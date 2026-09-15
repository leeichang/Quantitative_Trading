"""
外部驗證工具測試（CPCV / PBO / DSR）

## 為什麼要自己寫一份測試

上游 `HKUDS/Vibe-Trading` 有 8,053 行 quantlib 測試，但那是**他們的**測試。
CLAUDE.md 禁令 10 要求本專案的特徵與標記函式都要有比對**手算值**的測試，
拿別人的測試當自己的等於沒測——升級時才會發現介面早就變了。

這裡只測**本專案實際會呼叫的函式**，其餘不碰。

## 最有價值的那個測試

`test_matches_own_walk_forward_embargo`：用兩套獨立實作跑同一組參數，
隔離期數必須一致。

本專案的 `embargo_periods(60, 5) = 12`，上游的 purge 是用標籤區間重疊
判定的——兩條完全不同的路徑，答案應該相同。不同就代表有一邊錯了，
而且那會是自己看不出來的錯。
"""

from __future__ import annotations

import numpy as np
import pytest

from taiwan_quant.validation.external.crossvalidation import (
    combinatorial_purged_splits,
    purged_walk_forward_splits,
)
from taiwan_quant.validation.external.multipletesting import (
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
)
from taiwan_quant.validation.walk_forward import embargo_periods


# ══════════════════════════════════════════════════════════════
# purge 語意：手算
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_purge_is_bidirectional_around_the_test_block() -> None:
    """
    手算：10 個樣本切 5 塊（每塊 2 個），標籤跨 2 期，測試取第 3 塊。

    **關鍵：測試段要看的不是 [4,5]，是它的「資訊區間」。**
    測試樣本 5 的標籤要到 index 7 才揭曉，所以整段資訊延伸到 [4,7]。

    ```
    索引        0     1     2     3    [4     5]    6     7     8     9
    標籤區間  [0,2] [1,3] [2,4] [3,5] [4,6] [5,7] [6,8] [7,9] [8,9] [9,9]
                          剔除  剔除   ← 測試 →   剔除  剔除  保留  保留
    ```

    - i=2 [2,4]、i=3 [3,5]：標籤伸進測試段 → 剔除（**向前**洩漏）
    - i=6 [6,8]、i=7 [7,9]：落在測試標籤還沒揭曉的區間 → 剔除（**向後**洩漏）
    - i=0 [0,2]、i=1 [1,3]：整段早於 4 → 保留
    - i=8 [8,9]、i=9 [9,9]：整段晚於 7 → 保留

    預期訓練集 = [0, 1, 8, 9]，purge 掉 4 個。

    ## 這個測試抓到的事

    我第一版手算成 `[0, 1, 6, 7, 8, 9]`，只想到向前那一邊，測試因此紅燈。

    本專案自己的 `walk_forward.py` 只做**單向**隔離——那是對的，因為
    walk-forward 的訓練集永遠在測試段之前，向後那一邊根本不存在。
    但 CPCV 的訓練資料分布在測試塊的**兩側**，雙向才正確。

    兩邊語意不同不是 bug，是適用場景不同。混用才是 bug。
    """
    n = 10
    label_ends = np.minimum(np.arange(n) + 2, n - 1)

    splits = list(
        combinatorial_purged_splits(
            n, label_ends, n_groups=5, n_test_groups=1, embargo_fraction=0.0
        )
    )
    third = splits[2]

    assert third.test.tolist() == [4, 5]
    assert third.train.tolist() == [0, 1, 8, 9]
    assert third.purged == 4


@pytest.mark.unit
def test_train_and_test_never_overlap() -> None:
    """切分要保證的最低限度：同一個索引不可同時是訓練與測試。"""
    n = 60
    label_ends = np.minimum(np.arange(n) + 5, n - 1)

    for split in combinatorial_purged_splits(n, label_ends, n_groups=6, n_test_groups=2):
        assert not (set(split.train.tolist()) & set(split.test.tolist()))


# ══════════════════════════════════════════════════════════════
# CPCV 的路徑數：手算組合數
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_path_count_is_the_binomial_coefficient() -> None:
    """
    C(6,2) = 15、C(5,1) = 5、C(8,2) = 28。

    這個數字就是 CPCV 的全部價值來源：單一 walk-forward 只有 1 條路徑。
    """
    n = 120
    label_ends = np.minimum(np.arange(n) + 3, n - 1)

    for n_groups, n_test, expected in ((6, 2, 15), (5, 1, 5), (8, 2, 28)):
        paths = list(
            combinatorial_purged_splits(n, label_ends, n_groups, n_test)
        )
        assert len(paths) == expected


@pytest.mark.unit
def test_rejects_degenerate_group_settings() -> None:
    n = 40
    with pytest.raises(ValueError, match="n_test_groups"):
        list(combinatorial_purged_splits(n, None, n_groups=4, n_test_groups=4))
    with pytest.raises(ValueError, match="groups"):
        list(combinatorial_purged_splits(4, None, n_groups=10, n_test_groups=2))


# ══════════════════════════════════════════════════════════════
# 與本專案自己的 walk_forward 交叉比對 ← 最重要
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_matches_own_walk_forward_embargo() -> None:
    """
    兩套獨立實作必須算出同樣的隔離期數。

    ```
    本專案   embargo_periods(60, 5) = ceil(60/5) = 12      由持有期÷決策間隔推導
    上游     purge 掉標籤區間與測試段重疊的訓練樣本         由區間重疊判定
    ```

    推導方式完全不同，答案必須相同。不同就代表有一邊錯了。

    ⚠️ 只有 **walk-forward** 這一支可以這樣比。CPCV 的 purge 是雙向的
    （見 `test_purge_is_bidirectional_around_the_test_block`），數量本來
    就會比本專案的單向隔離多——那不是矛盾，是適用場景不同。
    """
    n = 370                                          # 1,850 交易日 / 每 5 日決策
    horizon, stride = 60, 5
    gap = embargo_periods(horizon, stride)           # = 12
    label_ends = np.minimum(np.arange(n) + gap, n - 1)

    folds = list(purged_walk_forward_splits(n, label_ends, n_folds=5))

    assert gap == 12
    assert all(fold.purged == gap for fold in folds), [f.purged for f in folds]


@pytest.mark.unit
def test_walk_forward_trains_only_on_the_past() -> None:
    """
    禁令 5：滾動訓練只能用預測期之前的資料。

    上游的 `purged_kfold_splits` 會訓練在測試段之後的資料上（對估計泛化
    是對的，對模擬實盤是錯的），所以本專案只用 walk-forward 這一支。
    這個測試就是守住這條界線。
    """
    n = 200
    label_ends = np.minimum(np.arange(n) + 12, n - 1)

    for fold in purged_walk_forward_splits(n, label_ends, n_folds=5):
        assert int(fold.train.max()) < int(fold.test.min())


@pytest.mark.unit
def test_walk_forward_window_expands() -> None:
    """預設 expanding：訓練集逐 fold 變長，與本專案 walk_forward.py 一致。"""
    n = 200
    label_ends = np.minimum(np.arange(n) + 12, n - 1)

    sizes = [len(f.train) for f in purged_walk_forward_splits(n, label_ends, n_folds=5)]

    assert sizes == sorted(sizes)
    assert sizes[-1] > sizes[0]


# ══════════════════════════════════════════════════════════════
# PBO：極端情形手推
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_pbo_is_zero_when_one_strategy_dominates_everywhere() -> None:
    """
    一支策略在每一段都明顯較好時，樣本內選它、樣本外它還是第一，
    **沒有任何一次落到中位數以下** → PBO = 0。

    這是 PBO 的下界，可以手推而不必知道 logit 的實際數值。
    """
    rng = np.random.default_rng(0)
    n_obs = 240
    good = 0.01 + rng.normal(0, 0.001, n_obs)
    bad = -0.01 + rng.normal(0, 0.001, n_obs)
    performance = np.column_stack([good, bad, bad, bad])

    result = probability_of_backtest_overfitting(performance, n_splits=8)

    assert result.pbo == pytest.approx(0.0, abs=1e-9)


@pytest.mark.unit
def test_pbo_detects_noise_as_overfitting_risk() -> None:
    """
    全是雜訊時，樣本內的贏家在樣本外的排名應該是隨機的，
    PBO 要明顯高於「有真優勢」的情形。

    這裡只斷言**相對關係**——PBO 的絕對值會隨抽樣波動，釘死數字等於
    拿程式輸出反填預期值。
    """
    rng = np.random.default_rng(7)
    n_obs, n_strategies = 240, 20
    noise = rng.normal(0, 1, size=(n_obs, n_strategies))

    pure_noise = probability_of_backtest_overfitting(noise, n_splits=8).pbo

    with_edge = noise.copy()
    with_edge[:, 3] += 0.35
    real_edge = probability_of_backtest_overfitting(with_edge, n_splits=8).pbo

    assert real_edge < pure_noise
    assert pure_noise > 0.2


@pytest.mark.unit
def test_pbo_needs_competitors_and_even_splits() -> None:
    rng = np.random.default_rng(1)
    single = rng.normal(0, 1, size=(100, 1))
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(single, n_splits=8)

    pair = rng.normal(0, 1, size=(100, 2))
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(pair, n_splits=7)   # 必須是偶數


# ══════════════════════════════════════════════════════════════
# DSR：性質而非數值
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_more_trials_raise_the_luck_bar() -> None:
    """
    掃越多組參數，「純靠運氣能拿到的最佳 Sharpe」越高，DSR 越低。

    這是多重測試校正的定義，不必知道 Bailey-López de Prado 的公式細節。
    """
    common = dict(observed_sharpe=1.2, trial_sharpe_std=0.5, n_observations=60)
    few = deflated_sharpe_ratio(n_trials=2, **common)
    many = deflated_sharpe_ratio(n_trials=200, **common)

    assert many.expected_maximum_sharpe > few.expected_maximum_sharpe
    assert many.deflated_sharpe_ratio < few.deflated_sharpe_ratio


@pytest.mark.unit
def test_observed_below_luck_bar_does_not_survive() -> None:
    """
    本專案的真實處境：掃 20 組、有效樣本 30、最佳 Sharpe 0.90。

    運氣的門檻（expected_maximum_sharpe）會高於 0.90，
    所以 `survives` 必須是 False——**最好的組合比運氣還差**。
    """
    result = deflated_sharpe_ratio(
        observed_sharpe=0.90, n_trials=20, trial_sharpe_std=0.5, n_observations=30
    )

    assert result.expected_maximum_sharpe > result.observed_sharpe
    assert result.survives is False


@pytest.mark.unit
def test_higher_observed_sharpe_raises_dsr() -> None:
    """同樣的掃描次數下，實際表現越好，DSR 越高。單調性。"""
    common = dict(n_trials=20, trial_sharpe_std=0.5, n_observations=60)
    weak = deflated_sharpe_ratio(observed_sharpe=0.5, **common)
    strong = deflated_sharpe_ratio(observed_sharpe=2.5, **common)

    assert strong.deflated_sharpe_ratio > weak.deflated_sharpe_ratio
    assert strong.survives is True
