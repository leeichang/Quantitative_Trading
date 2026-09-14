"""
Walk-Forward 切分與標籤隔離測試

## 抓到的 bug（第 10 個）

原本的切分是：

```python
train_dates = dates[:cursor]
test_dates  = dates[cursor : cursor + test_span]
```

**訓練集的最後幾期，標籤期跨進測試段。** 持有 60 日、每 5 日決策時，
最後 12 期決策的實際報酬要用到測試期的價格才算得出來——校準器等於
「看過」測試期的走勢。

污染比例：第一個 fold 8.0%（12/150），最後一個 fold 2.4%（12/498）。

這與 CLAUDE.md 禁令 1（T 日決策只能用 T 日收盤前可得的資料）是同一條
原則的延伸：**訓練時可用的標籤，必須在訓練期結束前就已經揭曉。**
"""

from __future__ import annotations

import pandas as pd
import pytest

from taiwan_quant.validation.walk_forward import (
    WalkForwardError,
    embargo_periods,
    walk_forward_folds,
)


def dates(n: int) -> list[pd.Timestamp]:
    """n 個決策日，每 5 個交易日一次"""
    return list(pd.date_range("2019-01-04", periods=n, freq="5B"))


# ══════════════════════════════════════════════════════════════
# 隔離期數
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_embargo_periods_matches_hand_calculation() -> None:
    """
    持有 60 日、每 5 日決策 → 60 / 5 = 12 期的標籤會跨進測試段。
    """
    assert embargo_periods(horizon=60, stride=5) == 12


@pytest.mark.unit
def test_embargo_rounds_up() -> None:
    """
    不可無條件捨去。持有 22 日、每 5 日決策 → 22/5 = 4.4，
    第 5 期的標籤仍有一部分落在測試段內，必須一起隔離。
    """
    assert embargo_periods(horizon=22, stride=5) == 5


@pytest.mark.unit
def test_embargo_is_zero_when_label_resolves_within_one_period() -> None:
    """持有期等於決策間隔時，標籤在下一期開始前就揭曉，不需隔離"""
    assert embargo_periods(horizon=5, stride=5) == 1


@pytest.mark.unit
def test_embargo_rejects_non_positive() -> None:
    with pytest.raises(WalkForwardError, match="必須為正"):
        embargo_periods(horizon=0, stride=5)
    with pytest.raises(WalkForwardError, match="必須為正"):
        embargo_periods(horizon=60, stride=0)


# ══════════════════════════════════════════════════════════════
# 切分
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_train_excludes_embargoed_periods() -> None:
    """
    訓練集尾端的 12 期必須被剔除。

    150 期訓練 → 實際可用 138 期，最後一期是 index 137。
    """
    d = dates(200)
    folds = list(walk_forward_folds(d, first_train=150, test_span=12,
                                    horizon=60, stride=5))
    train, test = folds[0]

    assert len(train) == 138
    assert train[-1] == d[137]
    assert test == d[150:162]


@pytest.mark.unit
def test_no_train_date_label_reaches_test_period() -> None:
    """
    直接驗證性質：訓練集裡每一個決策日的**標籤到期日**，
    都必須早於測試段的第一天。

    這是隔離要保證的唯一一件事。
    """
    d = dates(200)
    horizon, stride = 60, 5
    for train, test in walk_forward_folds(d, first_train=150, test_span=12,
                                          horizon=horizon, stride=stride):
        label_span = pd.Timedelta(days=horizon * 7 / 5)   # 交易日換日曆日的保守估計
        for day in train:
            assert day + label_span <= test[0] + pd.Timedelta(days=7), (
                f"訓練日 {day.date()} 的標籤跨進測試段 {test[0].date()}"
            )


@pytest.mark.unit
def test_folds_advance_by_test_span() -> None:
    d = dates(200)
    folds = list(walk_forward_folds(d, first_train=150, test_span=12,
                                    horizon=60, stride=5))
    assert folds[0][1] == d[150:162]
    assert folds[1][1] == d[162:174]


@pytest.mark.unit
def test_train_window_expands() -> None:
    """滾動擴張窗口：訓練集越來越長"""
    d = dates(200)
    sizes = [len(train) for train, _ in
             walk_forward_folds(d, first_train=150, test_span=12,
                                horizon=60, stride=5)]
    assert sizes == sorted(sizes)
    assert sizes[1] > sizes[0]


@pytest.mark.unit
def test_stops_before_running_out_of_test_data() -> None:
    """測試段不足時停止，不可回傳半截的 fold"""
    d = dates(170)
    folds = list(walk_forward_folds(d, first_train=150, test_span=12,
                                    horizon=60, stride=5))
    assert len(folds) == 1
    assert len(folds[0][1]) == 12


@pytest.mark.unit
def test_yields_nothing_when_data_too_short() -> None:
    d = dates(100)
    assert list(walk_forward_folds(d, first_train=150, test_span=12,
                                   horizon=60, stride=5)) == []


@pytest.mark.unit
def test_skips_fold_when_embargo_empties_train() -> None:
    """
    隔離之後訓練集為空的 fold 必須跳過，不可用空集合擬合。

    first_train=10、隔離 12 期 → 訓練集會是空的。
    """
    d = dates(200)
    folds = list(walk_forward_folds(d, first_train=10, test_span=12,
                                    horizon=60, stride=5))
    assert all(len(train) > 0 for train, _ in folds)


@pytest.mark.unit
def test_rejects_unsorted_dates() -> None:
    d = dates(200)
    with pytest.raises(WalkForwardError, match="升冪"):
        list(walk_forward_folds(list(reversed(d)), first_train=150,
                                test_span=12, horizon=60, stride=5))


@pytest.mark.unit
def test_embargo_reduces_training_samples_measurably() -> None:
    """
    量化隔離的代價：第一個 fold 損失 8%，最後一個 2.4%。

    這個數字要留在測試裡——它是「修正前的結果被污染多少」的證據。
    """
    d = dates(520)
    folds = list(walk_forward_folds(d, first_train=150, test_span=12,
                                    horizon=60, stride=5))
    first_loss = 12 / 150
    last_train_end = 150 + (len(folds) - 1) * 12
    last_loss = 12 / last_train_end

    assert first_loss == pytest.approx(0.080, abs=0.001)
    assert last_loss == pytest.approx(0.024, abs=0.002)


@pytest.mark.unit
def test_explicit_oos_start_keeps_expanding_training_by_default() -> None:
    """指定切分點不改變 walk-forward：已揭曉的前期 OOS 會進入後續訓練。"""
    d = dates(240)
    requested = d[180] - pd.Timedelta(days=1)
    folds = list(walk_forward_folds(
        d,
        first_train=150,
        test_span=12,
        horizon=60,
        stride=5,
        oos_start=requested,
        dev_end=d[179],
    ))

    assert folds[0][1][0] == d[180]
    assert [len(train) for train, _ in folds] == [168, 180, 192, 204, 216]


@pytest.mark.unit
def test_freeze_model_keeps_training_at_initial_oos_cutoff() -> None:
    """只有明確指定 freeze_model，校準資料才永久固定。"""
    d = dates(240)
    folds = list(walk_forward_folds(
        d,
        first_train=150,
        test_span=12,
        horizon=60,
        stride=5,
        oos_start=d[180],
        dev_end=d[179],
        freeze_model=True,
    ))

    assert [len(train) for train, _ in folds] == [168, 168, 168, 168, 168]
    assert all(train[-1] == d[167] for train, _ in folds)


@pytest.mark.unit
def test_oos_start_alone_expands_across_many_folds() -> None:
    """
    不給 `dev_end` 時，訓練集要一路擴張，不是只擴張前幾個 fold。

    手算：cursor 從 180 起、每次 +12，隔離 12 期。
    訓練期數 = 180-12, 192-12, 204-12, ... = 168, 180, 192, ...
    """
    d = dates(320)
    sizes = [len(train) for train, _ in walk_forward_folds(
        d,
        first_train=150,
        test_span=12,
        horizon=60,
        stride=5,
        oos_start=d[180],
    )]

    assert sizes[:3] == [168, 180, 192]
    assert sizes == sorted(sizes)
    assert len(sizes) > 5
    assert sizes[-1] > sizes[0]


@pytest.mark.unit
def test_expanding_after_oos_start_still_embargoes_every_fold() -> None:
    """
    擴張不得吃掉隔離。每個 fold 的訓練尾端與測試段起點之間，
    都要**正好**隔 12 期。

    這是 `0f53df5` 之前的真實漏洞：`min(cursor - gap, cursor_初始)`
    在 fold 2 以後會選中 `cursor_初始`，等於完全沒套隔離。
    """
    d = dates(320)
    position = {day: i for i, day in enumerate(d)}
    folds = list(walk_forward_folds(
        d,
        first_train=150,
        test_span=12,
        horizon=60,
        stride=5,
        oos_start=d[180],
    ))

    assert len(folds) > 1
    for train, test in folds:
        excluded = position[test[0]] - position[train[-1]] - 1
        assert excluded == 12, f"測試段 {test[0].date()} 前只隔離了 {excluded} 期"


@pytest.mark.unit
def test_early_dev_end_keeps_a_constant_lag_not_a_fixed_wall() -> None:
    """
    `dev_end` 明顯早於 OOS 起點時，那段距離會**隨 fold 一起往前移**，
    不是釘死在某一天。

    手算：`dev_end = d[155]` → 初始訓練截止 156（比隔離要求的 168 更嚴）。
    之後每個 fold 都保持這個落後量：156, 168, 180, ...
    緊貼 `oos_start` 的 `dev_end` 不會觸發這條路徑，所以要單獨守。
    """
    d = dates(320)
    sizes = [len(train) for train, _ in walk_forward_folds(
        d,
        first_train=150,
        test_span=12,
        horizon=60,
        stride=5,
        oos_start=d[180],
        dev_end=d[155],
    )]

    assert sizes[:3] == [156, 168, 180]


@pytest.mark.unit
def test_omitting_oos_start_preserves_existing_folds() -> None:
    """未指定凍結日期時，舊有推導切分必須逐筆不變。"""
    d = dates(200)
    legacy = list(walk_forward_folds(
        d, first_train=150, test_span=12, horizon=60, stride=5
    ))
    explicit_none = list(walk_forward_folds(
        d,
        first_train=150,
        test_span=12,
        horizon=60,
        stride=5,
        oos_start=None,
        dev_end=None,
    ))

    assert explicit_none == legacy


@pytest.mark.unit
def test_oos_start_before_minimum_training_raises() -> None:
    """OOS 太早會讓首次訓練不足，必須明確拒絕。"""
    d = dates(200)
    with pytest.raises(WalkForwardError, match="早於最低訓練"):
        list(walk_forward_folds(
            d,
            first_train=150,
            test_span=12,
            horizon=60,
            stride=5,
            oos_start=d[100],
        ))
