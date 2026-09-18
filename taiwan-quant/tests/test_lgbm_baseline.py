"""
LightGBM baseline 的測試

## 最重要的一個測試是標籤淨化的邊界

40 日標籤在決策日之後 41 個交易日才揭曉。訓練時包含尚未揭曉的列就是
look-ahead，而且**不會拋錯**——它只會讓模型看起來很準。

少減 1 就洩漏一期。`test_purge_boundary_is_exact` 逐位釘住那個邊界。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.validate_lgbm_baseline import random_gross_median
from taiwan_quant.models.lgbm_baseline import (
    DEFAULT_PARAMS,
    N_TRIALS,
    BaselineError,
    build_dataset_for_model,
    build_features,
    cross_sectional_rank,
    purged_training_positions,
    train_and_predict,
)


def synthetic_bars(n: int, seed: int, drift: float = 0.0002) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(drift, 0.015, n))
    return pd.DataFrame(
        {
            "open": close * (1 + rng.normal(0, 0.002, n)),
            "high": close * (1 + abs(rng.normal(0, 0.008, n))),
            "low": close * (1 - abs(rng.normal(0, 0.008, n))),
            "close": close,
            "volume": rng.uniform(1e6, 5e6, n),
            "foreign_net": rng.normal(0, 1e5, n),
            "trust_net": rng.normal(0, 1e4, n),
            "dealer_net": rng.normal(0, 1e4, n),
            "margin_balance": rng.uniform(1e6, 2e6, n),
            "short_balance": rng.uniform(1e4, 1e5, n),
        },
        index=pd.bdate_range("2018-01-01", periods=n),
    )


# ══════════════════════════════════════════════════════════════
# 標籤淨化：最重要的邊界
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
@pytest.mark.parametrize(
    "predict_position, horizon, expected_end",
    [
        (100, 40, 60),
        (100, 1, 99),
        (41, 40, 1),
        (40, 40, 0),
    ],
)
def test_purge_boundary_is_exact(
    predict_position: int, horizon: int, expected_end: int
) -> None:
    """
    手算：位置 d 的標籤用 `closes[d+1+horizon]`，在位置 d+1+horizon
    才揭曉。要在 p 預測需要 `d + 1 + horizon <= p`，即 `d <= p−horizon−1`，
    右開邊界是 `p − horizon`。

        p=100, h=40  →  最後可用的 d 是 59，end = 60
        p=41,  h=40  →  最後可用的 d 是 0，  end = 1
        p=40,  h=40  →  沒有可用的 d，       end = 0
    """
    _, end = purged_training_positions(predict_position, horizon)
    assert end == expected_end


@pytest.mark.unit
def test_last_training_row_label_resolves_before_prediction() -> None:
    """
    邊界的語意檢查：最後一列訓練資料的標籤揭曉日必須 <= 預測日。

    這是上一個測試的另一種寫法，故意重複——這個邊界錯了不會有任何
    其他測試抓到。
    """
    predict_position, horizon = 200, 40
    _, end = purged_training_positions(predict_position, horizon)
    last_training = end - 1

    assert last_training + 1 + horizon <= predict_position
    # 再往後一列就洩漏
    assert (last_training + 1) + 1 + horizon > predict_position


@pytest.mark.unit
def test_purge_respects_warmup() -> None:
    """暖機不足的位置不可進訓練（特徵是 NaN，但範圍也該明確）"""
    start, end = purged_training_positions(100, 40, warmup=70)
    assert (start, end) == (70, 70)


@pytest.mark.unit
def test_purge_rejects_nonpositive_horizon() -> None:
    with pytest.raises(BaselineError, match="horizon"):
        purged_training_positions(100, 0)


# ══════════════════════════════════════════════════════════════
# 目標：橫斷面排名
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_cross_sectional_rank_removes_market_drift() -> None:
    """
    手算：兩個日期，第二天所有名字都 +10%（純大盤漂移）。
    絕對報酬完全不同，但排名必須一樣。
    """
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2020-01-01", "2020-01-02"]), ["A", "B", "C"]],
        names=["date", "stock_id"],
    )
    values = pd.Series([0.01, 0.02, 0.03, 0.11, 0.12, 0.13], index=index)

    ranks = cross_sectional_rank(values)

    day_one = ranks.xs(pd.Timestamp("2020-01-01"), level="date")
    day_two = ranks.xs(pd.Timestamp("2020-01-02"), level="date")
    pd.testing.assert_series_equal(day_one, day_two)
    assert day_one.tolist() == [1 / 3, 2 / 3, 1.0]


@pytest.mark.unit
def test_cross_sectional_rank_ignores_missing() -> None:
    """缺值不參與排名，也不變成 0"""
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2020-01-01"]), ["A", "B", "C"]],
        names=["date", "stock_id"],
    )
    ranks = cross_sectional_rank(pd.Series([np.nan, 0.02, 0.03], index=index))

    assert np.isnan(ranks.iloc[0])
    assert ranks.iloc[1] == pytest.approx(0.5)
    assert ranks.iloc[2] == pytest.approx(1.0)


# ══════════════════════════════════════════════════════════════
# 特徵組裝
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_features_carry_both_families_and_keep_nan() -> None:
    """
    技術面 16 欄 + 籌碼 16 欄，缺值**保留 NaN**。

    LightGBM 原生支援 NaN。填 0 會讓「沒有籌碼資料」與「籌碼淨額為零」
    變成同一件事。
    """
    by_stock = {"A": synthetic_bars(300, seed=1), "B": synthetic_bars(300, seed=2)}

    features = build_features(by_stock)

    assert features.index.names == ["date", "stock_id"]
    assert features.shape[1] == 32
    # 暖機期必有 NaN，而且沒有被填掉
    assert features["momentum_120"].isna().any()


@pytest.mark.unit
def test_features_survive_missing_chip_columns() -> None:
    """缺整組籌碼欄的標的只保留技術面，不可整檔消失"""
    bars = synthetic_bars(300, seed=3)
    price_only = bars[["open", "high", "low", "close", "volume"]]

    features = build_features({"A": bars, "B": price_only})

    stocks = set(features.index.get_level_values("stock_id"))
    assert stocks == {"A", "B"}
    b_rows = features.xs("B", level="stock_id")
    assert b_rows["foreign_net_ratio_5"].isna().all()
    assert b_rows["momentum_20"].notna().any()


@pytest.mark.unit
def test_build_features_rejects_empty_input() -> None:
    with pytest.raises(BaselineError, match="沒有任何標的"):
        build_features({})


# ══════════════════════════════════════════════════════════════
# 標籤與 validate_oos_momentum 必須一致
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_label_matches_the_backtest_definition() -> None:
    """
    標籤必須是 **T+1 開盤買、T+horizon 收盤賣**。

    ## ⚠️ 這個測試第一版釘住了錯的慣例

    原本斷言 `closes[2] / opens[1]`，也就是 T+1+horizon 出場——那是
    horizon + 1 天。程式與測試是同時寫的，兩邊照同一個誤解，所以測試
    通過而行為是錯的。

    正確推導：T+1 開盤進、X 收盤出，持有天數 = X − T。要 horizon 天
    就是 X = T + horizon。`data/integrity.holding_dates` 是單一來源：
    進場 `decision_index + 1`、出場 `decision_index + holding_days`。

    手算：position=0、horizon=1
        進場 = opens[1] = 11.0
        出場 = closes[1] = 12.0      ← 不是 closes[2]
        報酬 = 12/11 − 1 = +9.0909%
    """
    bars = pd.DataFrame(
        {
            "open": [10.0, 11.0, 12.5, 13.0],
            "high": [10.5, 12.0, 13.5, 14.0],
            "low": [9.5, 10.5, 12.0, 12.5],
            "close": [10.0, 12.0, 13.0, 13.5],
            "volume": [1e6] * 4,
        },
        index=pd.bdate_range("2020-01-01", periods=4),
    )
    calendar = list(bars.index)

    dataset = build_dataset_for_model({"A": bars}, calendar, horizon=1)
    value = dataset.forward_return.xs("A", level="stock_id").iloc[0]

    assert value == pytest.approx(12.0 / 11.0 - 1)
    assert value == pytest.approx(0.090909, abs=1e-6)
    # 多算一天會得到 13/11 − 1 = +18.18%，差一倍
    assert value != pytest.approx(13.0 / 11.0 - 1)


@pytest.mark.unit
def test_label_agrees_with_the_canonical_holding_dates() -> None:
    """
    標籤的出場日必須與 `data/integrity.holding_dates` 完全一致。

    那個函式是持有期邊界的單一來源。兩份實作就是兩份會漂移的規則——
    這次的 off-by-one 正是因為它們各自算。
    """
    from taiwan_quant.data.integrity import holding_dates

    rng = np.random.default_rng(20260918)
    n = 300
    close = 100 * np.cumprod(1 + rng.normal(0.0002, 0.01, n))
    bars = pd.DataFrame(
        {
            "open": close * 0.999, "high": close * 1.01,
            "low": close * 0.99, "close": close,
            "volume": np.full(n, 1e6),
        },
        index=pd.bdate_range("2020-01-01", periods=n),
    )
    calendar = list(bars.index)
    horizon = 40

    dataset = build_dataset_for_model({"A": bars}, calendar, horizon=horizon)
    position = 100
    entry_day, exit_day = holding_dates(
        calendar[position], calendar, holding_days=horizon)

    expected = (
        bars["close"].loc[exit_day] / bars["open"].loc[entry_day] - 1
    )
    actual = dataset.forward_return.loc[(calendar[position], "A")]

    assert actual == pytest.approx(expected)


# ══════════════════════════════════════════════════════════════
# baseline 的性質
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_baseline_declares_one_trial() -> None:
    """
    未調參，所以 `n_trials` 是 1。

    這是 baseline 最有價值的性質：先前掃 126 組時 DSR 0.0042 不通過、
    掃 16 組時 PBO 0.625~0.750。**改動任何超參數就必須把 N_TRIALS 加上去。**
    """
    assert N_TRIALS == 1
    assert DEFAULT_PARAMS["seed"] == 20260917, "種子固定才可重現（禁令 7）"
    assert DEFAULT_PARAMS["deterministic"] is True


@pytest.mark.unit
def test_random_benchmark_is_invariant_to_candidate_input_order() -> None:
    realized = pd.Series({"A": 0.10, "B": -0.20, "C": 0.30, "D": 0.00})

    forward = random_gross_median(
        realized, ["A", "B", "C", "D"],
        n_positions=2, n_draws=20, seed=20260919,
    )
    reversed_order = random_gross_median(
        realized, ["D", "C", "B", "A"],
        n_positions=2, n_draws=20, seed=20260919,
    )

    assert forward == pytest.approx(reversed_order)


@pytest.mark.unit
def test_train_and_predict_returns_empty_when_starved() -> None:
    """
    訓練樣本不足時回空，不可用幾百列硬訓一個模型然後當成結果。
    """
    by_stock = {"A": synthetic_bars(300, seed=1), "B": synthetic_bars(300, seed=2)}
    calendar = list(by_stock["A"].index)
    dataset = build_dataset_for_model(by_stock, calendar, horizon=40)

    predicted, importance = train_and_predict(
        dataset, calendar, train_range=(0, 5), predict_position=200,
        candidates=("A", "B"),
    )

    assert predicted.empty
    assert importance == {}


@pytest.mark.unit
def test_train_and_predict_only_scores_the_given_candidates() -> None:
    """
    只能對時點標的池內的名字評分。多回傳一檔就等於在池外選股（禁令 2 的反面）。
    """
    by_stock = {f"S{i}": synthetic_bars(600, seed=i) for i in range(6)}
    calendar = list(by_stock["S0"].index)
    dataset = build_dataset_for_model(by_stock, calendar, horizon=40)
    train_range = purged_training_positions(500, 40, warmup=120)

    predicted, importance = train_and_predict(
        dataset, calendar, train_range, predict_position=500,
        candidates=("S0", "S1", "S2"),
    )

    assert set(predicted.index) <= {"S0", "S1", "S2"}
    assert len(importance) == 32, "特徵重要度要涵蓋全部 32 欄"


@pytest.mark.unit
def test_shuffle_seed_changes_the_model_but_not_the_shape() -> None:
    """
    `shuffle_seed` 在每個日期之內打亂標籤，用來偵測洩漏。

    要求：預測的 index 與形狀完全不變（資料結構沒動），但預測值要改變
    （標籤關係被破壞）。若打亂後預測一模一樣，代表打亂根本沒生效。
    """
    by_stock = {f"S{i}": synthetic_bars(600, seed=i) for i in range(6)}
    calendar = list(by_stock["S0"].index)
    dataset = build_dataset_for_model(by_stock, calendar, horizon=40)
    train_range = purged_training_positions(500, 40, warmup=250)

    honest, _ = train_and_predict(
        dataset, calendar, train_range, 500, tuple(by_stock))
    shuffled, _ = train_and_predict(
        dataset, calendar, train_range, 500, tuple(by_stock), shuffle_seed=7)

    assert list(honest.index) == list(shuffled.index)
    assert not np.allclose(honest.to_numpy(), shuffled.to_numpy())
