"""
對照組與資料集組裝測試

CLAUDE.md 要求每份回測報告都要並列等權買進持有與隨機進場。
對照組算錯比沒有對照更糟——第一版的 bug 是買進持有只涵蓋 7 個月、
策略涵蓋 2 年多，比較毫無意義卻看起來很正常。
"""

from __future__ import annotations

import pandas as pd
import pytest

from taiwan_quant.data.dataset import CHIP_COLUMNS, build_dataset
from taiwan_quant.validation.benchmarks import (
    buy_and_hold_return,
    effective_samples,
)


def price_frame(spec: dict[str, list[float]], start: str = "2024-01-01") -> pd.DataFrame:
    """組出 MultiIndex(stock_id, date) 的價格表"""
    frames = []
    for stock_id, closes in spec.items():
        index = pd.date_range(start, periods=len(closes), freq="B")
        frames.append(
            pd.DataFrame(
                {
                    "stock_id": stock_id,
                    "date": index,
                    "open": closes,
                    "high": [c * 1.01 for c in closes],
                    "low": [c * 0.99 for c in closes],
                    "close": closes,
                    "volume": [1_000_000] * len(closes),
                }
            )
        )
    return pd.concat(frames).set_index(["stock_id", "date"]).sort_index()


def by_stock(spec: dict[str, list[float]], start: str = "2024-01-01") -> dict[str, pd.DataFrame]:
    frame = price_frame(spec, start)
    return {
        sid: frame.xs(sid, level="stock_id") for sid in frame.index.get_level_values(0).unique()
    }


# ══════════════════════════════════════════════════════════════
# 等權買進持有
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_buy_and_hold_matches_hand_calculation() -> None:
    """
    A: 100 → 120  = +20%
    B: 100 →  90  = −10%
    等權平均 = (+0.20 − 0.10) / 2 = +5%
    """
    data = by_stock({"A": [100, 110, 120], "B": [100, 95, 90]})
    start, end = pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-03")

    assert buy_and_hold_return(data, start, end) == pytest.approx(0.05)


@pytest.mark.unit
def test_buy_and_hold_respects_window() -> None:
    """
    期間必須與策略完全一致，否則不可比。

    取 [第 2 天, 第 3 天]：100 → 120 只算 110 → 120 = +9.09%
    """
    data = by_stock({"A": [100, 110, 120]})
    start, end = pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")

    assert buy_and_hold_return(data, start, end) == pytest.approx(120 / 110 - 1)


@pytest.mark.unit
def test_buy_and_hold_aligns_by_date_not_position() -> None:
    """
    各檔起始日不同時必須以日期對齊。

    A 從 01-01 起、B 從 01-03 起。要求 [01-03, 01-05]：
        A 取第 3~5 根：100 → 102  = +2%
        B 取第 1~3 根：200 → 206  = +3%
    等權 = +2.5%
    """
    frames = {
        "A": pd.DataFrame(
            {"close": [98, 99, 100, 101, 102]},
            index=pd.date_range("2024-01-01", periods=5, freq="B"),
        ),
        "B": pd.DataFrame(
            {"close": [200, 203, 206]},
            index=pd.date_range("2024-01-03", periods=3, freq="B"),
        ),
    }
    result = buy_and_hold_return(
        frames, pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-05")
    )
    assert result == pytest.approx((0.02 + 0.03) / 2)


@pytest.mark.unit
def test_buy_and_hold_returns_none_when_window_empty() -> None:
    """
    無法計算時回 `None`，不是 0 或 NaN。

    0 會被誤讀成「買進持有剛好打平」，讓策略看起來贏了對照組。
    """
    data = by_stock({"A": [100, 110, 120]})
    result = buy_and_hold_return(
        data, pd.Timestamp("2030-01-01"), pd.Timestamp("2030-12-31")
    )
    assert result is None


@pytest.mark.unit
def test_buy_and_hold_skips_single_bar_series() -> None:
    """只有一根 K 無法算報酬，該檔跳過而非當成 0%"""
    data = by_stock({"A": [100, 110, 120], "B": [100]})
    start, end = pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-03")

    assert buy_and_hold_return(data, start, end) == pytest.approx(0.20)


@pytest.mark.unit
def test_buy_and_hold_rejects_reversed_window() -> None:
    data = by_stock({"A": [100, 110, 120]})
    with pytest.raises(ValueError, match="start"):
        buy_and_hold_return(data, pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-01"))


# ══════════════════════════════════════════════════════════════
# 有效獨立樣本數
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_effective_samples_discounts_overlap() -> None:
    """
    持有 20 日、每 5 日決策一次 → 每 4 個決策日才有 1 個不重疊。

    10 個決策日 → ceil(10 / 4) = 3 個有效樣本。
    """
    dates = list(pd.date_range("2024-01-01", periods=10, freq="B"))
    assert effective_samples(dates, horizon=20, stride=5) == 3


@pytest.mark.unit
def test_effective_samples_no_overlap_when_horizon_equals_stride() -> None:
    dates = list(pd.date_range("2024-01-01", periods=10, freq="B"))
    assert effective_samples(dates, horizon=5, stride=5) == 10


@pytest.mark.unit
def test_effective_samples_deduplicates_dates() -> None:
    """同一決策日選 3 檔算 1 個樣本，不是 3 個"""
    day = pd.Timestamp("2024-01-01")
    assert effective_samples([day, day, day], horizon=5, stride=5) == 1


@pytest.mark.unit
def test_effective_samples_empty() -> None:
    assert effective_samples([], horizon=20, stride=5) == 0


# ══════════════════════════════════════════════════════════════
# 資料集組裝
# ══════════════════════════════════════════════════════════════


def chip_frame(stock_ids: list[str], n: int, start: str = "2024-01-01") -> pd.DataFrame:
    frames = []
    for sid in stock_ids:
        index = pd.date_range(start, periods=n, freq="B")
        data = {"stock_id": sid, "date": index}
        data.update({col: [1000.0] * n for col in CHIP_COLUMNS})
        frames.append(pd.DataFrame(data))
    return pd.concat(frames).set_index(["stock_id", "date"]).sort_index()


@pytest.mark.unit
def test_build_dataset_joins_chips() -> None:
    prices = price_frame({"A": [100.0] * 10})
    chips = chip_frame(["A"], 10)
    result = build_dataset(["A"], prices, chips, min_length=5)

    assert set(CHIP_COLUMNS).issubset(result.by_stock["A"].columns)
    assert len(result.by_stock["A"]) == 10


@pytest.mark.unit
def test_build_dataset_drops_short_series() -> None:
    """
    序列過短的標的納入只會汙染全域時間軸。

    實測案例：7769 與籌碼 join 後只剩 71 根，卻被當成完整標的。
    """
    prices = price_frame({"A": [100.0] * 10, "B": [100.0] * 3})
    result = build_dataset(["A", "B"], prices, chips=None, min_length=5)

    assert list(result.by_stock) == ["A"]
    assert result.skipped == (("B", 3),)


@pytest.mark.unit
def test_build_dataset_records_missing_stocks() -> None:
    prices = price_frame({"A": [100.0] * 10})
    result = build_dataset(["A", "ZZZZ"], prices, chips=None, min_length=5)

    assert result.missing == ("ZZZZ",)


@pytest.mark.unit
def test_build_dataset_inner_join_shortens_series() -> None:
    """
    籌碼只有末段時 inner join 會把價格序列砍短——這正是 7769 的情形。
    砍短後若不足門檻，必須被剔除。
    """
    prices = price_frame({"A": [100.0] * 20})
    chips = chip_frame(["A"], 3, start="2024-01-01")
    result = build_dataset(["A"], prices, chips, min_length=10)

    assert result.by_stock == {}
    assert result.skipped == (("A", 3),)


@pytest.mark.unit
def test_build_dataset_keeps_prices_when_chips_absent_for_stock() -> None:
    """
    某檔沒有籌碼資料時保留價格（不 join），不是整檔丟掉。

    籌碼族的策略會在自己的 required_columns 檢查時跳過它。
    """
    prices = price_frame({"A": [100.0] * 10, "B": [100.0] * 10})
    chips = chip_frame(["A"], 10)
    result = build_dataset(["A", "B"], prices, chips, min_length=5)

    assert set(result.by_stock) == {"A", "B"}
    assert "foreign_net" in result.by_stock["A"].columns
    assert "foreign_net" not in result.by_stock["B"].columns


@pytest.mark.unit
def test_build_dataset_does_not_mutate_input() -> None:
    prices = price_frame({"A": [100.0] * 10})
    before = prices.copy()
    build_dataset(["A"], prices, chips=None, min_length=5)

    pd.testing.assert_frame_equal(prices, before)


# ══════════════════════════════════════════════════════════════
# 等權買進持有權益曲線
# ══════════════════════════════════════════════════════════════

from taiwan_quant.validation.benchmarks import equal_weight_equity  # noqa: E402


@pytest.mark.unit
def test_equal_weight_equity_starts_at_one() -> None:
    data = by_stock({"A": [100, 110, 120], "B": [50, 55, 60]})
    cal = list(pd.date_range("2024-01-01", periods=3, freq="B"))

    curve = equal_weight_equity(data, cal)
    assert curve.iloc[0] == pytest.approx(1.0)


@pytest.mark.unit
def test_equal_weight_equity_matches_hand_calculation() -> None:
    """
    A: 100 → 120 = 1.20
    B: 100 →  90 = 0.90
    等權 = (1.20 + 0.90) / 2 = 1.05
    """
    data = by_stock({"A": [100, 110, 120], "B": [100, 95, 90]})
    cal = list(pd.date_range("2024-01-01", periods=3, freq="B"))

    curve = equal_weight_equity(data, cal)
    assert curve.iloc[-1] == pytest.approx(1.05)


@pytest.mark.unit
def test_equal_weight_equity_forward_fills_missing_days() -> None:
    """停牌沿用前一日價格，不是歸零"""
    data = {
        "A": pd.DataFrame(
            {"close": [100.0, 110.0]},
            index=pd.DatetimeIndex(["2024-01-01", "2024-01-03"]),
        )
    }
    cal = list(pd.date_range("2024-01-01", periods=3, freq="D"))

    curve = equal_weight_equity(data, cal)
    assert curve.iloc[1] == pytest.approx(1.0)
    assert curve.iloc[2] == pytest.approx(1.1)


@pytest.mark.unit
def test_equal_weight_equity_skips_stocks_not_listed_at_start() -> None:
    """期初還沒上市的標的不納入——那是 look-ahead"""
    data = {
        "A": pd.DataFrame(
            {"close": [100.0, 120.0]},
            index=pd.DatetimeIndex(["2024-01-01", "2024-01-02"]),
        ),
        "B": pd.DataFrame(
            {"close": [100.0]},
            index=pd.DatetimeIndex(["2024-01-02"]),
        ),
    }
    cal = list(pd.DatetimeIndex(["2024-01-01", "2024-01-02"]))

    curve = equal_weight_equity(data, cal)
    assert curve.iloc[-1] == pytest.approx(1.20)


@pytest.mark.unit
def test_equal_weight_equity_rejects_empty_calendar() -> None:
    with pytest.raises(ValueError, match="日曆"):
        equal_weight_equity(by_stock({"A": [100.0, 110.0]}), [])


@pytest.mark.unit
def test_equal_weight_equity_rejects_when_nothing_listed_at_start() -> None:
    data = by_stock({"A": [100.0, 110.0]}, start="2025-01-01")
    cal = list(pd.date_range("2024-01-01", periods=2, freq="B"))

    with pytest.raises(ValueError, match="期初"):
        equal_weight_equity(data, cal)


# ══════════════════════════════════════════════════════════════
# 0050 買進持有（CLAUDE.md 明訂的必跑對照組）
#
# 先前四輪驗證都只有「等權買進持有」，缺 0050 本身——因為上游資料庫
# 沒有 ETF。長歷史回補之後 0050/0051/0056 都有完整 11 年資料，
# 這個缺口沒有理由再留著。
#
# 為什麼 0050 是比等權更嚴格的對照：它是**可以真的買到的東西**。
# 等權持有 150 檔需要每季調倉、付 150 次交易成本；0050 買一次就好。
# ══════════════════════════════════════════════════════════════

from taiwan_quant.validation.benchmarks import single_asset_equity  # noqa: E402


@pytest.mark.unit
def test_single_asset_equity_starts_at_one() -> None:
    data = by_stock({"0050": [100, 110, 120]})
    cal = list(pd.date_range("2024-01-01", periods=3, freq="B"))

    curve = single_asset_equity(data["0050"], cal)
    assert curve.iloc[0] == pytest.approx(1.0)


@pytest.mark.unit
def test_single_asset_equity_matches_hand_calculation() -> None:
    """100 → 120 = ×1.20"""
    data = by_stock({"0050": [100, 110, 120]})
    cal = list(pd.date_range("2024-01-01", periods=3, freq="B"))

    curve = single_asset_equity(data["0050"], cal)
    assert curve.iloc[-1] == pytest.approx(1.20)


@pytest.mark.unit
def test_single_asset_equity_forward_fills() -> None:
    """停牌沿用前一日價格，不是歸零"""
    bars = pd.DataFrame(
        {"close": [100.0, 110.0]},
        index=pd.DatetimeIndex(["2024-01-01", "2024-01-03"]),
    )
    cal = list(pd.date_range("2024-01-01", periods=3, freq="D"))

    curve = single_asset_equity(bars, cal)
    assert curve.iloc[1] == pytest.approx(1.0)
    assert curve.iloc[2] == pytest.approx(1.1)


@pytest.mark.unit
def test_single_asset_equity_uses_price_at_or_before_start() -> None:
    """
    期初基準取**不晚於起始日**的最後一筆價格。

    取起始日之後的第一筆等於用未來的價格當成本，那是 look-ahead。
    """
    bars = pd.DataFrame(
        {"close": [90.0, 100.0, 120.0]},
        index=pd.DatetimeIndex(["2023-12-28", "2024-01-02", "2024-01-03"]),
    )
    cal = list(pd.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-03"]))

    curve = single_asset_equity(bars, cal)
    # 2024-01-01 沒有報價 → 沿用 2023-12-28 的 90
    assert curve.iloc[0] == pytest.approx(1.0)
    assert curve.iloc[-1] == pytest.approx(120 / 90)


@pytest.mark.unit
def test_single_asset_equity_rejects_empty_calendar() -> None:
    data = by_stock({"0050": [100.0, 110.0]})
    with pytest.raises(ValueError, match="日曆"):
        single_asset_equity(data["0050"], [])


@pytest.mark.unit
def test_single_asset_equity_rejects_when_not_listed_at_start() -> None:
    """
    期初還沒上市時**拋錯**，不可用之後的第一筆價格當基準。

    回一條看起來正常的曲線，會讓報告拿一個起點錯誤的對照組去比。
    """
    bars = by_stock({"0050": [100.0, 110.0]}, start="2025-01-01")["0050"]
    cal = list(pd.date_range("2024-01-01", periods=2, freq="B"))

    with pytest.raises(ValueError, match="期初"):
        single_asset_equity(bars, cal)


@pytest.mark.unit
def test_single_asset_equity_max_drawdown_is_computable() -> None:
    """
    對照組的回撤要能跟策略比——只有總報酬時，
    「策略 +50% / 對照 +60%」看不出對照中途回撤了 40%。
    """
    bars = by_stock({"0050": [100, 120, 90, 95]})["0050"]
    cal = list(pd.date_range("2024-01-01", periods=4, freq="B"))

    curve = single_asset_equity(bars, cal)
    peak = curve.cummax()
    assert float(((peak - curve) / peak).max()) == pytest.approx(0.25)
