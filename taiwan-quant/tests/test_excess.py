"""
超額報酬標籤測試

## 為什麼要改標籤

現在的標籤是**絕對毛報酬**。在 2019-2026 這種等權 +689% 的多頭裡，
幾乎每一筆的標籤都是正的——模型學到的有一大部分是「市場會漲」，
不是「這檔比較強」。

選股是橫斷面任務，正確的目標是：

    超額報酬 = 個股報酬 − 同期間基準報酬

這樣才能把 beta 拿掉，隔離出真正的選股能力。

## 基準必須對齊同一個窗口

每筆持倉的進出場日期都不同（移動停損會提早出場），所以基準報酬要按
**該筆的實際進出場日**計算，不能用固定期間或當期平均。

用錯窗口不會拋錯，只會讓超額報酬多一層雜訊。
"""

from __future__ import annotations

import pandas as pd
import pytest

from taiwan_quant.labeling.excess import (
    ExcessError,
    benchmark_return,
    to_excess_return,
)


def index_series(values: list[float], start: str = "2024-01-01") -> pd.Series:
    return pd.Series(
        values,
        index=pd.date_range(start, periods=len(values), freq="B"),
        dtype=float,
    )


# ══════════════════════════════════════════════════════════════
# 基準報酬
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_benchmark_return_matches_hand_calculation() -> None:
    """指數 100 → 110，期間報酬 +10%"""
    idx = index_series([100.0, 105.0, 110.0])
    r = benchmark_return(idx, idx.index[0], idx.index[2])
    assert r == pytest.approx(0.10)


@pytest.mark.unit
def test_benchmark_uses_value_at_or_before_entry() -> None:
    """
    進場日若不在指數序列裡（停牌、假日），取**不晚於它**的最後一筆。

    取之後的第一筆等於用未來的值當成本——那是 look-ahead。
    """
    idx = pd.Series(
        [100.0, 120.0],
        index=pd.DatetimeIndex(["2024-01-01", "2024-01-05"]),
        dtype=float,
    )
    r = benchmark_return(idx, pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-05"))
    assert r == pytest.approx(0.20)


@pytest.mark.unit
def test_benchmark_returns_none_before_series_start() -> None:
    """
    進場日早於指數起點時回 `None`，不可外推。

    回 0 會讓那筆的超額報酬等於絕對報酬，靜默混入兩種標籤。
    """
    idx = index_series([100.0, 110.0], start="2024-06-03")
    assert benchmark_return(idx, pd.Timestamp("2024-01-01"),
                            pd.Timestamp("2024-06-04")) is None


@pytest.mark.unit
def test_benchmark_rejects_reversed_window() -> None:
    idx = index_series([100.0, 110.0, 120.0])
    with pytest.raises(ExcessError, match="不可晚於"):
        benchmark_return(idx, idx.index[2], idx.index[0])


@pytest.mark.unit
def test_benchmark_returns_none_on_non_positive_base() -> None:
    idx = index_series([0.0, 110.0, 120.0])
    assert benchmark_return(idx, idx.index[0], idx.index[2]) is None


# ══════════════════════════════════════════════════════════════
# 超額報酬
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_excess_matches_hand_calculation() -> None:
    """
    個股 +15%、同期基準 +10% → 超額 +5%（算術差）。

    用算術差而非幾何差，因為下游的成本模型也是算術扣除，
    兩者必須一致才不會在門檻比較時混用兩種尺度。
    """
    idx = index_series([100.0, 105.0, 110.0])
    r = to_excess_return(0.15, idx, idx.index[0], idx.index[2])
    assert r == pytest.approx(0.05)


@pytest.mark.unit
def test_excess_can_be_negative_in_bull_market() -> None:
    """
    多頭裡賺 8% 但基準賺 20% → 超額 −12%。

    **這正是改標籤的目的**：絕對報酬 +8% 看起來是好交易，
    但它其實輸給什麼都不做。
    """
    idx = index_series([100.0, 110.0, 120.0])
    r = to_excess_return(0.08, idx, idx.index[0], idx.index[2])
    assert r == pytest.approx(-0.12)


@pytest.mark.unit
def test_excess_is_none_when_benchmark_unavailable() -> None:
    """
    基準算不出來時整筆回 `None`，不可退回絕對報酬。

    混用兩種標籤會讓校準器同時學到兩種尺度，而且不會拋錯。
    """
    idx = index_series([100.0, 110.0], start="2024-06-03")
    assert to_excess_return(0.15, idx, pd.Timestamp("2024-01-01"),
                            pd.Timestamp("2024-06-04")) is None


@pytest.mark.unit
def test_excess_rejects_non_finite_gross() -> None:
    idx = index_series([100.0, 110.0, 120.0])
    with pytest.raises(ExcessError, match="有限值"):
        to_excess_return(float("nan"), idx, idx.index[0], idx.index[2])


@pytest.mark.unit
def test_excess_uses_each_positions_own_window() -> None:
    """
    每筆持倉用自己的進出場日算基準——移動停損會讓出場日不同。

    A 持有到第 3 天（基準 +10%），B 提早在第 2 天出場（基準 +5%）。
    兩筆的絕對報酬都是 +12%，但超額不同。
    """
    idx = index_series([100.0, 105.0, 110.0])
    a = to_excess_return(0.12, idx, idx.index[0], idx.index[2])
    b = to_excess_return(0.12, idx, idx.index[0], idx.index[1])

    assert a == pytest.approx(0.02)
    assert b == pytest.approx(0.07)
    assert a != b
