"""
還原股價計算測試

CLAUDE.md 禁令 12：一律用還原股價。

原本的還原價來自 yfinance，但它查不到已下市的股票——107 檔沒有還原價。
下市股票正是 survivorship 的關鍵樣本，不能因為拿不到還原價就放棄。

改由**官方除權息紀錄**推算：

    factor = 除權息參考價 / 除權息前收盤價

比對手算值，不拿程式輸出反填預期值。
"""

from __future__ import annotations

import pandas as pd
import pytest

from taiwan_quant.data.adjustment import (
    AdjustmentError,
    DividendEvent,
    back_adjust,
    events_from_dividend_result,
)


def closes(values: list[float], start: str = "2024-01-01") -> pd.Series:
    return pd.Series(
        values,
        index=pd.date_range(start, periods=len(values), freq="B"),
        dtype=float,
    )


# ══════════════════════════════════════════════════════════════
# 還原計算
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_no_events_returns_unchanged() -> None:
    """沒有除權息就不需要還原"""
    prices = closes([100.0, 110.0, 120.0])
    result = back_adjust(prices, [])
    pd.testing.assert_series_equal(result, prices)


@pytest.mark.unit
def test_single_event_matches_hand_calculation() -> None:
    """
    除權息日 2024-01-03，前收 100、參考價 95 → factor 0.95。

    **除息日之前**的價格要乘上 0.95，除息日當天與之後不動：

        01-01  100 × 0.95 = 95.0
        01-02  100 × 0.95 = 95.0
        01-03   95           ← 除息日當天已是除息後價格
        01-04   96
    """
    prices = closes([100.0, 100.0, 95.0, 96.0])
    events = [DividendEvent(pd.Timestamp("2024-01-03"), 100.0, 95.0)]

    result = back_adjust(prices, events)

    assert result.iloc[0] == pytest.approx(95.0)
    assert result.iloc[1] == pytest.approx(95.0)
    assert result.iloc[2] == pytest.approx(95.0)
    assert result.iloc[3] == pytest.approx(96.0)


@pytest.mark.unit
def test_multiple_events_compound() -> None:
    """
    兩次除息，因子要連乘。

        事件 A  2024-01-03  100 → 90   factor 0.90
        事件 B  2024-01-05   80 → 76   factor 0.95

    01-01 的價格在兩個事件之前 → × 0.90 × 0.95 = × 0.855
        100 × 0.855 = 85.5
    01-04 只在事件 B 之前 → × 0.95
         85 × 0.95 = 80.75
    """
    prices = closes([100.0, 100.0, 90.0, 85.0, 76.0])
    events = [
        DividendEvent(pd.Timestamp("2024-01-03"), 100.0, 90.0),
        DividendEvent(pd.Timestamp("2024-01-05"), 80.0, 76.0),
    ]

    result = back_adjust(prices, events)

    assert result.iloc[0] == pytest.approx(100.0 * 0.90 * 0.95)
    assert result.iloc[3] == pytest.approx(85.0 * 0.95)
    assert result.iloc[4] == pytest.approx(76.0)


@pytest.mark.unit
def test_event_after_price_range_adjusts_everything() -> None:
    """
    價格序列之後才發生的除息事件不影響任何一天。

    序列最後一天是 01-04，事件在 01-10——所有價格都在事件之前，
    所以**全部**要乘上 factor。這不是「忽略」，是正確的還原。
    """
    prices = closes([100.0, 100.0, 100.0, 100.0])
    events = [DividendEvent(pd.Timestamp("2024-01-10"), 100.0, 95.0)]

    result = back_adjust(prices, events)
    assert result.tolist() == pytest.approx([95.0] * 4)


@pytest.mark.unit
def test_event_before_price_range_has_no_effect() -> None:
    """序列開始之前的除息事件不該影響任何價格"""
    prices = closes([100.0, 110.0], start="2024-06-03")
    events = [DividendEvent(pd.Timestamp("2024-01-03"), 100.0, 95.0)]

    result = back_adjust(prices, events)
    pd.testing.assert_series_equal(result, prices)


@pytest.mark.unit
def test_stock_dividend_factor_can_exceed_typical_cash_range() -> None:
    """
    股票股利的因子比現金股利大得多。

    配股 1 元（每股配 0.1 股）：100 → 90.91，factor 0.9091。
    """
    prices = closes([100.0, 90.91])
    events = [DividendEvent(pd.Timestamp("2024-01-02"), 100.0, 90.91)]

    result = back_adjust(prices, events)
    assert result.iloc[0] == pytest.approx(90.91, abs=1e-6)


@pytest.mark.unit
def test_does_not_mutate_input() -> None:
    prices = closes([100.0, 100.0, 95.0])
    before = prices.copy()
    back_adjust(prices, [DividendEvent(pd.Timestamp("2024-01-03"), 100.0, 95.0)])
    pd.testing.assert_series_equal(prices, before)


@pytest.mark.unit
def test_rejects_non_positive_prices_in_event() -> None:
    """
    前收或參考價非正時**拋錯**。

    除以 0 會得到 inf，而 inf 不會讓後續的比較拋錯——它會靜默汙染
    整段還原價。必須在邊界擋掉。
    """
    with pytest.raises(AdjustmentError, match="必須為正"):
        DividendEvent(pd.Timestamp("2024-01-03"), 0.0, 95.0)
    with pytest.raises(AdjustmentError, match="必須為正"):
        DividendEvent(pd.Timestamp("2024-01-03"), 100.0, -1.0)


@pytest.mark.unit
def test_rejects_absurd_factor() -> None:
    """
    因子離 1 太遠代表資料有問題（例如前收與參考價來自不同檔）。

    放行會讓整段還原價偏掉數倍，而且曲線看起來仍然「正常」。
    """
    with pytest.raises(AdjustmentError, match="因子"):
        DividendEvent(pd.Timestamp("2024-01-03"), 100.0, 5.0)


@pytest.mark.unit
def test_factor_property() -> None:
    event = DividendEvent(pd.Timestamp("2024-01-03"), 100.0, 95.0)
    assert event.factor == pytest.approx(0.95)


# ══════════════════════════════════════════════════════════════
# 解析 FinMind 的 TaiwanStockDividendResult
# ══════════════════════════════════════════════════════════════


DIVIDEND_ROWS = [
    {"date": "2015-06-29", "stock_id": "2330",
     "before_price": 146.0, "after_price": 141.5,
     "stock_and_cache_dividend": 4.499875, "stock_or_cache_dividend": "息"},
    {"date": "2016-06-27", "stock_id": "2330",
     "before_price": 160.0, "after_price": 154.0,
     "stock_and_cache_dividend": 6.0, "stock_or_cache_dividend": "息"},
]


@pytest.mark.unit
def test_events_from_dividend_result_parses_rows() -> None:
    events = events_from_dividend_result(DIVIDEND_ROWS)

    assert len(events) == 2
    assert events[0].date == pd.Timestamp("2015-06-29")
    assert events[0].factor == pytest.approx(141.5 / 146.0)


@pytest.mark.unit
def test_events_from_dividend_result_sorts_by_date() -> None:
    events = events_from_dividend_result(list(reversed(DIVIDEND_ROWS)))
    assert [e.date for e in events] == sorted(e.date for e in events)


@pytest.mark.unit
def test_events_from_dividend_result_skips_unusable_rows() -> None:
    """
    無法解析的列**跳過**，不讓整檔失敗。

    但也不可補值——補 factor=1 等於宣稱那次除息沒有發生。
    """
    rows = DIVIDEND_ROWS + [
        {"date": "2017-06-29", "stock_id": "2330",
         "before_price": None, "after_price": 150.0},
        {"date": "", "stock_id": "2330", "before_price": 100.0, "after_price": 95.0},
        {"date": "2018-06-29", "stock_id": "2330",
         "before_price": 100.0, "after_price": 3.0},   # 因子離譜
    ]
    events = events_from_dividend_result(rows)
    assert len(events) == 2


@pytest.mark.unit
def test_events_from_dividend_result_empty() -> None:
    assert events_from_dividend_result([]) == []
