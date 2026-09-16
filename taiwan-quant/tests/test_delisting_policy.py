"""工作單 K：下市結算政策的手算測試。"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from taiwan_quant.validation.delisting import MissingPriceKind, settle_holding_period


@pytest.mark.unit
def test_delisted_position_settles_at_last_real_close() -> None:
    """
    T+1 以 100 元進場，持有期第 20 日永久停止交易，最後收盤 80 元。

    政策 B 在最後有價日結算，之後現金報酬為零，所以到第 40 日仍是 -20%。
    """
    calendar = list(pd.date_range("2023-01-02", periods=42, freq="B"))
    bars = pd.DataFrame(
        {
            "open": [100.0] * 21,
            "close": [100.0] * 20 + [80.0],
        },
        index=calendar[:21],
    )

    result = settle_holding_period(
        bars,
        entry_date=calendar[1],
        target_date=calendar[41],
        delisted_date=date(2023, 2, 10),
    )

    assert result.kind is MissingPriceKind.DELISTED
    assert result.exit_date == calendar[20]
    assert result.gross_return == pytest.approx(-0.20)


@pytest.mark.unit
def test_temporary_suspension_is_not_treated_as_delisting() -> None:
    """目標日缺價但之後恢復交易，必須標成停牌／缺資料，不准用舊價成交。"""
    calendar = list(pd.date_range("2023-01-02", periods=45, freq="B"))
    kept = calendar[:20] + calendar[42:]
    bars = pd.DataFrame({"open": 100.0, "close": 100.0}, index=kept)

    result = settle_holding_period(
        bars,
        entry_date=calendar[1],
        target_date=calendar[40],
        delisted_date=None,
    )

    assert result.kind is MissingPriceKind.SUSPENDED_OR_MISSING
    assert result.gross_return is None
    assert result.exit_date is None
