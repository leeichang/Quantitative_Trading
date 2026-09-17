"""工作單 J：檔數掃描彙總的手算測試。"""

from __future__ import annotations

import pytest

from scripts.diagnose_position_costs import resolve_pick_tier, summarize_rows
from taiwan_quant.config.costs import Tier


@pytest.mark.unit
def test_summarize_rows_matches_hand_calculation() -> None:
    """
    兩期手算：

    - 毛報酬：(10% + 2%) / 2 = 6%
    - 成本：(1% + 2%) / 2 = 1.5%
    - 淨報酬：(9% + 0%) / 2 = 4.5%
    - 每年兩趟的年化淨：(1.045)^2 - 1 = 9.2025%
    - 四筆成交中兩筆整股 = 50%；兩年只有一年平均淨報酬 > 0
    """
    rows = [
        {"year": 2022, "gross": 0.10, "cost": 0.01, "net": 0.09,
         "whole": 2, "trades": 2},
        {"year": 2023, "gross": 0.02, "cost": 0.02, "net": 0.00,
         "whole": 0, "trades": 2},
    ]

    result = summarize_rows(rows, trips_per_year=2.0)

    assert result["gross_per_trip"] == pytest.approx(0.06)
    assert result["cost_per_trip"] == pytest.approx(0.015)
    assert result["net_per_trip"] == pytest.approx(0.045)
    assert result["annualized_net"] == pytest.approx(0.092025)
    assert result["whole_ratio"] == pytest.approx(0.5)
    assert result["positive_years"] == 1
    assert result["years"] == 2


@pytest.mark.unit
def test_summarize_rows_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="不可為空"):
        summarize_rows([], trips_per_year=6.3)


@pytest.mark.unit
def test_position_cost_scan_routes_historical_mid_cap_to_mid_tier() -> None:
    """同一價格下，決策日不在 top 50 的股票必須走 0051 零股 0.4%。"""
    tier = resolve_pick_tier(
        "9999", actual_price=100.0, adjusted_price=80.0,
        amount=40_000, is_large=False,
    )
    assert tier is Tier.MID
