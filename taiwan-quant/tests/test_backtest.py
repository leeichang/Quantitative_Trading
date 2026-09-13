#!/usr/bin/env python3
"""
週頻回測引擎測試

CLAUDE.md 規格 13、15（來自 qlib-tw-trader 驗證的實測教訓）：

    13. 換手率是回測的一級輸出，不是事後才算
        實測：週換手 9.9% → 年化成本 5.51%；271.5% → 115.56%

    15. 成本敏感度至少三檔並列
        實測：+0.41%（無滑價）vs −19.08%（含滑價），差 19.5 個百分點

所以本引擎的介面刻意讓「換手率」與「成本情境」無法被忽略：
`run_backtest()` 回傳的 `BacktestResult` 一定帶 `weekly_turnover` 與
`annual_cost_drag`，而 `run_cost_sensitivity()` 一次跑完整組情境。

預期值全部手算在註解裡。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from taiwan_quant.backtest.engine import (
    BacktestResult,
    TradePlan,
    WeeklyOutcome,
    run_backtest,
    run_cost_sensitivity,
)
from taiwan_quant.config.costs import DEFAULT, FEE_TAX_ONLY, GROSS, Tier

pytestmark = pytest.mark.unit


def plan(
    week: str,
    stock_id: str = "2330",
    gross_return: float = 0.05,
    holding_days: int = 5,
    tier: str = "0050",
) -> TradePlan:
    """建一筆已實現的交易計畫"""
    return TradePlan(
        week_id=week,
        stock_id=stock_id,
        gross_return=gross_return,
        holding_days=holding_days,
        tier=Tier(tier),
    )


def weeks(n: int, gross_return: float = 0.05, positions: int = 1) -> list[TradePlan]:
    """產生 n 週、每週 `positions` 筆計畫"""
    return [
        plan(f"2026W{w:02d}", stock_id=f"s{p}", gross_return=gross_return)
        for w in range(1, n + 1)
        for p in range(positions)
    ]


# ══════════════════════════════════════════════════════════════
# 報酬彙總
# ══════════════════════════════════════════════════════════════


def test_gross_return_compounds_weekly() -> None:
    """
    手算：4 週、每週單一部位毛報酬 +5%
        累積 = 1.05^4 − 1 = 0.21550625
    """
    result = run_backtest(weeks(4, gross_return=0.05), cost=GROSS)
    assert result.gross_cumulative_return == pytest.approx(1.05**4 - 1)


def test_positions_within_week_are_equal_weighted() -> None:
    """
    同一週的多個部位等權平均，不是相加。

    手算：某週兩檔，+10% 與 0% → 該週報酬 = (0.10 + 0.00)/2 = 0.05
    """
    plans = [
        plan("2026W01", "A", gross_return=0.10),
        plan("2026W01", "B", gross_return=0.00),
    ]
    result = run_backtest(plans, cost=GROSS)
    assert result.gross_cumulative_return == pytest.approx(0.05)


def test_net_return_deducts_round_trip_cost() -> None:
    """
    手算：1 週、毛報酬 +5%、0050 分層、預設成本
        一趟來回成本率 = 1.071%
        淨報酬 = 0.05 − 0.01071 = 0.03929
    """
    result = run_backtest([plan("2026W01", gross_return=0.05)], cost=DEFAULT)
    assert result.net_cumulative_return == pytest.approx(0.03929)


def test_mid_tier_costs_more_than_large() -> None:
    """
    中型股滑價 0.4% vs 0.3%，來回差 0.2 個百分點。

    手算：0050 來回 1.071%、0051 來回 1.271%
        淨報酬差 = 0.01271 − 0.01071 = 0.002
    """
    large = run_backtest([plan("2026W01", tier="0050")], cost=DEFAULT)
    mid = run_backtest([plan("2026W01", tier="0051")], cost=DEFAULT)
    diff = large.net_cumulative_return - mid.net_cumulative_return
    assert diff == pytest.approx(0.002)


def test_gross_and_net_differ_by_cost() -> None:
    """毛淨必須並列且不同——只報一個就是 qlib-tw-trader 的錯"""
    result = run_backtest(weeks(10), cost=DEFAULT)
    assert result.gross_cumulative_return > result.net_cumulative_return


# ══════════════════════════════════════════════════════════════
# 換手率（規格 13）
# ══════════════════════════════════════════════════════════════


def test_turnover_is_one_when_fully_replaced_each_week() -> None:
    """
    每週換掉全部部位 → 週換手率 100%。

    手算：平均持有 5 天、一週 5 個交易日 → 每週剛好換一輪 → 1.0
    """
    result = run_backtest(weeks(10, gross_return=0.0), cost=DEFAULT)
    assert result.weekly_turnover == pytest.approx(1.0)


def test_turnover_halves_when_holding_twice_as_long() -> None:
    """
    平均持有 10 天 → 每週只換半輪 → 週換手率 50%。

    持有期是成本的主導因素，這條把它釘住。
    """
    plans = [plan(f"2026W{w:02d}", holding_days=10) for w in range(1, 11)]
    result = run_backtest(plans, cost=DEFAULT)
    assert result.weekly_turnover == pytest.approx(0.5)


def test_annual_cost_drag_matches_turnover() -> None:
    """
    手算：週換手 100%、0050 來回成本 1.071%
        年化拖累 = 1.0 × 52 × 0.01071 = 0.55692  →  55.69%
    """
    result = run_backtest(weeks(10, gross_return=0.0), cost=DEFAULT)
    assert result.annual_cost_drag == pytest.approx(0.55692, abs=1e-6)


def test_turnover_is_always_reported() -> None:
    """
    換手率是一級輸出——即使呼叫端用毛報酬模型也必須算出來。

    qlib-tw-trader 的問題正是換手率只在離線腳本算，API 路徑完全沒有，
    結果日度調倉的 271.5% 換手率長期無人察覺。
    """
    result = run_backtest(weeks(5), cost=GROSS)
    assert result.weekly_turnover > 0
    assert result.annual_cost_drag == pytest.approx(0.0), "毛報酬模型的拖累應為 0"


# ══════════════════════════════════════════════════════════════
# 風險指標
# ══════════════════════════════════════════════════════════════


def test_max_drawdown_on_known_path() -> None:
    """
    手算：週報酬 +10%, −20%, +5%（用毛報酬模型避免成本干擾）
        權益 1.0 → 1.10 → 0.88 → 0.924
        高點 1.10，谷底 0.88
        最大回撤 = (1.10 − 0.88)/1.10 = 0.2
    """
    plans = [
        plan("2026W01", gross_return=0.10),
        plan("2026W02", gross_return=-0.20),
        plan("2026W03", gross_return=0.05),
    ]
    result = run_backtest(plans, cost=GROSS)
    assert result.max_drawdown == pytest.approx(0.2)


def test_max_drawdown_is_zero_on_monotonic_rise() -> None:
    result = run_backtest(weeks(5, gross_return=0.05), cost=GROSS)
    assert result.max_drawdown == pytest.approx(0.0)


def test_win_rate_on_known_outcomes() -> None:
    """
    手算：4 週中 3 週為正 → 勝率 75%
    """
    plans = [
        plan("2026W01", gross_return=0.05),
        plan("2026W02", gross_return=0.05),
        plan("2026W03", gross_return=-0.05),
        plan("2026W04", gross_return=0.05),
    ]
    result = run_backtest(plans, cost=GROSS)
    assert result.win_rate == pytest.approx(0.75)


def test_sharpe_is_none_with_single_week() -> None:
    """單一樣本無法算標準差，必須回 None 而非 0 或 inf"""
    assert run_backtest([plan("2026W01")], cost=GROSS).sharpe is None


def test_sharpe_is_none_when_zero_variance() -> None:
    """報酬完全相同 → 標準差 0 → Sharpe 無定義，回 None"""
    assert run_backtest(weeks(5, gross_return=0.05), cost=GROSS).sharpe is None


def test_sharpe_on_known_series() -> None:
    """
    手算：週報酬 +10%, 0%（毛報酬模型）
        平均 = 0.05
        樣本標準差（ddof=1）= sqrt(((0.10−0.05)² + (0−0.05)²)/1) = 0.0707107
        年化 Sharpe = 0.05/0.0707107 × sqrt(52) = 0.70711 × 7.2111 = 5.0990
    """
    plans = [
        plan("2026W01", gross_return=0.10),
        plan("2026W02", gross_return=0.00),
    ]
    result = run_backtest(plans, cost=GROSS)
    assert result.sharpe == pytest.approx(0.05 / 0.07071068 * np.sqrt(52), rel=1e-5)


def test_annualized_return_from_weeks() -> None:
    """
    手算：52 週、每週 +0.1%（毛報酬）
        累積 = 1.001^52 − 1 = 0.0533...
        年化 = 相同（剛好一年）
    """
    result = run_backtest(weeks(52, gross_return=0.001), cost=GROSS)
    assert result.annualized_return == pytest.approx(result.gross_cumulative_return, rel=1e-6)


# ══════════════════════════════════════════════════════════════
# 週別明細
# ══════════════════════════════════════════════════════════════


def test_weekly_outcomes_preserve_order() -> None:
    result = run_backtest(weeks(3), cost=DEFAULT)
    assert [o.week_id for o in result.weekly] == ["2026W01", "2026W02", "2026W03"]


def test_weekly_outcome_records_positions_and_cost() -> None:
    plans = [plan("2026W01", "A"), plan("2026W01", "B")]
    outcome = run_backtest(plans, cost=DEFAULT).weekly[0]
    assert outcome.positions == 2
    assert outcome.cost_rate > 0
    assert outcome.gross_return > outcome.net_return


def test_weeks_sorted_even_if_input_unordered() -> None:
    """輸入順序亂掉不可影響複合順序——複合是有序運算"""
    plans = [plan("2026W03"), plan("2026W01"), plan("2026W02")]
    result = run_backtest(plans, cost=GROSS)
    assert [o.week_id for o in result.weekly] == ["2026W01", "2026W02", "2026W03"]


# ══════════════════════════════════════════════════════════════
# 輸入驗證
# ══════════════════════════════════════════════════════════════


def test_rejects_empty_plans() -> None:
    with pytest.raises(ValueError, match="不可為空"):
        run_backtest([], cost=DEFAULT)


def test_rejects_nonpositive_holding_days() -> None:
    with pytest.raises(ValueError, match="holding_days"):
        TradePlan("2026W01", "2330", 0.05, 0, Tier.LARGE)


def test_rejects_non_finite_return() -> None:
    """NaN 報酬會汙染整個複合結果，必須在邊界擋掉"""
    with pytest.raises(ValueError, match="gross_return"):
        TradePlan("2026W01", "2330", float("nan"), 5, Tier.LARGE)


def test_result_is_immutable() -> None:
    result = run_backtest(weeks(3), cost=DEFAULT)
    with pytest.raises(Exception):
        result.weekly_turnover = 0.0  # type: ignore[misc]


# ══════════════════════════════════════════════════════════════
# 成本敏感度（規格 15）
# ══════════════════════════════════════════════════════════════


def test_cost_sensitivity_covers_all_scenarios() -> None:
    """報告必須一次並列所有成本情境，不可只報一個"""
    results = run_cost_sensitivity(weeks(20))
    assert len(results) >= 3
    assert "無成本（毛報酬）" in results
    assert any("滑價" in k for k in results)


def test_cost_sensitivity_is_monotonic() -> None:
    """成本越高，淨報酬越低——順序反了就是算錯"""
    results = run_cost_sensitivity(weeks(20, gross_return=0.02))
    gross = results["無成本（毛報酬）"].net_cumulative_return
    fee_tax = results["僅手續費+證交稅（無滑價）"].net_cumulative_return
    full = results["6折+滑價（預設）"].net_cumulative_return
    assert gross > fee_tax > full


def test_cost_sensitivity_quantifies_slippage_impact() -> None:
    """
    滑價的影響必須可量化——這是 qlib-tw-trader 差 19.5 個百分點的來源。

    手算（1 週、毛報酬 0%、0050）：
        僅手續費+稅（無折扣上界不適用，預設 6 折）：0.001425×0.6×2 + 0.003 = 0.00471
        含滑價：0.00471 + 0.003×2 = 0.01071
        差 = 0.006（0.6 個百分點／趟）
    """
    plans = [plan("2026W01", gross_return=0.0)]
    fee_tax = run_backtest(plans, cost=FEE_TAX_ONLY).net_cumulative_return
    full = run_backtest(plans, cost=DEFAULT).net_cumulative_return
    assert fee_tax - full == pytest.approx(0.006, abs=1e-9)


def test_sensitivity_all_share_same_turnover() -> None:
    """
    換手率是策略屬性，與成本模型無關——所有情境的換手率必須相同。

    若不同就代表換手率計算被成本模型汙染了。
    """
    results = run_cost_sensitivity(weeks(20))
    turnovers = {r.weekly_turnover for r in results.values()}
    assert len(turnovers) == 1


# ══════════════════════════════════════════════════════════════
# 報告輸出
# ══════════════════════════════════════════════════════════════


def test_describe_includes_turnover_and_cost() -> None:
    """報告文字必須含換手率與成本拖累（規格 13）"""
    text = run_backtest(weeks(10), cost=DEFAULT).describe()
    assert "換手率" in text
    assert "成本拖累" in text
    assert "毛報酬" in text and "淨報酬" in text


# ══════════════════════════════════════════════════════════════
# 期間長度：年化與 Sharpe 的單位
#
# 實測 bug：validate_oos.py 用 40 個交易日為一期餵進引擎，
# 但引擎預設每期是一週（52 期/年），算出「年化 122,560%」與
# 「Sharpe 5.759」這種荒謬數字。
#
# 累積報酬不受影響（純複合），但**年化與 Sharpe 都依賴「一年幾期」**。
# 這個單位必須由呼叫端明示。
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_annualization_respects_periods_per_year() -> None:
    """
    手算：4 期、每期 +5%（毛報酬模型）
        累積 = 1.05^4 − 1 = 0.21550625

    每期一季（4 期/年）：年化 = 1.2155^(4/4) − 1 = 0.21550625
    每期一週（52 期/年）：年化 = 1.2155^(52/4) − 1 = 1.2155^13 − 1
                                = e^(13 × 0.19516) − 1 = 12.64 − 1 = 11.64

    兩者差 54 倍——同一組交易，年化差這麼多，單位錯了結論就全錯。
    """
    plans = weeks(4, gross_return=0.05)

    quarterly = run_backtest(plans, cost=GROSS, periods_per_year=4)
    assert quarterly.annualized_return == pytest.approx(1.05**4 - 1)

    weekly = run_backtest(plans, cost=GROSS, periods_per_year=52)
    assert weekly.annualized_return == pytest.approx(1.05**52 - 1, rel=1e-6)
    assert weekly.annualized_return > quarterly.annualized_return * 10


@pytest.mark.unit
def test_sharpe_respects_periods_per_year() -> None:
    """
    Sharpe 的年化因子是 sqrt(期數/年)。

    手算：期報酬 +10%, 0%
        平均 0.05、樣本標準差 0.07071068
        每年 4 期 → Sharpe = 0.05/0.07071068 × sqrt(4) = 1.41421
        每年 52 期 → × sqrt(52) = 5.09902
    """
    plans = [
        plan("2026W01", gross_return=0.10),
        plan("2026W02", gross_return=0.00),
    ]
    quarterly = run_backtest(plans, cost=GROSS, periods_per_year=4)
    assert quarterly.sharpe == pytest.approx(0.05 / 0.07071068 * 2.0, rel=1e-5)


@pytest.mark.unit
def test_cost_drag_respects_periods_per_year() -> None:
    """
    年化成本拖累也依賴期數。

    持有 40 日、每年約 252/40 = 6.3 期，換手率 100%（每期換一輪）
    → 年化來回 6.3 次，遠少於週頻的 52 次。
    """
    plans = [plan(f"2026W{w:02d}", holding_days=40) for w in range(1, 9)]

    quarterly = run_backtest(plans, cost=DEFAULT, periods_per_year=252 / 40)
    weekly = run_backtest(plans, cost=DEFAULT, periods_per_year=52)

    assert quarterly.annual_cost_drag < weekly.annual_cost_drag


@pytest.mark.unit
def test_default_periods_per_year_is_weekly() -> None:
    """未指定時維持週頻預設，不改變既有呼叫端行為"""
    plans = weeks(4, gross_return=0.05)
    assert (
        run_backtest(plans, cost=GROSS).annualized_return
        == run_backtest(plans, cost=GROSS, periods_per_year=52).annualized_return
    )


@pytest.mark.unit
def test_rejects_nonpositive_periods_per_year() -> None:
    with pytest.raises(ValueError, match="periods_per_year"):
        run_backtest(weeks(3), cost=GROSS, periods_per_year=0)
