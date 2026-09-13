"""
週頻回測引擎

依據 D7（週頻決策）與 CLAUDE.md 規格 13、15。

## 介面為什麼長這樣

兩條規格來自 qlib-tw-trader 的實測教訓，所以刻意設計成「無法忽略」：

**規格 13：換手率是一級輸出。**
`BacktestResult` 一定帶 `weekly_turnover` 與 `annual_cost_drag`，
即使呼叫端用毛報酬模型也會算出來。qlib-tw-trader 的問題正是換手率只在
離線腳本算、API 路徑完全沒有，結果日度調倉的 271.5% 週換手率長期無人察覺。

**規格 15：成本敏感度至少三檔並列。**
`run_cost_sensitivity()` 一次跑完 `SENSITIVITY_SET` 的所有情境。
實測「無滑價 +0.41%」vs「含滑價 −19.08%」差 19.5 個百分點——
只報一個數字等於沒報。

## 邊界

本引擎吃的是**已實現的交易計畫**（`TradePlan`），不做選股也不做標記。
職責分離：
    labeling/   決定進出場與毛報酬
    ranking/    決定每週挑哪幾檔
    backtest/   把計畫彙總成績效，扣成本，算換手率

這樣回測層不需要知道任何策略細節，也不可能偷看未來——它只看得到
已經發生的結果。
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from itertools import groupby

from taiwan_quant.config.costs import (
    DEFAULT,
    SENSITIVITY_SET,
    WEEKS_PER_YEAR,
    CostModel,
    Tier,
    annual_cost_drag,
)

TRADING_DAYS_PER_WEEK = 5.0


@dataclass(frozen=True)
class TradePlan:
    """
    一筆已實現的交易計畫。

    由 `labeling` 與 `ranking` 產出，回測層只負責彙總。
    """

    week_id: str
    """決策週（例如 2026W37），同一週的計畫會被等權合併"""

    stock_id: str

    gross_return: float
    """毛報酬率（不含交易成本）"""

    holding_days: int
    """實際持有交易日數，決定換手率"""

    tier: Tier
    """流動性分層，決定滑價"""

    def __post_init__(self) -> None:
        if self.holding_days < 1:
            raise ValueError(f"holding_days 至少為 1，得到 {self.holding_days}")
        if not math.isfinite(self.gross_return):
            raise ValueError(
                f"gross_return 必須為有限值，得到 {self.gross_return}。"
                "NaN 會汙染整個複合結果且不會拋錯——必須在邊界擋掉。"
            )


@dataclass(frozen=True)
class WeeklyOutcome:
    """單週彙總結果"""

    week_id: str
    positions: int
    gross_return: float
    net_return: float
    cost_rate: float
    """該週的平均來回成本率"""

    avg_holding_days: float


@dataclass(frozen=True)
class BacktestResult:
    """
    回測結果。

    毛淨並列是刻意的——只報一個數字就是 qlib-tw-trader 的錯。
    """

    cost_label: str
    weeks: int
    trades: int

    gross_cumulative_return: float
    net_cumulative_return: float
    annualized_return: float
    """以淨報酬年化"""

    sharpe: float | None
    """年化 Sharpe（無風險利率視為 0）；樣本不足或零變異時為 None"""

    max_drawdown: float
    win_rate: float

    weekly_turnover: float
    """平均週換手率（單邊）。規格 13 的一級輸出"""

    annual_cost_drag: float
    """由換手率推算的年化成本拖累"""

    weekly: tuple[WeeklyOutcome, ...]

    def describe(self) -> str:
        sharpe = f"{self.sharpe:.3f}" if self.sharpe is not None else "n/a"
        return "\n".join(
            [
                f"成本情境        {self.cost_label}",
                f"樣本            {self.weeks} 週 / {self.trades} 筆交易",
                f"毛報酬（累積）   {self.gross_cumulative_return * 100:+.2f}%",
                f"淨報酬（累積）   {self.net_cumulative_return * 100:+.2f}%",
                f"年化報酬        {self.annualized_return * 100:+.2f}%",
                f"Sharpe          {sharpe}",
                f"最大回撤        {self.max_drawdown * 100:.2f}%",
                f"勝率            {self.win_rate * 100:.1f}%",
                f"週換手率        {self.weekly_turnover * 100:.1f}%",
                f"年化成本拖累     {self.annual_cost_drag * 100:.2f}%",
            ]
        )


# ══════════════════════════════════════════════════════════════
# 內部計算
# ══════════════════════════════════════════════════════════════


def _compound(returns: list[float]) -> float:
    """序列複合為累積報酬"""
    acc = 1.0
    for r in returns:
        acc *= 1.0 + r
    return acc - 1.0


def _max_drawdown(returns: list[float]) -> float:
    """由週報酬序列算最大回撤（正數）"""
    equity = 1.0
    peak = 1.0
    worst = 0.0
    for r in returns:
        equity *= 1.0 + r
        peak = max(peak, equity)
        worst = max(worst, (peak - equity) / peak)
    return worst


def _sharpe(returns: list[float], periods_per_year: float) -> float | None:
    """
    年化 Sharpe（無風險利率視為 0）。

    年化因子是 sqrt(期數/年)。**這個期數必須與實際的期間長度相符**——
    用 40 個交易日為一期的資料套 52 期/年，Sharpe 會被高估約 2.9 倍。

    樣本 < 2 或零變異時回 None——不可回 0 或 inf，
    那會讓下游誤以為「算出來了、結果很差」而非「無法計算」。
    """
    if len(returns) < 2:
        return None
    sd = statistics.stdev(returns)
    if sd == 0:
        return None
    return statistics.fmean(returns) / sd * math.sqrt(periods_per_year)


def _annualize(cumulative: float, periods: int, periods_per_year: float) -> float:
    """
    累積報酬年化。

    同樣依賴「一年幾期」。實測 bug：用 40 交易日為一期的資料套
    52 期/年，8 期的 +198% 被年化成 +122,560%。
    """
    if periods <= 0:
        return 0.0
    growth = 1.0 + cumulative
    if growth <= 0:
        return -1.0
    return growth ** (periods_per_year / periods) - 1.0


def _weekly_turnover(plans: list[TradePlan]) -> float:
    """
    平均週換手率（單邊）。

        每週換手率 = 一週交易日數 / 平均持有交易日數

    持有 5 天 → 每週剛好換一輪（1.0）
    持有 10 天 → 每週換半輪（0.5）
    持有 1 天 → 每週換五輪（5.0）

    這條公式把「持有期」與「成本」直接綁在一起，讓規格 13 的
    「持有期是成本的主導因素」在程式裡看得見。
    """
    avg_holding = statistics.fmean(p.holding_days for p in plans)
    if avg_holding <= 0:
        return 0.0
    return TRADING_DAYS_PER_WEEK / avg_holding


# ══════════════════════════════════════════════════════════════
# 回測
# ══════════════════════════════════════════════════════════════


def run_backtest(
    plans: list[TradePlan],
    cost: CostModel = DEFAULT,
    cost_label: str = "",
    periods_per_year: float = WEEKS_PER_YEAR,
) -> BacktestResult:
    """
    把已實現的交易計畫彙總成績效。

    Args:
        plans: 交易計畫清單（可亂序，會依 week_id 排序）
        cost: 成本模型（一律來自 config/costs.py）
        cost_label: 情境名稱，用於報告
        periods_per_year: 一年有幾期。預設 52（週頻）。

            **每期不是一週時必須明示。** 年化報酬、Sharpe、成本拖累
            三者都依賴這個值。以 40 個交易日為一期時應傳 252/40 ≈ 6.3。

    Returns:
        BacktestResult（毛淨並列，含換手率與成本拖累）

    Raises:
        ValueError: 計畫清單為空

    同一週的多筆計畫等權平均（不是相加）——三檔各 +10% 是該週 +10%，
    不是 +30%。
    """
    if not plans:
        raise ValueError("交易計畫清單不可為空")
    if periods_per_year <= 0:
        raise ValueError(f"periods_per_year 必須為正，得到 {periods_per_year}")

    ordered = sorted(plans, key=lambda p: p.week_id)

    weekly: list[WeeklyOutcome] = []
    for week_id, group in groupby(ordered, key=lambda p: p.week_id):
        batch = list(group)
        gross = statistics.fmean(p.gross_return for p in batch)
        cost_rate = statistics.fmean(cost.round_trip_rate(p.tier) for p in batch)
        weekly.append(
            WeeklyOutcome(
                week_id=week_id,
                positions=len(batch),
                gross_return=gross,
                net_return=gross - cost_rate,
                cost_rate=cost_rate,
                avg_holding_days=statistics.fmean(p.holding_days for p in batch),
            )
        )

    gross_series = [o.gross_return for o in weekly]
    net_series = [o.net_return for o in weekly]

    net_cumulative = _compound(net_series)
    turnover = _weekly_turnover(ordered)
    # 拖累用實際換手率與該情境的成本率推算；毛報酬模型的成本率為 0
    representative_tier = ordered[0].tier
    drag = annual_cost_drag(
        turnover, cost, representative_tier, weeks_per_year=periods_per_year
    )

    return BacktestResult(
        cost_label=cost_label or "自訂成本模型",
        weeks=len(weekly),
        trades=len(ordered),
        gross_cumulative_return=_compound(gross_series),
        net_cumulative_return=net_cumulative,
        annualized_return=_annualize(net_cumulative, len(weekly), periods_per_year),
        sharpe=_sharpe(net_series, periods_per_year),
        max_drawdown=_max_drawdown(net_series),
        win_rate=sum(1 for r in net_series if r > 0) / len(net_series),
        weekly_turnover=turnover,
        annual_cost_drag=drag,
        weekly=tuple(weekly),
    )


def run_cost_sensitivity(
    plans: list[TradePlan],
    scenarios: dict[str, CostModel] | None = None,
    periods_per_year: float = WEEKS_PER_YEAR,
) -> dict[str, BacktestResult]:
    """
    一次跑完所有成本情境（CLAUDE.md 規格 15）。

    Args:
        plans: 交易計畫清單
        scenarios: 情境字典；None 表示用 `config.costs.SENSITIVITY_SET`

    Returns:
        {情境名稱: 結果}

    為什麼要強制並列：實測「無滑價 +0.41%」vs「含滑價 −19.08%」差
    19.5 個百分點。只報一個數字，讀者無法判斷結論有多脆弱。
    """
    scenarios = scenarios if scenarios is not None else SENSITIVITY_SET
    return {
        label: run_backtest(
            plans, cost=model, cost_label=label, periods_per_year=periods_per_year
        )
        for label, model in scenarios.items()
    }


def describe_sensitivity(results: dict[str, BacktestResult]) -> str:
    """把成本敏感度結果排成對照表"""
    header = (
        f"{'情境':<26}{'毛報酬':>10}{'淨報酬':>10}{'年化':>10}"
        f"{'Sharpe':>9}{'MaxDD':>9}{'勝率':>8}{'年化成本':>10}"
    )
    lines = ["─" * 92, header, "─" * 92]

    for label, r in results.items():
        sharpe = f"{r.sharpe:.3f}" if r.sharpe is not None else "n/a"
        lines.append(
            f"{label:<26}{r.gross_cumulative_return * 100:>9.2f}%"
            f"{r.net_cumulative_return * 100:>9.2f}%"
            f"{r.annualized_return * 100:>9.2f}%"
            f"{sharpe:>9}{r.max_drawdown * 100:>8.2f}%"
            f"{r.win_rate * 100:>7.1f}%{r.annual_cost_drag * 100:>9.2f}%"
        )

    first = next(iter(results.values()))
    lines.append("─" * 92)
    lines.append(
        f"週換手率 {first.weekly_turnover * 100:.1f}%"
        f"（策略屬性，與成本模型無關）｜"
        f"樣本 {first.weeks} 週 / {first.trades} 筆"
    )
    return "\n".join(lines)
