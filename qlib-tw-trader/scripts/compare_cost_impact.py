#!/usr/bin/env python3
"""
交易成本對回測績效的實測影響

直接呼叫 WalkForwardBacktester（不經 API），用不同成本模型跑同一段
Walk-Forward 區間，輸出毛報酬 vs 淨報酬對照與實測換手率。

為什麼不走 API：FastAPI 的 response schema 會過濾掉 WeekResult 新增的
gross_return / one_way_turnover / cost_pct 欄位，而這些正是本次要量測的。

用法：
    PYTHONPATH=. .venv/bin/python scripts/compare_cost_impact.py 2026W16 2026W35
    PYTHONPATH=. .venv/bin/python scripts/compare_cost_impact.py 2026W16 2026W35 --capital 400000

輸出：
    stdout 人可讀報告
    scripts/output/cost_impact_<timestamp>.json 原始數據
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

from config.costs import DEFAULT, GROSS, NO_DISCOUNT, CostModel, Tier
from src.repositories.database import get_session
from src.services.walk_forward_backtester import WalkForwardBacktester

OUTPUT_DIR = Path("scripts/output")
WEEKS_PER_YEAR = 52


@dataclass
class ScenarioResult:
    """單一成本情境的績效"""

    label: str
    weeks: int
    cumulative_return: float
    annualized_return: float
    sharpe: float | None
    max_drawdown: float
    win_rate: float
    avg_weekly_turnover: float
    total_cost_pct: float
    annualized_cost_drag: float


def _compound(weekly_pct: list[float]) -> float:
    """週報酬（%）複合為累積報酬（%）"""
    acc = 1.0
    for r in weekly_pct:
        acc *= 1 + r / 100
    return (acc - 1) * 100


def _max_drawdown(weekly_pct: list[float]) -> float:
    """由週報酬序列算最大回撤（%，正數）"""
    equity = 1.0
    peak = 1.0
    worst = 0.0
    for r in weekly_pct:
        equity *= 1 + r / 100
        peak = max(peak, equity)
        worst = max(worst, (peak - equity) / peak)
    return worst * 100


def _sharpe(weekly_pct: list[float]) -> float | None:
    """年化 Sharpe（無風險利率視為 0）"""
    if len(weekly_pct) < 2:
        return None
    mean = statistics.fmean(weekly_pct)
    sd = statistics.stdev(weekly_pct)
    if sd == 0:
        return None
    return (mean / sd) * math.sqrt(WEEKS_PER_YEAR)


def _annualize(cumulative_pct: float, weeks: int) -> float:
    """累積報酬年化（%）"""
    if weeks <= 0:
        return 0.0
    growth = 1 + cumulative_pct / 100
    if growth <= 0:
        return -100.0
    return (growth ** (WEEKS_PER_YEAR / weeks) - 1) * 100


def summarize(label: str, details: list, use_gross: bool) -> ScenarioResult:
    """把 weekly_details 匯總成一個情境結果"""
    returns = [
        (d.gross_return if use_gross else d.week_return)
        for d in details
        if (d.gross_return if use_gross else d.week_return) is not None
    ]
    turnovers = [d.one_way_turnover for d in details if d.one_way_turnover is not None]
    costs = [d.cost_pct for d in details if d.cost_pct is not None]

    weeks = len(returns)
    cumulative = _compound(returns)
    # 毛報酬情境本身不扣成本，成本欄位歸零以免誤讀
    total_cost = 0.0 if use_gross else sum(costs)

    return ScenarioResult(
        label=label,
        weeks=weeks,
        cumulative_return=cumulative,
        annualized_return=_annualize(cumulative, weeks),
        sharpe=_sharpe(returns),
        max_drawdown=_max_drawdown(returns),
        win_rate=100 * sum(1 for r in returns if r > 0) / weeks if weeks else 0.0,
        avg_weekly_turnover=statistics.fmean(turnovers) if turnovers else 0.0,
        total_cost_pct=total_cost,
        annualized_cost_drag=total_cost * WEEKS_PER_YEAR / weeks if weeks else 0.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="交易成本對回測績效的實測影響")
    parser.add_argument("start_week", help="起始週，例如 2026W16")
    parser.add_argument("end_week", help="結束週，例如 2026W35")
    parser.add_argument("--capital", type=float, default=400_000.0, help="初始資金")
    parser.add_argument("--positions", type=int, default=10, help="最大持倉檔數")
    args = parser.parse_args()

    scenarios: list[tuple[str, CostModel, Tier]] = [
        ("無成本（毛報酬）", GROSS, Tier.LARGE),
        ("原專案模型（手續費+證交稅，無滑價）", GROSS, Tier.LARGE),
        ("6折 + 0.3% 滑價（0050）", DEFAULT, Tier.LARGE),
        ("6折 + 0.4% 滑價（0051 中型股）", DEFAULT, Tier.MID),
        ("無折扣 + 0.3% 滑價", NO_DISCOUNT, Tier.LARGE),
    ]

    session = get_session()
    try:
        backtester = WalkForwardBacktester(session)

        # 同一次回測即可同時取得毛與淨（WeekReturnBreakdown 兩者並存），
        # 但不同費率要各跑一次。第一個情境用毛報酬欄位。
        results: list[ScenarioResult] = []
        market_returns: list[float] = []
        ic_summary: dict = {}

        for i, (label, cost, tier) in enumerate(scenarios):
            print(f"跑第 {i + 1}/{len(scenarios)} 個情境：{label} ...", flush=True)
            outcome = backtester.run(
                start_week_id=args.start_week,
                end_week_id=args.end_week,
                initial_capital=args.capital,
                max_positions=args.positions,
                cost=cost,
                tier=tier,
            )
            details = outcome.weekly_details
            use_gross = i == 0  # 第一個情境取毛報酬欄位
            results.append(summarize(label, details, use_gross=use_gross))

            if i == 0:
                market_returns = [
                    d.market_return for d in details if d.market_return is not None
                ]
                ic_summary = asdict(outcome.ic_analysis)
    finally:
        session.close()

    # 市場基準
    market_cum = _compound(market_returns)
    market_weeks = len(market_returns)

    # ── 報告 ──
    lines: list[str] = []
    add = lines.append

    add("=" * 96)
    add("交易成本對回測績效的實測影響 — qlib-tw-trader")
    add("=" * 96)
    add("")
    add(f"區間        {args.start_week} ~ {args.end_week}（{results[0].weeks} 週）")
    add(f"初始資金    {args.capital:,.0f} TWD")
    add(f"持倉檔數    Top-{args.positions}")
    add(f"調倉頻率    日度（原專案設計，每個預測日重選 Top-K）")
    add("")

    add("─" * 96)
    add("IC 分析")
    add("─" * 96)
    add(f"  平均 valid IC（驗證期）  {ic_summary.get('avg_valid_ic', 0):+.4f}")
    add(f"  平均 live IC（樣本外）   {ic_summary.get('avg_live_ic', 0):+.4f}")
    add(f"  IC 衰減                  {ic_summary.get('ic_decay', 0):.1f}%")
    add(f"  valid/live IC 相關係數   {ic_summary.get('ic_correlation') or 0:+.4f}")
    add("")

    add("─" * 96)
    add("成本情境對照")
    add("─" * 96)
    add(
        f"{'情境':<38}{'累積':>9}{'年化':>9}{'Sharpe':>9}"
        f"{'MaxDD':>9}{'勝率':>8}{'年化成本':>10}"
    )
    for r in results:
        sharpe = f"{r.sharpe:.3f}" if r.sharpe is not None else "n/a"
        add(
            f"{r.label:<38}{r.cumulative_return:>8.2f}%{r.annualized_return:>8.2f}%"
            f"{sharpe:>9}{r.max_drawdown:>8.2f}%{r.win_rate:>7.1f}%"
            f"{r.annualized_cost_drag:>9.2f}%"
        )
    add("")
    add(
        f"{'市場基準（等權 100 檔）':<38}{market_cum:>8.2f}%"
        f"{_annualize(market_cum, market_weeks):>8.2f}%"
    )
    add("")

    add("─" * 96)
    add("實測換手率")
    add("─" * 96)
    base = results[2]  # 6折 + 0.3% 滑價
    add(f"  平均週換手率（單邊）   {base.avg_weekly_turnover * 100:.1f}%")
    add(f"  年化來回次數           {base.avg_weekly_turnover * WEEKS_PER_YEAR:.1f}")
    add(f"  區間總成本             {base.total_cost_pct:.2f}%")
    add(f"  年化成本拖累           {base.annualized_cost_drag:.2f}%")
    add("")
    add("  對照 README 宣稱的 HoldDrop(K=10,H=3,D=1) 週換手率 9.9%：")
    ratio = base.avg_weekly_turnover / 0.099 if base.avg_weekly_turnover else 0
    add(f"    本次實測為其 {ratio:.1f} 倍")
    add("")

    add("=" * 96)
    gross = results[0]
    net = results[2]
    add("結論")
    add("=" * 96)
    add(
        f"  毛報酬 {gross.cumulative_return:+.2f}%  →  "
        f"扣成本後 {net.cumulative_return:+.2f}%"
        f"（成本吃掉 {gross.cumulative_return - net.cumulative_return:.2f} 個百分點）"
    )
    if gross.sharpe is not None and net.sharpe is not None:
        add(f"  Sharpe {gross.sharpe:.3f}  →  {net.sharpe:.3f}")
    add(f"  同期市場基準 {market_cum:+.2f}%")
    add(
        f"  對市場超額（扣成本後）{net.cumulative_return - market_cum:+.2f} 個百分點"
    )
    add("")
    add("本次量測的邊界（誠實聲明）：")
    add(f"  · 只有 {results[0].weeks} 週，樣本極少，不足以下統計結論。")
    add("    README 的 156 週需要約 16 小時訓練，本次只訓練 20 個模型（約 78 分）。")
    add("  · 使用日度調倉的 topk 策略（原專案 API 預設），不是 README 報告")
    add("    最佳績效的 HoldDrop 策略。換手率與成本拖累因此高出很多。")
    add("  · 因 FinMind 免費額度限制，PER／月營收／集保／借券資料不完整，")
    add("    停用了 63 個相關因子，實際訓練用 240 個（原 303 個）。")
    add("  · 未做 PBO / Deflated Sharpe 多重測試校正。")
    add("=" * 96)

    report = "\n".join(lines)
    print()
    print(report)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = {
        "start_week": args.start_week,
        "end_week": args.end_week,
        "capital": args.capital,
        "max_positions": args.positions,
        "ic_analysis": ic_summary,
        "market_cumulative_return": market_cum,
        "scenarios": [asdict(r) for r in results],
    }
    json_path = OUTPUT_DIR / f"cost_impact_{stamp}.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print(f"原始數據：{json_path}")


if __name__ == "__main__":
    main()
