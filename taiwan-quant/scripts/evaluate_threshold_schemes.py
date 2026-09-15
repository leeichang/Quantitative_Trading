#!/usr/bin/env python3
"""任務 E：只在截至 2023-12-29 的開發集並列評估門檻方案。"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.validate_oos_trailing import (  # noqa: E402
    RANDOM_SEED,
    WARMUP_DAYS,
    enumerate_decisions,
    infer_tiers,
    make_price_lookup,
    precompute_outcomes,
    precompute_scores,
    resolve_members,
    run_walk_forward,
    to_signals,
    trading_calendar,
)
from taiwan_quant.backtest.portfolio_sim import (  # noqa: E402
    PriceLookup,
    Signal,
    simulate_portfolio,
)
from taiwan_quant.config.costs import DEFAULT  # noqa: E402
from taiwan_quant.data.dataset import build_dataset  # noqa: E402
from taiwan_quant.data.loader import (  # noqa: E402
    DEFAULT_UNIVERSE_BASIS,
    HISTORY_DB_PATH,
    load_chips,
    load_prices,
)
from taiwan_quant.strategies.families import STRATEGY_FAMILIES  # noqa: E402
from taiwan_quant.validation.benchmarks import (  # noqa: E402
    ETF_BENCHMARKS,
    equal_weight_equity,
    equity_curve_statistics,
    etf_benchmark_curves,
)
from taiwan_quant.validation.thresholds import (  # noqa: E402
    annotate_tie_counts,
    filter_relative_top,
    select_periodic_rebalances,
)
from taiwan_quant.validation.trial_summary import summarize_trials  # noqa: E402

DEV_END = date(2023, 12, 29)
UNIVERSE_SIZE = 150
RANDOM_TRIALS = 100


def _all_members(db_path: Path, basis: str) -> list[str]:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        return [row[0] for row in con.execute(
            "SELECT DISTINCT stock_id FROM universe_history "
            "WHERE basis = ? ORDER BY stock_id",
            (basis,),
        )]
    finally:
        con.close()


def _scheme_signals(
    name: str,
    raw: list[Signal],
    calendar: list[pd.Timestamp],
    execution_lookup: PriceLookup,
) -> tuple[list[Signal], int, int]:
    if name.startswith("E1-top"):
        selected = filter_relative_top(raw, float(name.removeprefix("E1-top")))
        return selected, len(selected), 0
    if name == "E3-rebalance60":
        result = select_periodic_rebalances(raw, calendar, execution_lookup, 60, 3)
        return list(result.signals), result.candidates, result.rejected_missing_price
    return annotate_tie_counts(raw), len(raw), 0


def main() -> None:
    parser = argparse.ArgumentParser(description="開發集門檻方案並列掃描")
    parser.add_argument("--db", default=str(HISTORY_DB_PATH))
    parser.add_argument("--output", default="reports/threshold_sweep_dev.json")
    parser.add_argument("--horizon", type=int, default=60, choices=[60])
    parser.add_argument("--end", default=DEV_END.isoformat())
    args = parser.parse_args()
    end = date.fromisoformat(args.end)
    if end > DEV_END:
        parser.error("任務 E 不得載入 2024+；--end 最晚為 2023-12-29")

    db_path = Path(args.db)
    members = _all_members(db_path, DEFAULT_UNIVERSE_BASIS)
    load_ids = sorted(set(members) | set(ETF_BENCHMARKS))
    prices = load_prices(
        load_ids, start=date(2015, 1, 1), end=end, adjusted=True, db_path=db_path
    )
    price_ids = set(prices.index.get_level_values("stock_id"))
    price_by_stock = {
        stock_id: prices.xs(stock_id, level="stock_id")
        for stock_id in load_ids if stock_id in price_ids
    }
    chips = load_chips(members, start=date(2015, 1, 1), end=end, db_path=db_path)
    dataset = build_dataset(load_ids, prices, chips)
    by_stock = dataset.by_stock
    calendar = trading_calendar(by_stock)
    decision_dates = calendar[WARMUP_DAYS::5]
    close_lookup = make_price_lookup(by_stock)
    execution_lookup = make_price_lookup(by_stock, column="open", exact=True)
    tiers = infer_tiers(db_path, UNIVERSE_SIZE, DEFAULT_UNIVERSE_BASIS)

    print("預算三族分數與 60 日標記（只做一次）...", flush=True)
    started = time.time()
    scores = precompute_scores(by_stock)
    outcomes = precompute_outcomes(by_stock, args.horizon, decision_dates)
    print(f"預算完成：{time.time() - started:.1f}s", flush=True)

    rows: list[dict[str, object]] = []
    for size in (UNIVERSE_SIZE,):
        members_at = resolve_members(
            decision_dates, db_path, size, DEFAULT_UNIVERSE_BASIS
        )
        for family in STRATEGY_FAMILIES:
            by_date: dict = enumerate_decisions(
                outcomes, scores[family.name], decision_dates, members_at
            )
            raw_picks, random_picks, oos_start, _, _ = run_walk_forward(
                by_date,
                tiers,
                np.random.default_rng(RANDOM_SEED),
                None,
                args.horizon,
            )
            if oos_start is None:
                continue
            oos_calendar = [day for day in calendar if day >= oos_start]
            raw_signals = to_signals(raw_picks, calendar, tiers)
            random_totals: list[float] = []
            random_drawdowns: list[float] = []
            for trial in range(RANDOM_TRIALS):
                rng = np.random.default_rng(RANDOM_SEED + trial)
                trial_picks = [
                    (day, decision, float(rng.random()))
                    for day, decision, _ in random_picks
                ]
                random_result = simulate_portfolio(
                    to_signals(trial_picks, calendar, tiers),
                    close_lookup,
                    oos_calendar,
                    n_slots=3,
                    cost=DEFAULT,
                )
                random_totals.append(random_result.total_return)
                random_drawdowns.append(random_result.max_drawdown)
            random_summary = summarize_trials(random_totals, random_drawdowns)

            start_members = members_at.get(oos_start, set())
            equal_weight_data = {
                stock_id: price_by_stock[stock_id]
                for stock_id in start_members if stock_id in price_by_stock
            }
            equal_curve = equal_weight_equity(equal_weight_data, oos_calendar)
            equal_stats = equity_curve_statistics(equal_curve)
            etf_curves = etf_benchmark_curves(price_by_stock, oos_calendar)
            etf_stats = {
                etf: {
                    "total_return": equity_curve_statistics(curve).total_return,
                    "max_drawdown": equity_curve_statistics(curve).max_drawdown,
                    "sharpe": equity_curve_statistics(curve).sharpe,
                }
                for etf, curve in etf_curves.items()
            }

            candidates: list[tuple[str, list[Signal]]] = [
                (f"E1-top{pct}", raw_signals) for pct in (5, 10, 20)
            ]
            for edge_z in (1.5, 2.0, 2.5):
                picks, _, _, _, _ = run_walk_forward(
                    by_date,
                    tiers,
                    np.random.default_rng(RANDOM_SEED),
                    edge_z,
                    args.horizon,
                )
                candidates.append((f"E2-z{edge_z:g}", to_signals(picks, calendar, tiers)))
            candidates.append(("E3-rebalance60", raw_signals))

            for scheme, candidate_signals in candidates:
                selected, signal_count, selector_rejected = _scheme_signals(
                    scheme, candidate_signals, oos_calendar, execution_lookup
                )
                result = simulate_portfolio(
                    selected, close_lookup, oos_calendar, n_slots=3, cost=DEFAULT
                )
                rows.append({
                    "scheme": scheme,
                    "family": family.name,
                    "universe_size": size,
                    "signals": signal_count,
                    "trades": result.opened_signals,
                    "signal_trade_ratio": (
                        signal_count / result.opened_signals
                        if result.opened_signals else None
                    ),
                    "selector_rejected": selector_rejected,
                    "total_return": result.total_return,
                    "sharpe": result.sharpe,
                    "max_drawdown": result.max_drawdown,
                    "weekly_turnover": result.weekly_turnover,
                    "annualized_cost_drag": result.annualized_cost_drag,
                    "cost_model": "taiwan_quant.config.costs.DEFAULT",
                    "transactions": [
                        {
                            "stock_id": trade.stock_id,
                            "entry_date": trade.entry_date.date().isoformat(),
                            "exit_date": trade.exit_date.date().isoformat(),
                            "rank_score": trade.rank_score,
                            "tie_count": trade.tie_count,
                        }
                        for trade in result.trades
                    ],
                    "benchmarks": {
                        "equal_weight_150": {
                            "total_return": equal_stats.total_return,
                            "max_drawdown": equal_stats.max_drawdown,
                            "sharpe": equal_stats.sharpe,
                            "constituents": len(equal_weight_data),
                        },
                        "etf": etf_stats,
                        "random_100_median": {
                            "total_return": random_summary.median_total_return,
                            "max_drawdown": random_summary.median_max_drawdown,
                            "n_trials": random_summary.n_trials,
                        },
                    },
                })
                print(
                    f"{scheme:<17} {family.name:<8} U={size} "
                    f"sig/trade={signal_count}/{result.opened_signals} "
                    f"return={result.total_return * 100:+.2f}%",
                    flush=True,
                )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"end": end.isoformat(), "rows": rows},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"結果：{output}")


if __name__ == "__main__":
    main()
