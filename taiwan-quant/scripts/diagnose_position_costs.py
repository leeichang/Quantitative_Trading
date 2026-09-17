#!/usr/bin/env python3
"""工作單 J：只在開發集比較還原價與實際價的成本分層。"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.validate_oos_trailing as validation  # noqa: E402
from taiwan_quant.config.costs import DEFAULT, Tier, resolve_tier  # noqa: E402
from taiwan_quant.data.dataset import build_dataset  # noqa: E402
from taiwan_quant.data.etf_universe import is_etf  # noqa: E402
from taiwan_quant.data.integrity import select_holding_positions  # noqa: E402
from taiwan_quant.data.loader import (  # noqa: E402
    DEFAULT_UNIVERSE_BASIS,
    HISTORY_DB_PATH,
    RAW_OPEN_COLUMN,
    load_chips,
    load_prices,
)
from taiwan_quant.ranking.tie_break import (  # noqa: E402
    DEFAULT_TIE_SEED,
    deterministic_jitter,
)
from taiwan_quant.validation.delisting import load_delisted_dates  # noqa: E402

DEV_END = date(2023, 12, 29)
START = date(2015, 1, 1)
CAPITAL = 400_000.0
HOLDING_DAYS = 40
WARMUP_DAYS = 250
UNIVERSE_SIZE = 150
POSITION_COUNTS = (1, 3, 5, 10, 15, 20)
FAMILY = "動能突破"
WHOLE_TIERS = {Tier.LARGE_WHOLE, Tier.MID_WHOLE, Tier.ETF_WHOLE}


def summarize_rows(
    rows: list[dict[str, Any]], trips_per_year: float
) -> dict[str, float | int]:
    """彙總同一 N、同一成本政策；公式由單元測試手算釘住。"""
    if not rows:
        raise ValueError("rows 不可為空")
    frame = pd.DataFrame(rows)
    yearly = frame.groupby("year", sort=True)["net"].mean()
    gross = float(frame["gross"].mean())
    cost = float(frame["cost"].mean())
    net = float(frame["net"].mean())
    trades = int(frame["trades"].sum())
    whole = int(frame["whole"].sum())
    return {
        "periods": len(frame),
        "trades": trades,
        "gross_per_trip": gross,
        "cost_per_trip": cost,
        "net_per_trip": net,
        "annualized_net": float((1.0 + net) ** trips_per_year - 1.0),
        "whole_ratio": whole / trades if trades else 0.0,
        "positive_years": int((yearly > 0).sum()),
        "years": len(yearly),
    }


def _members(db_path: Path) -> list[str]:
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as con:
        return [
            row[0]
            for row in con.execute(
                "SELECT DISTINCT stock_id FROM universe_history WHERE basis = ?",
                (DEFAULT_UNIVERSE_BASIS,),
            )
        ]


def _frame(
    prices: pd.DataFrame, column: str, calendar: list[pd.Timestamp]
) -> pd.DataFrame:
    return prices[column].unstack("stock_id").reindex(calendar)


def resolve_pick_tier(
    sid: str,
    actual_price: float,
    adjusted_price: float,
    amount: float,
    *,
    is_large: bool,
) -> Tier:
    return resolve_tier(
        actual_price=actual_price,
        adjusted_price=adjusted_price,
        amount=amount,
        large=is_large,
        is_etf=is_etf(sid),
    )


def _cost_rate(amount: float, tier: Tier, *, exact: bool) -> float:
    if exact:
        return DEFAULT.round_trip_cost(amount, tier) / amount
    return DEFAULT.round_trip_rate(tier)


def run(db_path: Path, end: date) -> dict[str, Any]:
    """掃 N=1/3/5/10/15/20；硬限制在開發集。"""
    if end > DEV_END:
        raise ValueError(f"禁止載入 2024+；end 最晚為 {DEV_END}")

    members = _members(db_path)
    prices = load_prices(members, start=START, end=end, db_path=db_path)
    chips = load_chips(members, start=START, end=end, db_path=db_path)
    by_stock = build_dataset(members, prices, chips).by_stock
    calendar = validation.trading_calendar(by_stock)
    scores = validation.precompute_scores(by_stock)[FAMILY]
    adjusted_opens = _frame(prices, "open", calendar)
    actual_opens = _frame(prices, RAW_OPEN_COLUMN, calendar)
    adjusted_closes = _frame(prices, "close", calendar)
    delisted_dates = load_delisted_dates(db_path, as_of=end)

    decision_dates = [
        day
        for day in calendar[WARMUP_DAYS::HOLDING_DAYS]
        if calendar.index(day) + 1 + HOLDING_DAYS < len(calendar)
    ]
    members_at = validation.resolve_members(
        decision_dates, db_path, UNIVERSE_SIZE, DEFAULT_UNIVERSE_BASIS
    )
    large_at = validation.resolve_members(
        decision_dates, db_path, 50, DEFAULT_UNIVERSE_BASIS
    )
    policies = (
        "legacy_adjusted_rate",
        "adjusted_exact_cost",
        "actual_exact_cost",
    )
    rows: dict[int, dict[str, list[dict[str, Any]]]] = {
        n: {policy: [] for policy in policies} for n in POSITION_COUNTS
    }

    for day in decision_dates:
        allowed = tuple(members_at.get(day) or ())
        ranked = pd.Series(
            {sid: scores[sid].get(day, np.nan) for sid in allowed if sid in scores}
        ).dropna()
        ordered = sorted(
            ranked.index,
            key=lambda sid: (
                -float(ranked[sid]),
                deterministic_jitter(sid, DEFAULT_TIE_SEED),
            ),
        )
        selected = select_holding_positions(
            decision_date=day,
            calendar=calendar,
            ordered_candidates=ordered,
            opens=adjusted_opens,
            closes=adjusted_closes,
            holding_days=HOLDING_DAYS,
            n_positions=max(POSITION_COUNTS),
            delisted_dates=delisted_dates,
        )
        entry_day = calendar[calendar.index(day) + 1]
        large_members = large_at.get(day, set())
        for n in POSITION_COUNTS:
            if len(selected) < n:
                continue
            positions = selected[:n]
            picks = [position.stock_id for position in positions]
            gross = float(np.mean([position.gross_return for position in positions]))
            amount = CAPITAL / n
            policy_costs: dict[str, list[float]] = defaultdict(list)
            policy_whole: dict[str, int] = defaultdict(int)
            for sid in picks:
                adjusted = float(adjusted_opens.loc[entry_day, sid])
                actual = float(actual_opens.loc[entry_day, sid])
                if not np.isfinite(actual) or actual <= 0:
                    raise RuntimeError(f"{entry_day.date()} {sid} 缺實際開盤價")
                large = sid in large_members
                # 精確重現已發佈舊表：當時漏傳 large=，所有非 ETF 都走大型股。
                legacy = resolve_pick_tier(
                    sid, adjusted, adjusted, amount, is_large=True
                )
                adjusted_tier = resolve_pick_tier(
                    sid, adjusted, adjusted, amount, is_large=large
                )
                corrected = resolve_pick_tier(
                    sid, actual, adjusted, amount, is_large=large
                )
                for policy, tier, exact in (
                    ("legacy_adjusted_rate", legacy, False),
                    ("adjusted_exact_cost", adjusted_tier, True),
                    ("actual_exact_cost", corrected, True),
                ):
                    policy_costs[policy].append(_cost_rate(amount, tier, exact=exact))
                    policy_whole[policy] += int(tier in WHOLE_TIERS)
            for policy in policies:
                cost = float(np.mean(policy_costs[policy]))
                rows[n][policy].append(
                    {
                        "decision_date": day.date().isoformat(),
                        "year": day.year,
                        "gross": gross,
                        "cost": cost,
                        "net": gross - cost,
                        "whole": policy_whole[policy],
                        "trades": n,
                    }
                )

    trips_per_year = 252.0 / HOLDING_DAYS
    summaries = {
        str(n): {
            policy: summarize_rows(rows[n][policy], trips_per_year)
            for policy in policies
        }
        for n in POSITION_COUNTS
    }
    return {
        "start": START.isoformat(),
        "end": end.isoformat(),
        "holding_days": HOLDING_DAYS,
        "capital": CAPITAL,
        "position_counts": list(POSITION_COUNTS),
        "summaries": summaries,
        "periods": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=HISTORY_DB_PATH)
    parser.add_argument("--end", default=DEV_END.isoformat())
    parser.add_argument(
        "--output", type=Path, default=Path("reports/position_costs_dev.json")
    )
    args = parser.parse_args()
    end = date.fromisoformat(args.end)
    if end > DEV_END:
        parser.error(f"禁止載入 2024+；--end 最晚為 {DEV_END}")
    payload = run(args.db, end)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for n in POSITION_COUNTS:
        old = payload["summaries"][str(n)]["legacy_adjusted_rate"]
        new = payload["summaries"][str(n)]["actual_exact_cost"]
        print(
            f"N={n:>2} old cost={old['cost_per_trip']*100:.3f}% "
            f"whole={old['whole_ratio']*100:.1f}% | "
            f"new cost={new['cost_per_trip']*100:.3f}% "
            f"whole={new['whole_ratio']*100:.1f}%"
        )
    print(f"結果：{args.output}")


if __name__ == "__main__":
    main()
