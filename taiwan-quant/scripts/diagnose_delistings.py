#!/usr/bin/env python3
"""工作單 K：開發集缺未來價診斷與下市政策 A/B 對照。"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.validate_oos_trailing as validation  # noqa: E402
from scripts.diagnose_position_costs import (  # noqa: E402
    CAPITAL,
    DEV_END,
    FAMILY,
    HOLDING_DAYS,
    START,
    UNIVERSE_SIZE,
    WARMUP_DAYS,
    resolve_pick_tier,
)
from taiwan_quant.config.costs import DEFAULT  # noqa: E402
from taiwan_quant.data.dataset import build_dataset  # noqa: E402
from taiwan_quant.data.integrity import (  # noqa: E402
    complete_holding_decision_dates,
    holding_dates,
)
from taiwan_quant.data.loader import (  # noqa: E402
    DEFAULT_UNIVERSE_BASIS,
    HISTORY_DB_PATH,
    RAW_OPEN_COLUMN,
    load_chips,
    load_prices,
    load_universe_at,
)
from taiwan_quant.ranking.tie_break import (  # noqa: E402
    DEFAULT_TIE_SEED,
    deterministic_jitter,
)
from taiwan_quant.validation.delisting import (  # noqa: E402
    MissingPriceKind,
    settle_holding_period,
)

N_POSITIONS = 10


def _metadata(db_path: Path, end: date) -> dict[str, dict[str, Any]]:
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as con:
        rows = con.execute(
            """
            SELECT stock_id, name, industry,
                   CASE WHEN delisted_date <= ? THEN delisted_date END
            FROM stock_master
            """,
            (end.isoformat(),),
        ).fetchall()
    return {
        sid: {
            "name": name,
            "industry": industry,
            "delisted_date": date.fromisoformat(delisted) if delisted else None,
        }
        for sid, name, industry, delisted in rows
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


def _summarize(periods: list[dict[str, Any]]) -> dict[str, Any]:
    if not periods:
        return {"periods": 0, "trades": 0}
    frame = pd.DataFrame(periods)
    yearly = frame.groupby("year", sort=True)["net"].mean()
    return {
        "periods": len(frame),
        "trades": int(frame["trades"].sum()),
        "gross_per_trip": float(frame["gross"].mean()),
        "cost_per_trip": float(frame["cost"].mean()),
        "net_per_trip": float(frame["net"].mean()),
        "positive_years": int((yearly > 0).sum()),
        "years": len(yearly),
        "by_date_net": dict(zip(frame["decision_date"], frame["net"], strict=True)),
    }


def _paired(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> dict[str, float | int]:
    a_map = {row["decision_date"]: row["net"] for row in a}
    b_map = {row["decision_date"]: row["net"] for row in b}
    common = sorted(a_map.keys() & b_map.keys())
    differences = np.array([b_map[key] - a_map[key] for key in common], dtype=float)
    if not len(differences):
        return {"n": 0, "mean_difference": 0.0, "standard_error": 0.0}
    se = (
        float(differences.std(ddof=1) / math.sqrt(len(differences)))
        if len(differences) > 1 else 0.0
    )
    return {
        "n": len(differences),
        "mean_difference": float(differences.mean()),
        "standard_error": se,
    }


def run(db_path: Path, end: date) -> dict[str, Any]:
    if end > DEV_END:
        raise ValueError(f"禁止載入 2024+；end 最晚為 {DEV_END}")

    members = _members(db_path)
    metadata = _metadata(db_path, end)
    prices = load_prices(members, start=START, end=end, db_path=db_path)
    chips = load_chips(members, start=START, end=end, db_path=db_path)
    standard = build_dataset(members, prices, chips)
    relaxed = build_dataset(members, prices, chips, min_length=1)
    calendar = validation.trading_calendar(standard.by_stock)
    decision_dates = complete_holding_decision_dates(
        calendar,
        calendar[WARMUP_DAYS::HOLDING_DAYS],
        holding_days=HOLDING_DAYS,
    )
    members_at = validation.resolve_members(
        decision_dates, db_path, UNIVERSE_SIZE, DEFAULT_UNIVERSE_BASIS
    )
    large_at = validation.resolve_members(
        decision_dates, db_path, 50, DEFAULT_UNIVERSE_BASIS
    )
    market_ranks = {
        day: {
            str(row.stock_id): int(row.rank)
            for row in load_universe_at(
                day.date(), db_path=db_path, limit=UNIVERSE_SIZE,
                basis=DEFAULT_UNIVERSE_BASIS,
            ).stocks.itertuples()
        }
        for day in decision_dates
    }
    actual_opens = _frame(prices, RAW_OPEN_COLUMN, calendar)
    adjusted_opens = _frame(prices, "open", calendar)
    price_bars = {
        sid: prices.xs(sid, level="stock_id")
        for sid in set(prices.index.get_level_values("stock_id"))
    }

    short_delisted = {
        sid: length
        for sid, length in standard.skipped
        if metadata.get(sid, {}).get("delisted_date") is not None
    }

    def evaluate(
        by_stock: dict[str, pd.DataFrame],
        tracked: set[str] | None = None,
    ) -> tuple[
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
    ]:
        scores = validation.precompute_scores(by_stock)[FAMILY]
        missing_events: list[dict[str, Any]] = []
        policy_a: list[dict[str, Any]] = []
        policy_b: list[dict[str, Any]] = []
        tracked_candidates: list[dict[str, Any]] = []
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
            entry_day, target_day = holding_dates(
                day, calendar, holding_days=HOLDING_DAYS
            )
            settlements = {}
            tracked_by_sid: dict[str, dict[str, Any]] = {}
            for strategy_rank, sid in enumerate(ordered, start=1):
                item = metadata.get(sid, {})
                settlement = settle_holding_period(
                    price_bars[sid],
                    entry_date=entry_day,
                    target_date=target_day,
                    delisted_date=item.get("delisted_date"),
                )
                settlements[sid] = settlement
                if tracked and sid in tracked:
                    tracked_by_sid[sid] = {
                        "decision_date": day.date().isoformat(),
                        "stock_id": sid,
                        "name": item.get("name", ""),
                        "market_rank": market_ranks[day].get(sid),
                        "strategy_rank": strategy_rank,
                        "score": float(ranked[sid]),
                        "kind": settlement.kind.value,
                        "selected_policy_b": False,
                    }
                if settlement.kind is not MissingPriceKind.COMPLETE:
                    missing_events.append(
                        {
                            "decision_date": day.date().isoformat(),
                            "stock_id": sid,
                            "name": item.get("name", ""),
                            "market_rank": market_ranks[day].get(sid),
                            "strategy_rank": strategy_rank,
                            "score": float(ranked[sid]),
                            "kind": settlement.kind.value,
                            "known_delisted": item.get("delisted_date") is not None,
                            "delisting_known_by_target": (
                                item.get("delisted_date") is not None
                                and item["delisted_date"] <= target_day.date()
                            ),
                            "delisted_date": (
                                item["delisted_date"].isoformat()
                                if item.get("delisted_date") else None
                            ),
                            "last_price_date": price_bars[sid].index[-1].date().isoformat(),
                        }
                    )

            for policy, accepted in (
                (policy_a, {MissingPriceKind.COMPLETE}),
                (policy_b, {MissingPriceKind.COMPLETE, MissingPriceKind.DELISTED}),
            ):
                picks = [sid for sid in ordered if settlements[sid].kind in accepted][
                    :N_POSITIONS
                ]
                if len(picks) < N_POSITIONS:
                    continue
                if policy is policy_b:
                    for sid in picks:
                        if sid in tracked_by_sid:
                            tracked_by_sid[sid]["selected_policy_b"] = True
                returns = [float(settlements[sid].gross_return) for sid in picks]
                costs = []
                large_members = large_at.get(day, set())
                for sid in picks:
                    adjusted = float(adjusted_opens.loc[entry_day, sid])
                    actual = float(actual_opens.loc[entry_day, sid])
                    tier = resolve_pick_tier(
                        sid, actual, adjusted, CAPITAL / N_POSITIONS,
                        is_large=sid in large_members,
                    )
                    costs.append(
                        DEFAULT.round_trip_cost(CAPITAL / N_POSITIONS, tier)
                        / (CAPITAL / N_POSITIONS)
                    )
                gross = float(np.mean(returns))
                cost = float(np.mean(costs))
                policy.append(
                    {
                        "decision_date": day.date().isoformat(),
                        "year": day.year,
                        "gross": gross,
                        "cost": cost,
                        "net": gross - cost,
                        "trades": len(picks),
                        "stocks": picks,
                    }
                )
            tracked_candidates.extend(tracked_by_sid.values())
        return policy_a, policy_b, missing_events, tracked_candidates

    standard_a, standard_b, missing, _ = evaluate(standard.by_stock)
    augmented_by_stock = dict(standard.by_stock)
    augmented_by_stock.update(
        {sid: relaxed.by_stock[sid] for sid in short_delisted if sid in relaxed.by_stock}
    )
    _, relaxed_b, _, short_candidates = evaluate(
        augmented_by_stock, tracked=set(short_delisted)
    )
    return {
        "start": START.isoformat(),
        "end": end.isoformat(),
        "holding_days": HOLDING_DAYS,
        "n_positions": N_POSITIONS,
        "policy_a": _summarize(standard_a),
        "policy_b": _summarize(standard_b),
        "policy_b_minus_a": _paired(standard_a, standard_b),
        "missing_future_prices": missing,
        "short_delisted": {
            "excluded": short_delisted,
            "candidate_events": short_candidates,
            "policy_b_standard": _summarize(standard_b),
            "policy_b_relaxed": _summarize(relaxed_b),
            "relaxed_minus_standard": _paired(standard_b, relaxed_b),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=HISTORY_DB_PATH)
    parser.add_argument("--end", default=DEV_END.isoformat())
    parser.add_argument(
        "--output", type=Path, default=Path("reports/delisting_policy_dev.json")
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
    a, b = payload["policy_a"], payload["policy_b"]
    print(
        f"A：{a['trades']} 筆，毛/趟 {a['gross_per_trip']*100:+.3f}%，"
        f"淨/趟 {a['net_per_trip']*100:+.3f}%"
    )
    print(
        f"B：{b['trades']} 筆，毛/趟 {b['gross_per_trip']*100:+.3f}%，"
        f"淨/趟 {b['net_per_trip']*100:+.3f}%"
    )
    print(f"缺未來價事件：{len(payload['missing_future_prices'])}")
    print(f"結果：{args.output}")


if __name__ == "__main__":
    main()
