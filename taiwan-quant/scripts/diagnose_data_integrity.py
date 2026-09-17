#!/usr/bin/env python3
"""工作單 M：價格缺口、候選池身分與決策進出場日期完整性報告。"""

from __future__ import annotations

import argparse
import bisect
import json
import sqlite3
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from taiwan_quant.data.integrity import find_gap_runs  # noqa: E402
from taiwan_quant.data.loader import DEFAULT_UNIVERSE_BASIS, HISTORY_DB_PATH  # noqa: E402

DEV_END = date(2023, 12, 29)
START = date(2015, 1, 1)
WARMUP_DAYS = 250
HOLDING_DAYS = 40
STRIDE = 40
UNIVERSE_SIZE = 150


def _calendar(con: sqlite3.Connection, end: date) -> list[pd.Timestamp]:
    rows = con.execute(
        """
        SELECT date
        FROM stock_daily
        WHERE date BETWEEN ? AND ?
          AND open > 0 AND high > 0 AND low > 0 AND close > 0
        GROUP BY date
        HAVING COUNT(*) >= 100
        ORDER BY date
        """,
        (START.isoformat(), end.isoformat()),
    ).fetchall()
    return [pd.Timestamp(row[0]) for row in rows]


def _price_dates(con: sqlite3.Connection, end: date) -> dict[str, list[pd.Timestamp]]:
    rows = con.execute(
        """
        SELECT stock_id, date
        FROM stock_daily
        WHERE date BETWEEN ? AND ?
          AND open > 0 AND high > 0 AND low > 0 AND close > 0
        ORDER BY stock_id, date
        """,
        (START.isoformat(), end.isoformat()),
    )
    result: dict[str, list[pd.Timestamp]] = defaultdict(list)
    for stock_id, value in rows:
        result[str(stock_id)].append(pd.Timestamp(value))
    return dict(result)


def _metadata(con: sqlite3.Connection) -> dict[str, dict[str, str]]:
    rows = con.execute("SELECT stock_id, name, industry FROM stock_master")
    return {
        str(stock_id): {"name": name or "", "industry": industry or ""}
        for stock_id, name, industry in rows
    }


def _snapshots(
    con: sqlite3.Connection, end: date
) -> tuple[list[pd.Timestamp], dict[pd.Timestamp, dict[str, int]]]:
    rows = con.execute(
        """
        SELECT as_of_date, stock_id, rank
        FROM universe_history
        WHERE basis = ? AND as_of_date <= ? AND rank <= ?
        ORDER BY as_of_date, rank
        """,
        (DEFAULT_UNIVERSE_BASIS, end.isoformat(), UNIVERSE_SIZE),
    )
    by_date: dict[pd.Timestamp, dict[str, int]] = defaultdict(dict)
    for value, stock_id, rank in rows:
        by_date[pd.Timestamp(value)][str(stock_id)] = int(rank)
    dates = sorted(by_date)
    return dates, dict(by_date)


def _snapshot_at(
    day: pd.Timestamp,
    dates: list[pd.Timestamp],
    by_date: dict[pd.Timestamp, dict[str, int]],
) -> dict[str, int]:
    position = bisect.bisect_right(dates, day) - 1
    return by_date[dates[position]] if position >= 0 else {}


def run(db_path: Path, end: date) -> dict[str, Any]:
    if end > DEV_END:
        raise ValueError(f"診斷禁止載入 2024+；end 最晚為 {DEV_END}")
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as con:
        calendar = _calendar(con, end)
        observed_by_stock = _price_dates(con, end)
        metadata = _metadata(con)
        snapshot_dates, snapshots = _snapshots(con, end)

    decision_dates = [
        day
        for day in calendar[WARMUP_DAYS::STRIDE]
        if calendar.index(day) + 1 + HOLDING_DAYS < len(calendar)
    ]
    ever_candidates = {
        stock_id for members in snapshots.values() for stock_id in members
    }
    relevant: dict[tuple[str, pd.Timestamp], list[dict[str, str]]] = defaultdict(list)
    for decision_day in decision_dates:
        idx = calendar.index(decision_day)
        entry_day = calendar[idx + 1]
        exit_day = calendar[idx + 1 + HOLDING_DAYS]
        for stock_id in _snapshot_at(decision_day, snapshot_dates, snapshots):
            relevant[(stock_id, entry_day)].append(
                {"decision_date": decision_day.date().isoformat(), "role": "T+1"}
            )
            relevant[(stock_id, exit_day)].append(
                {"decision_date": decision_day.date().isoformat(), "role": "T+H"}
            )

    rows: list[dict[str, Any]] = []
    affected_events = 0
    candidate_stocks: set[str] = set()
    candidate_gap_days = 0
    candidate_single_day_gaps = 0
    candidate_blocks = 0
    candidate_block_days = 0
    active_candidate_stocks: set[str] = set()
    active_candidate_gap_days = 0
    for stock_id, observed in observed_by_stock.items():
        for run_item in find_gap_runs(observed, calendar):
            start_idx = calendar.index(run_item.start)
            dates = calendar[start_idx : start_idx + run_item.length]
            day_details = []
            in_universe_any = False
            for gap_day in dates:
                rank = _snapshot_at(gap_day, snapshot_dates, snapshots).get(stock_id)
                impacts = relevant.get((stock_id, gap_day), [])
                affected_events += len(impacts)
                if rank is not None:
                    in_universe_any = True
                    active_candidate_gap_days += 1
                    active_candidate_stocks.add(stock_id)
                day_details.append(
                    {
                        "date": gap_day.date().isoformat(),
                        "market_rank": rank,
                        "decision_impacts": impacts,
                    }
                )
            ever_candidate = stock_id in ever_candidates
            if ever_candidate:
                candidate_stocks.add(stock_id)
                candidate_gap_days += run_item.length
                if run_item.shape == "single":
                    candidate_single_day_gaps += 1
                else:
                    candidate_blocks += 1
                    candidate_block_days += run_item.length
            item = metadata.get(stock_id, {})
            rows.append(
                {
                    "stock_id": stock_id,
                    "name": item.get("name", ""),
                    "industry": item.get("industry", ""),
                    "start": run_item.start.date().isoformat(),
                    "end": run_item.end.date().isoformat(),
                    "length": run_item.length,
                    "shape": run_item.shape,
                    "ever_in_top150": ever_candidate,
                    "in_top150_during_gap": in_universe_any,
                    "days": day_details,
                }
            )

    singles = [row for row in rows if row["shape"] == "single"]
    blocks = [row for row in rows if row["shape"] == "block"]
    return {
        "start": START.isoformat(),
        "end": end.isoformat(),
        "market_calendar_days": len(calendar),
        "summary": {
            "stocks_with_gaps": len({row["stock_id"] for row in rows}),
            "gap_days": sum(row["length"] for row in rows),
            "single_day_gaps": len(singles),
            "block_count": len(blocks),
            "block_days": sum(row["length"] for row in blocks),
            "candidate_stocks_with_gaps": len(candidate_stocks),
            "candidate_gap_days": candidate_gap_days,
            "candidate_single_day_gaps": candidate_single_day_gaps,
            "candidate_block_count": candidate_blocks,
            "candidate_block_days": candidate_block_days,
            "active_candidate_stocks_with_gaps": len(active_candidate_stocks),
            "active_candidate_gap_days": active_candidate_gap_days,
            "decision_entry_or_exit_impacts": affected_events,
        },
        "gaps": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=HISTORY_DB_PATH)
    parser.add_argument("--end", default=DEV_END.isoformat())
    parser.add_argument(
        "--output", type=Path, default=Path("reports/data_integrity_before.json")
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
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"結果：{args.output}")


if __name__ == "__main__":
    main()
