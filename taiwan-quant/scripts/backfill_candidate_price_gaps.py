#!/usr/bin/env python3
"""以 FinMind 交叉驗證前 150 歷史成員缺口，只回補官方確有交易的價格列。"""

from __future__ import annotations

import argparse
import getpass
import json
import sqlite3
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.backfill_finmind_history import (  # noqa: E402
    DATASETS,
    USER_AGENT,
    api_token,
    call,
)
from taiwan_quant.data.loader import HISTORY_DB_PATH  # noqa: E402

DEV_END = date(2023, 12, 29)


def _is_tradeable_quote(row: dict[str, Any]) -> bool:
    try:
        values = (
            float(row["open"]),
            float(row["max"]),
            float(row["min"]),
            float(row["close"]),
        )
    except (KeyError, TypeError, ValueError):
        return False
    return min(values) > 0


def _repair_prices(
    con: sqlite3.Connection,
    stock_id: str,
    rows: list[dict[str, Any]],
    expected_dates: set[str],
) -> int:
    """只補診斷已列出的缺口；既有完整列絕不覆寫。"""
    now = datetime.now().isoformat(timespec="seconds")
    payload = []
    for row in rows:
        day = str(row.get("date", ""))
        if day not in expected_dates:
            continue
        try:
            open_, high, low, close = (
                float(row["open"]),
                float(row["max"]),
                float(row["min"]),
                float(row["close"]),
            )
            volume = int(row.get("Trading_Volume") or 0)
        except (KeyError, TypeError, ValueError):
            continue
        if not _is_tradeable_quote(row):
            continue
        payload.append(
            (stock_id, day, open_, high, low, close, volume, now)
        )
    if not payload:
        return 0
    before = con.total_changes
    con.executemany(
        """
        INSERT INTO stock_daily
            (stock_id, date, open, high, low, close, volume, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(stock_id, date) DO UPDATE SET
            open = excluded.open,
            high = excluded.high,
            low = excluded.low,
            close = excluded.close,
            volume = excluded.volume,
            created_at = excluded.created_at
        WHERE stock_daily.open IS NULL OR stock_daily.open <= 0
           OR stock_daily.high IS NULL OR stock_daily.high <= 0
           OR stock_daily.low IS NULL OR stock_daily.low <= 0
           OR stock_daily.close IS NULL OR stock_daily.close <= 0
        """,
        payload,
    )
    return con.total_changes - before


def run(
    *,
    db_path: Path,
    gap_report: Path,
    output: Path,
    delay: float,
    token_override: str | None = None,
    all_stocks: bool = False,
) -> dict[str, Any]:
    report = json.loads(gap_report.read_text(encoding="utf-8"))
    if date.fromisoformat(report["end"]) > DEV_END:
        raise ValueError(f"禁止用 2024+ 診斷輸入；最晚為 {DEV_END}")
    candidate_gaps = (
        report["gaps"]
        if all_stocks
        else [row for row in report["gaps"] if row["ever_in_top150"]]
    )
    stocks = sorted({row["stock_id"] for row in candidate_gaps})
    expected = {
        (row["stock_id"], item["date"])
        for row in candidate_gaps
        for item in row["days"]
    }
    expected_by_stock: dict[str, set[str]] = {
        stock_id: {day for sid, day in expected if sid == stock_id}
        for stock_id in stocks
    }
    token = token_override or api_token()
    if token is None:
        raise RuntimeError("FINMIND_API_TOKEN 未設定；不以匿名額度執行回補")

    classifications: list[dict[str, Any]] = []
    source_dates: dict[str, set[str]] = {}
    failures: list[dict[str, str]] = []
    repaired = 0
    with sqlite3.connect(db_path, timeout=30) as con:
        before_rows = int(con.execute("SELECT COUNT(*) FROM stock_daily").fetchone()[0])
        with httpx.Client(headers={"User-Agent": USER_AGENT}) as client:
            for index, stock_id in enumerate(stocks, start=1):
                result = call(
                    client,
                    {
                        "dataset": DATASETS["prices"],
                        "data_id": stock_id,
                        "start_date": report["start"],
                        "end_date": report["end"],
                    },
                    token,
                )
                if not result.ok:
                    failures.append({"stock_id": stock_id, "message": result.message})
                    continue
                source_dates[stock_id] = {
                    str(row.get("date"))
                    for row in result.rows
                    if row.get("date") is not None and _is_tradeable_quote(row)
                }
                repaired += _repair_prices(
                    con, stock_id, result.rows, expected_by_stock[stock_id]
                )
                if index % 20 == 0:
                    con.commit()
                time.sleep(delay)
        con.commit()
        after_rows = int(con.execute("SELECT COUNT(*) FROM stock_daily").fetchone()[0])

    for stock_id, day in sorted(expected):
        classifications.append(
            {
                "stock_id": stock_id,
                "date": day,
                "source_has_trade": day in source_dates.get(stock_id, set()),
                "classification": (
                    "data_missing"
                    if day in source_dates.get(stock_id, set())
                    else "no_trade_or_suspension"
                ),
            }
        )

    payload: dict[str, Any] = {
        "source": "FinMind TaiwanStockPrice",
        "start": report["start"],
        "end": report["end"],
        "stocks_checked": len(stocks),
        "gap_days_checked": len(expected),
        "database_rows_before": before_rows,
        "database_rows_after": after_rows,
        "rows_repaired": repaired,
        "rows_inserted": after_rows - before_rows,
        "rows_updated": repaired - (after_rows - before_rows),
        "source_confirmed_missing_rows": sum(
            item["source_has_trade"] for item in classifications
        ),
        "source_confirmed_no_trade_rows": sum(
            not item["source_has_trade"] for item in classifications
        ),
        "failures": failures,
        "classifications": classifications,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=HISTORY_DB_PATH)
    parser.add_argument(
        "--gap-report", type=Path, default=Path("reports/data_integrity_before.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/data_gap_backfill.json")
    )
    parser.add_argument("--delay", type=float, default=0.2)
    parser.add_argument(
        "--prompt-token", action="store_true", help="以隱藏輸入讀取 token，不落盤"
    )
    parser.add_argument(
        "--all-stocks", action="store_true", help="核對報告中的全部股票，不只歷史前 150"
    )
    args = parser.parse_args()
    token_override = getpass.getpass("FinMind token: ") if args.prompt_token else None
    payload = run(
        db_path=args.db,
        gap_report=args.gap_report,
        output=args.output,
        delay=args.delay,
        token_override=token_override,
        all_stocks=args.all_stocks,
    )
    print(json.dumps({key: value for key, value in payload.items()
                      if key not in {"classifications"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
