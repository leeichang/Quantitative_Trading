#!/usr/bin/env python3
"""
還原收盤價回補（2015 起）

## 為什麼需要

CLAUDE.md 禁令 12：**一律用還原股價。**

`loader._apply_adjustment()` 用 `adj_close / close` 當因子，同比例套到
OHLC。但 `stock_daily_adj` 只有 100 檔、2023-01 起——價格回補到 2015 之後，
沒有還原價的那一段會被標記成 `is_adjusted=False`，等於禁令 12 破功。

## 為什麼用 yfinance

**既有的 `stock_daily_adj` 本來就是 yfinance 的 Adj Close。** 實測比對
（2330 / 2881 / 2317 / 1301，各 892 根）：

    相對誤差中位數  0.000002 ~ 0.000040      （DB 存兩位小數的捨入）
    超過 0.5% 的列  0 / 3568

用同一個來源往前拉，方法學才不會在 2023-01 產生斷點。

### 這裡不會踩到「yfinance OHLC 是分割還原價」的坑

本腳本**只取 Adj Close**，不取 OHLC。因子是

    adj_close(yfinance) / close(TWSE 原始成交價)

分子含分割與股利還原、分母是原始價，比值正好就是完整的還原因子。
既有管線一直是這樣組的。

### 已下市的股票拿不到

yfinance 查不到下市股票。那些列會保持未還原，`loader` 會在
`is_adjusted` 欄位標記。這是已知限制，必須隨報告揭露。

## 寫入原則

`INSERT OR IGNORE`——既有的已驗證列永不覆寫（禁令 9）。

用法：
    .venv/bin/python scripts/backfill_adj_close.py
    .venv/bin/python scripts/backfill_adj_close.py --start 2015-01-01
    .venv/bin/python scripts/backfill_adj_close.py --status
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
import warnings
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

warnings.filterwarnings("ignore")

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "history.db"
"""與 backfill_finmind_history.py 同一個庫，不碰 qlib-tw-trader 的 data.db"""

BATCH_SIZE = 40
"""一次向 yfinance 要幾檔。太大容易整批失敗，太小請求次數過多"""

BATCH_DELAY = 1.0
BUSY_TIMEOUT_MS = 30_000
"""價格回補可能同時在寫，給 SQLite 足夠的等待時間"""

SUFFIX_CANDIDATES = (".TW", ".TWO")
"""上市 .TW、上櫃 .TWO。標的池以上市為主，但保留後備"""


def ensure_schema(con: sqlite3.Connection) -> None:
    con.executescript("""
        CREATE TABLE IF NOT EXISTS adj_backfill_log (
            stock_id   TEXT NOT NULL PRIMARY KEY,
            status     TEXT NOT NULL,
            rows       INTEGER NOT NULL DEFAULT 0,
            message    TEXT NOT NULL DEFAULT '',
            fetched_at TEXT NOT NULL
        );
    """)
    con.commit()


def target_stocks(con: sqlite3.Connection, start: date, end: date) -> list[str]:
    """
    需要還原價的標的：在 stock_daily 有資料、但 stock_daily_adj 缺的。

    只挑真的缺的，讓腳本可以重複執行而不浪費請求。
    """
    rows = con.execute(
        """
        SELECT d.stock_id, COUNT(*) AS missing
        FROM stock_daily AS d
        LEFT JOIN stock_daily_adj AS a
               ON a.stock_id = d.stock_id AND a.date = d.date
        WHERE d.date >= ? AND d.date <= ? AND a.adj_close IS NULL
        GROUP BY d.stock_id
        HAVING missing > 0
        ORDER BY missing DESC
        """,
        (start.isoformat(), end.isoformat()),
    ).fetchall()
    return [r[0] for r in rows]


def already_done(con: sqlite3.Connection) -> set[str]:
    rows = con.execute(
        "SELECT stock_id FROM adj_backfill_log WHERE status IN ('ok', 'unavailable')"
    ).fetchall()
    return {r[0] for r in rows}


def fetch_batch(stock_ids: list[str], start: date, end: date) -> dict[str, object]:
    """
    批次向 yfinance 取 Adj Close。

    Returns:
        {代號: Series}；查不到的代號不會出現在結果裡
    """
    import pandas as pd
    import yfinance as yf

    tickers = {f"{sid}.TW": sid for sid in stock_ids}
    frame = yf.download(
        list(tickers),
        start=start.isoformat(),
        end=end.isoformat(),
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    if frame is None or frame.empty:
        return {}

    out: dict[str, object] = {}
    for ticker, sid in tickers.items():
        try:
            if isinstance(frame.columns, pd.MultiIndex):
                series = frame[ticker]["Adj Close"]
            else:
                series = frame["Adj Close"]
        except KeyError:
            continue
        series = series.dropna()
        if not series.empty:
            out[sid] = series
    return out


def store(con: sqlite3.Connection, stock_id: str, series: object) -> int:
    """寫入 stock_daily_adj（不覆寫既有列）"""
    now = datetime.now().isoformat(timespec="seconds")
    payload = [
        (stock_id, ts.date().isoformat(), float(value), now)
        for ts, value in series.items()  # type: ignore[union-attr]
        if float(value) > 0
    ]
    if not payload:
        return 0
    con.executemany(
        "INSERT OR IGNORE INTO stock_daily_adj "
        "(stock_id, date, adj_close, created_at) VALUES (?, ?, ?, ?)",
        payload,
    )
    return len(payload)


def print_status(con: sqlite3.Connection) -> None:
    print("=" * 76)
    print("還原價回補進度")
    print("=" * 76)
    rows = con.execute(
        "SELECT status, COUNT(*) FROM adj_backfill_log GROUP BY status"
    ).fetchall()
    if not rows:
        print("  尚未開始")
    for status, n in rows:
        print(f"  {status:<14}{n:>5} 檔")

    n, ids, lo, hi = con.execute(
        "SELECT COUNT(*), COUNT(DISTINCT stock_id), MIN(date), MAX(date) "
        "FROM stock_daily_adj"
    ).fetchone()
    print(f"\n  stock_daily_adj  {n:,} 列  {ids} 檔  {lo} ~ {hi}")

    covered, total = con.execute(
        """
        SELECT SUM(CASE WHEN a.adj_close IS NOT NULL THEN 1 ELSE 0 END), COUNT(*)
        FROM stock_daily AS d
        LEFT JOIN stock_daily_adj AS a
               ON a.stock_id = d.stock_id AND a.date = d.date
        """
    ).fetchone()
    if total:
        print(f"  還原價覆蓋率     {covered:,} / {total:,} = {covered / total * 100:.1f}%")
    print("=" * 76)


def main() -> None:
    parser = argparse.ArgumentParser(description="還原收盤價回補")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default=date.today().isoformat())
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--db", default=str(DB_PATH))
    args = parser.parse_args()

    con = sqlite3.connect(Path(args.db), timeout=BUSY_TIMEOUT_MS / 1000)
    con.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
    ensure_schema(con)

    if args.status:
        print_status(con)
        con.close()
        return

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    pending = [s for s in target_stocks(con, start, end) if s not in already_done(con)]
    print("=" * 76)
    print("還原收盤價回補（yfinance Adj Close）")
    print("=" * 76)
    print(f"區間 {start} ~ {end}｜待處理 {len(pending)} 檔｜批次 {args.batch_size}")
    print("寫入模式 INSERT OR IGNORE（既有資料永不覆寫）")
    print()

    if not pending:
        print_status(con)
        con.close()
        return

    now = datetime.now().isoformat(timespec="seconds")
    total_rows = 0
    unavailable: list[str] = []
    started = time.time()

    for i in range(0, len(pending), args.batch_size):
        batch = pending[i : i + args.batch_size]
        try:
            fetched = fetch_batch(batch, start, end)
        except Exception as exc:  # noqa: BLE001 — 單批失敗不該中斷整輪
            print(f"  批次 {i // args.batch_size + 1} 失敗：{type(exc).__name__} {exc}")
            for sid in batch:
                con.execute(
                    "INSERT OR REPLACE INTO adj_backfill_log VALUES (?, ?, ?, ?, ?)",
                    (sid, "error", 0, f"{type(exc).__name__}: {exc}"[:300], now),
                )
            con.commit()
            time.sleep(BATCH_DELAY * 3)
            continue

        for sid in batch:
            series = fetched.get(sid)
            if series is None:
                unavailable.append(sid)
                con.execute(
                    "INSERT OR REPLACE INTO adj_backfill_log VALUES (?, ?, ?, ?, ?)",
                    (sid, "unavailable", 0, "yfinance 查無資料（多為已下市）", now),
                )
                continue
            rows = store(con, sid, series)
            total_rows += rows
            con.execute(
                "INSERT OR REPLACE INTO adj_backfill_log VALUES (?, ?, ?, ?, ?)",
                (sid, "ok", rows, "", now),
            )

        con.commit()
        done = min(i + args.batch_size, len(pending))
        elapsed = time.time() - started
        eta = elapsed / done * (len(pending) - done)
        print(f"  {done}/{len(pending)}  已寫入 {total_rows:,} 列"
              f"  查無 {len(unavailable)} 檔"
              f"  已花 {elapsed / 60:.1f} 分  預估剩 {eta / 60:.1f} 分", flush=True)
        time.sleep(BATCH_DELAY)

    print()
    if unavailable:
        print(f"yfinance 查無 {len(unavailable)} 檔（多為已下市，"
              f"這些列會保持未還原並由 loader 標記）：")
        print(f"  {unavailable[:20]}{' ...' if len(unavailable) > 20 else ''}")
        print()

    print_status(con)
    con.close()


if __name__ == "__main__":
    main()
