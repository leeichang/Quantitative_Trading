#!/usr/bin/env python3
"""
還原股價回補 — 由官方除權息紀錄推算（補 yfinance 的缺口）

## 為什麼需要

`backfill_adj_close.py` 走 yfinance 的 Adj Close，但它**查不到已下市的
股票**——長歷史回補後有 107 檔沒有還原價。

下市股票正是 survivorship 的關鍵樣本（2015 年標的池裡的日月光、矽品
都已下市）。因為拿不到還原價就排除它們，等於把好不容易解掉的
survivorship bias 又放回來。

## 方法

FinMind 的 `TaiwanStockDividendResult` 免費層可用，且對下市股票也有
紀錄（實測 2311 日月光 4 筆、2325 矽品 3 筆）。直接給：

```
before_price   除權息前收盤價
after_price    除權息參考價
factor = after_price / before_price
```

計算邏輯與測試見 `taiwan_quant/data/adjustment.py`。

## 與 yfinance 的比對（實測 6 檔）

```
代號    除息次數   中位相對誤差      最大      >1% 的列
2330      34      0.000000     0.008152        0
1301      13      0.000000     0.006313        0
2412      13      0.000082     0.008249        0
2454      16      0.000016     0.023307        3
2317      13      0.000000     0.203089      920
2881      18      0.000000     0.022222      963
```

2317 與 2881 的大差異**不是本方法的錯**。查證兩者的因子變動日：

```
2317  除權息法   12 個變動日，全部是真實除息日
      yfinance   42 個變動日，含 2016-08-22 ~ 09-13 的連續段
```

那段連續變動不是公司行為，是 yfinance 自己的資料抖動。
**除權息法只在官方除息日變動，方法學上更乾淨。**

## 但仍然只補缺口，不覆寫

既有的 yfinance 還原價已與 qlib-tw-trader 的 production 資料驗證過
（中位誤差 2e-6）。覆寫它會讓先前所有的驗證結論不可比。

所以：**只寫入目前完全沒有還原價的標的**（`INSERT OR IGNORE`）。
代價是資料庫裡混了兩種方法，必須隨報告揭露。

用法：
    .venv/bin/python scripts/backfill_adj_from_dividends.py
    .venv/bin/python scripts/backfill_adj_from_dividends.py --status
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from datetime import date, datetime
from pathlib import Path

import httpx
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from taiwan_quant.data.adjustment import (  # noqa: E402
    back_adjust,
    events_from_dividend_result,
)

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "history.db"

API = "https://api.finmindtrade.com/api/v4/data"
DATASET = "TaiwanStockDividendResult"
USER_AGENT = "Mozilla/5.0"

TOKEN_ENV = "FINMIND_API_TOKEN"
PLACEHOLDER_TOKENS = frozenset({"", "your", "your_token", "changeme", "none"})

REQUEST_DELAY = 1.5
QUOTA_BACKOFF = 90.0
MAX_RETRIES = 4

METHOD = "dividend_result"
"""寫進 log 的方法標記，讓下游分得出哪些列是哪種方法算的"""


def api_token() -> str | None:
    raw = (os.environ.get(TOKEN_ENV) or "").strip()
    return None if raw.lower() in PLACEHOLDER_TOKENS else raw


def ensure_schema(con: sqlite3.Connection) -> None:
    con.executescript("""
        CREATE TABLE IF NOT EXISTS adj_dividend_log (
            stock_id TEXT PRIMARY KEY,
            status   TEXT NOT NULL,
            events   INTEGER NOT NULL DEFAULT 0,
            rows     INTEGER NOT NULL DEFAULT 0,
            method   TEXT NOT NULL DEFAULT '',
            message  TEXT NOT NULL DEFAULT '',
            fetched_at TEXT NOT NULL
        );
    """)
    con.commit()


def missing_adj_stocks(con: sqlite3.Connection) -> list[str]:
    """
    完全沒有還原價的標的。

    **只挑「完全沒有」的**，不碰部分覆蓋的——後者混用兩種方法會在
    接縫處產生假跳空。
    """
    rows = con.execute(
        """
        SELECT d.stock_id, COUNT(*) AS bars
        FROM stock_daily AS d
        WHERE NOT EXISTS (
            SELECT 1 FROM stock_daily_adj AS a WHERE a.stock_id = d.stock_id
        )
        GROUP BY d.stock_id
        ORDER BY bars DESC
        """
    ).fetchall()
    return [r[0] for r in rows]


def done_ids(con: sqlite3.Connection) -> set[str]:
    rows = con.execute(
        "SELECT stock_id FROM adj_dividend_log WHERE status IN ('ok', 'no_events')"
    ).fetchall()
    return {r[0] for r in rows}


def fetch_dividends(client: httpx.Client, stock_id: str,
                    token: str | None) -> tuple[bool, list[dict], str]:
    """抓單檔的除權息紀錄；額度限制時退避重試"""
    params = {"dataset": DATASET, "data_id": stock_id,
              "start_date": "2010-01-01", "end_date": date.today().isoformat()}
    if token:
        params["token"] = token

    for attempt in range(MAX_RETRIES):
        try:
            response = client.get(API, params=params, timeout=60)
            if response.status_code == 200:
                body = response.json()
                if body.get("status") == 200:
                    return True, body.get("data") or [], ""
                message = str(body.get("msg", ""))[:200]
                if "level" in message.lower() or "limit" in message.lower():
                    time.sleep(QUOTA_BACKOFF)
                    continue
                return False, [], message
            if response.status_code in (402, 429, 500, 502, 503):
                time.sleep(QUOTA_BACKOFF)
                continue
            return False, [], f"HTTP {response.status_code}"
        except (httpx.HTTPError, ValueError) as exc:
            time.sleep(10.0 * (attempt + 1))
            if attempt == MAX_RETRIES - 1:
                return False, [], f"{type(exc).__name__}: {exc}"
    return False, [], "重試耗盡（多為額度限制）"


def store(con: sqlite3.Connection, stock_id: str, adjusted: pd.Series) -> int:
    now = datetime.now().isoformat(timespec="seconds")
    payload = [
        (stock_id, ts.date().isoformat(), float(value), now)
        for ts, value in adjusted.items()
        if pd.notna(value) and float(value) > 0
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
    print("=" * 78)
    print("除權息還原價回補進度")
    print("=" * 78)
    for status, n, rows in con.execute(
        "SELECT status, COUNT(*), SUM(rows) FROM adj_dividend_log GROUP BY status"
    ):
        print(f"  {status:<14}{n:>5} 檔   寫入 {rows or 0:>9,} 列")

    n, ids = con.execute(
        "SELECT COUNT(*), COUNT(DISTINCT stock_id) FROM stock_daily_adj").fetchone()
    covered, total = con.execute(
        """
        SELECT SUM(CASE WHEN a.adj_close IS NOT NULL THEN 1 ELSE 0 END), COUNT(*)
        FROM stock_daily AS d
        LEFT JOIN stock_daily_adj AS a
               ON a.stock_id = d.stock_id AND a.date = d.date
        """
    ).fetchone()
    print(f"\n  stock_daily_adj   {n:,} 列  {ids} 檔")
    if total:
        print(f"  還原價覆蓋率      {covered:,} / {total:,} = {covered/total*100:.2f}%")

    universe_cov = con.execute(
        """
        SELECT ROUND(SUM(CASE WHEN a.adj_close IS NOT NULL THEN 1 ELSE 0 END)
                     * 100.0 / COUNT(*), 2)
        FROM stock_daily d
        JOIN (SELECT DISTINCT stock_id FROM stock_universe_history) u USING (stock_id)
        LEFT JOIN stock_daily_adj a ON a.stock_id = d.stock_id AND a.date = d.date
        """
    ).fetchone()[0]
    print(f"  標的池成員覆蓋率   {universe_cov}%")
    print("=" * 78)


def main() -> None:
    parser = argparse.ArgumentParser(description="由除權息紀錄回補還原價")
    parser.add_argument("--delay", type=float, default=REQUEST_DELAY)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--db", default=str(DB_PATH))
    args = parser.parse_args()

    con = sqlite3.connect(Path(args.db), timeout=30)
    ensure_schema(con)

    if args.status:
        print_status(con)
        con.close()
        return

    token = api_token()
    pending = [s for s in missing_adj_stocks(con) if s not in done_ids(con)]
    if args.limit:
        pending = pending[: args.limit]

    print("=" * 78)
    print("還原價回補 — 由官方除權息紀錄推算")
    print("=" * 78)
    print(f"資料庫 {args.db}")
    print(f"認證   {'已設定 token' if token else '匿名（免費層）'}")
    print(f"待處理 {len(pending)} 檔（目前完全沒有還原價的標的）")
    print("寫入模式 INSERT OR IGNORE（既有 yfinance 還原價永不覆寫）")
    print()

    if not pending:
        print_status(con)
        con.close()
        return

    counts = {"ok": 0, "no_events": 0, "error": 0}
    written = 0
    started = time.time()

    with httpx.Client(headers={"User-Agent": USER_AGENT}) as client:
        for i, stock_id in enumerate(pending, start=1):
            ok, rows, message = fetch_dividends(client, stock_id, token)
            now = datetime.now().isoformat(timespec="seconds")

            if not ok:
                con.execute(
                    "INSERT OR REPLACE INTO adj_dividend_log VALUES (?,?,?,?,?,?,?)",
                    (stock_id, "error", 0, 0, METHOD, message, now))
                counts["error"] += 1
            else:
                events = events_from_dividend_result(rows)
                closes = pd.read_sql_query(
                    "SELECT date, close FROM stock_daily WHERE stock_id = ? "
                    "ORDER BY date", con, params=(stock_id,),
                    parse_dates=["date"]).set_index("date")["close"].astype(float)

                if closes.empty:
                    con.execute(
                        "INSERT OR REPLACE INTO adj_dividend_log VALUES (?,?,?,?,?,?,?)",
                        (stock_id, "error", len(events), 0, METHOD, "無價格資料", now))
                    counts["error"] += 1
                else:
                    # 沒有除息紀錄時還原價等於原始價——那是正確的結果，
                    # 不是「沒資料」。仍然寫入，讓覆蓋率反映真實情況。
                    adjusted = back_adjust(closes, events)
                    n = store(con, stock_id, adjusted)
                    written += n
                    status = "ok" if events else "no_events"
                    con.execute(
                        "INSERT OR REPLACE INTO adj_dividend_log VALUES (?,?,?,?,?,?,?)",
                        (stock_id, status, len(events), n, METHOD, "", now))
                    counts[status] += 1

            if i % 20 == 0 or i == len(pending):
                con.commit()
                elapsed = time.time() - started
                eta = elapsed / i * (len(pending) - i)
                print(f"  {i}/{len(pending)}  {stock_id}  寫入 {written:,} 列  "
                      f"有除息 {counts['ok']} 無除息 {counts['no_events']} "
                      f"錯 {counts['error']}  "
                      f"已花 {elapsed/60:.1f} 分  剩 {eta/60:.1f} 分", flush=True)

            time.sleep(args.delay)

    con.commit()
    print()
    print_status(con)
    con.close()


if __name__ == "__main__":
    main()
