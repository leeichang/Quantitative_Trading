#!/usr/bin/env python3
"""
TWSE 歷史資料回補（2015 起）

## 為什麼要做

目前資料只有 2023-01 起約 915 個交易日。扣掉 250 天暖機 + 訓練期後，
OOS 只剩 1.5 年，而且全是多頭。`04_路線A驗證結果.md` 的每一條限制都
指向同一件事：**樣本太少**。

回補到 2015 可以讓 OOS 涵蓋 2018 年的貿易戰修正與 2022 年的空頭。

## 來源與驗證

三個 TWSE 官方端點，都以 `date=YYYYMMDD` 取單日：

```
MI_INDEX    每日收盤行情（全部）   OHLCV，含 ETF 與當年掛牌的所有證券
T86         三大法人買賣超         個股別
MI_MARGN    融資融券彙總           個股別
```

**與既有資料的重疊期交叉驗證**（2024-01-03、2025-07-15）：

```
MI_INDEX    97 / 97 完全吻合
T86         97 / 97 完全吻合
MI_MARGN    93 / 93 完全吻合
```

驗證過程抓到一個會污染八年資料的 bug：T86 在 2015 年是 16 欄、2018 年
起 19 欄。用固定索引解析時 97 檔只有 1 檔吻合，而且不拋錯。已改成依
欄位名稱取值，見 `taiwan_quant/data/twse_history.py`。

## 為什麼不用 yfinance

yfinance 的 OHLC 是**分割還原價**，TWSE 是**原始成交價**。實測 2881
富邦金因股票股利被 Yahoo 記為 split，四個價格欄都差 2.4390%。在 2023-01
接起來會產生假跳空。而且 yfinance 查不到已下市的股票——那正是
survivorship bias 的來源。

## 寫入原則

- **一律 `INSERT OR IGNORE`**：既有的已驗證資料永不被覆寫（禁令 9）
- **可中斷續跑**：`twse_backfill_log` 記錄每個 (端點, 日期) 的狀態
- **非交易日標記為 `no_data`**，不重試
- **解析失敗標記為 `error` 並保留訊息**，可事後單獨重跑

用法：
    .venv/bin/python scripts/backfill_twse_history.py                    # 2015-01 ~ 2023-01
    .venv/bin/python scripts/backfill_twse_history.py --endpoints prices
    .venv/bin/python scripts/backfill_twse_history.py --retry-errors
    .venv/bin/python scripts/backfill_twse_history.py --status           # 只看進度
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from taiwan_quant.data.twse_history import (  # noqa: E402
    TwseParseError,
    parse_mi_index,
    parse_mi_margn,
    parse_t86,
)

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "history.db"
"""
**獨立的資料庫檔**，不寫進 qlib-tw-trader 的 data.db。

兩個理由：

1. **鎖競爭。** qlib-tw-trader 的 uvicorn 服務長時間開著同一個 DB，
   實測回補行程會卡死（WAL 30 秒零成長、CPU 0.85 秒/5 分鐘）。
2. **禁令 9。** 已驗證的 production 資料完全不碰，比 INSERT OR IGNORE
   更徹底。

既然 TWSE 與既有資料在重疊期 97/97 吻合，就整段重抓 2015~今天——
一個來源、零接縫，不需要在 2023-01 拼接。
"""

USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
REQUEST_DELAY = 0.6
"""
請求間隔（秒）。

實測零延遲 8/8 成功，但要連續跑數千次。0.6 秒是禮貌性設定——
被 TWSE 暫時封鎖的代價（整批重來）遠大於多花的時間。
"""

MAX_RETRIES = 3
RETRY_BACKOFF = 5.0

ENDPOINTS = {
    "prices": "https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX"
              "?date={d}&type=ALLBUT0999&response=json",
    "institutional": "https://www.twse.com.tw/rwd/zh/fund/T86"
                     "?date={d}&selectType=ALL&response=json",
    "margin": "https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN"
              "?date={d}&selectType=ALL&response=json",
}

DEFAULT_START = date(2015, 1, 1)
DEFAULT_END = date.today()
"""整段重抓。重疊期已驗證與既有資料完全吻合，不需要拼接"""


@dataclass(frozen=True)
class FetchOutcome:
    """單次抓取的結果"""

    status: str
    """ok / no_data / error"""

    rows: int
    message: str = ""


# ══════════════════════════════════════════════════════════════
# 資料庫
# ══════════════════════════════════════════════════════════════


def ensure_schema(con: sqlite3.Connection) -> None:
    """
    建立資料表。

    schema 刻意與 qlib-tw-trader 的 data.db 完全一致，
    讓 taiwan-quant 的 loader 只要換 db_path 就能用。
    """
    con.executescript("""
        PRAGMA journal_mode = WAL;

        CREATE TABLE IF NOT EXISTS stock_daily (
            id        INTEGER PRIMARY KEY,
            stock_id  VARCHAR(10) NOT NULL,
            date      DATE NOT NULL,
            open      NUMERIC(10, 2) NOT NULL,
            high      NUMERIC(10, 2) NOT NULL,
            low       NUMERIC(10, 2) NOT NULL,
            close     NUMERIC(10, 2) NOT NULL,
            volume    INTEGER NOT NULL,
            created_at DATETIME NOT NULL,
            CONSTRAINT uq_stock_daily UNIQUE (stock_id, date)
        );
        CREATE INDEX IF NOT EXISTS ix_stock_daily_stock_id ON stock_daily (stock_id);
        CREATE INDEX IF NOT EXISTS ix_stock_daily_date ON stock_daily (date);

        CREATE TABLE IF NOT EXISTS stock_daily_adj (
            id        INTEGER PRIMARY KEY,
            stock_id  VARCHAR(10) NOT NULL,
            date      DATE NOT NULL,
            adj_close NUMERIC(10, 2) NOT NULL,
            created_at DATETIME NOT NULL,
            CONSTRAINT uq_stock_daily_adj UNIQUE (stock_id, date)
        );
        CREATE INDEX IF NOT EXISTS ix_adj_stock_id ON stock_daily_adj (stock_id);

        CREATE TABLE IF NOT EXISTS stock_daily_institutional (
            id          INTEGER PRIMARY KEY,
            stock_id    VARCHAR(10) NOT NULL,
            date        DATE NOT NULL,
            foreign_buy  INTEGER NOT NULL,
            foreign_sell INTEGER NOT NULL,
            trust_buy    INTEGER NOT NULL,
            trust_sell   INTEGER NOT NULL,
            dealer_buy   INTEGER NOT NULL,
            dealer_sell  INTEGER NOT NULL,
            created_at   DATETIME NOT NULL,
            CONSTRAINT uq_stock_daily_inst UNIQUE (stock_id, date)
        );
        CREATE INDEX IF NOT EXISTS ix_inst_stock_id ON stock_daily_institutional (stock_id);

        CREATE TABLE IF NOT EXISTS stock_daily_margin (
            id        INTEGER PRIMARY KEY,
            stock_id  VARCHAR(10) NOT NULL,
            date      DATE NOT NULL,
            margin_buy     INTEGER NOT NULL,
            margin_sell    INTEGER NOT NULL,
            margin_balance INTEGER NOT NULL,
            short_buy      INTEGER NOT NULL,
            short_sell     INTEGER NOT NULL,
            short_balance  INTEGER NOT NULL,
            created_at     DATETIME NOT NULL,
            CONSTRAINT uq_stock_daily_margin UNIQUE (stock_id, date)
        );
        CREATE INDEX IF NOT EXISTS ix_margin_stock_id ON stock_daily_margin (stock_id);

        CREATE TABLE IF NOT EXISTS twse_backfill_log (
            endpoint    TEXT NOT NULL,
            trade_date  TEXT NOT NULL,
            status      TEXT NOT NULL,
            rows        INTEGER NOT NULL DEFAULT 0,
            message     TEXT NOT NULL DEFAULT '',
            fetched_at  TEXT NOT NULL,
            PRIMARY KEY (endpoint, trade_date)
        );

        CREATE TABLE IF NOT EXISTS stock_universe_history (
            as_of_date  TEXT NOT NULL,
            stock_id    TEXT NOT NULL,
            name        TEXT NOT NULL,
            turnover    REAL NOT NULL,
            rank        INTEGER NOT NULL,
            is_etf      INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (as_of_date, stock_id)
        );
        CREATE INDEX IF NOT EXISTS ix_universe_history_date
            ON stock_universe_history (as_of_date);
    """)
    con.commit()


def completed_dates(con: sqlite3.Connection, endpoint: str) -> set[str]:
    """已成功或確認無資料的日期——不再重抓"""
    rows = con.execute(
        "SELECT trade_date FROM twse_backfill_log "
        "WHERE endpoint = ? AND status IN ('ok', 'no_data')",
        (endpoint,),
    ).fetchall()
    return {r[0] for r in rows}


def log_outcome(
    con: sqlite3.Connection, endpoint: str, day: str, outcome: FetchOutcome
) -> None:
    con.execute(
        "INSERT OR REPLACE INTO twse_backfill_log "
        "(endpoint, trade_date, status, rows, message, fetched_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (endpoint, day, outcome.status, outcome.rows,
         outcome.message[:500], datetime.now().isoformat(timespec="seconds")),
    )


# ══════════════════════════════════════════════════════════════
# 抓取
# ══════════════════════════════════════════════════════════════


def fetch_json(client: httpx.Client, url: str) -> dict | None:
    """
    抓一個 JSON，失敗時退避重試。

    Returns:
        解析後的 dict；重試耗盡仍失敗時回 None（由呼叫端記為 error）
    """
    for attempt in range(MAX_RETRIES):
        try:
            response = client.get(url, timeout=30)
            if response.status_code == 200:
                return response.json()
            # 429 / 5xx：退避後重試
            time.sleep(RETRY_BACKOFF * (attempt + 1))
        except (httpx.HTTPError, ValueError):
            time.sleep(RETRY_BACKOFF * (attempt + 1))
    return None


def store_prices(con: sqlite3.Connection, payload: dict, day: str) -> FetchOutcome:
    """寫入 stock_daily（不覆寫既有列）"""
    try:
        quotes = parse_mi_index(payload, trade_date=day)
    except TwseParseError as exc:
        return _classify(exc)

    now = datetime.now().isoformat(timespec="seconds")
    con.executemany(
        "INSERT OR IGNORE INTO stock_daily "
        "(stock_id, date, open, high, low, close, volume, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [(q.stock_id, q.date, q.open, q.high, q.low, q.close, q.volume, now)
         for q in quotes],
    )
    return FetchOutcome("ok", len(quotes))


def store_institutional(con: sqlite3.Connection, payload: dict, day: str) -> FetchOutcome:
    try:
        rows = parse_t86(payload, trade_date=day)
    except TwseParseError as exc:
        return _classify(exc)

    now = datetime.now().isoformat(timespec="seconds")
    con.executemany(
        "INSERT OR IGNORE INTO stock_daily_institutional "
        "(stock_id, date, foreign_buy, foreign_sell, trust_buy, trust_sell, "
        " dealer_buy, dealer_sell, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [(r.stock_id, r.date, r.foreign_buy, r.foreign_sell,
          r.trust_buy, r.trust_sell, r.dealer_buy, r.dealer_sell, now)
         for r in rows],
    )
    return FetchOutcome("ok", len(rows))


def store_margin(con: sqlite3.Connection, payload: dict, day: str) -> FetchOutcome:
    try:
        rows = parse_mi_margn(payload, trade_date=day)
    except TwseParseError as exc:
        return _classify(exc)

    now = datetime.now().isoformat(timespec="seconds")
    con.executemany(
        "INSERT OR IGNORE INTO stock_daily_margin "
        "(stock_id, date, margin_buy, margin_sell, margin_balance, "
        " short_buy, short_sell, short_balance, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [(r.stock_id, r.date, r.margin_buy, r.margin_sell, r.margin_balance,
          r.short_buy, r.short_sell, r.short_balance, now)
         for r in rows],
    )
    return FetchOutcome("ok", len(rows))


def _classify(exc: TwseParseError) -> FetchOutcome:
    """
    區分「那天沒開市」與「解析壞了」。

    這個區分很重要：`no_data` 不會重試，`error` 會。把解析 bug 誤標成
    `no_data`，那天的資料就永久缺失且無人察覺。
    """
    text = str(exc)
    if "stat 不是 OK" in text:
        return FetchOutcome("no_data", 0, text)
    return FetchOutcome("error", 0, text)


STORE = {
    "prices": store_prices,
    "institutional": store_institutional,
    "margin": store_margin,
}


# ══════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════


def weekdays(start: date, end: date) -> Iterator[date]:
    """列出區間內的平日（週末必定不開市，先濾掉省請求）"""
    day = start
    while day <= end:
        if day.weekday() < 5:
            yield day
        day += timedelta(days=1)


def run_endpoint(
    con: sqlite3.Connection,
    client: httpx.Client,
    endpoint: str,
    start: date,
    end: date,
    delay: float,
    retry_errors: bool,
) -> None:
    """回補單一端點的整個區間"""
    done = completed_dates(con, endpoint)
    if retry_errors:
        pass  # done 已排除 error 狀態，會自動重試

    pending = [d for d in weekdays(start, end) if d.isoformat() not in done]
    total = len(pending)
    print(f"\n[{endpoint}] 待處理 {total} 天"
          f"（已完成 {len(done)} 天）", flush=True)
    if not total:
        return

    url_template = ENDPOINTS[endpoint]
    store = STORE[endpoint]
    counts = {"ok": 0, "no_data": 0, "error": 0}
    started = time.time()

    for i, day in enumerate(pending, start=1):
        iso = day.isoformat()
        payload = fetch_json(client, url_template.format(d=day.strftime("%Y%m%d")))

        if payload is None:
            outcome = FetchOutcome("error", 0, "HTTP 重試耗盡")
        else:
            outcome = store(con, payload, iso)

        log_outcome(con, endpoint, iso, outcome)
        counts[outcome.status] += 1

        if i % 50 == 0 or i == total:
            con.commit()
            elapsed = time.time() - started
            eta = elapsed / i * (total - i)
            print(f"  {i}/{total}  {iso}  "
                  f"ok={counts['ok']} 無資料={counts['no_data']} 錯誤={counts['error']}"
                  f"  已花 {elapsed/60:.1f} 分  預估剩 {eta/60:.1f} 分", flush=True)

        time.sleep(delay)

    con.commit()
    print(f"[{endpoint}] 完成：{counts}", flush=True)


def print_status(con: sqlite3.Connection) -> None:
    """列出各端點與各表的目前狀態"""
    print("=" * 76)
    print("回補進度")
    print("=" * 76)
    rows = con.execute(
        "SELECT endpoint, status, COUNT(*), MIN(trade_date), MAX(trade_date) "
        "FROM twse_backfill_log GROUP BY endpoint, status ORDER BY endpoint, status"
    ).fetchall()
    if not rows:
        print("  尚未開始")
    for endpoint, status, n, lo, hi in rows:
        print(f"  {endpoint:<15}{status:<10}{n:>6} 天   {lo} ~ {hi}")

    print()
    print("資料表現況：")
    for table in ("stock_daily", "stock_daily_institutional",
                  "stock_daily_margin", "stock_daily_adj"):
        try:
            n, lo, hi = con.execute(
                f"SELECT COUNT(*), MIN(date), MAX(date) FROM {table}"
            ).fetchone()
            ids = con.execute(
                f"SELECT COUNT(DISTINCT stock_id) FROM {table}"
            ).fetchone()[0]
            print(f"  {table:<28}{n:>9,} 列  {ids:>5} 檔   {lo} ~ {hi}")
        except sqlite3.Error as exc:
            print(f"  {table:<28}讀取失敗：{exc}")

    errors = con.execute(
        "SELECT endpoint, trade_date, message FROM twse_backfill_log "
        "WHERE status = 'error' ORDER BY trade_date LIMIT 5"
    ).fetchall()
    if errors:
        n = con.execute(
            "SELECT COUNT(*) FROM twse_backfill_log WHERE status = 'error'"
        ).fetchone()[0]
        print()
        print(f"錯誤 {n} 筆（前 5 筆）：")
        for endpoint, day, message in errors:
            print(f"  {endpoint} {day}  {message[:80]}")
    print("=" * 76)


def main() -> None:
    parser = argparse.ArgumentParser(description="TWSE 歷史資料回補")
    parser.add_argument("--start", default=DEFAULT_START.isoformat())
    parser.add_argument("--end", default=DEFAULT_END.isoformat())
    parser.add_argument("--endpoints", nargs="*", default=list(ENDPOINTS),
                        choices=list(ENDPOINTS))
    parser.add_argument("--delay", type=float, default=REQUEST_DELAY)
    parser.add_argument("--retry-errors", action="store_true",
                        help="重試先前標記為 error 的日期")
    parser.add_argument("--status", action="store_true", help="只印進度，不抓取")
    parser.add_argument("--db", default=str(DB_PATH))
    args = parser.parse_args()

    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    ensure_schema(con)

    if args.status:
        print_status(con)
        con.close()
        return

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    print("=" * 76)
    print("TWSE 歷史資料回補")
    print("=" * 76)
    print(f"區間 {start} ~ {end}｜端點 {args.endpoints}｜間隔 {args.delay}s")
    print(f"資料庫 {db_path}")
    print("寫入模式 INSERT OR IGNORE（既有資料永不覆寫）")

    with httpx.Client(headers={"User-Agent": USER_AGENT}) as client:
        for endpoint in args.endpoints:
            run_endpoint(con, client, endpoint, start, end,
                         args.delay, args.retry_errors)

    print()
    print_status(con)
    con.close()


if __name__ == "__main__":
    main()
