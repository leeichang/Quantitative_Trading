#!/usr/bin/env python3
"""
FinMind 歷史資料回補（2015 起）

## 為什麼改用 FinMind

原本走 TWSE 官方日報，重疊期交叉驗證 97/97 完全吻合。但連續抓取數百次
之後被 HiNet CDN 的 WAF 擋下：

    HTTP 307  「因為安全性考量，您所執行的頁面無法呈現。」

那是防爬蟲機制，**不繞過**。改用 FinMind——它是為程式化存取設計的 API，
而且請求經濟性好上兩個量級：

```
TWSE      一天一請求  ×  2,850 天  ×  3 個端點  =  8,550 次
FinMind   一檔一請求（整段 11 年） ×  N 檔  ×  3 個資料集
```

實測 2330 的 11 年價格：一次請求、2,850 列、0.6 秒。

### 數值一致性

FinMind 2015-01-05 的 2330：`open 140.5 / max 140.5 / min 137.5`
與 TWSE MI_INDEX 同日完全相同。兩者都是**原始成交價**。

## survivorship 怎麼解

```
TaiwanStockInfo        現存上市證券
TaiwanStockDelisting   725 檔下市證券及下市日期
```

2015 之後才下市的股票，在我們的回測區間內是真實可交易的標的。
**只用今天的名單回溯歷史，就是把它們全部漏掉。**

歷史標的池由價格資料裡的 `Trading_money`（成交金額）排名建出，
逐季快照。那是流動性代理，不是市值——市值需要股數，免費層沒有。

## 免費層的限制（必須揭露）

```
TaiwanStockPriceAdj   還原股價    需付費 → 改用 yfinance Adj Close
不指定 data_id 的批次查詢          需付費 → 一檔一請求
請求頻率                            有限額 → 遇到就退避重試
```

## 寫入位置

`taiwan-quant/data/history.db`，**不碰 qlib-tw-trader 的 data.db**。
schema 與其完全一致，loader 只要換 `db_path` 就能用。

用法：
    .venv/bin/python scripts/backfill_finmind_history.py --phase master
    .venv/bin/python scripts/backfill_finmind_history.py --phase prices
    .venv/bin/python scripts/backfill_finmind_history.py --phase chips
    .venv/bin/python scripts/backfill_finmind_history.py --status
"""

from __future__ import annotations

import argparse
import os
import re
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "history.db"

API = "https://api.finmindtrade.com/api/v4/data"
USER_AGENT = "Mozilla/5.0"

TOKEN_ENV = "FINMIND_API_TOKEN"
"""
可選的 API token（環境變數）。

**不得硬編碼。** 沒設也能跑，只是走匿名的低額度。
qlib-tw-trader 的 `.env` 裡目前是佔位字串 `your`，視同未設定。
"""

PLACEHOLDER_TOKENS = frozenset({"", "your", "your_token", "changeme", "none"})

REQUEST_DELAY = 1.2
"""請求間隔。FinMind 免費層有頻率限制，寧可慢也不要被鎖"""

QUOTA_BACKOFF = 90.0
"""撞到額度時的等待秒數"""

MAX_RETRIES = 4

ORDINARY_CODE = re.compile(r"^\d{4}$")

DEFAULT_START = "2015-01-01"

DATASETS = {
    "prices": "TaiwanStockPrice",
    "institutional": "TaiwanStockInstitutionalInvestorsBuySell",
    "margin": "TaiwanStockMarginPurchaseShortSale",
}

INVESTOR_MAP = {
    # FinMind 的三大法人是長格式，一列一種法人。
    # 外資取「Foreign_Investor」窄定義，與既有 2023~2026 資料一致
    # （實測：既有 foreign_buy 等於 TWSE 的外陸資買進，不含外資自營商）。
    "Foreign_Investor": "foreign",
    "Investment_Trust": "trust",
    "Dealer_self": "dealer",
    "Dealer_Hedging": "dealer",
}
"""自營商合併自行買賣與避險，與既有 schema 一致"""

UNIVERSE_SIZE = 150
"""每季快照取成交金額前幾名。D2 要求 0050(50) + 0051(100) = 150 檔"""


@dataclass(frozen=True)
class ApiResult:
    ok: bool
    rows: list[dict]
    message: str = ""


# ══════════════════════════════════════════════════════════════
# API
# ══════════════════════════════════════════════════════════════


def api_token() -> str | None:
    """讀取 token；未設定或仍是佔位字串時回 None（走匿名層）"""
    raw = (os.environ.get(TOKEN_ENV) or "").strip()
    return None if raw.lower() in PLACEHOLDER_TOKENS else raw


def call(client: httpx.Client, params: dict, token: str | None) -> ApiResult:
    """
    呼叫 FinMind，遇到額度限制時退避重試。

    Returns:
        ApiResult；重試耗盡仍失敗時 `ok=False`，由呼叫端記錄後續可重跑
    """
    payload = dict(params)
    if token:
        payload["token"] = token

    for attempt in range(MAX_RETRIES):
        try:
            response = client.get(API, params=payload, timeout=90)
            if response.status_code == 200:
                body = response.json()
                if body.get("status") == 200:
                    return ApiResult(True, body.get("data") or [])
                message = str(body.get("msg", ""))[:200]
                # 額度／等級問題：等一下再試
                if "level" in message.lower() or "limit" in message.lower():
                    time.sleep(QUOTA_BACKOFF)
                    continue
                return ApiResult(False, [], message)
            if response.status_code in (402, 429, 500, 502, 503):
                time.sleep(QUOTA_BACKOFF)
                continue
            return ApiResult(False, [], f"HTTP {response.status_code}")
        except (httpx.HTTPError, ValueError) as exc:
            time.sleep(10.0 * (attempt + 1))
            if attempt == MAX_RETRIES - 1:
                return ApiResult(False, [], f"{type(exc).__name__}: {exc}")
    return ApiResult(False, [], "重試耗盡（多為額度限制）")


# ══════════════════════════════════════════════════════════════
# 資料庫
# ══════════════════════════════════════════════════════════════


def ensure_schema(con: sqlite3.Connection) -> None:
    """schema 與 qlib-tw-trader 的 data.db 一致，loader 換路徑即可用"""
    con.executescript("""
        PRAGMA journal_mode = WAL;

        CREATE TABLE IF NOT EXISTS stock_daily (
            id INTEGER PRIMARY KEY,
            stock_id VARCHAR(10) NOT NULL, date DATE NOT NULL,
            open NUMERIC(10,2) NOT NULL, high NUMERIC(10,2) NOT NULL,
            low NUMERIC(10,2) NOT NULL, close NUMERIC(10,2) NOT NULL,
            volume INTEGER NOT NULL, created_at DATETIME NOT NULL,
            CONSTRAINT uq_stock_daily UNIQUE (stock_id, date));
        CREATE INDEX IF NOT EXISTS ix_sd_id ON stock_daily (stock_id);
        CREATE INDEX IF NOT EXISTS ix_sd_date ON stock_daily (date);

        CREATE TABLE IF NOT EXISTS stock_daily_adj (
            id INTEGER PRIMARY KEY,
            stock_id VARCHAR(10) NOT NULL, date DATE NOT NULL,
            adj_close NUMERIC(10,2) NOT NULL, created_at DATETIME NOT NULL,
            CONSTRAINT uq_stock_daily_adj UNIQUE (stock_id, date));
        CREATE INDEX IF NOT EXISTS ix_adj_id ON stock_daily_adj (stock_id);

        CREATE TABLE IF NOT EXISTS stock_daily_institutional (
            id INTEGER PRIMARY KEY,
            stock_id VARCHAR(10) NOT NULL, date DATE NOT NULL,
            foreign_buy INTEGER NOT NULL, foreign_sell INTEGER NOT NULL,
            trust_buy INTEGER NOT NULL, trust_sell INTEGER NOT NULL,
            dealer_buy INTEGER NOT NULL, dealer_sell INTEGER NOT NULL,
            created_at DATETIME NOT NULL,
            CONSTRAINT uq_stock_daily_inst UNIQUE (stock_id, date));
        CREATE INDEX IF NOT EXISTS ix_inst_id ON stock_daily_institutional (stock_id);

        CREATE TABLE IF NOT EXISTS stock_daily_margin (
            id INTEGER PRIMARY KEY,
            stock_id VARCHAR(10) NOT NULL, date DATE NOT NULL,
            margin_buy INTEGER NOT NULL, margin_sell INTEGER NOT NULL,
            margin_balance INTEGER NOT NULL, short_buy INTEGER NOT NULL,
            short_sell INTEGER NOT NULL, short_balance INTEGER NOT NULL,
            created_at DATETIME NOT NULL,
            CONSTRAINT uq_stock_daily_margin UNIQUE (stock_id, date));
        CREATE INDEX IF NOT EXISTS ix_mg_id ON stock_daily_margin (stock_id);

        -- 證券主檔：含已下市，這是 survivorship 的關鍵
        CREATE TABLE IF NOT EXISTS stock_master (
            stock_id TEXT PRIMARY KEY,
            name TEXT NOT NULL DEFAULT '',
            industry TEXT NOT NULL DEFAULT '',
            market TEXT NOT NULL DEFAULT '',
            delisted_date TEXT
        );

        CREATE TABLE IF NOT EXISTS finmind_backfill_log (
            dataset TEXT NOT NULL, stock_id TEXT NOT NULL,
            status TEXT NOT NULL, rows INTEGER NOT NULL DEFAULT 0,
            message TEXT NOT NULL DEFAULT '', fetched_at TEXT NOT NULL,
            PRIMARY KEY (dataset, stock_id));

        CREATE TABLE IF NOT EXISTS stock_universe_history (
            as_of_date TEXT NOT NULL, stock_id TEXT NOT NULL,
            turnover REAL NOT NULL, rank INTEGER NOT NULL,
            PRIMARY KEY (as_of_date, stock_id));
        CREATE INDEX IF NOT EXISTS ix_uh_date ON stock_universe_history (as_of_date);
    """)
    con.commit()


def log(con: sqlite3.Connection, dataset: str, stock_id: str,
        status: str, rows: int, message: str = "") -> None:
    con.execute(
        "INSERT OR REPLACE INTO finmind_backfill_log VALUES (?, ?, ?, ?, ?, ?)",
        (dataset, stock_id, status, rows, message[:300],
         datetime.now().isoformat(timespec="seconds")),
    )


def done_ids(con: sqlite3.Connection, dataset: str) -> set[str]:
    rows = con.execute(
        "SELECT stock_id FROM finmind_backfill_log "
        "WHERE dataset = ? AND status IN ('ok', 'empty')",
        (dataset,),
    ).fetchall()
    return {r[0] for r in rows}


# ══════════════════════════════════════════════════════════════
# Phase：證券主檔
# ══════════════════════════════════════════════════════════════


def phase_master(con: sqlite3.Connection, client: httpx.Client,
                 token: str | None) -> None:
    """
    建立證券主檔：現存上市 + 2015 後下市。

    **下市的必須留下。** 它們在回測區間內是真實可交易的標的，
    漏掉就是 survivorship bias（禁令 2）。
    """
    print("[master] 取得證券清單 ...", flush=True)
    info = call(client, {"dataset": "TaiwanStockInfo"}, token)
    if not info.ok:
        print(f"  ✗ 失敗：{info.message}")
        return

    listed = {
        r["stock_id"]: (r.get("stock_name", ""), r.get("industry_category", ""),
                        r.get("type", ""))
        for r in info.rows
        if ORDINARY_CODE.match(str(r.get("stock_id", "")))
    }
    print(f"  現存 4 位數證券 {len(listed)} 檔")

    time.sleep(REQUEST_DELAY)
    delisted = call(client, {"dataset": "TaiwanStockDelisting"}, token)
    delist_map: dict[str, str] = {}
    if delisted.ok:
        for r in delisted.rows:
            sid = str(r.get("stock_id", ""))
            day = str(r.get("date", ""))
            if ORDINARY_CODE.match(sid) and day >= DEFAULT_START:
                delist_map[sid] = day
                listed.setdefault(sid, (r.get("stock_name", ""), "", "delisted"))
        print(f"  2015 後下市 {len(delist_map)} 檔（納入，避免 survivorship bias）")
    else:
        print(f"  ⚠️  下市清單取得失敗：{delisted.message}")

    con.executemany(
        "INSERT OR REPLACE INTO stock_master "
        "(stock_id, name, industry, market, delisted_date) VALUES (?, ?, ?, ?, ?)",
        [(sid, meta[0], meta[1], meta[2], delist_map.get(sid))
         for sid, meta in listed.items()],
    )
    con.commit()
    print(f"[master] 完成：主檔 {len(listed)} 檔", flush=True)


# ══════════════════════════════════════════════════════════════
# Phase：價格
# ══════════════════════════════════════════════════════════════


def store_prices(con: sqlite3.Connection, stock_id: str, rows: list[dict]) -> int:
    now = datetime.now().isoformat(timespec="seconds")
    payload = []
    for r in rows:
        try:
            o, h, low, c = (float(r["open"]), float(r["max"]),
                            float(r["min"]), float(r["close"]))
        except (KeyError, TypeError, ValueError):
            continue
        # 當天沒有成交時 FinMind 會給 0，補 0 會讓報酬率算出 −100%
        if min(o, h, low, c) <= 0:
            continue
        payload.append((stock_id, str(r["date"]), o, h, low, c,
                        int(r.get("Trading_Volume") or 0), now))
    if payload:
        con.executemany(
            "INSERT OR IGNORE INTO stock_daily "
            "(stock_id, date, open, high, low, close, volume, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)", payload)
    return len(payload)


def store_institutional(con: sqlite3.Connection, stock_id: str,
                        rows: list[dict]) -> int:
    """
    FinMind 的三大法人是**長格式**（一列一種法人），要先樞紐。

    自營商的 Dealer_self 與 Dealer_Hedging 兩列相加，與既有 schema 一致。
    """
    now = datetime.now().isoformat(timespec="seconds")
    daily: dict[str, dict[str, int]] = {}
    for r in rows:
        key = INVESTOR_MAP.get(str(r.get("name", "")))
        if key is None:
            continue
        bucket = daily.setdefault(str(r["date"]), {})
        bucket[f"{key}_buy"] = bucket.get(f"{key}_buy", 0) + int(r.get("buy") or 0)
        bucket[f"{key}_sell"] = bucket.get(f"{key}_sell", 0) + int(r.get("sell") or 0)

    payload = [
        (stock_id, day,
         v.get("foreign_buy", 0), v.get("foreign_sell", 0),
         v.get("trust_buy", 0), v.get("trust_sell", 0),
         v.get("dealer_buy", 0), v.get("dealer_sell", 0), now)
        for day, v in daily.items()
    ]
    if payload:
        con.executemany(
            "INSERT OR IGNORE INTO stock_daily_institutional "
            "(stock_id, date, foreign_buy, foreign_sell, trust_buy, trust_sell,"
            " dealer_buy, dealer_sell, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", payload)
    return len(payload)


def store_margin(con: sqlite3.Connection, stock_id: str, rows: list[dict]) -> int:
    now = datetime.now().isoformat(timespec="seconds")
    payload = []
    for r in rows:
        try:
            payload.append((
                stock_id, str(r["date"]),
                int(r.get("MarginPurchaseBuy") or 0),
                int(r.get("MarginPurchaseSell") or 0),
                int(r.get("MarginPurchaseTodayBalance") or 0),
                int(r.get("ShortSaleBuy") or 0),
                int(r.get("ShortSaleSell") or 0),
                int(r.get("ShortSaleTodayBalance") or 0),
                now,
            ))
        except (KeyError, TypeError, ValueError):
            continue
    if payload:
        con.executemany(
            "INSERT OR IGNORE INTO stock_daily_margin "
            "(stock_id, date, margin_buy, margin_sell, margin_balance,"
            " short_buy, short_sell, short_balance, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", payload)
    return len(payload)


STORE = {
    "prices": store_prices,
    "institutional": store_institutional,
    "margin": store_margin,
}


def fetch_dataset(con: sqlite3.Connection, client: httpx.Client, token: str | None,
                  kind: str, stock_ids: list[str], start: str, end: str,
                  delay: float) -> None:
    """逐檔抓取一個資料集的整段區間"""
    dataset = DATASETS[kind]
    pending = [s for s in stock_ids if s not in done_ids(con, dataset)]
    total = len(pending)
    print(f"\n[{kind}] 待處理 {total} 檔（共 {len(stock_ids)} 檔）", flush=True)
    if not total:
        return

    store = STORE[kind]
    counts = {"ok": 0, "empty": 0, "error": 0}
    written = 0
    started = time.time()

    for i, sid in enumerate(pending, start=1):
        result = call(client, {
            "dataset": dataset, "data_id": sid,
            "start_date": start, "end_date": end,
        }, token)

        if not result.ok:
            log(con, dataset, sid, "error", 0, result.message)
            counts["error"] += 1
        elif not result.rows:
            log(con, dataset, sid, "empty", 0, "區間內無資料")
            counts["empty"] += 1
        else:
            rows = store(con, sid, result.rows)
            written += rows
            log(con, dataset, sid, "ok", rows)
            counts["ok"] += 1

        if i % 20 == 0 or i == total:
            con.commit()
            elapsed = time.time() - started
            eta = elapsed / i * (total - i)
            print(f"  {i}/{total}  {sid}  寫入 {written:,} 列  "
                  f"ok={counts['ok']} 空={counts['empty']} 錯={counts['error']}  "
                  f"已花 {elapsed/60:.1f} 分  預估剩 {eta/60:.1f} 分", flush=True)

        time.sleep(delay)

    con.commit()
    print(f"[{kind}] 完成：{counts}，共寫入 {written:,} 列", flush=True)


# ══════════════════════════════════════════════════════════════
# Phase：歷史標的池
# ══════════════════════════════════════════════════════════════


def build_universe_history(con: sqlite3.Connection) -> None:
    """
    由價格資料建立逐季的標的池快照。

    排名依據是**該季的日均成交金額**（close × volume 的季平均）。

    這是流動性代理，不是市值——市值要股數，FinMind 免費層沒有。
    但它已經解掉 survivorship：排名只用「當季實際有交易的股票」，
    後來下市的在它還活著的季度依然入池。
    """
    print("\n[universe] 建立逐季標的池快照 ...", flush=True)
    con.execute("DELETE FROM stock_universe_history")

    quarters = [r[0] for r in con.execute(
        "SELECT DISTINCT substr(date, 1, 4) || '-Q' || "
        "  ((CAST(substr(date, 6, 2) AS INTEGER) + 2) / 3) "
        "FROM stock_daily ORDER BY 1"
    ).fetchall()]

    inserted = 0
    for quarter in quarters:
        year, q = quarter.split("-Q")
        start_month = (int(q) - 1) * 3 + 1
        lo = f"{year}-{start_month:02d}-01"
        hi = f"{year}-{start_month + 2:02d}-31"

        rows = con.execute(
            "SELECT stock_id, AVG(close * volume) AS turnover "
            "FROM stock_daily WHERE date >= ? AND date <= ? "
            "GROUP BY stock_id HAVING COUNT(*) >= 20 "
            "ORDER BY turnover DESC LIMIT ?",
            (lo, hi, UNIVERSE_SIZE),
        ).fetchall()

        con.executemany(
            "INSERT OR REPLACE INTO stock_universe_history VALUES (?, ?, ?, ?)",
            [(lo, sid, float(turnover), rank)
             for rank, (sid, turnover) in enumerate(rows, start=1)],
        )
        inserted += len(rows)

    con.commit()
    print(f"[universe] 完成：{len(quarters)} 個季度快照、{inserted} 筆", flush=True)


# ══════════════════════════════════════════════════════════════
# 狀態
# ══════════════════════════════════════════════════════════════


def print_status(con: sqlite3.Connection) -> None:
    print("=" * 80)
    print("FinMind 回補進度")
    print("=" * 80)
    rows = con.execute(
        "SELECT dataset, status, COUNT(*) FROM finmind_backfill_log "
        "GROUP BY dataset, status ORDER BY dataset, status"
    ).fetchall()
    if not rows:
        print("  尚未開始")
    for dataset, status, n in rows:
        print(f"  {dataset:<45}{status:<9}{n:>6} 檔")

    print()
    for table in ("stock_daily", "stock_daily_adj",
                  "stock_daily_institutional", "stock_daily_margin"):
        try:
            n, ids, lo, hi = con.execute(
                f"SELECT COUNT(*), COUNT(DISTINCT stock_id), MIN(date), MAX(date) "
                f"FROM {table}").fetchone()
            print(f"  {table:<28}{n:>10,} 列 {ids:>5} 檔  {lo} ~ {hi}")
        except sqlite3.Error as exc:
            print(f"  {table:<28}讀取失敗 {exc}")

    master, delisted = con.execute(
        "SELECT COUNT(*), SUM(CASE WHEN delisted_date IS NOT NULL THEN 1 ELSE 0 END) "
        "FROM stock_master").fetchone()
    print(f"  {'stock_master':<28}{master or 0:>10,} 檔"
          f"（其中已下市 {delisted or 0} 檔）")

    snaps = con.execute(
        "SELECT COUNT(DISTINCT as_of_date), MIN(as_of_date), MAX(as_of_date) "
        "FROM stock_universe_history").fetchone()
    if snaps[0]:
        print(f"  {'stock_universe_history':<28}{snaps[0]:>10} 個季度快照"
              f"  {snaps[1]} ~ {snaps[2]}")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="FinMind 歷史資料回補")
    parser.add_argument("--phase", nargs="*",
                        default=["master", "prices", "chips", "universe"],
                        choices=["master", "prices", "chips", "universe"])
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=date.today().isoformat())
    parser.add_argument("--delay", type=float, default=REQUEST_DELAY)
    parser.add_argument("--limit", type=int, default=0,
                        help="只處理前 N 檔（測試用）")
    parser.add_argument("--markets", nargs="*", default=["twse", "delisted"],
                        help="納入哪些市場。D2 的 0050+0051 都是上市，"
                             "所以預設排除上櫃（tpex）以節省請求")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--db", default=str(DB_PATH))
    args = parser.parse_args()

    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path, timeout=30)
    ensure_schema(con)

    if args.status:
        print_status(con)
        con.close()
        return

    token = api_token()
    print("=" * 80)
    print("FinMind 歷史資料回補")
    print("=" * 80)
    print(f"區間 {args.start} ~ {args.end}｜階段 {args.phase}｜間隔 {args.delay}s")
    print(f"資料庫 {db_path}")
    print(f"認證   {'已設定 token' if token else '匿名（免費層，額度較低）'}")

    with httpx.Client(headers={"User-Agent": USER_AGENT}) as client:
        if "master" in args.phase:
            phase_master(con, client, token)

        placeholders = ",".join("?" * len(args.markets))
        candidates = [r[0] for r in con.execute(
            f"SELECT stock_id FROM stock_master WHERE market IN ({placeholders}) "
            "ORDER BY stock_id", args.markets).fetchall()]
        print(f"\n候選標的 {len(candidates)} 檔（市場 {args.markets}）")
        if args.limit:
            candidates = candidates[: args.limit]

        if "prices" in args.phase:
            fetch_dataset(con, client, token, "prices", candidates,
                          args.start, args.end, args.delay)

        if "universe" in args.phase:
            build_universe_history(con)

        if "chips" in args.phase:
            # 籌碼只抓曾經進過標的池的股票——請求次數減半以上
            members = [r[0] for r in con.execute(
                "SELECT DISTINCT stock_id FROM stock_universe_history "
                "ORDER BY stock_id").fetchall()]
            if not members:
                print("\n[chips] 標的池為空，先跑 --phase universe")
            else:
                print(f"\n[chips] 標的池歷來成員 {len(members)} 檔")
                for kind in ("institutional", "margin"):
                    fetch_dataset(con, client, token, kind, members,
                                  args.start, args.end, args.delay)

    print()
    print_status(con)
    con.close()


if __name__ == "__main__":
    main()
