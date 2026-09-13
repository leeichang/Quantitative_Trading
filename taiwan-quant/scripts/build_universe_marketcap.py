#!/usr/bin/env python3
"""
歷史標的池 — 改用市值排名

## 為什麼要改

第一版用**日均成交金額**排名，因為 FinMind 免費層沒有發行股數。
拿今天真實的 0050+0051 名單去驗證，只抓到 **109 / 150 = 72.7%**。

漏掉的是同一類股票：

```
1102 亞泥、1402 遠東新、1503 士電、2002 中鋼、2105 正新、2207 和泰車
```

**大市值但低週轉的傳產。** 成交金額排名系統性偏向熱門股，漏掉穩定的
權值股——而 0050 / 0051 是**市值加權指數**，正好相反。

## 資料來源

TWSE `MI_QFIIS`（外資及陸資持股統計）的「發行股數」欄位，支援任意
歷史日期。只要每季一次，47 個請求就能覆蓋 11 年。

```
市值 = 當日收盤價 × 發行股數
```

解析沿用 `taiwan_quant/data/market_cap.py`（已有測試），不重寫。

## 兩種排名並存

`stock_universe_history` 加上 `basis` 欄位（`turnover` / `market_cap`），
兩份快照同時保留。理由：

1. 可以直接比較兩種代理對真實成分股的命中率
2. 先前的驗證結果是用 turnover 版跑的，刪掉就無法重現

`loader.load_universe_at()` 預設取 `market_cap`。

## 仍然是代理，不是真名單

市值排名接近 0050 / 0051 的編製方式，但不等同——真實指數還有流動性
門檻、產業分散、自由流通量調整等規則，而且成分股審核每季由指數公司
人工核定。

**真實的歷史成分股在免費來源取不到**（元大只公布現況，指數公司的歷史
審核公告沒有 API）。這條限制留著。

用法：
    .venv/bin/python scripts/build_universe_marketcap.py
    .venv/bin/python scripts/build_universe_marketcap.py --validate
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datetime import timedelta  # noqa: E402

from taiwan_quant.data.market_cap import (  # noqa: E402
    MarketCapError,
    fetch_issued_shares,
)

SHARES_LOOKBACK_DAYS = 10
"""
發行股數往前找幾天。

快照日是季初（1/1、4/1、7/1、10/1），**常常不是交易日**——實測 47 期
裡有 23 期落在元旦或週末。那些日期 MI_QFIIS 回 `stat=OK` 但 `data=[]`，
解析層會拋 MarketCapError。

往前找最近一個有資料的交易日，與 `closes_on_or_before` 的作法一致。
**只往前找，不往後**——往後就是 look-ahead。
"""


def issued_shares_on_or_before(
    day: date, lookback: int = SHARES_LOOKBACK_DAYS
) -> tuple[dict[str, int], date] | None:
    """
    取不晚於 `day` 的最近一份發行股數。

    Returns:
        (發行股數, 實際取到的日期)；`lookback` 天內都找不到時回 None

    **只往前找。** 往後找等於在季初就知道之後才公布的股數。
    """
    for back in range(lookback + 1):
        probe = day - timedelta(days=back)
        try:
            return fetch_issued_shares(probe), probe
        except MarketCapError:
            time.sleep(0.4)
    return None

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "history.db"

UNIVERSE_SIZE = 150
"""D2 要求 0050(50) + 0051(100) = 150 檔"""

REQUEST_DELAY = 2.0
"""TWSE 的 WAF 對連續請求敏感。每季一次只有 47 個請求，慢一點無妨"""

BASIS_MARKET_CAP = "market_cap"
BASIS_TURNOVER = "turnover"


def ensure_schema(con: sqlite3.Connection) -> None:
    """
    加上 `basis` 欄位，讓兩種排名並存。

    舊資料（沒有 basis 欄位）一律標記為 turnover——那是它實際的算法。
    """
    columns = {r[1] for r in con.execute("PRAGMA table_info(stock_universe_history)")}
    if "basis" not in columns:
        con.execute(
            "ALTER TABLE stock_universe_history "
            f"ADD COLUMN basis TEXT NOT NULL DEFAULT '{BASIS_TURNOVER}'"
        )
    if "market_cap" not in columns:
        con.execute(
            "ALTER TABLE stock_universe_history "
            "ADD COLUMN market_cap REAL NOT NULL DEFAULT 0"
        )

    # 原主鍵是 (as_of_date, stock_id)，兩種 basis 會撞鍵。改建新表。
    con.executescript(f"""
        CREATE TABLE IF NOT EXISTS universe_history (
            as_of_date TEXT NOT NULL,
            basis      TEXT NOT NULL,
            stock_id   TEXT NOT NULL,
            metric     REAL NOT NULL,
            rank       INTEGER NOT NULL,
            PRIMARY KEY (as_of_date, basis, stock_id)
        );
        CREATE INDEX IF NOT EXISTS ix_uh2_date_basis
            ON universe_history (as_of_date, basis);

        INSERT OR IGNORE INTO universe_history (as_of_date, basis, stock_id, metric, rank)
        SELECT as_of_date, '{BASIS_TURNOVER}', stock_id, turnover, rank
        FROM stock_universe_history;
    """)
    con.commit()


def snapshot_dates(con: sqlite3.Connection) -> list[str]:
    """沿用既有的季度快照日期，讓兩種排名可以逐期對照"""
    return [r[0] for r in con.execute(
        "SELECT DISTINCT as_of_date FROM stock_universe_history ORDER BY as_of_date")]


def closes_on_or_before(con: sqlite3.Connection, day: str) -> dict[str, float]:
    """
    取不晚於 `day` 的最新收盤價。

    快照日是季初，常常不是交易日；取之後的第一筆會是 look-ahead。
    """
    rows = con.execute(
        """
        SELECT stock_id, close FROM stock_daily
        WHERE date = (
            SELECT MAX(date) FROM stock_daily AS d2
            WHERE d2.stock_id = stock_daily.stock_id AND d2.date <= ?
        )
        """,
        (day,),
    ).fetchall()
    return {sid: float(c) for sid, c in rows if c and float(c) > 0}


def build(con: sqlite3.Connection, delay: float) -> None:
    dates = snapshot_dates(con)
    print(f"季度快照 {len(dates)} 期：{dates[0]} ~ {dates[-1]}")
    print()

    done = {r[0] for r in con.execute(
        "SELECT DISTINCT as_of_date FROM universe_history WHERE basis = ?",
        (BASIS_MARKET_CAP,))}
    pending = [d for d in dates if d not in done]
    if not pending:
        print("市值快照已全部建立")
        return

    started = time.time()
    for i, day in enumerate(pending, start=1):
        found = issued_shares_on_or_before(date.fromisoformat(day))
        if found is None:
            print(f"  {day}  ✗ 前 {SHARES_LOOKBACK_DAYS} 天內都取不到發行股數",
                  flush=True)
            time.sleep(delay)
            continue
        shares, shares_date = found
        if shares_date.isoformat() != day:
            print(f"  {day}  ⓘ 非交易日，改用 {shares_date} 的發行股數", flush=True)

        closes = closes_on_or_before(con, day)
        caps = {
            sid: closes[sid] * n
            for sid, n in shares.items()
            if sid in closes
        }
        if not caps:
            print(f"  {day}  ✗ 無可用市值（發行股數 {len(shares)} 檔、"
                  f"收盤價 {len(closes)} 檔）", flush=True)
            time.sleep(delay)
            continue

        top = sorted(caps.items(), key=lambda kv: -kv[1])[:UNIVERSE_SIZE]
        con.executemany(
            "INSERT OR REPLACE INTO universe_history "
            "(as_of_date, basis, stock_id, metric, rank) VALUES (?, ?, ?, ?, ?)",
            [(day, BASIS_MARKET_CAP, sid, cap, rank)
             for rank, (sid, cap) in enumerate(top, start=1)],
        )
        con.commit()

        if i % 10 == 0 or i == len(pending):
            elapsed = time.time() - started
            eta = elapsed / i * (len(pending) - i)
            print(f"  {i}/{len(pending)}  {day}  前 150 檔已寫入"
                  f"（候選 {len(caps)} 檔）  已花 {elapsed/60:.1f} 分"
                  f"  剩 {eta/60:.1f} 分", flush=True)

        time.sleep(delay)


def validate(con: sqlite3.Connection) -> None:
    """
    用今天真實的 0050 + 0051 名單驗證兩種代理的命中率。

    這是唯一能檢驗代理品質的方式——歷史名單取不到，但現況可以。
    """
    from taiwan_quant.data.constituents import fetch_universe_constituents

    print("=" * 74)
    print("代理品質驗證（對照今日真實 0050 + 0051 成分股）")
    print("=" * 74)
    try:
        snapshots = fetch_universe_constituents()
    except Exception as exc:  # noqa: BLE001 — 網路失敗不該讓整個腳本掛掉
        print(f"取得真實名單失敗：{type(exc).__name__} {exc}")
        return

    real: set[str] = set()
    for snap in snapshots.values():
        real |= set(snap.stock_ids)
    print(f"真實名單 {len(real)} 檔")
    print()

    latest = con.execute("SELECT MAX(as_of_date) FROM universe_history").fetchone()[0]
    print(f"{'排名依據':<14}{'檔數':>6}{'命中':>7}{'命中率':>9}   漏掉的前幾檔")
    print("-" * 74)
    for basis, label in ((BASIS_MARKET_CAP, "市值"), (BASIS_TURNOVER, "成交金額")):
        proxy = {r[0] for r in con.execute(
            "SELECT stock_id FROM universe_history WHERE as_of_date = ? AND basis = ?",
            (latest, basis))}
        if not proxy:
            print(f"{label:<14}（尚未建立）")
            continue
        hit = real & proxy
        missed = sorted(real - proxy)
        print(f"{label:<14}{len(proxy):>6}{len(hit):>7}"
              f"{len(hit)/len(real)*100:>8.1f}%   {missed[:6]}")
    print("-" * 74)
    print(f"快照日 {latest}")
    print("=" * 74)


def main() -> None:
    parser = argparse.ArgumentParser(description="建立市值排名的歷史標的池")
    parser.add_argument("--delay", type=float, default=REQUEST_DELAY)
    parser.add_argument("--validate", action="store_true",
                        help="只驗證代理品質，不抓取")
    parser.add_argument("--db", default=str(DB_PATH))
    args = parser.parse_args()

    con = sqlite3.connect(Path(args.db), timeout=30)
    ensure_schema(con)

    if args.validate:
        validate(con)
        con.close()
        return

    print("=" * 74)
    print("歷史標的池 — 市值排名")
    print("=" * 74)
    print(f"資料庫 {args.db}｜每期取前 {UNIVERSE_SIZE} 檔｜間隔 {args.delay}s")
    print()

    build(con, args.delay)

    print()
    for basis in (BASIS_MARKET_CAP, BASIS_TURNOVER):
        n, dates = con.execute(
            "SELECT COUNT(*), COUNT(DISTINCT as_of_date) FROM universe_history "
            "WHERE basis = ?", (basis,)).fetchone()
        print(f"  {basis:<12}{dates:>4} 期  {n:>7,} 筆")

    print()
    validate(con)
    con.close()


if __name__ == "__main__":
    main()
