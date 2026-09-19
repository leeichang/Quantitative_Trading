"""
單檔補歷史：用 TWSE 月報端點，請求數比逐日走訪少兩個數量級

## 為什麼不用 backfill_twse_history.py

那支走的是「一天拿全市場」（`MI_INDEX?type=ALLBUT0999`）。要補 2015 起
的完整歷史就是 3,052 次請求，而 **TWSE 在約 35 次之後就開始限流**，
實測 2026-09-19 從 2015-03-04 起連續回 `HTTP 重試耗盡`，接著整個 IP
被擋：

```
HTTP 307
因為安全性考量，您所執行的頁面無法呈現。
```

補幾檔新標的不需要全市場掃描。月報端點一次拿一個月：

```
逐日全市場   00712 從 2017-10 起 → 2,180 次請求
單檔月報     同上               →   108 次請求
```

## 限流

`DELAY` 預設 3.5 秒。TWSE 沒有公開的正式配額，實測逐日走訪在 0.6 秒
間隔下會被擋，所以這裡取一個明顯保守的值。**被擋之後要等，不是加重試**
——重試只會延長封鎖。

`--delay` 可以調大，不建議調小。

## 只補缺的月份

已經有資料的月份會跳過（依 `stock_daily` 實際列數判斷，不另建 log 表）。
所以中斷後重跑是安全的，也不會重複打同一個月。

## 用法

    python scripts/backfill_symbol_history.py --symbols 00712 00878 006208
    python scripts/backfill_symbol_history.py --symbols 00712 --start 2017-10 --delay 5
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from datetime import date, datetime
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from taiwan_quant.data.loader import HISTORY_DB_PATH  # noqa: E402
from taiwan_quant.data.twse_history import (  # noqa: E402
    NO_TRADE_MARKERS,
    is_ordinary_security,
    parse_number,
)

ENDPOINT = (
    "https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY"
    "?date={ym}01&stockNo={symbol}&response=json"
)
DELAY = 3.5
"""請求間隔（秒）。實測 0.6 秒會被 TWSE 擋，這裡取明顯保守的值"""

HEADERS = {"User-Agent": "Mozilla/5.0"}
TIMEOUT = 30.0
DEFAULT_START = "2015-01"


class BackfillError(RuntimeError):
    """輸入不合法，或 TWSE 回了非預期內容。"""


def months(start: str, end: str) -> list[str]:
    """`YYYY-MM` 區間展開成 `YYYYMM` 清單（含頭尾）。"""
    try:
        first = datetime.strptime(start, "%Y-%m").date().replace(day=1)
        last = datetime.strptime(end, "%Y-%m").date().replace(day=1)
    except ValueError as exc:
        raise BackfillError(f"月份格式必須是 YYYY-MM：{start} / {end}") from exc
    if first > last:
        raise BackfillError(f"起始月 {start} 不可晚於結束月 {end}")

    out: list[str] = []
    cursor = first
    while cursor <= last:
        out.append(cursor.strftime("%Y%m"))
        cursor = (cursor.replace(day=28) + __import__("datetime").timedelta(days=7)
                  ).replace(day=1)
    return out


def existing_months(con: sqlite3.Connection, symbol: str) -> set[str]:
    """已有資料的月份。用實際列數判斷，不另建 log 表"""
    return {
        row[0].replace("-", "")
        for row in con.execute(
            "SELECT DISTINCT substr(date, 1, 7) FROM stock_daily WHERE stock_id = ?",
            (symbol,),
        )
    }


def parse_month(payload: dict, symbol: str) -> list[tuple[str, float, float, float, float, int]]:
    """
    解析月報。

    ⚠️ 民國年：`112/10/02` → `2023-10-02`。**不可以用字串前四碼當西元年**。

    OHLC 任一為 `--` 的列剔除——那是當天沒成交，補 0 會讓報酬算出 −100%。
    """
    if str(payload.get("stat")) != "OK":
        raise BackfillError(f"{symbol}：stat = {payload.get('stat')!r}")

    rows: list[tuple[str, float, float, float, float, int]] = []
    for row in payload.get("data") or []:
        if len(row) < 7:
            continue
        roc = str(row[0]).strip()
        parts = roc.split("/")
        if len(parts) != 3:
            continue
        try:
            iso = f"{int(parts[0]) + 1911:04d}-{int(parts[1]):02d}-{int(parts[2]):02d}"
        except ValueError:
            continue

        if any(str(row[i]).strip().lower() in NO_TRADE_MARKERS for i in (3, 4, 5, 6)):
            continue
        prices = [parse_number(row[i]) for i in (3, 4, 5, 6)]
        if any(p is None or p <= 0 for p in prices):
            continue
        volume = parse_number(row[1])
        rows.append((iso, prices[0], prices[1], prices[2], prices[3],
                     int(volume) if volume is not None else 0))
    return rows


def store(con: sqlite3.Connection, symbol: str, rows: list) -> int:
    """`INSERT OR IGNORE`——既有列永不覆寫。"""
    now = datetime.now().isoformat(timespec="seconds")
    before = con.total_changes
    con.executemany(
        "INSERT OR IGNORE INTO stock_daily "
        "(stock_id, date, open, high, low, close, volume, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [(symbol, d, o, h, low, c, v, now) for d, o, h, low, c, v in rows],
    )
    con.commit()
    return con.total_changes - before


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--start", default=DEFAULT_START, help="YYYY-MM")
    parser.add_argument(
        "--end", default=date.today().strftime("%Y-%m"), help="YYYY-MM"
    )
    parser.add_argument("--delay", type=float, default=DELAY)
    parser.add_argument("--db", default=str(HISTORY_DB_PATH))
    args = parser.parse_args()

    if args.delay < 1.0:
        parser.error("--delay 低於 1 秒會被 TWSE 擋，實測 0.6 秒觸發封鎖")
    rejected = [s for s in args.symbols if not is_ordinary_security(s)]
    if rejected:
        parser.error(
            f"這些代號不是一般證券（槓桿／反向／權證／特別股）：{rejected}"
        )

    wanted = months(args.start, args.end)
    con = sqlite3.connect(args.db)
    try:
        for symbol in args.symbols:
            have = existing_months(con, symbol)
            todo = [m for m in wanted if m not in have]
            print(f"\n=== {symbol} ===")
            print(f"  區間 {args.start} ~ {args.end}｜共 {len(wanted)} 月"
                  f"｜已有 {len(wanted) - len(todo)}｜待補 {len(todo)}")
            if not todo:
                continue

            added = blocked = empty = 0
            with httpx.Client(headers=HEADERS, timeout=TIMEOUT) as client:
                for index, ym in enumerate(todo, start=1):
                    url = ENDPOINT.format(ym=ym, symbol=symbol)
                    try:
                        response = client.get(url)
                        if response.status_code != 200:
                            blocked += 1
                            print(f"  {ym} HTTP {response.status_code}"
                                  f"——很可能是限流，停止此標的")
                            break
                        rows = parse_month(response.json(), symbol)
                    except BackfillError as exc:
                        empty += 1
                        print(f"  {ym} {exc}")
                        time.sleep(args.delay)
                        continue
                    except Exception as exc:  # noqa: BLE001
                        blocked += 1
                        print(f"  {ym} 失敗：{type(exc).__name__}——停止此標的")
                        break

                    n = store(con, symbol, rows)
                    added += n
                    if index % 12 == 0 or index == len(todo):
                        print(f"  {ym} 累計新增 {added:,} 列"
                              f"（{index}/{len(todo)} 月）", flush=True)
                    time.sleep(args.delay)

            total = con.execute(
                "SELECT COUNT(*), MIN(date), MAX(date) FROM stock_daily "
                "WHERE stock_id = ?", (symbol,),
            ).fetchone()
            print(f"  新增 {added:,} 列｜空月 {empty}｜中斷 {blocked}")
            print(f"  現況 {total[0]:,} 列  {total[1]} ~ {total[2]}")
            if blocked:
                print("  ⚠️ 被中斷。**等一段時間再跑**，不要調小 --delay。")
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
