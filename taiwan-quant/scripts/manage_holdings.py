"""
持股帳本：每天跑這一支，它告訴你今天要做什麼

## 用法

    # 每日報告（今天要賣什麼、手上有什麼、還有幾個位子）
    python scripts/manage_holdings.py

    # 記一筆買進（出場日由交易日曆自動推算）
    python scripts/manage_holdings.py --buy 2884 --price 46.5 --shares 400

    # 記一筆賣出
    python scripts/manage_holdings.py --sell 2884 --price 50.1

## 「今天要賣」是機械的，不是預測

`planned_exit_date` = 買進日 + `HOLDING_DAYS` 個交易日。

⚠️ **這不是「現在是高點」。** 這個專案試過動態出場：移動停損在 1.56 年
多頭樣本（915 個交易日）上是 +372.79%，資料拉到 2,850 個交易日之後是
**−68.91%**。那個正報酬是環境的產物，不是抓頂能力。

帳本只做記帳與到期提醒。**要我預測高點我做不到，而且試過的人賠了。**

## 位子上限怎麼來的

40 萬 ÷ 每檔至少 2 萬 = 20 個位子。低於 2 萬時零股成本（0.3~0.4%）
會把 40 日 +1.7% 的期望值吃掉大半。

事件每天 0~3 檔而你每天只能開 0.5 個新倉（20 ÷ 40），
**所以大部分日子沒有位子可開，那是正常的。**
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from taiwan_quant.config.costs import DEFAULT as DEFAULT_COST  # noqa: E402
from taiwan_quant.config.costs import resolve_tier  # noqa: E402
from taiwan_quant.data.calendar import due_trading_date  # noqa: E402
from taiwan_quant.data.etf_universe import is_etf  # noqa: E402
from taiwan_quant.data.loader import (  # noqa: E402
    DEFAULT_UNIVERSE_BASIS,
    HISTORY_DB_PATH,
    load_universe_at,
)
from taiwan_quant.holdings import (  # noqa: E402
    Holding,
    HoldingsError,
    closed_holdings,
    due_on,
    open_holdings,
    record_purchase,
    record_sale,
)

CAPITAL = 400_000.0
HOLDING_DAYS = 40
MIN_POSITION = 20_000.0
"""每檔最低金額。低於此零股成本會吃掉期望值"""

MAX_SLOTS = int(CAPITAL // MIN_POSITION)
LARGE_TIER_SIZE = 50


def _calendar(db_path: Path) -> list[date]:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        return [
            date.fromisoformat(row[0])
            for row in con.execute(
                "SELECT DISTINCT date FROM stock_daily ORDER BY date"
            )
        ]
    finally:
        con.close()


def _last_close(db_path: Path, stock_id: str) -> tuple[date, float] | None:
    """最後一筆收盤價。用未還原價——那是你帳戶上看到的數字"""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        row = con.execute(
            "SELECT date, close FROM stock_daily WHERE stock_id = ? "
            "ORDER BY date DESC LIMIT 1",
            (stock_id,),
        ).fetchone()
    finally:
        con.close()
    if row is None:
        return None
    return date.fromisoformat(row[0]), float(row[1])


def _cost_for(db_path: Path, stock_id: str, price: float, amount: float) -> float:
    """逐檔來回成本（禁令 8）。買進當下固定，事後不重算"""
    try:
        large = stock_id in set(
            load_universe_at(
                date.today(), db_path=db_path, limit=LARGE_TIER_SIZE,
                basis=DEFAULT_UNIVERSE_BASIS,
            ).stock_ids
        )
    except Exception:  # noqa: BLE001
        large = False
    tier = resolve_tier(
        actual_price=price, adjusted_price=price,
        amount=amount, large=large, is_etf=is_etf(stock_id),
    )
    return DEFAULT_COST.round_trip_rate(tier)


def do_buy(db_path: Path, args: argparse.Namespace) -> int:
    buy_date = date.fromisoformat(args.date) if args.date else date.today()
    amount = args.price * args.shares
    if amount < MIN_POSITION:
        print(f"⚠️ 這筆只有 {amount:,.0f} 元，低於 {MIN_POSITION:,.0f} 的下限。")
        print("   零股成本（0.3~0.4%）會吃掉大半期望值。仍然照實記錄。")

    exit_date = due_trading_date(_calendar(db_path), buy_date, HOLDING_DAYS)
    holding = Holding(
        stock_id=args.buy,
        buy_date=buy_date.isoformat(),
        buy_price=args.price,
        shares=args.shares,
        strategy_version=args.source,
        planned_exit_date=exit_date.isoformat(),
        round_trip_cost=_cost_for(db_path, args.buy, args.price, amount),
        opened_at=datetime.now().astimezone().isoformat(),
        note=args.note,
    )
    record_purchase(db_path, holding)
    print(f"已記錄買進：{holding.stock_id} {holding.shares} 股 @ "
          f"{holding.buy_price:.2f}（{amount:,.0f} 元）")
    print(f"  來源 {holding.strategy_version}"
          f"｜來回成本 {holding.round_trip_cost:.3%}")
    print(f"  計畫出場日 {exit_date}（買進日 +{HOLDING_DAYS} 個交易日）")
    print(f"  ⚠️ 那是機械到期日，不是預測的高點。")
    return 0


def do_sell(db_path: Path, args: argparse.Namespace) -> int:
    sell_date = date.fromisoformat(args.date) if args.date else date.today()
    candidates = [h for h in open_holdings(db_path) if h.stock_id == args.sell]
    if not candidates:
        raise HoldingsError(f"{args.sell} 沒有未平倉的持股")
    if args.buy_date:
        buy_date = args.buy_date
    elif len(candidates) == 1:
        buy_date = candidates[0].buy_date
    else:
        raise HoldingsError(
            f"{args.sell} 有 {len(candidates)} 批未平倉"
            f"（{', '.join(h.buy_date for h in candidates)}），"
            "請用 --buy-date 指定哪一批"
        )

    closed = record_sale(
        db_path, stock_id=args.sell, buy_date=buy_date,
        sell_date=sell_date.isoformat(), sell_price=args.price,
    )
    print(f"已記錄賣出：{args.sell} @ {args.price:.2f}")
    print(f"  毛報酬 {closed.gross_return:+.3%}"
          f"｜成本 {closed.holding.round_trip_cost:.3%}"
          f"｜淨報酬 {closed.net_return:+.3%}")
    print(f"  損益 {closed.profit:+,.0f} 元")
    if closed.gross_return > 0 > closed.net_return:
        print("  ⚠️ 毛賺但淨賠——成本地板吃掉了。")
    return 0


def do_report(db_path: Path, today: date) -> int:
    open_rows = open_holdings(db_path)
    due_rows = due_on(db_path, today)

    print(f"=== {today} 持股報告 ===\n")

    if due_rows:
        print(f"【今天要賣】{len(due_rows)} 檔（機械到期，不是預測高點）")
        for h in due_rows:
            quote = _last_close(db_path, h.stock_id)
            overdue = (today - date.fromisoformat(h.planned_exit_date)).days
            mark = f"，已過期 {overdue} 天" if overdue > 0 else ""
            line = (f"  {h.stock_id}  {h.shares} 股 @ {h.buy_price:.2f}"
                    f"｜出場日 {h.planned_exit_date}{mark}")
            if quote:
                as_of, price = quote
                gross = price / h.buy_price - 1
                net = gross - h.round_trip_cost
                line += (f"\n      最後收盤 {price:.2f}（{as_of}）"
                         f"｜毛 {gross:+.2%}｜淨 {net:+.2%}"
                         f"｜{h.buy_price * h.shares * net:+,.0f} 元")
            print(line)
        print()
    else:
        print("【今天要賣】無\n")

    used = len(open_rows)
    print(f"【持倉】{used} 檔，位子 {used}/{MAX_SLOTS}"
          f"（空 {MAX_SLOTS - used}）")
    if open_rows:
        total_cost = sum(h.amount for h in open_rows)
        unrealised = 0.0
        for h in open_rows:
            quote = _last_close(db_path, h.stock_id)
            if not quote:
                continue
            net = quote[1] / h.buy_price - 1 - h.round_trip_cost
            unrealised += h.amount * net
        print(f"  投入 {total_cost:,.0f} 元"
              f"｜未實現（已扣成本）{unrealised:+,.0f} 元")
        print(f"  現金約 {CAPITAL - total_cost:,.0f} 元")
    print()

    closed_rows = closed_holdings(db_path)
    if closed_rows:
        profit = sum(c.profit for c in closed_rows)
        wins = sum(1 for c in closed_rows if c.net_return > 0)
        print(f"【已實現】{len(closed_rows)} 筆"
              f"｜淨損益 {profit:+,.0f} 元"
              f"｜勝率 {wins}/{len(closed_rows)}")
        print(f"  ⚠️ {len(closed_rows)} 筆還不足以判斷策略好壞——"
              f"前推驗證第一階段需要 173 個事件日。")
    else:
        print("【已實現】尚無平倉紀錄")

    if MAX_SLOTS - used > 0:
        print(f"\n→ 有 {MAX_SLOTS - used} 個空位。今天的訊號用 "
              f"record_forward_events.py 看。")
    else:
        print(f"\n→ 位子滿了。今天有訊號也開不了倉，等到期。")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(HISTORY_DB_PATH))
    parser.add_argument("--buy", metavar="代號")
    parser.add_argument("--sell", metavar="代號")
    parser.add_argument("--price", type=float)
    parser.add_argument("--shares", type=int)
    parser.add_argument("--date", help="成交日，預設今天")
    parser.add_argument("--buy-date", help="賣出時指定是哪一批")
    parser.add_argument("--source", default="margin_spike_h40@v1")
    parser.add_argument("--note", default="")
    args = parser.parse_args()

    db_path = Path(args.db)
    if args.buy and args.sell:
        parser.error("--buy 與 --sell 不可同時使用")
    if args.buy:
        if args.price is None or args.shares is None:
            parser.error("--buy 需要 --price 與 --shares")
        return do_buy(db_path, args)
    if args.sell:
        if args.price is None:
            parser.error("--sell 需要 --price")
        return do_sell(db_path, args)
    return do_report(db_path, date.today())


if __name__ == "__main__":
    raise SystemExit(main())
