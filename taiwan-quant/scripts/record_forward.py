#!/usr/bin/env python3
"""
前推預測記錄與結算（動能突破 N=10）

## 為什麼這是唯一會收斂的證據

開發集可以反覆看，所以它證明不了什麼。OOS 區間看一次就髒了
（2024-01 ~ 2026-08 已經在 `momentum_top10_h40@oos-2026-09-16` 用掉）。

**只有前推預測是往前長的**：時間走一天就多一天證據，而且從來沒有被
看過。它不消耗任何區間。

```
每 40 個交易日一筆決策 × 10 檔
1 年    6 期    仍然太少
3 年   19 期    與 2024-2026 那次 OOS 同量級
5 年   32 期    開始有意義
```

## 為什麼從「校準器 + 移動停損」換成「原始分數 Top 10」

實測（`2026-09-16_漲停預測力與成本結構.md`）：校準器（12 等級）→ 門檻
→ 槽位佇列三層在毀訊號。

```
經過三層        60 日 CPCV 中位數 +5.06%｜5% 分位 −27.86%
直接用原始分數   40 日 CPCV 中位數 +94.39%｜5% 分位 +13.68%｜15/15 路徑為正
```

樣本外（2024-01 ~ 2026-08，已用掉）：累積淨 +138.83%、Sharpe 1.08，
超過隨機 10 檔的 95% 分位（+120.97%），但**輸 0050 買進持有
（+242.15%、Sharpe 1.90）**。

⚠️ **所以這不是「已證明能賺錢的策略」。** 它是目前證據最完整的一組，
記進帳本是為了讓時間累積乾淨樣本，不是背書。

## 參數固定（禁令 7、8）

不接受策略參數的命令列覆寫。參數變了就換 `STRATEGY_VERSION`——帳本的
唯一鍵用 `strategy_version`，兩個版本會並存而不是互相覆蓋。

用法：
    .venv/bin/python scripts/record_forward.py            # 產生並記錄
    .venv/bin/python scripts/record_forward.py --settle    # 回填已到期的
    .venv/bin/python scripts/record_forward.py --dry-run   # 只看不寫
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from taiwan_quant.config.costs import (  # noqa: E402
    DEFAULT as DEFAULT_COST,
    resolve_tier,
)
from taiwan_quant.data.calendar import (  # noqa: E402
    due_trading_date,
    load_trading_calendar,
)
from taiwan_quant.data.dataset import build_dataset  # noqa: E402
from taiwan_quant.data.etf_universe import is_etf, merge_etf_candidates  # noqa: E402
from taiwan_quant.data.loader import (  # noqa: E402
    FROZEN_DATA_START,
    HISTORY_DB_PATH,
    load_chips,
    load_prices,
)
from taiwan_quant.forward_predictions import (  # noqa: E402
    ForwardPrediction,
    list_unsettled_predictions,
    record_forward_predictions,
    settle_forward_prediction,
)
from taiwan_quant.ranking.tie_break import (  # noqa: E402
    DEFAULT_TIE_SEED,
    deterministic_jitter,
)

import scripts.validate_oos_trailing as V  # noqa: E402

# ══════════════════════════════════════════════════════════════
# 固定參數（禁令 7、8）——參數變了就換 STRATEGY_VERSION
# ══════════════════════════════════════════════════════════════

FAMILY = "動能突破"
HOLDING_DAYS = 40
N_POSITIONS = 10
CAPITAL = 400_000.0
UNIVERSE_SIZE = 150
UNIVERSE_BASIS = "market_cap"
MIN_CANDIDATES = 30
INCLUDE_ETFS = False
"""
ETF 暫不納入候選。

`--include-etfs` 的效果在 2024-01 ~ 2026-08 無法驗證（那個區間已用掉），
所以帳本先記不含 ETF 的版本。要記含 ETF 的版本就換 `STRATEGY_VERSION`，
兩者會在帳本裡並存、在同一段未來上被比較——那才是帳本的價值。
"""

STRATEGY_VERSION = "momentum_top10_h40@v1"

PARAMS = {
    "family": FAMILY,
    "holding_days": HOLDING_DAYS,
    "n_positions": N_POSITIONS,
    "capital": CAPITAL,
    "universe_size": UNIVERSE_SIZE,
    "universe_basis": UNIVERSE_BASIS,
    "min_candidates": MIN_CANDIDATES,
    "include_etfs": INCLUDE_ETFS,
    "cost_model": "taiwan_quant.config.costs.DEFAULT",
    "cost_per_name": "resolve_tier(price, capital/n_positions, is_etf)",
    "tie_seed": DEFAULT_TIE_SEED,
    "ranking": "raw score, no calibrator/threshold/slot-queue",
    "entry": "T+1 open",
    "exit": f"T+{HOLDING_DAYS} close",
}
PARAMS_JSON = json.dumps(PARAMS, ensure_ascii=False, sort_keys=True)


def latest_price_date(db_path: Path) -> date:
    """查詢資料截止日；只讀 metadata，不繞過價格凍結守門"""
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as con:
        value = con.execute("SELECT MAX(date) FROM stock_daily").fetchone()[0]
    if value is None:
        raise RuntimeError("stock_daily 無資料")
    return date.fromisoformat(value)


def generate(db_path: Path, as_of: date) -> list[ForwardPrediction]:
    """
    產生 `as_of` 當日的前推預測。

    Args:
        db_path: SQLite 檔案路徑
        as_of: 決策日（資料截止日）

    Returns:
        最多 `N_POSITIONS` 筆預測

    ⚠️ `entry_price` 用的是 **`as_of` 的收盤價**，因為 T+1 開盤價此刻還
    不存在。實際成交價會不同——結算時用真正的 T+1 開盤價重算報酬。
    """
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    members = [
        row[0]
        for row in con.execute(
            "SELECT DISTINCT stock_id FROM universe_history WHERE basis = ?",
            (UNIVERSE_BASIS,),
        )
    ]
    con.close()
    members = list(merge_etf_candidates(tuple(members), include=True))

    unlock = as_of >= FROZEN_DATA_START
    reason = "record_forward 產生前推預測" if unlock else None
    prices = load_prices(members, start=date(2015, 1, 1), end=as_of, adjusted=True,
                         db_path=db_path, unlock_frozen=unlock, frozen_reason=reason)
    chips = load_chips(members, start=date(2015, 1, 1), end=as_of, db_path=db_path,
                       unlock_frozen=unlock, frozen_reason=reason)
    by_stock = build_dataset(members, prices, chips).by_stock

    calendar = V.trading_calendar(by_stock)
    day = calendar[-1]
    scores = V.precompute_scores(by_stock)[FAMILY]

    allowed = merge_etf_candidates(
        tuple(V.resolve_members([day], db_path, UNIVERSE_SIZE, UNIVERSE_BASIS)
              .get(day) or ()),
        include=INCLUDE_ETFS,
    )
    ranked = pd.Series(
        {sid: scores[sid].get(day, np.nan) for sid in allowed if sid in scores}
    ).dropna()
    if len(ranked) < MIN_CANDIDATES:
        raise RuntimeError(
            f"{day.date()} 只有 {len(ranked)} 檔可評分，低於 {MIN_CANDIDATES}"
        )

    ordered = sorted(
        ranked.index,
        key=lambda sid: (-float(ranked[sid]),
                         deterministic_jitter(sid, DEFAULT_TIE_SEED)),
    )[:N_POSITIONS]

    trading_calendar = load_trading_calendar(db_path)
    due = due_trading_date(trading_calendar, day.date(), HOLDING_DAYS)
    predicted_at = datetime.now().astimezone().isoformat()
    amount = CAPITAL / N_POSITIONS

    output: list[ForwardPrediction] = []
    for rank, sid in enumerate(ordered, start=1):
        close = float(by_stock[sid]["close"].loc[day])
        tier = resolve_tier(price=close, amount=amount, is_etf=is_etf(sid))
        output.append(ForwardPrediction(
            predicted_at=predicted_at,
            data_asof=str(day.date()),
            strategy_version=STRATEGY_VERSION,
            params_json=PARAMS_JSON,
            stock_id=sid,
            rank=rank,
            score=float(ranked[sid]),
            entry_price=close,
            due_date=due.isoformat(),
            round_trip_cost=DEFAULT_COST.round_trip_rate(tier),
        ))
    return output


def settle(db_path: Path) -> tuple[int, int]:
    """
    結算已到期的預測。

    Returns:
        (實際結算筆數, 待結算總筆數)

    **只結算已有足夠未來交易日的**。報酬用 T+1 開盤買、T+HOLDING_DAYS
    收盤賣重算——`entry_price` 記的是決策日收盤，不是成交價。
    """
    pending = list_unsettled_predictions(db_path)
    if not pending:
        return 0, 0

    calendar = load_trading_calendar(db_path)
    position = {d: i for i, d in enumerate(calendar)}
    stock_ids = sorted({item.stock_id for item in pending})
    earliest = min(date.fromisoformat(item.data_asof) for item in pending)
    latest = calendar[-1]
    prices = load_prices(
        stock_ids, start=earliest, end=latest, adjusted=True, db_path=db_path,
        unlock_frozen=latest >= FROZEN_DATA_START,
        frozen_reason="record_forward --settle 回填實現報酬",
    )

    settled = 0
    for item in pending:
        params = json.loads(item.params_json)
        horizon = int(params.get("holding_days", HOLDING_DAYS))
        decision = date.fromisoformat(item.data_asof)
        if decision not in position:
            continue
        i = position[decision]
        if i + 1 + horizon >= len(calendar):
            continue                      # 還沒到期
        try:
            bars = prices.xs(item.stock_id, level="stock_id")
        except KeyError:
            continue
        entry_day, exit_day = calendar[i + 1], calendar[i + 1 + horizon]
        if entry_day not in bars.index or exit_day not in bars.index:
            continue                      # 停牌等缺價，不回填舊價假裝成交
        entry = float(bars.loc[entry_day, "open"])
        exit_ = float(bars.loc[exit_day, "close"])
        if entry <= 0:
            continue
        if settle_forward_prediction(
            db_path, item,
            realized_return=exit_ / entry - 1.0,
            settled_at=datetime.now(UTC),
        ):
            settled += 1
    return settled, len(pending)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="前推預測記錄與結算（無策略參數可調）")
    parser.add_argument("--settle", action="store_true", help="結算已到期預測")
    parser.add_argument("--dry-run", action="store_true", help="只顯示不寫入")
    parser.add_argument("--db", type=Path, default=HISTORY_DB_PATH)
    parser.add_argument("--as-of", default=None, help="決策日；預設資料庫最新日")
    args = parser.parse_args()

    if args.settle:
        done, pending = settle(args.db)
        print(f"已結算 {done} / {pending} 筆待結算預測")
        return

    as_of = date.fromisoformat(args.as_of) if args.as_of else latest_price_date(args.db)
    predictions = generate(args.db, as_of)

    print(f"版本 {STRATEGY_VERSION}｜資料截止 {predictions[0].data_asof}"
          f"｜揭曉日 {predictions[0].due_date}")
    print(f"每檔 {CAPITAL / N_POSITIONS:,.0f} 元｜{FAMILY}｜持有 {HOLDING_DAYS} 日")
    print()
    print(f"{'排名':>4}{'代號':>8}{'分數':>9}{'決策日收盤':>12}"
          f"{'成本分層':>12}{'來回成本':>10}")
    print("-" * 58)
    for p in predictions:
        tier = resolve_tier(price=p.entry_price, amount=CAPITAL / N_POSITIONS,
                            is_etf=is_etf(p.stock_id))
        print(f"{p.rank:>4}{p.stock_id:>8}{p.score:>9.4f}{p.entry_price:>12,.2f}"
              f"{tier.value:>12}{p.round_trip_cost*100:>9.3f}%")
    print("-" * 58)
    print(f"平均來回成本 {np.mean([p.round_trip_cost for p in predictions])*100:.3f}%")
    print()
    if args.dry_run:
        print("--dry-run：未寫入帳本")
        return
    inserted = record_forward_predictions(args.db, predictions)
    print(f"產生 {len(predictions)} 筆，新增 {inserted} 筆"
          + ("（0 筆代表這個版本在這個資料日已記錄過）" if inserted == 0 else ""))
    print()
    print("⚠️  這是目前證據最完整的一組，不是已證明能賺錢的策略。")
    print("    樣本外（已用掉的區間）超過隨機 95% 分位，但輸 0050 買進持有。")


if __name__ == "__main__":
    main()
