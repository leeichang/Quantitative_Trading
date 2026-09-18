"""
把融資暴增事件記進前推帳本（含同期虛無抽樣）

## 為什麼要另開一支

`scripts/record_forward.py` 記的是排序型策略：每 40 個交易日選前 10 檔。
事件型策略的節奏完全不同——**事件每天都可能發生，而且一天只有 0~3 檔。**

```
排序型   每 40 日記一次，每次 10 檔
事件型   每個交易日記一次，每次 0~3 檔（實測 2.7 檔/日）
```

⚠️ **這支必須每個交易日跑,不是每 40 日。** 若照排序型的節奏每 40 日跑
一次,只會抓到 1/40 的事件,2.9 年累積約 18 筆——那要 40 倍的時間才能
達到同樣的檢定力。

## 同期虛無抽樣

每一次記策略的同時,記一組**檔數相同、從同一天的標的池隨機抽**的名單。

這不消耗禁令 6 的區間：抽樣只用決策日當天為止的標的池,不看未來報酬。
報酬和策略一起在到期日結算。

**沒有它,揭曉時沒有任何可比對的對照組。** 目前帳本裡的兩個動能版本
就是這個狀態。

## 依據

`../qlib-tw-trader/docs/原理說明/2026-09-18_融資暴增之後的超額報酬.md`

開發集(2015-01 ~ 2023-12-29)實測：融資暴增後 H=40 超額 +1.709%／趟、
SE 0.328%、t = +5.21,過 20 格網格的 Bonferroni 門檻。
**只有 H=40 淨值為正**（H≤20 的超額低於 1.081% 的成本地板）。

## 已知限制

⚠️ 同一檔可能在數日內重複觸發（實測 2884 在 7 個交易日內觸發 3 次）。
開發集的研究按「日」叢聚,那處理了同日相關,**沒有處理同一檔跨日重複**。
帳本會照實記,結算時要檢查重複度。

## 到期日用單一來源

`data/calendar.due_trading_date` 已經處理「日曆剩餘未來交易日不足 horizon」
的情形（真正的前推預測必然如此）——它錨在 `as_of` 加上歷史中位跨度，
而不是錨在日曆末端。我本來自己寫了一份，在日曆末端就拋錯。

**這正是本週反覆記錄的那個毛病：重寫一份已經有單一來源的東西。**

## 參數固定（禁令 7、8）

不接受命令列覆寫策略參數。參數變了就換 `STRATEGY_VERSION`。
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from taiwan_quant.config.costs import DEFAULT as DEFAULT_COST  # noqa: E402
from taiwan_quant.data.calendar import due_trading_date  # noqa: E402
from taiwan_quant.config.costs import resolve_tier  # noqa: E402
from taiwan_quant.data.etf_universe import is_etf  # noqa: E402
from taiwan_quant.data.loader import (  # noqa: E402
    DEFAULT_UNIVERSE_BASIS,
    HISTORY_DB_PATH,
    load_chips,
    load_prices,
    load_universe_at,
)
from taiwan_quant.forward_predictions import (  # noqa: E402
    ForwardPrediction,
    record_forward_predictions,
)
from taiwan_quant.validation.event_study import expanding_quantile_mask  # noqa: E402

# ── 固定參數（禁令 7、8）──────────────────────────────────
STRATEGY_VERSION = "margin_spike_h40@v1"
NULL_VERSION = "margin_spike_null_h40@v1"

CAPITAL = 400_000.0
HOLDING_DAYS = 40
N_POSITIONS = 10
"""部位上限。事件型常常少於這個數——那是策略的性質,不是缺陷"""

UNIVERSE_SIZE = 150
LARGE_TIER_SIZE = 50
QUANTILE = 0.99
REFRESH_EVERY = 20
MIN_OBSERVATIONS = 5000
MARGIN_WINDOW = 20
RAW_CLOSE_COLUMN = "raw_close"
DIVERGENCE_TOLERANCE = 0.01

PARAMS = {
    "event": "margin_balance_spike",
    "quantile": QUANTILE,
    "refresh_every": REFRESH_EVERY,
    "min_observations": MIN_OBSERVATIONS,
    "margin_window": MARGIN_WINDOW,
    "holding_days": HOLDING_DAYS,
    "n_positions_cap": N_POSITIONS,
    "capital": CAPITAL,
    "universe_size": UNIVERSE_SIZE,
    "universe_basis": DEFAULT_UNIVERSE_BASIS,
    "cost_model": "resolve_tier() per name",
    "exit": f"T+{HOLDING_DAYS} close",
    "entry": "T+1 open",
    "evidence": "2026-09-18_融資暴增之後的超額報酬.md",
    "dev_excess_per_trip": 0.01709,
    "dev_standard_error": 0.00328,
    "dev_t": 5.21,
}


class RecordError(RuntimeError):
    """資料不足或名單為空。"""


def _members(db_path: Path) -> list[str]:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        return [
            row[0]
            for row in con.execute(
                "SELECT DISTINCT stock_id FROM universe_history "
                "WHERE basis = ? ORDER BY stock_id",
                (DEFAULT_UNIVERSE_BASIS,),
            )
        ]
    finally:
        con.close()


def _trading_calendar(db_path: Path) -> list[date]:
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


def spike_names(
    closes: pd.DataFrame, margin: pd.DataFrame, day: pd.Timestamp
) -> list[str]:
    """
    決策日當天觸發融資暴增的股票，依分數高低排序。

    分數 = 當日融資餘額變動 ÷ 自身 20 日平均餘額。除以自身均值才不會
    讓大型股永遠佔據極端值。門檻只用 ≤ t 的資料（禁令 1）。
    """
    scaled = margin.diff() / margin.rolling(MARGIN_WINDOW).mean().abs().replace(
        0, np.nan
    )
    mask = expanding_quantile_mask(
        scaled,
        quantile=QUANTILE,
        refresh_every=REFRESH_EVERY,
        min_observations=MIN_OBSERVATIONS,
    )
    if day not in mask.index:
        raise RecordError(f"{day.date()} 不在價格索引內")
    fired = [column for column in mask.columns if bool(mask.at[day, column])]
    return sorted(fired, key=lambda sid: -float(scaled.at[day, sid]))


def _build(
    *,
    version: str,
    params: dict[str, object],
    names: list[str],
    scores: dict[str, float],
    day: pd.Timestamp,
    due: date,
    predicted_at: str,
    prices: pd.DataFrame,
    large_members: set[str],
    amount: float,
) -> list[ForwardPrediction]:
    """把名單轉成帳本列。成本逐檔算（禁令 8）。"""
    payload = json.dumps(params, ensure_ascii=False, sort_keys=True)
    rows: list[ForwardPrediction] = []
    for rank, sid in enumerate(names, start=1):
        row = prices.loc[(sid, day)]
        adjusted_close = float(row["close"])
        actual_close = float(row[RAW_CLOSE_COLUMN])

        # 決策日當天兩價通常相同（還原錨在載入範圍最後一天），但那是巧合
        # 不是保證。差太多時停下來，不要靜默用錯的價格分成本層級。
        if adjusted_close > 0 and (
            abs(actual_close / adjusted_close - 1) > DIVERGENCE_TOLERANCE
        ):
            raise RecordError(
                f"{sid} 在 {day.date()} 的實際價 {actual_close:.2f} 與還原價 "
                f"{adjusted_close:.2f} 相差 "
                f"{abs(actual_close / adjusted_close - 1):.1%}，"
                "請先確認還原價的錨定日"
            )

        tier = resolve_tier(
            actual_price=actual_close,
            adjusted_price=adjusted_close,
            amount=amount,
            large=sid in large_members,
            is_etf=is_etf(sid),
        )
        rows.append(
            ForwardPrediction(
                predicted_at=predicted_at,
                data_asof=str(day.date()),
                strategy_version=version,
                params_json=payload,
                stock_id=sid,
                rank=rank,
                score=float(scores[sid]),
                entry_price=actual_close,
                due_date=due.isoformat(),
                round_trip_cost=DEFAULT_COST.round_trip_rate(tier),
            )
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(HISTORY_DB_PATH))
    parser.add_argument(
        "--as-of", default=None, help="決策日，預設為資料最後一個交易日"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="只印出不寫入帳本"
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    members = _members(db_path)
    calendar = _trading_calendar(db_path)
    as_of = (
        date.fromisoformat(args.as_of) if args.as_of else calendar[-1]
    )
    print(f"決策日 {as_of}｜交易日曆 {len(calendar)} 日｜標的 {len(members)} 檔")

    prices = load_prices(
        members, start=date(2015, 1, 1), end=as_of, adjusted=True, db_path=db_path
    )
    closes = prices["close"].unstack("stock_id").sort_index()
    chips = load_chips(members, start=date(2015, 1, 1), end=as_of, db_path=db_path)
    margin = chips["margin_balance"].unstack("stock_id").reindex(
        index=closes.index, columns=closes.columns
    )

    day = pd.Timestamp(as_of)
    universe = set(
        load_universe_at(as_of, db_path=db_path, limit=UNIVERSE_SIZE,
                         basis=DEFAULT_UNIVERSE_BASIS).stock_ids
    )
    large_members = set(
        load_universe_at(as_of, db_path=db_path, limit=LARGE_TIER_SIZE,
                         basis=DEFAULT_UNIVERSE_BASIS).stock_ids
    )

    fired = [sid for sid in spike_names(closes, margin, day) if sid in universe]
    fired = fired[:N_POSITIONS]
    if not fired:
        print("⚠️ 當日沒有任何標的觸發融資暴增——不寫入任何列。")
        print("   事件型策略常常空手，那是策略的性質。明天再跑。")
        return 0

    scaled = margin.diff() / margin.rolling(MARGIN_WINDOW).mean().abs().replace(
        0, np.nan
    )
    scores = {sid: float(scaled.at[day, sid]) for sid in fired}

    # 同期虛無：檔數相同、從同一天的標的池隨機抽。種子綁決策日，可重現。
    pool = sorted(
        sid for sid in universe
        if (sid, day) in prices.index and sid not in fired
    )
    if len(pool) < len(fired):
        raise RecordError(f"標的池 {len(pool)} 檔不足以抽 {len(fired)} 檔虛無")
    rng = np.random.default_rng(int(as_of.strftime("%Y%m%d")))
    drawn = [pool[i] for i in rng.choice(len(pool), size=len(fired), replace=False)]
    null_scores = {sid: 0.0 for sid in drawn}

    due = due_trading_date(calendar, as_of, HOLDING_DAYS)
    predicted_at = datetime.now().astimezone().isoformat()
    amount = CAPITAL / N_POSITIONS

    common = dict(
        day=day, due=due, predicted_at=predicted_at, prices=prices,
        large_members=large_members, amount=amount,
    )
    rows = _build(
        version=STRATEGY_VERSION, params=PARAMS,
        names=fired, scores=scores, **common,
    ) + _build(
        version=NULL_VERSION,
        params={**PARAMS, "event": "random_same_count",
                "seed": int(as_of.strftime("%Y%m%d"))},
        names=drawn, scores=null_scores, **common,
    )

    print(f"\n到期日 {due}（T+{HOLDING_DAYS} 交易日）｜每筆 {amount:,.0f} 元")
    print(f"\n{'版本':30s} {'#':>2s} {'代號':>6s} {'分數':>9s} "
          f"{'進場參考價':>10s} {'來回成本':>8s}")
    for row in rows:
        print(f"{row.strategy_version:30s} {row.rank:2d} {row.stock_id:>6s} "
              f"{row.score:9.4f} {row.entry_price:10.2f} "
              f"{row.round_trip_cost:8.3%}")

    if args.dry_run:
        print("\n--dry-run：未寫入帳本。")
        return 0

    inserted = record_forward_predictions(db_path, rows)
    print(f"\n已寫入 {inserted} 列（重複鍵不覆寫）。")
    print(f"揭曉日 {due}。")
    print("\n⚠️ 這支必須**每個交易日**跑。每 40 日跑一次只會抓到 1/40 的事件。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
