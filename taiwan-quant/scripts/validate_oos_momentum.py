#!/usr/bin/env python3
"""
動能突破 N=10 的一次性樣本外驗證

## ⚠️ 這個腳本只該跑一次

CLAUDE.md 禁令 6：**OOS 區間只跑一次，不得回頭調參。**

2024-01-01 ~ 2026-09-11 在本次之前**完全沒有被載入過**。所有的參數
選擇都在開發集（2015-01 ~ 2023-12-29）上完成，過程記錄於：

```
2026-09-16_漲停預測力與成本結構.md
2026-09-16_最佳箱的逐年穩定度.md
2026-09-16_持有期120日的實測結果.md
```

跑完之後這個區間就不再乾淨。**結果不論好壞都如實寫進報告。**

## 為什麼這一組參數（開發集上的依據）

現行系統把分數經過三層：校準器（12 等級）→ 門檻（edge_z）→ 槽位佇列。
實測那三層在毀訊號：

```
經過三層        60 日 CPCV 中位數 +5.06%｜5% 分位 −27.86%
直接用原始分數   40 日 CPCV 中位數 +94.39%｜5% 分位 +13.68%｜15/15 路徑為正
```

開發集上的證據：

```
毛報酬          4.89%/趟（隨機 10 檔 2.33%，隨機 95% 分位 3.11%）
逐年為正        7/8 年，最差 −1.6%
CPCV            15 條路徑全為正，最差 +3%
成本安全邊際     損益兩平滑價 2.210%，實證 0.094% → 23.5 倍
look-ahead      物理截斷測試 0 筆不一致
survivorship    時點標的池（跳過會虛增 39 pp）
```

開發集上**唯一不通過**的是 DSR 0.0042——因為總共掃了 126 組
（3 族 × 7 持有期 × 6 檔數），運氣的期望最佳 Sharpe 是 1.3054 而觀察到
0.8566。**DSR 說「這可能是從 126 組裡挑出來的運氣」，而樣本外是唯一
能分辨的方法。** 這就是本次的目的。

## 固定參數（禁令 7、8）

不接受任何策略參數的命令列覆寫——那會讓「只跑一次」失去意義。
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from taiwan_quant.data.dataset import build_dataset  # noqa: E402
from taiwan_quant.data.loader import (  # noqa: E402
    HISTORY_DB_PATH,
    load_chips,
    load_price_views,
)
from taiwan_quant.validation.benchmarks import (  # noqa: E402
    equity_curve_statistics,
)
from taiwan_quant.data.etf_universe import (  # noqa: E402
    is_etf,
    merge_etf_candidates,
)
from taiwan_quant.config.costs import (  # noqa: E402
    DEFAULT as DEFAULT_COST,
    Tier,
    resolve_tier,
)
from taiwan_quant.ranking.tie_break import DEFAULT_TIE_SEED, deterministic_jitter  # noqa: E402

import scripts.validate_oos_trailing as V  # noqa: E402

# ══════════════════════════════════════════════════════════════
# 固定參數——不可由命令列覆寫（禁令 6、7、8）
# ══════════════════════════════════════════════════════════════

FAMILY = "動能突破"
HOLDING_DAYS = 40
DECISION_STRIDE = 40
"""決策間隔等於持有期：不重疊，每筆標籤在下一個決策日前就揭曉"""

N_POSITIONS = 10
CAPITAL = 400_000.0
UNIVERSE_SIZE = 150
UNIVERSE_BASIS = "market_cap"
MIN_CANDIDATES = 30

REFERENCE_TIER = Tier.LARGE_WHOLE
"""
對照組（等權、隨機）用的代表性分層。

策略本身的成本**逐檔**由 `resolve_tier` 決定，不用這個常數——買不起整張
的走零股、ETF 走 ETF 分層。這裡只是讓對照組有一個固定的參考成本，
否則隨機組合每次抽到不同標的會連成本一起變動，就不是純粹的對照了。
"""

ROUND_TRIP = DEFAULT_COST.round_trip_rate(REFERENCE_TIER)

OOS_START = date(2024, 1, 1)
DEV_END = date(2023, 12, 29)
STRATEGY_VERSION = "momentum_top10_h40@oos-2026-09-16"


def build_frame(
    by_stock: dict[str, pd.DataFrame], column: str, calendar: list[pd.Timestamp]
) -> pd.DataFrame:
    return pd.DataFrame(
        {sid: bars[column].astype(float) for sid, bars in by_stock.items()}
    ).reindex(calendar)


def run(db_path: Path, unlock: bool, reason: str | None,
        include_etfs: bool = False) -> dict:
    """在 OOS 區間跑一次，回傳可落盤的結果"""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    members = [
        row[0]
        for row in con.execute(
            "SELECT DISTINCT stock_id FROM universe_history WHERE basis = ?",
            (UNIVERSE_BASIS,),
        )
    ]
    latest = con.execute("SELECT MAX(date) FROM stock_daily").fetchone()[0]
    con.close()
    end = date.fromisoformat(latest)

    # ETF **一律載入**（對照組需要），但只有 include_etfs 時才進候選名單。
    # 第一版忘了這件事，導致必跑對照組的 0050/0051/0056 全部印不出來。
    members = list(merge_etf_candidates(tuple(members), include=True))

    # 從 2015 載入是為了讓 T 日的分數有足夠歷史；分數只用 T 日及之前的資料
    # （物理截斷測試已驗證無 look-ahead），決策日則嚴格限制在 OOS 區間內。
    price_views = load_price_views(
        members, start=date(2015, 1, 1), end=end, db_path=db_path,
        unlock_frozen=unlock, frozen_reason=reason,
    )
    chips = load_chips(members, start=date(2015, 1, 1), end=end, db_path=db_path,
                       unlock_frozen=unlock, frozen_reason=reason)
    by_stock = build_dataset(members, price_views.adjusted, chips).by_stock
    actual_by_stock = {
        sid: price_views.actual.xs(sid, level="stock_id")
        for sid in members if sid in price_views.actual.index.get_level_values("stock_id")
    }

    calendar = V.trading_calendar(by_stock)
    scores = V.precompute_scores(by_stock)[FAMILY]
    opens = build_frame(by_stock, "open", calendar)
    actual_opens = build_frame(actual_by_stock, "open", calendar)
    closes = build_frame(by_stock, "close", calendar)

    oos_ts = pd.Timestamp(OOS_START)
    anchor = next(i for i, d in enumerate(calendar) if d >= oos_ts)
    decision_dates = calendar[anchor::DECISION_STRIDE]
    members_at = V.resolve_members(
        decision_dates, db_path, UNIVERSE_SIZE, UNIVERSE_BASIS
    )
    large_at = V.resolve_members(
        decision_dates, db_path, 50, UNIVERSE_BASIS
    )
    forward = closes.shift(-HOLDING_DAYS) / opens.shift(-1) - 1

    trades, per_period = [], []
    for day in decision_dates:
        allowed = merge_etf_candidates(
            tuple(members_at.get(day) or ()), include=include_etfs)
        if not allowed:
            continue
        ranked = pd.Series(
            {sid: scores[sid].get(day, np.nan) for sid in allowed if sid in scores}
        ).dropna()
        realized = forward.loc[day].dropna()
        common = ranked.index.intersection(realized.index)
        if len(common) < MIN_CANDIDATES:
            continue
        ordered = sorted(
            common,
            key=lambda sid: (-float(ranked[sid]),
                             deterministic_jitter(sid, DEFAULT_TIE_SEED)),
        )
        picks = ordered[:N_POSITIONS]
        gross = float(realized[picks].mean())

        # 成本逐檔決定（禁令 3：一律呼叫 config/costs.py）。
        # 第一版把 0.671% 寫死在腳本裡，那繞過了單一來源，而且無法反映
        # 「買不起整張就是零股」與「ETF 跳動單位細 10 倍」這兩件事。
        entry_day = calendar[calendar.index(day) + 1]
        large_members = large_at.get(day, set())
        per_name_cost = []
        for sid in picks:
            adjusted_price = float(opens.loc[entry_day, sid])
            actual_price = float(actual_opens.loc[entry_day, sid])
            tier = resolve_tier(
                                actual_price=actual_price,
                                adjusted_price=adjusted_price,
                                amount=CAPITAL / N_POSITIONS,
                                large=sid in large_members,
                                is_etf=is_etf(sid))
            per_name_cost.append(DEFAULT_COST.round_trip_rate(tier))
        cost = float(np.mean(per_name_cost))

        per_period.append({"decision_date": str(day.date()),
                           "gross": gross, "cost": cost, "net": gross - cost,
                           "n_candidates": int(len(common))})
        for sid, c in zip(picks, per_name_cost, strict=True):
            adjusted_price = float(opens.loc[entry_day, sid])
            actual_price = float(actual_opens.loc[entry_day, sid])
            trades.append({"decision_date": str(day.date()), "stock_id": sid,
                           "score": float(ranked[sid]),
                           "gross_return": float(realized[sid]),
                           "entry_price": adjusted_price,
                           "actual_entry_price": actual_price,
                           "round_trip_cost": c,
                           "tier": resolve_tier(
                               actual_price=actual_price,
                               adjusted_price=adjusted_price,
                               amount=CAPITAL / N_POSITIONS,
                               large=sid in large_members,
                               is_etf=is_etf(sid)).value,
                           "is_etf": is_etf(sid)})

    nets = np.array([p["net"] for p in per_period])
    equity = pd.Series(np.cumprod(1 + nets),
                       index=pd.to_datetime([p["decision_date"] for p in per_period]))
    trips_per_year = 252 / HOLDING_DAYS

    # 對照組：同一個 OOS 區間
    first, last = decision_dates[0], calendar[min(
        calendar.index(decision_dates[-1]) + HOLDING_DAYS, len(calendar) - 1)]
    bench = {}
    for sid in ("0050", "0051", "0056"):
        if sid not in closes.columns:
            continue
        curve = closes[sid].loc[first:last].dropna()
        if len(curve) < 2:
            continue
        stats = equity_curve_statistics(curve)
        bench[sid] = {"total_return": stats.total_return,
                      "max_drawdown": stats.max_drawdown, "sharpe": stats.sharpe}

    rng = np.random.default_rng(20260916)
    trials = []
    for _ in range(100):
        vals = []
        for day in decision_dates:
            allowed = members_at.get(day)
            if not allowed:
                continue
            realized = forward.loc[day].reindex(allowed).dropna()
            if len(realized) < MIN_CANDIDATES:
                continue
            idx = rng.choice(len(realized), N_POSITIONS, replace=False)
            vals.append(float(realized.iloc[idx].mean()) - ROUND_TRIP)
        if vals:
            trials.append(float(np.prod(1 + np.array(vals)) - 1))

    eq_weight = []
    for day in decision_dates:
        allowed = members_at.get(day)
        if not allowed:
            continue
        realized = forward.loc[day].reindex(allowed).dropna()
        if len(realized) < MIN_CANDIDATES:
            continue
        eq_weight.append(float(realized.mean()) - ROUND_TRIP)

    return {
        "strategy_version": STRATEGY_VERSION,
        "generated_at": datetime.now().astimezone().isoformat(),
        "parameters": {
            "family": FAMILY, "holding_days": HOLDING_DAYS,
            "decision_stride": DECISION_STRIDE, "n_positions": N_POSITIONS,
            "capital": CAPITAL, "universe_size": UNIVERSE_SIZE,
            "universe_basis": UNIVERSE_BASIS, "min_candidates": MIN_CANDIDATES,
            "cost_model": "taiwan_quant.config.costs.DEFAULT",
            "reference_tier": REFERENCE_TIER.value,
            "reference_round_trip": ROUND_TRIP,
            "strategy_cost": "per-name via resolve_tier()",
            "tie_seed": DEFAULT_TIE_SEED, "dev_end": str(DEV_END),
            "include_etfs": include_etfs,
            "oos_start": str(OOS_START), "data_end": str(end),
        },
        "n_periods": len(per_period),
        "n_trades": len(trades),
        "gross_per_trip": float(np.mean([p["gross"] for p in per_period])),
        "net_per_trip": float(nets.mean()),
        "cumulative_net": float(np.prod(1 + nets) - 1),
        "annualized_net": float((1 + nets.mean()) ** trips_per_year - 1),
        "sharpe": float(nets.mean() / nets.std(ddof=1) * np.sqrt(trips_per_year))
        if len(nets) > 1 and nets.std(ddof=1) > 0 else None,
        "max_drawdown": float(((equity.cummax() - equity) / equity.cummax()).max()),
        "periods_positive": int((nets > 0).sum()),
        "benchmarks": {
            "etf": bench,
            "equal_weight_universe": float(np.prod(1 + np.array(eq_weight)) - 1)
            if eq_weight else None,
            "random_median": float(np.median(trials)) if trials else None,
            "random_p05": float(np.percentile(trials, 5)) if trials else None,
            "random_p95": float(np.percentile(trials, 95)) if trials else None,
        },
        "per_period": per_period,
        "trades": trades,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="動能突破 N=10 的一次性 OOS 驗證（無策略參數可調）")
    parser.add_argument("--db", type=Path, default=HISTORY_DB_PATH)
    parser.add_argument("--out", type=Path,
                        default=Path("reports/oos_momentum_top10_h40.json"))
    parser.add_argument(
        "--include-etfs", action="store_true",
        help="把 0050/0051/0056 加進候選。⚠️ 需要新的 OOS 區間，"
             "不可在已用過的 2024-01~2026-08 上驗證（禁令 6）",
    )
    parser.add_argument("--unlock-frozen", action="store_true")
    parser.add_argument("--frozen-reason", default=None)
    args = parser.parse_args()
    if args.unlock_frozen and not (args.frozen_reason or "").strip():
        parser.error("--unlock-frozen 必須提供非空白的 --frozen-reason")

    r = run(args.db, args.unlock_frozen, args.frozen_reason,
            include_etfs=args.include_etfs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(r, ensure_ascii=False, indent=2))

    p = r["parameters"]
    print("=" * 74)
    print("動能突破 N=10 一次性樣本外驗證")
    print("=" * 74)
    print(f"版本   {r['strategy_version']}")
    print(f"OOS    {p['oos_start']} ~ {p['data_end']}（開發集止於 {p['dev_end']}）")
    print(f"參數   {p['family']}｜持有 {p['holding_days']} 日｜N={p['n_positions']}"
          f"｜標的池 {p['universe_basis']} 前 {p['universe_size']}")
    print(f"成本   {p['cost_model']}｜策略逐檔分層"
          f"｜對照組參考 {p['reference_tier']} {p['reference_round_trip']*100:.3f}%")
    print(f"       實際平均來回 {np.mean([x['cost'] for x in r['per_period']])*100:.3f}%")
    print()
    print(f"期數 {r['n_periods']}｜交易 {r['n_trades']} 筆"
          f"｜為正 {r['periods_positive']}/{r['n_periods']}")
    print(f"毛/趟 {r['gross_per_trip']*100:+.2f}%｜淨/趟 {r['net_per_trip']*100:+.2f}%")
    print(f"累積淨報酬 {r['cumulative_net']*100:+.2f}%｜年化 {r['annualized_net']*100:+.2f}%")
    sh = r["sharpe"]
    print(f"Sharpe {sh:.3f}" if sh is not None else "Sharpe n/a", end="")
    print(f"｜最大回撤 {r['max_drawdown']*100:.2f}%")
    print()
    b = r["benchmarks"]
    print("必跑對照組（同一 OOS 區間）")
    for sid, s in b["etf"].items():
        shr = f"{s['sharpe']:.2f}" if s["sharpe"] is not None else "n/a"
        print(f"  {sid} 買進持有   {s['total_return']*100:+8.2f}%"
              f"  Sharpe {shr}  MaxDD {s['max_drawdown']*100:.2f}%")
    if b["equal_weight_universe"] is not None:
        print(f"  等權全池        {b['equal_weight_universe']*100:+8.2f}%")
    if b["random_median"] is not None:
        print(f"  隨機 {N_POSITIONS} 檔中位  {b['random_median']*100:+8.2f}%"
              f"  （5%~95%：{b['random_p05']*100:+.2f}% ~ {b['random_p95']*100:+.2f}%）")
    print()
    print(f"原始證據 {args.out}")
    print("=" * 74)
    print("⚠️  此 OOS 區間至此已被使用。依禁令 6 不得再回頭調參後重跑。")
    print("=" * 74)


if __name__ == "__main__":
    main()
