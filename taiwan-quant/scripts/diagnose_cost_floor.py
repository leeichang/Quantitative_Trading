#!/usr/bin/env python3
"""
40 萬資金的成本地板：每個槓桿值多少（開發集）

## 為什麼降成本比提升預測力值得先做

```
漲停預測力    隨機 2.47% → Top3 16.62%    6.7 倍，已經是實測到的最強訊號
週頻淨期望    毛 0.85%/週 vs 成本 1.071%   仍然是負的
```

**訊號再強都擋不住頻率乘上來的成本。** 而降成本不需要新證據——它是
算術，不是預測。

## 這份診斷輸出什麼

主要輸出是**成本欄**與**整股比例**，不是報酬。

⚠️ 掃 N × H 是 20 格參數空間。**不要從裡面挑最好的淨報酬那一格**——
那正是 DSR 0.0042 的來源（當時掃 126 組）。報酬欄只是讓成本的代價
看得見，要當結論必須先過 PBO / DSR。

## 三個修正（都讓成本變高）

### 1. 可負擔性用實際價，不是還原價

`resolve_tier` 用 `amount >= price * 1000` 判斷整股。還原價錨在最新日，
2016 年平均只有實際價的 86.7%，13.4% 的名字在 40 元門檻上分層錯邊，
方向是「看起來買得起整張」。

### 2. 用絕對金額算成本，不用比率

`CostModel.round_trip_rate()` 的 docstring 明寫「**忽略最低手續費**」。
最低手續費 20 元的生效門檻是 **23,392 元**（6 折）：

```
N=15   每檔 26,667   名目費 22.8 元   不咬
N=20   每檔 20,000   名目費 17.1 元   咬住 → 實際費率 0.100% 而非 0.0855%
```

所以一律用 `round_trip_cost(amount, tier)`。

### 3. 成本逐檔算完再平均，不是先平均價格

分層是門檻函數（整股／零股二分），對它取平均與先平均再分層不同。

## 禁令 6

一律跑開發集（預設 `--end 2023-12-29`）。
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.validate_oos_trailing as V  # noqa: E402, N812
from taiwan_quant.config.costs import (  # noqa: E402
    DEFAULT as DEFAULT_COST,
)
from taiwan_quant.config.costs import (
    FEE_DISCOUNT_DEFAULT,
    CostModel,
    Tier,
    resolve_tier,
)
from taiwan_quant.data.dataset import build_dataset  # noqa: E402
from taiwan_quant.data.etf_universe import is_etf, merge_etf_candidates  # noqa: E402
from taiwan_quant.data.integrity import (  # noqa: E402
    complete_holding_decision_dates,
    forward_returns,
)
from taiwan_quant.data.loader import (  # noqa: E402
    HISTORY_DB_PATH,
    RAW_OPEN_COLUMN,
    load_chips,
    load_prices,
)
from taiwan_quant.ranking.tie_break import (  # noqa: E402
    DEFAULT_TIE_SEED,
    deterministic_jitter,
)

FAMILY = "動能突破"
CAPITAL = 400_000.0
UNIVERSE_SIZE = 150
UNIVERSE_BASIS = "market_cap"

LARGE_TIER_SIZE = 50
"""
市值前 50 名視為 0050 級（滑價 0.3%），51~150 視為中型 100（0.4%）。

禁令 4 的明文分層。第一版全部走 `resolve_tier` 的預設 `large=True`，
於是排名 51~150 的名字也被套 0.3%——**那份成本地板表的每一格都低估
了成本**。這是 Codex 在平行做同一個任務時抓到的。
"""
MIN_CANDIDATES = 30
WARMUP = 750

DEV_END = date(2023, 12, 29)

N_GRID = (3, 5, 10, 15, 20)
H_GRID = (40, 60, 80, 120)

CURRENT_N = 10
CURRENT_H = 40
"""現行參數，用來當對照基準"""

DISCOUNT_GRID = (1.0, FEE_DISCOUNT_DEFAULT, 0.28)
"""手續費折扣：無折扣 / 6 折（預設）/ 2.8 折（大戶）"""


def sweep(
    n_positions: int,
    holding_days: int,
    calendar: list[pd.Timestamp],
    scores: dict[str, dict],
    members_at_cache: dict[int, dict],
    large_at_cache: dict[int, dict],
    opens: pd.DataFrame,
    raw_opens: pd.DataFrame,
    forward_cache: dict[int, pd.DataFrame],
    cost: CostModel,
    include_etfs: bool,
) -> dict | None:
    """
    跑一組 (N, H)，回傳成本與報酬彙總。

    決策間隔等於持有期：不重疊，每筆標籤在下一個決策日前就揭曉。
    """
    amount = CAPITAL / n_positions
    forward = forward_cache[holding_days]
    members_at = members_at_cache[holding_days]
    large_at = large_at_cache[holding_days]
    decision_dates = [
        day
        for day in calendar[WARMUP::holding_days]
        if calendar.index(day) + 1 + holding_days < len(calendar)
    ]

    periods: list[dict] = []
    for day in decision_dates:
        allowed = merge_etf_candidates(
            tuple(members_at.get(day) or ()), include=include_etfs
        )
        if not allowed:
            continue
        ranked = pd.Series(
            {sid: scores[sid].get(day, np.nan) for sid in allowed if sid in scores}
        ).dropna()
        realized = forward.loc[day].dropna()
        common = ranked.index.intersection(realized.index)
        if len(common) < MIN_CANDIDATES:
            continue

        picks = sorted(
            common,
            key=lambda sid: (
                -float(ranked[sid]),
                deterministic_jitter(sid, DEFAULT_TIE_SEED),
            ),
        )[:n_positions]

        entry_day = calendar[calendar.index(day) + 1]
        large_members = large_at.get(day, set())
        rates, whole = [], 0
        for sid in picks:
            actual = float(raw_opens.loc[entry_day, sid])
            adjusted = float(opens.loc[entry_day, sid])
            if not (np.isfinite(actual) and actual > 0
                    and np.isfinite(adjusted) and adjusted > 0):
                continue
            tier = resolve_tier(
                actual_price=actual,
                adjusted_price=adjusted,
                amount=amount,
                # 禁令 4：0050 成分股 0.3%、中型 100 為 0.4%。
                # 第一版全部走預設的 large=True，等於把排名 51~150 的
                # 名字也套 0.3%——**每一格的成本都被低估**。
                large=sid in large_members,
                is_etf=is_etf(sid),
            )
            # 絕對金額 ÷ 金額 才含最低手續費；round_trip_rate() 明文忽略它
            rates.append(cost.round_trip_cost(amount, tier) / amount)
            whole += tier in (Tier.LARGE_WHOLE, Tier.MID_WHOLE, Tier.ETF_WHOLE)
        if not rates:
            continue

        periods.append(
            {
                "decision_date": str(day.date()),
                "gross": float(realized[picks].mean()),
                "cost": float(np.mean(rates)),
                "whole_ratio": whole / len(rates),
            }
        )

    if not periods:
        return None

    frame = pd.DataFrame(periods)
    frame["net"] = frame["gross"] - frame["cost"]
    frame["year"] = pd.to_datetime(frame["decision_date"]).dt.year
    trips = 252 / holding_days
    nets = frame["net"].to_numpy()
    yearly = frame.groupby("year")["net"].mean()

    return {
        "n_positions": n_positions,
        "holding_days": holding_days,
        "amount": amount,
        "periods": len(frame),
        "whole_ratio": float(frame["whole_ratio"].mean()),
        "cost_per_trip": float(frame["cost"].mean()),
        # 規格 13：換手率與年化成本拖累是一級輸出
        "annual_cost_drag": float(frame["cost"].mean() * trips),
        "gross_per_trip": float(frame["gross"].mean()),
        "net_per_trip": float(nets.mean()),
        "annualised_net": float((1 + nets.mean()) ** trips - 1),
        "positive_years": f"{int((yearly > 0).sum())}/{len(yearly)}",
        "series": {
            row["decision_date"]: row["net"] for row in frame.to_dict("records")
        },
    }


def paired_within_horizon(results: list[dict]) -> None:
    """
    **只在同一個 H 之內**配對比較不同的 N。

    ## 為什麼不可以跨 H 配對

    第一版這樣做了，結果是錯的：`net_per_trip` 在 H=40 是 40 日報酬、
    在 H=120 是 120 日報酬。把兩者相減再叫「淨差異」等於在量持有期長度，
    不是在量參數好壞。

    當時 20 格裡有 3 格 |t| > 2，**全部是 H=120**——那個「顯著」完全是
    這個錯造出來的。

    跨 H 的比較只能看年化欄，而且**不是配對的**：決策日不同、期數不同
    （H=120 只有 12 期），所以不附標準誤，也不可宣稱顯著。
    """
    print("── 同一持有期內，不同 N 的配對差異（基準 N=10）" + "─" * 18)
    print(f"{'設定':<16}{'期數':>6}{'淨差異':>10}{'標準誤':>9}{'t':>7}{'判讀':>14}")
    for horizon in H_GRID:
        same = [r for r in results if r["holding_days"] == horizon]
        base = next((r for r in same if r["n_positions"] == CURRENT_N), None)
        if base is None:
            continue
        for item in same:
            if item is base:
                continue
            shared = sorted(set(base["series"]) & set(item["series"]))
            label = f"N={item['n_positions']}, H={horizon}"
            diff = np.array(
                [item["series"][day] - base["series"][day] for day in shared]
            )
            if len(diff) < 8 or diff.std(ddof=1) == 0:
                print(f"{label:<16}{len(diff):>6}{'—':>10}{'—':>9}{'—':>7}"
                      f"{'期數不足':>14}")
                continue
            stderr = diff.std(ddof=1) / np.sqrt(len(diff))
            t_stat = diff.mean() / stderr
            verdict = "量不出差別" if abs(t_stat) < 2 else (
                "顯著較差" if t_stat < 0 else "顯著較好"
            )
            print(f"{label:<16}{len(diff):>6}{diff.mean():>9.2%}{stderr:>9.2%}"
                  f"{t_stat:>7.2f}{verdict:>14}")
        print()

    print("⚠️  跨持有期不可配對比較：H=40 的「淨/趟」是 40 日報酬、H=120 的是")
    print("    120 日報酬。跨 H 只能看年化欄，而 H=120 只有 12 期——先前實測")
    print("    H=120 的 PBO 是 0.971、DSR 算不出來，那才是它的判定。")
    print()


def report(payload: dict) -> None:
    print(f"\n開發集 {payload['dev_end']} 為止｜資金 {CAPITAL:,.0f} 元\n")

    print("── 成本地板：N × H " + "─" * 46)
    print(f"{'N':>3}{'H':>5}{'每檔金額':>11}{'整股比例':>9}{'成本/趟':>9}"
          f"{'年化成本':>9}{'毛/趟':>8}{'淨/趟':>8}{'年化淨':>9}{'期數':>6}{'逐年正':>8}")
    for item in payload["grid"]:
        marker = "  ←現行" if (
            item["n_positions"] == CURRENT_N and item["holding_days"] == CURRENT_H
        ) else ""
        print(
            f"{item['n_positions']:>3}{item['holding_days']:>5}"
            f"{item['amount']:>11,.0f}{item['whole_ratio']:>9.1%}"
            f"{item['cost_per_trip']:>9.3%}{item['annual_cost_drag']:>9.1%}"
            f"{item['gross_per_trip']:>8.2%}{item['net_per_trip']:>8.2%}"
            f"{item['annualised_net']:>9.1%}{item['periods']:>6}"
            f"{item['positive_years']:>8}{marker}"
        )
    print()

    paired_within_horizon(payload["grid"])

    print("── 手續費折扣值多少（N=10, H=40）" + "─" * 30)
    print(f"{'折扣':>8}{'成本/趟':>10}{'vs 6 折':>10}")
    for row in payload["discounts"]:
        print(f"{row['discount']:>8.2f}{row['cost_per_trip']:>10.3%}"
              f"{row['delta_vs_default']:>+10.3%}")
    print()

    print("── ETF 納入候選（N=10, H=40）" + "─" * 34)
    for row in payload["etf"]:
        label = "含 ETF" if row["include_etfs"] else "不含 ETF"
        print(f"{label:>10}  整股比例 {row['whole_ratio']:>6.1%}"
              f"  成本/趟 {row['cost_per_trip']:.3%}"
              f"  毛/趟 {row['gross_per_trip']:.2%}")
    print()
    print("   完全相同，而且不是接線錯誤——ETF 每期都在候選池、都有分數，")
    print("   只是從來沒進前 10：")
    for row in payload["etf_ranks"]:
        print(f"     {row['stock_id']}  出現 {row['periods']} 期"
              f"｜排名中位 {row['median_rank']:.0f}"
              f"｜最佳 {row['best_rank']}"
              f"｜進前 10 的期數 {row['top10_periods']}")
    print("   結構性原因：分散的一籃子在定義上不會出現動能突破。")
    print("   ETF 成本分層與 is_etf 仍然需要——對照組要用。")
    print()

    print("⚠️  上表是 20 格參數空間。**不要從裡面挑淨報酬最好的那一格**——")
    print("    DSR 0.0042 就是掃 126 組的結果。這份診斷的輸出是成本欄。")


def run(db_path: Path, end: date, unlock: bool, reason: str | None) -> dict:
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

    prices = load_prices(members, start=date(2015, 1, 1), end=end, adjusted=True,
                         db_path=db_path, unlock_frozen=unlock, frozen_reason=reason)
    chips = load_chips(members, start=date(2015, 1, 1), end=end, db_path=db_path,
                       unlock_frozen=unlock, frozen_reason=reason)
    by_stock = build_dataset(members, prices, chips).by_stock

    calendar = V.trading_calendar(by_stock)
    scores = V.precompute_scores(by_stock)[FAMILY]

    def frame(column: str) -> pd.DataFrame:
        return pd.DataFrame(
            {sid: bars[column].astype(float) for sid, bars in by_stock.items()}
        ).reindex(calendar)

    opens, closes = frame("open"), frame("close")
    raw_opens = frame(RAW_OPEN_COLUMN)

    forward_cache, members_at_cache, large_at_cache = {}, {}, {}
    for horizon in H_GRID:
        forward_cache[horizon] = forward_returns(
            opens, closes, holding_days=horizon
        )
        dates = complete_holding_decision_dates(
            calendar, calendar[WARMUP::horizon], holding_days=horizon
        )
        members_at_cache[horizon] = V.resolve_members(
            dates, db_path, UNIVERSE_SIZE, UNIVERSE_BASIS
        )
        # 市值前 50 才是 0050 級（滑價 0.3%），其餘走中型 100（0.4%）
        large_at_cache[horizon] = V.resolve_members(
            dates, db_path, LARGE_TIER_SIZE, UNIVERSE_BASIS
        )

    grid = []
    for n_positions in N_GRID:
        for horizon in H_GRID:
            item = sweep(n_positions, horizon, calendar, scores, members_at_cache,
                         large_at_cache, opens, raw_opens, forward_cache,
                         DEFAULT_COST, include_etfs=False)
            if item is not None:
                grid.append(item)
        print(f"  N={n_positions} 完成", flush=True)

    discounts = []
    for discount in DISCOUNT_GRID:
        item = sweep(CURRENT_N, CURRENT_H, calendar, scores, members_at_cache,
                     large_at_cache, opens, raw_opens, forward_cache,
                     CostModel(fee_discount=discount), include_etfs=False)
        if item is not None:
            discounts.append({"discount": discount,
                              "cost_per_trip": item["cost_per_trip"]})
    default_cost = next(
        row["cost_per_trip"] for row in discounts
        if row["discount"] == FEE_DISCOUNT_DEFAULT
    )
    for row in discounts:
        row["delta_vs_default"] = row["cost_per_trip"] - default_cost

    etf = []
    for include in (False, True):
        item = sweep(CURRENT_N, CURRENT_H, calendar, scores, members_at_cache,
                     large_at_cache, opens, raw_opens, forward_cache,
                     DEFAULT_COST, include_etfs=include)
        if item is not None:
            etf.append({"include_etfs": include,
                        "whole_ratio": item["whole_ratio"],
                        "cost_per_trip": item["cost_per_trip"],
                        "gross_per_trip": item["gross_per_trip"]})

    return {"dev_end": str(end), "capital": CAPITAL,
            "grid": grid, "discounts": discounts, "etf": etf,
            "etf_ranks": etf_rank_audit(
                calendar, scores, members_at_cache[CURRENT_H],
                forward_cache[CURRENT_H]
            )}


def etf_rank_audit(
    calendar: list[pd.Timestamp],
    scores: dict[str, dict],
    members_at: dict,
    forward: pd.DataFrame,
) -> list[dict]:
    """
    ETF 在排序裡實際落在第幾名。

    「含 ETF 與不含 ETF 結果完全相同」有兩種可能的原因，而它們的意義
    完全不同：

        接線錯誤   ETF 沒進候選池 → 要修
        排不進去   ETF 在池裡但排名太後 → 這個槓桿本來就沒用

    **不查就分不出來。** 所以這裡逐期記錄 ETF 的實際排名。
    """
    from taiwan_quant.data.etf_universe import ETF_CANDIDATES

    ranks: dict[str, list[int]] = {sid: [] for sid in ETF_CANDIDATES}
    decision_dates = [
        day
        for day in calendar[WARMUP::CURRENT_H]
        if calendar.index(day) + 1 + CURRENT_H < len(calendar)
    ]
    for day in decision_dates:
        allowed = merge_etf_candidates(
            tuple(members_at.get(day) or ()), include=True
        )
        ranked = pd.Series(
            {sid: scores[sid].get(day, np.nan) for sid in allowed if sid in scores}
        ).dropna()
        realized = forward.loc[day].dropna()
        common = ranked.index.intersection(realized.index)
        if len(common) < MIN_CANDIDATES:
            continue
        ordered = sorted(
            common,
            key=lambda sid: (
                -float(ranked[sid]),
                deterministic_jitter(sid, DEFAULT_TIE_SEED),
            ),
        )
        for sid in ETF_CANDIDATES:
            if sid in ordered:
                ranks[sid].append(ordered.index(sid) + 1)

    return [
        {
            "stock_id": sid,
            "periods": len(values),
            "median_rank": float(np.median(values)),
            "best_rank": int(min(values)),
            "top10_periods": int(sum(1 for r in values if r <= CURRENT_N)),
        }
        for sid, values in sorted(ranks.items())
        if values
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=HISTORY_DB_PATH)
    parser.add_argument(
        "--end", type=date.fromisoformat, default=DEV_END,
        help="資料載入上限。預設開發集結束日；改動等於動用 OOS（禁令 6）",
    )
    parser.add_argument("--unlock-frozen", action="store_true")
    parser.add_argument("--reason", default=None)
    parser.add_argument("--out", type=Path,
                        default=Path("reports/cost_floor_dev.json"))
    args = parser.parse_args()

    payload = run(args.db, args.end, args.unlock_frozen, args.reason)
    report(payload)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\n原始輸出：{args.out}")


if __name__ == "__main__":
    main()
