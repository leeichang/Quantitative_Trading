#!/usr/bin/env python3
"""
訊號排序能力快篩（逐期 Rank IC）

## 為什麼要有這個腳本

先前用「全部樣本池在一起算一個 Spearman」判定訊號有沒有排序能力，
那是方法學錯誤。而且我**跑了 7 次完整 OOS 回測（每次 25 分鐘）才發現
判定方式本身是錯的**。

這個腳本幾分鐘就能給出同樣的答案，**應該是第一步，不是第八步**。

## 它同時回答兩個問題

```
1. 訊號有沒有橫斷面排序能力？          逐期 Rank IC + 重疊修正的 t 值
2. 改用超額報酬當標籤有沒有差？        兩種標籤並列
```

第二個問題很重要：現在的標籤是絕對毛報酬，而 2019-2026 等權買進持有
+689%。模型可能只是學到「市場會漲」。

## 判準

```
|t| > 2          排序能力不是運氣
平均 IC > 0.02   落在專業界認為可用的下緣
IC > 0 的期數    應明顯高於 50%
逐年不能集中     單一年份撐起全部 = 假訊號
```

**IC 顯著不等於能賺錢。** 能否覆蓋交易成本是另一回事。

用法：
    .venv/bin/python scripts/diagnose_rank_ic.py
    .venv/bin/python scripts/diagnose_rank_ic.py --horizons 20 60
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from taiwan_quant.data.dataset import build_dataset  # noqa: E402
from taiwan_quant.data.loader import (  # noqa: E402
    HISTORY_DB_PATH,
    load_chips,
    load_prices,
)
from taiwan_quant.labeling.excess import to_excess_return  # noqa: E402
from taiwan_quant.strategies.families import STRATEGY_FAMILIES  # noqa: E402
from taiwan_quant.validation.benchmarks import equal_weight_equity  # noqa: E402
from taiwan_quant.validation.rank_ic import RankICError, evaluate_rank_ic  # noqa: E402

import scripts.validate_oos_trailing as V  # noqa: E402

MIN_NAMES = 30
"""單期至少要有幾檔才納入。太窄的橫斷面算出來的 IC 幾乎全是雜訊"""


def main() -> None:
    parser = argparse.ArgumentParser(description="訊號排序能力快篩")
    parser.add_argument("--horizons", type=int, nargs="*", default=[20, 60])
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2026-09-11")
    parser.add_argument("--universe-size", type=int, default=150)
    parser.add_argument("--basis", default="market_cap")
    parser.add_argument("--db", default=str(HISTORY_DB_PATH))
    args = parser.parse_args()

    db_path = Path(args.db)
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    members = [r[0] for r in con.execute(
        "SELECT DISTINCT stock_id FROM universe_history WHERE basis = ? "
        "ORDER BY stock_id", (args.basis,))]
    con.close()

    print("=" * 76)
    print("訊號排序能力快篩 — 逐期橫斷面 Rank IC")
    print("=" * 76)
    print(f"標的池 {len(members)} 檔（{args.basis}）｜期間 {args.start} ~ {args.end}")
    print()

    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    prices = load_prices(members, start=start, end=end, adjusted=True, db_path=db_path)
    chips = load_chips(members, start=start, end=end, db_path=db_path)
    dataset = build_dataset(members, prices, chips)
    by_stock = dataset.by_stock
    print(dataset.describe())

    calendar = V.trading_calendar(by_stock)
    position = {d: i for i, d in enumerate(calendar)}
    decision_dates = calendar[V.WARMUP_DAYS :: V.DECISION_STRIDE]
    members_at = V.resolve_members(
        decision_dates, db_path, args.universe_size, args.basis
    )

    # 等權基準的逐日權益曲線，供超額報酬使用
    benchmark = equal_weight_equity(by_stock, calendar)
    print(f"基準（等權 {len(by_stock)} 檔）"
          f"{calendar[0].date()} ~ {calendar[-1].date()}："
          f"{(float(benchmark.iloc[-1]) - 1) * 100:+.2f}%")
    print()

    print("預算分數序列 ...", flush=True)
    t0 = time.time()
    scores = V.precompute_scores(by_stock)
    print(f"  完成，耗時 {time.time() - t0:.1f}s\n")

    for horizon in args.horizons:
        print(f"預算 horizon={horizon} 的標記 ...", flush=True)
        t0 = time.time()
        outcomes = V.precompute_outcomes(by_stock, horizon, decision_dates)
        print(f"  {len(outcomes)} 檔、耗時 {time.time() - t0:.1f}s\n")

        print("=" * 76)
        print(f"持有 {horizon} 日")
        print("=" * 76)
        print(f"{'策略族':<10}{'標籤':<10}{'平均 IC':>10}{'ICIR':>8}"
              f"{'t 值':>8}{'IC>0':>8}{'期數':>7}{'有效期':>8}  判定")
        print("-" * 76)

        reports: dict[tuple[str, str], object] = {}
        for family in STRATEGY_FAMILIES:
            by_date = V.enumerate_decisions(
                outcomes, scores[family.name], decision_dates, members_at
            )

            for label, use_excess in (("絕對報酬", False), ("超額報酬", True)):
                obs: dict[pd.Timestamp, tuple[np.ndarray, np.ndarray]] = {}
                for day, batch in by_date.items():
                    s_list, r_list = [], []
                    for item in batch:
                        if use_excess:
                            # 用**該筆自己的**進出場日。移動停損會讓
                            # 同一天進場的兩檔在不同日出場，用固定窗口
                            # 會多一層雜訊。
                            entry_pos = position.get(day)
                            if entry_pos is None:
                                continue
                            exit_pos = min(
                                entry_pos + item.holding_days, len(calendar) - 1
                            )
                            value = to_excess_return(
                                item.gross_return, benchmark,
                                calendar[entry_pos], calendar[exit_pos],
                            )
                            if value is None:
                                continue
                        else:
                            value = item.gross_return
                        s_list.append(item.score)
                        r_list.append(value)
                    if len(s_list) >= MIN_NAMES:
                        obs[day] = (np.asarray(s_list), np.asarray(r_list))

                try:
                    rep = evaluate_rank_ic(
                        obs, horizon=horizon, stride=V.DECISION_STRIDE,
                        min_names=MIN_NAMES,
                    )
                except RankICError as exc:
                    print(f"{family.name:<10}{label:<10}  無法計算：{exc}")
                    continue

                reports[(family.name, label)] = rep
                verdict = "顯著" if rep.is_significant else "無法排除運氣"
                print(f"{family.name:<10}{label:<10}{rep.mean_ic:>+10.4f}"
                      f"{rep.icir:>+8.3f}{rep.t_stat:>+8.2f}"
                      f"{rep.positive_rate * 100:>7.1f}%{rep.n_periods:>7}"
                      f"{rep.effective_periods:>8.1f}  {verdict}")

        print("-" * 76)
        print()

        # 最強的那組印逐年拆解
        if reports:
            best = max(reports.items(), key=lambda kv: abs(kv[1].t_stat))  # type: ignore[union-attr]
            (fam, label), rep = best
            print(f"逐年拆解（|t| 最大者：{fam} × {label}）")
            print(rep.describe())  # type: ignore[union-attr]
            print()

    print("=" * 76)
    print("判準：|t| > 2 且平均 IC > 0.02 才值得往下做完整回測")
    print("⚠️  IC 顯著不等於能賺錢——能否覆蓋交易成本是另一回事。")
    print("=" * 76)


if __name__ == "__main__":
    main()
