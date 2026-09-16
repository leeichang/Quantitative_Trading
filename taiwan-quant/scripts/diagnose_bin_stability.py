#!/usr/bin/env python3
"""
最佳箱的逐年穩定度

## 要回答的問題

`2026-09-16_箱內排序能力與優勢集中度.md` 發現：三族的優勢都擠在一個薄箱。

```
策略族      選哪箱   報酬    占候選    每期檔數
動能突破     箱 0    7.53%    3.0%     4.3 檔
均值回歸     箱 5    7.45%    1.2%     1.7 檔
籌碼跟隨     箱 3    3.37%   14.2%    18.4 檔
```

動能突破的 7.53% 是 1,597 個樣本算出來的，箱內 t 值只有 0.18。

**那 7.53% 是真的，還是少數幾年撐起來的？**

## 為什麼要看超額而不是絕對報酬

2020~2021 是大多頭，任何箱的絕對報酬都會很好看；2022 是空頭，全部難看。
直接比絕對報酬只會量到市場方向。

所以主要看的是：

```
超額 = 該箱當年平均報酬 − 當年**所有候選**的平均報酬
```

這隔離掉市場水準，剩下的才是選股能力。

## 判準

```
超額為正的年數 ≥ 多數，且沒有單一年份撐起全部   → 優勢穩定
集中在 1~2 年                                  → 雜訊，不該當成能力
```

⚠️ 用全樣本擬合的校準器（不切分）——要問的是「這個箱的優勢在時間上穩不穩」
這個結構性問題，不是樣本外表現。**這個數字不能當成 OOS 結論。**

用法：
    .venv/bin/python scripts/diagnose_bin_stability.py
    .venv/bin/python scripts/diagnose_bin_stability.py --horizons 60 120
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from taiwan_quant.config.costs import DEFAULT, Tier  # noqa: E402
from taiwan_quant.data.dataset import build_dataset  # noqa: E402
from taiwan_quant.data.loader import (  # noqa: E402
    HISTORY_DB_PATH,
    load_chips,
    load_prices,
)
from taiwan_quant.strategies.families import STRATEGY_FAMILIES  # noqa: E402
from taiwan_quant.validation.binning import find_bin  # noqa: E402
from taiwan_quant.validation.calibration import (  # noqa: E402
    CalibrationError,
    fit_return_calibrator,
)

import scripts.validate_oos_trailing as V  # noqa: E402

ROUND_TRIP = DEFAULT.round_trip_rate(Tier.MID)
"""中型股一趟來回成本，當作「這個超額有沒有意義」的參考線"""


def rows_by_year_and_bin(
    by_date: dict[pd.Timestamp, list], calibrator
) -> tuple[dict[tuple[int, int], list[float]], dict[int, list[float]]]:
    """
    把每筆決策按 (年份, 箱) 分組，同時收集每年的全體候選。

    Returns:
        ((年, 箱) → 報酬清單, 年 → 全體候選報酬清單)

    落在箱外的候選不計入分箱，但**仍計入當年全體**——它們是真的候選，
    只是校準器不願評估。從分母拿掉會讓超額虛高。
    """
    per_bin: dict[tuple[int, int], list[float]] = defaultdict(list)
    per_year: dict[int, list[float]] = defaultdict(list)

    for day, candidates in by_date.items():
        year = int(day.year)
        for item in candidates:
            per_year[year].append(item.gross_return)
            index = find_bin(item.score, calibrator.bins)
            if index is not None:
                per_bin[(year, index)].append(item.gross_return)

    return per_bin, per_year


def picked_bin(calibrator) -> int:
    """排序會選中的箱——`predict()` 回箱平均報酬，所以是平均最高的那箱"""
    means = [b.mean_return for b in calibrator.bins]
    return int(np.argmax(means))


def main() -> None:
    parser = argparse.ArgumentParser(description="最佳箱的逐年穩定度")
    parser.add_argument("--horizons", type=int, nargs="*", default=[60])
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2023-12-29")
    parser.add_argument("--universe-size", type=int, default=V.UNIVERSE_SIZE)
    parser.add_argument("--basis", default="market_cap")
    parser.add_argument(
        "--min-samples",
        type=int,
        default=30,
        help="單年單箱少於這個數就標記出來——那一年的平均不可信",
    )
    parser.add_argument("--db", default=str(HISTORY_DB_PATH))
    args = parser.parse_args()

    db_path = Path(args.db)
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    members = [
        row[0]
        for row in con.execute(
            "SELECT DISTINCT stock_id FROM universe_history WHERE basis = ? "
            "ORDER BY stock_id",
            (args.basis,),
        )
    ]
    con.close()

    print("=" * 78)
    print("最佳箱的逐年穩定度 — 那個優勢是真的，還是少數幾年撐起來的？")
    print("=" * 78)
    print(f"標的池 {len(members)} 檔（{args.basis}）｜期間 {args.start} ~ {args.end}")
    print(f"參考線：一趟來回成本 {ROUND_TRIP * 100:.3f}%")
    print()

    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    prices = load_prices(members, start=start, end=end, adjusted=True, db_path=db_path)
    chips = load_chips(members, start=start, end=end, db_path=db_path)
    dataset = build_dataset(members, prices, chips)
    by_stock = dataset.by_stock
    print(dataset.describe())

    calendar = V.trading_calendar(by_stock)
    decision_dates = calendar[V.WARMUP_DAYS :: V.DECISION_STRIDE]
    members_at = V.resolve_members(
        decision_dates, db_path, args.universe_size, args.basis
    )

    print("預算分數序列 ...", flush=True)
    t0 = time.time()
    scores = V.precompute_scores(by_stock)
    print(f"  完成，耗時 {time.time() - t0:.1f}s\n")

    for horizon in args.horizons:
        print(f"預算 horizon={horizon} 的標記 ...", flush=True)
        t0 = time.time()
        outcomes = V.precompute_outcomes(by_stock, horizon, decision_dates)
        print(f"  {len(outcomes)} 檔、耗時 {time.time() - t0:.1f}s\n")

        for family in STRATEGY_FAMILIES:
            by_date = V.enumerate_decisions(
                outcomes, scores[family.name], decision_dates, members_at
            )
            all_decisions = [d for day in sorted(by_date) for d in by_date[day]]
            if not all_decisions:
                continue

            try:
                calibrator = fit_return_calibrator(
                    np.array([d.score for d in all_decisions]),
                    np.array([d.gross_return for d in all_decisions]),
                    n_bins=V.CALIBRATION_BINS,
                    min_samples_per_bin=V.CALIBRATION_MIN_SAMPLES,
                )
            except CalibrationError as exc:
                print(f"{family.name}：校準失敗：{exc}")
                continue

            target = picked_bin(calibrator)
            bucket = calibrator.bins[target]
            per_bin, per_year = rows_by_year_and_bin(by_date, calibrator)

            print("=" * 78)
            print(f"【{family.name}】持有 {horizon} 日｜排序會選中箱 {target}"
                  f"（全期平均 {bucket.mean_return * 100:.2f}%、{bucket.n_samples} 樣本）")
            print("=" * 78)
            print(f"{'年份':<8}{'該箱樣本':>10}{'該箱報酬':>11}"
                  f"{'全體候選':>11}{'超額':>11}   ")
            print("-" * 78)

            excesses: list[tuple[int, float, int]] = []
            for year in sorted(per_year):
                rows = per_bin.get((year, target), [])
                overall = float(np.mean(per_year[year]))
                if not rows:
                    print(f"{year:<8}{0:>10}{'—':>11}{overall * 100:>10.2f}%"
                          f"{'—':>11}   該箱當年無候選")
                    continue
                mean = float(np.mean(rows))
                excess = mean - overall
                flag = "  ⚠️ 樣本少" if len(rows) < args.min_samples else ""
                print(f"{year:<8}{len(rows):>10}{mean * 100:>10.2f}%"
                      f"{overall * 100:>10.2f}%{excess * 100:>+10.2f} pp{flag}")
                excesses.append((year, excess, len(rows)))

            print("-" * 78)
            if not excesses:
                print("  該箱在任何一年都沒有候選\n")
                continue

            positive = [e for e in excesses if e[1] > 0]
            values = np.array([e[1] for e in excesses])
            weights = np.array([e[2] for e in excesses], dtype=float)
            weighted = float(np.average(values, weights=weights))

            # 拿掉最好的一年，看還剩多少——雜訊的典型特徵是全靠一年
            best_year, best_excess, _ = max(excesses, key=lambda e: e[1])
            without_best = [e for e in excesses if e[0] != best_year]
            wb_weighted = (
                float(np.average(
                    [e[1] for e in without_best],
                    weights=[e[2] for e in without_best],
                ))
                if without_best else 0.0
            )

            # 前後期對比：優勢是否在衰退。拿掉最好那年看不出時間趨勢——
            # 一個穩定的優勢與一個正在消失的優勢，兩者的「加權平均」可以
            # 完全相同。**最後一年最重要**，它最接近尚未使用的評估區間。
            midpoint = len(excesses) // 2
            first_half = excesses[:midpoint]
            second_half = excesses[midpoint:]
            fh = float(np.average(
                [e[1] for e in first_half], weights=[e[2] for e in first_half]
            )) if first_half else 0.0
            sh = float(np.average(
                [e[1] for e in second_half], weights=[e[2] for e in second_half]
            )) if second_half else 0.0
            last_year, last_excess, _ = excesses[-1]

            print(f"  超額為正   {len(positive)} / {len(excesses)} 年")
            print(f"  加權平均超額           {weighted * 100:+.2f} pp")
            print(f"  最好的一年             {best_year}"
                  f"（超額 {best_excess * 100:+.2f} pp）")
            print(f"  拿掉最好那年後         {wb_weighted * 100:+.2f} pp")
            print(f"  前半期 {first_half[0][0]}~{first_half[-1][0]}"
                  f"        {fh * 100:+.2f} pp")
            print(f"  後半期 {second_half[0][0]}~{second_half[-1][0]}"
                  f"        {sh * 100:+.2f} pp")
            print(f"  最後一年 {last_year}            {last_excess * 100:+.2f} pp")
            print(f"  來回成本參考線         {ROUND_TRIP * 100:.3f}%")
            if sh < fh and last_excess < 0:
                print("  ⚠️  超額在衰退且最後一年轉負——"
                      "開發集尾端最接近評估區間，這是壞訊號。")

            if wb_weighted <= 0:
                print("  → **全靠一年**。拿掉之後超額轉負，這是雜訊不是能力。")
            elif wb_weighted < ROUND_TRIP:
                print("  → 拿掉最好那年後，超額不足以覆蓋來回成本。")
            elif len(positive) <= len(excesses) / 2:
                print("  → 超額為正的年數不到半數，穩定度不足。")
            else:
                print("  → 超額在多數年份為正且不靠單一年份，優勢看起來穩定。")
            print()

    print("=" * 78)
    print("判準：拿掉最好的一年後，加權超額仍需 > 來回成本，且多數年份為正")
    print("⚠️  這是全樣本擬合的結構性診斷，不是樣本外結論。")
    print("=" * 78)


if __name__ == "__main__":
    main()
