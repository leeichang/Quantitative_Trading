#!/usr/bin/env python3
"""
箱內排序能力診斷

## 這個腳本要回答的問題

`rank_score` 只有 12 個相異值（`CALIBRATION_BINS = 6` × 2 個流動性分層），
要排 150 檔。第 3 名平均有 3.8 檔同分，Top 3 裡 1~2 個位置是平手決定的。

修 tie-break 只把偏差移除（動能突破虛增的 17.54 pp），**變異還在**：
CPCV 全距 126 ~ 191 pp，離驗收目標「數十 pp」還很遠。

根源是排序解析度。三個可能的解法：

```
1. 提高 CALIBRATION_BINS          每箱要 ≥ 50 樣本，6 → 12 箱需兩倍樣本
2. 換連續模型（Ridge / LightGBM）  還沒有 baseline
3. 排序改用原始分數，校準只當門檻  最便宜
```

**第 3 個的前提是「校準器分不出來的差異，原始分數分得出來」。**
這個腳本就是驗證那個前提。

## 怎麼讀輸出

```
箱內 |t| > 2 且平均 IC > 0.02     原始分數在箱內還有排序能力 → 解法 3 可行
箱內 IC ≈ 0                      分數的資訊已經被分箱吃乾了 → 解法 3 沒用，
                                 而且解法 1（更多箱）也救不了
```

第二種情況是**壞消息但重要**：它代表問題不在「箱太粗」，而在「分數本身
的解析度就只有這麼多」。那時候只剩解法 2。

## 為什麼要分箱算，不能算全體

全體 Rank IC 混進了「箱與箱之間」的排序能力——那部分校準器**已經用上
了**。要問的是箱內還剩多少，所以必須在箱內算。

重疊修正逐箱獨立做：每一箱自己是一條時間序列（每個決策日一個 IC），
把 6 箱混在一起會讓有效期數虛增 6 倍。

用法：
    .venv/bin/python scripts/diagnose_within_bin_ic.py
    .venv/bin/python scripts/diagnose_within_bin_ic.py --horizons 60 --min-names 12
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
from taiwan_quant.validation.rank_ic import (  # noqa: E402
    RankICError,
    evaluate_rank_ic,
)

import scripts.validate_oos_trailing as V  # noqa: E402


def observations_by_bin(
    by_date: dict[pd.Timestamp, list],
    calibrator,
    min_names: int,
) -> dict[int, dict[pd.Timestamp, tuple[np.ndarray, np.ndarray]]]:
    """
    把每個決策日的候選按校準箱分組。

    Args:
        by_date: 決策日 → 候選清單
        calibrator: 已擬合的 `ReturnCalibrator`
        min_names: 單箱單日至少要有幾檔才納入

    Returns:
        箱索引 → {決策日: (分數陣列, 報酬陣列)}

    落在箱外（`find_bin` 回 `None`）的候選直接丟掉——那是校準器不願意
    評估的，不是「箱 0」。
    """
    grouped: dict[int, dict[pd.Timestamp, list[tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for day, candidates in by_date.items():
        for item in candidates:
            index = find_bin(item.score, calibrator.bins)
            if index is None:
                continue
            grouped[index][day].append((item.score, item.gross_return))

    result: dict[int, dict[pd.Timestamp, tuple[np.ndarray, np.ndarray]]] = {}
    for index, per_day in grouped.items():
        usable = {
            day: (
                np.array([s for s, _ in rows]),
                np.array([r for _, r in rows]),
            )
            for day, rows in per_day.items()
            if len(rows) >= min_names
        }
        if usable:
            result[index] = usable
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="箱內排序能力診斷")
    parser.add_argument("--horizons", type=int, nargs="*", default=[60])
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument(
        "--end",
        default="2023-12-29",
        help="資料載入上限。這是診斷，一律停在開發集結束日（禁令 6）",
    )
    parser.add_argument("--universe-size", type=int, default=V.UNIVERSE_SIZE)
    parser.add_argument("--basis", default="market_cap")
    parser.add_argument(
        "--min-names",
        type=int,
        default=12,
        help="單箱單日至少幾檔才算 IC。太少的話算出來幾乎全是雜訊",
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
    print("箱內排序能力 — 校準器分不出來的差異，原始分數分得出來嗎？")
    print("=" * 78)
    print(f"標的池 {len(members)} 檔（{args.basis}）｜期間 {args.start} ~ {args.end}")
    print(f"單箱單日最少 {args.min_names} 檔")
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

            print("=" * 78)
            print(f"【{family.name}】持有 {horizon} 日")
            print("=" * 78)

            # 先印箱本身的樣子：校準器的排序是否單調
            print(f"{'箱':<4}{'分數下界':>10}{'分數上界':>10}{'樣本':>8}"
                  f"{'箱平均報酬':>12}")
            print("-" * 78)
            for i, bucket in enumerate(calibrator.bins):
                print(f"{i:<4}{bucket.lo:>10.4f}{bucket.hi:>10.4f}"
                      f"{bucket.n_samples:>8}{bucket.mean_return * 100:>11.2f}%")
            print()

            grouped = observations_by_bin(by_date, calibrator, args.min_names)
            if not grouped:
                print("  沒有任何箱在任何一天湊到足夠檔數\n")
                continue

            print(f"{'箱':<4}{'期數':>7}{'有效期':>9}{'平均 IC':>10}{'ICIR':>8}"
                  f"{'t 值':>8}{'IC>0':>8}  判定")
            print("-" * 78)

            verdicts: list[tuple[int, float, float]] = []
            for index in sorted(grouped):
                try:
                    report = evaluate_rank_ic(
                        grouped[index],
                        horizon=horizon,
                        stride=V.DECISION_STRIDE,
                        min_names=args.min_names,
                    )
                except RankICError as exc:
                    print(f"{index:<4}  無法計算：{exc}")
                    continue

                verdict = "顯著" if report.is_significant else "無法排除運氣"
                print(f"{index:<4}{report.n_periods:>7}{report.effective_periods:>9.1f}"
                      f"{report.mean_ic:>+10.4f}{report.icir:>+8.3f}"
                      f"{report.t_stat:>+8.2f}{report.positive_rate * 100:>7.1f}%"
                      f"  {verdict}")
                verdicts.append((index, report.mean_ic, report.t_stat))

            print("-" * 78)
            if verdicts:
                significant = [v for v in verdicts if abs(v[2]) > 2.0]
                mean_abs_ic = float(np.mean([abs(v[1]) for v in verdicts]))
                print(f"  {len(significant)} / {len(verdicts)} 箱達 |t| > 2"
                      f"｜箱內 |平均 IC| 的平均 {mean_abs_ic:.4f}")
                if significant and mean_abs_ic > 0.02:
                    print("  → 原始分數在箱內仍有排序能力，"
                          "「排序用原始分數、校準只當門檻」可行")
                else:
                    print("  → 分數的資訊已被分箱吃乾。換排序鍵沒用，")
                    print("    加更多箱也救不了——要換連續模型。")
            print()

    print("=" * 78)
    print("判準：箱內 |t| > 2 且平均 IC > 0.02 才算「箱內還有排序能力」")
    print("⚠️  箱內 IC 顯著不等於能賺錢——能否覆蓋交易成本是另一回事。")
    print("=" * 78)


if __name__ == "__main__":
    main()
