#!/usr/bin/env python3
"""
CPCV 路徑分布診斷（任務 H）

## 這個腳本要回答的問題

`03_待辦與改進方向.md` 開頭那張表：

```
組合             v6         v7          差異
均值回歸 × 60    +954.87%   +123.93%   −831 pp
均值回歸 × 120   +294.62%   +802.42%   +508 pp
等權對照          +701.37%   +688.79%    −13 pp   ← 對照組只動 13 pp
```

v6 → v7 只補了 23 期缺漏的市值快照，**策略參數完全沒動**。

當時的結論「不是 bug，是路徑相依」是對的，但無法量化——單一 walk-forward
只給一條路徑。CPCV 用同一份資料生出 15 條，直接給分布。

## 怎麼讀輸出

```
全距 ≤ 數十 pp      估計量穩定，結論可信
全距 = 數百 pp      估計量本身的變異蓋過訊號，那個點估計不該被當成能力
```

⚠️ **15 條路徑不是 15 個獨立樣本。** 共用同一份歷史，只是切法不同。
用它否定比用它肯定可靠得多。

## 與 walk-forward 的關係

CPCV 的訓練集會用到測試段**之後**的資料，所以它**不能**拿來宣稱實盤
表現（禁令 5）。它量的是估計量的穩定度。實盤模擬仍然要用
`validate_oos_trailing.py`。兩個一起看，不是二選一。

用法：
    .venv/bin/python scripts/diagnose_cpcv.py
    .venv/bin/python scripts/diagnose_cpcv.py --horizons 60 --end 2023-12-29
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

from taiwan_quant.backtest.portfolio_sim import simulate_portfolio  # noqa: E402
from taiwan_quant.config.costs import Tier  # noqa: E402
from taiwan_quant.data.dataset import build_dataset  # noqa: E402
from taiwan_quant.data.loader import (  # noqa: E402
    HISTORY_DB_PATH,
    load_chips,
    load_prices,
)
from taiwan_quant.strategies.families import STRATEGY_FAMILIES  # noqa: E402
from taiwan_quant.validation.calibration import (  # noqa: E402
    CalibrationError,
    fit_return_calibrator,
)
from taiwan_quant.validation.cpcv import (  # noqa: E402
    CPCVError,
    cpcv_folds,
    summarize_paths,
)
from taiwan_quant.validation.external.multipletesting import (  # noqa: E402
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
)
from taiwan_quant.validation.fold_signals import (  # noqa: E402
    select_fold_signals,
    standard_errors_by_bin,
)

import scripts.validate_oos_trailing as V  # noqa: E402


def run_one_family(
    family_name: str,
    by_date: dict[pd.Timestamp, list],
    tiers: dict[str, Tier],
    calendar: list[pd.Timestamp],
    price_lookup,
    horizon: int,
    edge_z: float,
    slots: int,
) -> tuple[list[float], int]:
    """
    跑完一個策略族的所有 CPCV 路徑。

    Returns:
        (每條路徑的總報酬, 校準失敗的路徑數)

    校準失敗的路徑**跳過而不是記 0**——記 0 會把「算不出來」混進
    「算出來是 0」，兩者意義完全不同。
    """
    decision_dates = sorted(by_date)
    path_returns: list[float] = []
    failures = 0

    for train_dates, test_dates in cpcv_folds(
        decision_dates, horizon=horizon, stride=V.DECISION_STRIDE
    ):
        train_decisions = [d for day in train_dates for d in by_date[day]]
        if not train_decisions:
            failures += 1
            continue

        try:
            calibrator = fit_return_calibrator(
                np.array([d.score for d in train_decisions]),
                np.array([d.gross_return for d in train_decisions]),
                n_bins=V.CALIBRATION_BINS,
                min_samples_per_bin=V.CALIBRATION_MIN_SAMPLES,
            )
        except CalibrationError:
            failures += 1
            continue

        picks = select_fold_signals(
            {day: by_date[day] for day in test_dates},
            calibrator,
            standard_errors_by_bin(calibrator.bins),
            tiers,
            edge_z,
        )
        if not picks:
            path_returns.append(0.0)      # 沒有訊號 = 不進場 = 0 報酬，這是真實結果
            continue

        result = simulate_portfolio(
            V.to_signals(picks, calendar, tiers),
            price_lookup,
            calendar,
            n_slots=slots,
        )
        path_returns.append(float(result.total_return))

    return path_returns, failures


def full_sample_curves(
    outcomes,
    scores,
    decision_dates: list[pd.Timestamp],
    members_at,
    tiers: dict[str, Tier],
    calendar: list[pd.Timestamp],
    price_lookup,
    horizon: int,
    edge_z: float,
    slots: int,
) -> dict[str, pd.Series]:
    """
    每個策略族的全樣本逐日報酬序列。

    PBO 與 DSR 吃的是**時間序列**：每一期一個報酬，同一時點跨策略可比。
    CPCV 的 15 條路徑總報酬不是序列——它們彼此在時間上重疊，長度也不是
    「期數」。硬餵進去會算出沒有意義的數字。

    這裡用同一套門檻、同一個校準流程，但**不切分**：整段資料擬合一次、
    整段套用一次。這只是為了拿到可比的報酬序列，**不是樣本外結果**。
    """
    curves: dict[str, pd.Series] = {}
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
        except CalibrationError:
            continue

        picks = select_fold_signals(
            by_date, calibrator, standard_errors_by_bin(calibrator.bins),
            tiers, edge_z,
        )
        if not picks:
            continue
        result = simulate_portfolio(
            V.to_signals(picks, calendar, tiers), price_lookup, calendar,
            n_slots=slots,
        )
        curves[family.name] = result.equity.pct_change().dropna()
    return curves


def report_multiple_testing(
    curves: dict[str, pd.Series], n_horizons: int, horizon: int
) -> None:
    """
    PBO 與 DSR。CLAUDE.md「多重測試校正」那一節要求的兩個數字。

    ⚠️ **DSR 的 `n_observations` 必須是有效樣本數，不是交易日數。**
    60 日持有時相鄰日的報酬幾乎完全重疊，2,196 個交易日只值
    2196 / 60 ≈ 36 個獨立觀測。餵原始日數會讓 DSR 灌水：

    ```
    n = 2196（逐日，重疊）   DSR = 1.0000
    n = 36（有效樣本）        DSR = 0.9741
    ```

    這與 `validation/rank_ic.py` 的重疊修正是同一條原則。
    """
    if len(curves) < 2:
        print("  策略族不足 2 個，無法比較排名")
        return

    width = min(len(s) for s in curves.values())
    matrix = np.column_stack([s.to_numpy()[:width] for s in curves.values()])
    try:
        pbo = probability_of_backtest_overfitting(matrix, n_splits=8)
        print(f"  PBO（{len(curves)} 個策略族、{width} 個交易日）：{pbo.pbo:.3f}")
        print("    " + ("⚠️  > 0.5，判定過擬合，不得進 Top 3"
                        if pbo.pbo > 0.5 else "未達過擬合門檻"))
    except ValueError as exc:
        print(f"  PBO 無法計算：{exc}")

    best_name = max(curves, key=lambda k: float(curves[k].mean()))
    best = curves[best_name].to_numpy()
    if best.std(ddof=1) <= 0:
        return
    daily_sharpe = float(best.mean() / best.std(ddof=1))
    effective = max(2, len(best) // horizon)      # 重疊修正，見 docstring
    dsr = deflated_sharpe_ratio(
        observed_sharpe=daily_sharpe * np.sqrt(V.TRADING_DAYS_PER_YEAR),
        n_trials=len(curves) * n_horizons,
        trial_sharpe_std=0.5,
        n_observations=effective,
    )
    print(f"\n  DSR（表現最好的 {best_name}）")
    print(f"    觀察到的年化 Sharpe   {dsr.observed_sharpe:.4f}")
    print(f"    運氣的期望最佳值       {dsr.expected_maximum_sharpe:.4f}")
    print(f"    有效樣本數             {effective}"
          f"（{len(best)} 個交易日 ÷ {horizon} 日持有）")
    print(f"    DSR                   {dsr.deflated_sharpe_ratio:.4f}")
    print(f"    通過？                 {dsr.survives}")


def main() -> None:
    parser = argparse.ArgumentParser(description="CPCV 路徑分布診斷")
    parser.add_argument("--horizons", type=int, nargs="*", default=[60])
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument(
        "--end",
        default="2023-12-29",
        help="資料載入上限。掃參數階段務必停在開發集結束日（禁令 6）",
    )
    parser.add_argument("--universe-size", type=int, default=V.UNIVERSE_SIZE)
    parser.add_argument("--basis", default="market_cap")
    parser.add_argument("--slots", type=int, default=V.TOP_N)
    parser.add_argument("--edge-z", type=float, default=1.0)
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

    print("=" * 76)
    print("CPCV 路徑分布診斷 — 估計量對切分方式有多敏感")
    print("=" * 76)
    print(f"標的池 {len(members)} 檔（{args.basis}）｜期間 {args.start} ~ {args.end}")
    print(f"槽位 {args.slots}｜edge_z {args.edge_z}")
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
    tiers = V.infer_tiers(db_path, args.universe_size, args.basis)
    price_lookup = V.make_price_lookup(by_stock)

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

        per_family: dict[str, list[float]] = {}
        for family in STRATEGY_FAMILIES:
            by_date = V.enumerate_decisions(
                outcomes, scores[family.name], decision_dates, members_at
            )
            try:
                returns, failures = run_one_family(
                    family.name, by_date, tiers, calendar, price_lookup,
                    horizon, args.edge_z, args.slots,
                )
            except CPCVError as exc:
                print(f"{family.name}：無法切分：{exc}")
                continue

            if not returns:
                print(f"{family.name}：{failures} 條路徑全部校準失敗")
                continue

            per_family[family.name] = returns
            dist = summarize_paths(returns)
            print(f"\n【{family.name}】（{failures} 條校準失敗已排除）")
            print(dist.describe())

        # 多重測試校正：用**逐日報酬序列**，不是 CPCV 的路徑總報酬
        #
        # ⚠️ 第一版把 15 條路徑的總報酬當成 PBO/DSR 的輸入，那是誤用：
        #    這兩個統計量要的是時間序列（每期一個報酬），15 個總報酬
        #    不是序列。實跑時被上游擋下來（「need at least 30 observations」），
        #    算是撿到——換個參數就會安靜地算出一個沒有意義的數字。
        if len(per_family) >= 2:
            print("\n" + "-" * 76)
            print("多重測試校正（用全樣本逐日報酬，與 CPCV 路徑分開算）")
            curves = full_sample_curves(
                outcomes, scores, decision_dates, members_at, tiers,
                calendar, price_lookup, horizon, args.edge_z, args.slots,
            )
            report_multiple_testing(
                curves, n_horizons=len(args.horizons), horizon=horizon
            )

        print()

    print("=" * 76)
    print("判準：全距數百 pp 代表估計量本身的變異蓋過訊號，")
    print("      那個點估計不該被當成策略能力。")
    print("⚠️  CPCV 量的是穩定度，不是實盤表現——實盤模擬用 validate_oos_trailing.py。")
    print("=" * 76)


if __name__ == "__main__":
    main()
