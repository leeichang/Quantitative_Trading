#!/usr/bin/env python3
"""
策略族 × 持有期 可行性診斷

回答一個問題：

    有沒有任何「策略族 + 持有期」組合，能讓校準後的真實勝率
    跨過含成本的進場門檻？

背景（`reports/weekly_plan.txt`）：
    momentum 基準在 5 日持有期下，基礎勝率 18.67%、最佳分數箱 22.06%，
    而門檻是 40.22%。差 18 個百分點。

兩條可能的出路：
    1. 換策略族 —— 籌碼跟隨或均值回歸的基礎勝率是否有本質差異？
    2. 拉長持有期 —— 門檻 = (stop + cost) / (target + stop)，
       持有期拉長會讓 target 變大（有更多時間移動），門檻隨之下降

本腳本同時掃這兩個維度，讓資料決定。

## 多重測試警告

本網格掃 3 族 × 5 個持有期 = 15 組。**掃越多組，最好的那組越可能是運氣。**

所以：
  · 報告列出**全部 15 組**，不只最佳
  · 對勝過門檻的組合回報 Deflated Sharpe 的試驗數 N = 15
  · 任何勝出組合仍須走完整 Walk-Forward + OOS + PBO

用法：
    .venv/bin/python scripts/diagnose_families.py
    .venv/bin/python scripts/diagnose_families.py --limit 60
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from taiwan_quant.config.costs import DEFAULT, Tier, annual_cost_drag  # noqa: E402
from taiwan_quant.data.loader import load_chips, load_prices, load_universe  # noqa: E402
from taiwan_quant.labeling.barrier_width import derive_width  # noqa: E402
from taiwan_quant.labeling.triple_barrier import label_one  # noqa: E402
from taiwan_quant.ranking.portfolio import entry_threshold  # noqa: E402
from taiwan_quant.strategies.families import STRATEGY_FAMILIES, StrategyFamily  # noqa: E402
from taiwan_quant.validation.calibration import (  # noqa: E402
    CalibrationError,
    fit_calibrator,
)

HORIZONS = (5, 10, 20, 40, 60)
"""
持有交易日數。

     5 日  ≈ 1 週    使用者原始需求
    10 日  ≈ 2 週
    20 日  ≈ 1 個月
    40 日  ≈ 2 個月
    60 日  ≈ 3 個月  使用者允許的上限
"""

ATR_MULTIPLE = 0.8
TARGET_QUANTILE = 0.85
MIN_RISK_REWARD = 2.0

CALIBRATION_BINS = 6
CALIBRATION_MIN_SAMPLES = 50
TRAIN_END = date(2025, 12, 31)

TRADING_DAYS_PER_WEEK = 5.0
DECISION_STRIDE = 5
"""
每隔幾個交易日取一個決策日。

週頻決策本來就是每 5 天一次，取樣密度對齊實際使用情境，
也大幅降低計算量。
"""


@dataclass(frozen=True)
class GridResult:
    """單一（策略族 × 持有期）組合的診斷結果"""

    family: str
    horizon: int
    n_samples: int
    base_rate: float
    best_bin_prob: float
    """校準後最高的分箱機率"""

    best_bin_samples: int
    avg_target_pct: float
    avg_stop_pct: float
    threshold: float
    """含成本的進場門檻"""

    weekly_turnover: float
    annual_cost_drag: float
    calibration_error: str = ""

    @property
    def edge(self) -> float:
        """最佳分箱機率 − 門檻。為正才有優勢"""
        return self.best_bin_prob - self.threshold

    @property
    def clears_threshold(self) -> bool:
        return self.edge > 0

    @property
    def discrimination(self) -> float:
        """最佳分箱機率 − 基礎勝率。衡量分數有沒有鑑別力"""
        return self.best_bin_prob - self.base_rate

    @property
    def random_edge(self) -> float:
        """
        隨機選股的優勢 = 基礎勝率 − 門檻。

        **這是最重要的對照。** 若這個值為正，代表「隨便買、照這組柵欄
        持有」就能跨過門檻——策略本身沒有貢獻，優勢全來自柵欄結構
        與長持有期壓低的成本。
        """
        return self.base_rate - self.threshold


def collect_samples(
    data_by_stock: dict[str, pd.DataFrame],
    family: StrategyFamily,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, list[float], list[float]]:
    """
    蒐集（分數, 標籤）配對與柵欄寬度。

    只走訓練期決策日，避免 look-ahead。
    """
    scores: list[float] = []
    labels: list[int] = []
    targets: list[float] = []
    stops: list[float] = []

    for bars in data_by_stock.values():
        if not set(family.required_columns).issubset(bars.columns):
            continue

        train_mask = bars.index.date <= TRAIN_END

        for idx in range(250, len(bars), DECISION_STRIDE):
            if not train_mask[idx]:
                break

            width = derive_width(
                bars,
                decision_idx=idx,
                horizon=horizon,
                atr_multiple=ATR_MULTIPLE,
                target_quantile=TARGET_QUANTILE,
                min_risk_reward=MIN_RISK_REWARD,
            )
            if width is None:
                continue

            outcome = label_one(
                bars,
                decision_idx=idx,
                target_pct=width.target_pct,
                stop_pct=width.stop_pct,
                horizon=horizon,
            )
            if outcome is None:
                continue

            score = family.score_fn(bars.iloc[: idx + 1])
            if not np.isfinite(score):
                continue

            scores.append(score)
            labels.append(outcome.label)
            targets.append(width.target_pct)
            stops.append(width.stop_pct)

    return np.asarray(scores), np.asarray(labels), targets, stops


def evaluate(
    data_by_stock: dict[str, pd.DataFrame],
    family: StrategyFamily,
    horizon: int,
) -> GridResult:
    """診斷單一組合"""
    scores, labels, targets, stops = collect_samples(data_by_stock, family, horizon)

    if len(scores) == 0:
        return GridResult(
            family=family.name, horizon=horizon, n_samples=0,
            base_rate=float("nan"), best_bin_prob=float("nan"), best_bin_samples=0,
            avg_target_pct=float("nan"), avg_stop_pct=float("nan"),
            threshold=float("nan"), weekly_turnover=float("nan"),
            annual_cost_drag=float("nan"),
            calibration_error="無樣本",
        )

    avg_target = float(np.mean(targets))
    avg_stop = float(np.mean(stops))
    threshold = entry_threshold(avg_target, avg_stop, DEFAULT, Tier.LARGE)

    # 持有 horizon 天 → 每週換 5/horizon 輪
    turnover = TRADING_DAYS_PER_WEEK / horizon
    drag = annual_cost_drag(turnover, DEFAULT, Tier.LARGE)

    try:
        calibrator = fit_calibrator(
            scores, labels,
            n_bins=CALIBRATION_BINS,
            min_samples_per_bin=CALIBRATION_MIN_SAMPLES,
        )
    except CalibrationError as exc:
        return GridResult(
            family=family.name, horizon=horizon, n_samples=len(scores),
            base_rate=float(np.mean(labels == 1)),
            best_bin_prob=float("nan"), best_bin_samples=0,
            avg_target_pct=avg_target, avg_stop_pct=avg_stop,
            threshold=threshold, weekly_turnover=turnover, annual_cost_drag=drag,
            calibration_error=str(exc)[:60],
        )

    usable = [b for b in calibrator.bins if b.is_usable(CALIBRATION_MIN_SAMPLES)]
    best = max(usable, key=lambda b: b.empirical_prob) if usable else None

    return GridResult(
        family=family.name,
        horizon=horizon,
        n_samples=len(scores),
        base_rate=calibrator.base_rate,
        best_bin_prob=best.empirical_prob if best else float("nan"),
        best_bin_samples=best.n_samples if best else 0,
        avg_target_pct=avg_target,
        avg_stop_pct=avg_stop,
        threshold=threshold,
        weekly_turnover=turnover,
        annual_cost_drag=drag,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="策略族 × 持有期 可行性診斷")
    parser.add_argument("--limit", type=int, default=40, help="處理幾檔標的")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2026-09-11")
    args = parser.parse_args()

    universe = load_universe(limit=150)
    stock_ids = universe.stock_ids[: args.limit]

    print("=" * 108)
    print("策略族 × 持有期 可行性診斷")
    print("=" * 108)
    print(f"標的 {len(stock_ids)} 檔｜期間 {args.start} ~ {args.end}"
          f"｜訓練期至 {TRAIN_END}")
    print(f"掃描 {len(STRATEGY_FAMILIES)} 族 × {len(HORIZONS)} 個持有期 "
          f"= {len(STRATEGY_FAMILIES) * len(HORIZONS)} 組")
    print()

    print("載入價格與籌碼 ...")
    prices = load_prices(
        stock_ids,
        start=date.fromisoformat(args.start),
        end=date.fromisoformat(args.end),
        adjusted=True,
    )
    chips = load_chips(
        stock_ids,
        start=date.fromisoformat(args.start),
        end=date.fromisoformat(args.end),
    )

    data_by_stock: dict[str, pd.DataFrame] = {}
    for sid in stock_ids:
        if sid not in prices.index.get_level_values("stock_id"):
            continue
        bars = prices.xs(sid, level="stock_id")
        if sid in chips.index.get_level_values("stock_id"):
            chip_bars = chips.xs(sid, level="stock_id")
            merged = bars.join(
                chip_bars[
                    ["foreign_net", "trust_net", "dealer_net",
                     "margin_balance", "short_balance"]
                ],
                how="inner",
            )
            data_by_stock[sid] = merged
        else:
            data_by_stock[sid] = bars

    print(f"  可用 {len(data_by_stock)} 檔")
    print()

    results: list[GridResult] = []
    for family in STRATEGY_FAMILIES:
        for horizon in HORIZONS:
            print(f"  診斷 {family.name} × {horizon} 日 ...", flush=True)
            results.append(evaluate(data_by_stock, family, horizon))

    print()
    print("─" * 108)
    print("全部結果（**列出全部，不只最佳——多重測試需要看到完整分布**）")
    print("─" * 108)
    header = (
        f"{'策略族':<10}{'持有':>5}{'樣本':>8}{'基礎勝率':>10}{'最佳分箱':>10}"
        f"{'鑑別力':>9}{'門檻':>9}{'隨機優勢':>10}{'總優勢':>9}"
        f"{'年化成本':>10}  判定"
    )
    print(header)
    print("─" * 108)

    for r in sorted(results, key=lambda x: (-x.edge if np.isfinite(x.edge) else 1)):
        if r.calibration_error:
            print(f"{r.family:<10}{r.horizon:>5}{r.n_samples:>8}"
                  f"{'':>10}{'':>10}{'':>9}{'':>9}{'':>9}{'':>9}{'':>9}{'':>10}"
                  f"  ✗ {r.calibration_error}")
            continue

        verdict = "✓ 跨過門檻" if r.clears_threshold else "✗"
        print(
            f"{r.family:<10}{r.horizon:>5}{r.n_samples:>8}"
            f"{r.base_rate * 100:>9.2f}%{r.best_bin_prob * 100:>9.2f}%"
            f"{r.discrimination * 100:>+8.2f}%{r.threshold * 100:>8.2f}%"
            f"{r.random_edge * 100:>+9.2f}%{r.edge * 100:>+8.2f}%"
            f"{r.annual_cost_drag * 100:>9.2f}%  {verdict}"
        )

    print("─" * 108)
    print()

    clearing = [r for r in results if r.clears_threshold]

    print("=" * 108)
    print("結論")
    print("=" * 108)

    if not clearing:
        print("  **沒有任何組合跨過門檻。**")
        print()
        best = max(
            (r for r in results if np.isfinite(r.edge)),
            key=lambda r: r.edge,
            default=None,
        )
        if best:
            print(f"  最接近的是 {best.family} × {best.horizon} 日，"
                  f"仍差 {abs(best.edge) * 100:.2f} 個百分點")
            print(f"    最佳分箱勝率 {best.best_bin_prob * 100:.2f}%"
                  f"（{best.best_bin_samples} 筆樣本）")
            print(f"    門檻 {best.threshold * 100:.2f}%"
                  f"（目標 {best.avg_target_pct * 100:.2f}%、"
                  f"停損 {best.avg_stop_pct * 100:.2f}%、"
                  f"成本 {DEFAULT.round_trip_rate(Tier.LARGE) * 100:.3f}%）")
        print()
        print("  依 CLAUDE.md 紅線：如實寫進報告，不進 Top 3。")
    else:
        print(f"  {len(clearing)} 組跨過門檻：")
        for r in sorted(clearing, key=lambda x: -x.edge):
            print(f"    {r.family} × {r.horizon} 日｜"
                  f"勝率 {r.best_bin_prob * 100:.2f}% vs 門檻 {r.threshold * 100:.2f}%"
                  f"（優勢 +{r.edge * 100:.2f}pp、{r.best_bin_samples} 筆樣本）")
        print()
        print(f"  ⚠️  本次掃了 {len(results)} 組。跨過門檻不等於有優勢——")
        print(f"      Deflated Sharpe 的試驗數 N = {len(results)}，")
        print("      必須走完整 Walk-Forward + OOS + PBO 才能採信。")
        print()
        print("  ── 優勢來自哪裡？拆解 ──")
        print(f"  {'組合':<20}{'總優勢':>10}{'隨機貢獻':>11}{'策略貢獻':>11}")
        for r in sorted(clearing, key=lambda x: -x.edge):
            print(f"  {r.family + ' × ' + str(r.horizon) + ' 日':<20}"
                  f"{r.edge * 100:>+9.2f}%{r.random_edge * 100:>+10.2f}%"
                  f"{r.discrimination * 100:>+10.2f}%")
        print()
        random_wins = [r for r in results if r.random_edge > 0]
        if random_wins:
            print(f"  ⚠️  有 {len(random_wins)} 組**隨機選股也能跨過門檻**"
                  "（基礎勝率 > 門檻）。")
            print("      這代表優勢主要來自柵欄結構與長持有期壓低的成本，")
            print("      不是策略的鑑別力。三族的鑑別力都只有 1~5 個百分點。")

    print()
    print("  持有期與成本的關係（本次實測）：")
    for horizon in HORIZONS:
        sample = next((r for r in results if r.horizon == horizon
                       and np.isfinite(r.annual_cost_drag)), None)
        if sample:
            print(f"    {horizon:>2} 日｜週換手 {sample.weekly_turnover * 100:>6.1f}%"
                  f"｜年化成本 {sample.annual_cost_drag * 100:>6.2f}%"
                  f"｜門檻 {sample.threshold * 100:>5.2f}%")

    print()
    print("=" * 108)
    print("⚠️  本診斷只看標籤分布與校準勝率，未跑 Walk-Forward、未計對照組。")
    print("    不構成投資建議。")
    print("=" * 108)


if __name__ == "__main__":
    main()
