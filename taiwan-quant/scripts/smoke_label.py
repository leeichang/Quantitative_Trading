#!/usr/bin/env python3
"""
端到端煙霧測試：在真實台股資料上跑柵欄推導 + triple-barrier 標記

目的不是產生投資建議，而是驗證三件事：

1. 資料載入層接得上 qlib-tw-trader 的 SQLite
2. 柵欄寬度能從 ATR 與歷史報酬分布推導出合理值（不是寫死的）
3. triple-barrier 標記在真實資料上三類標籤都出現，且分布不退化

用法：
    .venv/bin/python scripts/smoke_label.py
    .venv/bin/python scripts/smoke_label.py --stocks 2330 2454 2317 --horizon 5
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from taiwan_quant.config.costs import DEFAULT, Tier, annual_cost_drag  # noqa: E402
from taiwan_quant.data.loader import (  # noqa: E402
    coverage_report,
    load_prices,
    load_universe,
)
from taiwan_quant.labeling.barrier_width import derive_width  # noqa: E402
from taiwan_quant.labeling.triple_barrier import label_one  # noqa: E402


def label_with_derived_width(
    bars: pd.DataFrame,
    horizon: int,
    min_risk_reward: float,
) -> pd.DataFrame:
    """
    對每個決策日先推導柵欄寬度、再標記。

    寬度推導不出來（資料不足或 R:R 不達標）的決策日會被略過——
    這正是設計意圖：不硬湊 R:R（CLAUDE.md 進場門檻）。
    """
    records: list[dict[str, object]] = []
    index: list[pd.Timestamp] = []

    for decision_idx in range(len(bars)):
        width = derive_width(
            bars,
            decision_idx=decision_idx,
            horizon=horizon,
            min_risk_reward=min_risk_reward,
        )
        if width is None:
            continue

        result = label_one(
            bars,
            decision_idx=decision_idx,
            target_pct=width.target_pct,
            stop_pct=width.stop_pct,
            horizon=horizon,
        )
        if result is None:
            continue

        index.append(bars.index[decision_idx])
        records.append(
            {
                "label": result.label,
                "target_pct": width.target_pct,
                "stop_pct": width.stop_pct,
                "risk_reward": width.risk_reward,
                "atr_value": width.atr_value,
                "entry_price": result.entry_price,
                "exit_price": result.exit_price,
                "gross_return": result.gross_return,
                "holding_days": result.holding_days,
                "exit_reason": result.exit_reason,
            }
        )

    return pd.DataFrame(records, index=pd.DatetimeIndex(index, name="decision_date"))


def main() -> None:
    parser = argparse.ArgumentParser(description="triple-barrier 端到端煙霧測試")
    parser.add_argument("--stocks", nargs="*", default=None, help="股票代號；預設取標的池前 10 檔")
    parser.add_argument("--horizon", type=int, default=5, help="時間柵（交易日）")
    parser.add_argument("--min-rr", type=float, default=2.0, help="R:R 門檻")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2026-09-11")
    args = parser.parse_args()

    print("=" * 84)
    print("端到端煙霧測試：柵欄推導 + triple-barrier 標記")
    print("=" * 84)
    print()

    print("─" * 84)
    print("上游資料覆蓋率")
    print("─" * 84)
    print(coverage_report().to_string(index=False))
    print()

    universe = load_universe(limit=150)
    print("─" * 84)
    print("標的池")
    print("─" * 84)
    print(f"  可用 {len(universe.stocks)} 檔")
    for note in universe.warning.describe():
        print(f"  ⚠️  {note}")
    print()

    stock_ids = args.stocks or universe.stock_ids[:10]
    prices = load_prices(
        stock_ids,
        start=date.fromisoformat(args.start),
        end=date.fromisoformat(args.end),
        adjusted=True,
    )

    print("─" * 84)
    print(f"標記結果（horizon={args.horizon} 交易日、R:R 門檻 {args.min_rr}）")
    print("─" * 84)
    header = (
        f"{'標的':<8}{'可標記':>7}{'+1':>6}{'0':>6}{'-1':>6}"
        f"{'勝率':>8}{'平均目標':>10}{'平均停損':>10}{'平均R:R':>9}{'平均持有':>9}"
    )
    print(header)

    summaries: list[pd.DataFrame] = []
    for stock_id in stock_ids:
        if stock_id not in prices.index.get_level_values("stock_id"):
            print(f"{stock_id:<8}  無資料")
            continue

        bars = prices.xs(stock_id, level="stock_id")
        labeled = label_with_derived_width(bars, args.horizon, args.min_rr)

        if labeled.empty:
            print(f"{stock_id:<8}{0:>7}   （無任何決策日通過 R:R 門檻）")
            continue

        counts = labeled["label"].value_counts()
        n_up = int(counts.get(1, 0))
        n_flat = int(counts.get(0, 0))
        n_down = int(counts.get(-1, 0))
        win_rate = 100 * n_up / (n_up + n_down) if (n_up + n_down) else 0.0

        print(
            f"{stock_id:<8}{len(labeled):>7}{n_up:>6}{n_flat:>6}{n_down:>6}"
            f"{win_rate:>7.1f}%{labeled['target_pct'].mean() * 100:>9.2f}%"
            f"{labeled['stop_pct'].mean() * 100:>9.2f}%"
            f"{labeled['risk_reward'].mean():>9.2f}"
            f"{labeled['holding_days'].mean():>9.1f}"
        )
        summaries.append(labeled.assign(stock_id=stock_id))

    if not summaries:
        print("\n沒有任何標記結果，無法彙總。")
        return

    combined = pd.concat(summaries)
    print()

    print("─" * 84)
    print("全體彙總")
    print("─" * 84)
    counts = combined["label"].value_counts().sort_index()
    total = len(combined)
    for label_value, count in counts.items():
        name = {1: "+1 觸目標", 0: " 0 到期", -1: "-1 觸停損"}[int(label_value)]
        print(f"  {name}   {count:>6}  ({count / total * 100:>5.1f}%)")
    print(f"  合計     {total:>6}")
    print()

    reasons = combined["exit_reason"].value_counts()
    print("  出場原因：", {str(k): int(v) for k, v in reasons.items()})
    print()

    n_up = int(counts.get(1, 0))
    n_down = int(counts.get(-1, 0))
    base_rate = n_up / (n_up + n_down) if (n_up + n_down) else 0.0
    print(f"  基礎勝率（排除到期）  {base_rate * 100:.1f}%")
    print(f"  平均毛報酬            {combined['gross_return'].mean() * 100:+.3f}%")
    print(f"  平均持有天數          {combined['holding_days'].mean():.2f}")
    print()

    print("─" * 84)
    print("成本檢查（CLAUDE.md 規格 13：換手率是一級輸出）")
    print("─" * 84)
    avg_hold = combined["holding_days"].mean()
    # 週頻決策、平均持有 avg_hold 天 → 每週約換 5/avg_hold 次完整部位
    weekly_turnover = 5.0 / avg_hold if avg_hold > 0 else 0.0
    drag = annual_cost_drag(weekly_turnover, DEFAULT, Tier.LARGE)
    round_trip = DEFAULT.round_trip_rate(Tier.LARGE)

    print(f"  一趟來回成本率        {round_trip * 100:.3f}%")
    print(f"  平均持有天數          {avg_hold:.2f}")
    print(f"  隱含週換手率          {weekly_turnover * 100:.1f}%")
    print(f"  年化成本拖累          {drag * 100:.2f}%")
    print()
    print(f"  平均毛報酬 / 趟        {combined['gross_return'].mean() * 100:+.3f}%")
    print(f"  扣一趟成本後           {(combined['gross_return'].mean() - round_trip) * 100:+.3f}%")
    print()

    print("=" * 84)
    print("這個煙霧測試證明／不證明什麼")
    print("=" * 84)
    print("  ✓ 證明：資料管線、柵欄推導、標記邏輯在真實資料上可運作")
    print("  ✓ 證明：柵欄寬度來自 ATR 與歷史報酬分布，不是寫死的")
    print("  ✗ 不證明：任何預測能力。以上只是「標籤的歷史分布」，")
    print("           沒有模型、沒有樣本外、沒有對照組。")
    print("  ✗ 不構成投資建議。")
    print("=" * 84)


if __name__ == "__main__":
    main()
