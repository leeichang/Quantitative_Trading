#!/usr/bin/env python3
"""
柵欄參數可行性診斷

**這不是效能最佳化，不可用來選「最賺錢」的參數。**

用途：回答一個純粹的可行性問題——
    「在這組參數下，triple-barrier 標記能不能產出可用的訓練集？」

判準（與報酬完全無關）：
    1. 覆蓋率    有多少比例的決策日通過 R:R 門檻？太低 → 樣本不足
    2. 類別分布  +1 / 0 / −1 三類是否都有足夠樣本？
                 某類為 0 → 模型會退化成常數預測
    3. 到期占比  「0（時間柵到期）」占比過高 → 柵欄相對 horizon 太寬，
                 標籤幾乎不帶資訊

為什麼要診斷：smoke_label.py 實測發現 D7 原始規格
（horizon=5、R:R>=2.0、atr_multiple=1.5、target_quantile=0.7）
在真實資料上只有 0.4% 覆蓋率且 **+1 類別為 0**，規格內部矛盾。

多重測試警告（CLAUDE.md 多重測試校正）：
    本網格只看「標籤分布」，不看任何報酬或績效指標。
    選定參數後的策略績效仍須走完整 Walk-Forward + OOS + PBO，
    不可因為這裡掃過參數就跳過校正。

用法：
    .venv/bin/python scripts/diagnose_barrier_params.py
    .venv/bin/python scripts/diagnose_barrier_params.py --stocks 2330 2317 2454 --top 20
"""

from __future__ import annotations

import argparse
import itertools
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from taiwan_quant.data.loader import load_prices, load_universe  # noqa: E402
from taiwan_quant.labeling.barrier_width import derive_width  # noqa: E402
from taiwan_quant.labeling.triple_barrier import label_one  # noqa: E402

MIN_CLASS_SAMPLES = 30
"""每類最少樣本數。低於此值訓練會不穩定"""

MIN_COVERAGE = 0.10
"""最低覆蓋率。低於 10% 代表門檻把絕大多數決策日擋掉了"""

MAX_TIME_BARRIER_SHARE = 0.85
"""到期占比上限。超過代表柵欄相對 horizon 太寬，標籤資訊量低"""


@dataclass(frozen=True)
class ParamGrid:
    """待掃描的參數組合"""

    horizon: int
    atr_multiple: float
    target_quantile: float
    min_risk_reward: float

    def label(self) -> str:
        return (
            f"h={self.horizon} atr×{self.atr_multiple} "
            f"q={self.target_quantile} rr>={self.min_risk_reward}"
        )


@dataclass(frozen=True)
class GridResult:
    """單一參數組合的標籤分布"""

    params: ParamGrid
    decision_days: int
    labeled: int
    n_up: int
    n_flat: int
    n_down: int
    avg_target_pct: float
    avg_stop_pct: float
    avg_holding_days: float

    @property
    def coverage(self) -> float:
        return self.labeled / self.decision_days if self.decision_days else 0.0

    @property
    def time_barrier_share(self) -> float:
        return self.n_flat / self.labeled if self.labeled else 0.0

    @property
    def min_class_size(self) -> int:
        return min(self.n_up, self.n_flat, self.n_down)

    @property
    def is_usable(self) -> bool:
        """三個可行性判準全部通過"""
        return (
            self.coverage >= MIN_COVERAGE
            and self.min_class_size >= MIN_CLASS_SAMPLES
            and self.time_barrier_share <= MAX_TIME_BARRIER_SHARE
        )

    def reasons(self) -> list[str]:
        """不可用的原因"""
        notes: list[str] = []
        if self.coverage < MIN_COVERAGE:
            notes.append(f"覆蓋率 {self.coverage * 100:.1f}% < {MIN_COVERAGE * 100:.0f}%")
        if self.min_class_size < MIN_CLASS_SAMPLES:
            notes.append(
                f"最小類別 {self.min_class_size} < {MIN_CLASS_SAMPLES}"
                f"（+1={self.n_up} 0={self.n_flat} −1={self.n_down}）"
            )
        if self.time_barrier_share > MAX_TIME_BARRIER_SHARE:
            notes.append(f"到期占比 {self.time_barrier_share * 100:.0f}% 過高")
        return notes


def evaluate(bars_by_stock: dict[str, pd.DataFrame], params: ParamGrid) -> GridResult:
    """對一組參數統計全體標籤分布"""
    decision_days = 0
    n_up = n_flat = n_down = 0
    targets: list[float] = []
    stops: list[float] = []
    holds: list[int] = []

    for bars in bars_by_stock.values():
        decision_days += len(bars)
        for decision_idx in range(len(bars)):
            width = derive_width(
                bars,
                decision_idx=decision_idx,
                horizon=params.horizon,
                atr_multiple=params.atr_multiple,
                target_quantile=params.target_quantile,
                min_risk_reward=params.min_risk_reward,
            )
            if width is None:
                continue
            result = label_one(
                bars,
                decision_idx=decision_idx,
                target_pct=width.target_pct,
                stop_pct=width.stop_pct,
                horizon=params.horizon,
            )
            if result is None:
                continue

            targets.append(width.target_pct)
            stops.append(width.stop_pct)
            holds.append(result.holding_days)
            if result.label == 1:
                n_up += 1
            elif result.label == 0:
                n_flat += 1
            else:
                n_down += 1

    labeled = n_up + n_flat + n_down
    return GridResult(
        params=params,
        decision_days=decision_days,
        labeled=labeled,
        n_up=n_up,
        n_flat=n_flat,
        n_down=n_down,
        avg_target_pct=sum(targets) / len(targets) if targets else float("nan"),
        avg_stop_pct=sum(stops) / len(stops) if stops else float("nan"),
        avg_holding_days=sum(holds) / len(holds) if holds else float("nan"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="柵欄參數可行性診斷")
    parser.add_argument("--stocks", nargs="*", default=None, help="預設取標的池前 10 檔")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2026-09-11")
    parser.add_argument("--top", type=int, default=15, help="列出前幾組")
    args = parser.parse_args()

    universe = load_universe(limit=150)
    stock_ids = args.stocks or universe.stock_ids[:10]

    prices = load_prices(
        stock_ids,
        start=date.fromisoformat(args.start),
        end=date.fromisoformat(args.end),
        adjusted=True,
    )
    bars_by_stock = {
        stock_id: prices.xs(stock_id, level="stock_id")
        for stock_id in stock_ids
        if stock_id in prices.index.get_level_values("stock_id")
    }

    grids = [
        ParamGrid(h, m, q, rr)
        for h, m, q, rr in itertools.product(
            (5, 10),                    # horizon：D7 要求 5 日；10 日作對照
            (0.8, 1.0, 1.5),            # atr_multiple：停損寬度
            (0.60, 0.70, 0.85),         # target_quantile：目標分位數
            (1.0, 1.5, 2.0),            # min_risk_reward：R:R 門檻
        )
    ]

    print("=" * 104)
    print("柵欄參數可行性診斷")
    print("=" * 104)
    print(f"標的 {len(bars_by_stock)} 檔｜期間 {args.start} ~ {args.end}")
    print(f"掃描 {len(grids)} 組參數")
    print()
    print("判準（**與報酬完全無關**）：")
    print(f"  覆蓋率 >= {MIN_COVERAGE * 100:.0f}%｜每類 >= {MIN_CLASS_SAMPLES} 筆"
          f"｜到期占比 <= {MAX_TIME_BARRIER_SHARE * 100:.0f}%")
    print()

    results = [evaluate(bars_by_stock, g) for g in grids]
    results.sort(key=lambda r: (r.is_usable, r.min_class_size, r.coverage), reverse=True)

    header = (
        f"{'參數':<34}{'覆蓋率':>8}{'可標記':>8}"
        f"{'+1':>7}{'0':>7}{'-1':>7}{'到期%':>7}"
        f"{'目標':>8}{'停損':>8}{'可用':>6}"
    )
    print(header)
    print("─" * 104)

    for r in results[: args.top]:
        target = f"{r.avg_target_pct * 100:.2f}%" if r.labeled else "n/a"
        stop = f"{r.avg_stop_pct * 100:.2f}%" if r.labeled else "n/a"
        print(
            f"{r.params.label():<34}{r.coverage * 100:>7.1f}%{r.labeled:>8}"
            f"{r.n_up:>7}{r.n_flat:>7}{r.n_down:>7}"
            f"{r.time_barrier_share * 100:>6.0f}%{target:>8}{stop:>8}"
            f"{'✓' if r.is_usable else '✗':>6}"
        )

    usable = [r for r in results if r.is_usable]
    print()
    print("=" * 104)
    print(f"可用組合：{len(usable)} / {len(results)}")
    print("=" * 104)

    if usable:
        print()
        print("通過可行性判準的組合（**這不代表它們會賺錢**）：")
        for r in usable:
            print(
                f"  {r.params.label():<34}"
                f"覆蓋 {r.coverage * 100:>5.1f}%｜"
                f"+1/0/−1 = {r.n_up}/{r.n_flat}/{r.n_down}｜"
                f"目標 {r.avg_target_pct * 100:.2f}%｜停損 {r.avg_stop_pct * 100:.2f}%"
            )
    else:
        print()
        print("⚠️  沒有任何組合通過可行性判準。可能需要：")
        print("   · 放寬 horizon（D7 要求 5 日，但 5 日內不易觸及 ATR 導出的目標）")
        print("   · 降低 R:R 門檻（目前 2.0 在 5 日 horizon 下幾乎不可滿足）")
        print("   · 收緊停損（降低 atr_multiple）")

    # D7 原始規格的下場
    d7 = next(
        (
            r
            for r in results
            if r.params == ParamGrid(5, 1.5, 0.70, 2.0)
        ),
        None,
    )
    if d7:
        print()
        print("─" * 104)
        print("D7 原始規格（horizon=5、atr×1.5、q=0.70、R:R>=2.0）的實測下場")
        print("─" * 104)
        print(f"  覆蓋率      {d7.coverage * 100:.2f}%（{d7.labeled} / {d7.decision_days} 個決策日）")
        print(f"  類別分布    +1={d7.n_up}  0={d7.n_flat}  −1={d7.n_down}")
        for reason in d7.reasons():
            print(f"  ✗ {reason}")
        print()
        print("  結論：D7 的參數組合在真實資料上內部矛盾。")
        print("        ATR 導出的停損（約 3%）搭配 R:R>=2 會要求約 7% 的目標，")
        print("        而 5 個交易日內觸及 7% 的機率極低 → +1 類別幾乎為空。")

    print()
    print("=" * 104)
    print("⚠️  本診斷只看標籤分布，不看報酬。選定參數後的績效仍須走完整")
    print("    Walk-Forward + OOS + PBO 校正（CLAUDE.md 多重測試校正）。")
    print("    不構成投資建議。")
    print("=" * 104)


if __name__ == "__main__":
    main()
