"""
逐期橫斷面 Rank IC

## 為什麼需要這個模組

選股是**橫斷面**任務：今天這 150 檔裡，誰會比較強。

先前用「全部樣本池在一起算一個 Spearman」判定訊號有沒有排序能力。
那是方法學錯誤，而且方向完全相反：

```
策略族      全域 Spearman   逐期 Rank IC   IC>0 的期數比例
動能突破      −0.0077        +0.0384        58.2%
籌碼跟隨      −0.0018        +0.0281        60.2%
均值回歸      −0.0132        −0.0369        39.1%
```

全域版把橫斷面排序與時序變異混在一起——在一段所有股票都漲 20% 的期間，
相關係數被「哪幾天報酬高」主導，而不是「那天哪幾檔排得好」。

**用錯指標的代價**：我跑了 7 次完整 OOS 回測（每次 25 分鐘），才發現
判定訊號的方式本身是錯的。這個工具幾分鐘就能給出同樣的答案。

## 三件必須做對的事

### 1. 用排名，不用數值

一檔 +500% 的極端報酬會主導 Pearson 相關係數。台股這 7 年有 40~135 倍
的 AI 股，用 Pearson 等於在量那幾檔。

### 2. 重疊必須修正

持有 60 日但每 5 日決策 → 連續 12 期的持有期重疊。用 512 期算 t 值會
高估 sqrt(12) ≈ 3.5 倍的顯著性。

    有效期數 = 期數 ÷ (持有期 ÷ 決策間隔)

### 3. 算不出來就回 None，不回 0

橫斷面太窄、或分數全部並列時，算出來的 IC 是雜訊。回 0 會被平均進去，
把真實的 IC 稀釋掉。

實測踩過：校準器只分 6 箱時，同一期有 10 檔並列同分。

## 判準

`|t| > 2` 視為顯著。但**這不是「能賺錢」的判準**——IC 顯著只代表
排序能力不是運氣，能不能覆蓋交易成本是另一回事（見
`validation/benchmarks.py` 與成本模型）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

DEFAULT_MIN_NAMES = 10
"""
單期至少要有幾檔才算。

太窄的橫斷面算出來的 Spearman 幾乎全是雜訊——5 檔的 IC 標準誤約 0.5。
"""

SIGNIFICANCE_T = 2.0
"""|t| 超過此值視為顯著"""


class RankICError(RuntimeError):
    """Rank IC 無法計算"""


def period_rank_ic(
    scores: np.ndarray,
    returns: np.ndarray,
    min_names: int = DEFAULT_MIN_NAMES,
) -> float | None:
    """
    單期的橫斷面 Rank IC（Spearman）。

    Args:
        scores: 該期各標的的分數
        returns: 對應的實際報酬
        min_names: 至少要有幾檔才算

    Returns:
        Rank IC；下列情況回 `None`（**不是 0**）：
          · 有效配對少於 `min_names`
          · 分數全部相同（沒有排序可言）
          · 報酬全部相同

    Raises:
        RankICError: 兩個陣列長度不同
    """
    scores = np.asarray(scores, dtype=float)
    returns = np.asarray(returns, dtype=float)
    if len(scores) != len(returns):
        raise RankICError(f"長度必須相同：{len(scores)} vs {len(returns)}")

    usable = np.isfinite(scores) & np.isfinite(returns)
    if usable.sum() < min_names:
        return None

    s, r = scores[usable], returns[usable]
    # 全部並列時 Spearman 回 NaN，而 NaN 平均進去會汙染整體
    if np.ptp(s) == 0 or np.ptp(r) == 0:
        return None

    ic, _ = stats.spearmanr(s, r)
    return float(ic) if np.isfinite(ic) else None


@dataclass(frozen=True)
class RankICReport:
    """跨期彙總"""

    mean_ic: float
    std_ic: float
    t_stat: float
    """已用有效期數修正的 t 值"""

    positive_rate: float
    n_periods: int
    skipped: int
    """因橫斷面過窄或全部並列而跳過的期數"""

    effective_periods: float
    """重疊修正後的有效期數"""

    horizon: int
    stride: int
    by_year: dict[int, float]
    """逐年平均 IC。單一年份撐起全部結果是常見的假訊號"""

    @property
    def icir(self) -> float:
        """IC 的資訊比率（平均 ÷ 標準差），未做重疊修正"""
        return self.mean_ic / self.std_ic if self.std_ic else 0.0

    @property
    def is_significant(self) -> bool:
        return abs(self.t_stat) >= SIGNIFICANCE_T

    def describe(self) -> str:
        verdict = "顯著" if self.is_significant else "**無法排除運氣**"
        lines = [
            "─" * 62,
            f"逐期橫斷面 Rank IC（持有 {self.horizon} 日，每 {self.stride} 日決策）",
            "─" * 62,
            f"  平均 Rank IC   {self.mean_ic:+.4f}",
            f"  IC 標準差      {self.std_ic:.4f}",
            f"  ICIR           {self.icir:+.3f}",
            f"  IC > 0 的期數  {self.positive_rate * 100:.1f}%",
            f"  期數           {self.n_periods}（跳過 {self.skipped}）",
            f"  有效期數       {self.effective_periods:.1f}"
            f"（重疊修正：每 {max(1, self.horizon // self.stride)} 期才獨立）",
            f"  t 值           {self.t_stat:+.2f}  → {verdict}",
            "",
            "  逐年平均 IC：",
        ]
        for year, value in sorted(self.by_year.items()):
            lines.append(f"    {year}  {value:+.4f}")
        lines.append("─" * 62)
        lines.append(
            "  ⓘ IC 顯著只代表排序能力不是運氣，"
            "能否覆蓋交易成本是另一回事。"
        )
        return "\n".join(lines)


def evaluate_rank_ic(
    observations: dict[pd.Timestamp, tuple[np.ndarray, np.ndarray]],
    horizon: int,
    stride: int,
    min_names: int = DEFAULT_MIN_NAMES,
) -> RankICReport:
    """
    彙總多期的 Rank IC。

    Args:
        observations: {決策日: (分數陣列, 報酬陣列)}
        horizon: 持有交易日數（用於重疊修正）
        stride: 決策間隔交易日數
        min_names: 單期至少幾檔

    Returns:
        RankICReport

    Raises:
        RankICError: 沒有任何一期可算

    **重疊修正是必要的。** 持有 60 日、每 5 日決策時，連續 12 期的
    持有期重疊，用原始期數算 t 值會高估約 3.5 倍。
    """
    if horizon < 1 or stride < 1:
        raise RankICError(f"horizon 與 stride 都必須為正：{horizon}, {stride}")

    ics: list[float] = []
    dates: list[pd.Timestamp] = []
    skipped = 0

    for day in sorted(observations):
        scores, returns = observations[day]
        ic = period_rank_ic(scores, returns, min_names=min_names)
        if ic is None:
            skipped += 1
            continue
        ics.append(ic)
        dates.append(day)

    if not ics:
        raise RankICError(
            f"沒有任何一期可算（全部 {skipped} 期都因橫斷面過窄或全部並列被跳過）"
        )

    series = pd.Series(ics, index=pd.DatetimeIndex(dates))
    mean_ic = float(series.mean())
    std_ic = float(series.std(ddof=1)) if len(series) > 1 else 0.0

    overlap = max(1, horizon // stride)
    effective = len(series) / overlap
    t_stat = (
        mean_ic / std_ic * math.sqrt(effective) if std_ic > 0 and effective > 0 else 0.0
    )

    return RankICReport(
        mean_ic=mean_ic,
        std_ic=std_ic,
        t_stat=float(t_stat),
        positive_rate=float((series > 0).mean()),
        n_periods=len(series),
        skipped=skipped,
        effective_periods=float(effective),
        horizon=horizon,
        stride=stride,
        by_year={int(y): float(v) for y, v in series.groupby(series.index.year).mean().items()},
    )
