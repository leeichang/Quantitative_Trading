"""
量測一個微小擾動會傳播多遠

## 為什麼需要這個

`03_待辦與改進方向.md` 開頭記了主線最有力的一個證據：v6 → v7 只補了
23 期缺漏的市值快照，**策略參數、門檻、槽位數完全沒動**，結果

```
均值回歸 × 60      +954.87% → +123.93%    −831 pp
均值回歸 × 120     +294.62% → +802.42%    +508 pp
等權對照           +701.37% → +688.79%     −13 pp
```

對照組只動 13 pp，策略擺盪數百 pp。待辦把原因寫成**路徑相依**：
60 日持有下 99.8% 的訊號被槽位擋掉，實際成交哪一百筆取決於
「槽位空出來時誰在排隊」。

**那是一個機制假說，從來沒有被直接量測。** 這個模組量它。

## 量什麼

給同一組訊號做一次最小擾動（例如移掉第一個決策日的一檔），比對兩次
模擬**實際成交的 (代號, 進場日) 集合**：

```
shared / only_baseline / only_perturbed      集合差異
first_divergence                             第一次分歧的日期
last_divergence                              最後一次分歧的日期
```

`last_divergence` 是關鍵。槽位排隊下，一次擾動會改變後續每個槽位的
釋放時點，**分歧一路傳到序列末端**。定期換倉下每個節點獨立選股，
分歧應該只出現在被擾動的那一期。

## 這個模組不做什麼

不判斷哪個方案更好。它只回答「擾動傳播多遠」——報酬高低是另一件事，
而路徑相依會讓報酬高低本身變成抽樣結果。
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd

Fill = tuple[str, pd.Timestamp]
"""一筆實際成交：(代號, 進場日)。出場日由持有期決定，不是自由度"""


@dataclass(frozen=True)
class DivergenceResult:
    """兩次模擬之間，實際成交集合的差異。"""

    shared: int
    only_baseline: int
    only_perturbed: int
    first_divergence: pd.Timestamp | None
    last_divergence: pd.Timestamp | None

    @property
    def baseline_fills(self) -> int:
        return self.shared + self.only_baseline

    @property
    def perturbed_fills(self) -> int:
        return self.shared + self.only_perturbed

    @property
    def jaccard(self) -> float:
        """交集 ÷ 聯集。1.0 代表完全相同，0.0 代表毫無重疊"""
        union = self.shared + self.only_baseline + self.only_perturbed
        if union == 0:
            return 1.0
        return self.shared / union

    @property
    def diverged(self) -> bool:
        return self.only_baseline > 0 or self.only_perturbed > 0

    def describe(self) -> str:
        if not self.diverged:
            return f"完全相同（{self.shared} 筆成交）"
        span = (
            f"{self.first_divergence.date()} ~ {self.last_divergence.date()}"
            if self.first_divergence is not None
            and self.last_divergence is not None
            else "—"
        )
        return (
            f"重疊 {self.jaccard:.1%}"
            f"｜共有 {self.shared}"
            f"｜僅基準 {self.only_baseline}"
            f"｜僅擾動 {self.only_perturbed}"
            f"｜分歧區間 {span}"
        )


def fill_signature(trades: Iterable[object]) -> frozenset[Fill]:
    """
    把成交紀錄縮成 (代號, 進場日) 的集合。

    只取這兩個欄位是刻意的：報酬與成本由標記決定，**同一檔同一天進場
    就是同一筆交易**。把報酬納入比對會讓價格資料的微小差異看起來像
    路徑分歧。

    接受任何有 `stock_id` 與 `entry_date` 屬性的物件
    （`portfolio_sim.ClosedTrade` 符合）。
    """
    return frozenset(
        (str(trade.stock_id), pd.Timestamp(trade.entry_date))  # type: ignore[attr-defined]
        for trade in trades
    )


def divergence(
    baseline: Iterable[object],
    perturbed: Iterable[object],
) -> DivergenceResult:
    """
    比對兩次模擬的成交集合。

    `first_divergence` / `last_divergence` 取自**只出現在其中一邊**的
    成交日期。兩邊完全相同時皆為 `None`。
    """
    left = fill_signature(baseline)
    right = fill_signature(perturbed)

    only_left = left - right
    only_right = right - left
    differing = sorted(day for _, day in (only_left | only_right))

    return DivergenceResult(
        shared=len(left & right),
        only_baseline=len(only_left),
        only_perturbed=len(only_right),
        first_divergence=differing[0] if differing else None,
        last_divergence=differing[-1] if differing else None,
    )


def propagation_ratio(
    result: DivergenceResult,
    perturbation_date: pd.Timestamp,
    calendar: list[pd.Timestamp],
) -> float:
    """
    分歧涵蓋了擾動之後多少比例的時間軸。

    `1.0` 代表一路傳到末端（槽位排隊的預期行為），接近 `0.0` 代表
    擾動被關在原地（定期換倉的預期行為）。

    Args:
        result: `divergence` 的輸出
        perturbation_date: 擾動施加的決策日
        calendar: 升冪交易日曆

    Returns:
        [0, 1] 的比例。沒有分歧時回 0.0。

    Raises:
        ValueError: 日曆為空，或擾動日不在日曆內
    """
    if not calendar:
        raise ValueError("交易日曆不可為空")
    if perturbation_date not in calendar:
        raise ValueError(f"擾動日 {perturbation_date} 不在交易日曆內")
    if result.last_divergence is None:
        return 0.0

    start = calendar.index(perturbation_date)
    remaining = len(calendar) - 1 - start
    if remaining <= 0:
        return 0.0

    reach = calendar.index(result.last_divergence) - start
    return max(0.0, min(1.0, reach / remaining))
