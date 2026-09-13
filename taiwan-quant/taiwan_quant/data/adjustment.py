"""
還原股價計算（由官方除權息紀錄推算）

## 為什麼需要

CLAUDE.md 禁令 12：**一律用還原股價。**

原本的還原價來自 yfinance 的 Adj Close，但它**查不到已下市的股票**——
長歷史回補後有 107 檔沒有還原價。

而下市股票正是 survivorship 的關鍵樣本（2015 年標的池裡的日月光、
矽品都已下市）。因為拿不到還原價就把它們排除，等於把好不容易解掉的
survivorship bias 又放回來。

## 方法

FinMind 的 `TaiwanStockDividendResult` 免費層可用，且**對下市股票也有
紀錄**（實測 2311 日月光 4 筆、2325 矽品 3 筆）。它直接給：

```
before_price   除權息前收盤價
after_price    除權息參考價
```

比值就是還原因子：

    factor = after_price / before_price

## 還原方向

**除權息日之前**的價格要乘上因子，當天與之後不動：

```
adj_close(t) = close(t) × Π { factor(e) : e.date > t }
```

這是標準的向後還原（back-adjustment）——讓過去的價格降到與現在同一個
基準，所以最新一天的還原價等於原始價。方向反了會讓報酬率整段偏掉，
而曲線看起來仍然「正常」。

## 兩道邊界檢查

```
前收 / 參考價必須為正      除以 0 得到 inf，而 inf 不會讓比較拋錯
因子必須落在 [0.5, 1.5]   離 1 太遠代表資料有問題（例如兩欄來自不同檔）
```

放行任何一種，整段還原價會偏掉數倍而且沒有任何警訊。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

FACTOR_MIN = 0.5
FACTOR_MAX = 1.5
"""
還原因子的合理範圍。

台股單次除權息很少讓參考價低於前收的一半（那要配發超過 50% 的股息）。
超出範圍時拋錯而不是照用——資料錯誤比漏一次除息嚴重得多。
"""


class AdjustmentError(ValueError):
    """除權息資料不可用"""


@dataclass(frozen=True)
class DividendEvent:
    """一次除權息事件"""

    date: pd.Timestamp
    """除權息交易日。**這一天的價格已經是除息後的價格**"""

    before_price: float
    after_price: float

    def __post_init__(self) -> None:
        for label, value in (("before_price", self.before_price),
                             ("after_price", self.after_price)):
            if not math.isfinite(value) or value <= 0:
                raise AdjustmentError(
                    f"{label} 必須為正的有限值，得到 {value}。"
                    "除以 0 會得到 inf，而 inf 不會讓後續比較拋錯——"
                    "它會靜默汙染整段還原價。"
                )

        factor = self.after_price / self.before_price
        if not FACTOR_MIN <= factor <= FACTOR_MAX:
            raise AdjustmentError(
                f"還原因子 {factor:.4f} 超出合理範圍 "
                f"[{FACTOR_MIN}, {FACTOR_MAX}]"
                f"（前收 {self.before_price}、參考價 {self.after_price}）。"
                "多半是兩欄來自不同標的或不同日期。"
            )

    @property
    def factor(self) -> float:
        return self.after_price / self.before_price


def back_adjust(
    closes: pd.Series,
    events: list[DividendEvent],
) -> pd.Series:
    """
    由除權息事件算出還原收盤價。

    Args:
        closes: 原始收盤價序列（**不會被修改**），index 為日期
        events: 除權息事件（順序不拘）

    Returns:
        與 `closes` 同索引的還原價序列

    除權息日**當天**的價格已經是除息後的價格，所以只有嚴格早於
    事件日的價格需要乘上因子。
    """
    if not events:
        return closes.copy()

    adjusted = closes.astype(float).copy()
    for event in events:
        mask = adjusted.index < event.date
        if mask.any():
            adjusted.loc[mask] = adjusted.loc[mask] * event.factor
    return adjusted


def events_from_dividend_result(rows: list[dict]) -> list[DividendEvent]:
    """
    把 FinMind `TaiwanStockDividendResult` 的回應轉成事件清單。

    Args:
        rows: API 回傳的 `data` 陣列

    Returns:
        依日期升冪排序的事件

    無法解析或因子離譜的列**直接跳過**，不讓整檔失敗。但也不補值——
    補 `factor=1` 等於宣稱那次除息沒有發生，那比漏掉更糟。
    """
    events: list[DividendEvent] = []
    for row in rows:
        raw_date = str(row.get("date") or "").strip()
        if not raw_date:
            continue
        try:
            day = pd.Timestamp(raw_date)
        except (ValueError, TypeError):
            continue

        before, after = row.get("before_price"), row.get("after_price")
        if before is None or after is None:
            continue
        try:
            events.append(DividendEvent(day, float(before), float(after)))
        except (AdjustmentError, TypeError, ValueError):
            continue

    return sorted(events, key=lambda e: e.date)
