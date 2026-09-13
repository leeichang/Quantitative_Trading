"""
超額報酬標籤

## 為什麼要改標籤

原本的標籤是**絕對毛報酬**。在 2019-2026 這種等權 +689% 的多頭裡，
幾乎每一筆的標籤都是正的——模型學到的有一大部分是「市場會漲」，
不是「這檔比較強」。

實測佐證：三個策略族的前 3 名平均毛報酬 6.45%，但同期全市場平均
3.13%。**一半以上的報酬來自 beta，不是選股。**

選股是橫斷面任務，正確的目標是：

```
超額報酬 = 個股報酬 − 同期間基準報酬
```

## 三個實作決定

### 1. 用算術差，不用幾何差

下游的成本模型是算術扣除（`gross − round_trip_rate`）。標籤用幾何差
會讓門檻比較時混用兩種尺度，而且不會拋錯。

### 2. 基準按**每筆自己的進出場日**算

移動停損會讓出場日不同——同一個決策日進場的兩檔，可能一檔 20 天就
觸停損、一檔抱滿 60 天。用固定期間或當期平均會多一層雜訊。

### 3. 算不出基準就整筆回 `None`

不可退回絕對報酬。混用兩種標籤會讓校準器同時學到兩種尺度，
而且完全不會拋錯——這正是 CLAUDE.md 反覆強調的「無法計算」與
「算出來很差」必須區分。
"""

from __future__ import annotations

import math

import pandas as pd


class ExcessError(ValueError):
    """超額報酬無法計算"""


def benchmark_return(
    index: pd.Series,
    entry_date: pd.Timestamp,
    exit_date: pd.Timestamp,
) -> float | None:
    """
    基準在指定窗口的報酬。

    Args:
        index: 基準的權益曲線或價格序列（index 為日期，升冪）
        entry_date / exit_date: 該筆持倉的實際進出場日

    Returns:
        期間報酬率；下列情況回 `None`（**不外推**）：
          · 進場日早於序列起點
          · 出場日早於序列起點
          · 基準起始值非正

    Raises:
        ExcessError: 進場日晚於出場日

    進出場日不在序列裡時（假日、停牌），取**不晚於它**的最後一筆。
    取之後的第一筆等於用未來的值當成本。
    """
    if entry_date > exit_date:
        raise ExcessError(f"entry_date {entry_date} 不可晚於 exit_date {exit_date}")

    before_entry = index.loc[index.index <= entry_date]
    before_exit = index.loc[index.index <= exit_date]
    if before_entry.empty or before_exit.empty:
        return None

    base = float(before_entry.iloc[-1])
    final = float(before_exit.iloc[-1])
    if base <= 0 or not math.isfinite(base) or not math.isfinite(final):
        return None

    return final / base - 1.0


def to_excess_return(
    gross_return: float,
    index: pd.Series,
    entry_date: pd.Timestamp,
    exit_date: pd.Timestamp,
) -> float | None:
    """
    把絕對毛報酬換成相對基準的超額報酬。

    Args:
        gross_return: 該筆的絕對毛報酬
        index: 基準序列
        entry_date / exit_date: 該筆持倉的實際進出場日

    Returns:
        超額報酬（算術差）；基準算不出來時回 `None`

    Raises:
        ExcessError: `gross_return` 非有限值

    **多頭裡的正報酬可能是負超額**——賺 8% 但基準賺 20%，超額是 −12%。
    那筆交易其實輸給什麼都不做。這正是改標籤的目的。
    """
    if not math.isfinite(gross_return):
        raise ExcessError(
            f"gross_return 必須為有限值，得到 {gross_return}。"
            "NaN 不會讓後續比較拋錯，只會靜默汙染整組標籤。"
        )

    base = benchmark_return(index, entry_date, exit_date)
    if base is None:
        return None
    return gross_return - base
