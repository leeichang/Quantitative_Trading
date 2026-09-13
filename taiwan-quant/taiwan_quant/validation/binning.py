"""
分箱規則（校準層共用）

## 為什麼要抽出來

系統有兩種校準器：

    Calibrator         分數 → P(+1)      （固定目標的 triple-barrier 用）
    ReturnCalibrator   分數 → E[報酬]    （無固定目標的移動停損用）

兩者的箱界規則必須**完全一致**，否則同一個分數在兩邊會查到不同的箱，
比較結果就不可信。與其寫兩份再祈禱它們同步，不如只寫一份。

## 箱界規則

等寬分箱，左閉右開，**最後一箱含右端點**：

    [lo, hi)  [lo, hi)  ...  [lo, hi]

等頻分箱會讓箱界隨樣本浮動，不易解讀也不易跨期比對，所以不用。
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Bin(Protocol):
    """任何帶箱界與樣本數的校準箱"""

    lo: float
    hi: float
    n_samples: int


def bin_edges(score_min: float, score_max: float, n_bins: int) -> np.ndarray:
    """等寬箱界，長度為 n_bins + 1"""
    return np.linspace(score_min, score_max, n_bins + 1)


def bin_mask(
    scores: np.ndarray, edges: np.ndarray, index: int, n_bins: int
) -> np.ndarray:
    """
    第 `index` 箱的布林遮罩。

    最後一箱含右端點，否則分數等於 score_max 的樣本會被整個排除掉。
    """
    lo, hi = float(edges[index]), float(edges[index + 1])
    upper = scores <= hi if index == n_bins - 1 else scores < hi
    return (scores >= lo) & upper


def find_bin(score: float, bins: tuple[Bin, ...]) -> int | None:
    """
    找出分數落在哪一箱。

    **規則必須與 `bin_mask` 相同**，否則 predict 會查到與擬合時不同的箱。

    所有分數相同（箱界退化為 lo == hi）時，回傳最後一個箱——
    此時每一箱的條件都成立，取最後一箱與 `bin_mask` 的累積結果一致。
    """
    for i, bucket in enumerate(bins):
        is_last = i == len(bins) - 1
        in_bin = (
            bucket.lo <= score <= bucket.hi
            if is_last
            else bucket.lo <= score < bucket.hi
        )
        if in_bin and (bucket.n_samples > 0 or is_last):
            return i
    return None
