"""
CPCV（combinatorial purged cross-validation）包裝層

## 為什麼需要這個模組

`03_待辦與改進方向.md` 開頭那張表是整份文件最重要的證據：

```
組合             v6         v7          差異
均值回歸 × 60    +954.87%   +123.93%   −831 pp
均值回歸 × 120   +294.62%   +802.42%   +508 pp
等權對照          +701.37%   +688.79%    −13 pp   ← 對照組只動 13 pp
```

v6 → v7 只補了 23 期缺漏的市值快照，**策略參數完全沒動**。

當時的結論「不是 bug，是路徑相依」是對的，但**無法量化**——單一
walk-forward 只給一條路徑，你無從知道那條路徑有多大代表性。

CPCV 用同一份資料生出 C(n_groups, n_test_groups) 條路徑，把「重跑一次
結果就變」從缺陷變成可量測的統計量。本專案規模（370 個決策期）下
預設得到 **15 條路徑**。

## 這層負責什麼

上游 `external.crossvalidation` 吃的是「樣本數 + 標籤結束索引」，
本專案手上的是「決策日清單 + 持有交易日數 + 決策間隔」。

```
horizon=60 交易日、stride=5 交易日
    → ceil(60/5) = 12 個決策期
    → label_end_indices[i] = min(i + 12, n - 1)
```

翻譯錯了**不會拋錯**，只會讓 purge 少剔除幾期，結果變好看。
所以 `label_end_indices` 有手算值測試。

## purge 是雙向的

CPCV 的訓練資料分布在測試塊的**兩側**，所以剔除也是雙向：

```
索引        0     1     2     3    [4     5]    6     7     8     9
標籤區間  [0,2] [1,3] [2,4] [3,5] [4,6] [5,7] [6,8] [7,9] [8,9] [9,9]
                      剔除  剔除   ← 測試 →   剔除  剔除  保留  保留
```

i=6、i=7 被剔除，是因為測試樣本 5 的標籤要到 index 7 才揭曉——那段
期間的資料還「屬於」測試集。

本專案的 `walk_forward.py` 只做單向隔離，那是對的：walk-forward 的訓練
集永遠在測試段之前，向後那一邊根本不存在。**兩邊語意不同不是矛盾，
是適用場景不同。**

## ⚠️ 15 條路徑不是 15 個獨立樣本

它們共用同一份歷史，只是切法不同。CPCV 回答的是「這條策略對切分方式
有多敏感」，不是「多了 14 倍證據」。

**用它否定（變異太大 → 結論不可靠）比用它肯定可靠得多。**
`PathDistribution.describe()` 會把這句話印進報告，不要拿掉。

## 這不是 walk-forward 的替代品

CPCV 的訓練集會用到測試段**之後**的資料，所以它不能拿來宣稱「實盤會
這樣運作」（那違反禁令 5）。它量的是估計量的穩定度，實盤模擬仍然要用
`walk_forward.py`。兩個一起看，不是二選一。
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from taiwan_quant.validation.external.crossvalidation import (
    combinatorial_purged_splits,
)

DEFAULT_GROUPS = 6
DEFAULT_TEST_GROUPS = 2
"""C(6,2) = 15 條路徑。組數再多，每塊的測試期會短到算不出有意義的報酬"""


class CPCVError(RuntimeError):
    """切分參數不合法"""


@dataclass(frozen=True)
class PathDistribution:
    """一組 CPCV 路徑的報酬分布"""

    returns: tuple[float, ...]
    n_paths: int
    median: float
    p05: float
    p95: float
    spread: float
    """全距（最大 − 最小）。任務 E 的驗收數字就是它"""

    def describe(self) -> str:
        """報告用的文字，含「不是獨立樣本」的警語"""
        return (
            f"CPCV 路徑分布（{self.n_paths} 條）\n"
            f"  中位數  {self.median * 100:+8.2f}%\n"
            f"  5% 分位 {self.p05 * 100:+8.2f}%\n"
            f"  95% 分位{self.p95 * 100:+8.2f}%\n"
            f"  全距    {self.spread * 100:8.2f} pp\n"
            f"\n"
            f"⚠️  這 {self.n_paths} 條路徑共用同一份歷史，只是切法不同，\n"
            f"    **不是 {self.n_paths} 個獨立樣本**。它量的是「對切分方式有多敏感」，\n"
            f"    用來否定（變異太大 → 結論不可靠）比用來肯定可靠得多。"
        )


def label_end_indices(n_periods: int, horizon: int, stride: int) -> np.ndarray:
    """
    每個決策期的標籤在第幾期揭曉。

    Args:
        n_periods: 決策期數
        horizon: 持有交易日數
        stride: 決策間隔交易日數

    Returns:
        長度 `n_periods` 的索引陣列，超出尾端的夾到最後一期

    Raises:
        CPCVError: 任一參數非正

    **無條件進位**，與 `walk_forward.embargo_periods` 同一條理由：
    持有 22 日、每 5 日決策時 22/5 = 4.4，第 5 期的標籤仍有一部分沒揭曉。
    """
    if n_periods < 1 or horizon < 1 or stride < 1:
        raise CPCVError(
            f"n_periods、horizon、stride 都必須為正，"
            f"得到 {n_periods}, {horizon}, {stride}"
        )
    gap = math.ceil(horizon / stride)
    return np.minimum(np.arange(n_periods) + gap, n_periods - 1)


def cpcv_folds(
    decision_dates: Sequence[pd.Timestamp],
    horizon: int,
    stride: int,
    n_groups: int = DEFAULT_GROUPS,
    n_test_groups: int = DEFAULT_TEST_GROUPS,
) -> Iterator[tuple[list[pd.Timestamp], list[pd.Timestamp]]]:
    """
    產生 C(n_groups, n_test_groups) 組 (訓練日, 測試日)。

    Args:
        decision_dates: 全部決策日（升冪）
        horizon: 持有交易日數
        stride: 決策間隔交易日數
        n_groups: 切成幾塊
        n_test_groups: 每次held out 幾塊

    Yields:
        (訓練日清單, 測試日清單)——回傳**日期**不是索引

    Raises:
        CPCVError: 日期未升冪、期數不足以分組，或參數不合法

    ⚠️ 訓練集會包含測試段**之後**的決策日。這對估計穩定度是對的，
    但不能拿來宣稱實盤表現（禁令 5）——那要用 `walk_forward_folds`。
    """
    days = list(decision_dates)
    if any(b < a for a, b in zip(days, days[1:], strict=False)):
        raise CPCVError("決策日必須依日期升冪排序")

    label_ends = label_end_indices(len(days), horizon, stride)
    try:
        splits = combinatorial_purged_splits(
            len(days), label_ends, n_groups=n_groups, n_test_groups=n_test_groups
        )
        for split in splits:
            yield [days[i] for i in split.train], [days[i] for i in split.test]
    except ValueError as exc:
        raise CPCVError(f"CPCV 切分失敗：{exc}") from exc


def summarize_paths(returns: Sequence[float]) -> PathDistribution:
    """
    把各路徑的報酬整理成分布。

    Args:
        returns: 每條路徑的總報酬（小數，0.15 = +15%）

    Returns:
        `PathDistribution`

    Raises:
        CPCVError: 沒有任何路徑

    百分位用 numpy 的線性內插。路徑數少（預設 15）時分位數本來就粗，
    **全距比分位數更該看**——它不受內插方式影響。
    """
    values = np.asarray(list(returns), dtype=float)
    if values.size == 0:
        raise CPCVError("至少要有一條路徑才能算分布")

    return PathDistribution(
        returns=tuple(float(v) for v in values),
        n_paths=int(values.size),
        median=float(np.median(values)),
        p05=float(np.percentile(values, 5)),
        p95=float(np.percentile(values, 95)),
        spread=float(values.max() - values.min()),
    )
