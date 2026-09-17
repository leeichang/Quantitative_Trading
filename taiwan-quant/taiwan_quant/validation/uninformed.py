"""
無資訊對照組：與策略同結構、但與標籤無關

## 為什麼需要這個

CLAUDE.md 原本的必跑對照組是「隨機進場（100 次模擬取中位數）」。
2026-09-17 實測證明**那個門檻對任何基於因子的策略都太低**：

```
真模型      淨/趟 +3.64%
標籤打亂    淨/趟 +2.94%    ← 保留 81%
隨機 10 檔  淨/趟 +1.05%

打亂 − 隨機 = +1.88%／趟   SE 0.81%   t = 2.32   顯著較好
```

一個證明沒有任何預測資訊的模型，顯著打敗隨機選股。

原因：**用任意函數取前 N 檔 ≠ 均勻隨機抽 N 檔。** 任意函數會繼承一個
因子傾斜，而因子傾斜本身就賺錢。

```
持有全池              2.45%
隨機 10 檔（中位）     2.14%
任意函數取前 10        3.95%    ← +1.50 pp 只來自「集中選股」
```

## 兩種策略類型，兩種對照組

```
模型類   打亂標籤後重訓            models/lgbm_baseline.train_and_predict(shuffle_seed=...)
分數類   同一特徵集的隨機權重組合   本模組
```

## 為什麼是「隨機權重」而不是「隨機分數」

手工分數的形式是固定權重的特徵混合：

```python
_blend(parts=[_squash(row["momentum_20"], ...), ...], weights=[2.0, 1.5, ...])
```

所以對照組要保留**同一個形式**，只把手挑的權重換成任意權重。
隨機分數（每檔亂給一個數）退化成隨機選股，那正是被證明太弱的門檻。

## ⚠️ 權重在時間上必須固定

一次抽樣 = 一組權重，套用到**所有**決策日。

手工分數的權重不隨時間變，所以對照組也不能變。每期重抽權重會讓選股
在時間上失去持續性，退化回隨機選股——又變成太弱的門檻。

`test_weights_are_constant_across_dates` 釘住這件事。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


class UninformedError(RuntimeError):
    """對照組輸入不合法"""


DEFAULT_DRAWS = 200
"""抽樣次數。與 CLAUDE.md 隨機對照組的 100 次同量級，多一點讓分位穩定"""


@dataclass(frozen=True)
class NullDistribution:
    """
    一組抽樣的結果分布。

    ⚠️ 這些抽樣**不是獨立策略**，它們共用同一份歷史與同一組特徵。
    用它算分位（「真策略落在第幾百分位」）是對的；用它算「有幾組賺錢」
    然後宣稱統計顯著是錯的。
    """

    draws: tuple[float, ...]
    """每次抽樣的淨報酬／趟"""

    observed: float
    """真策略的淨報酬／趟"""

    def percentile_of_observed(self) -> float:
        """真策略落在抽樣分布的第幾百分位（0~1）"""
        if not self.draws:
            raise UninformedError("沒有抽樣結果")
        values = np.asarray(self.draws, dtype=float)
        return float((values < self.observed).mean())

    def describe(self) -> list[str]:
        """人可讀的摘要，供報告直接引用"""
        values = np.asarray(self.draws, dtype=float)
        return [
            f"真策略 {self.observed:+.2%}／趟"
            f"｜落在隨機權重分布的第 {self.percentile_of_observed():.0%} 百分位",
            f"抽樣 {len(values)} 次：中位 {np.median(values):+.2%}"
            f"｜5% 分位 {np.quantile(values, 0.05):+.2%}"
            f"｜95% 分位 {np.quantile(values, 0.95):+.2%}",
            "⚠️ 抽樣共用同一份歷史，不是獨立策略。用它算分位可以，"
            "用它宣稱顯著不行。",
        ]


def cross_sectional_ranks(features: pd.DataFrame) -> pd.DataFrame:
    """
    每個特徵在當日候選之內的分位（0~1）。

    Args:
        features: 單一決策日的特徵表（index 為 stock_id，column 為特徵）

    Returns:
        同形狀的分位表。全缺值的特徵欄回 NaN

    Raises:
        UninformedError: 候選少於 2 檔

    取分位是為了**尺度無關**：`momentum_120` 與 `rsi_14` 的量級差兩個
    數量級，直接加權會讓量級大的那個主導，而那是量級的效果不是權重的。
    """
    if len(features) < 2:
        raise UninformedError(f"候選只有 {len(features)} 檔，無從排序")
    return features.rank(pct=True)


def random_weight_scores(
    ranks: pd.DataFrame, weights: np.ndarray
) -> pd.Series:
    """
    分位表 × 權重 → 分數。

    Args:
        ranks: `cross_sectional_ranks` 的輸出
        weights: 與 `ranks` 欄數等長的權重

    Returns:
        每檔的分數。全特徵皆缺的標的為 NaN

    Raises:
        UninformedError: 權重長度不符

    缺值以當日分位中位（0.5）代入——**不是 0**。0 代表「該特徵最低」，
    那會讓缺籌碼資料的標的被系統性壓低，變成一個沒宣告的篩選條件。
    """
    if len(weights) != ranks.shape[1]:
        raise UninformedError(
            f"權重長度 {len(weights)} 與特徵數 {ranks.shape[1]} 不符"
        )
    filled = ranks.fillna(0.5)
    scored = filled.to_numpy(dtype="float64") @ np.asarray(weights, dtype="float64")
    result = pd.Series(scored, index=ranks.index)
    return result.where(ranks.notna().any(axis=1))


def draw_weights(
    n_features: int,
    n_draws: int = DEFAULT_DRAWS,
    seed: int = 20260917,
    sparsity: int | None = None,
) -> np.ndarray:
    """
    抽 `n_draws` 組權重，每組長度 `n_features`。

    Args:
        n_features: 特徵數
        n_draws: 抽樣次數
        seed: 隨機種子（禁令 7：必須可重現）
        sparsity: 每組只有這麼多個非零權重；`None` 代表全部非零

    Returns:
        shape `(n_draws, n_features)` 的權重矩陣

    Raises:
        UninformedError: 參數非正，或 `sparsity` 超過 `n_features`

    標準常態分布：有正有負，所以對照組不預設「特徵越高越好」。
    手挑權重全為正（`_blend` 的 weights），但方向是靠 `_squash(-x)`
    內建的——**手挑同時選了特徵和方向**，所以對照組必須允許雙向，
    否則它連「RSI 越低越好」都表達不出來，那是被綁住手的對照組。

    ## ⚠️ 稀疏度必須與被比較的策略一致

    實測（`reports/hand_vs_uninformed_dev.json` 第一版）：32 個特徵的
    密集隨機權重，其淨/趟中位只有 +1.45%，僅比純隨機選股（+1.05%）
    高 0.4 pp。而標籤打亂的 LightGBM 是 +2.94%。

    **兩個虛無假設不等價。** 原因是 LightGBM 的樹只在少數特徵上分裂
    （稀疏、集中），而 32 個特徵的隨機線性組合會互相抵銷、接近雜訊。

    手工分數也是稀疏的：

    ```
    動能突破    4 個特徵   weights [3.0, 2.0, 2.0, 1.0]
    籌碼跟隨    6 個特徵
    均值回歸    4 個特徵   weights [2.5, 2.0, 1.5, 3.0]
    ```

    所以比較某個族時，`sparsity` 要設成**那個族自己用的特徵數**。
    用密集對照組會得到一個過鬆的門檻——正是這整條規格要防的事。

    **每一列是一組固定權重，套用到所有決策日。** 見模組說明。
    """
    if n_features < 1:
        raise UninformedError(f"n_features 至少為 1，得到 {n_features}")
    if n_draws < 1:
        raise UninformedError(f"n_draws 至少為 1，得到 {n_draws}")
    rng = np.random.default_rng(seed)
    if sparsity is None:
        return rng.normal(0.0, 1.0, size=(n_draws, n_features))
    if not 1 <= sparsity <= n_features:
        raise UninformedError(
            f"sparsity 必須落在 1~{n_features}，得到 {sparsity}"
        )
    weights = np.zeros((n_draws, n_features))
    for row in range(n_draws):
        chosen = rng.choice(n_features, size=sparsity, replace=False)
        weights[row, chosen] = rng.normal(0.0, 1.0, size=sparsity)
    return weights
