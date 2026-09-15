"""
平手排序

## 抓到的 bug（第 12 個）

三個地方都用同一個排序鍵：

```python
sorted(candidates, key=lambda s: (-s.rank_score, s.stock_id))
#                                               ^^^^^^^^^^^ bug
```

實測（動能突破、開發集）：33,900 筆通過門檻的候選，**只有 12 個相異
`rank_score`**。

```
CALIBRATION_BINS = 6          ReturnCalibrator.predict() 回傳的是箱平均值
× 2 個流動性分層（成本不同）
= 12 個相異值
```

從 ~150 檔選 Top 3，第 3 名平均有 **3.8 檔同分**：

```
2019-06-27  候選 146｜與第 3 名同分 6 檔｜選中 ['2303', '2379', '1102']
2021-02-18  候選 144｜與第 3 名同分 6 檔｜選中 ['2301', '2303', '2344']
```

**Top 3 裡通常有 1~2 個位置是股票代號決定的，不是訊號決定的。**

這也解釋了 v6/v7 那次擺盪（`03_待辦與改進方向.md` 開頭）：標的池補了
23 期快照，平手名單重排，整條權益曲線跟著變。

## 為什麼代號是最糟的次要鍵

| 次要鍵 | 與報酬相關 | 可重現 | 可量測任意性 |
|---|---|---|---|
| `stock_id` | 否，但**與公司特性系統性相關** | 是 | **否** |
| 確定性抖動 | 否 | 是 | **是** |

代號不只是無資訊，它是**有偏**的：台股代號與上市年份、產業、規模都
相關，用它排序等於在模型裡偷偷塞進一個沒有宣告的因子。

更關鍵的是**可量測性**。抖動帶 seed 之後，換個 seed 重跑就知道有多少
結果來自平手運氣——這正是 CPCV 想回答的問題。字母序給你一個任意答案，
而且沒有辦法知道它有多任意。

## 為什麼不用原始分數當次要鍵

`Signal` 身上沒有原始分數，只有經過分箱的 `rank_score`。加欄位要改所有
建構點。而且用原始分數等於假設「校準器分不出來的差異，原始分數分得
出來」——那是沒有證據的假設。

## ⚠️ 不可用內建 `hash()`

CPython 的 `hash(str)` 每個行程都會換 salt（除非設 `PYTHONHASHSEED`），
同一份程式今天跑跟明天跑會排出不同順序，回測就不可重現。

這裡用 `blake2b`，跨行程、跨平台、跨版本都穩定。

## 這修的是偏差，不是變異

抖動讓平手不再偏向低號碼，但**平手本身還在**。真正的解法是讓
`rank_score` 有更高的解析度（12 個等級排 150 檔，本來就太粗）。
這個模組只保證：分不出來的時候，不要假裝分得出來，也不要偷偷用一個
有偏的代理。
"""

from __future__ import annotations

import hashlib
from typing import Protocol

DEFAULT_TIE_SEED = 20260915
"""預設 seed。換它重跑可以量測有多少結果來自平手運氣"""

_JITTER_BYTES = 8
_JITTER_SCALE = float(1 << (_JITTER_BYTES * 8))


class RankableProtocol(Protocol):
    """排序只需要這兩個欄位"""

    stock_id: str
    rank_score: float


def deterministic_jitter(stock_id: str, seed: int = DEFAULT_TIE_SEED) -> float:
    """
    `[0, 1)` 區間的確定性偽隨機值。

    Args:
        stock_id: 股票代號
        seed: 換它就換一組排序

    Returns:
        同一組 `(stock_id, seed)` 永遠回同一個值

    用 `blake2b` 而不是內建 `hash()`——後者每個行程換 salt，會讓回測
    不可重現。
    """
    digest = hashlib.blake2b(
        stock_id.encode("utf-8"),
        digest_size=_JITTER_BYTES,
        salt=str(seed).encode("utf-8")[:16],
    ).digest()
    return int.from_bytes(digest, "big") / _JITTER_SCALE


def ordering_key(
    signal: RankableProtocol, seed: int = DEFAULT_TIE_SEED
) -> tuple[float, float]:
    """
    排序鍵：分數優先，平手用抖動。

    Args:
        signal: 任何帶 `stock_id` 與 `rank_score` 的物件
        seed: 平手抖動的 seed

    Returns:
        `(-rank_score, jitter)`，可直接餵給 `sorted(key=...)`

    分數高的永遠排前面；**抖動只在完全同分時才作用**。
    """
    return (-signal.rank_score, deterministic_jitter(signal.stock_id, seed))
