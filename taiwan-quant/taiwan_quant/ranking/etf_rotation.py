"""
ETF 輪動：依過去報酬排序，持有前 k 檔

## 為什麼值得做這條

所有股票策略都撞同一面牆——**過路費**：

```
              來回成本    年化成本拖累
股票 N=10 H=40  1.081%      6.8%
ETF  H=120      0.571%      1.2%     ← 5.7 倍
```

40 萬做 1.08% 的來回，年化先送掉 6.8%，而台股長期報酬約 12~16%。
**一個 1 pp／年的優勢在 ETF 區間活得下來，在股票區間活不下來。**

這個結構論證不依賴任何回測數字，所以它是目前最值得投入的方向。

## 為什麼邏輯要抽成模組

第一版寫在臨時腳本裡，跑出「年化 81.8%」。那不是發現，是索引錯位：

```python
cl = cl.dropna(how="any")     # cl 被過濾了
cl.iloc[i] / op.iloc[i + 1]   # 但 op 沒有 → iloc 位置對應到不同日期
```

修正後是 12.5%。**一個沒有測試的臨時計算，錯了不會有人發現。**

## 時點紀律（禁令 1）

```
排序    只用 ≤ T 的收盤價
進場    T+1 開盤
出場    T+1+H 收盤
```

`trailing_momentum` 明確接受 `position`（整數位置）而不是日期，因為
`iloc` 的錯位正是上面那個 bug 的成因——位置語意必須顯式。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class RotationError(RuntimeError):
    """輪動輸入不合法"""


def aligned_views(*frames: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    """
    把多個價格矩陣對齊到共同、無缺值的索引與欄位。

    Args:
        *frames: 價格矩陣（index 為交易日，column 為 stock_id）

    Returns:
        與輸入同順序的矩陣，全部共用同一個 index 與 column

    Raises:
        RotationError: 沒有共同的日期或標的

    ## 這個函式存在的唯一理由

    `iloc[i]` 在兩個列數不同的矩陣上**指向不同日期**。第一版的
    「年化 81.8%」就是這樣來的：收盤價做了 `dropna` 而開盤價沒有。

    **所有用 `iloc` 對位的地方都必須先經過這裡。**
    """
    if not frames:
        raise RotationError("至少要傳一個矩陣")

    index = frames[0].dropna(how="any").index
    columns = frames[0].columns
    for frame in frames[1:]:
        index = index.intersection(frame.dropna(how="any").index)
        columns = columns.intersection(frame.columns)

    if len(index) == 0:
        raise RotationError("這些矩陣沒有共同的無缺值交易日")
    if len(columns) < 2:
        raise RotationError(f"共同標的只有 {len(columns)} 檔，無法排序")

    ordered = sorted(columns)
    return tuple(frame.loc[index, ordered] for frame in frames)


def trailing_momentum(
    closes: pd.DataFrame, position: int, lookback: int
) -> pd.Series:
    """
    第 `position` 個交易日（0-based）回看 `lookback` 日的報酬。

    Args:
        closes: 還原收盤價，index 必須已對齊（見 `aligned_views`）
        position: 決策日的整數位置。**只用 ≤ position 的資料**
        lookback: 回看的交易日數

    Returns:
        每檔的區間報酬。算不出的為 NaN

    Raises:
        RotationError: `position` 不足 `lookback`，或參數非正

    刻意收 `position` 而不是日期：`iloc` 的位置語意必須顯式，
    因為位置錯位不會拋錯，只會安靜地算錯。
    """
    if lookback < 1:
        raise RotationError(f"lookback 至少為 1，得到 {lookback}")
    if position < lookback:
        raise RotationError(
            f"position={position} 不足回看 {lookback} 日；暖機期不可省"
        )
    if position >= len(closes):
        raise RotationError(f"position={position} 超出範圍 {len(closes)}")

    return closes.iloc[position] / closes.iloc[position - lookback] - 1.0


def top_k(scores: pd.Series, k: int) -> list[str]:
    """
    取分數最高的 k 檔，平手用代號排序。

    Args:
        scores: 每檔的分數
        k: 取幾檔

    Returns:
        `stock_id` 清單，最多 k 檔

    Raises:
        RotationError: `k` 非正

    ## 為什麼這裡可以用代號破平手

    `ranking/tie_break.py` 用確定性抖動取代字母序，理由是台股代號與
    上市年份／產業／規模系統性相關——那個偏差在 150 檔裡有實際影響
    （實測虛增 17.5 pp）。

    **8 檔 ETF 的情境不同**：它們的代號是發行順序（0050~0061），
    而連續報酬幾乎不可能完全相等。平手在這裡是浮點巧合，不是
    解析度不足。維持簡單且可重現即可。
    """
    if k < 1:
        raise RotationError(f"k 至少為 1，得到 {k}")
    usable = scores.dropna()
    if usable.empty:
        return []
    ordered = sorted(usable.index, key=lambda sid: (-float(usable[sid]), sid))
    return ordered[:k]


def holding_return(
    closes: pd.DataFrame,
    opens: pd.DataFrame,
    position: int,
    horizon: int,
    picks: list[str],
) -> float:
    """
    T+1 開盤買、T+1+horizon 收盤賣的等權毛報酬。

    Args:
        closes / opens: 已對齊的還原價矩陣
        position: 決策日的整數位置
        horizon: 持有交易日數
        picks: 持有清單

    Returns:
        等權平均毛報酬

    Raises:
        RotationError: 視野不足、`picks` 為空，或價格不合法

    **不含成本。** 成本由 `config/costs.py` 逐檔決定（禁令 3），
    而它需要實際成交價來判斷整股／零股，不屬於這一層。
    """
    if not picks:
        raise RotationError("picks 不可為空")
    if horizon < 1:
        raise RotationError(f"horizon 至少為 1，得到 {horizon}")
    exit_position = position + 1 + horizon
    if exit_position >= len(closes):
        raise RotationError(
            f"視野不足：position={position} + 1 + {horizon} 超出 {len(closes)}"
        )

    returns = []
    for stock_id in picks:
        entry = float(opens.iloc[position + 1][stock_id])
        exit_price = float(closes.iloc[exit_position][stock_id])
        if not (np.isfinite(entry) and entry > 0 and np.isfinite(exit_price)):
            raise RotationError(
                f"{stock_id} 在位置 {position + 1} / {exit_position} 價格不合法；"
                "對齊過的矩陣不該出現這個，請檢查 aligned_views"
            )
        returns.append(exit_price / entry - 1.0)
    return float(np.mean(returns))
