"""
漲停事件的判定

## 為什麼要獨立成模組

漲停率（隨機 2.47% → 動能突破 Top3 16.62%，**6.7 倍**）是目前實測到
最強的預測力，但那份分析是臨時算的，沒有留下可重用、可測試的程式。

一個沒有測試的臨時計算，下次重跑不保證得到同一個數字。

## 判定方式：依跳動單位精算，不用容差

漲停價是 `參考價 × 1.1` 依跳動單位**向下**取整，所以實際漲幅通常低於
10%（例如 27.90 的漲停價是 30.65，+9.86%）。

第一版用一個 0.4 pp 的「容差」處理這件事，但那是一個要辯護的參數。
跳動單位是公告規則，用它精算就不需要容差。見 `limit_up_price`。

⚠️ 前提是還原因子正確。`_apply_adjustment` 的因子沿用邏輯修過一個
實測 bug（0050 在 2025-06-18 的 1:4 分割造成 84.20% 假回撤），
所以那一層必須先對。

## 台股漲跌幅限制

```
2015-06-01 起   ±10%
之前            ±7%
```

資料庫從 2015-01-05 開始，所以**前五個月是 ±7% 制度**。本模組預設
±10%，跨越那段時要明確傳 `limit=0.07`，否則 2015 上半年的漲停會被漏掉。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LIMIT_UP_PCT = 0.10
"""現行台股單日漲幅上限（2015-06-01 起）"""

LEGACY_LIMIT_UP_PCT = 0.07
"""2015-06-01 之前的上限。資料庫起點 2015-01-05 落在這個制度內"""

TICK_BANDS: tuple[tuple[float, float], ...] = (
    (10.0, 0.01),
    (50.0, 0.05),
    (100.0, 0.10),
    (500.0, 0.50),
    (1000.0, 1.00),
    (float("inf"), 5.00),
)
"""
台股股票的跳動單位：`(價格上界, 跳動單位)`，上界為開區間。

這是**公告規則**，不是估計參數。用它精算漲停價，就不需要一個「容差」
——容差是要辯護的參數，跳動單位是查得到的事實。

⚠️ ETF 的跳動單位不同（實測 0050 / 0056 的最小跳動是 0.01，比同價位
股票細 4~10 倍）。本表只適用股票。
"""


def tick_size(price: float) -> float:
    """
    該價位的跳動單位。

    Args:
        price: 價格

    Returns:
        跳動單位

    Raises:
        ValueError: `price` 非正
    """
    if price <= 0:
        raise ValueError(f"price 必須為正，得到 {price}")
    return next(tick for upper, tick in TICK_BANDS if price < upper)


def limit_up_price(reference: float, limit: float = LIMIT_UP_PCT) -> float:
    """
    漲停價：`reference × (1 + limit)` 依跳動單位**向下**取整。

    Args:
        reference: 參考價（前一日收盤）
        limit: 漲幅上限

    Returns:
        漲停價

    Raises:
        ValueError: `reference` 非正或 `limit` 非正

    手算案例：

        790.00 × 1.10 = 869.00   跳動 1.00（500~1000）  漲停 869.00  +10.00%
         27.90 × 1.10 =  30.69   跳動 0.05（10~50）     漲停  30.65   +9.86%
          9.99 × 1.10 =  10.989  跳動 0.05（跨到 10 以上）漲停 10.95   +9.61%

    第三例是**最壞情況**：跨價格帶時折讓最大。跳動單位取決於取整後的
    價格所在的帶，所以要迭代一次——`10.989` 落在 10~50 帶（跳動 0.05），
    不是 0~10 帶（0.01）。
    """
    if reference <= 0:
        raise ValueError(f"reference 必須為正，得到 {reference}")
    if limit <= 0:
        raise ValueError(f"limit 必須為正，得到 {limit}")

    target = reference * (1.0 + limit)
    tick = tick_size(target)
    capped = np.floor(round(target / tick, 9)) * tick
    # 取整後可能掉回下一個帶（例如 target 剛好在帶界之上），再收斂一次
    if capped > 0 and tick_size(capped) != tick:
        tick = tick_size(capped)
        capped = np.floor(round(target / tick, 9)) * tick
    return float(round(capped, 4))


def limit_up_events(
    closes: pd.DataFrame, limit: float = LIMIT_UP_PCT
) -> pd.DataFrame:
    """
    整個價格矩陣的漲停事件：收盤價達到當日漲停價。

    Args:
        closes: 還原收盤價（index 為交易日，column 為 stock_id）
        limit: 漲幅上限

    Returns:
        同形狀的布林矩陣。每檔的第一天必為 False（沒有前收）

    ## 為什麼用還原價

    直覺會說「±10% 是對**實際報價**的限制，所以要用未還原價」。
    實際相反：交易所在除權息日也會調整參考價，所以還原後的報酬才是
    限制實際適用的對象。

    實測（2015-06 起，2.8M 筆日報酬）兩者都有超過 10% 的殘留：

    ```
    報酬區間          未還原價    還原價
    [0.100, 0.105)     2,879     2,399
    [0.105, 0.200)     1,359     1,057
    ```

    還原價少一些（現金減資會把未還原的參考價往上調，看起來像暴漲），
    但沒有消掉——IPO 首日與長期停牌後恢復交易**沒有漲跌幅限制**。
    那些不是判定錯誤，是真的沒有上限的日子。

    ⚠️ 用 `>=` 比較而非 `==`：上面那些沒有上限的日子會超過算出來的
    漲停價，把它們算成漲停比漏掉安全——它們確實是「當天大漲」。
    """
    limits = limit_up_price_frame(closes.shift(1), limit)
    return (closes >= limits).where(limits.notna(), False).astype(bool)


def limit_up_price_frame(
    references: pd.DataFrame, limit: float = LIMIT_UP_PCT
) -> pd.DataFrame:
    """
    整個矩陣的漲停價（向量化）。

    Args:
        references: 參考價矩陣（通常是 `closes.shift(1)`）
        limit: 漲幅上限

    Returns:
        同形狀的漲停價；參考價缺值或非正處為 NaN

    ## 為什麼有兩個版本

    `limit_up_price` 是純量版，測試比對手算值用；這個是矩陣版，實際
    計算用（逐格 Python 呼叫在 330k 格上會慢兩個數量級）。

    ⚠️ **兩者不可各自實作規則**，否則遲早有一份會漂移。
    `test_vectorised_matches_scalar` 拿隨機價格比對兩者完全相等。
    """
    if limit <= 0:
        raise ValueError(f"limit 必須為正，得到 {limit}")

    values = references.to_numpy(dtype="float64", copy=True)
    valid = np.isfinite(values) & (values > 0)
    target = np.where(valid, values * (1.0 + limit), np.nan)

    def ticks_for(prices: np.ndarray) -> np.ndarray:
        """每個價位的跳動單位；非正或缺值處回 NaN"""
        result = np.full(prices.shape, np.nan)
        remaining = np.isfinite(prices) & (prices > 0)
        # 由小到大套用，先命中的帶勝出——與 tick_size 的 next() 同語意
        for upper, tick in TICK_BANDS:
            hit = remaining & (prices < upper)
            result[hit] = tick
            remaining &= ~hit
        return result

    tick = ticks_for(target)
    capped = np.floor(np.round(target / tick, 9)) * tick
    # 取整後可能掉回下一個帶，再收斂一次（與純量版同樣只迭代一次）
    retick = ticks_for(capped)
    changed = np.isfinite(retick) & (retick != tick)
    if changed.any():
        tick = np.where(changed, retick, tick)
        capped = np.floor(np.round(target / tick, 9)) * tick

    return pd.DataFrame(
        np.round(capped, 4), index=references.index, columns=references.columns
    )


def hit_within(events: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    未來 `horizon` 個交易日內是否出現過漲停。

    Args:
        events: `limit_up_events` 的輸出
        horizon: 往前看幾個交易日

    Returns:
        同形狀的布林矩陣。`[t, sid]` 為 True 代表 sid 在
        `(t, t+horizon]` 之間至少漲停一次

    ⚠️ **這是標籤，不是特徵。** 它用到 t 之後的資料，只能當預測目標，
    絕對不可以進特徵集（禁令 1）。

    區間是**左開右閉**：t 當天的漲停不算，因為決策是在 t 日收盤後做的，
    那時 t 日的漲停已經發生、買不到了。
    """
    if horizon < 1:
        raise ValueError(f"horizon 至少為 1，得到 {horizon}")

    # `rolling` 的視窗是**往後看**的（[t−w+1, t]），所以不能直接用它做
    # 前瞻標籤。先 shift(-horizon) 把 t+horizon 搬到 t，再做長度 horizon
    # 的後向 rolling，合起來就是 (t, t+horizon]：
    #
    #     shift(-h)[t]                = events[t+h]
    #     rolling(h).max() 在 t       = max(shift(-h)[t−h+1 .. t])
    #                                 = max(events[t+1 .. t+h])
    #
    # 第一版寫 `shift(-1).rolling(horizon)`，那算出來是
    # max(events[t−h+2 .. t+1])——**大部分視窗落在過去**，horizon 變大
    # 反而往回看得更遠。
    shifted = events.shift(-horizon).astype("float64")
    forward = shifted.rolling(window=horizon, min_periods=1).max()
    return forward.fillna(0.0).astype(bool)


def locked_all_day(
    highs: pd.DataFrame, lows: pd.DataFrame, events: pd.DataFrame
) -> pd.DataFrame:
    """
    漲停鎖死（一價到底）的日子——**這種買不到**。

    Args:
        highs / lows: 還原最高／最低價
        events: `limit_up_events` 的輸出

    Returns:
        同形狀的布林矩陣

    高 = 低代表全天只有一個價位。漲停且一價到底，等於開盤就鎖上，
    掛買單排不到。實測這種情形佔漲停的 4.7%。

    **回測若不扣掉這些，等於假設買得到買不到的東西。**
    """
    return events & highs.eq(lows)
