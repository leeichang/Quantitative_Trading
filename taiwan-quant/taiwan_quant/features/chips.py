"""
籌碼面特徵

台股特有的資料優勢——美股沒有逐日的三大法人買賣超與融資融券餘額。
D7 的「籌碼跟隨」策略族用這組特徵。

## 公布時序（D7 標註此族 look-ahead 風險最高）

    T 日 09:00-13:30   交易發生
    T 日 13:30         收盤價確定
    T 日 15:00-18:00   三大法人買賣超公布      ← 收盤已過
    T+1 09:00          進場

所以：

    ✓ 決策日 T 可以用 T 日的籌碼資料（因為是盤後才決策）
    ✗ 但**進場必須在 T+1 開盤**，不可用 T 日收盤價進場

第二條由 `labeling/triple_barrier.py` 的 `LABEL_ENTRY_OFFSET = 1` 保證。
本模組只負責「特徵不偷看 T 之後的資料」，由物理截斷掃描把關。

## 資料可用性

實測上游（qlib-tw-trader SQLite）覆蓋率：

    stock_daily_institutional    99.8%  ✓ 三大法人可用
    stock_daily_margin           95.4%  ✓ 融資融券可用
    stock_daily_shareholding      7.1%  ✗ 集保不可用
    stock_daily_securities_lending 2.8%  ✗ 借券不可用

因此本模組只用前兩者。呼叫端應先用 `data.loader.coverage_report()`
確認資料夠不夠，缺欄位時 `build_chips()` 會明確報錯而非產出全 NaN
（全 NaN 會讓 dropna 清空整個訓練集——qlib-tw-trader 訓練失敗的原因）。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
import pandas as pd

CHIPS_REQUIRED_COLUMNS = (
    "foreign_net",
    "trust_net",
    "dealer_net",
    "margin_balance",
    "short_balance",
    "volume",
    "close",
)
"""
本模組需要的欄位。

foreign_net / trust_net / dealer_net  三大法人買賣超（股數，正為買超）
margin_balance / short_balance        融資 / 融券餘額
volume                                成交量（用來把買賣超正規化）
close                                 收盤價
"""

INSTITUTION_COLUMNS = ("foreign_net", "trust_net", "dealer_net")

EPSILON = 1e-12

Direction = Literal["buy", "sell"]


def _validate(bars: pd.DataFrame) -> None:
    missing = set(CHIPS_REQUIRED_COLUMNS) - set(bars.columns)
    if missing:
        raise ValueError(
            f"缺少必要欄位：{sorted(missing)}。"
            "籌碼資料不完整時請先確認 coverage_report()，"
            "不可用全 NaN 的特徵訓練——那會讓 dropna 清空整個樣本。"
        )
    if not bars.index.is_monotonic_increasing:
        raise ValueError("索引必須依日期升冪排序")


# ══════════════════════════════════════════════════════════════
# 個別特徵
# ══════════════════════════════════════════════════════════════


def net_buy_ratio(bars: pd.DataFrame, column: str, window: int) -> pd.Series:
    """
    N 日累計買賣超相對同期累計成交量。

        sum(net, N) / sum(volume, N)

    用成交量正規化的理由：買超 10,000 股對台積電是噪音、對小型股是主力進場。
    絕對股數無法跨標的比較。

    成交量累計為 0（整段停牌）時回 NaN——沒有成交就無法定義佔比。
    """
    net_sum = bars[column].rolling(window, min_periods=window).sum()
    volume_sum = bars["volume"].rolling(window, min_periods=window).sum()
    return net_sum / volume_sum.where(volume_sum > EPSILON)


def consecutive_net_buy_days(
    bars: pd.DataFrame,
    column: str,
    direction: Direction = "buy",
) -> pd.Series:
    """
    連續買（賣）超天數。

    D7 籌碼跟隨族的核心訊號。

    Args:
        column: 要看哪個法人
        direction: "buy" 算連續買超、"sell" 算連續賣超

    規則：
      · 買超為 0（持平）**中斷**連續，不是「維持不變」
      · 方向反轉立刻歸零

    實作只用 cumsum 技巧，完全不涉及未來資料：
    以「非連續事件」為分組鍵做組內累計計數。
    """
    net = bars[column]
    is_streak = (net > 0) if direction == "buy" else (net < 0)

    # 每次 streak 中斷就換一個 group id，組內累計計數即為連續天數
    group = (~is_streak).cumsum()
    counts = is_streak.groupby(group).cumsum()
    return counts.where(is_streak, 0).astype("int64")


def net_buy_streak_strength(bars: pd.DataFrame) -> pd.Series:
    """
    三大法人合力方向（−3 ~ +3）。

        每個法人買超記 +1、賣超記 −1、持平記 0，三者相加

    三家同步買超（+3）比單一法人買超更有訊息量。
    """
    signs = [np.sign(bars[col]).fillna(0.0) for col in INSTITUTION_COLUMNS]
    return sum(signs).astype("int64")


def institution_net_agreement(bars: pd.DataFrame, window: int) -> pd.Series:
    """
    N 日內三大法人方向一致度（0~1）。

        |sum(strength, N)| / (3 × N)

    1 = 整段期間三家都同方向；0 = 互相抵銷。
    取絕對值，因為「一致看空」與「一致看多」都是強訊號。
    """
    strength = net_buy_streak_strength(bars).astype("float64")
    rolled = strength.rolling(window, min_periods=window).sum()
    return rolled.abs() / (3.0 * window)


def margin_balance_change(bars: pd.DataFrame, window: int = 5) -> pd.Series:
    """
    融資餘額 N 日變化率。

        margin_balance[T] / margin_balance[T − N] − 1

    融資大增常伴隨散戶追高，是反向指標的常見輸入。
    前期餘額為 0 時回 NaN 而非 inf。
    """
    balance = bars["margin_balance"]
    base = balance.shift(window)
    return balance / base.where(base > EPSILON) - 1.0


def short_margin_ratio(bars: pd.DataFrame) -> pd.Series:
    """
    券資比（融券餘額 / 融資餘額）。

    偏高代表空方力道強，也是軋空行情的前提條件。
    """
    margin = bars["margin_balance"]
    return bars["short_balance"] / margin.where(margin > EPSILON)


def margin_to_volume(bars: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    融資餘額相對 N 日均量。

    用均量正規化，讓不同流動性的標的可比較。
    """
    vol_ma = bars["volume"].rolling(window, min_periods=window).mean()
    return bars["margin_balance"] / vol_ma.where(vol_ma > EPSILON)


# ══════════════════════════════════════════════════════════════
# 特徵目錄
# ══════════════════════════════════════════════════════════════

CHIPS_FEATURES: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    # 外資：權重最高，動向最受關注
    "foreign_net_ratio_1": lambda b: net_buy_ratio(b, "foreign_net", 1),
    "foreign_net_ratio_5": lambda b: net_buy_ratio(b, "foreign_net", 5),
    "foreign_net_ratio_20": lambda b: net_buy_ratio(b, "foreign_net", 20),
    "foreign_buy_streak": lambda b: consecutive_net_buy_days(b, "foreign_net", "buy"),
    "foreign_sell_streak": lambda b: consecutive_net_buy_days(b, "foreign_net", "sell"),
    # 投信：季底作帳行為明顯，短線動能來源
    "trust_net_ratio_5": lambda b: net_buy_ratio(b, "trust_net", 5),
    "trust_net_ratio_20": lambda b: net_buy_ratio(b, "trust_net", 20),
    "trust_buy_streak": lambda b: consecutive_net_buy_days(b, "trust_net", "buy"),
    # 自營商：波動大，單獨訊號較弱，主要看合力
    "dealer_net_ratio_5": lambda b: net_buy_ratio(b, "dealer_net", 5),
    # 三大法人合力
    "institution_strength": net_buy_streak_strength,
    "institution_agreement_5": lambda b: institution_net_agreement(b, 5),
    "institution_agreement_20": lambda b: institution_net_agreement(b, 20),
    # 融資融券
    "margin_change_5": lambda b: margin_balance_change(b, 5),
    "margin_change_20": lambda b: margin_balance_change(b, 20),
    "short_margin_ratio": short_margin_ratio,
    "margin_to_volume_20": lambda b: margin_to_volume(b, 20),
}
"""
籌碼面特徵目錄（16 個）。

只用覆蓋率 > 95% 的上游資料（三大法人、融資融券）。集保與借券資料
覆蓋率僅 7.1% / 2.8%，不納入——不完整的特徵會讓 dropna 清空樣本。
"""


def build_chips(bars: pd.DataFrame) -> pd.DataFrame:
    """
    計算整組籌碼面特徵。

    Args:
        bars: 含 `CHIPS_REQUIRED_COLUMNS` 的日頻資料（**不會被修改**）

    Returns:
        與 bars 同索引的特徵 DataFrame。視窗不足處為 NaN；
        無限值一律轉 NaN。

    Raises:
        ValueError: 缺少必要欄位或索引未排序
    """
    _validate(bars)

    frame = pd.DataFrame(
        {name: fn(bars) for name, fn in CHIPS_FEATURES.items()},
        index=bars.index,
    )
    return frame.replace([np.inf, -np.inf], np.nan)
