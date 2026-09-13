"""
技術面特徵

每個函式只吃「決策日及之前」的資料——所有 rolling / shift 都是往後看，
沒有任何 `shift(-n)`、也沒有任何全樣本統計量（`mean()` / `std()` 直接
對整段序列呼叫）。

為什麼要特別強調後者：

    def zscore(bars):
        close = bars["close"]
        return (close - close.mean()) / close.std()   # ← 洩漏

這段沒有任何未來參照語法，靜態掃描看不到，但它吃了整段樣本，資料一變長
歷史值就改變。本模組一律用 `rolling(...)` 而非裸 `mean()`，並由
`tests/test_features.py` 的物理截斷掃描把關（CLAUDE.md 規格 16）。

除零防護：所有比率型特徵都要處理分母為 0（停牌日零成交量、完全平盤導致
標準差為 0）。inf 進到模型會讓訓練崩潰或給出荒謬權重。
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")

EPSILON = 1e-12
"""除零防護用的極小值"""


def _validate(bars: pd.DataFrame) -> None:
    missing = set(REQUIRED_COLUMNS) - set(bars.columns)
    if missing:
        raise ValueError(f"缺少必要欄位：{sorted(missing)}")
    if not bars.index.is_monotonic_increasing:
        raise ValueError("日 K 索引必須依日期升冪排序")


# ══════════════════════════════════════════════════════════════
# 個別特徵
# ══════════════════════════════════════════════════════════════


def ma_ratio(bars: pd.DataFrame, window: int) -> pd.Series:
    """
    收盤相對 N 日均線的偏離率。

        (close / MA(close, N)) − 1

    > 0 表示站在均線上方。均線含當日收盤（決策日收盤後才算，不算洩漏）。
    """
    ma = bars["close"].rolling(window, min_periods=window).mean()
    return bars["close"] / ma.replace(0.0, np.nan) - 1.0


def momentum(bars: pd.DataFrame, window: int) -> pd.Series:
    """
    N 日報酬率。

        (close[T] / close[T − N]) − 1

    注意 `pct_change(periods=N)` 的方向：正的 periods 是往**過去**看，
    這是合法的。寫成負數就變成看未來。
    """
    return bars["close"].pct_change(periods=window)


def rsi(bars: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    相對強弱指標（0~100）。

        RS  = 平均漲幅 / 平均跌幅
        RSI = 100 − 100 / (1 + RS)

    單邊行情的邊界：
      · 只漲不跌 → 平均跌幅 0 → RSI = 100
      · 只跌不漲 → 平均漲幅 0 → RSI = 0
    用 Wilder 原始定義的簡單移動平均版本（非指數平滑），
    確保「只用過去 window 天」這件事一目了然。
    """
    delta = bars["close"].diff()
    gain = delta.clip(lower=0.0).rolling(window, min_periods=window).mean()
    loss = (-delta.clip(upper=0.0)).rolling(window, min_periods=window).mean()

    rsi_values = pd.Series(np.nan, index=bars.index, dtype="float64")
    valid = gain.notna() & loss.notna()

    both_zero = valid & (gain <= EPSILON) & (loss <= EPSILON)
    only_gain = valid & (loss <= EPSILON) & ~both_zero
    normal = valid & (loss > EPSILON)

    rsi_values[both_zero] = 50.0                       # 完全平盤視為中性
    rsi_values[only_gain] = 100.0
    rs = gain[normal] / loss[normal]
    rsi_values[normal] = 100.0 - 100.0 / (1.0 + rs)

    return rsi_values


def bollinger_position(bars: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.Series:
    """
    收盤在布林通道中的相對位置。

        0 = 下軌、0.5 = 中軌、1 = 上軌
        突破上軌 > 1、跌破下軌 < 0

    標準差為 0（完全平盤）時回 0.5 而非 inf/NaN——平盤本身就是中性狀態，
    沒有理由讓它變成缺值或無限大。
    """
    close = bars["close"]
    middle = close.rolling(window, min_periods=window).mean()
    std = close.rolling(window, min_periods=window).std(ddof=0)

    band_width = 2.0 * num_std * std
    position = pd.Series(np.nan, index=bars.index, dtype="float64")

    valid = middle.notna() & std.notna()
    flat = valid & (band_width <= EPSILON)
    normal = valid & (band_width > EPSILON)

    position[flat] = 0.5
    lower = middle[normal] - num_std * std[normal]
    position[normal] = (close[normal] - lower) / band_width[normal]

    return position


def volume_ratio(bars: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    成交量相對 N 日均量的倍數。

        volume / VOL_MA(N)

    均量含當日。均量為 0（整段停牌）時回 NaN 而非 inf——
    「沒有成交」無法定義量能倍數，硬給數字會誤導模型。
    """
    volume = bars["volume"].astype("float64")
    vol_ma = volume.rolling(window, min_periods=window).mean()
    return volume / vol_ma.where(vol_ma > EPSILON)


def true_range_ratio(bars: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    ATR 相對收盤價（波動度）。

    True Range 含跳空項，否則會低估波動度：

        TR = max(high − low, |high − prev_close|, |low − prev_close|)
    """
    high, low = bars["high"], bars["low"]
    prev_close = bars["close"].shift(1)

    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    true_range.iloc[0] = np.nan   # 第一筆沒有前收

    atr = true_range.rolling(window, min_periods=window).mean()
    close = bars["close"]
    return atr / close.where(close > EPSILON)


def high_low_position(bars: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    收盤在 N 日高低區間中的位置（0~1），即 Stochastic 的 %K 分子部分。

        (close − min(low, N)) / (max(high, N) − min(low, N))
    """
    highest = bars["high"].rolling(window, min_periods=window).max()
    lowest = bars["low"].rolling(window, min_periods=window).min()
    span = highest - lowest
    return (bars["close"] - lowest) / span.where(span > EPSILON)


def gap_ratio(bars: pd.DataFrame) -> pd.Series:
    """
    開盤跳空幅度。

        (open[T] / close[T − 1]) − 1
    """
    prev_close = bars["close"].shift(1)
    return bars["open"] / prev_close.where(prev_close > EPSILON) - 1.0


def intraday_range(bars: pd.DataFrame) -> pd.Series:
    """
    當日振幅相對開盤價。

        (high − low) / open
    """
    open_ = bars["open"]
    return (bars["high"] - bars["low"]) / open_.where(open_ > EPSILON)


def close_position_in_bar(bars: pd.DataFrame) -> pd.Series:
    """
    收盤在當日高低之間的位置（0 = 收最低、1 = 收最高）。

    量價分析常用：收盤靠上代表買方掌控。
    高低相同（一價到底，例如漲停鎖死）時回 0.5。
    """
    high, low, close = bars["high"], bars["low"], bars["close"]
    span = high - low
    position = pd.Series(0.5, index=bars.index, dtype="float64")
    normal = span > EPSILON
    position[normal] = (close[normal] - low[normal]) / span[normal]
    return position


# ══════════════════════════════════════════════════════════════
# 特徵目錄
# ══════════════════════════════════════════════════════════════

TECHNICAL_FEATURES: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    # 均線偏離：短中長三段
    "ma_ratio_5": lambda b: ma_ratio(b, 5),
    "ma_ratio_20": lambda b: ma_ratio(b, 20),
    "ma_ratio_60": lambda b: ma_ratio(b, 60),
    # 動能：D7 的動能突破族用這組
    "momentum_5": lambda b: momentum(b, 5),
    "momentum_20": lambda b: momentum(b, 20),
    "momentum_60": lambda b: momentum(b, 60),
    "momentum_120": lambda b: momentum(b, 120),
    # 超買超賣：D7 的均值回歸族用這組
    "rsi_14": lambda b: rsi(b, 14),
    "bollinger_position_20": lambda b: bollinger_position(b, 20),
    "high_low_position_20": lambda b: high_low_position(b, 20),
    # 量能
    "volume_ratio_5": lambda b: volume_ratio(b, 5),
    "volume_ratio_20": lambda b: volume_ratio(b, 20),
    # 波動度：柵欄寬度也用 ATR，這裡的版本供模型當特徵
    "true_range_ratio_14": lambda b: true_range_ratio(b, 14),
    # 日內型態
    "gap_ratio": gap_ratio,
    "intraday_range": intraday_range,
    "close_position_in_bar": close_position_in_bar,
}
"""
技術面特徵目錄。

刻意保持小規模（16 個）。qlib-tw-trader 有 303 個因子，實測結果是
live IC 為負——因子數量不是優勢來源。先讓少量特徵通過完整驗證，
再談擴充（CLAUDE.md：先有 baseline 再談複雜度）。
"""


def build_technical(bars: pd.DataFrame) -> pd.DataFrame:
    """
    計算整組技術面特徵。

    Args:
        bars: 日 K（**不會被修改**），需含 open/high/low/close/volume

    Returns:
        與 bars 同索引的特徵 DataFrame，欄位為 `TECHNICAL_FEATURES` 的鍵。
        視窗不足處為 NaN；無限值一律轉為 NaN
        （inf 進模型會讓訓練崩潰或給出荒謬權重）。
    """
    _validate(bars)

    frame = pd.DataFrame(
        {name: fn(bars) for name, fn in TECHNICAL_FEATURES.items()},
        index=bars.index,
    )
    return frame.replace([np.inf, -np.inf], np.nan)
