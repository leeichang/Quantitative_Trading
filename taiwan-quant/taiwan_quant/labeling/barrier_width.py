"""
柵欄寬度推導

CLAUDE.md 核心建模方式禁止人工寫死 target_pct / stop_pct。本模組把兩者
從資料推導出來：

    stop_pct   = atr_multiple × ATR(n) / close
                 停損距離由**波動度**決定。波動大的股票停損要放寬，
                 否則會被日常雜訊掃出場。

    target_pct = 決策日之前「已實現」的未來 horizon 日報酬分布之指定分位數
                 目標價由**歷史經驗**決定，而不是拍一個 +8%。

    若 target_pct / stop_pct < min_risk_reward → 放棄該筆（回傳 None）
                 **不可硬拉目標價去湊 R:R。**那等於用一個不現實的目標價
                 換一個好看的風報比。

意見來源（../docs/需求規劃/202609/來源原文/02_ChatGPT_台股量化軟體比較.md）：

    不要人工設定 Buy = Current Price × 0.96，而是讓模型研究
    「歷史上這種型態，什麼價格位置進場最有效？」

反 look-ahead（CLAUDE.md 禁令 1）：報酬分位數只取決策日之前**已經走完
horizon 的**樣本。決策日當下還沒實現的報酬不可納入。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("high", "low", "close")


@dataclass(frozen=True)
class BarrierWidth:
    """
    柵欄寬度推導結果。

    帶出完整推導依據，否則無法回答「為什麼當時停損設在這個距離」
    （CLAUDE.md 禁令 7、8 要求可稽核）。
    """

    target_pct: float
    stop_pct: float

    atr_value: float
    atr_period: int
    atr_multiple: float
    target_quantile: float
    lookback: int
    sample_size: int
    """算目標分位數時實際用到的樣本數"""

    @property
    def risk_reward(self) -> float:
        return self.target_pct / self.stop_pct


def _validate(bars: pd.DataFrame) -> None:
    missing = set(REQUIRED_COLUMNS) - set(bars.columns)
    if missing:
        raise ValueError(f"缺少必要欄位：{sorted(missing)}")
    if not bars.index.is_monotonic_increasing:
        raise ValueError("日 K 索引必須依日期升冪排序")


def atr(bars: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range。

    True Range 取三者最大，**必須含跳空項**，否則會嚴重低估波動度：

        TR = max(high − low,
                 |high − prev_close|,
                 |low  − prev_close|)

    Args:
        bars: 日 K（不會被修改），需含 high/low/close
        period: 平均視窗

    Returns:
        與 bars 同索引的 Series。前 period 筆為 NaN
        （視窗不足時不用不完整資料硬算）。
    """
    if period < 1:
        raise ValueError(f"period 至少為 1，得到 {period}")
    _validate(bars)

    high = bars["high"]
    low = bars["low"]
    prev_close = bars["close"].shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    # 第一筆沒有前收，True Range 不可用
    true_range.iloc[0] = np.nan

    return true_range.rolling(window=period, min_periods=period).mean()


def _past_horizon_returns(
    close: pd.Series,
    decision_idx: int,
    horizon: int,
    lookback: int,
) -> np.ndarray:
    """
    取決策日之前「已實現」的 horizon 日報酬樣本。

    第 i 天的 horizon 報酬 = close[i + horizon] / close[i] − 1，
    只有 i + horizon <= decision_idx 才算已實現。
    """
    last_start = decision_idx - horizon
    if last_start < 0:
        return np.empty(0)

    first_start = max(0, last_start - lookback + 1)
    starts = np.arange(first_start, last_start + 1)
    if starts.size == 0:
        return np.empty(0)

    values = close.to_numpy(dtype=float)
    return values[starts + horizon] / values[starts] - 1.0


def derive_width(
    bars: pd.DataFrame,
    decision_idx: int,
    horizon: int,
    atr_period: int = 14,
    atr_multiple: float = 1.5,
    target_quantile: float = 0.7,
    min_risk_reward: float = 2.0,
    lookback: int = 250,
    min_stop_pct: float = 0.015,
    max_stop_pct: float = 0.10,
    min_target_pct: float = 0.02,
) -> BarrierWidth | None:
    """
    推導單一決策日的柵欄寬度。

    Args:
        bars: 日 K（不會被修改）
        decision_idx: 決策日的位置索引
        horizon: 時間柵長度（交易日）
        atr_period: ATR 視窗
        atr_multiple: 停損 = atr_multiple × ATR / close
        target_quantile: 目標取歷史 horizon 報酬的哪個分位數
        min_risk_reward: R:R 門檻，不足則放棄（CLAUDE.md 進場門檻要求 >= 2.0）
        lookback: 算報酬分位數時回看幾個樣本
        min_stop_pct / max_stop_pct: 停損距離的地板與天花板
        min_target_pct: 目標距離的地板

    Returns:
        推導結果；資料不足或 R:R 不達標時回傳 None（**不猜測、不湊數**）
    """
    _validate(bars)
    if horizon < 1:
        raise ValueError(f"horizon 至少為 1，得到 {horizon}")
    if not 0.0 < target_quantile < 1.0:
        raise ValueError(f"target_quantile 必須落在 (0, 1)，得到 {target_quantile}")
    if not 0 <= decision_idx < len(bars):
        raise ValueError(f"decision_idx {decision_idx} 超出資料範圍 0..{len(bars) - 1}")

    # ── 停損：由 ATR 決定 ──
    atr_series = atr(bars, period=atr_period)
    atr_value = atr_series.iloc[decision_idx]
    if not np.isfinite(atr_value):
        return None

    close = float(bars["close"].iloc[decision_idx])
    if not np.isfinite(close) or close <= 0:
        return None

    raw_stop = atr_multiple * float(atr_value) / close
    stop_pct = min(max(raw_stop, min_stop_pct), max_stop_pct)

    # ── 目標：由歷史已實現報酬分位數決定 ──
    samples = _past_horizon_returns(bars["close"], decision_idx, horizon, lookback)
    if samples.size == 0:
        return None

    if not np.isfinite(samples).all():
        # 上游資料洞（停牌、缺漏）會讓 np.quantile 回 NaN。
        # 絕不可放行：NaN 不會讓門檻比較拋錯，而是讓 `NaN < 門檻` 為 False，
        # 靜默通過檢查。實測案例：2317 在 2025-07-30 有一列 OHLC 全為 NULL。
        return None

    raw_target = float(np.quantile(samples, target_quantile))
    target_pct = max(raw_target, min_target_pct)

    if not (np.isfinite(target_pct) and np.isfinite(stop_pct)) or target_pct <= 0:
        return None

    width = BarrierWidth(
        target_pct=target_pct,
        stop_pct=stop_pct,
        atr_value=float(atr_value),
        atr_period=atr_period,
        atr_multiple=atr_multiple,
        target_quantile=target_quantile,
        lookback=lookback,
        sample_size=int(samples.size),
    )

    # ── R:R 門檻：不足就放棄，不硬拉目標價湊數 ──
    if width.risk_reward < min_risk_reward:
        return None

    return width
