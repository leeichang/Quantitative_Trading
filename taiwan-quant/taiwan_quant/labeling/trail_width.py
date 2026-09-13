"""
移動停損寬度推導

CLAUDE.md 禁止人工寫死柵欄寬度。`trail_pct` 必須從資料推導。

## 為什麼不能照抄 `barrier_width.derive_width`

固定停損與移動停損的受測次數不同：

    固定停損   進場時設定一次，只有價格跌破才觸發
    移動停損   每天隨高點上移，**整個持有期都在被測試**

同樣寬度的停損，移動版被觸發的機率高得多。直接沿用 `atr_multiple = 0.8`
會把所有部位在趨勢中途掃出場——那就回到「上檔被封死」的老問題。

## 推導方式

直接量測這檔股票**在一個持有期內通常會回落多少**：

    對每個已走完的 horizon 長度視窗：
        區間最高 = cummax(high)
        回落     = 1 − low / 區間最高
        該視窗的最大回落 = max(回落)

    trail_pct = 這些最大回落的第 q 分位數

這個量測方式與 `trailing_stop.label_trailing` 的觸發邏輯**完全一致**
（以 high 累積峰值、以 low 判定觸發），所以分位數的意義是直白的：

    q = 0.80  →  歷史上 80% 的持有期，這個停損不會被觸發

ATR 不足以取代它。ATR 量的是**單日**振幅，移動停損承受的是**多日累積**
回落——兩者可以差好幾倍。ATR 仍然計算並記錄，作為稽核依據。

## 為什麼 q 預設 0.80

這是唯一真正的參數，它控制一組對立的代價：

    q 太小  →  被日常雜訊掃出場，永遠跟不到趨勢
    q 太大  →  停損形同虛設，回吐過多獲利

0.80 表示容忍五次持有期裡最深的那一次以外的所有回落。實際值應由
`scripts/diagnose_trail_quantile.py` 在訓練期掃描決定，此處只是預設。

## 上下限是防呆，不是調參旋鈕

    地板 = 一趟來回成本（取自 config/costs.py，禁令 3）
           比交易成本還窄的停損沒有意義——被掃出場的代價大於它保護的金額
    天花板 = 30%
           再寬就不是風險控制了

兩者正常情況下都不會綁定。綁定時 `floor_applied` / `cap_applied` 會標記，
讓報告看得見「這檔的寬度是被夾住的」。

## 反 look-ahead（禁令 1）

回落樣本只取決策日當天或之前**就已經走完**的視窗。決策日之後的暴跌
不可用來放寬停損——那會讓回測漂亮得不合理。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from taiwan_quant.config.costs import DEFAULT, Tier
from taiwan_quant.labeling.barrier_width import atr

REQUIRED_COLUMNS = ("high", "low", "close")

DEFAULT_PULLBACK_QUANTILE = 0.80
DEFAULT_LOOKBACK = 250

MIN_TRAIL_PCT = DEFAULT.round_trip_rate(Tier.LARGE)
"""地板：一趟來回成本。禁令 3——費率一律取自 config/costs.py"""

MAX_TRAIL_PCT = 0.30
"""天花板：再寬就不是風險控制"""


@dataclass(frozen=True)
class TrailWidth:
    """
    移動停損寬度推導結果。

    帶出完整推導依據，否則無法回答「為什麼當時停損設在這個距離」
    （禁令 7、8 要求可稽核）。
    """

    trail_pct: float
    """夾在上下限之內的最終值"""

    raw_trail_pct: float
    """夾之前的原始分位數。與 trail_pct 不同時代表上下限有綁定"""

    pullback_quantile: float
    horizon: int
    lookback: int
    sample_size: int
    """實際用到的視窗數"""

    atr_value: float
    atr_period: int
    """ATR 不參與推導，只作為稽核時的波動度對照"""

    @property
    def floor_applied(self) -> bool:
        return self.raw_trail_pct < MIN_TRAIL_PCT

    @property
    def cap_applied(self) -> bool:
        return self.raw_trail_pct > MAX_TRAIL_PCT


def _validate(bars: pd.DataFrame) -> None:
    missing = set(REQUIRED_COLUMNS) - set(bars.columns)
    if missing:
        raise ValueError(f"缺少必要欄位：{sorted(missing)}")
    if not bars.index.is_monotonic_increasing:
        raise ValueError("日 K 索引必須依日期升冪排序")


def max_pullback_samples(
    bars: pd.DataFrame,
    decision_idx: int,
    horizon: int,
    lookback: int = DEFAULT_LOOKBACK,
) -> np.ndarray:
    """
    取決策日之前已走完的 horizon 長度視窗，各自的最大回落。

    Args:
        bars: 日 K（**不會被修改**）
        decision_idx: 決策日的位置索引
        horizon: 視窗長度（交易日），須與持有期一致
        lookback: 最多回看幾個視窗

    Returns:
        一維陣列，每個元素是一個視窗的最大回落率（0.12 = 回落 12%）。
        可用視窗不足時回空陣列。

    視窗 [s, s + horizon − 1] 只有在 `s + horizon − 1 <= decision_idx`
    時才算已走完。每個視窗的峰值**從視窗起點重新累積**——視窗之前的
    高點與這次持有無關。
    """
    _validate(bars)
    if horizon < 1:
        raise ValueError(f"horizon 至少為 1，得到 {horizon}")

    last_start = decision_idx - horizon + 1
    if last_start < 0:
        return np.empty(0)

    first_start = max(0, last_start - lookback + 1)
    stop = last_start + horizon  # 最後一個視窗的結束位置（不含）

    high = bars["high"].to_numpy(dtype=float)[first_start:stop]
    low = bars["low"].to_numpy(dtype=float)[first_start:stop]
    if high.size < horizon:
        return np.empty(0)

    # 滑動視窗：shape = (視窗數, horizon)
    high_windows = sliding_window_view(high, horizon)
    low_windows = sliding_window_view(low, horizon)

    # 與 label_trailing 一致：以 high 累積峰值、以 low 判定回落
    peaks = np.maximum.accumulate(high_windows, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        drawdown = 1.0 - low_windows / peaks

    return drawdown.max(axis=1)


def derive_trail_width(
    bars: pd.DataFrame,
    decision_idx: int,
    horizon: int,
    pullback_quantile: float = DEFAULT_PULLBACK_QUANTILE,
    lookback: int = DEFAULT_LOOKBACK,
    atr_period: int = 14,
) -> TrailWidth | None:
    """
    推導單一決策日的移動停損寬度。

    Args:
        bars: 日 K（不會被修改）
        decision_idx: 決策日的位置索引
        horizon: 最長持有交易日數
        pullback_quantile: 取歷史最大回落的哪個分位數
        lookback: 回看幾個視窗
        atr_period: ATR 視窗（僅作稽核對照，不參與推導）

    Returns:
        推導結果；資料不足或含非有限值時回 `None`（**不猜測、不湊數**）
    """
    _validate(bars)
    if horizon < 1:
        raise ValueError(f"horizon 至少為 1，得到 {horizon}")
    if not 0.0 < pullback_quantile < 1.0:
        raise ValueError(
            f"pullback_quantile 必須落在 (0, 1)，得到 {pullback_quantile}"
        )
    if not 0 <= decision_idx < len(bars):
        raise ValueError(
            f"decision_idx {decision_idx} 超出資料範圍 0..{len(bars) - 1}"
        )

    samples = max_pullback_samples(bars, decision_idx, horizon, lookback)
    if samples.size == 0:
        return None

    if not np.isfinite(samples).all():
        # 上游資料洞（停牌、缺漏）會讓 np.quantile 回 NaN。絕不可放行：
        # NaN 不會讓門檻比較拋錯，而是讓 `NaN < 門檻` 為 False，靜默通過。
        return None

    raw = float(np.quantile(samples, pullback_quantile))
    if not np.isfinite(raw) or raw < 0:
        return None

    atr_value = atr(bars, period=atr_period).iloc[decision_idx]
    if not np.isfinite(atr_value):
        return None

    return TrailWidth(
        trail_pct=min(max(raw, MIN_TRAIL_PCT), MAX_TRAIL_PCT),
        raw_trail_pct=raw,
        pullback_quantile=pullback_quantile,
        horizon=horizon,
        lookback=lookback,
        sample_size=int(samples.size),
        atr_value=float(atr_value),
        atr_period=atr_period,
    )
