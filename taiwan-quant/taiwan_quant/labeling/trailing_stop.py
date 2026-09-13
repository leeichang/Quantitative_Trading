"""
移動停損標記（路線 A）

## 為什麼換掉 triple-barrier

樣本外驗證結論（`docs/需求規劃/202609/03_策略可行性最終結論.md`）：

    9 組策略全部輸給等權買進持有，差距達 85 個百分點。

機制原因很明確：**triple-barrier 的目標價把上檔封死了。**

    目標 +19.6%（40 日）→ 碰到就出場
    但同期市場漲了 262%

每次觸及目標就出場，就錯過後續續漲。強勢多頭裡，任何「設定目標價就走」
的策略都會系統性輸給單純持有。

## 解法

拿掉上柵，改成移動停損：

    初始停損 = 進場價 × (1 − trail_pct)
    每天更新 = max(現有停損, 期間最高價 × (1 − trail_pct))   ← **只升不降**
    出場     = 盤中觸及停損線，或時間柵到期

上檔不封死 → 趨勢延續時一路跟上；回檔超過 trail_pct → 鎖住已實現獲利。

這是 Kimi 在原始對話的建議（見 `來源原文/06_Kimi`）：

    改採「拉回分批買 + 紀律停損」的波段計畫即可。

## 與 triple-barrier 共通的規則

換標記方式不改變這四條：

    1. 進場是 **T+1 開盤**（T 日特徵要收盤後才算得出來）
    2. 用 high/low 判定觸發，不是 close
    3. 跳空穿越停損時以**開盤價**成交（不可假設停在停損價）
    4. 結果無法確定時回 `None`，不猜

## 標籤語意的差異

triple-barrier 是三分類（先碰上柵 / 下柵 / 到期）。
移動停損沒有固定目標，所以標籤是**報酬的正負號**：

    +1   實際報酬 > 0
    −1   實際報酬 <= 0

報酬為 0 視為未獲利（保守）。校準層要改用「每箱的平均實際報酬」
而非「命中率」——沒有固定目標時，命中率不足以描述期望值。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

REQUIRED_COLUMNS = ("open", "high", "low", "close")

ExitReason = Literal["trailing_stop", "time"]


@dataclass(frozen=True)
class TrailingSpec:
    """
    移動停損規格。

    `trail_pct` 由 ATR 推導（同 `barrier_width.derive_width` 的做法），
    不可人工寫死。
    """

    entry_price: float
    trail_pct: float
    max_horizon: int
    """最長持有交易日數。時間柵仍然存在，但它是上限而非主要出場機制"""

    def __post_init__(self) -> None:
        if self.entry_price <= 0:
            raise ValueError(f"entry_price 必須為正，得到 {self.entry_price}")
        if not 0.0 < self.trail_pct < 1.0:
            raise ValueError(f"trail_pct 必須落在 (0, 1)，得到 {self.trail_pct}")
        if self.max_horizon < 1:
            raise ValueError(f"max_horizon 至少為 1，得到 {self.max_horizon}")

    @property
    def initial_stop(self) -> float:
        """進場時的停損線"""
        return self.entry_price * (1 - self.trail_pct)

    def stop_from_peak(self, peak: float) -> float:
        """由期間最高價推算的停損線"""
        return peak * (1 - self.trail_pct)


@dataclass(frozen=True)
class TrailingExit:
    """單筆移動停損的出場結果"""

    label: int
    """+1 獲利 / −1 未獲利（報酬 <= 0）"""

    entry_price: float
    exit_price: float
    gross_return: float
    """毛報酬率（不含交易成本；成本由回測層依 config/costs.py 扣除）"""

    highest_price: float
    """
    期間最高價。

    必須記錄——否則無法回答「當時停損為什麼設在那個價位」
    （CLAUDE.md 禁令 7、8 要求可稽核）。
    """

    holding_days: int
    exit_reason: ExitReason
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp


def _validate(bars: pd.DataFrame) -> None:
    missing = set(REQUIRED_COLUMNS) - set(bars.columns)
    if missing:
        raise ValueError(f"缺少必要欄位：{sorted(missing)}")
    if not bars.index.is_monotonic_increasing:
        raise ValueError("日 K 索引必須依日期升冪排序，否則觸發順序判斷會錯")


def label_trailing(
    bars: pd.DataFrame,
    decision_idx: int,
    trail_pct: float,
    max_horizon: int,
) -> TrailingExit | None:
    """
    標記單一決策日的移動停損結果。

    Args:
        bars: 日 K（**不會被修改**），index 為日期（升冪）
        decision_idx: 決策日的位置索引（T 日）
        trail_pct: 移動停損幅度（0.10 = 從高點回落 10% 出場）
        max_horizon: 最長持有交易日數

    Returns:
        出場結果；標籤無法確定時回 `None`。

        「無法確定」只有一種情況：可得的未來 K 棒少於 `max_horizon`，
        **且**在這些 K 棒內未觸及停損。此時結果取決於還沒發生的交易日。

        反之，停損若已在可得範圍內觸及，結果就是確定的。

    決策日自己的價格完全不參與判定——它屬於過去，只用來算特徵。
    """
    _validate(bars)

    entry_idx = decision_idx + 1
    if entry_idx >= len(bars):
        return None

    last_idx = entry_idx + max_horizon - 1
    window_complete = last_idx < len(bars)
    window = bars.iloc[entry_idx : min(last_idx, len(bars) - 1) + 1]

    entry_price = float(window["open"].iloc[0])
    spec = TrailingSpec(
        entry_price=entry_price, trail_pct=trail_pct, max_horizon=max_horizon
    )

    peak = entry_price
    stop = spec.initial_stop

    for offset, (bar_date, bar) in enumerate(window.iterrows(), start=1):
        bar_open = float(bar["open"])
        bar_high = float(bar["high"])
        bar_low = float(bar["low"])

        # 先判定出場：當日的高點還沒發生時，停損線仍是昨日收盤後的水準。
        # 若先用今日高點更新停損再判定，等於用當日盤中最高點回頭保護當日
        # 的低點——那是 look-ahead。
        if bar_low <= stop:
            # 跳空跌破時以開盤價成交，不可假設停在停損線
            exit_price = min(stop, bar_open)
            return _build(spec, exit_price, peak, offset, "trailing_stop", window, bar_date)

        # 未出場才更新高點與停損（只升不降）
        if bar_high > peak:
            peak = bar_high
            stop = max(stop, spec.stop_from_peak(peak))

    # 未觸停損。可得 K 棒不足 horizon 時結果未定 → 留空
    if not window_complete:
        return None

    final_date = window.index[-1]
    final_close = float(window["close"].iloc[-1])
    return _build(spec, final_close, peak, max_horizon, "time", window, final_date)


def _build(
    spec: TrailingSpec,
    exit_price: float,
    peak: float,
    holding_days: int,
    reason: ExitReason,
    window: pd.DataFrame,
    exit_date: pd.Timestamp,
) -> TrailingExit:
    """組裝出場結果"""
    gross_return = exit_price / spec.entry_price - 1
    return TrailingExit(
        label=1 if gross_return > 0 else -1,
        entry_price=spec.entry_price,
        exit_price=exit_price,
        gross_return=gross_return,
        highest_price=peak,
        holding_days=holding_days,
        exit_reason=reason,
        entry_date=window.index[0],
        exit_date=exit_date,
    )


def label_trailing_series(
    bars: pd.DataFrame,
    trail_pct: float,
    max_horizon: int,
) -> pd.DataFrame:
    """
    批次標記所有可行的決策日。

    Returns:
        DataFrame，**index 為決策日**（不是進場日）。

        索引對齊決策日是關鍵：特徵在決策日計算，label 必須對齊決策日
        才能訓練。對齊錯就等於 look-ahead（CLAUDE.md 禁令 1）。
    """
    _validate(bars)

    records: list[dict[str, object]] = []
    index: list[pd.Timestamp] = []

    for decision_idx in range(len(bars)):
        result = label_trailing(bars, decision_idx, trail_pct, max_horizon)
        if result is None:
            continue
        index.append(bars.index[decision_idx])
        records.append(
            {
                "label": result.label,
                "entry_price": result.entry_price,
                "exit_price": result.exit_price,
                "gross_return": result.gross_return,
                "highest_price": result.highest_price,
                "holding_days": result.holding_days,
                "exit_reason": result.exit_reason,
                "entry_date": result.entry_date,
                "exit_date": result.exit_date,
            }
        )

    return pd.DataFrame(records, index=pd.DatetimeIndex(index, name=bars.index.name))
