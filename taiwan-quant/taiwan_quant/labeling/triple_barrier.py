"""
Triple-barrier 標記

把「未來 N 天會漲多少」這個難題，改成「未來 N 天內先碰到目標價還是先碰到
停損價」這個可分類、可回測、可直接對應交易計畫的問題。

依據 ../docs/需求規劃/202609/01_決策紀錄.md 的 D7，以及 CLAUDE.md 的核心建模方式。

規則（逐條對應 CLAUDE.md）：

    T 日收盤決策 → T+1 **開盤**進場
        上柵 = entry × (1 + target_pct)
        下柵 = entry × (1 − stop_pct)
        時間柵 = 進場後 horizon 個交易日

    y = +1  先觸及上柵
        −1  先觸及下柵
         0  時間柵到期都沒觸及

四個容易寫錯的細節（AGENTS.md 會逐條審查）：

1. **用 high/low 判定觸發，不是 close。**用 close 會系統性低估觸發率，
   因為盤中早就碰到柵欄卻沒反映。
2. **同一根 K 同時觸及兩柵 → 保守判 −1。**日線看不出盤中路徑，
   不可假設「先漲到目標才跌」。
3. **進場價是 T+1 開盤，不是 T 日收盤。**T 日的特徵要等收盤才算得出來，
   三大法人資料更要等盤後 15:00–18:00。用 T 日收盤進場屬 look-ahead。
4. **跳空穿越柵欄時以開盤價成交，不是柵欄價。**停損單遇跳空會以市價成交，
   用柵欄價會高估停損執行品質。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

REQUIRED_COLUMNS = ("open", "high", "low", "close")

ExitReason = Literal["upper", "lower", "both", "time"]


@dataclass(frozen=True)
class BarrierSpec:
    """
    柵欄規格。

    target_pct / stop_pct 由呼叫端依 ATR 分位數與歷史相似型態報酬分布推導，
    **不可人工寫死**（CLAUDE.md 核心建模方式）。
    """

    entry_price: float
    target_pct: float
    stop_pct: float
    horizon: int

    def __post_init__(self) -> None:
        if self.entry_price <= 0:
            raise ValueError(f"entry_price 必須為正，得到 {self.entry_price}")
        if self.target_pct <= 0:
            raise ValueError(f"target_pct 必須為正，得到 {self.target_pct}")
        if self.stop_pct <= 0:
            raise ValueError(f"stop_pct 必須為正，得到 {self.stop_pct}")
        if self.horizon < 1:
            raise ValueError(f"horizon 至少為 1，得到 {self.horizon}")

    @property
    def upper_price(self) -> float:
        """上柵（目標價）"""
        return self.entry_price * (1 + self.target_pct)

    @property
    def lower_price(self) -> float:
        """下柵（失效價）"""
        return self.entry_price * (1 - self.stop_pct)

    @property
    def risk_reward(self) -> float:
        """風險報酬比。CLAUDE.md 進場門檻要求 >= 2.0"""
        return self.target_pct / self.stop_pct


@dataclass(frozen=True)
class BarrierLabel:
    """單筆標記結果。不可變，避免下游誤改。"""

    label: int
    """+1 觸上柵 / −1 觸下柵 / 0 時間柵到期"""

    entry_price: float
    """進場價（T+1 開盤）"""

    exit_price: float
    """出場價"""

    gross_return: float
    """毛報酬率（不含交易成本；成本由回測層依 config/costs.py 扣除）"""

    holding_days: int
    """持有交易日數（進場日算第 1 天）"""

    exit_reason: ExitReason
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp


def _validate_bars(bars: pd.DataFrame) -> None:
    """系統邊界驗證：欄位齊全、索引排序"""
    missing = set(REQUIRED_COLUMNS) - set(bars.columns)
    if missing:
        raise ValueError(f"缺少必要欄位：{sorted(missing)}")
    if not bars.index.is_monotonic_increasing:
        raise ValueError("日 K 索引必須依日期升冪排序，否則觸柵順序判斷會錯")


def label_one(
    bars: pd.DataFrame,
    decision_idx: int,
    target_pct: float,
    stop_pct: float,
    horizon: int,
) -> BarrierLabel | None:
    """
    標記單一決策日。

    Args:
        bars: 日 K，index 為日期（升冪），欄位含 open/high/low/close
        decision_idx: 決策日在 bars 中的位置索引（T 日）
        target_pct: 上柵寬度（0.08 = +8%）
        stop_pct: 下柵寬度（0.04 = −4%）
        horizon: 時間柵長度（進場後幾個交易日）

    Returns:
        標記結果；標籤無法確定時回傳 None（**不猜測**）

        「無法確定」只有一種情況：可得的未來 K 棒少於 horizon，**且**
        在這些 K 棒內兩柵都沒觸及。此時 label 究竟是 ±1 還是 0 取決於
        還沒發生的交易日，必須留空。

        反之，若柵欄在可得範圍內已經觸及，label 就是確定的——
        後面還有幾天都不影響結果。

    決策日自己的價格完全不參與判定——它屬於「過去」，只用來算特徵。
    """
    _validate_bars(bars)

    entry_idx = decision_idx + 1
    if entry_idx >= len(bars):
        return None

    last_idx = entry_idx + horizon - 1
    window_complete = last_idx < len(bars)
    window = bars.iloc[entry_idx : min(last_idx, len(bars) - 1) + 1]
    entry_price = float(window["open"].iloc[0])
    spec = BarrierSpec(
        entry_price=entry_price,
        target_pct=target_pct,
        stop_pct=stop_pct,
        horizon=horizon,
    )

    for offset, (bar_date, bar) in enumerate(window.iterrows(), start=1):
        high = float(bar["high"])
        low = float(bar["low"])
        bar_open = float(bar["open"])

        touched_upper = high >= spec.upper_price
        touched_lower = low <= spec.lower_price

        if touched_upper and touched_lower:
            # 同一根 K 同時觸及兩柵：日線看不出盤中路徑，保守判停損
            return _build(spec, spec.lower_price, offset, "both", window, bar_date)

        if touched_upper:
            # 跳空開高穿越上柵時實際成交在開盤價；否則成交在柵欄價（限價單）
            return _build(spec, max(spec.upper_price, bar_open), offset, "upper", window, bar_date)

        if touched_lower:
            # 跳空開低跌破下柵時實際成交在開盤價；不可假設停在柵欄價
            return _build(spec, min(spec.lower_price, bar_open), offset, "lower", window, bar_date)

    # 兩柵都沒觸及。若可得 K 棒不足 horizon，label 取決於還沒發生的交易日 → 留空
    if not window_complete:
        return None

    # 時間柵到期：以最後一根 K 的收盤出場
    final_date = window.index[-1]
    final_close = float(window["close"].iloc[-1])
    return _build(spec, final_close, horizon, "time", window, final_date)


def _build(
    spec: BarrierSpec,
    exit_price: float,
    holding_days: int,
    reason: ExitReason,
    window: pd.DataFrame,
    exit_date: pd.Timestamp,
) -> BarrierLabel:
    """組裝標記結果"""
    label_map: dict[ExitReason, int] = {"upper": 1, "lower": -1, "both": -1, "time": 0}
    return BarrierLabel(
        label=label_map[reason],
        entry_price=spec.entry_price,
        exit_price=exit_price,
        gross_return=exit_price / spec.entry_price - 1,
        holding_days=holding_days,
        exit_reason=reason,
        entry_date=window.index[0],
        exit_date=exit_date,
    )


def label_series(
    bars: pd.DataFrame,
    target_pct: float,
    stop_pct: float,
    horizon: int,
) -> pd.DataFrame:
    """
    批次標記所有可行的決策日。

    Args:
        bars: 日 K（不會被修改）
        target_pct / stop_pct / horizon: 同 label_one

    Returns:
        DataFrame，**index 為決策日**（不是進場日）。

        索引對齊決策日是關鍵：特徵在決策日計算，label 必須對齊決策日
        才能訓練。對齊錯就等於 look-ahead（CLAUDE.md 禁令 1）。

    未來 K 棒不足的尾端決策日會被略過。
    """
    _validate_bars(bars)

    records: list[dict[str, object]] = []
    index: list[pd.Timestamp] = []

    # 需要 1 根進場 + horizon 根，所以最後一個可行決策日是 len - horizon - 1
    for decision_idx in range(len(bars) - horizon):
        result = label_one(bars, decision_idx, target_pct, stop_pct, horizon)
        if result is None:
            continue
        index.append(bars.index[decision_idx])
        records.append(
            {
                "label": result.label,
                "entry_price": result.entry_price,
                "exit_price": result.exit_price,
                "gross_return": result.gross_return,
                "holding_days": result.holding_days,
                "exit_reason": result.exit_reason,
                "entry_date": result.entry_date,
                "exit_date": result.exit_date,
            }
        )

    return pd.DataFrame(records, index=pd.DatetimeIndex(index, name=bars.index.name))
