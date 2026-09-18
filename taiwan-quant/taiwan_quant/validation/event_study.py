"""
事件研究：條件在「事件已經發生」之後的前推報酬

## 為什麼要換成事件法

排序法的檢定力已經用盡。實測 N=10、H=40、36 期，每趟 sd ≈ 15.4 pp，
2 SE 能偵測的最小效應是 **5.0%／趟**，而策略實測效應是 3.1%／趟——
**低於偵測門檻**。要證明 3%／趟需要 105 期 ≈ 17 年，而開發集只有 5.7 年。

事件法改變的是**觀測單位**：

```
排序法   36 個決策期          最小可測效應 ≈ 5.0%／趟
事件法   ~1,500 個交易日      SE 縮小 √(1500/36) ≈ 6.5 倍
                            最小可測效應 ≈ 0.8%／趟
```

同樣的資料，換一個切法就多出 6 倍檢定力。**這不是換一個策略，
是換一個問法。**

## 觀測單位必須是「日」而不是「股票-事件」

漲停在橫斷面上會叢聚——大盤噴的那天幾十檔一起漲停，它們的後續報酬
高度相關。把每個 (股票, 日) 當獨立樣本會把 n 虛增幾十倍，標準誤跟著
虛減，什麼都會變顯著。

所以：**先算每日事件組的平均，再把「日」當樣本。**

## 必須配對同日全池

事件叢聚在強勢market。若拿事件組的絕對報酬去跟長期平均比，量到的會是
「強勢market之後大盤漲」而不是「事件之後這些股票漲」。

所以每日都減掉**同一天全池的平均報酬**。那消掉大盤共同變動，
剩下的才是事件的橫斷面超額。

## 進場價必須是 T+1 開盤

漲停當天收盤買不到——那是漲停。隔天開盤會跳空，而**跳空就是這個
策略的主要成本**。用 `opens.shift(-1)` 進場把那個跳空算進來了，
不可以改用 T 日收盤，那會憑空賺到跳空幅度。

```
forward[T] = closes[T + horizon] / opens[T + 1] - 1
```

與 `data/integrity.holding_dates` 同一個慣例。

## 這個模組不做什麼

不挑格子。事件 × 持有期是一張網格，**從裡面挑最好的一格就是過擬合**。
輸出要整張網格一起看，並附多重測試的代價。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


class EventStudyError(ValueError):
    """事件研究的輸入不合法。"""


@dataclass(frozen=True)
class EventStudyResult:
    """一個 (事件, 持有期) 組合的逐日配對結果。"""

    event: str
    horizon: int

    n_events: int
    """事件發生的 (股票, 日) 總數。**不是樣本數**"""

    n_dates: int
    """有事件的交易日數。**這才是樣本數**"""

    event_mean: float
    """事件組的逐日平均報酬，再對日取平均"""

    baseline_mean: float
    """同日全池的平均報酬，再對日取平均"""

    paired: tuple[float, ...]
    """逐日配對差異（事件組 − 同日全池）。bootstrap 吃這個"""

    @property
    def excess(self) -> float:
        """平均超額。逐日配對差異的平均"""
        if not self.paired:
            return float("nan")
        return float(np.mean(self.paired))

    @property
    def standard_error(self) -> float:
        if len(self.paired) < 2:
            return float("nan")
        return float(np.std(self.paired, ddof=1) / np.sqrt(len(self.paired)))

    @property
    def t_stat(self) -> float:
        se = self.standard_error
        if not np.isfinite(se) or se == 0.0:
            return float("nan")
        return self.excess / se

    @property
    def events_per_date(self) -> float:
        if self.n_dates == 0:
            return float("nan")
        return self.n_events / self.n_dates

    def describe(self) -> str:
        return (
            f"{self.event}｜H={self.horizon}"
            f"｜事件 {self.n_events} 筆／{self.n_dates} 日"
            f"（每日 {self.events_per_date:.1f} 檔）"
            f"｜超額 {self.excess:+.3%}"
            f"｜SE {self.standard_error:.3%}"
            f"｜t = {self.t_stat:+.2f}"
        )


def forward_returns(
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    *,
    horizon: int,
) -> pd.DataFrame:
    """
    T+1 開盤進、T+horizon 收盤出。

    **與 `data/integrity.holding_dates` 同一個慣例。** 2026-09-18 之前
    主線有三個檔案寫成 `shift(-1 - horizon)`，多持有一天，那個錯誤
    推翻了當時的主要結論——所以這裡只有一份實作。

    Raises:
        EventStudyError: horizon < 1，或兩張表的形狀不一致
    """
    if horizon < 1:
        raise EventStudyError(f"horizon 至少為 1，得到 {horizon}")
    if opens.shape != closes.shape:
        raise EventStudyError(
            f"opens 與 closes 形狀必須相同，得到 {opens.shape} 與 {closes.shape}"
        )
    return closes.shift(-horizon) / opens.shift(-1) - 1.0


def per_date_mean(
    returns: pd.DataFrame, mask: pd.DataFrame
) -> pd.Series:
    """
    每一天，取 `mask` 為真且報酬有值的那些股票的平均報酬。

    兩張表必須同索引同欄位。當天沒有任何有效樣本的日期回 NaN，
    由呼叫端決定要不要丟掉——**不要在這裡靜默跳過**。
    """
    if returns.shape != mask.shape:
        raise EventStudyError(
            f"報酬與遮罩形狀必須相同，得到 {returns.shape} 與 {mask.shape}"
        )
    selected = returns.where(mask.astype(bool))
    return selected.mean(axis=1, skipna=True)


def event_study(
    *,
    event: str,
    horizon: int,
    returns: pd.DataFrame,
    event_mask: pd.DataFrame,
    universe_mask: pd.DataFrame,
) -> EventStudyResult:
    """
    逐日配對：事件組平均 − 同日全池平均。

    Args:
        event: 事件名稱，只用於報告
        horizon: 持有期，只用於報告（`returns` 必須已經是該持有期）
        returns: forward 報酬，由 `forward_returns` 產生
        event_mask: 事件是否在 (股票, 日) 發生
        universe_mask: 該 (股票, 日) 是否在當時標的池內（禁令 2）

    Returns:
        `EventStudyResult`。只保留**事件組與全池當日都有值**的日期——
        少了任一邊就無法配對，而用不同天的兩個平均相減是錯的。

    Raises:
        EventStudyError: 形狀不一致
    """
    in_event = event_mask.astype(bool) & universe_mask.astype(bool)
    in_universe = universe_mask.astype(bool)

    event_daily = per_date_mean(returns, in_event)
    base_daily = per_date_mean(returns, in_universe)

    both = event_daily.notna() & base_daily.notna()
    paired = (event_daily[both] - base_daily[both]).to_numpy(dtype=float)

    return EventStudyResult(
        event=event,
        horizon=horizon,
        n_events=int((in_event & returns.notna()).to_numpy().sum()),
        n_dates=int(both.sum()),
        event_mean=float(event_daily[both].mean()) if both.any() else float("nan"),
        baseline_mean=float(base_daily[both].mean()) if both.any() else float("nan"),
        paired=tuple(paired.tolist()),
    )


def expanding_quantile_mask(
    values: pd.DataFrame,
    *,
    quantile: float,
    refresh_every: int,
    min_observations: int,
) -> pd.DataFrame:
    """
    只用 ≤ t 的資料算分位門檻，標出超過門檻的 (股票, 日)。（禁令 1）

    用全期分布定義門檻是 look-ahead——那等於先知道整段歷史的值域。
    門檻移動很慢，所以每 `refresh_every` 個交易日重算一次；
    重算時只用當日及之前的資料。

    Args:
        values: 要取分位的值（例如外資買超比率）
        quantile: 分位，(0, 1)
        refresh_every: 幾個交易日重算一次門檻
        min_observations: 門檻生效所需的最少歷史筆數

    Raises:
        EventStudyError: 分位不在 (0, 1)，或 refresh_every < 1
    """
    if not 0.0 < quantile < 1.0:
        raise EventStudyError(f"分位必須在 (0, 1) 之間，得到 {quantile}")
    if refresh_every < 1:
        raise EventStudyError(f"refresh_every 必須為正，得到 {refresh_every}")

    flat = values.to_numpy(dtype=float)
    thresholds = np.full(len(values), np.nan)
    current = np.nan
    seen: list[float] = []

    for position in range(len(values)):
        row = flat[position]
        if position % refresh_every == 0:
            finite = np.asarray(seen, dtype=float)
            finite = finite[np.isfinite(finite)]
            if finite.size >= min_observations:
                current = float(np.quantile(finite, quantile))
        thresholds[position] = current
        seen.extend(row[np.isfinite(row)].tolist())

    threshold_frame = pd.DataFrame(
        np.repeat(thresholds[:, None], values.shape[1], axis=1),
        index=values.index,
        columns=values.columns,
    )
    return (values > threshold_frame) & threshold_frame.notna()
