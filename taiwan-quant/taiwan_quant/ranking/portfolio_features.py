"""
投組約束的四個輸入，全部在 T 日可得

## 為什麼要這個模組

`constraints.py` 定義了約束**規則**（同產業 ≤ 2、高波動 ≤ 1、
相關係數 < 0.7、至少 1 支 defensive），但它的 `PortfolioCandidate`
Protocol 只宣告「候選要有這些屬性」，沒有說**那些屬性怎麼算出來**。

舊路徑（`portfolio.py` / `trailing_portfolio.py`）各自在上游填好再傳進去。
動能突破 N=10 走的是完全不同的選股流程（原始分數排序），所以需要一份
共用的、**時點安全**的計算。寫在腳本裡的話下一個腳本又會重寫一次。

## 四個輸入與它們的時點紀律（禁令 1）

```
industry        stock_master.industry      ⚠️ 現行分類，不是時點快照（見下）
atr_percentile  ATR(14)/close 的橫斷面分位   只用 ≤ T 的 K 線
beta            對 0050 的 250 日迴歸係數    只用 ≤ T 的報酬
correlations    60 日日報酬的兩兩相關        只用 ≤ T 的報酬
```

### ⚠️ `industry` 不是時點資料

`stock_master` 是一張**現況表**，沒有產業分類的生效日期。所以 2017 年的
決策會看到該公司 **2026 年**的產業分類。

這是一個已知的時代錯置，不隱藏。判斷它可接受的理由：

- 產業分類是**風險分群**，不是報酬預測變數。它的用途是「不要讓 6 檔
  金融股同時入選」，而不是「金融股會漲」
- 台股產業分類極少變動，且變動時通常是新增細類（如「其他電子類」拆出
  「電子零組件業」），不是跨大類搬家

但它仍然是禁令 1 的一個缺口。**拿到帶生效日的分類資料後應該換掉。**

### ATR 必須先除以股價

ATR 的單位是新台幣。1560（790 元）的 ATR 天生就比 2801（27.9 元）大
一個數量級——直接取橫斷面分位等於在排序股價高低，不是排序波動度。

所以一律用 `ATR(14) / close`（當日波動佔股價的比例）再取分位。

### 相關係數缺值一律拒絕

`constraints.py` 的 `CORRELATION_UNKNOWN` 是保守拒絕：查不到就不讓它進。
本模組只回報**算得出來的**配對，缺的就讓它缺——把缺值填 0 等於宣稱
「這兩檔無關」，那會讓高度相關的標的同時入選，正是約束要防的事。
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from taiwan_quant.labeling.barrier_width import atr

ATR_PERIOD = 14
BETA_LOOKBACK = 250
BETA_BENCHMARK = "0050"
CORRELATION_LOOKBACK = 60

MIN_BETA_OVERLAP = 120
"""beta 至少要這麼多筆共同交易日，否則不算（回 NaN → 視為非 defensive）"""

ATR_RATIO_DECIMALS = 6
"""
取分位前先把 ATR% 量化到小數 6 位。

浮點乘法會讓數學上相同的兩個 ATR% 差 ~1e-16，而 `rank()` 把它們當成
不同值，於是波動度排名多出一個**沒有資訊的先後**——`HIGH_VOLATILITY_PERCENTILE`
剛好切在中間時，這個雜訊會決定誰佔用高波動額度。

這與 `tie_break.py` 修的是同一類問題：分不出來的時候不要假裝分得出來。
ATR% 實測落在 0.005 ~ 0.10，1e-6 的顆粒度遠細於任何有意義的差異。
"""


class PortfolioFeatureError(RuntimeError):
    """約束輸入無法計算"""


def load_industries(db_path: Path) -> dict[str, str]:
    """
    讀 `stock_master.industry`。

    Returns:
        `stock_id` 到產業名稱。**沒有分類的股票不會出現在回傳值裡**——
        讓呼叫端自己決定怎麼處理缺值，比在這裡填 "未分類" 好：
        "未分類" 會被當成一個真的產業，然後所有缺值股票互相擠掉。

    實測 2,597 檔裡 2,551 檔（98.2%）有分類。
    """
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as con:
        rows = con.execute(
            "SELECT stock_id, industry FROM stock_master "
            "WHERE industry IS NOT NULL AND TRIM(industry) <> ''"
        ).fetchall()
    return {stock_id: industry for stock_id, industry in rows}


def atr_ratio_frame(
    by_stock: dict[str, pd.DataFrame], period: int = ATR_PERIOD
) -> pd.DataFrame:
    """
    每檔每日的 ATR% = ATR(period) / close。

    Args:
        by_stock: 每檔的日 K（需含 high/low/close）
        period: ATR 視窗

    Returns:
        index 為日期、column 為 `stock_id` 的矩陣。視窗不足處為 NaN。

    **這是整段歷史一次算完的版本**，因為 `atr()` 每次呼叫都會算完整序列
    再取最後一筆——逐日呼叫是 O(天數²)。診斷要掃 ~50 個決策日 × 150 檔，
    逐日版本會慢兩個數量級。

    ⚠️ 逐日與整段兩個版本**不可各自實作 ATR%**，否則遲早有一份會漂移。
    `atr_percentiles()` 直接從這裡取值。
    """
    ratios = {}
    for stock_id, bars in by_stock.items():
        series = atr(bars, period=period)
        close = bars["close"].astype(float)
        ratios[stock_id] = (series / close.where(close > 0)).round(
            ATR_RATIO_DECIMALS
        )
    return pd.DataFrame(ratios)


def percentiles_at(
    ratio_frame: pd.DataFrame, day: pd.Timestamp, members: tuple[str, ...]
) -> dict[str, float]:
    """
    把 T 日的 ATR% 轉成橫斷面分位。

    Args:
        ratio_frame: `atr_ratio_frame` 的輸出
        day: 決策日
        members: 參與橫斷面比較的標的池

    Returns:
        `stock_id` 到分位（0~1）。缺值的不會出現。

    分位是**在 `members` 之內**排名，不是全市場。理由：約束問的是
    「這一檔在今天的候選池裡算不算高波動」，拿它跟池外的股票比沒有意義。

    `rank(pct=True)` 預設 `method="average"`，所以量化後相同的 ATR% 會
    共享同一個分位——分不出來的時候不排出先後。
    """
    present = [sid for sid in members if sid in ratio_frame.columns]
    if not present or day not in ratio_frame.index:
        return {}
    row = ratio_frame.loc[day, present].dropna()
    if row.empty:
        return {}
    ranked = row.rank(pct=True)
    return {str(k): float(v) for k, v in ranked.items()}


def atr_percentiles(
    by_stock: dict[str, pd.DataFrame],
    day: pd.Timestamp,
    members: tuple[str, ...],
    period: int = ATR_PERIOD,
) -> dict[str, float]:
    """
    T 日的 ATR% 橫斷面分位（單日便利版，供實際推播使用）。

    Args:
        by_stock: 每檔的日 K（需含 high/low/close）
        day: 決策日。**只用 ≤ day 的 K 線**
        members: 參與橫斷面比較的標的池
        period: ATR 視窗

    Returns:
        `stock_id` 到分位（0~1）。算不出 ATR 的不會出現。

    推播只有一天、10 檔，所以這裡為了時點紀律明確而先截斷再算：
    `bars.loc[:day]` 保證不可能看到 T 之後的 K 線。回測請用
    `atr_ratio_frame` + `percentiles_at`，兩者共用同一段計算。
    """
    truncated = {
        stock_id: bars.loc[:day]
        for stock_id, bars in by_stock.items()
        if stock_id in members and len(bars.loc[:day]) > period
    }
    if not truncated:
        return {}
    frame = atr_ratio_frame(truncated, period=period)
    return percentiles_at(frame, frame.index[-1], members)


def betas(
    closes: pd.DataFrame,
    day: pd.Timestamp,
    members: tuple[str, ...],
    benchmark: str = BETA_BENCHMARK,
    lookback: int = BETA_LOOKBACK,
) -> dict[str, float]:
    """
    對大盤（預設 0050）的 beta，用 ≤ day 的 `lookback` 個交易日日報酬。

    Args:
        closes: 還原收盤價矩陣（index 為交易日，column 為 stock_id）
        day: 決策日
        members: 要算的標的
        benchmark: 基準標的
        lookback: 報酬視窗長度

    Returns:
        `stock_id` 到 beta。共同交易日不足 `MIN_BETA_OVERLAP` 的不會出現。

    Raises:
        PortfolioFeatureError: `closes` 裡沒有 `benchmark`

    beta = Cov(r_i, r_m) / Var(r_m)，用普通最小平方的閉式解，不引進
    statsmodels——這裡不需要標準誤與 p 值。
    """
    if benchmark not in closes.columns:
        raise PortfolioFeatureError(
            f"closes 缺少基準標的 {benchmark}；beta 無法計算。"
            "載入價格時必須一併載入基準（ETF 一律載入，見 etf_universe）"
        )

    window = closes.loc[:day].tail(lookback + 1)
    returns = window.pct_change().iloc[1:]
    market = returns[benchmark]
    market_var = float(market.var(ddof=1))
    if not np.isfinite(market_var) or market_var <= 0:
        return {}

    result: dict[str, float] = {}
    for stock_id in members:
        if stock_id not in returns.columns or stock_id == benchmark:
            continue
        pair = pd.concat([returns[stock_id], market], axis=1).dropna()
        if len(pair) < MIN_BETA_OVERLAP:
            continue
        own, mkt = pair.iloc[:, 0], pair.iloc[:, 1]
        variance = float(mkt.var(ddof=1))
        if variance <= 0:
            continue
        result[stock_id] = float(own.cov(mkt)) / variance
    return result


def return_correlations(
    closes: pd.DataFrame,
    day: pd.Timestamp,
    members: tuple[str, ...],
    lookback: int = CORRELATION_LOOKBACK,
) -> dict[tuple[str, str], float]:
    """
    ≤ day 的 `lookback` 日日報酬兩兩相關係數。

    Args:
        closes: 還原收盤價矩陣
        day: 決策日
        members: 要算的標的
        lookback: 報酬視窗長度

    Returns:
        `(a, b)` 到相關係數，**只存一個方向**（`constraints.correlation()`
        兩個方向都查）。算不出來的配對不會出現。

    ⚠️ CLAUDE.md 寫的是「60 日報酬相關係數」。這裡用**60 個交易日的日
    報酬序列**算相關，不是「60 日持有期報酬」的相關——後者在同一個決策
    日上每檔只有一個數，算不出相關係數。
    """
    present = tuple(sid for sid in members if sid in closes.columns)
    if len(present) < 2:
        return {}

    window = closes.loc[:day, list(present)].tail(lookback + 1)
    returns = window.pct_change().iloc[1:]
    matrix = returns.corr(min_periods=max(2, lookback // 2))

    result: dict[tuple[str, str], float] = {}
    for i, a in enumerate(present):
        for b in present[i + 1:]:
            rho = matrix.at[a, b]
            if np.isfinite(rho):
                result[(a, b)] = float(rho)
    return result


@dataclass(frozen=True)
class RankedCandidate:
    """
    動能突破的候選，實作 `constraints.PortfolioCandidate`。

    刻意**不帶** `target_pct` / `prob_up` / `trail_pct`——那些是校準器與
    移動停損的概念，動能突破沒有。帳本改版時已經學過一次：硬留欄位就得
    填假值，而假值比缺值更糟。
    """

    stock_id: str
    industry: str
    rank_score: float
    atr_percentile: float
    beta: float

    @property
    def is_high_volatility(self) -> bool:
        from taiwan_quant.ranking.constraints import HIGH_VOLATILITY_PERCENTILE

        return self.atr_percentile > HIGH_VOLATILITY_PERCENTILE

    @property
    def is_defensive(self) -> bool:
        from taiwan_quant.ranking.constraints import DEFENSIVE_BETA_MAX

        return self.beta < DEFENSIVE_BETA_MAX


UNCLASSIFIED = "未分類"
"""
產業缺值時的佔位值。

**它是一個真的產業標籤**，所以「同產業 ≤ 2」也會套用到它——這是刻意的
保守做法：分類不明的股票不該無限量入選。
"""


def build_candidates(
    ordered: list[str],
    scores: dict[str, float],
    industries: dict[str, str],
    atr_pct: dict[str, float],
    beta_map: dict[str, float],
) -> list[RankedCandidate]:
    """
    把已排序的代號組成候選清單。

    Args:
        ordered: 已依分數排序的 `stock_id`（含 tie-break）
        scores: 原始分數
        industries: `load_industries` 的輸出
        atr_pct: `atr_percentiles` 的輸出
        beta_map: `betas` 的輸出

    Returns:
        與 `ordered` 同順序的候選。**順序必須保留**——`greedy_pick`
        依序貪婪挑選，重排等於改變選股結果。

    缺值處理：
        產業缺 → `UNCLASSIFIED`（仍受同產業上限約束）
        ATR 缺 → 0.0（視為低波動，不佔用高波動額度）
        beta 缺 → `inf`（視為非 defensive，不能用來滿足 defensive 要求）

    ATR 與 beta 的缺值方向刻意相反：兩者都選「不讓缺值取得好處」的那一邊。
    ATR 缺值若填 1.0 會讓它被當成高波動而佔用額度（過度懲罰）；beta 缺值
    若填 0 會讓它假裝是 defensive 而滿足約束（放過真正的風險）。
    """
    return [
        RankedCandidate(
            stock_id=stock_id,
            industry=industries.get(stock_id, UNCLASSIFIED),
            rank_score=float(scores[stock_id]),
            atr_percentile=atr_pct.get(stock_id, 0.0),
            beta=beta_map.get(stock_id, float("inf")),
        )
        for stock_id in ordered
    ]
