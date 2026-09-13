"""
對照組

CLAUDE.md 要求每份回測報告都要並列：

    等權買進持有（同一標的池）
    0050 買進持有
    隨機進場（100 次模擬取中位數）

**若策略在 OOS 跑不贏前兩者，就如實寫進報告，不要調參數到看起來贏。**

## 為什麼要獨立成模組並寫測試

對照組算錯比沒有對照更糟——它讓錯誤的結論看起來經過驗證。

實測踩過的兩個坑：

    期間錯位   買進持有只涵蓋 7 個月、策略涵蓋 2 年多
    位置對齊   各檔起始日不同，用位置索引取起點會落在不同日期

兩者都不會拋錯，只會讓數字看起來很正常。

## 0050 對照的現況

上游資料庫不含 ETF，目前只能用等權組合代理。取得 0050 日線後應補上。
"""

from __future__ import annotations

import math

import pandas as pd


def buy_and_hold_return(
    by_stock: dict[str, pd.DataFrame],
    start: pd.Timestamp,
    end: pd.Timestamp,
    price_column: str = "close",
) -> float | None:
    """
    等權買進持有：`start` 買進、`end` 賣出。

    Args:
        by_stock: {股票代號: 日 K}，index 為日期
        start / end: 期間端點（含）。**必須與策略的實際交易期間對齊**
        price_column: 取哪個價格欄位

    Returns:
        等權平均報酬率；期間內沒有任何標的可算時回 `None`。

        回 `None` 而非 0 或 NaN 是刻意的：0 會被誤讀成「買進持有剛好
        打平」，讓策略看起來贏了對照組。

    Raises:
        ValueError: start 晚於 end

    以**日期**對齊而非位置索引——各檔起始日不同，位置索引會落在不同日期。
    """
    if start > end:
        raise ValueError(f"start {start} 不可晚於 end {end}")

    returns: list[float] = []
    for bars in by_stock.values():
        window = bars.loc[(bars.index >= start) & (bars.index <= end)]
        if len(window) < 2:
            # 一根 K 算不出報酬。該檔跳過，不可當成 0%
            continue
        entry = float(window[price_column].iloc[0])
        exit_price = float(window[price_column].iloc[-1])
        if entry <= 0 or not math.isfinite(entry) or not math.isfinite(exit_price):
            continue
        returns.append(exit_price / entry - 1.0)

    if not returns:
        return None
    return sum(returns) / len(returns)


def effective_samples(
    decision_dates: list[pd.Timestamp],
    horizon: int,
    stride: int,
) -> int:
    """
    有效獨立樣本數。

    Args:
        decision_dates: 所有決策日（可含重複，同一日選多檔算一個樣本）
        horizon: 持有交易日數
        stride: 決策間隔交易日數

    Returns:
        不重疊的樣本數

    持有 `horizon` 天但每 `stride` 天決策一次 → 持有期重疊。
    每 `horizon / stride` 個決策日才有一個不重疊的樣本。

    報告必須同時列出總筆數與有效樣本數——**用總筆數算統計顯著性會
    嚴重高估**。持有 60 日、每 5 日決策的 12 筆「樣本」其實只有 1 筆
    獨立資訊。
    """
    if horizon < 1:
        raise ValueError(f"horizon 至少為 1，得到 {horizon}")
    if stride < 1:
        raise ValueError(f"stride 至少為 1，得到 {stride}")

    unique = sorted(set(decision_dates))
    if not unique:
        return 0

    step = max(1, horizon // stride)
    return len(unique[::step])


def equal_weight_equity(
    by_stock: dict[str, pd.DataFrame],
    calendar: list[pd.Timestamp],
    price_column: str = "close",
) -> pd.Series:
    """
    等權買進持有的**逐日權益曲線**（起點 1.0）。

    Args:
        by_stock: {股票代號: 日 K}
        calendar: 模擬日曆（升冪）
        price_column: 取哪個價格欄位

    Returns:
        與 calendar 同長度的 Series

    Raises:
        ValueError: 日曆為空，或期初沒有任何標的可買

    只買**期初就存在**的標的，之後不調倉。期間內缺某天報價時沿用
    前一日價格——停牌不等於歸零。

    為什麼要曲線而不只是總報酬：對照組的最大回撤要能跟策略比。
    只有總報酬時，「策略 +50% / 對照 +60%」看起來只差 10 個百分點，
    但若對照組中途回撤 40%、策略只回撤 8%，那是完全不同的東西。
    """
    if not calendar:
        raise ValueError("交易日曆不可為空")

    start = calendar[0]
    bases: dict[str, float] = {}
    for stock_id, bars in by_stock.items():
        window = bars.loc[bars.index <= start]
        if window.empty:
            continue
        price = float(window[price_column].iloc[-1])
        if price > 0 and math.isfinite(price):
            bases[stock_id] = price

    if not bases:
        raise ValueError(f"期初 {start} 沒有任何標的可買")

    # 逐檔重取樣到日曆，缺值沿用前一日（停牌不等於歸零）
    ratios = []
    for stock_id, base in bases.items():
        series = by_stock[stock_id][price_column]
        aligned = series.reindex(pd.DatetimeIndex(calendar), method="ffill")
        ratios.append(aligned / base)

    return pd.concat(ratios, axis=1).mean(axis=1)


ETF_BENCHMARKS = ("0050", "0051", "0056")
"""
CLAUDE.md 明訂「0050 買進持有」是必跑對照組。

長歷史回補之前上游資料庫不含 ETF，所以前四輪驗證都只有等權買進持有。
回補後 0050 / 0051 / 0056 都有完整 2015~2026 資料（含還原價），
這個缺口沒有理由再留著。

**0050 是比等權更嚴格的對照**：它是可以真的買到的單一標的。
等權持有 150 檔需要每季調倉、付 150 次交易成本；0050 買一次就好。
策略連它都贏不了，就沒有存在的理由。
"""


def single_asset_equity(
    bars: pd.DataFrame,
    calendar: list[pd.Timestamp],
    price_column: str = "close",
) -> pd.Series:
    """
    單一標的買進持有的**逐日權益曲線**（起點 1.0）。

    Args:
        bars: 該標的的日 K，index 為日期
        calendar: 模擬日曆（升冪）
        price_column: 取哪個價格欄位

    Returns:
        與 calendar 同長度的 Series

    Raises:
        ValueError: 日曆為空，或期初該標的尚未上市

    期初基準取**不晚於起始日**的最後一筆價格——取起始日之後的第一筆
    等於用未來的價格當成本，那是 look-ahead。

    期初尚未上市時拋錯，不可退而用之後的第一筆：回一條看起來正常的
    曲線，會讓報告拿一個起點錯誤的對照組去比。
    """
    if not calendar:
        raise ValueError("交易日曆不可為空")

    start = calendar[0]
    window = bars.loc[bars.index <= start]
    if window.empty:
        raise ValueError(
            f"期初 {start} 時該標的尚未上市（最早報價 "
            f"{bars.index.min() if len(bars) else 'n/a'}）"
        )

    base = float(window[price_column].iloc[-1])
    if base <= 0 or not math.isfinite(base):
        raise ValueError(f"期初價格不可用：{base}")

    aligned = bars[price_column].reindex(pd.DatetimeIndex(calendar), method="ffill")
    return (aligned / base).astype(float)


def etf_benchmark_curves(
    by_stock: dict[str, pd.DataFrame],
    calendar: list[pd.Timestamp],
    etf_ids: tuple[str, ...] = ETF_BENCHMARKS,
    price_column: str = "close",
) -> dict[str, pd.Series]:
    """
    一次算出所有可用的 ETF 對照曲線。

    Returns:
        {etf_id: 權益曲線}。資料不足的 ETF **直接略過**，不放進結果——
        回一條空的或全 1 的曲線會被誤讀成「那檔剛好沒漲跌」。
    """
    curves: dict[str, pd.Series] = {}
    for etf_id in etf_ids:
        bars = by_stock.get(etf_id)
        if bars is None or bars.empty:
            continue
        try:
            curves[etf_id] = single_asset_equity(bars, calendar, price_column)
        except ValueError:
            continue
    return curves
