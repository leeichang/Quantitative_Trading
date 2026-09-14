"""門檻方案的純函式。

相對門檻只在同一決策日內比較，避免把市場時序水準誤當選股能力；定期
換倉則只建立固定交易日節點的候選，讓策略不再依賴槽位何時偶然釋放。
"""

from __future__ import annotations

import math

import pandas as pd

from taiwan_quant.backtest.portfolio_sim import Signal


def filter_relative_top(
    signals: list[Signal],
    top_percent: float,
) -> list[Signal]:
    """每個決策日只保留排名分數前 ``top_percent`` 的訊號。"""
    if not 0 < top_percent <= 100:
        raise ValueError(f"top_percent 必須落在 (0, 100]，得到 {top_percent}")

    by_date: dict[pd.Timestamp, list[Signal]] = {}
    for signal in signals:
        by_date.setdefault(signal.decision_date, []).append(signal)

    selected: list[Signal] = []
    for day in sorted(by_date):
        ranked = sorted(by_date[day], key=lambda item: (-item.rank_score, item.stock_id))
        count = max(1, math.ceil(len(ranked) * top_percent / 100.0))
        selected.extend(ranked[:count])
    return selected


def select_periodic_rebalances(
    signals: list[Signal],
    calendar: list[pd.Timestamp],
    rebalance_every: int,
    top_n: int,
) -> list[Signal]:
    """每隔固定交易日只保留該日 Top N，忽略節點之間的訊號。"""
    if rebalance_every < 1:
        raise ValueError(f"rebalance_every 必須為正，得到 {rebalance_every}")
    if top_n < 1:
        raise ValueError(f"top_n 必須為正，得到 {top_n}")
    if not calendar:
        return []

    scheduled = set(calendar[::rebalance_every])
    by_date: dict[pd.Timestamp, list[Signal]] = {}
    for signal in signals:
        if signal.decision_date in scheduled:
            by_date.setdefault(signal.decision_date, []).append(signal)

    selected: list[Signal] = []
    for day in sorted(by_date):
        ranked = sorted(by_date[day], key=lambda item: (-item.rank_score, item.stock_id))
        selected.extend(ranked[:top_n])
    return selected
