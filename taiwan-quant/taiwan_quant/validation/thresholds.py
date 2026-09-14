"""門檻方案的純函式。

相對門檻只在同一決策日內比較，避免把市場時序水準誤當選股能力；定期
換倉則只建立固定交易日節點的候選，讓策略不再依賴槽位何時偶然釋放。
"""

from __future__ import annotations

import math
from dataclasses import replace

import pandas as pd

from taiwan_quant.backtest.portfolio_sim import PriceLookup, Signal


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
    price_lookup: PriceLookup,
    rebalance_every: int,
    top_n: int,
) -> list[Signal]:
    """固定節點選 Top N，並在下個節點強制出場、依節點價格重算報酬。"""
    if rebalance_every < 1:
        raise ValueError(f"rebalance_every 必須為正，得到 {rebalance_every}")
    if top_n < 1:
        raise ValueError(f"top_n 必須為正，得到 {top_n}")
    if not calendar:
        return []

    nodes = list(calendar[::rebalance_every])
    if nodes[-1] != calendar[-1]:
        nodes.append(calendar[-1])
    scheduled = set(nodes[:-1])
    by_date: dict[pd.Timestamp, list[Signal]] = {}
    for signal in signals:
        if signal.decision_date in scheduled:
            by_date.setdefault(signal.decision_date, []).append(signal)

    selected: list[Signal] = []
    next_node = {day: nodes[index + 1] for index, day in enumerate(nodes[:-1])}
    for day in sorted(by_date):
        ranked = sorted(by_date[day], key=lambda item: (-item.rank_score, item.stock_id))
        exit_day = next_node[day]
        for signal in ranked[:top_n]:
            entry_price = price_lookup(signal.stock_id, day)
            exit_price = price_lookup(signal.stock_id, exit_day)
            if entry_price is None or exit_price is None or entry_price <= 0:
                continue
            selected.append(replace(
                signal,
                exit_date=exit_day,
                gross_return=exit_price / entry_price - 1.0,
            ))
    return selected
