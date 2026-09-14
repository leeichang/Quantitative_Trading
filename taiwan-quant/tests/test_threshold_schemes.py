"""任務 E 的門檻與定期換倉性質測試。"""

from __future__ import annotations

import pandas as pd
import pytest

from taiwan_quant.backtest.portfolio_sim import Signal
from taiwan_quant.config.costs import Tier
from taiwan_quant.validation.thresholds import (
    filter_relative_top,
    select_periodic_rebalances,
)


def _signal(day: pd.Timestamp, stock_id: str, score: float) -> Signal:
    return Signal(
        decision_date=day,
        exit_date=day + pd.Timedelta(days=120),
        stock_id=stock_id,
        gross_return=0.10,
        rank_score=score,
        tier=Tier.LARGE,
    )


@pytest.mark.unit
def test_relative_top_filters_each_date_independently() -> None:
    """每日期望報酬前 20%；5 檔手算後各留 1 檔。"""
    days = list(pd.date_range("2023-01-02", periods=2, freq="B"))
    signals = [
        _signal(day, f"{day.day}-{rank}", float(rank))
        for day in days
        for rank in range(1, 6)
    ]

    selected = filter_relative_top(signals, top_percent=20)

    assert [(s.decision_date, s.rank_score) for s in selected] == [
        (days[0], 5.0),
        (days[1], 5.0),
    ]


@pytest.mark.unit
def test_relative_top_rounds_up_and_is_deterministic_on_ties() -> None:
    """3 檔取前 10% 至少留 1 檔；同分時代號小者勝出。"""
    day = pd.Timestamp("2023-01-02")
    signals = [_signal(day, sid, 1.0) for sid in ("C", "A", "B")]

    selected = filter_relative_top(signals, top_percent=10)

    assert [s.stock_id for s in selected] == ["A"]


@pytest.mark.unit
@pytest.mark.parametrize("value", [0, -1, 101])
def test_relative_top_rejects_invalid_percent(value: float) -> None:
    with pytest.raises(ValueError, match="top_percent"):
        filter_relative_top([], top_percent=value)


@pytest.mark.unit
def test_periodic_rebalance_uses_exact_calendar_intervals_and_top_three() -> None:
    """每 4 個交易日重選，只用節點當日分數最高的 3 檔。"""
    calendar = list(pd.date_range("2023-01-02", periods=9, freq="B"))
    signals = [
        _signal(day, sid, score)
        for day in calendar
        for sid, score in (("A", 1.0), ("B", 4.0), ("C", 3.0), ("D", 2.0))
    ]

    selected = select_periodic_rebalances(
        signals, calendar, rebalance_every=4, top_n=3
    )

    assert sorted({s.decision_date for s in selected}) == [
        calendar[0], calendar[4], calendar[8]
    ]
    assert [s.stock_id for s in selected[:3]] == ["B", "C", "D"]
    assert len(selected) == 9


@pytest.mark.unit
def test_periodic_rebalance_rejects_non_positive_inputs() -> None:
    with pytest.raises(ValueError, match="rebalance_every"):
        select_periodic_rebalances([], [pd.Timestamp("2023-01-02")], 0, 3)
    with pytest.raises(ValueError, match="top_n"):
        select_periodic_rebalances([], [pd.Timestamp("2023-01-02")], 60, 0)
