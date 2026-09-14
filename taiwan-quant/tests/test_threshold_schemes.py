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
from scripts.validate_oos_trailing import make_price_lookup


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
    """每 4 個交易日重選，並在下個節點強制出場、重算區間報酬。"""
    calendar = list(pd.date_range("2023-01-02", periods=10, freq="B"))
    signals = [
        _signal(day, sid, score)
        for day in calendar
        for sid, score in (("A", 1.0), ("B", 4.0), ("C", 3.0), ("D", 2.0))
    ]

    prices = {sid: 100.0 for sid in ("A", "B", "C", "D")}

    def lookup(stock_id: str, day: pd.Timestamp) -> float | None:
        multiplier = 1.0 + calendar.index(day) / 100.0
        return prices[stock_id] * multiplier

    result = select_periodic_rebalances(
        signals, calendar, lookup, rebalance_every=4, top_n=3
    )
    selected = list(result.signals)

    assert sorted({s.decision_date for s in selected}) == [
        calendar[1], calendar[5]
    ]
    assert [s.stock_id for s in selected[:3]] == ["B", "C", "D"]
    assert len(selected) == 6
    assert all(s.exit_date == calendar[5] for s in selected[:3])
    assert all(s.exit_date == calendar[9] for s in selected[3:])
    assert selected[0].gross_return == pytest.approx(1.05 / 1.01 - 1.0)
    assert selected[0].entry_price == pytest.approx(101.0)
    assert result.candidates == 6
    assert result.rejected_missing_price == 0


@pytest.mark.unit
def test_periodic_rebalance_audits_top_candidate_with_missing_price() -> None:
    calendar = list(pd.date_range("2023-01-02", periods=4, freq="B"))
    signals = [_signal(calendar[0], "A", 2.0), _signal(calendar[0], "B", 1.0)]

    def lookup(stock_id: str, day: pd.Timestamp) -> float | None:
        return None if stock_id == "A" else 100.0

    result = select_periodic_rebalances(
        signals, calendar, lookup, rebalance_every=2, top_n=2
    )

    assert result.candidates == 2
    assert result.rejected_missing_price == 1
    assert [signal.stock_id for signal in result.signals] == ["B"]


@pytest.mark.unit
def test_execution_price_does_not_fill_forward_across_suspension() -> None:
    """成交價必須精確命中 T+1；只有逐日市值可沿用舊收盤價。"""
    first, suspended = pd.Timestamp("2023-01-02"), pd.Timestamp("2023-01-03")
    bars = pd.DataFrame({"open": [100.0], "close": [101.0]}, index=[first])

    execution = make_price_lookup({"A": bars}, column="open", exact=True)
    mark_to_market = make_price_lookup({"A": bars}, column="close")

    assert execution("A", suspended) is None
    assert mark_to_market("A", suspended) == pytest.approx(101.0)


@pytest.mark.unit
def test_periodic_rebalance_rejects_non_positive_inputs() -> None:
    with pytest.raises(ValueError, match="rebalance_every"):
        select_periodic_rebalances(
            [], [pd.Timestamp("2023-01-02")], lambda _s, _d: 1.0, 0, 3
        )
    with pytest.raises(ValueError, match="top_n"):
        select_periodic_rebalances(
            [], [pd.Timestamp("2023-01-02")], lambda _s, _d: 1.0, 60, 0
        )
