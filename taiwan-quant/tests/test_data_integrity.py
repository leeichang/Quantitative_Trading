"""工作單 M：資料缺口分類與決策日期守門。"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from taiwan_quant.data.dataset import build_dataset
from taiwan_quant.data.integrity import (
    DataIntegrityError,
    assert_holding_price_completeness,
    complete_holding_decision_dates,
    find_gap_runs,
    holding_dates,
    select_holding_positions,
)


@pytest.mark.unit
def test_find_gap_runs_separates_single_days_and_blocks() -> None:
    calendar = pd.to_datetime(
        ["2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05", "2023-01-06"]
    )
    observed = pd.to_datetime(["2023-01-02", "2023-01-05", "2023-01-06"])

    runs = find_gap_runs(observed, calendar)

    assert [(run.start.date().isoformat(), run.length, run.shape) for run in runs] == [
        ("2023-01-03", 2, "block")
    ]


@pytest.mark.unit
def test_build_dataset_preserves_price_day_when_chips_are_missing() -> None:
    dates = pd.to_datetime(["2023-01-02", "2023-01-03", "2023-01-04"])
    index = pd.MultiIndex.from_product([["2330"], dates], names=["stock_id", "date"])
    prices = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "volume": [1000, 1100, 1200],
        },
        index=index,
    )
    chip_index = pd.MultiIndex.from_tuples(
        [("2330", dates[0]), ("2330", dates[2])], names=["stock_id", "date"]
    )
    chips = pd.DataFrame(
        {
            "foreign_net": [1.0, 2.0],
            "trust_net": [1.0, 2.0],
            "dealer_net": [1.0, 2.0],
            "margin_balance": [10.0, 11.0],
            "short_balance": [3.0, 4.0],
        },
        index=chip_index,
    )

    bars = build_dataset(["2330"], prices, chips, min_length=1).by_stock["2330"]

    assert bars.index.tolist() == dates.tolist()
    assert pd.isna(bars.loc[dates[1], "foreign_net"])


@pytest.mark.unit
def test_holding_price_guard_rejects_missing_exit_instead_of_dropna() -> None:
    calendar = list(pd.to_datetime(["2023-01-02", "2023-01-03", "2023-01-04"]))
    opens = pd.DataFrame({"2330": [100.0, 101.0, 102.0]}, index=calendar)
    closes = pd.DataFrame({"2330": [100.5, 101.5, float("nan")]}, index=calendar)

    with pytest.raises(DataIntegrityError, match=r"2330.*T\+H.*2023-01-04"):
        assert_holding_price_completeness(
            decision_date=calendar[0],
            calendar=calendar,
            candidates=["2330"],
            opens=opens,
            closes=closes,
            holding_days=2,
        )


@pytest.mark.unit
def test_selection_guard_exits_at_decision_plus_h_not_entry_plus_h() -> None:
    calendar = list(pd.date_range("2023-01-02", periods=5, freq="B"))
    opens = pd.DataFrame({"A": [99.0, 100.0, 101.0, 102.0, 103.0]}, index=calendar)
    closes = pd.DataFrame({"A": [99.0, 100.0, 101.0, 110.0, float("nan")]}, index=calendar)

    selected = select_holding_positions(
        decision_date=calendar[0],
        calendar=calendar,
        ordered_candidates=["A"],
        opens=opens,
        closes=closes,
        holding_days=3,
        n_positions=1,
        delisted_dates={},
    )

    # T+1 以 100 進場、T+3 以 110 出場；不可錯查 T+4 的 NaN。
    assert selected[0].gross_return == pytest.approx(0.10)


@pytest.mark.unit
def test_holding_dates_and_tail_filter_share_t_plus_h_boundary() -> None:
    calendar = list(pd.date_range("2023-01-02", periods=6, freq="B"))

    entry, target = holding_dates(calendar[1], calendar, holding_days=3)
    kept = complete_holding_decision_dates(
        calendar,
        [calendar[0], calendar[1], calendar[2], calendar[3]],
        holding_days=3,
    )

    assert entry == calendar[2]
    assert target == calendar[4]
    assert kept == [calendar[0], calendar[1], calendar[2]]


@pytest.mark.unit
def test_selection_guard_allows_delisted_position_with_policy_b_return() -> None:
    calendar = list(pd.date_range("2023-01-02", periods=5, freq="B"))
    opens = pd.DataFrame({"A": [100.0, 100.0, 90.0, float("nan"), float("nan")]}, index=calendar)
    closes = pd.DataFrame({"A": [100.0, 100.0, 80.0, float("nan"), float("nan")]}, index=calendar)

    selected = select_holding_positions(
        decision_date=calendar[0],
        calendar=calendar,
        ordered_candidates=["A"],
        opens=opens,
        closes=closes,
        holding_days=3,
        n_positions=1,
        delisted_dates={"A": date(2023, 1, 4)},
    )

    assert [position.stock_id for position in selected] == ["A"]
    assert selected[0].gross_return == pytest.approx(-0.20)
    assert selected[0].kind.value == "delisted"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("stock_id", "decision", "entry", "last_trade", "target", "delisted"),
    [
        ("3474", "2016-11-09", "2016-11-10", "2016-12-05", "2017-01-06", date(2016, 12, 6)),
        ("2325", "2018-03-30", "2018-04-02", "2018-04-27", "2018-05-31", date(2018, 4, 30)),
        ("2311", "2018-03-30", "2018-04-02", "2018-04-27", "2018-05-31", date(2018, 4, 30)),
    ],
)
def test_known_delistings_do_not_trip_integrity_guard(
    stock_id: str,
    decision: str,
    entry: str,
    last_trade: str,
    target: str,
    delisted: date,
) -> None:
    calendar = list(pd.to_datetime([decision, entry, last_trade, target]))
    opens = pd.DataFrame({stock_id: [99.0, 100.0, 82.0, float("nan")]}, index=calendar)
    closes = pd.DataFrame({stock_id: [99.0, 100.0, 80.0, float("nan")]}, index=calendar)

    selected = select_holding_positions(
        decision_date=calendar[0],
        calendar=calendar,
        ordered_candidates=[stock_id],
        opens=opens,
        closes=closes,
        holding_days=3,
        n_positions=1,
        delisted_dates={stock_id: delisted},
    )

    assert selected[0].stock_id == stock_id
    assert selected[0].kind.value == "delisted"
    assert selected[0].gross_return == pytest.approx(-0.20)


@pytest.mark.unit
def test_selection_guard_rejects_suspended_position_in_top_n() -> None:
    calendar = list(pd.date_range("2023-01-02", periods=6, freq="B"))
    opens = pd.DataFrame({"A": [100.0, 100.0, 100.0, 100.0, 101.0, 101.0]}, index=calendar)
    closes = pd.DataFrame({"A": [100.0, 100.0, 100.0, float("nan"), 101.0, 101.0]}, index=calendar)

    with pytest.raises(DataIntegrityError, match="A.*suspended_or_missing"):
        select_holding_positions(
            decision_date=calendar[0],
            calendar=calendar,
            ordered_candidates=["A"],
            opens=opens,
            closes=closes,
            holding_days=3,
            n_positions=1,
            delisted_dates={"A": None},
        )


@pytest.mark.unit
def test_selection_guard_checks_replacement_after_missing_entry() -> None:
    calendar = list(pd.date_range("2023-01-02", periods=6, freq="B"))
    opens = pd.DataFrame(
        {
            "NO_ENTRY": [100.0, float("nan"), 100.0, 100.0, 100.0, 100.0],
            "REPLACEMENT": [100.0, 100.0, 100.0, float("nan"), 101.0, 101.0],
        },
        index=calendar,
    )
    closes = opens.copy()

    with pytest.raises(DataIntegrityError, match="REPLACEMENT.*suspended_or_missing"):
        select_holding_positions(
            decision_date=calendar[0],
            calendar=calendar,
            ordered_candidates=["NO_ENTRY", "REPLACEMENT"],
            opens=opens,
            closes=closes,
            holding_days=3,
            n_positions=1,
            delisted_dates={},
        )


@pytest.mark.unit
def test_selection_guard_does_not_check_names_below_filled_top_n() -> None:
    calendar = list(pd.date_range("2023-01-02", periods=6, freq="B"))
    opens = pd.DataFrame(
        {"BUY": [100.0] * 6, "BELOW": [100.0, 100.0, 100.0, float("nan"), 101.0, 101.0]},
        index=calendar,
    )
    closes = opens.copy()

    selected = select_holding_positions(
        decision_date=calendar[0],
        calendar=calendar,
        ordered_candidates=["BUY", "BELOW"],
        opens=opens,
        closes=closes,
        holding_days=3,
        n_positions=1,
        delisted_dates={},
    )

    assert [position.stock_id for position in selected] == ["BUY"]
