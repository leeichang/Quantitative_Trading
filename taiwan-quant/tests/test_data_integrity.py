"""工作單 M：資料缺口分類與決策日期守門。"""

from __future__ import annotations

import pandas as pd
import pytest

from taiwan_quant.data.dataset import build_dataset
from taiwan_quant.data.integrity import (
    DataIntegrityError,
    assert_holding_price_completeness,
    find_gap_runs,
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
            holding_days=1,
        )
