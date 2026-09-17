"""價格缺口回補的不可變條件。"""

from __future__ import annotations

import sqlite3

import pytest

from scripts.backfill_candidate_price_gaps import _repair_prices


@pytest.fixture
def price_db() -> sqlite3.Connection:
    con = sqlite3.connect(":memory:")
    con.execute(
        """
        CREATE TABLE stock_daily (
            stock_id TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            created_at TEXT,
            PRIMARY KEY (stock_id, date)
        )
        """
    )
    yield con
    con.close()


def _row(day: str, *, volume: object = 1000) -> dict[str, object]:
    return {
        "date": day,
        "open": 10,
        "max": 12,
        "min": 9,
        "close": 11,
        "Trading_Volume": volume,
    }


@pytest.mark.unit
def test_repair_only_expected_gap_with_complete_source_volume(
    price_db: sqlite3.Connection,
) -> None:
    price_db.execute(
        "INSERT INTO stock_daily VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("2330", "2020-01-02", 20, 21, 19, 20.5, 500, "original"),
    )
    rows = [
        _row("2020-01-02"),  # 完整既有列，不可覆寫
        _row("2020-01-03"),  # 唯一允許補入的日期
        _row("2020-01-06"),  # 不在診斷缺口，不可補入
        _row("2020-01-07", volume=""),  # 成交量未知，不可猜成 0
    ]

    repaired = _repair_prices(
        price_db,
        "2330",
        rows,
        {"2020-01-02", "2020-01-03", "2020-01-07"},
    )

    assert repaired == 1
    stored = price_db.execute(
        "SELECT date, open, volume FROM stock_daily ORDER BY date"
    ).fetchall()
    assert stored == [("2020-01-02", 20.0, 500), ("2020-01-03", 10.0, 1000)]
