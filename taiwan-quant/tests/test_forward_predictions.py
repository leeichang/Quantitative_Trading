"""前推預測落盤與結算測試。"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from taiwan_quant.forward_predictions import (
    ForwardPrediction,
    initialize_forward_store,
    list_unsettled_predictions,
    record_forward_predictions,
    settle_forward_prediction,
)


def prediction() -> ForwardPrediction:
    return ForwardPrediction(
        predicted_at="2026-09-13T14:00:00+08:00",
        data_asof="2026-09-11",
        strategy_version="trailing_stop_portfolio@v0.1.0",
        family="籌碼跟隨",
        horizon=60,
        stock_id="2330",
        rank=1,
        score=0.8,
        expected_return=0.12,
        trail_pct=0.08,
        entry_price=1200.0,
        due_date="2026-12-04",
    )


@pytest.mark.unit
def test_initialize_creates_forward_predictions_schema(tmp_path: Path) -> None:
    db = tmp_path / "forward.db"
    initialize_forward_store(db)

    con = sqlite3.connect(db)
    columns = {row[1] for row in con.execute("PRAGMA table_info(forward_predictions)")}
    con.close()
    assert columns == {
        "predicted_at", "data_asof", "strategy_version", "family", "horizon",
        "stock_id", "rank", "score", "expected_return", "trail_pct",
        "entry_price", "due_date", "realized_return", "settled_at",
    }


@pytest.mark.unit
def test_record_is_idempotent_for_same_prediction_key(tmp_path: Path) -> None:
    db = tmp_path / "forward.db"
    item = prediction()
    record_forward_predictions(db, [item])
    record_forward_predictions(db, [item])

    assert list_unsettled_predictions(db) == [item]


@pytest.mark.unit
def test_settle_fills_realized_return_once(tmp_path: Path) -> None:
    db = tmp_path / "forward.db"
    item = prediction()
    record_forward_predictions(db, [item])
    settled_at = datetime(2026, 12, 4, tzinfo=timezone.utc)

    changed = settle_forward_prediction(
        db, item, realized_return=0.15, settled_at=settled_at
    )
    changed_again = settle_forward_prediction(
        db, item, realized_return=0.99, settled_at=settled_at
    )

    assert changed is True
    assert changed_again is False
    assert list_unsettled_predictions(db) == []

    con = sqlite3.connect(db)
    row = con.execute(
        "SELECT realized_return, settled_at FROM forward_predictions"
    ).fetchone()
    con.close()
    assert row == pytest.approx((0.15, settled_at.isoformat()))
