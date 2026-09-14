"""前推預測落盤與結算測試。"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from taiwan_quant.forward_predictions import (
    ForwardPrediction,
    ForwardStoreError,
    initialize_forward_store,
    list_unsettled_predictions,
    record_forward_predictions,
    settle_forward_prediction,
)


def prediction(**overrides: object) -> ForwardPrediction:
    base = dict(
        predicted_at="2026-09-13T14:00:00+08:00",
        data_asof="2026-09-11",
        strategy_version="trailing_stop_portfolio@v0.1.0",
        edge_z=0.5,
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
    return ForwardPrediction(**{**base, **overrides})  # type: ignore[arg-type]


@pytest.mark.unit
def test_initialize_creates_forward_predictions_schema(tmp_path: Path) -> None:
    db = tmp_path / "forward.db"
    initialize_forward_store(db)

    con = sqlite3.connect(db)
    columns = {row[1] for row in con.execute("PRAGMA table_info(forward_predictions)")}
    con.close()
    assert columns == {
        "predicted_at", "data_asof", "strategy_version", "edge_z", "family",
        "horizon", "stock_id", "rank", "score", "expected_return", "trail_pct",
        "entry_price", "due_date", "realized_return", "settled_at",
    }


@pytest.mark.unit
def test_schema_does_not_create_a_redundant_index(tmp_path: Path) -> None:
    """
    table-level `UNIQUE(...)` 本來就會建索引。再手動 `CREATE UNIQUE INDEX`
    在同一組欄位上，只是雙倍儲存與雙倍寫入成本。

    預期只剩兩個自動索引：PRIMARY KEY 一個、UNIQUE 一個。
    """
    db = tmp_path / "forward.db"
    initialize_forward_store(db)

    con = sqlite3.connect(db)
    names = [
        row[0]
        for row in con.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'index' AND tbl_name = 'forward_predictions'"
        )
    ]
    con.close()

    assert all(name.startswith("sqlite_autoindex_") for name in names), names
    assert len(names) == 2


@pytest.mark.unit
def test_different_edge_z_on_same_day_are_both_recorded(tmp_path: Path) -> None:
    """
    禁令 8：必須保存 backtest parameters。

    `edge_z` 不進**唯一鍵與主鍵**時，同一天掃兩組門檻的第二筆會被
    `INSERT OR IGNORE` 靜默丟掉，帳本會長成「同一版本同一天，參數不明」。

    這裡刻意讓兩筆共用同一個 `predicted_at`——只靠時間戳差異來避開碰撞
    是假的保護，同一次執行掃兩組參數時時間戳就是一樣的。
    """
    db = tmp_path / "forward.db"
    loose = prediction(edge_z=0.3)
    strict = prediction(edge_z=0.8)

    assert loose.predicted_at == strict.predicted_at
    assert record_forward_predictions(db, [loose]) == 1
    assert record_forward_predictions(db, [strict]) == 1
    assert {item.edge_z for item in list_unsettled_predictions(db)} == {0.3, 0.8}


@pytest.mark.unit
def test_rejects_incompatible_legacy_schema(tmp_path: Path) -> None:
    """
    舊表缺 `edge_z` 時要明確拋錯，不可靜默沿用——
    `CREATE TABLE IF NOT EXISTS` 對既有表是無聲的。
    """
    db = tmp_path / "forward.db"
    con = sqlite3.connect(db)
    con.execute(
        "CREATE TABLE forward_predictions (predicted_at TEXT, stock_id TEXT)"
    )
    con.commit()
    con.close()

    with pytest.raises(ForwardStoreError, match="edge_z"):
        initialize_forward_store(db)


@pytest.mark.unit
def test_record_is_idempotent_for_same_prediction_key(tmp_path: Path) -> None:
    db = tmp_path / "forward.db"
    item = prediction()
    record_forward_predictions(db, [item])
    record_forward_predictions(db, [item])

    assert list_unsettled_predictions(db) == [item]


@pytest.mark.unit
def test_rerun_timestamp_does_not_duplicate_official_prediction(tmp_path: Path) -> None:
    """同一資料、版本、策略與標的只能有一筆官方前推預測。"""
    db = tmp_path / "forward.db"
    first = prediction()
    rerun = ForwardPrediction(
        **{**first.__dict__, "predicted_at": "2026-09-13T14:05:00+08:00"}
    )

    assert record_forward_predictions(db, [first]) == 1
    assert record_forward_predictions(db, [rerun]) == 0
    assert list_unsettled_predictions(db) == [first]


@pytest.mark.unit
def test_settle_fills_realized_return_once(tmp_path: Path) -> None:
    db = tmp_path / "forward.db"
    item = prediction()
    record_forward_predictions(db, [item])
    settled_at = datetime(2026, 12, 4, tzinfo=UTC)

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
    assert row[0] == pytest.approx(0.15)
    assert row[1] == settled_at.isoformat()
