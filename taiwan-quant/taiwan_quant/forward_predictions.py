"""不可變的前推預測帳本。

預測一旦寫入只能在到期後補上實現報酬，不允許覆寫原始分數或參數。
"""

from __future__ import annotations

import sqlite3
from dataclasses import astuple, dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class ForwardPrediction:
    predicted_at: str
    data_asof: str
    strategy_version: str
    family: str
    horizon: int
    stock_id: str
    rank: int
    score: float
    expected_return: float | None
    trail_pct: float
    entry_price: float
    due_date: str


SCHEMA = """
CREATE TABLE IF NOT EXISTS forward_predictions (
    predicted_at TEXT NOT NULL,
    data_asof TEXT NOT NULL,
    strategy_version TEXT NOT NULL,
    family TEXT NOT NULL,
    horizon INTEGER NOT NULL,
    stock_id TEXT NOT NULL,
    rank INTEGER NOT NULL,
    score REAL NOT NULL,
    expected_return REAL,
    trail_pct REAL NOT NULL,
    entry_price REAL NOT NULL,
    due_date TEXT NOT NULL,
    realized_return REAL,
    settled_at TEXT,
    PRIMARY KEY (predicted_at, family, horizon, stock_id),
    UNIQUE (data_asof, strategy_version, family, horizon, stock_id)
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_forward_prediction_identity
ON forward_predictions (data_asof, strategy_version, family, horizon, stock_id);
"""


def initialize_forward_store(db_path: Path) -> None:
    """建立前推預測表；不更動資料庫中的其他表。"""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as con:
        con.executescript(SCHEMA)


def record_forward_predictions(
    db_path: Path, predictions: list[ForwardPrediction]
) -> int:
    """只新增不存在的預測；同一主鍵重跑不覆寫歷史紀錄。"""
    initialize_forward_store(db_path)
    before: int
    after: int
    with sqlite3.connect(db_path) as con:
        before = con.total_changes
        con.executemany(
            """
            INSERT OR IGNORE INTO forward_predictions (
                predicted_at, data_asof, strategy_version, family, horizon,
                stock_id, rank, score, expected_return, trail_pct, entry_price,
                due_date
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [astuple(item) for item in predictions],
        )
        after = con.total_changes
    return after - before


def list_unsettled_predictions(db_path: Path) -> list[ForwardPrediction]:
    """依到期日列出尚未結算的預測。"""
    initialize_forward_store(db_path)
    with sqlite3.connect(db_path) as con:
        rows = con.execute(
            """
            SELECT predicted_at, data_asof, strategy_version, family, horizon,
                   stock_id, rank, score, expected_return, trail_pct,
                   entry_price, due_date
            FROM forward_predictions
            WHERE settled_at IS NULL
            ORDER BY due_date, predicted_at, family, rank
            """
        ).fetchall()
    return [ForwardPrediction(*row) for row in rows]


def settle_forward_prediction(
    db_path: Path,
    prediction: ForwardPrediction,
    realized_return: float,
    settled_at: datetime,
) -> bool:
    """結算一筆未結算預測；已結算資料不可被第二次覆寫。"""
    initialize_forward_store(db_path)
    with sqlite3.connect(db_path) as con:
        cursor = con.execute(
            """
            UPDATE forward_predictions
            SET realized_return = ?, settled_at = ?
            WHERE predicted_at = ? AND family = ? AND horizon = ?
              AND stock_id = ? AND settled_at IS NULL
            """,
            (
                realized_return,
                settled_at.isoformat(),
                prediction.predicted_at,
                prediction.family,
                prediction.horizon,
                prediction.stock_id,
            ),
        )
    return cursor.rowcount == 1
