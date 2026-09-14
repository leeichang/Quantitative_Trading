"""
不可變的前推預測帳本

預測一旦寫入只能在到期後補上實現報酬，不允許覆寫原始分數或參數。

## 為什麼 `edge_z` 必須進表（CLAUDE.md 禁令 8）

`edge_z` 是 CLI 可調、而且會改變選股結果的參數。它不進唯一鍵時，
同一天掃兩組門檻的第二筆會被 `INSERT OR IGNORE` **靜默丟掉**，
帳本會長成「同一版本同一天，參數不明」。

禁令 8 要求「必須保存 backtest parameters」——帳本是唯一會隨時間
收斂的證據，參數不齊等於整份證據作廢。

## 唯一鍵與主鍵是兩回事

```
PRIMARY KEY  (predicted_at, edge_z, family, horizon, stock_id)
             實體紀錄，含產生時刻

UNIQUE       (data_asof, strategy_version, edge_z, family, horizon, stock_id)
             官方預測，每組參數一筆
```

`predicted_at` 每次執行都不同，所以主鍵擋不住重跑。真正的冪等保證
來自 UNIQUE：同一份資料、同一版程式、同一組參數，只會有一筆。

⚠️ `edge_z` 兩邊都要有。只加進 UNIQUE 的話，**同一次執行**掃兩組門檻
時 `predicted_at` 相同，第二筆會撞上主鍵被靜默丟掉——靠時間戳差異
避開碰撞是假的保護。
"""

from __future__ import annotations

import sqlite3
from dataclasses import astuple, dataclass, fields
from datetime import datetime
from pathlib import Path


class ForwardStoreError(RuntimeError):
    """帳本結構與現行欄位不相容"""


@dataclass(frozen=True)
class ForwardPrediction:
    predicted_at: str
    data_asof: str
    strategy_version: str
    edge_z: float
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
    edge_z REAL NOT NULL,
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
    PRIMARY KEY (predicted_at, edge_z, family, horizon, stock_id),
    UNIQUE (data_asof, strategy_version, edge_z, family, horizon, stock_id)
);
"""
"""
table-level `UNIQUE(...)` 本身就會建索引（`sqlite_autoindex_..._2`），
不要再手動 `CREATE UNIQUE INDEX` 在同一組欄位上——那是雙倍儲存與
雙倍寫入成本，換不到任何東西。
"""

_COLUMNS = tuple(field.name for field in fields(ForwardPrediction))
_PLACEHOLDERS = ", ".join("?" * len(_COLUMNS))
_COLUMN_LIST = ", ".join(_COLUMNS)


def initialize_forward_store(db_path: Path) -> None:
    """
    建立前推預測表；不更動資料庫中的其他表。

    Raises:
        ForwardStoreError: 既有表缺少現行欄位

    `CREATE TABLE IF NOT EXISTS` 對既有表是**無聲的**，所以建完要驗一次
    欄位。舊帳本缺 `edge_z` 時寧可拋錯，也不要靜默寫進不完整的結構。
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as con:
        con.executescript(SCHEMA)
        existing = {
            row[1] for row in con.execute("PRAGMA table_info(forward_predictions)")
        }
    missing = [name for name in _COLUMNS if name not in existing]
    if missing:
        raise ForwardStoreError(
            f"{db_path} 的 forward_predictions 缺少欄位 {missing}；"
            "這是舊版結構，請先備份後重建該表"
        )


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
            f"INSERT OR IGNORE INTO forward_predictions ({_COLUMN_LIST}) "
            f"VALUES ({_PLACEHOLDERS})",
            [astuple(item) for item in predictions],
        )
        after = con.total_changes
    return after - before


def list_unsettled_predictions(db_path: Path) -> list[ForwardPrediction]:
    """依到期日列出尚未結算的預測。"""
    initialize_forward_store(db_path)
    with sqlite3.connect(db_path) as con:
        rows = con.execute(
            f"SELECT {_COLUMN_LIST} FROM forward_predictions "
            "WHERE settled_at IS NULL "
            "ORDER BY due_date, predicted_at, family, rank"
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
