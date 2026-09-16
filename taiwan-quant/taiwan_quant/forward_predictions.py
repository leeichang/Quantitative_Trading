"""
不可變的前推預測帳本

預測一旦寫入只能在到期後補上實現報酬，不允許覆寫原始分數或參數。

## 為什麼這是唯一會收斂的證據

開發集可以反覆看，所以它證明不了什麼。OOS 區間看一次就髒了
（2024-01 ~ 2026-08 已經用掉）。**只有前推預測是往前長的**：時間走一天
就多一天證據，而且從來沒有被看過。

```
每 40 個交易日一筆決策 × 10 檔
1 年    6 期    仍然太少
3 年   19 期    與 2024-2026 那次 OOS 同量級
5 年   32 期    開始有意義
```

慢，但它不消耗任何區間。

## 身分收斂到 `strategy_version`（禁令 7）

先前的 schema 把可調參數散成欄位（`edge_z`、`trail_pct`、
`expected_return`），結果出過一個 bug：`edge_z` 沒進唯一鍵，同一天掃兩組
門檻的第二筆被 `INSERT OR IGNORE` 靜默丟掉。

當時的修法是把 `edge_z` 加進主鍵與唯一鍵——**但那只對 `edge_z` 有效**。
下一個可調參數出現時同樣的 bug 會再來一次。

現在：**參數變了就換 `strategy_version`**。

```
PRIMARY KEY (predicted_at, strategy_version, stock_id)
UNIQUE      (data_asof, strategy_version, stock_id)
```

`params_json` 存完整參數供稽核，但不進鍵。禁令 7 要的就是「能反查是哪版
程式、哪組參數」——一個欄位承擔那個身分，比十個欄位各自承擔一部分可靠。

## `round_trip_cost` 逐筆存（禁令 8）

實測 N=10 時 86.2% 的持倉走零股（1.071%）、13.8% 走整股（0.671%）——
只存一個「代表值」會讓事後無法還原當時的淨報酬。
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import astuple, dataclass, fields
from datetime import datetime
from pathlib import Path


class ForwardStoreError(RuntimeError):
    """帳本結構或內容不合法"""


@dataclass(frozen=True)
class ForwardPrediction:
    """
    一筆前推預測。

    只有**產出時可知**的欄位。`realized_return` 與 `settled_at` 不在這裡
    ——它們是到期後才補的，放進來會讓「未結算」這個狀態變得可以偽造。
    """

    predicted_at: str
    """預測產生的實際時間（含時區）"""

    data_asof: str
    """用到的資料截止日"""

    strategy_version: str
    """程式版本 + 參數組，例如 `momentum_top10_h40@v1`。**這就是身分**"""

    params_json: str
    """完整參數的 JSON，供稽核。不進唯一鍵"""

    stock_id: str
    rank: int
    score: float
    entry_price: float
    due_date: str
    """標籤揭曉日。用實際交易日曆推算，不可用 `pd.offsets.BDay`"""

    round_trip_cost: float
    """這一筆的來回成本，由 `config.costs.resolve_tier` 逐檔決定"""


SCHEMA = """
CREATE TABLE IF NOT EXISTS forward_predictions (
    predicted_at TEXT NOT NULL,
    data_asof TEXT NOT NULL,
    strategy_version TEXT NOT NULL,
    params_json TEXT NOT NULL,
    stock_id TEXT NOT NULL,
    rank INTEGER NOT NULL,
    score REAL NOT NULL,
    entry_price REAL NOT NULL,
    due_date TEXT NOT NULL,
    round_trip_cost REAL NOT NULL,
    realized_return REAL,
    settled_at TEXT,
    PRIMARY KEY (predicted_at, strategy_version, stock_id),
    UNIQUE (data_asof, strategy_version, stock_id)
);
"""
"""
table-level `UNIQUE(...)` 本身就會建索引，不要再手動 `CREATE UNIQUE INDEX`
在同一組欄位上——那是雙倍儲存與雙倍寫入成本，換不到任何東西。
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
    欄位。舊結構（含 `edge_z`）缺 `params_json`，寧可拋錯也不要靜默沿用。
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
            "這是舊版結構（可能含 edge_z / trail_pct），請先備份後重建該表"
        )


def _validate(prediction: ForwardPrediction) -> None:
    """寫入前檢查。不合法的內容寧可拋錯，不要事後才發現讀不出來"""
    try:
        params = json.loads(prediction.params_json)
    except json.JSONDecodeError as exc:
        raise ForwardStoreError(
            f"{prediction.stock_id} 的 params_json 不是合法 JSON：{exc}"
        ) from exc
    if not isinstance(params, dict) or not params:
        raise ForwardStoreError(
            f"{prediction.stock_id} 的 params_json 不可為空——"
            "禁令 8 要求保存參數"
        )
    if not prediction.strategy_version.strip():
        raise ForwardStoreError("strategy_version 不可為空白")
    if prediction.entry_price <= 0:
        raise ForwardStoreError(
            f"entry_price 必須為正，得到 {prediction.entry_price}"
        )
    if prediction.round_trip_cost < 0:
        raise ForwardStoreError(
            f"round_trip_cost 不可為負，得到 {prediction.round_trip_cost}"
        )


def record_forward_predictions(
    db_path: Path, predictions: list[ForwardPrediction]
) -> int:
    """
    只新增不存在的預測；同一唯一鍵重跑不覆寫歷史紀錄。

    Args:
        db_path: SQLite 檔案路徑
        predictions: 待寫入的預測

    Returns:
        實際新增的列數

    Raises:
        ForwardStoreError: 任一筆內容不合法，或既有表是舊結構

    **全部驗證通過才寫入。** 一筆不合法就整批拋錯，不要寫一半。
    """
    for prediction in predictions:
        _validate(prediction)
    initialize_forward_store(db_path)
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
            "ORDER BY due_date, predicted_at, strategy_version, rank"
        ).fetchall()
    return [ForwardPrediction(*row) for row in rows]


def settle_forward_prediction(
    db_path: Path,
    prediction: ForwardPrediction,
    realized_return: float,
    settled_at: datetime,
) -> bool:
    """
    結算一筆未結算預測。

    Returns:
        `True` 代表這次真的寫入；`False` 代表它早已結算

    已結算的資料**不可被第二次覆寫**——那是帳本存在的理由。
    """
    initialize_forward_store(db_path)
    with sqlite3.connect(db_path) as con:
        cursor = con.execute(
            """
            UPDATE forward_predictions
            SET realized_return = ?, settled_at = ?
            WHERE predicted_at = ? AND strategy_version = ? AND stock_id = ?
              AND settled_at IS NULL
            """,
            (
                realized_return,
                settled_at.isoformat(),
                prediction.predicted_at,
                prediction.strategy_version,
                prediction.stock_id,
            ),
        )
    return cursor.rowcount == 1
