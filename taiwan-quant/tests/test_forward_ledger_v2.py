"""
前推帳本改記動能突破 N=10 的測試

## 為什麼要改

帳本原本是為「校準器 + 移動停損」那條路設計的：

```
edge_z          校準器門檻的 z 值      → 動能突破沒有校準器
trail_pct       移動停損寬度           → 動能突破是固定 40 日出場
expected_return 校準器的期望報酬       → 同上
```

而那條路已經被實測否定：`2026-09-16_漲停預測力與成本結構.md` 顯示
校準器（12 等級）→ 門檻 → 槽位佇列三層在毀訊號，60 日 CPCV 中位數
只有 +5.06%、5% 分位 −27.86%；直接用原始分數排序則是 +94.39% / +13.68%。

帳本一列都還沒寫（實測確認 `forward_predictions` 表不存在），所以現在
是唯一能無痛改 schema 的時機。

## 設計：`strategy_version` 就是身分

禁令 7：「必須保存 strategy version——每次產出都要能反查是哪版程式、
哪組參數」。

所以把可調參數編進 `strategy_version`，而不是散成一堆欄位：

```
momentum_top10_h40@v1     程式版本 + 參數組
```

`params_json` 存完整參數供稽核，但**唯一鍵只用
(data_asof, strategy_version, stock_id)**——參數變了 `strategy_version`
就變，不需要把每個參數都放進鍵裡。

這解決了先前 `edge_z` 那個問題的根源：當時的修法是「把 edge_z 加進主鍵
與唯一鍵」，但那只對 `edge_z` 有效。下一個可調參數出現時同樣的 bug 會
再來一次。**把身分收斂到一個欄位才是根治。**
"""

from __future__ import annotations

import json
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


def momentum_prediction(**overrides: object) -> ForwardPrediction:
    """動能突破 N=10 的一筆預測——沒有 edge_z、沒有 trail_pct"""
    base = dict(
        predicted_at="2026-09-16T20:00:00+08:00",
        data_asof="2026-09-11",
        strategy_version="momentum_top10_h40@v1",
        params_json=json.dumps(
            {"family": "動能突破", "holding_days": 40, "n_positions": 10,
             "universe_size": 150, "universe_basis": "market_cap"},
            ensure_ascii=False, sort_keys=True,
        ),
        stock_id="2330",
        rank=1,
        score=0.8734,
        entry_price=1200.0,
        due_date="2026-11-10",
        round_trip_cost=0.01071,
    )
    return ForwardPrediction(**{**base, **overrides})  # type: ignore[arg-type]


# ══════════════════════════════════════════════════════════════
# 新 schema
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_schema_replaces_calibrator_fields_with_params_json(tmp_path: Path) -> None:
    """
    `edge_z` / `trail_pct` / `expected_return` / `family` / `horizon`
    都從欄位變成 `params_json` 的內容。

    理由：它們是「校準器 + 移動停損」專屬的。動能突破沒有校準器，
    硬留著就得填 NULL 或假值——假值比缺值更糟，它看起來像真的。
    """
    db = tmp_path / "forward.db"
    initialize_forward_store(db)

    con = sqlite3.connect(db)
    columns = {row[1] for row in con.execute("PRAGMA table_info(forward_predictions)")}
    con.close()

    assert columns == {
        "predicted_at", "data_asof", "strategy_version", "params_json",
        "stock_id", "rank", "score", "entry_price", "due_date",
        "round_trip_cost", "realized_return", "settled_at",
    }
    for gone in ("edge_z", "trail_pct", "expected_return", "family", "horizon"):
        assert gone not in columns, f"{gone} 應該已移進 params_json"


@pytest.mark.unit
def test_cost_is_recorded_per_prediction(tmp_path: Path) -> None:
    """
    `round_trip_cost` 要逐筆存。

    禁令 8：必須保存 backtest parameters，含成本模型設定。實測發現
    N=10 時 86.2% 的持倉走零股（1.071%）、13.8% 走整股（0.671%）——
    只存一個「代表值」會讓事後無法還原當時的淨報酬。
    """
    db = tmp_path / "forward.db"
    record_forward_predictions(db, [
        momentum_prediction(stock_id="2330", round_trip_cost=0.01071),
        momentum_prediction(stock_id="0050", rank=2, round_trip_cost=0.00100),
    ])

    costs = {p.stock_id: p.round_trip_cost for p in list_unsettled_predictions(db)}

    assert costs == {"2330": 0.01071, "0050": 0.00100}


# ══════════════════════════════════════════════════════════════
# 身分收斂到 strategy_version
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_same_version_same_day_same_name_is_recorded_once(tmp_path: Path) -> None:
    """重跑不可產生第二筆官方預測。"""
    db = tmp_path / "forward.db"
    first = momentum_prediction()
    rerun = momentum_prediction(predicted_at="2026-09-16T20:05:00+08:00")

    assert record_forward_predictions(db, [first]) == 1
    assert record_forward_predictions(db, [rerun]) == 0
    assert list_unsettled_predictions(db) == [first]


@pytest.mark.unit
def test_different_versions_coexist(tmp_path: Path) -> None:
    """
    參數變了就換 `strategy_version`，兩者並存——這是帳本的重點：
    能比較不同版本在**同一段未來**的表現。

    刻意共用同一個 `predicted_at`：同一次執行記兩個版本時時間戳相同，
    靠時間戳差異避開碰撞是假的保護（先前 `edge_z` 就是這樣被丟掉的）。
    """
    db = tmp_path / "forward.db"
    v1 = momentum_prediction(strategy_version="momentum_top10_h40@v1")
    v2 = momentum_prediction(strategy_version="momentum_top10_h60@v1")

    assert v1.predicted_at == v2.predicted_at
    assert record_forward_predictions(db, [v1]) == 1
    assert record_forward_predictions(db, [v2]) == 1
    assert {p.strategy_version for p in list_unsettled_predictions(db)} == {
        "momentum_top10_h40@v1", "momentum_top10_h60@v1"
    }


@pytest.mark.unit
def test_params_json_must_be_valid_json(tmp_path: Path) -> None:
    """
    `params_json` 存的是稽核依據。不是合法 JSON 就等於沒存——
    寧可寫入時拋錯，不要事後才發現讀不出來。
    """
    db = tmp_path / "forward.db"
    with pytest.raises(ForwardStoreError, match="JSON"):
        record_forward_predictions(db, [momentum_prediction(params_json="不是 JSON")])


@pytest.mark.unit
def test_params_json_must_not_be_empty(tmp_path: Path) -> None:
    """空物件等於沒記參數，違反禁令 8。"""
    db = tmp_path / "forward.db"
    with pytest.raises(ForwardStoreError, match="不可為空"):
        record_forward_predictions(db, [momentum_prediction(params_json="{}")])


# ══════════════════════════════════════════════════════════════
# 結算
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_settle_fills_realized_return_once(tmp_path: Path) -> None:
    db = tmp_path / "forward.db"
    item = momentum_prediction()
    record_forward_predictions(db, [item])
    at = datetime(2026, 11, 10, tzinfo=UTC)

    assert settle_forward_prediction(db, item, realized_return=0.152,
                                     settled_at=at) is True
    assert settle_forward_prediction(db, item, realized_return=0.999,
                                     settled_at=at) is False
    assert list_unsettled_predictions(db) == []

    con = sqlite3.connect(db)
    row = con.execute(
        "SELECT realized_return, settled_at FROM forward_predictions"
    ).fetchone()
    con.close()
    assert row[0] == pytest.approx(0.152)
    assert row[1] == at.isoformat()


@pytest.mark.unit
def test_legacy_schema_is_rejected(tmp_path: Path) -> None:
    """
    舊表（含 edge_z）要明確拋錯。`CREATE TABLE IF NOT EXISTS` 對既有表
    是無聲的，靜默沿用舊結構會讓寫入失敗或欄位錯位。
    """
    db = tmp_path / "forward.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE forward_predictions "
                "(predicted_at TEXT, edge_z REAL, stock_id TEXT)")
    con.commit()
    con.close()

    with pytest.raises(ForwardStoreError, match="params_json"):
        initialize_forward_store(db)
