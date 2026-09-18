"""
持股帳本：記錄**你實際買了什麼**，而不是策略說了什麼

## 為什麼要和前推帳本分開

`forward_predictions` 記的是策略的輸出——每天觸發的全部名單，用來累積
證據。**它不是你的持股。**

兩者必然不同：

```
前推帳本   每天 0~3 檔，全部記，40 萬買不完（穩態 108 檔）
持股帳本   你實際買的，受 20 檔上限與現金限制，會跳過大部分訊號
```

混在一起會讓兩個問題都答不了：策略的效應要看全部訊號，
你的績效要看實際成交價與實際股數。

## 出場是機械的，不是預測

`planned_exit_date` 是決策日 +40 個交易日，由交易日曆推算。

⚠️ **這不是「預測的最高點」。** 這個專案試過動態出場：移動停損在
1.56 年多頭樣本（915 個交易日）上是 +372.79%，資料拉到 2,850 個交易日
之後是 **−68.91%**（見 `原理說明/2026-09-12_移動停損與多槽位組合原理.md`
與 09-13 的更正）。**那個正報酬是環境的產物。**

抓頂比抓方向更難，而方向本身都還沒測穩。帳本只做記帳，不做預測。

## 成本存在買進當下

`round_trip_cost` 由 `config.costs.resolve_tier` 逐檔決定，買進時就固定。
事後用「代表值」重算會讓實際淨報酬無法還原（禁令 8）。
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, fields
from datetime import date, datetime
from pathlib import Path


class HoldingsError(RuntimeError):
    """持股帳本的結構或內容不合法。"""


@dataclass(frozen=True)
class Holding:
    """
    一筆實際持股。

    只有**買進當下可知**的欄位。賣出資訊由 `record_sale` 補，
    放進來會讓「未平倉」這個狀態變得可以偽造。
    """

    stock_id: str
    buy_date: str
    """實際成交日（不是決策日）"""

    buy_price: float
    """實際成交價。**不是參考價**——績效要用真的"""

    shares: int
    strategy_version: str
    """來源策略，或 `manual`。用來分辨哪些部位該算進策略績效"""

    planned_exit_date: str
    """機械出場日（T+40 交易日）。不是預測的高點"""

    round_trip_cost: float
    opened_at: str
    note: str = ""

    @property
    def amount(self) -> float:
        return self.buy_price * self.shares


@dataclass(frozen=True)
class ClosedHolding:
    """已平倉的持股，含實際淨報酬。"""

    holding: Holding
    sell_date: str
    sell_price: float
    closed_at: str

    @property
    def gross_return(self) -> float:
        return self.sell_price / self.holding.buy_price - 1.0

    @property
    def net_return(self) -> float:
        """扣掉買進當下固定的來回成本"""
        return self.gross_return - self.holding.round_trip_cost

    @property
    def profit(self) -> float:
        return self.holding.amount * self.net_return


SCHEMA = """
CREATE TABLE IF NOT EXISTS holdings (
    stock_id TEXT NOT NULL,
    buy_date TEXT NOT NULL,
    buy_price REAL NOT NULL,
    shares INTEGER NOT NULL,
    strategy_version TEXT NOT NULL,
    planned_exit_date TEXT NOT NULL,
    round_trip_cost REAL NOT NULL,
    opened_at TEXT NOT NULL,
    note TEXT NOT NULL DEFAULT '',
    sell_date TEXT,
    sell_price REAL,
    closed_at TEXT,
    PRIMARY KEY (stock_id, buy_date)
);
"""
"""
主鍵是 (代號, 買進日)。同一天加碼同一檔要併成一列，不是兩列——
兩列會讓平均成本無法還原。不同日買的同一檔是不同列，那是不同批。
"""

_COLUMNS = tuple(field.name for field in fields(Holding))


def initialize_holdings_store(db_path: Path) -> None:
    """建表。`CREATE TABLE IF NOT EXISTS` 對既有表無聲，所以建完要驗。"""
    with sqlite3.connect(db_path) as con:
        con.execute(SCHEMA)
        columns = {row[1] for row in con.execute("PRAGMA table_info(holdings)")}
    required = set(_COLUMNS) | {"sell_date", "sell_price", "closed_at"}
    missing = required - columns
    if missing:
        raise HoldingsError(
            f"holdings 表缺少欄位 {sorted(missing)}——可能是舊結構，請先遷移"
        )


def _validate(holding: Holding) -> None:
    if not holding.stock_id.strip():
        raise HoldingsError("stock_id 不可為空")
    if holding.buy_price <= 0:
        raise HoldingsError(f"買進價必須為正，得到 {holding.buy_price}")
    if holding.shares <= 0:
        raise HoldingsError(f"股數必須為正，得到 {holding.shares}")
    if not 0.0 <= holding.round_trip_cost < 1.0:
        raise HoldingsError(
            f"來回成本必須在 [0, 1) 之間，得到 {holding.round_trip_cost}"
        )
    for label, value in (
        ("buy_date", holding.buy_date),
        ("planned_exit_date", holding.planned_exit_date),
    ):
        try:
            date.fromisoformat(value)
        except ValueError as exc:
            raise HoldingsError(f"{label} 必須是 ISO 日期，得到 {value!r}") from exc
    if holding.planned_exit_date < holding.buy_date:
        raise HoldingsError(
            f"出場日 {holding.planned_exit_date} 不可早於買進日 {holding.buy_date}"
        )


def record_purchase(db_path: Path, holding: Holding) -> None:
    """
    記一筆買進。

    Raises:
        HoldingsError: 內容不合法，或同一檔同一天已有紀錄

    **同鍵重複直接拋錯,不靜默忽略。** 前推帳本用 `INSERT OR IGNORE` 是
    因為重跑同一天應該冪等；持股不同——重複買進通常是打錯,要讓你看到。
    """
    _validate(holding)
    initialize_holdings_store(db_path)
    placeholders = ", ".join("?" * len(_COLUMNS))
    with sqlite3.connect(db_path) as con:
        try:
            con.execute(
                f"INSERT INTO holdings ({', '.join(_COLUMNS)}) "
                f"VALUES ({placeholders})",
                tuple(getattr(holding, name) for name in _COLUMNS),
            )
        except sqlite3.IntegrityError as exc:
            raise HoldingsError(
                f"{holding.stock_id} 在 {holding.buy_date} 已有紀錄——"
                "同日加碼請併成一列（修改股數與均價），不要記兩列"
            ) from exc


def record_sale(
    db_path: Path,
    *,
    stock_id: str,
    buy_date: str,
    sell_date: str,
    sell_price: float,
) -> ClosedHolding:
    """
    記一筆賣出並回傳結果。

    Raises:
        HoldingsError: 找不到該筆、已平倉、或賣價不合法
    """
    if sell_price <= 0:
        raise HoldingsError(f"賣出價必須為正，得到 {sell_price}")
    initialize_holdings_store(db_path)

    with sqlite3.connect(db_path) as con:
        row = con.execute(
            f"SELECT {', '.join(_COLUMNS)}, sell_date FROM holdings "
            "WHERE stock_id = ? AND buy_date = ?",
            (stock_id, buy_date),
        ).fetchone()
        if row is None:
            raise HoldingsError(f"找不到 {stock_id} 在 {buy_date} 的持股")
        if row[-1] is not None:
            raise HoldingsError(
                f"{stock_id} ({buy_date}) 已於 {row[-1]} 平倉，不可重複賣出"
            )
        closed_at = datetime.now().astimezone().isoformat()
        con.execute(
            "UPDATE holdings SET sell_date = ?, sell_price = ?, closed_at = ? "
            "WHERE stock_id = ? AND buy_date = ?",
            (sell_date, sell_price, closed_at, stock_id, buy_date),
        )

    holding = Holding(**dict(zip(_COLUMNS, row[:-1])))
    return ClosedHolding(
        holding=holding, sell_date=sell_date,
        sell_price=sell_price, closed_at=closed_at,
    )


def open_holdings(db_path: Path) -> list[Holding]:
    """未平倉的持股，依計畫出場日升冪——最急的排前面。"""
    initialize_holdings_store(db_path)
    with sqlite3.connect(db_path) as con:
        rows = con.execute(
            f"SELECT {', '.join(_COLUMNS)} FROM holdings "
            "WHERE sell_date IS NULL ORDER BY planned_exit_date, stock_id"
        ).fetchall()
    return [Holding(**dict(zip(_COLUMNS, row))) for row in rows]


def closed_holdings(db_path: Path) -> list[ClosedHolding]:
    """已平倉的持股，依賣出日升冪。"""
    initialize_holdings_store(db_path)
    with sqlite3.connect(db_path) as con:
        rows = con.execute(
            f"SELECT {', '.join(_COLUMNS)}, sell_date, sell_price, closed_at "
            "FROM holdings WHERE sell_date IS NOT NULL ORDER BY sell_date, stock_id"
        ).fetchall()
    return [
        ClosedHolding(
            holding=Holding(**dict(zip(_COLUMNS, row[: len(_COLUMNS)]))),
            sell_date=row[len(_COLUMNS)],
            sell_price=row[len(_COLUMNS) + 1],
            closed_at=row[len(_COLUMNS) + 2],
        )
        for row in rows
    ]


def due_on(db_path: Path, today: date) -> list[Holding]:
    """
    計畫出場日已到（或已過）的未平倉持股。

    **這是機械的到期提醒,不是「現在是高點」。** 過期的也會列出來——
    漏賣比早賣更需要看見。
    """
    return [
        holding
        for holding in open_holdings(db_path)
        if date.fromisoformat(holding.planned_exit_date) <= today
    ]
