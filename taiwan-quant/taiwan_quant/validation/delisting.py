"""下市與暫時缺價的明確分類及結算政策。"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from pathlib import Path

import pandas as pd


class MissingPriceKind(StrEnum):
    """持有期終點價格狀態。"""

    COMPLETE = "complete"
    DELISTED = "delisted"
    SUSPENDED_OR_MISSING = "suspended_or_missing"
    MISSING_ENTRY = "missing_entry"


@dataclass(frozen=True)
class HoldingSettlement:
    """持有期的可稽核結算結果。"""

    kind: MissingPriceKind
    gross_return: float | None
    exit_date: pd.Timestamp | None


def load_delisted_dates(
    db_path: Path,
    *,
    as_of: date,
) -> dict[str, date | None]:
    """載入截至 as_of 已知的下市日，避免倒用未來下市事件。"""
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as con:
        rows = con.execute(
            "SELECT stock_id, delisted_date FROM stock_master"
        ).fetchall()
    return {
        stock_id: (
            parsed if (parsed := date.fromisoformat(value)) <= as_of else None
        )
        if value
        else None
        for stock_id, value in rows
    }


def settle_holding_period(
    bars: pd.DataFrame,
    *,
    entry_date: pd.Timestamp,
    target_date: pd.Timestamp,
    delisted_date: date | None,
) -> HoldingSettlement:
    """
    用真實存在的價格結算，不以前值回填假裝成交。

    正常情況用 T+1 開盤到目標日收盤；已知永久下市且目標日無價時，政策 B
    用進場後最後有價日收盤結算，之後視為持有現金到原到期日。若之後恢復
    交易或沒有下市證據，只標記缺價，不擅自成交。
    """
    if entry_date not in bars.index:
        return HoldingSettlement(MissingPriceKind.MISSING_ENTRY, None, None)
    entry = float(bars.loc[entry_date, "open"])
    if not math.isfinite(entry) or entry <= 0:
        return HoldingSettlement(MissingPriceKind.MISSING_ENTRY, None, None)

    if target_date in bars.index:
        exit_price = float(bars.loc[target_date, "close"])
        if math.isfinite(exit_price) and exit_price > 0:
            return HoldingSettlement(
                MissingPriceKind.COMPLETE,
                exit_price / entry - 1.0,
                target_date,
            )

    if bool((bars.index > target_date).any()):
        return HoldingSettlement(
            MissingPriceKind.SUSPENDED_OR_MISSING, None, None
        )

    eligible = bars.loc[(bars.index >= entry_date) & (bars.index < target_date)]
    delisting_known_by_target = (
        delisted_date is not None and delisted_date <= target_date.date()
    )
    if delisting_known_by_target and not eligible.empty:
        valid = eligible["close"].astype(float)
        valid = valid[valid.map(lambda value: math.isfinite(value) and value > 0)]
        if not valid.empty:
            exit_day = pd.Timestamp(valid.index[-1])
            return HoldingSettlement(
                MissingPriceKind.DELISTED,
                float(valid.iloc[-1]) / entry - 1.0,
                exit_day,
            )

    return HoldingSettlement(MissingPriceKind.SUSPENDED_OR_MISSING, None, None)
