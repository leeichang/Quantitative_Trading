"""交易日缺口分類與決策相關價格守門。"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date

import pandas as pd

from taiwan_quant.validation.delisting import (
    MissingPriceKind,
    settle_holding_period,
)


class DataIntegrityError(RuntimeError):
    """候選股票在進出場日期缺少真實價格。"""


@dataclass(frozen=True)
class GapRun:
    """某檔股票在自身有價期間內的一段連續缺口。"""

    start: pd.Timestamp
    end: pd.Timestamp
    length: int

    @property
    def shape(self) -> str:
        return "single" if self.length == 1 else "block"


@dataclass(frozen=True)
class HoldingPosition:
    """守門後確定會成交且可結算的部位。"""

    stock_id: str
    gross_return: float
    kind: MissingPriceKind


def find_gap_runs(
    observed_dates: Iterable[pd.Timestamp],
    market_calendar: Sequence[pd.Timestamp],
) -> tuple[GapRun, ...]:
    """依市場交易日找出首末有價日之間的缺口，並合併連續日期。"""
    observed = pd.DatetimeIndex(observed_dates).sort_values().unique()
    calendar = pd.DatetimeIndex(market_calendar).sort_values().unique()
    if observed.empty or calendar.empty:
        return ()

    active = calendar[(calendar >= observed[0]) & (calendar <= observed[-1])]
    missing = active.difference(observed)
    if missing.empty:
        return ()

    calendar_positions = {day: idx for idx, day in enumerate(calendar)}
    runs: list[GapRun] = []
    start = previous = pd.Timestamp(missing[0])
    for current_value in missing[1:]:
        current = pd.Timestamp(current_value)
        if calendar_positions[current] != calendar_positions[previous] + 1:
            runs.append(
                GapRun(
                    start=start,
                    end=previous,
                    length=calendar_positions[previous] - calendar_positions[start] + 1,
                )
            )
            start = current
        previous = current
    runs.append(
        GapRun(
            start=start,
            end=previous,
            length=calendar_positions[previous] - calendar_positions[start] + 1,
        )
    )
    return tuple(runs)


def _has_tradeable_price(frame: pd.DataFrame, day: pd.Timestamp, stock_id: str) -> bool:
    if day not in frame.index or stock_id not in frame.columns:
        return False
    value = float(frame.at[day, stock_id])
    return math.isfinite(value) and value > 0


def assert_holding_price_completeness(
    *,
    decision_date: pd.Timestamp,
    calendar: Sequence[pd.Timestamp],
    candidates: Iterable[str],
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    holding_days: int,
) -> None:
    """候選池任一股票缺 T+1 開盤或 T+H 收盤時立即失敗。"""
    dates = list(calendar)
    try:
        decision_index = dates.index(decision_date)
    except ValueError as exc:
        raise DataIntegrityError(f"決策日 {decision_date.date()} 不在交易日曆") from exc

    entry_index = decision_index + 1
    exit_index = entry_index + holding_days
    if exit_index >= len(dates):
        raise DataIntegrityError(
            f"決策日 {decision_date.date()} 後不足 {holding_days} 個交易日"
        )
    entry_day = dates[entry_index]
    exit_day = dates[exit_index]

    problems: list[str] = []
    for stock_id in candidates:
        if not _has_tradeable_price(opens, entry_day, stock_id):
            problems.append(f"{stock_id} T+1 {entry_day.date()}")
        if not _has_tradeable_price(closes, exit_day, stock_id):
            problems.append(f"{stock_id} T+H {exit_day.date()}")
    if problems:
        raise DataIntegrityError(
            f"決策日 {decision_date.date()} 的候選價格不完整：" + ", ".join(problems)
        )


def select_holding_positions(
    *,
    decision_date: pd.Timestamp,
    calendar: Sequence[pd.Timestamp],
    ordered_candidates: Iterable[str],
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    holding_days: int,
    n_positions: int,
    delisted_dates: Mapping[str, date | None],
) -> tuple[HoldingPosition, ...]:
    """依排名選滿 N 檔；共用下市分類，並檢查每個實際遞補者。"""
    if n_positions < 1:
        raise ValueError("n_positions 必須至少為 1")
    dates = list(calendar)
    try:
        decision_index = dates.index(decision_date)
    except ValueError as exc:
        raise DataIntegrityError(f"決策日 {decision_date.date()} 不在交易日曆") from exc

    entry_index = decision_index + 1
    target_index = entry_index + holding_days
    if target_index >= len(dates):
        raise DataIntegrityError(
            f"決策日 {decision_date.date()} 後不足 {holding_days} 個交易日"
        )
    entry_date = dates[entry_index]
    target_date = dates[target_index]

    selected: list[HoldingPosition] = []
    for stock_id in ordered_candidates:
        if stock_id not in opens.columns or stock_id not in closes.columns:
            continue
        bars = pd.concat(
            [opens[stock_id].rename("open"), closes[stock_id].rename("close")],
            axis=1,
        ).dropna(how="all")
        settlement = settle_holding_period(
            bars,
            entry_date=entry_date,
            target_date=target_date,
            delisted_date=delisted_dates.get(stock_id),
        )
        if settlement.kind is MissingPriceKind.MISSING_ENTRY:
            continue
        if settlement.kind is MissingPriceKind.SUSPENDED_OR_MISSING:
            raise DataIntegrityError(
                f"決策日 {decision_date.date()} 的實際候選價格不完整："
                f"{stock_id} {settlement.kind.value} T+H {target_date.date()}"
            )
        if settlement.gross_return is None:
            raise AssertionError("可結算部位缺少 gross_return")
        selected.append(
            HoldingPosition(
                stock_id=stock_id,
                gross_return=settlement.gross_return,
                kind=settlement.kind,
            )
        )
        if len(selected) == n_positions:
            break
    return tuple(selected)
