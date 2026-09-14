"""
交易日曆推算測試

## 抓到的 bug（第 11 個）

`record_forward.py` 原本用 `pd.offsets.BDay(horizon)` 推標籤揭曉日：

```python
due_date = (pd.Timestamp(as_of) + pd.offsets.BDay(horizon)).date()   # ← bug
```

`BDay` 只跳週末，**不管台股休市**。實測 2015-2026 的 2,850 個交易日：

```
交易日數   實際中位跨度   BDay 推算   誤差
  20         29 日          28 日      -1
  60         89 日          84 日      -5
 120        180 日         168 日     -12
```

單次最糟差 14 天：2025-01-02 起算 60 個交易日，實際揭曉日是
2025-04-10，BDay 推得 2025-03-27——中間卡了農曆年與 228。

## 為什麼一定要推估，不能查

前推預測的揭曉日在**未來**，資料庫裡還沒有那些交易日。所以只能推估。
推估要用歷史上同樣交易日數的實際跨度中位數，不是週末近似。

日曆真的涵蓋到期日時（例如補記過去的預測），就直接數，不推估。
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from taiwan_quant.data.calendar import (
    CalendarError,
    due_trading_date,
    median_calendar_span,
)


def weekdays(start: date, count: int, skip: set[date] | None = None) -> list[date]:
    """連續營業日，可挖掉指定的休市日"""
    skip = skip or set()
    out: list[date] = []
    cursor = start
    while len(out) < count:
        if cursor.weekday() < 5 and cursor not in skip:
            out.append(cursor)
        cursor += timedelta(days=1)
    return out


# ══════════════════════════════════════════════════════════════
# 日曆涵蓋得到：直接數
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_counts_actual_trading_days_when_calendar_covers_due_date() -> None:
    """
    手算：2025-01-06（一）起算 5 個交易日。

    01-07、01-08、01-09、01-10、01-13（週末跳過）→ 揭曉日 01-13。
    """
    cal = weekdays(date(2025, 1, 6), 30)

    assert due_trading_date(cal, date(2025, 1, 6), horizon=5) == date(2025, 1, 13)


@pytest.mark.unit
def test_holiday_pushes_due_date_one_trading_day_later() -> None:
    """
    手算：同上但 01-09 休市。

    01-07、01-08、01-10、01-13、01-14 → 揭曉日 01-14，比無休市晚一天。
    `BDay` 會漏掉這一天。
    """
    cal = weekdays(date(2025, 1, 6), 30, skip={date(2025, 1, 9)})

    assert due_trading_date(cal, date(2025, 1, 6), horizon=5) == date(2025, 1, 14)


@pytest.mark.unit
def test_as_of_need_not_be_a_trading_day() -> None:
    """
    決策日落在週末時，從它之後的第一個交易日開始數。

    手算：2025-01-11（六）起算 5 個交易日 → 01-13、14、15、16、17。
    """
    cal = weekdays(date(2025, 1, 6), 30)

    assert due_trading_date(cal, date(2025, 1, 11), horizon=5) == date(2025, 1, 17)


# ══════════════════════════════════════════════════════════════
# 日曆不夠長：用歷史中位跨度推估
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_median_span_of_pure_weekday_calendar_is_exactly_one_week() -> None:
    """
    純營業日日曆（無休市）裡，任意 5 個交易日的跨度恆為 7 日曆日。

    週一到下週一、週二到下週二，都是 +7。中位數必然是 7。
    """
    cal = weekdays(date(2025, 1, 6), 60)

    assert median_calendar_span(cal, horizon=5) == 7


@pytest.mark.unit
def test_estimates_from_as_of_when_calendar_runs_out() -> None:
    """
    真正的前推預測：揭曉日在資料庫最後一天之後，只能推估。

    純營業日日曆的 5 交易日中位跨度是 7 日 → 最後一天 + 7 日。
    """
    cal = weekdays(date(2025, 1, 6), 60)
    last = cal[-1]

    assert due_trading_date(cal, last, horizon=5) == last + timedelta(days=7)


@pytest.mark.unit
def test_estimate_anchors_on_as_of_not_on_calendar_end() -> None:
    """
    決策日已經超出日曆時，推估要從**決策日**往後推，不是從日曆末端。

    否則補記舊預測會全部擠在同一天。
    """
    cal = weekdays(date(2025, 1, 6), 60)
    future = cal[-1] + timedelta(days=30)

    assert due_trading_date(cal, future, horizon=5) == future + timedelta(days=7)


@pytest.mark.unit
def test_partial_future_coverage_still_estimates() -> None:
    """
    日曆只剩 3 個未來交易日、但需要 5 個時，不可回傳最後一天充數。
    """
    cal = weekdays(date(2025, 1, 6), 60)
    as_of = cal[-4]

    result = due_trading_date(cal, as_of, horizon=5)

    assert result > cal[-1]
    assert result == as_of + timedelta(days=7)


# ══════════════════════════════════════════════════════════════
# 邊界
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_rejects_non_positive_horizon() -> None:
    cal = weekdays(date(2025, 1, 6), 30)
    with pytest.raises(CalendarError, match="必須為正"):
        due_trading_date(cal, date(2025, 1, 6), horizon=0)


@pytest.mark.unit
def test_rejects_calendar_too_short_to_estimate() -> None:
    """
    日曆比持有期還短時無法推估，必須明確拋錯而不是猜一個數字。
    """
    cal = weekdays(date(2025, 1, 6), 3)
    with pytest.raises(CalendarError, match="不足"):
        due_trading_date(cal, cal[-1], horizon=5)


@pytest.mark.unit
def test_rejects_unsorted_calendar() -> None:
    cal = weekdays(date(2025, 1, 6), 30)
    with pytest.raises(CalendarError, match="升冪"):
        due_trading_date(list(reversed(cal)), cal[0], horizon=5)


# ══════════════════════════════════════════════════════════════
# integration：真實台股日曆
# ══════════════════════════════════════════════════════════════


@pytest.mark.integration
def test_real_taiwan_calendar_beats_bday_across_lunar_new_year() -> None:
    """
    用真實日曆重現那 14 天的誤差。

    2025-01-02 起算 60 個交易日：實際 2025-04-10，BDay 推得 2025-03-27。
    """
    import sqlite3

    from taiwan_quant.data.loader import HISTORY_DB_PATH

    if not HISTORY_DB_PATH.exists():
        pytest.skip("history.db 不存在")

    con = sqlite3.connect(f"file:{HISTORY_DB_PATH}?mode=ro", uri=True)
    cal = [
        date.fromisoformat(row[0])
        for row in con.execute("SELECT DISTINCT date FROM stock_daily ORDER BY date")
    ]
    con.close()

    assert due_trading_date(cal, date(2025, 1, 2), horizon=60) == date(2025, 4, 10)
    assert median_calendar_span(cal, horizon=60) == 89
