"""
交易日曆推算

## 為什麼不能用 `pd.offsets.BDay`

`BDay` 只跳週末，**不管台股休市**。實測 2015-2026 的 2,850 個交易日：

```
交易日數   實際中位跨度   BDay 推算   誤差
  20         29 日          28 日      -1
  60         89 日          84 日      -5
 120        180 日         168 日     -12
```

單次最糟差 14 天：2025-01-02 起算 60 個交易日，實際揭曉日 2025-04-10，
`BDay` 推得 2025-03-27——中間卡了農曆年與 228。

## 為什麼一定要推估，不能一律查表

前推預測的揭曉日在**未來**，資料庫裡還沒有那些交易日。所以到期日只能
推估；能查得到的時候（例如補記過去的預測）就直接數，不推估。

推估用歷史上同樣交易日數的實際跨度**中位數**，不是週末近似。中位數比
平均數穩健：農曆年那幾週會把平均往上拉，但它一年只出現一次。
"""

from __future__ import annotations

import sqlite3
from bisect import bisect_right
from datetime import date, timedelta
from pathlib import Path
from statistics import median


class CalendarError(RuntimeError):
    """日曆不合法或不足以推算"""


def load_trading_calendar(db_path: Path) -> list[date]:
    """
    從 `stock_daily` 取出所有交易日（升冪、去重）。

    Args:
        db_path: SQLite 檔案路徑

    Returns:
        交易日清單

    Raises:
        CalendarError: 資料表為空

    唯讀開啟，不會被回補流程的寫入鎖卡住。
    """
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as con:
        rows = con.execute(
            "SELECT DISTINCT date FROM stock_daily ORDER BY date"
        ).fetchall()
    if not rows:
        raise CalendarError(f"{db_path} 的 stock_daily 沒有任何交易日")
    return [date.fromisoformat(row[0]) for row in rows]


def median_calendar_span(calendar: list[date], horizon: int) -> int:
    """
    歷史上 `horizon` 個交易日實際跨越幾個日曆日（中位數）。

    Args:
        calendar: 交易日清單（升冪）
        horizon: 交易日數

    Returns:
        日曆日數

    Raises:
        CalendarError: `horizon` 非正，或日曆短到湊不出一組跨度
    """
    _validate(calendar, horizon)
    if len(calendar) <= horizon:
        raise CalendarError(
            f"日曆只有 {len(calendar)} 個交易日，不足以推算 {horizon} 日跨度"
        )
    spans = [
        (calendar[i + horizon] - calendar[i]).days
        for i in range(len(calendar) - horizon)
    ]
    return int(median(spans))


def due_trading_date(calendar: list[date], as_of: date, horizon: int) -> date:
    """
    決策日之後第 `horizon` 個交易日——標籤揭曉日。

    Args:
        calendar: 交易日清單（升冪）
        as_of: 決策日；不必是交易日
        horizon: 持有交易日數

    Returns:
        揭曉日。日曆涵蓋得到就是實際交易日，否則是推估值

    Raises:
        CalendarError: 參數不合法，或日曆短到無法推估

    日曆剩餘的未來交易日不足 `horizon` 個時（真正的前推預測必然如此），
    回傳 `as_of + 歷史中位跨度`。**錨點是 `as_of`，不是日曆末端**——
    否則補記舊預測會全部擠在同一天。
    """
    _validate(calendar, horizon)
    future = calendar[bisect_right(calendar, as_of) :]
    if len(future) >= horizon:
        return future[horizon - 1]
    return as_of + timedelta(days=median_calendar_span(calendar, horizon))


def _validate(calendar: list[date], horizon: int) -> None:
    if horizon < 1:
        raise CalendarError(f"horizon 必須為正，得到 {horizon}")
    if not calendar:
        raise CalendarError("交易日曆是空的")
    if any(b < a for a, b in zip(calendar, calendar[1:], strict=False)):
        raise CalendarError("交易日曆必須依日期升冪排序")
