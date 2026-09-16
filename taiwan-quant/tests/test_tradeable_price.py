"""
成本分層必須用實際成交價，不是還原價

## 問題

`config.costs.resolve_tier(price, amount, is_etf)` 用 `price` 判斷
「這一檔買不買得起整張」：

```python
whole = amount >= price * LOT_SIZE      # LOT_SIZE = 1000
```

N=10 時 `amount = 40,000`，門檻是股價 **40 元**。

但回測傳進去的是還原價。`load_prices(adjusted=True)` 用 `adj_close`
覆寫 `close` 並同比例調整 OHL，所以下游拿不到實際價。

## 幅度（全庫、每年 6 月）

```
年度   還原/實際平均   40 元門檻分層不同的比例
2016      0.8666          13.41%
2020      0.8715          10.89%
2024      0.9259           3.46%
```

還原是**回溯調整、錨在最新日**，所以越早的日期還原價越低。
開發集（2015~2023）正是落差最大的那一段。

## 方向對策略有利

```
還原價偏低  →  看起來買得起整張  →  走 0050-lot（0.671%）
實際價較高  →  其實只能買零股    →  應走 0050（1.071%）
```

**成本被低估。** 與第 14 個 bug（拿 N=3 的部位大小論證 N=10 的成本）
同一類：錯的方向一律是讓結果變好看。

## 解法：兩種價格各有明確欄位

報酬用還原價（禁令 12），可負擔性用實際價。**不共用一個欄位靠命名
區分**——那正是這個 bug 的成因。
"""

from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path

import pytest

from taiwan_quant.config.costs import LOT_SIZE, Tier, resolve_tier
from taiwan_quant.data.loader import (
    RAW_CLOSE_COLUMN,
    RAW_OPEN_COLUMN,
    load_prices,
)


@pytest.fixture
def split_adjusted_db(tmp_path: Path) -> Path:
    """
    一檔股票，還原價是實際價的一半（相當於歷史上做過 1:2 分割）。

    實際價 84 元 → 一張 84,000 元
    還原價 42 元 → 一張 42,000 元

    `amount = 50,000` 時：實際買不起整張，還原「看起來」買得起。
    這就是 bug 的最小重現。
    """
    db = tmp_path / "prices.db"
    con = sqlite3.connect(db)
    con.executescript(
        """
        CREATE TABLE stock_daily (
            stock_id TEXT, date TEXT,
            open REAL, high REAL, low REAL, close REAL, volume REAL
        );
        CREATE TABLE stock_daily_adj (stock_id TEXT, date TEXT, adj_close REAL);
        """
    )
    rows = [("1234", f"2020-01-{day:02d}", 84.0, 85.0, 83.0, 84.0, 1_000_000.0)
            for day in range(1, 11)]
    con.executemany("INSERT INTO stock_daily VALUES (?,?,?,?,?,?,?)", rows)
    con.executemany(
        "INSERT INTO stock_daily_adj VALUES (?,?,?)",
        [("1234", f"2020-01-{day:02d}", 42.0) for day in range(1, 11)],
    )
    con.commit()
    con.close()
    return db


@pytest.mark.unit
def test_adjusted_load_keeps_the_tradeable_price(split_adjusted_db: Path) -> None:
    """
    `adjusted=True` 時 `close` 是還原價，`raw_close` 是實際成交價。

    兩者都要在，因為它們回答不同的問題：
        close      這一趟賺了多少（禁令 12 要求還原）
        raw_close  這一筆買不買得起整張（成本分層）
    """
    prices = load_prices(
        ["1234"], start=date(2020, 1, 1), end=date(2020, 1, 10),
        adjusted=True, db_path=split_adjusted_db,
    )

    assert prices["close"].eq(42.0).all(), "close 必須是還原價"
    assert prices[RAW_CLOSE_COLUMN].eq(84.0).all(), "raw_close 必須是實際價"
    # 進場價是 T+1 開盤，所以開盤也要留實際價；用收盤近似會差一天的
    # 盤中幅度，而「差不多夠用」正是這個 bug 的成因
    assert prices["open"].eq(42.0).all(), "open 必須同比例還原"
    assert prices[RAW_OPEN_COLUMN].eq(84.0).all(), "raw_open 必須是實際價"


@pytest.mark.unit
def test_unadjusted_load_has_identical_columns(split_adjusted_db: Path) -> None:
    """
    `adjusted=False` 時 `close` 本身就是實際價，`raw_close` 與它相同。

    **欄位一律存在**，呼叫端不需要分支判斷——少一個分支就少一個
    「忘記處理」的可能。
    """
    prices = load_prices(
        ["1234"], start=date(2020, 1, 1), end=date(2020, 1, 10),
        adjusted=False, db_path=split_adjusted_db,
    )

    assert prices["close"].eq(84.0).all()
    assert prices[RAW_CLOSE_COLUMN].equals(prices["close"])


@pytest.mark.unit
def test_tier_differs_between_the_two_prices(split_adjusted_db: Path) -> None:
    """
    手算：`amount = 50,000`

        還原價 42 元 × 1000 = 42,000 ≤ 50,000  →  整股（0050-lot, 0.671%）
        實際價 84 元 × 1000 = 84,000 > 50,000  →  零股（0050,     1.071%）

    這 0.4 pp 就是 bug 的大小。
    """
    prices = load_prices(
        ["1234"], start=date(2020, 1, 1), end=date(2020, 1, 10),
        adjusted=True, db_path=split_adjusted_db,
    )
    adjusted_price = float(prices["close"].iloc[0])
    tradeable_price = float(prices[RAW_CLOSE_COLUMN].iloc[0])
    amount = 50_000.0

    assert amount >= adjusted_price * LOT_SIZE
    assert amount < tradeable_price * LOT_SIZE
    assert resolve_tier(price=adjusted_price, amount=amount) is Tier.LARGE_WHOLE
    assert resolve_tier(price=tradeable_price, amount=amount) is Tier.LARGE


@pytest.mark.unit
def test_missing_adjustment_still_reports_the_tradeable_price(
    tmp_path: Path,
) -> None:
    """
    整檔沒有還原價時 `_apply_adjustment` 保留原價（因子 1.0）。
    那時 `raw_close` 仍必須等於實際價，**不可是 NaN**。

    缺值靜默退回還原價是這個 bug 的原始形態，不可換個地方重演。
    """
    db = tmp_path / "noadj.db"
    con = sqlite3.connect(db)
    con.executescript(
        """
        CREATE TABLE stock_daily (
            stock_id TEXT, date TEXT,
            open REAL, high REAL, low REAL, close REAL, volume REAL
        );
        CREATE TABLE stock_daily_adj (stock_id TEXT, date TEXT, adj_close REAL);
        """
    )
    con.executemany(
        "INSERT INTO stock_daily VALUES (?,?,?,?,?,?,?)",
        [("9999", f"2020-01-{day:02d}", 60.0, 61.0, 59.0, 60.0, 500_000.0)
         for day in range(1, 11)],
    )
    con.commit()
    con.close()

    prices = load_prices(
        ["9999"], start=date(2020, 1, 1), end=date(2020, 1, 10),
        adjusted=True, db_path=db,
    )

    assert prices[RAW_CLOSE_COLUMN].notna().all()
    assert prices[RAW_CLOSE_COLUMN].eq(60.0).all()
