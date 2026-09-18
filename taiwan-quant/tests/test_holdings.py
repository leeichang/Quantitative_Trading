"""
持股帳本的測試

承重的三條：

1. **淨報酬要扣來回成本**——只報毛報酬會讓你以為賺了,而 1.071% 的成本
   吃掉的是 40 日 +1.7% 期望值的六成。
2. **重複買進要拋錯**——前推帳本用 `INSERT OR IGNORE` 是為了重跑冪等,
   持股不同:同鍵重複通常是打錯,靜默忽略會讓你以為記進去了。
3. **重複賣出要拋錯**——否則第二次會覆寫第一次的成交價,績效就假了。
"""

from __future__ import annotations

from datetime import date

import pytest

from taiwan_quant.holdings import (
    Holding,
    HoldingsError,
    closed_holdings,
    due_on,
    open_holdings,
    record_purchase,
    record_sale,
)


def _holding(**overrides) -> Holding:
    base = dict(
        stock_id="2884",
        buy_date="2026-09-12",
        buy_price=46.30,
        shares=400,
        strategy_version="margin_spike_h40@v1",
        planned_exit_date="2026-11-10",
        round_trip_cost=0.01071,
        opened_at="2026-09-12T14:00:00+08:00",
        note="",
    )
    base.update(overrides)
    return Holding(**base)


@pytest.fixture
def db(tmp_path):
    return tmp_path / "holdings.db"


# ══════════════════════════════════════════════════════════════
# 買進
# ══════════════════════════════════════════════════════════════


def test_purchase_is_readable_back(db):
    record_purchase(db, _holding())

    rows = open_holdings(db)

    assert len(rows) == 1
    assert rows[0].stock_id == "2884"
    assert rows[0].shares == 400
    assert rows[0].amount == pytest.approx(46.30 * 400)


def test_duplicate_purchase_on_the_same_day_raises(db):
    """同日加碼要併成一列。記兩列會讓均價無法還原"""
    record_purchase(db, _holding())

    with pytest.raises(HoldingsError, match="已有紀錄"):
        record_purchase(db, _holding(shares=200))


def test_same_stock_on_a_different_day_is_a_separate_lot(db):
    record_purchase(db, _holding())
    record_purchase(db, _holding(buy_date="2026-10-01",
                                 planned_exit_date="2026-11-28"))

    assert len(open_holdings(db)) == 2


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"buy_price": 0.0}, "買進價"),
        ({"buy_price": -1.0}, "買進價"),
        ({"shares": 0}, "股數"),
        ({"shares": -5}, "股數"),
        ({"round_trip_cost": 1.5}, "來回成本"),
        ({"round_trip_cost": -0.01}, "來回成本"),
        ({"stock_id": "  "}, "stock_id"),
        ({"buy_date": "2026/09/12"}, "buy_date"),
        ({"planned_exit_date": "2026-09-01"}, "出場日"),
    ],
)
def test_invalid_purchase_fails_fast(db, overrides, message):
    with pytest.raises(HoldingsError, match=message):
        record_purchase(db, _holding(**overrides))


# ══════════════════════════════════════════════════════════════
# 賣出與淨報酬
# ══════════════════════════════════════════════════════════════


def test_net_return_subtracts_the_round_trip_cost(db):
    """
    **承重測試。** 買 46.30 賣 50.00 是毛 +7.991%，
    扣 1.071% 的來回成本之後是淨 +6.920%。

    手算：50.00 / 46.30 − 1 = 0.0799136...
    """
    record_purchase(db, _holding())

    closed = record_sale(
        db, stock_id="2884", buy_date="2026-09-12",
        sell_date="2026-11-10", sell_price=50.00,
    )

    assert closed.gross_return == pytest.approx(50.00 / 46.30 - 1)
    assert closed.gross_return == pytest.approx(0.079914, abs=1e-6)
    assert closed.net_return == pytest.approx(0.079914 - 0.01071, abs=1e-6)
    assert closed.net_return < closed.gross_return


def test_profit_is_net_and_scaled_by_amount(db):
    record_purchase(db, _holding())

    closed = record_sale(
        db, stock_id="2884", buy_date="2026-09-12",
        sell_date="2026-11-10", sell_price=50.00,
    )

    assert closed.profit == pytest.approx(46.30 * 400 * closed.net_return)


def test_a_small_gain_can_be_a_net_loss(db):
    """毛 +0.5% 在 1.071% 的成本下是淨虧。這是成本地板的日常樣貌"""
    record_purchase(db, _holding())

    closed = record_sale(
        db, stock_id="2884", buy_date="2026-09-12",
        sell_date="2026-11-10", sell_price=46.30 * 1.005,
    )

    assert closed.gross_return > 0
    assert closed.net_return < 0
    assert closed.profit < 0


def test_selling_twice_raises(db):
    record_purchase(db, _holding())
    record_sale(db, stock_id="2884", buy_date="2026-09-12",
                sell_date="2026-11-10", sell_price=50.0)

    with pytest.raises(HoldingsError, match="已於.*平倉"):
        record_sale(db, stock_id="2884", buy_date="2026-09-12",
                    sell_date="2026-11-11", sell_price=99.0)


def test_selling_an_unknown_holding_raises(db):
    with pytest.raises(HoldingsError, match="找不到"):
        record_sale(db, stock_id="9999", buy_date="2026-09-12",
                    sell_date="2026-11-10", sell_price=50.0)


def test_a_sold_holding_leaves_the_open_list(db):
    record_purchase(db, _holding())
    record_sale(db, stock_id="2884", buy_date="2026-09-12",
                sell_date="2026-11-10", sell_price=50.0)

    assert open_holdings(db) == []
    assert len(closed_holdings(db)) == 1


@pytest.mark.parametrize("price", [0.0, -1.0])
def test_invalid_sell_price_raises(db, price):
    record_purchase(db, _holding())

    with pytest.raises(HoldingsError, match="賣出價"):
        record_sale(db, stock_id="2884", buy_date="2026-09-12",
                    sell_date="2026-11-10", sell_price=price)


# ══════════════════════════════════════════════════════════════
# 到期提醒
# ══════════════════════════════════════════════════════════════


def test_due_on_includes_holdings_at_or_past_the_planned_exit(db):
    record_purchase(db, _holding())

    assert due_on(db, date(2026, 11, 9)) == []
    assert len(due_on(db, date(2026, 11, 10))) == 1
    # 過期的也要列出來——漏賣比早賣更需要看見
    assert len(due_on(db, date(2026, 12, 25))) == 1


def test_due_on_skips_already_sold_holdings(db):
    record_purchase(db, _holding())
    record_sale(db, stock_id="2884", buy_date="2026-09-12",
                sell_date="2026-11-10", sell_price=50.0)

    assert due_on(db, date(2026, 12, 25)) == []


def test_open_holdings_are_ordered_by_urgency(db):
    record_purchase(db, _holding(stock_id="1111",
                                 planned_exit_date="2026-12-01"))
    record_purchase(db, _holding(stock_id="2222",
                                 planned_exit_date="2026-11-10"))

    assert [h.stock_id for h in open_holdings(db)] == ["2222", "1111"]
