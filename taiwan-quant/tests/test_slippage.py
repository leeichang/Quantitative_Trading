"""
跳動單位滑價模型的測試

承重的三條：

1. **半價差是算出來的，不是假設**——手算跳動單位，不拿另一條程式路徑比對。
2. **零股倍數不可小於 1**——零股簿比整股薄，不會更便宜。
3. **當日無成交回 inf，不是 0**——買不到的東西不該被估成零成本。
"""

from __future__ import annotations

import math

import pytest

from taiwan_quant.config.slippage import (
    SlippageError,
    affordable_whole_lots,
    estimate,
    half_spread,
    market_impact,
)


# ══════════════════════════════════════════════════════════════
# half_spread
# ══════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "price, tick, expected",
    [
        (9.82, 0.01, 0.01 / 2 / 9.82),      # <10   跳動 0.01
        (46.30, 0.05, 0.05 / 2 / 46.30),    # <50   跳動 0.05
        (70.0, 0.10, 0.10 / 2 / 70.0),      # <100  跳動 0.10
        (234.5, 0.50, 0.50 / 2 / 234.5),    # <500  跳動 0.50
        (790.0, 1.00, 1.00 / 2 / 790.0),    # <1000 跳動 1.00
        (2410.0, 5.00, 5.00 / 2 / 2410.0),  # >=1000 跳動 5.00
    ],
)
def test_half_spread_follows_the_tick_table(price, tick, expected):
    """手算：半價差 = 跳動單位 ÷ 2 ÷ 價格"""
    assert half_spread(price) == pytest.approx(expected)


def test_half_spread_is_far_below_the_legacy_odd_lot_spec():
    """
    **這是換模型的理由。** 禁令 4 的零股規格是 0.3% 單邊，
    而跳動單位隱含值在每一個價位都遠低於它。
    """
    from taiwan_quant.config.costs import SLIPPAGE, Tier

    legacy = SLIPPAGE[Tier.LARGE]
    assert legacy == pytest.approx(0.003)
    for price in (46.30, 234.5, 2410.0, 5765.0):
        assert half_spread(price) < legacy / 2, (
            f"{price} 元的半價差 {half_spread(price):.4%} "
            f"應遠低於規格 {legacy:.2%}"
        )


@pytest.mark.parametrize("price", [0.0, -1.0])
def test_half_spread_rejects_non_positive_price(price):
    with pytest.raises(SlippageError, match="價格"):
        half_spread(price)


# ══════════════════════════════════════════════════════════════
# market_impact
# ══════════════════════════════════════════════════════════════


def test_market_impact_scales_with_position_share():
    """手算：4 萬 ÷ 4 億 × 0.1 = 1e-5"""
    assert market_impact(40_000, 400_000_000) == pytest.approx(1e-5)


def test_market_impact_is_material_for_illiquid_names():
    """40 萬買日成交額 200 萬的股票是 20% 的量，衝擊不可忽略"""
    assert market_impact(400_000, 2_000_000) == pytest.approx(0.02)


def test_market_impact_is_infinite_when_nothing_traded():
    """
    **當天無成交回 inf，不是 0。**

    估成 0 等於假設買得到買不到的東西——那是這個專案反覆踩到的那類錯。
    """
    assert math.isinf(market_impact(40_000, 0.0))
    assert math.isinf(market_impact(40_000, -1.0))


def test_market_impact_rejects_negative_inputs():
    with pytest.raises(SlippageError, match="金額"):
        market_impact(-1.0, 1_000_000)
    with pytest.raises(SlippageError, match="係數"):
        market_impact(40_000, 1_000_000, coefficient=-0.1)


# ══════════════════════════════════════════════════════════════
# estimate
# ══════════════════════════════════════════════════════════════


def test_whole_lots_carry_no_odd_lot_premium():
    result = estimate(
        price=9.82, amount=40_000, daily_turnover=60_000_000,
        whole_lots=True, odd_lot_multiplier=2.0,
    )

    assert result.odd_lot_premium == pytest.approx(0.0)
    assert result.total == pytest.approx(result.half_spread + result.market_impact)


def test_odd_lot_premium_scales_with_the_multiplier():
    """倍數 2.0 → 加成等於一倍半價差；倍數 1.0 → 加成為零"""
    kwargs = dict(price=46.30, amount=40_000,
                  daily_turnover=60_000_000, whole_lots=False)

    one = estimate(**kwargs, odd_lot_multiplier=1.0)
    two = estimate(**kwargs, odd_lot_multiplier=2.0)
    three = estimate(**kwargs, odd_lot_multiplier=3.0)

    assert one.odd_lot_premium == pytest.approx(0.0)
    assert two.odd_lot_premium == pytest.approx(one.half_spread)
    assert three.odd_lot_premium == pytest.approx(2 * one.half_spread)


def test_multiplier_below_one_is_rejected():
    """
    **零股不可能比整股便宜。** 委託簿更薄，價差只會更寬。
    允許小於 1 會讓「把零股調到比整股便宜」變成一個可以偷偷做的調參。
    """
    with pytest.raises(SlippageError, match="不可小於 1"):
        estimate(price=46.30, amount=40_000, daily_turnover=1e8,
                 whole_lots=False, odd_lot_multiplier=0.5)


def test_estimate_decomposition_sums_to_total():
    result = estimate(price=2410.0, amount=40_000,
                      daily_turnover=5_000_000_000, whole_lots=False)

    assert result.total == pytest.approx(
        result.half_spread + result.odd_lot_premium + result.market_impact
    )


# ══════════════════════════════════════════════════════════════
# affordable_whole_lots
# ══════════════════════════════════════════════════════════════


def test_high_priced_stocks_are_never_whole_lots_at_this_capital():
    """
    2330 一張 241 萬——40 萬永遠買不起，**而那與流動性無關**。

    兩段式滑價把「買不起一張」當成「流動性差」，這條把它們分開。
    """
    assert not affordable_whole_lots(2410.0, 400_000)
    assert not affordable_whole_lots(2410.0, 40_000)
    assert affordable_whole_lots(2410.0, 2_410_000)


def test_cheap_etfs_are_whole_lots_at_the_same_capital():
    """同樣 13.3 萬，9.82 元的標的買得起 13 張"""
    assert affordable_whole_lots(9.82, 133_333)


def test_affordable_rejects_non_positive_price():
    with pytest.raises(SlippageError, match="價格"):
        affordable_whole_lots(0.0, 100_000)
