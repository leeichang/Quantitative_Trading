#!/usr/bin/env python3
"""
成本模型測試（D3 / D4）

每個預期值都手算在註解裡。**不可拿程式輸出反填**
（CLAUDE.md 禁令 10 / AGENTS.md 檢查項 24）。

實測背景：qlib-tw-trader 驗證顯示「含滑價 vs 不含滑價」在 20 週回測上
差了 19.5 個百分點（+0.41% vs −19.08%）。成本不是小數點問題。
"""

from __future__ import annotations

import pytest

from taiwan_quant.config.costs import (
    DEFAULT,
    FEE_DISCOUNT_DEFAULT,
    FEE_RATE,
    FEE_TAX_ONLY,
    GROSS,
    MIN_FEE,
    NO_DISCOUNT,
    SENSITIVITY_SET,
    SLIPPAGE,
    TAX_RATE,
    CostModel,
    Tier,
    annual_cost_drag,
    sharpe_after_cost,
)

pytestmark = pytest.mark.unit

TOL = 1e-9


# ══════════════════════════════════════════════════════════════
# 常數：必須等於法規／券商實際數字
# ══════════════════════════════════════════════════════════════


def test_statutory_constants() -> None:
    """手續費法定上限 0.1425%、證交稅 0.3%、最低手續費 20 元"""
    assert FEE_RATE == 0.001425
    assert TAX_RATE == 0.003
    assert MIN_FEE == 20.0
    assert FEE_DISCOUNT_DEFAULT == 0.6


def test_odd_lot_slippage_is_not_round_lot_value() -> None:
    """
    零股滑價必須明顯高於整張常見的 0.1%（D4）。

    40 萬資金買不起一張台積電（2,430 × 1,000 = 243 萬），只能做零股，
    零股價差比整張寬，沿用 0.1% 會低估成本。
    """
    assert SLIPPAGE[Tier.LARGE] == 0.003
    assert SLIPPAGE[Tier.MID] == 0.004
    assert SLIPPAGE[Tier.MID] > SLIPPAGE[Tier.LARGE], "中型股滑價應高於大型股"


# ══════════════════════════════════════════════════════════════
# 手續費與最低收費
# ══════════════════════════════════════════════════════════════


def test_commission_above_min_fee() -> None:
    """手算：100,000 × 0.001425 × 0.6 = 85.5 元（> 20，取 85.5）"""
    assert DEFAULT.commission(100_000) == pytest.approx(85.5, abs=TOL)


def test_commission_hits_min_fee_floor() -> None:
    """手算：10,000 × 0.001425 × 0.6 = 8.55 元（< 20，取 20）"""
    assert DEFAULT.commission(10_000) == pytest.approx(20.0, abs=TOL)


def test_min_fee_threshold() -> None:
    """
    門檻 = 20 / (0.001425 × 0.6)
         = 20 / 0.000855
         = 23,391.8128... 元

    低於此金額的交易實際費率高於 0.0855%。以 40 萬做零股三檔，常出現。
    """
    threshold = DEFAULT.min_amount_above_min_fee()
    assert threshold == pytest.approx(23_391.8128, abs=1e-3)
    assert DEFAULT.commission(threshold - 100) == pytest.approx(20.0, abs=TOL)
    assert DEFAULT.commission(threshold + 100) > 20.0


def test_commission_is_sign_agnostic() -> None:
    """負金額（賣出表示法）取絕對值"""
    assert DEFAULT.commission(-100_000) == DEFAULT.commission(100_000)


def test_no_discount_commission() -> None:
    """手算：100,000 × 0.001425 × 1.0 = 142.5 元"""
    assert NO_DISCOUNT.commission(100_000) == pytest.approx(142.5, abs=TOL)


# ══════════════════════════════════════════════════════════════
# 單邊總成本
# ══════════════════════════════════════════════════════════════


def test_buy_cost_large_tier() -> None:
    """
    買進 100,000 元、0050 分層：
        手續費 100,000 × 0.001425 × 0.6 =  85.5
        滑價   100,000 × 0.003           = 300.0
        合計                              = 385.5
    買進不收證交稅。
    """
    assert DEFAULT.buy_cost(100_000, Tier.LARGE) == pytest.approx(385.5, abs=TOL)


def test_sell_cost_large_tier() -> None:
    """
    賣出 100,000 元、0050 分層：
        手續費 85.5 + 證交稅 300.0 + 滑價 300.0 = 685.5
    """
    assert DEFAULT.sell_cost(100_000, Tier.LARGE) == pytest.approx(685.5, abs=TOL)


def test_sell_minus_buy_equals_tax() -> None:
    """賣買價差恰為證交稅"""
    amount = 100_000
    diff = DEFAULT.sell_cost(amount) - DEFAULT.buy_cost(amount)
    assert diff == pytest.approx(amount * TAX_RATE, abs=TOL)


def test_round_trip_cost_absolute() -> None:
    """手算：385.5 + 685.5 = 1,071.0 元（佔 100,000 為 1.071%）"""
    assert DEFAULT.round_trip_cost(100_000) == pytest.approx(1071.0, abs=TOL)


def test_mid_tier_costs_more() -> None:
    """中型股滑價 0.4% vs 0.3%，買進差 100,000 × 0.001 = 100 元"""
    diff = DEFAULT.buy_cost(100_000, Tier.MID) - DEFAULT.buy_cost(100_000, Tier.LARGE)
    assert diff == pytest.approx(100.0, abs=TOL)


# ══════════════════════════════════════════════════════════════
# 成本率
# ══════════════════════════════════════════════════════════════


def test_round_trip_rate_default() -> None:
    """
    6 折、0050、忽略最低手續費：
        手續費 0.001425 × 0.6 × 2 = 0.00171
        證交稅                     = 0.003
        滑價   0.003 × 2           = 0.006
        合計                       = 0.01071  →  1.071%
    """
    assert DEFAULT.round_trip_rate(Tier.LARGE) == pytest.approx(0.01071, abs=TOL)


def test_round_trip_rate_mid_tier() -> None:
    """0.00171 + 0.003 + 0.004 × 2 = 0.01271 → 1.271%"""
    assert DEFAULT.round_trip_rate(Tier.MID) == pytest.approx(0.01271, abs=TOL)


def test_round_trip_rate_no_discount() -> None:
    """0.001425 × 2 + 0.003 + 0.006 = 0.01185 → 1.185%"""
    assert NO_DISCOUNT.round_trip_rate(Tier.LARGE) == pytest.approx(0.01185, abs=TOL)


def test_fee_tax_only_reproduces_qlib_tw_trader_model() -> None:
    """
    重現 qlib-tw-trader 的成本模型（無滑價，取無折扣上界）：
        0.001425 × 2 + 0.003 = 0.00585 → 0.585%

    與含滑價的 1.071% 差 0.486 個百分點／趟。這 0.486 在 20 週回測上
    被放大成 19.5 個百分點的績效差距。
    """
    model = CostModel(fee_discount=1.0, apply_slippage=False)
    assert model.round_trip_rate(Tier.LARGE) == pytest.approx(0.00585, abs=TOL)
    gap = NO_DISCOUNT.round_trip_rate(Tier.LARGE) - model.round_trip_rate(Tier.LARGE)
    assert gap == pytest.approx(0.006, abs=TOL)


def test_gross_model_has_zero_cost() -> None:
    """毛報酬模型完全不收費"""
    assert GROSS.round_trip_rate(Tier.LARGE) == pytest.approx(0.0, abs=TOL)
    assert GROSS.buy_cost(100_000) == pytest.approx(0.0, abs=TOL)
    assert GROSS.sell_cost(100_000) == pytest.approx(0.0, abs=TOL)


def test_one_way_rates() -> None:
    """
    買進單邊：0.001425 × 0.6 + 0.003           = 0.003855
    賣出單邊：0.001425 × 0.6 + 0.003 + 0.003   = 0.006855
    兩者相加必須等於來回 0.01071
    """
    assert DEFAULT.one_way_rate(is_sell=False) == pytest.approx(0.003855, abs=TOL)
    assert DEFAULT.one_way_rate(is_sell=True) == pytest.approx(0.006855, abs=TOL)
    total = DEFAULT.one_way_rate(False) + DEFAULT.one_way_rate(True)
    assert total == pytest.approx(DEFAULT.round_trip_rate(), abs=TOL)


def test_fee_tax_only_excludes_slippage_only() -> None:
    """FEE_TAX_ONLY 保留手續費與稅，只關掉滑價"""
    assert FEE_TAX_ONLY.commission(100_000) == pytest.approx(85.5, abs=TOL)
    assert FEE_TAX_ONLY.slippage(100_000) == pytest.approx(0.0, abs=TOL)
    assert FEE_TAX_ONLY.tax(100_000) == pytest.approx(300.0, abs=TOL)


# ══════════════════════════════════════════════════════════════
# 換手率 → 年化拖累（CLAUDE.md 規格 13）
# ══════════════════════════════════════════════════════════════


def test_annual_drag_at_holddrop_turnover() -> None:
    """
    qlib-tw-trader README 的 HoldDrop(K=10,H=3,D=1) 週換手率 9.9%。

    手算：0.099 × 52 = 5.148 趟/年
          5.148 × 0.01071
            = 5.148 × 0.01    = 0.05148
            + 5.148 × 0.00071 = 0.00365508
            = 0.05513508                     →  年化 5.51%
    """
    drag = annual_cost_drag(0.099, DEFAULT, Tier.LARGE)
    assert drag == pytest.approx(0.05513508, abs=1e-8)


def test_annual_drag_at_measured_daily_rebalance_turnover() -> None:
    """
    實測 qlib-tw-trader 日度調倉的週換手率為 271.5%。

    手算：2.715 × 52 = 141.18 趟/年
          141.18 × 0.01071 = 1.51203...      →  年化 151.2%

    這個數字說明日度調倉在台股成本結構下不可執行。
    """
    drag = annual_cost_drag(2.715, DEFAULT, Tier.LARGE)
    assert drag == pytest.approx(2.715 * 52 * 0.01071, abs=TOL)
    assert drag > 1.0, "年化拖累應超過 100%"


def test_annual_drag_scales_linearly() -> None:
    """換手率翻倍，拖累翻倍"""
    assert annual_cost_drag(0.20) == pytest.approx(2 * annual_cost_drag(0.10), abs=TOL)


def test_annual_drag_zero_when_no_turnover() -> None:
    assert annual_cost_drag(0.0) == 0.0


def test_annual_drag_rejects_negative_turnover() -> None:
    with pytest.raises(ValueError, match="weekly_turnover"):
        annual_cost_drag(-0.1)


# ══════════════════════════════════════════════════════════════
# Sharpe 解析估算
# ══════════════════════════════════════════════════════════════


def test_sharpe_after_cost() -> None:
    """
    對 qlib-tw-trader README 宣稱的績效做成本敏感度：
        年化 55.1%、Sharpe 1.724、週換手 9.9%

    手算：
        隱含波動度 = 0.551 / 1.724   = 0.31960557
        拖累       = 0.05513508
        淨報酬     = 0.551 − 0.05513508 = 0.49586492
        淨 Sharpe  = 0.49586492 / 0.31960557 = 1.55149
    """
    drag, net_return, net_sharpe = sharpe_after_cost(0.551, 1.724, 0.099, DEFAULT, Tier.LARGE)
    assert drag == pytest.approx(0.05513508, abs=1e-8)
    assert net_return == pytest.approx(0.49586492, abs=1e-8)
    assert net_sharpe == pytest.approx(1.55149, abs=1e-5)


def test_sharpe_after_cost_rejects_nonpositive_sharpe() -> None:
    """毛 Sharpe 非正時無法反推波動度"""
    with pytest.raises(ValueError, match="gross_sharpe"):
        sharpe_after_cost(0.5, 0.0, 0.1)


# ══════════════════════════════════════════════════════════════
# 模型不變性與參數驗證
# ══════════════════════════════════════════════════════════════


def test_cost_model_is_frozen() -> None:
    """不可變，避免回測中途被改掉而無人察覺"""
    with pytest.raises(Exception):
        DEFAULT.fee_discount = 0.5  # type: ignore[misc]


def test_invalid_discount_rejected() -> None:
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="fee_discount"):
            CostModel(fee_discount=bad)


def test_toggles_reduce_cost_monotonically() -> None:
    """關掉滑價/最低收費只能讓成本變小"""
    amount = 5_000  # 小額，會觸發最低手續費
    assert CostModel(apply_slippage=False).buy_cost(amount) < DEFAULT.buy_cost(amount)
    assert CostModel(apply_min_fee=False).buy_cost(amount) < DEFAULT.buy_cost(amount)


def test_sensitivity_set_covers_required_scenarios() -> None:
    """
    CLAUDE.md 規格 15 要求回測報告至少並列三檔成本情境。
    敏感度組合必須涵蓋：無成本、含手續費稅、含滑價。
    """
    assert len(SENSITIVITY_SET) >= 3
    rates = [m.round_trip_rate(Tier.LARGE) for m in SENSITIVITY_SET.values()]
    assert min(rates) == pytest.approx(0.0, abs=TOL), "必須有一檔毛報酬對照"
    assert max(rates) > 0.01, "必須有一檔含滑價的完整成本"
    # 情境之間必須真的不同，否則並列沒有意義
    assert len(set(round(r, 8) for r in rates)) == len(rates)
