#!/usr/bin/env python3
"""
成本模型單元測試（D3 / D4）

每個預期值都在註解裡手算，可自行驗算，不是拿程式輸出反填。

執行：
    .venv/bin/python -m pytest tests/test_costs.py -v

單獨執行並產出對照表：
    PYTHONPATH=. .venv/bin/python tests/test_costs.py
"""

from __future__ import annotations

import pytest

from config.costs import (
    DEFAULT,
    FEE_DISCOUNT_DEFAULT,
    FEE_RATE,
    GROSS,
    MIN_FEE,
    NO_DISCOUNT,
    SLIPPAGE,
    TAX_RATE,
    CostModel,
    Tier,
    annual_cost_drag,
    sharpe_after_cost,
)

TOL = 1e-9


# ══════════════════════════════════════════════════════════════
# 常數：必須等於法規/券商實際數字
# ══════════════════════════════════════════════════════════════


def test_fee_rate_is_statutory_cap() -> None:
    """手續費法定上限 0.1425%"""
    assert FEE_RATE == 0.001425


def test_tax_rate_is_statutory() -> None:
    """證交稅 0.3%"""
    assert TAX_RATE == 0.003


def test_min_fee_is_20() -> None:
    """最低手續費 20 元。這是附件 quant_tw_pipeline.py 漏掉的一條"""
    assert MIN_FEE == 20.0


def test_default_discount_is_6_fold() -> None:
    assert FEE_DISCOUNT_DEFAULT == 0.6


def test_odd_lot_slippage_is_not_round_lot_value() -> None:
    """
    零股滑價必須明顯高於整張常見的 0.1%。

    40 萬資金買不起一張台積電（243 萬），只能做零股，
    零股價差比整張寬，沿用 0.1% 會低估成本。
    """
    assert SLIPPAGE[Tier.LARGE] == 0.003
    assert SLIPPAGE[Tier.MID] == 0.004
    assert SLIPPAGE[Tier.MID] > SLIPPAGE[Tier.LARGE], "中型股滑價應高於大型股"


# ══════════════════════════════════════════════════════════════
# 手續費：含最低收費
# ══════════════════════════════════════════════════════════════


def test_commission_above_min_fee() -> None:
    """
    手算：100,000 × 0.001425 × 0.6 = 85.5 元
    85.5 > 20，取 85.5
    """
    assert DEFAULT.commission(100_000) == pytest.approx(85.5, abs=TOL)


def test_commission_hits_min_fee_floor() -> None:
    """
    手算：10,000 × 0.001425 × 0.6 = 8.55 元
    8.55 < 20，取 20（最低收費生效）
    """
    assert DEFAULT.commission(10_000) == pytest.approx(20.0, abs=TOL)


def test_min_fee_threshold_is_23391() -> None:
    """
    最低手續費生效門檻 = 20 / (0.001425 × 0.6)
                       = 20 / 0.000855
                       = 23,391.81... 元

    低於這個金額的交易，實際費率高於 0.0855%。
    以 40 萬資金做零股三檔，這種小額單會常出現。
    """
    threshold = DEFAULT.min_amount_above_min_fee()
    assert threshold == pytest.approx(20 / 0.000855, abs=1e-6)
    assert threshold == pytest.approx(23_391.8128, abs=1e-3)

    # 門檻附近的行為：略低 → 收 20；略高 → 按比例
    assert DEFAULT.commission(threshold - 100) == pytest.approx(20.0, abs=TOL)
    assert DEFAULT.commission(threshold + 100) > 20.0


def test_commission_is_sign_agnostic() -> None:
    """負金額（賣出/空單表示法）取絕對值"""
    assert DEFAULT.commission(-100_000) == DEFAULT.commission(100_000)


def test_no_discount_commission_doubles_roughly() -> None:
    """
    手算：100,000 × 0.001425 × 1.0 = 142.5 元
    """
    assert NO_DISCOUNT.commission(100_000) == pytest.approx(142.5, abs=TOL)


# ══════════════════════════════════════════════════════════════
# 買賣單邊總成本
# ══════════════════════════════════════════════════════════════


def test_buy_cost_large_tier() -> None:
    """
    買進 100,000 元，0050 分層：
        手續費 100,000 × 0.001425 × 0.6 = 85.5
        滑價   100,000 × 0.003           = 300.0
        合計                              = 385.5
    買進「不」收證交稅。
    """
    assert DEFAULT.buy_cost(100_000, Tier.LARGE) == pytest.approx(385.5, abs=TOL)


def test_sell_cost_large_tier() -> None:
    """
    賣出 100,000 元，0050 分層：
        手續費 100,000 × 0.001425 × 0.6 = 85.5
        證交稅 100,000 × 0.003           = 300.0
        滑價   100,000 × 0.003           = 300.0
        合計                              = 685.5
    """
    assert DEFAULT.sell_cost(100_000, Tier.LARGE) == pytest.approx(685.5, abs=TOL)


def test_sell_cost_exceeds_buy_cost_by_tax() -> None:
    """賣買價差恰為證交稅"""
    amount = 100_000
    diff = DEFAULT.sell_cost(amount) - DEFAULT.buy_cost(amount)
    assert diff == pytest.approx(amount * TAX_RATE, abs=TOL)


def test_mid_tier_costs_more_than_large() -> None:
    """
    中型股滑價 0.4% vs 大型股 0.3%，差 0.1%：
        100,000 × 0.001 = 100 元
    """
    diff = DEFAULT.buy_cost(100_000, Tier.MID) - DEFAULT.buy_cost(100_000, Tier.LARGE)
    assert diff == pytest.approx(100.0, abs=TOL)


def test_trade_cost_dispatches_correctly() -> None:
    """trade_cost 介面與 scripts/evaluate_models.py 的 calc_trade_cost 對齊"""
    assert DEFAULT.trade_cost(100_000, is_sell=False) == DEFAULT.buy_cost(100_000)
    assert DEFAULT.trade_cost(100_000, is_sell=True) == DEFAULT.sell_cost(100_000)


# ══════════════════════════════════════════════════════════════
# 成本率（回測用）
# ══════════════════════════════════════════════════════════════


def test_round_trip_rate_default() -> None:
    """
    一趟來回成本率（6 折、0050、忽略最低手續費）：
        手續費 0.001425 × 0.6 × 2 = 0.00171
        證交稅                     = 0.003
        滑價   0.003 × 2           = 0.006
        合計                       = 0.01071  →  1.071%
    """
    assert DEFAULT.round_trip_rate(Tier.LARGE) == pytest.approx(0.01071, abs=TOL)


def test_round_trip_rate_mid_tier() -> None:
    """
    中型股：0.00171 + 0.003 + 0.004 × 2 = 0.01271  →  1.271%
    """
    assert DEFAULT.round_trip_rate(Tier.MID) == pytest.approx(0.01271, abs=TOL)


def test_round_trip_rate_no_discount() -> None:
    """
    無折扣：0.001425 × 2 + 0.003 + 0.006 = 0.01185  →  1.185%
    """
    assert NO_DISCOUNT.round_trip_rate(Tier.LARGE) == pytest.approx(0.01185, abs=TOL)


def test_original_project_model_round_trip() -> None:
    """
    重現 qlib-tw-trader 原本 scripts/evaluate_models.py 的成本率
    （無滑價、無折扣以取上界）：
        0.001425 × 2 + 0.003 = 0.00585  →  0.585%

    與我們含滑價的 1.071% 相差 0.486 個百分點。
    高換手策略下這個差距會被放大數十倍。
    """
    assert GROSS.round_trip_rate(Tier.LARGE) == pytest.approx(0.00585, abs=TOL)
    assert DEFAULT.round_trip_rate(Tier.LARGE) - GROSS.round_trip_rate(
        Tier.LARGE
    ) == pytest.approx(0.00486, abs=TOL)


def test_one_way_rate_buy_has_no_tax() -> None:
    """
    買進單邊：0.001425 × 0.6 + 0.003 = 0.003855
    """
    assert DEFAULT.one_way_rate(is_sell=False) == pytest.approx(0.003855, abs=TOL)


def test_one_way_rate_sell_includes_tax() -> None:
    """
    賣出單邊：0.001425 × 0.6 + 0.003 + 0.003 = 0.006855
    """
    assert DEFAULT.one_way_rate(is_sell=True) == pytest.approx(0.006855, abs=TOL)


def test_one_way_rates_sum_to_round_trip() -> None:
    """單邊相加必須等於來回"""
    total = DEFAULT.one_way_rate(False) + DEFAULT.one_way_rate(True)
    assert total == pytest.approx(DEFAULT.round_trip_rate(), abs=TOL)


# ══════════════════════════════════════════════════════════════
# 換手率 → 年化拖累
# ══════════════════════════════════════════════════════════════


def test_annual_drag_readme_holddrop_turnover() -> None:
    """
    README 最佳策略 HoldDrop(K=10,H=3,D=1) 週換手率 9.9%。

    手算：0.099 × 52 = 5.148 趟來回/年
          5.148 × 0.01071
            = 5.148 × 0.01    = 0.05148
            + 5.148 × 0.00071 = 0.00365508
            = 0.05513508                    →  年化吃掉約 5.51%
    """
    drag = annual_cost_drag(0.099, DEFAULT, Tier.LARGE)
    assert drag == pytest.approx(0.099 * 52 * 0.01071, abs=TOL)
    assert drag == pytest.approx(0.05513508, abs=1e-8)


def test_annual_drag_scales_linearly_with_turnover() -> None:
    """換手率翻倍，拖累翻倍"""
    assert annual_cost_drag(0.20) == pytest.approx(2 * annual_cost_drag(0.10), abs=TOL)


def test_annual_drag_zero_when_no_turnover() -> None:
    """不交易就沒有交易成本"""
    assert annual_cost_drag(0.0) == 0.0


def test_high_turnover_drag_is_catastrophic() -> None:
    """
    API 回測走日度調倉。假設每日換掉 30% 部位：
        週換手率 = 0.30 × 5 = 1.5（150%）
        年化來回 = 1.5 × 52 = 78 趟
        拖累     = 78 × 0.01071 = 0.83538  →  年化吃掉 83.5%

    這說明為什麼「回測不扣成本」不是小數點問題——
    日度調倉在台股的成本結構下根本不可執行。
    """
    drag = annual_cost_drag(1.5, DEFAULT, Tier.LARGE)
    assert drag == pytest.approx(0.83538, abs=1e-5)
    assert drag > 0.5, "高換手拖累應超過 50%"


# ══════════════════════════════════════════════════════════════
# Sharpe 影響推算
# ══════════════════════════════════════════════════════════════


def test_sharpe_after_cost_readme_numbers() -> None:
    """
    對 README 宣稱的最佳策略做成本敏感度：
        年化報酬 55.1%、Sharpe 1.724、週換手 9.9%

    手算：
        隱含波動度 = 0.551 / 1.724 = 0.31960557
        拖累       = 0.099 × 52 × 0.01071 = 0.05513508
        淨報酬     = 0.551 - 0.05513508    = 0.49586492
        淨 Sharpe  = 0.49586492 / 0.31960557 = 1.5515

    註：README 的 1.724 已含手續費與證交稅但「不含滑價」，
        所以這裡算出的差額主要來自滑價與我們較保守的分層設定。
    """
    drag, net_return, net_sharpe = sharpe_after_cost(
        gross_annual_return=0.551,
        gross_sharpe=1.724,
        weekly_turnover=0.099,
        cost=DEFAULT,
        tier=Tier.LARGE,
    )
    assert drag == pytest.approx(0.05513508, abs=1e-8)
    assert net_return == pytest.approx(0.49586492, abs=1e-8)

    implied_vol = 0.551 / 1.724
    assert net_sharpe == pytest.approx(0.49586492 / implied_vol, abs=1e-8)
    assert net_sharpe == pytest.approx(1.55149, abs=1e-5)
    assert 1.5 < net_sharpe < 1.6, f"淨 Sharpe 落點異常：{net_sharpe}"


def test_sharpe_after_cost_rejects_nonpositive_sharpe() -> None:
    """毛 Sharpe 非正時無法反推波動度，必須拒絕"""
    with pytest.raises(ValueError, match="gross_sharpe"):
        sharpe_after_cost(0.5, 0.0, 0.1)


# ══════════════════════════════════════════════════════════════
# 模型不變性與參數驗證
# ══════════════════════════════════════════════════════════════


def test_cost_model_is_frozen() -> None:
    """成本模型不可變（避免回測中途被改掉而無人察覺）"""
    with pytest.raises(Exception):
        DEFAULT.fee_discount = 0.5  # type: ignore[misc]


def test_invalid_discount_rejected() -> None:
    """折扣必須落在 (0, 1]"""
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="fee_discount"):
            CostModel(fee_discount=bad)


def test_toggles_reduce_cost_monotonically() -> None:
    """關掉滑價/最低收費只能讓成本變小，不能變大"""
    amount = 5_000  # 小額，會觸發最低手續費
    full = CostModel()
    no_slip = CostModel(apply_slippage=False)
    no_min = CostModel(apply_min_fee=False)

    assert no_slip.buy_cost(amount) < full.buy_cost(amount)
    assert no_min.buy_cost(amount) < full.buy_cost(amount)


# ══════════════════════════════════════════════════════════════
# 獨立執行：對照表
# ══════════════════════════════════════════════════════════════


def build_report() -> str:
    lines: list[str] = []
    add = lines.append

    add("=" * 84)
    add("台股成本模型對照表 — config/costs.py")
    add("=" * 84)
    add("")
    add(f"FEE_RATE = {FEE_RATE}   TAX_RATE = {TAX_RATE}   MIN_FEE = {MIN_FEE}")
    add(f"SLIPPAGE = {{0050: {SLIPPAGE[Tier.LARGE]}, 0051: {SLIPPAGE[Tier.MID]}}}")
    add("")

    add("─" * 84)
    add("一趟來回成本率")
    add("─" * 84)
    add(f"{'模型':<26}{'0050':>12}{'0051':>12}")
    rows = [
        ("6折 + 滑價（本專案預設）", DEFAULT),
        ("無折扣 + 滑價（敏感度）", NO_DISCOUNT),
        ("原專案模型（無滑價）", GROSS),
    ]
    for label, model in rows:
        add(
            f"{label:<26}{model.round_trip_rate(Tier.LARGE) * 100:>11.3f}%"
            f"{model.round_trip_rate(Tier.MID) * 100:>11.3f}%"
        )
    add("")
    gap = DEFAULT.round_trip_rate() - GROSS.round_trip_rate()
    add(f"預設與原專案模型的差距：{gap * 100:.3f} 個百分點／趟")
    add("")

    add("─" * 84)
    add("最低手續費影響")
    add("─" * 84)
    threshold = DEFAULT.min_amount_above_min_fee()
    add(f"門檻金額：{threshold:,.0f} 元（低於此金額，實際費率高於名目 0.0855%）")
    for amt in (5_000, 10_000, 20_000, 23_392, 50_000, 117_600):
        fee = DEFAULT.commission(amt)
        add(
            f"  部位 {amt:>9,} 元 → 手續費 {fee:>8.2f} 元 "
            f"（實際費率 {fee / amt * 100:.4f}%）"
        )
    add("")

    add("─" * 84)
    add("換手率 → 年化成本拖累（0050 分層、6 折、含滑價）")
    add("─" * 84)
    add(f"{'週換手率':<14}{'年化來回次數':>16}{'年化拖累':>14}")
    for wt, note in [
        (0.099, "README HoldDrop(K=10,H=3,D=1)"),
        (0.20, ""),
        (0.50, ""),
        (1.00, ""),
        (1.50, "日度調倉（假設每日換 30%）"),
    ]:
        drag = annual_cost_drag(wt, DEFAULT, Tier.LARGE)
        add(f"{wt * 100:>10.1f}%{wt * 52:>16.2f}{drag * 100:>13.2f}%   {note}")
    add("")

    add("─" * 84)
    add("對 README 宣稱績效的成本敏感度")
    add("─" * 84)
    add("README：HoldDrop(K=10,H=3,D=1) 年化 55.1% / Sharpe 1.724 / 週換手 9.9%")
    add("        （該數字已含手續費與證交稅，但不含滑價）")
    add("")
    add(f"{'情境':<28}{'年化拖累':>10}{'淨年化':>10}{'淨 Sharpe':>12}")
    for label, model, tier in [
        ("6折 + 0.3% 滑價", DEFAULT, Tier.LARGE),
        ("6折 + 0.4% 滑價（中型股）", DEFAULT, Tier.MID),
        ("無折扣 + 0.3% 滑價", NO_DISCOUNT, Tier.LARGE),
    ]:
        drag, net_r, net_s = sharpe_after_cost(0.551, 1.724, 0.099, model, tier)
        add(
            f"{label:<28}{drag * 100:>9.2f}%{net_r * 100:>9.2f}%{net_s:>12.3f}"
        )
    add("")
    add("=" * 84)
    add("這張表的邊界（誠實聲明）：")
    add("  · 以上是「解析估算」，不是重跑回測。假設成本只降低報酬、不改變波動度。")
    add("  · 真正的實測需要 162 個模型完整重訓 + 回測，見")
    add("    docs/需求規劃/202609/驗證/ 的成本接線實測。")
    add("  · README 的週換手 9.9% 是它自己報的數字，我未獨立驗證。")
    add("=" * 84)

    return "\n".join(lines)


if __name__ == "__main__":
    print(build_report())
