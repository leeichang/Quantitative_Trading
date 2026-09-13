#!/usr/bin/env python3
"""
柵欄寬度推導測試

CLAUDE.md 核心建模方式明確禁止人工寫死 target_pct / stop_pct：

    `target_pct` / `stop_pct` 不可人工寫死，須由 ATR 分位數與歷史相似
    型態報酬分布推導

理由（來自 ChatGPT 來源的意見，見 來源原文/02_ChatGPT）：
    不要人工設定 Buy = Current Price × 0.96，而是讓模型研究
    「歷史上這種型態，什麼價格位置進場最有效？」

做法：
    stop_pct   = atr_multiple × ATR(n) / close      波動度決定停損距離
    target_pct = 未來 horizon 日報酬分布的指定分位數（歷史經驗決定目標）
    並強制 R:R >= 門檻，不足則放棄該筆（不硬拉目標價湊 R:R）

每個預期值都手算在註解裡。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from taiwan_quant.labeling.barrier_width import (
    BarrierWidth,
    atr,
    derive_width,
)

pytestmark = pytest.mark.unit


def make_bars(rows: list[tuple[str, float, float, float, float]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close"])
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()


def flat_bars(n: int, price: float = 100.0, span: float = 2.0) -> pd.DataFrame:
    """產生固定振幅的日 K：每天 high = price + span/2、low = price − span/2"""
    rows = [
        (
            f"2026-{1 + i // 28:02d}-{1 + i % 28:02d}",
            price,
            price + span / 2,
            price - span / 2,
            price,
        )
        for i in range(n)
    ]
    return make_bars(rows)


# ══════════════════════════════════════════════════════════════
# ATR
# ══════════════════════════════════════════════════════════════


def test_atr_on_constant_range() -> None:
    """
    固定振幅、無跳空的資料，ATR 應等於當日振幅。

    手算：high = 101、low = 99、前收 = 100
        True Range = max(high−low, |high−prev_close|, |low−prev_close|)
                   = max(2, 1, 1) = 2
        ATR(14) = 2 的平均 = 2.0
    """
    bars = flat_bars(30, price=100.0, span=2.0)
    result = atr(bars, period=14)
    assert result.iloc[-1] == pytest.approx(2.0)


def test_atr_first_period_is_nan() -> None:
    """
    前 period 根資料不足，ATR 必須為 NaN，不可用不完整視窗硬算。

    period = 14 → 前 14 筆（index 0..13）為 NaN，第 15 筆（index 14）才有值。
    第 1 筆另外因為沒有前收，True Range 本身也不可用。
    """
    bars = flat_bars(20, price=100.0, span=2.0)
    result = atr(bars, period=14)
    assert result.iloc[:14].isna().all()
    assert not np.isnan(result.iloc[14])


def test_atr_accounts_for_gaps() -> None:
    """
    跳空必須反映在 True Range 上，否則會低估波動度。

    D1  close 100
    D2  high 105、low 104（整根在前收之上，跳空開高）
        True Range = max(105−104, |105−100|, |104−100|) = max(1, 5, 4) = 5
        ← 若只用 high−low 會得到 1，嚴重低估
    """
    bars = make_bars([
        ("2026-01-05", 100.0, 100.0, 100.0, 100.0),
        ("2026-01-06", 104.0, 105.0, 104.0, 104.5),
    ])
    result = atr(bars, period=1)
    assert result.iloc[1] == pytest.approx(5.0)


def test_atr_rejects_bad_period() -> None:
    bars = flat_bars(30)
    with pytest.raises(ValueError, match="period"):
        atr(bars, period=0)


def test_atr_validates_columns() -> None:
    bad = pd.DataFrame({"close": [1.0, 2.0]}, index=pd.to_datetime(["2026-01-05", "2026-01-06"]))
    with pytest.raises(ValueError, match="缺少必要欄位"):
        atr(bad, period=14)


# ══════════════════════════════════════════════════════════════
# derive_width：停損由 ATR 決定
# ══════════════════════════════════════════════════════════════


def test_stop_pct_from_atr_multiple() -> None:
    """
    手算：ATR = 2.0、close = 100.0、atr_multiple = 1.5
        stop_pct = 1.5 × 2.0 / 100.0 = 0.03（3%）
    """
    bars = flat_bars(60, price=100.0, span=2.0)
    width = derive_width(
        bars,
        decision_idx=len(bars) - 1,
        horizon=5,
        atr_period=14,
        atr_multiple=1.5,
        target_quantile=0.7,
        min_risk_reward=0.0,   # 本測試只驗 stop，關掉 R:R 門檻
    )
    assert width is not None
    assert width.stop_pct == pytest.approx(0.03)


def test_stop_pct_scales_with_volatility() -> None:
    """
    波動度加倍，停損距離加倍。

    span 2.0 → ATR 2.0 → stop 3%
    span 4.0 → ATR 4.0 → stop 6%
    """
    low_vol = derive_width(
        flat_bars(60, price=100.0, span=2.0),
        decision_idx=59, horizon=5, atr_multiple=1.5,
        target_quantile=0.7, min_risk_reward=0.0,
    )
    high_vol = derive_width(
        flat_bars(60, price=100.0, span=4.0),
        decision_idx=59, horizon=5, atr_multiple=1.5,
        target_quantile=0.7, min_risk_reward=0.0,
    )
    assert low_vol.stop_pct == pytest.approx(0.03)
    assert high_vol.stop_pct == pytest.approx(0.06)


def test_stop_pct_respects_floor_and_cap() -> None:
    """
    停損距離要有上下限，避免極端波動下算出荒謬的值。

    · 極低波動（span 0.02 → ATR 0.02 → 1.5×0.02/100 = 0.0003）
      應被 floor 拉到 min_stop_pct
    · 極高波動（span 40 → ATR 40 → 1.5×40/100 = 0.6）
      應被 cap 壓到 max_stop_pct
    """
    tiny = derive_width(
        flat_bars(60, price=100.0, span=0.02),
        decision_idx=59, horizon=5, atr_multiple=1.5,
        target_quantile=0.7, min_risk_reward=0.0,
        min_stop_pct=0.015, max_stop_pct=0.10,
    )
    huge = derive_width(
        flat_bars(60, price=100.0, span=40.0),
        decision_idx=59, horizon=5, atr_multiple=1.5,
        target_quantile=0.7, min_risk_reward=0.0,
        min_stop_pct=0.015, max_stop_pct=0.10,
    )
    assert tiny.stop_pct == pytest.approx(0.015)
    assert huge.stop_pct == pytest.approx(0.10)


# ══════════════════════════════════════════════════════════════
# derive_width：目標由歷史報酬分布決定
# ══════════════════════════════════════════════════════════════


def test_target_pct_from_historical_return_quantile() -> None:
    """
    目標價取「歷史上未來 horizon 日報酬」的指定分位數。

    構造：價格每天固定漲 1%，horizon = 5
        每個 5 日報酬 = 1.01^5 − 1 = 0.0510100501...
        分布退化為單一值，任何分位數都等於它
    預期 target_pct ≈ 0.05101
    """
    price = 100.0
    rows = []
    for i in range(80):
        rows.append((f"2026-{1 + i // 28:02d}-{1 + i % 28:02d}", price, price * 1.001, price * 0.999, price))
        price *= 1.01
    bars = make_bars(rows)

    width = derive_width(
        bars,
        decision_idx=len(bars) - 1,
        horizon=5,
        atr_period=14,
        atr_multiple=1.5,
        target_quantile=0.7,
        min_risk_reward=0.0,
        lookback=60,
    )
    assert width.target_pct == pytest.approx(1.01**5 - 1, rel=1e-3)


def test_target_uses_only_past_returns() -> None:
    """
    反 look-ahead（CLAUDE.md 禁令 1）：目標分位數只能用決策日之前
    **已實現**的 horizon 報酬。

    構造：前 70 天平盤（5 日報酬 ≈ 0），第 70 天之後暴漲。
    在 decision_idx = 69 推導時，不可看到後面的暴漲。
        → target_pct 應接近 0（會被 min_target_pct 拉起來），
          而不是反映暴漲的大報酬
    """
    rows = []
    price = 100.0
    for i in range(70):
        rows.append((f"2026-{1 + i // 28:02d}-{1 + i % 28:02d}", price, price * 1.001, price * 0.999, price))
    for i in range(70, 100):
        price *= 1.10
        rows.append((f"2026-{1 + i // 28:02d}-{1 + i % 28:02d}", price, price * 1.001, price * 0.999, price))
    bars = make_bars(rows)

    width = derive_width(
        bars,
        decision_idx=69,
        horizon=5,
        atr_period=14,
        atr_multiple=1.5,
        target_quantile=0.9,
        min_risk_reward=0.0,
        lookback=60,
        min_target_pct=0.02,
    )
    assert width.target_pct == pytest.approx(0.02), (
        f"target_pct = {width.target_pct}，疑似看到了決策日之後的暴漲"
    )


def test_returns_none_when_history_insufficient() -> None:
    """歷史不足以算 ATR 或報酬分位數 → 回傳 None，不猜測"""
    bars = flat_bars(10, price=100.0, span=2.0)
    assert derive_width(bars, decision_idx=9, horizon=5, atr_period=14) is None


# ══════════════════════════════════════════════════════════════
# R:R 門檻
# ══════════════════════════════════════════════════════════════


def test_rejects_when_risk_reward_below_threshold() -> None:
    """
    R:R 不足門檻 → 回傳 None（放棄該筆），**不可硬拉目標價湊數**。

    構造：波動大（stop 寬）但歷史報酬小（target 窄）
        span 8.0 → ATR 8.0 → stop = 1.5 × 8 / 100 = 0.12
        平盤 → 5 日報酬分位數 ≈ 0 → target 被 floor 拉到 0.02
        R:R = 0.02 / 0.12 = 0.167 < 2.0  → 放棄
    """
    bars = flat_bars(80, price=100.0, span=8.0)
    width = derive_width(
        bars,
        decision_idx=79,
        horizon=5,
        atr_period=14,
        atr_multiple=1.5,
        target_quantile=0.7,
        min_risk_reward=2.0,
        min_target_pct=0.02,
        max_stop_pct=0.20,
    )
    assert width is None


def test_accepts_when_risk_reward_meets_threshold() -> None:
    """
    R:R 達標 → 回傳結果。

    構造：低波動 + 明顯上漲趨勢
        span 1.0 → ATR 1.0 → stop = 1.5 × 1 / 100 = 0.015
        每天漲 1% → 5 日報酬 ≈ 0.05101 → target ≈ 0.05101
        R:R = 0.05101 / 0.015 = 3.40 >= 2.0  → 接受
    """
    price = 100.0
    rows = []
    for i in range(80):
        rows.append((f"2026-{1 + i // 28:02d}-{1 + i % 28:02d}", price, price + 0.5, price - 0.5, price))
        price *= 1.01
    bars = make_bars(rows)

    width = derive_width(
        bars,
        decision_idx=79,
        horizon=5,
        atr_period=14,
        atr_multiple=1.5,
        target_quantile=0.7,
        min_risk_reward=2.0,
        lookback=60,
    )
    assert width is not None
    assert width.risk_reward >= 2.0
    assert width.risk_reward == pytest.approx(width.target_pct / width.stop_pct)


# ══════════════════════════════════════════════════════════════
# BarrierWidth 結構與不可變性
# ══════════════════════════════════════════════════════════════


def test_barrier_width_is_immutable() -> None:
    bars = flat_bars(60, price=100.0, span=2.0)
    width = derive_width(
        bars, decision_idx=59, horizon=5, atr_multiple=1.5,
        target_quantile=0.7, min_risk_reward=0.0,
    )
    with pytest.raises(Exception):
        width.stop_pct = 0.5  # type: ignore[misc]


def test_derive_width_does_not_mutate_input() -> None:
    """不可變性：推導不得改動傳入的日 K"""
    bars = flat_bars(60, price=100.0, span=2.0)
    before = bars.copy(deep=True)
    derive_width(
        bars, decision_idx=59, horizon=5, atr_multiple=1.5,
        target_quantile=0.7, min_risk_reward=0.0,
    )
    pd.testing.assert_frame_equal(bars, before)


def test_barrier_width_reports_inputs_for_audit() -> None:
    """
    結果要帶出推導依據（ATR 值、分位數、lookback），
    否則無法回答「為什麼當時停損設在這個距離」（禁令 7、8）。
    """
    bars = flat_bars(60, price=100.0, span=2.0)
    width = derive_width(
        bars, decision_idx=59, horizon=5, atr_period=14, atr_multiple=1.5,
        target_quantile=0.7, min_risk_reward=0.0, lookback=40,
    )
    assert width.atr_value == pytest.approx(2.0)
    assert width.atr_period == 14
    assert width.atr_multiple == pytest.approx(1.5)
    assert width.target_quantile == pytest.approx(0.7)
    assert width.lookback == 40
    assert width.sample_size > 0


# ══════════════════════════════════════════════════════════════
# NaN 防護
#
# 實測 bug（smoke_label.py 抓到）：2317 在 2025-07-30 有一列 OHLC 全為
# NULL（上游資料洞）。單一 NaN 汙染 np.quantile → target_pct = NaN →
# risk_reward = NaN → `NaN < min_risk_reward` 為 False → **靜默通過門檻**。
#
# 教訓：任何門檻比較前必須先排除非有限值。NaN 不會讓比較拋錯，
# 它會讓比較「看起來通過」。
# ══════════════════════════════════════════════════════════════


def test_nan_in_close_must_not_pass_risk_reward_gate() -> None:
    """
    價格序列含 NaN 時必須回傳 None，不可讓 NaN 的 R:R 混過門檻。

    構造：80 根 K，其中第 40 根 OHLC 全為 NaN（模擬上游資料洞）。
    """
    rows = [
        (f"2026-{1 + i // 28:02d}-{1 + i % 28:02d}", 100.0, 101.0, 99.0, 100.0)
        for i in range(80)
    ]
    bars = make_bars(rows)
    bars = bars.copy()
    bars.iloc[40, :] = np.nan

    width = derive_width(
        bars, decision_idx=79, horizon=5, atr_period=14,
        atr_multiple=1.5, target_quantile=0.7, min_risk_reward=2.0, lookback=60,
    )
    assert width is None, "含 NaN 的資料不得產出柵欄寬度"


def test_nan_atr_returns_none() -> None:
    """ATR 為 NaN（視窗內有資料洞）時必須回傳 None"""
    rows = [
        (f"2026-{1 + i // 28:02d}-{1 + i % 28:02d}", 100.0, 101.0, 99.0, 100.0)
        for i in range(80)
    ]
    bars = make_bars(rows)
    bars = bars.copy()
    bars.iloc[75, :] = np.nan   # 落在 decision_idx 的 ATR 視窗內

    width = derive_width(
        bars, decision_idx=79, horizon=5, atr_period=14,
        atr_multiple=1.5, target_quantile=0.7, min_risk_reward=0.0,
    )
    assert width is None


def test_derived_width_values_are_always_finite() -> None:
    """正常資料下，產出的 target/stop/R:R 必須都是有限值"""
    price = 100.0
    rows = []
    for i in range(80):
        rows.append((f"2026-{1 + i // 28:02d}-{1 + i % 28:02d}", price, price + 0.5, price - 0.5, price))
        price *= 1.01
    bars = make_bars(rows)

    width = derive_width(
        bars, decision_idx=79, horizon=5, atr_period=14,
        atr_multiple=1.5, target_quantile=0.7, min_risk_reward=2.0, lookback=60,
    )
    assert width is not None
    assert np.isfinite(width.target_pct)
    assert np.isfinite(width.stop_pct)
    assert np.isfinite(width.risk_reward)
    assert np.isfinite(width.atr_value)
