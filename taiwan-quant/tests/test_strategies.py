#!/usr/bin/env python3
"""
策略族測試

D7：三族同場比較

    A. 動能突破    20/60 日動能 + 量能放大 + 站上均線
    B. 籌碼跟隨    外資／投信連續買超 + 融資使用率低檔
    C. 均值回歸    RSI 超賣 + 觸及布林下軌 + 未跌破長均線

三族共用同一套 triple-barrier 標記、同一套成本模型、同一套驗證流程，
唯一的差別是**分數怎麼算**。這個切分讓「哪一族有優勢」變成可比較的問題。

分數的絕對值不重要——校準會把它映射成真實機率。重要的是**方向**：
分數高的樣本，實際命中率必須比較高。所以測試重點在：

  1. 方向正確（多頭情境分數高、空頭情境分數低）
  2. 缺資料時回 NaN 而不是猜
  3. 通過物理截斷掃描（不偷看未來）
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from taiwan_quant.strategies.families import (
    MIN_HISTORY,
    STRATEGY_FAMILIES,
    StrategyFamily,
    chips_following_score,
    mean_reversion_score,
    momentum_breakout_score,
)
from taiwan_quant.validation.lookahead import scan_builder

pytestmark = pytest.mark.unit


def trend_bars(n: int, daily_return: float, volume: int = 1000) -> pd.DataFrame:
    """等比趨勢日 K"""
    close = 100.0 * np.cumprod(np.full(n, 1 + daily_return))
    index = pd.date_range("2026-01-01", periods=n, freq="B", name="date")
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.full(n, volume, dtype="int64"),
        },
        index=index,
    )


def noisy_bars(n: int = 300, seed: int = 3, drift: float = 0.0005) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(drift, 0.02, n))
    index = pd.date_range("2026-01-01", periods=n, freq="B", name="date")
    return pd.DataFrame(
        {
            "open": close * (1 + rng.normal(0, 0.004, n)),
            "high": close * (1 + rng.uniform(0.002, 0.02, n)),
            "low": close * (1 - rng.uniform(0.002, 0.02, n)),
            "close": close,
            "volume": rng.integers(1_000, 50_000, n),
        },
        index=index,
    )


def chips_frame(
    bars: pd.DataFrame,
    foreign_net: float,
    trust_net: float = 0.0,
    dealer_net: float = 0.0,
    margin_balance: float = 100_000.0,
    short_balance: float = 10_000.0,
) -> pd.DataFrame:
    """把日 K 補成籌碼特徵需要的格式"""
    return bars.assign(
        foreign_net=foreign_net,
        trust_net=trust_net,
        dealer_net=dealer_net,
        margin_balance=margin_balance,
        short_balance=short_balance,
    )


# ══════════════════════════════════════════════════════════════
# A. 動能突破
# ══════════════════════════════════════════════════════════════


def test_momentum_higher_in_uptrend_than_downtrend() -> None:
    """
    方向正確性：上升趨勢的分數必須高於下降趨勢。

    方向錯了整個策略會反向操作，而回測仍會跑出數字——
    這是最容易被忽略又最致命的錯誤。
    """
    up = momentum_breakout_score(trend_bars(300, 0.004))
    down = momentum_breakout_score(trend_bars(300, -0.004))
    assert up > down


def test_momentum_rewards_volume_expansion() -> None:
    """
    同樣的漲幅，量能放大時分數應較高。

    構造：只放大**最後 3 天**的量。

    不可放大整個 20 日視窗——volume_ratio_20 = volume / 20日均量，
    整段放大會讓分母同步上移，比值回到 1.0，測不出差異。
    手算：(17×1000 + 3×5000)/20 = 1,600，比值 = 5000/1600 = 3.125
    """
    base = trend_bars(300, 0.003, volume=1000)
    expanded = base.copy()
    expanded.iloc[-3:, expanded.columns.get_loc("volume")] = 5000

    assert momentum_breakout_score(expanded) > momentum_breakout_score(base)


def test_momentum_bounded() -> None:
    """分數必須有界，否則極端行情會產生離群值汙染校準分箱"""
    for drift in (-0.01, -0.002, 0.0, 0.002, 0.01):
        score = momentum_breakout_score(trend_bars(300, drift))
        assert 0.0 <= score <= 1.0


def test_momentum_nan_when_insufficient_history() -> None:
    """視窗不足回 NaN，不可用不完整資料算"""
    assert np.isnan(momentum_breakout_score(trend_bars(30, 0.003)))


# ══════════════════════════════════════════════════════════════
# B. 籌碼跟隨
# ══════════════════════════════════════════════════════════════


def test_chips_higher_on_sustained_foreign_buying() -> None:
    """外資持續買超的分數必須高於持續賣超"""
    bars = noisy_bars(300)
    buying = chips_following_score(chips_frame(bars, foreign_net=50_000))
    selling = chips_following_score(chips_frame(bars, foreign_net=-50_000))
    assert buying > selling


def test_chips_rewards_institutional_agreement() -> None:
    """
    三大法人同步買超的分數，必須高於只有外資買、其餘賣。

    D7 的籌碼族核心假設就是「合力比單一法人有訊息量」。
    """
    bars = noisy_bars(300)
    aligned = chips_following_score(
        chips_frame(bars, foreign_net=30_000, trust_net=10_000, dealer_net=5_000)
    )
    conflicted = chips_following_score(
        chips_frame(bars, foreign_net=30_000, trust_net=-10_000, dealer_net=-5_000)
    )
    assert aligned > conflicted


def test_chips_penalizes_high_margin_usage() -> None:
    """
    融資餘額相對均量偏高 = 散戶追高，是反向指標。

    構造：同樣的法人買超，融資餘額差 100 倍。
    """
    bars = noisy_bars(300)
    low_margin = chips_following_score(
        chips_frame(bars, foreign_net=30_000, margin_balance=10_000)
    )
    high_margin = chips_following_score(
        chips_frame(bars, foreign_net=30_000, margin_balance=5_000_000)
    )
    assert low_margin > high_margin


def test_chips_bounded() -> None:
    bars = noisy_bars(300)
    for net in (-100_000, 0, 100_000):
        score = chips_following_score(chips_frame(bars, foreign_net=net))
        assert 0.0 <= score <= 1.0


def test_chips_nan_when_columns_missing() -> None:
    """
    缺籌碼欄位回 NaN，不可拋錯。

    上游籌碼資料常常不完整，整批標的裡少數幾檔缺資料是常態，
    不該讓整輪掃描中斷。
    """
    assert np.isnan(chips_following_score(noisy_bars(300)))


def test_chips_nan_when_insufficient_history() -> None:
    assert np.isnan(chips_following_score(chips_frame(noisy_bars(30), 1000)))


# ══════════════════════════════════════════════════════════════
# C. 均值回歸
# ══════════════════════════════════════════════════════════════


def test_mean_reversion_higher_when_oversold() -> None:
    """
    同一檔股票，急跌後的分數必須高於急跌前。

    構造：長期上升（60MA 之上不變）→ 最後 8 天連續下跌。
    這隔離出「超賣程度」這一個變數。

    注意基準不是「純上升趨勢的分數應該低」——純上升趨勢本來就是
    **超買**，均值回歸族本來就該給低分。這裡比的是同一檔的前後差異。
    """
    base = trend_bars(300, 0.003)

    oversold = base.copy()
    pivot = float(base["close"].iloc[-9])
    dropped = pivot * np.cumprod(np.full(8, 0.97))
    for col, factor in (("open", 1.0), ("close", 1.0), ("high", 1.01), ("low", 0.99)):
        oversold.iloc[-8:, oversold.columns.get_loc(col)] = dropped * factor

    assert mean_reversion_score(oversold) > mean_reversion_score(base)


def test_mean_reversion_penalizes_broken_long_ma() -> None:
    """
    同樣超賣，但跌破 60 日均線的分數必須較低。

    「接下墜的刀」與「拉回承接」的差別就在這條。

    構造：兩檔都在最後 8 天急跌 3%/日（超賣程度相同），
    差別在前段趨勢——一檔長期上升（跌完仍在 60MA 之上），
    一檔長期下跌（早已跌破）。這隔離出「長均線是否守住」這一個變數。
    """
    def with_selloff(drift: float) -> pd.DataFrame:
        bars = trend_bars(300, drift).copy()
        pivot = float(bars["close"].iloc[-9])
        dropped = pivot * np.cumprod(np.full(8, 0.97))
        for col, factor in (("open", 1.0), ("close", 1.0), ("high", 1.01), ("low", 0.99)):
            bars.iloc[-8:, bars.columns.get_loc(col)] = dropped * factor
        return bars

    intact = with_selloff(0.004)    # 長期上升，跌完仍守住 60MA
    broken = with_selloff(-0.004)   # 長期下跌，早已跌破

    assert mean_reversion_score(intact) > mean_reversion_score(broken)


def test_mean_reversion_bounded() -> None:
    for drift in (-0.006, 0.0, 0.006):
        score = mean_reversion_score(trend_bars(300, drift))
        assert 0.0 <= score <= 1.0


def test_mean_reversion_nan_when_insufficient_history() -> None:
    assert np.isnan(mean_reversion_score(trend_bars(30, 0.003)))


# ══════════════════════════════════════════════════════════════
# 三族的方向必須真的不同
# ══════════════════════════════════════════════════════════════


def test_momentum_and_mean_reversion_disagree_on_trend() -> None:
    """
    強勢上漲時，動能族看多、均值回歸族相對保守。

    若兩族在同一情境給出相同排序，那它們就是同一個策略換個名字，
    「三族比較」失去意義。
    """
    strong = trend_bars(300, 0.006)
    mild = trend_bars(300, 0.001)

    momentum_gap = momentum_breakout_score(strong) - momentum_breakout_score(mild)
    reversion_gap = mean_reversion_score(strong) - mean_reversion_score(mild)

    assert momentum_gap > 0, "動能族在強勢上漲時應給高分"
    assert momentum_gap > reversion_gap, "兩族對趨勢的反應應明顯不同"


# ══════════════════════════════════════════════════════════════
# 目錄
# ══════════════════════════════════════════════════════════════


def test_registry_has_three_families() -> None:
    assert len(STRATEGY_FAMILIES) == 3
    assert {f.name for f in STRATEGY_FAMILIES} == {"動能突破", "籌碼跟隨", "均值回歸"}


def test_registry_declares_required_columns() -> None:
    """
    每族要宣告需要哪些欄位，呼叫端才能先檢查資料夠不夠，
    而不是跑到一半才發現缺欄位。
    """
    by_name = {f.name: f for f in STRATEGY_FAMILIES}
    assert "foreign_net" in by_name["籌碼跟隨"].required_columns
    assert "foreign_net" not in by_name["動能突破"].required_columns


def test_family_is_immutable() -> None:
    with pytest.raises(Exception):
        STRATEGY_FAMILIES[0].name = "x"  # type: ignore[misc]


def test_all_families_produce_finite_scores_on_real_shaped_data() -> None:
    """三族在同一份完整資料上都要能算出有限分數"""
    bars = chips_frame(noisy_bars(300), foreign_net=10_000)
    for family in STRATEGY_FAMILIES:
        score = family.score_fn(bars)
        assert np.isfinite(score), f"{family.name} 算不出分數"


# ══════════════════════════════════════════════════════════════
# 物理截斷掃描（規格 16）
# ══════════════════════════════════════════════════════════════


def test_all_families_pass_truncation_scan() -> None:
    """
    三族的分數序列都必須通過物理截斷測試。

    策略族是模型的輸入端，若它偷看未來，後面所有驗證都白做。
    """
    bars = chips_frame(noisy_bars(400), foreign_net=10_000)

    def builder(frame: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                family.name: _rolling_scores(frame, family)
                for family in STRATEGY_FAMILIES
            },
            index=frame.index,
        )

    result = scan_builder(builder, bars, cut_indices=[300, 330, 360])

    assert result.positive_control_passed, result.positive_control_detail
    assert result.is_clean, result.describe()


def _rolling_scores(frame: pd.DataFrame, family: StrategyFamily) -> pd.Series:
    """對每個決策日算分數，供截斷掃描使用"""
    values = [np.nan] * len(frame)
    for idx in range(250, len(frame)):
        values[idx] = family.score_fn(frame.iloc[: idx + 1])
    return pd.Series(values, index=frame.index, dtype="float64")


# ══════════════════════════════════════════════════════════════
# 向量化分數序列（長歷史回測用）
#
# `score_fn(bars)` 每次呼叫都重建整張特徵表，只為了取最後一列。
# 11 年 × 586 檔 × 3 族 = 百萬次呼叫，那個固定開銷變成瓶頸。
#
# `score_series(bars)` 只建一次特徵表，逐列套用同一組運算。
# **結果必須逐點完全相同**——只要有任何差異，之前所有的驗證結論
# 在新資料上就不可比。
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_score_series_matches_score_fn_pointwise() -> None:
    """對每一個決策點，序列版與逐次呼叫版必須給出相同的分數"""
    bars = chips_frame(noisy_bars(400), foreign_net=10_000)
    for family in STRATEGY_FAMILIES:
        if not set(family.required_columns).issubset(bars.columns):
            continue
        series = family.score_series(bars)
        assert len(series) == len(bars)

        for idx in range(MIN_HISTORY, len(bars), 37):
            expected = family.score_fn(bars.iloc[: idx + 1])
            actual = float(series.iloc[idx])
            if np.isnan(expected):
                assert np.isnan(actual), f"{family.name} @ {idx} 應為 NaN"
            else:
                assert actual == pytest.approx(expected, abs=1e-9), (
                    f"{family.name} @ {idx}：序列 {actual} vs 逐次 {expected}"
                )


@pytest.mark.unit
def test_score_series_is_nan_before_min_history() -> None:
    """歷史不足的位置必須是 NaN，不可用不完整的視窗硬算"""
    bars = chips_frame(noisy_bars(400), foreign_net=10_000)
    for family in STRATEGY_FAMILIES:
        if not set(family.required_columns).issubset(bars.columns):
            continue
        series = family.score_series(bars)
        assert series.iloc[: MIN_HISTORY - 1].isna().all(), family.name


@pytest.mark.unit
def test_score_series_returns_all_nan_when_columns_missing() -> None:
    """缺欄位時整段回 NaN，不拋錯——整批掃描不該因少數標的中斷"""
    bars = noisy_bars(400)
    chips_family = next(f for f in STRATEGY_FAMILIES if f.name == "籌碼跟隨")
    assert chips_family.score_series(bars).isna().all()


@pytest.mark.unit
def test_score_series_index_matches_bars() -> None:
    """索引必須對齊日 K，否則決策日會錯位（等同 look-ahead）"""
    bars = chips_frame(noisy_bars(400), foreign_net=10_000)
    family = STRATEGY_FAMILIES[0]
    assert family.score_series(bars).index.equals(bars.index)


@pytest.mark.unit
def test_score_series_does_not_mutate_input() -> None:
    bars = chips_frame(noisy_bars(400), foreign_net=10_000)
    before = bars.copy()
    STRATEGY_FAMILIES[0].score_series(bars)
    pd.testing.assert_frame_equal(bars, before)
