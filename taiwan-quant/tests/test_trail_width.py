"""
移動停損寬度推導測試

CLAUDE.md 禁止人工寫死柵欄寬度。`trail_pct` 必須從資料推導，
且推導依據要能稽核（禁令 7、8）。

比對手算值，不拿程式輸出反填預期值。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from taiwan_quant.config.costs import DEFAULT, Tier
from taiwan_quant.labeling.trail_width import (
    MAX_TRAIL_PCT,
    TrailWidth,
    derive_trail_width,
    max_pullback_samples,
)


def make_bars(
    highs: list[float],
    lows: list[float] | None = None,
    closes: list[float] | None = None,
) -> pd.DataFrame:
    """由高點序列構造日 K；未指定時低點與收盤依固定比例推得"""
    lows = lows if lows is not None else [h * 0.99 for h in highs]
    closes = closes if closes is not None else [(h + low) / 2 for h, low in zip(highs, lows)]
    index = pd.date_range("2024-01-01", periods=len(highs), freq="B")
    return pd.DataFrame(
        {
            "open": closes,
            "high": highs,
            "low": lows,
            "close": closes,
        },
        index=index,
    )


# ══════════════════════════════════════════════════════════════
# max_pullback_samples：與 label_trailing 的觸發邏輯必須一致
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_pullback_matches_hand_calculation() -> None:
    """
    手算：

        high  100  110  106  120   95
        low    98  104  100  112   88

        區間最高（累積）  100  110  110  120  120
        回落 = 1 − low / 區間最高
              0.02000  0.05455  0.09091  0.06667  0.26667

        最大回落 = 0.26667
    """
    bars = make_bars(
        highs=[100, 110, 106, 120, 95],
        lows=[98, 104, 100, 112, 88],
    )
    samples = max_pullback_samples(bars, decision_idx=4, horizon=5, lookback=250)

    assert samples.size == 1
    assert samples[0] == pytest.approx(1 - 88 / 120, abs=1e-9)


@pytest.mark.unit
def test_pullback_peak_resets_each_window() -> None:
    """
    每個視窗獨立計算——視窗起點之前的高點不算數。

        high  200  100  101  102
        low   198   99  100  101

    horizon=3、decision_idx=3 → 兩個已走完視窗：

        視窗 [0,2]  峰值 200      最大回落 = 1 − 99/200 = 0.505
        視窗 [1,3]  峰值 100→102  最大回落 = 1 − 99/100 = 0.010

    第二個視窗才是關鍵：它看不到第 0 根的 200，所以回落是 1%
    而不是 50%。峰值若不重置，這裡會算出 0.505。
    """
    bars = make_bars(
        highs=[200, 100, 101, 102],
        lows=[198, 99, 100, 101],
    )
    samples = max_pullback_samples(bars, decision_idx=3, horizon=3, lookback=250)

    assert samples.size == 2
    assert samples[0] == pytest.approx(1 - 99 / 200, abs=1e-9)
    assert samples[1] == pytest.approx(0.01, abs=1e-9)


@pytest.mark.unit
def test_pullback_only_uses_completed_windows() -> None:
    """
    反 look-ahead（禁令 1）：只取在決策日當天或之前就已經走完的視窗。

    12 根 K、horizon=5、decision_idx=8 →
    視窗起點可為 0..4（結束於 4..8），共 5 個。第 9 根以後不可見。
    """
    bars = make_bars(highs=list(range(100, 112)))
    samples = max_pullback_samples(bars, decision_idx=8, horizon=5, lookback=250)

    assert samples.size == 5


@pytest.mark.unit
def test_pullback_future_crash_is_invisible() -> None:
    """
    決策日之後的崩跌**不可**影響決策日的樣本。

    這是最容易寫錯也最致命的地方：用了未來的暴跌去放寬停損，
    回測會漂亮得不合理。
    """
    calm = [100 + i for i in range(30)]
    bars_calm = make_bars(highs=calm)

    crashed = calm[:20] + [40] * 10
    bars_crash = make_bars(highs=crashed)

    s_calm = max_pullback_samples(bars_calm, decision_idx=19, horizon=5, lookback=250)
    s_crash = max_pullback_samples(bars_crash, decision_idx=19, horizon=5, lookback=250)

    np.testing.assert_allclose(s_calm, s_crash)


@pytest.mark.unit
def test_pullback_respects_lookback() -> None:
    bars = make_bars(highs=list(range(100, 200)))
    samples = max_pullback_samples(bars, decision_idx=90, horizon=5, lookback=10)
    assert samples.size == 10


@pytest.mark.unit
def test_pullback_insufficient_history_returns_empty() -> None:
    bars = make_bars(highs=[100, 101, 102])
    samples = max_pullback_samples(bars, decision_idx=1, horizon=5, lookback=250)
    assert samples.size == 0


# ══════════════════════════════════════════════════════════════
# derive_trail_width
# ══════════════════════════════════════════════════════════════


def noisy_bars(amplitude: float, n: int = 300, seed: int = 3) -> pd.DataFrame:
    """構造指定振幅的震盪上行序列"""
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0.0005, amplitude, n))
    high = close * (1 + amplitude)
    low = close * (1 - amplitude)
    index = pd.date_range("2023-01-02", periods=n, freq="B")
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close}, index=index
    )


@pytest.mark.unit
def test_higher_volatility_gives_wider_trail() -> None:
    """波動大的股票停損要放寬，否則被日常雜訊掃出場"""
    calm = derive_trail_width(noisy_bars(0.005), decision_idx=250, horizon=20)
    wild = derive_trail_width(noisy_bars(0.03), decision_idx=250, horizon=20)

    assert calm is not None and wild is not None
    assert wild.trail_pct > calm.trail_pct


@pytest.mark.unit
def test_higher_quantile_gives_wider_trail() -> None:
    """分位數越高 = 容忍越罕見的回落 = 停損越寬"""
    bars = noisy_bars(0.015)
    tight = derive_trail_width(
        bars, decision_idx=250, horizon=20, pullback_quantile=0.50
    )
    loose = derive_trail_width(
        bars, decision_idx=250, horizon=20, pullback_quantile=0.90
    )

    assert tight is not None and loose is not None
    assert loose.trail_pct > tight.trail_pct


@pytest.mark.unit
def test_trail_floor_is_round_trip_cost() -> None:
    """
    比一趟來回成本還窄的停損沒有意義——被掃出場的代價大於它保護的金額。

    地板取自 config/costs.py（禁令 3），不在此處重寫費率。
    """
    flat = noisy_bars(0.0001)
    width = derive_trail_width(flat, decision_idx=250, horizon=20)

    assert width is not None
    assert width.trail_pct >= DEFAULT.round_trip_rate(Tier.LARGE)
    assert width.floor_applied is True


@pytest.mark.unit
def test_trail_is_capped() -> None:
    """極端波動不可推出 90% 的「停損」——那已經不是風險控制"""
    wild = noisy_bars(0.20)
    width = derive_trail_width(wild, decision_idx=250, horizon=20)

    assert width is not None
    assert width.trail_pct <= MAX_TRAIL_PCT
    assert width.cap_applied is True


@pytest.mark.unit
def test_insufficient_history_returns_none() -> None:
    bars = noisy_bars(0.01, n=30)
    assert derive_trail_width(bars, decision_idx=3, horizon=20) is None


@pytest.mark.unit
def test_non_finite_data_returns_none() -> None:
    """
    上游資料洞會讓 np.quantile 回 NaN，而 `NaN < 門檻` 為 False——
    靜默通過檢查。實測案例：2317 在 2025-07-30 有一列 OHLC 全為 NULL。
    """
    bars = noisy_bars(0.01).copy()
    bars.iloc[100] = np.nan
    assert derive_trail_width(bars, decision_idx=250, horizon=20) is None


@pytest.mark.unit
def test_reports_audit_fields() -> None:
    """禁令 7、8：要能反查「當時停損為什麼設在這個距離」"""
    bars = noisy_bars(0.015)
    width = derive_trail_width(bars, decision_idx=250, horizon=20, lookback=120)

    assert width is not None
    assert width.horizon == 20
    assert width.lookback == 120
    assert width.pullback_quantile == pytest.approx(0.80)
    assert width.sample_size == 120
    assert np.isfinite(width.atr_value)
    assert width.raw_trail_pct > 0


@pytest.mark.unit
def test_is_immutable() -> None:
    width = derive_trail_width(noisy_bars(0.015), decision_idx=250, horizon=20)
    assert width is not None
    with pytest.raises(Exception):
        width.trail_pct = 0.5  # type: ignore[misc]


@pytest.mark.unit
def test_rejects_invalid_quantile() -> None:
    bars = noisy_bars(0.01)
    with pytest.raises(ValueError, match="pullback_quantile"):
        derive_trail_width(bars, decision_idx=250, horizon=20, pullback_quantile=1.0)


@pytest.mark.unit
def test_rejects_out_of_range_decision_idx() -> None:
    bars = noisy_bars(0.01, n=50)
    with pytest.raises(ValueError, match="decision_idx"):
        derive_trail_width(bars, decision_idx=50, horizon=20)


@pytest.mark.unit
def test_rejects_missing_columns() -> None:
    bars = noisy_bars(0.01).drop(columns=["low"])
    with pytest.raises(ValueError, match="缺少必要欄位"):
        derive_trail_width(bars, decision_idx=250, horizon=20)


# ══════════════════════════════════════════════════════════════
# 與 label_trailing 串接
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_derived_width_feeds_labeler() -> None:
    """推導出的寬度可直接餵給標記器，不需要人工介入"""
    from taiwan_quant.labeling.trailing_stop import label_trailing

    bars = noisy_bars(0.015)
    width = derive_trail_width(bars, decision_idx=250, horizon=20)
    assert width is not None

    exit_result = label_trailing(
        bars, decision_idx=250, trail_pct=width.trail_pct, max_horizon=20
    )
    assert exit_result is not None
    assert exit_result.exit_reason in {"trailing_stop", "time"}
