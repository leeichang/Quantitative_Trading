#!/usr/bin/env python3
"""
特徵測試

兩層保障：
  1. **手算值比對** —— 每個特徵的預期值都在註解裡手算（禁令 10）
  2. **物理截斷掃描** —— 整組特徵過 `validation.lookahead`（禁令 1、規格 16）

第 2 層是重點：手算只能驗「算得對不對」，截斷掃描才能驗「有沒有偷看未來」。
全樣本標準化這種洩漏，手算測試完全看不出來。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from taiwan_quant.features.technical import (
    TECHNICAL_FEATURES,
    build_technical,
    bollinger_position,
    ma_ratio,
    momentum,
    rsi,
    volume_ratio,
)
from taiwan_quant.validation.lookahead import scan_builder

pytestmark = pytest.mark.unit


def make_bars(rows: list[tuple[str, float, float, float, float, int]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()


def ramp_bars(n: int, start: float = 100.0, step: float = 1.0) -> pd.DataFrame:
    """等差上升的日 K，讓均線與動能可以手算"""
    rows = [
        (
            f"2026-{1 + i // 28:02d}-{1 + i % 28:02d}",
            start + i * step,
            start + i * step + 1,
            start + i * step - 1,
            start + i * step,
            1000,
        )
        for i in range(n)
    ]
    return make_bars(rows)


def random_bars(n: int = 120, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.02, n))
    index = pd.date_range("2026-01-01", periods=n, freq="B", name="date")
    return pd.DataFrame(
        {
            "open": close * (1 + rng.normal(0, 0.004, n)),
            "high": close * (1 + rng.uniform(0.001, 0.02, n)),
            "low": close * (1 - rng.uniform(0.001, 0.02, n)),
            "close": close,
            "volume": rng.integers(1_000, 50_000, n),
        },
        index=index,
    )


# ══════════════════════════════════════════════════════════════
# ma_ratio：收盤相對均線
# ══════════════════════════════════════════════════════════════


def test_ma_ratio_on_ramp() -> None:
    """
    等差序列 100, 101, ..., 上升 step=1，window=5。

    在第 5 個索引（收盤 104）時：
        MA5 = (100+101+102+103+104)/5 = 102
        ma_ratio = 104/102 − 1 = 0.019607843...
    """
    bars = ramp_bars(10)
    result = ma_ratio(bars, window=5)
    assert result.iloc[4] == pytest.approx(104 / 102 - 1)


def test_ma_ratio_first_window_is_nan() -> None:
    """視窗不足時必為 NaN，不可用不完整視窗硬算"""
    result = ma_ratio(ramp_bars(10), window=5)
    assert result.iloc[:4].isna().all()
    assert not np.isnan(result.iloc[4])


def test_ma_ratio_is_zero_on_flat_prices() -> None:
    """平盤時收盤等於均線，比率為 0"""
    bars = ramp_bars(10, start=100.0, step=0.0)
    assert ma_ratio(bars, window=5).iloc[9] == pytest.approx(0.0)


# ══════════════════════════════════════════════════════════════
# momentum：N 日報酬
# ══════════════════════════════════════════════════════════════


def test_momentum_on_ramp() -> None:
    """
    等差序列，window=5，在索引 5（收盤 105）時：
        5 日前收盤 = 100
        momentum = 105/100 − 1 = 0.05
    """
    result = momentum(ramp_bars(10), window=5)
    assert result.iloc[5] == pytest.approx(0.05)


def test_momentum_sign_follows_trend() -> None:
    """下跌趨勢動能必須為負——符號錯了整個策略方向會反"""
    down = ramp_bars(20, start=200.0, step=-2.0)
    assert momentum(down, window=5).iloc[10] < 0


def test_momentum_first_window_is_nan() -> None:
    result = momentum(ramp_bars(10), window=5)
    assert result.iloc[:5].isna().all()


# ══════════════════════════════════════════════════════════════
# rsi
# ══════════════════════════════════════════════════════════════


def test_rsi_is_100_on_monotonic_rise() -> None:
    """
    只漲不跌 → 平均跌幅為 0 → RSI = 100。

    手算：RSI = 100 − 100/(1 + RS)，RS = 平均漲/平均跌 → ∞
    """
    result = rsi(ramp_bars(40), window=14)
    assert result.iloc[-1] == pytest.approx(100.0)


def test_rsi_is_zero_on_monotonic_fall() -> None:
    result = rsi(ramp_bars(40, start=200.0, step=-1.0), window=14)
    assert result.iloc[-1] == pytest.approx(0.0)


def test_rsi_is_50_on_alternating_equal_moves() -> None:
    """
    漲跌幅相同且交替 → 平均漲 = 平均跌 → RS = 1 → RSI = 50。

    構造：100, 101, 100, 101, ... 漲 1 跌 1
    """
    rows = []
    for i in range(60):
        price = 100.0 + (i % 2)
        rows.append((f"2026-{1 + i // 28:02d}-{1 + i % 28:02d}", price, price + 0.5, price - 0.5, price, 1000))
    result = rsi(make_bars(rows), window=14)
    assert result.iloc[-1] == pytest.approx(50.0, abs=1e-6)


def test_rsi_bounded_zero_to_hundred() -> None:
    result = rsi(random_bars(200), window=14).dropna()
    assert (result >= 0).all() and (result <= 100).all()


# ══════════════════════════════════════════════════════════════
# bollinger_position
# ══════════════════════════════════════════════════════════════


def test_bollinger_position_is_half_at_middle_band() -> None:
    """
    收盤等於中軌時，位置 = 0.5（0 = 下軌、1 = 上軌）。

    構造：平盤資料，收盤永遠等於 20 日均線。
    標準差為 0 時要避免除零 → 回 0.5（中性），不可回 inf 或 NaN。
    """
    bars = ramp_bars(40, start=100.0, step=0.0)
    assert bollinger_position(bars, window=20).iloc[-1] == pytest.approx(0.5)


def test_bollinger_position_above_one_when_breaking_upper() -> None:
    """
    突破上軌時位置 > 1。

    構造：前 30 天平盤 100，最後一天跳到 200 → 遠在上軌之上。
    """
    rows = [
        (f"2026-{1 + i // 28:02d}-{1 + i % 28:02d}", 100.0, 100.5, 99.5, 100.0, 1000)
        for i in range(30)
    ]
    rows.append(("2026-02-03", 200.0, 201.0, 199.0, 200.0, 1000))
    result = bollinger_position(make_bars(rows), window=20)
    assert result.iloc[-1] > 1.0


def test_bollinger_position_handles_zero_std() -> None:
    """標準差為 0（完全平盤）不可產生 inf 或 NaN"""
    result = bollinger_position(ramp_bars(40, step=0.0), window=20)
    assert np.isfinite(result.iloc[-1])


# ══════════════════════════════════════════════════════════════
# volume_ratio
# ══════════════════════════════════════════════════════════════


def test_volume_ratio_on_known_values() -> None:
    """
    手算：前 5 天量都是 1000，第 6 天量 3000，window=5
        VOL_MA5（含當日）= (1000+1000+1000+1000+3000)/5 = 1400
        volume_ratio = 3000/1400 = 2.142857...
    """
    rows = [
        (f"2026-01-{5 + i:02d}", 100.0, 101.0, 99.0, 100.0, 1000)
        for i in range(5)
    ]
    rows.append(("2026-01-10", 100.0, 101.0, 99.0, 100.0, 3000))
    result = volume_ratio(make_bars(rows), window=5)
    assert result.iloc[-1] == pytest.approx(3000 / 1400)


def test_volume_ratio_is_one_on_constant_volume() -> None:
    assert volume_ratio(ramp_bars(20), window=5).iloc[-1] == pytest.approx(1.0)


def test_volume_ratio_handles_zero_volume() -> None:
    """停牌日量為 0 不可產生 inf"""
    rows = [
        (f"2026-01-{5 + i:02d}", 100.0, 101.0, 99.0, 100.0, 0)
        for i in range(10)
    ]
    result = volume_ratio(make_bars(rows), window=5)
    assert not np.isinf(result.iloc[-1])


# ══════════════════════════════════════════════════════════════
# build_technical：整組特徵
# ══════════════════════════════════════════════════════════════


def test_build_technical_returns_all_declared_features() -> None:
    """產出的欄位必須與 TECHNICAL_FEATURES 宣告一致，不可默默多或少"""
    frame = build_technical(random_bars(200))
    assert set(frame.columns) == set(TECHNICAL_FEATURES)


def test_build_technical_preserves_index() -> None:
    bars = random_bars(200)
    frame = build_technical(bars)
    pd.testing.assert_index_equal(frame.index, bars.index)


def test_build_technical_does_not_mutate_input() -> None:
    bars = random_bars(200)
    before = bars.copy(deep=True)
    build_technical(bars)
    pd.testing.assert_frame_equal(bars, before)


def test_build_technical_validates_columns() -> None:
    bad = pd.DataFrame({"close": [1.0, 2.0]}, index=pd.to_datetime(["2026-01-05", "2026-01-06"]))
    with pytest.raises(ValueError, match="缺少必要欄位"):
        build_technical(bad)


def test_build_technical_has_no_infinite_values() -> None:
    """
    任何特徵出現 inf 都會讓後續模型訓練崩潰或給出荒謬權重。

    inf 常來自除以 0（零成交量、零標準差），必須在特徵層擋掉。
    """
    frame = build_technical(random_bars(300))
    assert not np.isinf(frame.to_numpy(dtype=float)).any(), "特徵含 inf"


def test_build_technical_tail_has_no_nan() -> None:
    """
    資料充足時，最後一列不該有 NaN——否則最新決策日算不出特徵。

    250 列足以填滿所有視窗（最長 120 日動能）。
    """
    frame = build_technical(random_bars(300))
    assert frame.iloc[-1].notna().all(), f"最後一列仍有 NaN：{frame.iloc[-1][frame.iloc[-1].isna()].index.tolist()}"


# ══════════════════════════════════════════════════════════════
# 物理截斷掃描（規格 16）—— 這一節是重點
# ══════════════════════════════════════════════════════════════


def test_all_technical_features_pass_truncation_scan() -> None:
    """
    整組技術特徵必須通過物理截斷測試。

    `scan_builder` 預設會跑正控制組（4 個故意作弊的特徵），
    抓不到就拋錯，所以這裡的「通過」不是假通過。
    """
    result = scan_builder(build_technical, random_bars(300))

    assert result.positive_control_passed, result.positive_control_detail
    assert result.is_clean, result.describe()
    assert len(result.clean_features) == len(TECHNICAL_FEATURES)


def test_truncation_scan_covers_multiple_cut_points() -> None:
    """單一切點可能剛好躲過洩漏，必須多點驗證"""
    result = scan_builder(build_technical, random_bars(300))
    any_report = next(iter(result.reports.values()))
    assert any_report.cuts_tested >= 3
