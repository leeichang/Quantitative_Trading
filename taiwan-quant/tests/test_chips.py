#!/usr/bin/env python3
"""
籌碼面特徵測試

D7 的籌碼跟隨族 look-ahead 風險最高，因為三大法人資料的公布時序特殊：

    T 日盤中 09:00-13:30  交易發生
    T 日收盤 13:30        收盤價確定
    T 日盤後 15:00-18:00  三大法人買賣超公布   ← 收盤已過
    T+1 開盤 09:00        進場

所以「用 T 日籌碼、以 T 日收盤價進場」是 look-ahead。
決策日 T 可以用 T 日籌碼（盤後才決策），但進場必須在 T+1 開盤。
這條由 `triple_barrier` 保證，本模組只負責「特徵不偷看 T 之後」。

本檔測試兩件事：
  1. 特徵算得對（手算值比對）
  2. 特徵不偷看未來（物理截斷掃描）
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from taiwan_quant.features.chips import (
    CHIPS_FEATURES,
    CHIPS_REQUIRED_COLUMNS,
    build_chips,
    consecutive_net_buy_days,
    margin_balance_change,
    net_buy_ratio,
    net_buy_streak_strength,
)
from taiwan_quant.validation.lookahead import scan_builder

pytestmark = pytest.mark.unit


def make_chips(rows: list[dict]) -> pd.DataFrame:
    """
    建立測試用籌碼資料。

    rows 每筆需含 date 與 CHIPS_REQUIRED_COLUMNS 的欄位。
    """
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()


def chips_row(
    day: int,
    foreign: float = 0.0,
    trust: float = 0.0,
    dealer: float = 0.0,
    margin: float = 100_000.0,
    short: float = 10_000.0,
    volume: float = 1_000_000.0,
    close: float = 100.0,
) -> dict:
    return {
        "date": f"2026-{1 + day // 28:02d}-{1 + day % 28:02d}",
        "foreign_net": foreign,
        "trust_net": trust,
        "dealer_net": dealer,
        "margin_balance": margin,
        "short_balance": short,
        "volume": volume,
        "close": close,
    }


def random_chips(n: int = 150, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return make_chips(
        [
            chips_row(
                i,
                foreign=float(rng.normal(0, 5_000)),
                trust=float(rng.normal(0, 1_000)),
                dealer=float(rng.normal(0, 500)),
                margin=float(100_000 + rng.normal(0, 5_000)),
                short=float(10_000 + rng.normal(0, 1_000)),
                volume=float(rng.integers(500_000, 2_000_000)),
                close=float(100 * (1 + rng.normal(0, 0.02))),
            )
            for i in range(n)
        ]
    )


# ══════════════════════════════════════════════════════════════
# net_buy_ratio：買賣超相對成交量
# ══════════════════════════════════════════════════════════════


def test_net_buy_ratio_on_known_values() -> None:
    """
    手算：外資買超 50,000 股、成交量 1,000,000 股，window=1
        net_buy_ratio = 50,000 / 1,000,000 = 0.05
    """
    bars = make_chips([chips_row(0, foreign=50_000, volume=1_000_000)])
    assert net_buy_ratio(bars, "foreign_net", window=1).iloc[0] == pytest.approx(0.05)


def test_net_buy_ratio_accumulates_over_window() -> None:
    """
    手算：連續 3 天各買超 10,000，成交量各 1,000,000，window=3
        累計買超 30,000 / 累計成交量 3,000,000 = 0.01
    """
    bars = make_chips([chips_row(i, foreign=10_000, volume=1_000_000) for i in range(5)])
    assert net_buy_ratio(bars, "foreign_net", window=3).iloc[4] == pytest.approx(0.01)


def test_net_buy_ratio_is_negative_on_net_sell() -> None:
    """賣超必須為負——符號錯了策略方向會反"""
    bars = make_chips([chips_row(i, foreign=-20_000, volume=1_000_000) for i in range(5)])
    assert net_buy_ratio(bars, "foreign_net", window=3).iloc[4] < 0


def test_net_buy_ratio_window_insufficient_is_nan() -> None:
    bars = make_chips([chips_row(i, foreign=10_000) for i in range(5)])
    result = net_buy_ratio(bars, "foreign_net", window=3)
    assert result.iloc[:2].isna().all()


def test_net_buy_ratio_handles_zero_volume() -> None:
    """停牌日成交量為 0 不可產生 inf"""
    bars = make_chips([chips_row(i, foreign=10_000, volume=0.0) for i in range(5)])
    result = net_buy_ratio(bars, "foreign_net", window=3)
    assert not np.isinf(result.dropna()).any()


# ══════════════════════════════════════════════════════════════
# consecutive_net_buy_days：連續買超天數
# ══════════════════════════════════════════════════════════════


def test_consecutive_counts_positive_streak() -> None:
    """
    連 4 天買超 → 第 4 天的連續天數為 4。

    D7 籌碼跟隨族的核心訊號就是這個。
    """
    bars = make_chips([chips_row(i, foreign=1_000) for i in range(4)])
    assert consecutive_net_buy_days(bars, "foreign_net").iloc[3] == 4


def test_consecutive_resets_on_sell() -> None:
    """
    買、買、賣、買 → 連續天數為 1, 2, 0, 1

    賣超那天必須歸零，不可只是「不增加」。
    """
    bars = make_chips([
        chips_row(0, foreign=1_000),
        chips_row(1, foreign=1_000),
        chips_row(2, foreign=-1_000),
        chips_row(3, foreign=1_000),
    ])
    result = consecutive_net_buy_days(bars, "foreign_net")
    assert list(result) == [1, 2, 0, 1]


def test_consecutive_zero_net_breaks_streak() -> None:
    """買超為 0（持平）不算買超，連續中斷"""
    bars = make_chips([
        chips_row(0, foreign=1_000),
        chips_row(1, foreign=0.0),
        chips_row(2, foreign=1_000),
    ])
    assert list(consecutive_net_buy_days(bars, "foreign_net")) == [1, 0, 1]


def test_consecutive_counts_negative_streak_when_asked() -> None:
    """也要能算連續賣超（空方訊號）"""
    bars = make_chips([chips_row(i, foreign=-1_000) for i in range(3)])
    assert consecutive_net_buy_days(bars, "foreign_net", direction="sell").iloc[2] == 3


# ══════════════════════════════════════════════════════════════
# net_buy_streak_strength：三大法人合力
# ══════════════════════════════════════════════════════════════


def test_streak_strength_counts_agreeing_institutions() -> None:
    """
    三大法人同步買超 → 強度 3；兩買一賣 → 強度 1（2 − 1）。

    手算：買超記 +1、賣超記 −1、持平記 0
    """
    all_buy = make_chips([chips_row(0, foreign=100, trust=100, dealer=100)])
    assert net_buy_streak_strength(all_buy).iloc[0] == 3

    mixed = make_chips([chips_row(0, foreign=100, trust=100, dealer=-100)])
    assert net_buy_streak_strength(mixed).iloc[0] == 1


def test_streak_strength_all_sell_is_negative_three() -> None:
    bars = make_chips([chips_row(0, foreign=-1, trust=-1, dealer=-1)])
    assert net_buy_streak_strength(bars).iloc[0] == -3


def test_streak_strength_flat_is_zero() -> None:
    bars = make_chips([chips_row(0)])
    assert net_buy_streak_strength(bars).iloc[0] == 0


# ══════════════════════════════════════════════════════════════
# margin_balance_change：融資餘額變化
# ══════════════════════════════════════════════════════════════


def test_margin_change_on_known_values() -> None:
    """
    手算：融資餘額 100,000 → 105,000，window=1
        變化率 = 105,000/100,000 − 1 = 0.05
    """
    bars = make_chips([
        chips_row(0, margin=100_000),
        chips_row(1, margin=105_000),
    ])
    assert margin_balance_change(bars, window=1).iloc[1] == pytest.approx(0.05)


def test_margin_change_handles_zero_base() -> None:
    """前期餘額為 0 不可產生 inf"""
    bars = make_chips([chips_row(0, margin=0.0), chips_row(1, margin=1_000.0)])
    result = margin_balance_change(bars, window=1)
    assert not np.isinf(result.dropna()).any()


# ══════════════════════════════════════════════════════════════
# build_chips：整組特徵
# ══════════════════════════════════════════════════════════════


def test_build_chips_returns_all_declared_features() -> None:
    frame = build_chips(random_chips(200))
    assert set(frame.columns) == set(CHIPS_FEATURES)


def test_build_chips_preserves_index() -> None:
    bars = random_chips(200)
    pd.testing.assert_index_equal(build_chips(bars).index, bars.index)


def test_build_chips_does_not_mutate_input() -> None:
    bars = random_chips(200)
    before = bars.copy(deep=True)
    build_chips(bars)
    pd.testing.assert_frame_equal(bars, before)


def test_build_chips_validates_columns() -> None:
    """
    籌碼資料常常不完整（FinMind 額度撞牆）。缺欄位必須明確報錯，
    不可默默產出全 NaN 的特徵——那會讓 dropna 清空整個訓練集
    （qlib-tw-trader 訓練失敗的原因）。
    """
    bad = pd.DataFrame(
        {"foreign_net": [1.0, 2.0]},
        index=pd.to_datetime(["2026-01-05", "2026-01-06"]),
    )
    with pytest.raises(ValueError, match="缺少必要欄位"):
        build_chips(bad)


def test_build_chips_has_no_infinite_values() -> None:
    frame = build_chips(random_chips(300))
    assert not np.isinf(frame.to_numpy(dtype=float)).any(), "特徵含 inf"


def test_build_chips_tail_has_no_nan() -> None:
    """資料充足時最新決策日不該有 NaN"""
    frame = build_chips(random_chips(300))
    missing = frame.iloc[-1][frame.iloc[-1].isna()].index.tolist()
    assert not missing, f"最後一列仍有 NaN：{missing}"


def test_required_columns_documented() -> None:
    """
    `CHIPS_REQUIRED_COLUMNS` 要與實際需求一致，
    讓呼叫端能先用 `coverage_report()` 檢查資料夠不夠。
    """
    assert set(CHIPS_REQUIRED_COLUMNS) == {
        "foreign_net", "trust_net", "dealer_net",
        "margin_balance", "short_balance", "volume", "close",
    }


# ══════════════════════════════════════════════════════════════
# 物理截斷掃描（規格 16）
# ══════════════════════════════════════════════════════════════


def test_all_chips_features_pass_truncation_scan() -> None:
    """
    籌碼特徵必須通過物理截斷測試。

    這一條對籌碼族格外重要——D7 標註它「look-ahead 風險最高」。
    `scan_builder` 的正控制組確保這裡的通過不是假通過。
    """
    bars = random_chips(300)
    # 掃描器需要 OHLC 才能跑內建正控制組，補上與 close 一致的欄位
    scan_input = bars.assign(
        open=bars["close"], high=bars["close"] * 1.01, low=bars["close"] * 0.99
    )

    result = scan_builder(build_chips, scan_input)

    assert result.positive_control_passed, result.positive_control_detail
    assert result.is_clean, result.describe()
    assert len(result.clean_features) == len(CHIPS_FEATURES)
