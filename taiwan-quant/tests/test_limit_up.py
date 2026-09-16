"""
漲停判定的測試

## 為什麼每個值都手算

漲停率是目前實測到最強的預測力（隨機 2.47% → Top3 16.62%），而原本
那份分析沒有留下可測試的程式。一個沒有測試的臨時計算，下次重跑不保證
得到同一個數字。

## 為什麼不用「容差」

漲停價是 `參考價 × 1.1` 依跳動單位**向下**取整，所以實際漲幅通常低於
10%。第一版用一個 0.4 pp 的容差處理，但那是一個要辯護的參數。

跳動單位是公告規則，用它精算就不需要容差——而且第一版那個容差的
測試案例本身是編錯的：790 元的跳動單位是 1 元，漲停價是
790 × 1.1 = **869**（+10.00%），不是我當時寫的 865（+9.49%）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from taiwan_quant.labeling.limit_up import (
    LEGACY_LIMIT_UP_PCT,
    LIMIT_UP_PCT,
    TICK_BANDS,
    hit_within,
    limit_up_events,
    limit_up_price,
    limit_up_price_frame,
    locked_all_day,
    tick_size,
)


def series(values: list[float]) -> pd.Series:
    return pd.Series(values, index=pd.date_range("2020-01-01", periods=len(values)))


# ══════════════════════════════════════════════════════════════
# 跳動單位與漲停價
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
@pytest.mark.parametrize(
    "price, expected_tick",
    [
        (9.99, 0.01),
        (10.00, 0.05),
        (49.95, 0.05),
        (50.00, 0.10),
        (99.90, 0.10),
        (100.00, 0.50),
        (499.50, 0.50),
        (500.00, 1.00),
        (999.00, 1.00),
        (1000.00, 5.00),
        (5000.00, 5.00),
    ],
)
def test_tick_size_band_boundaries(price: float, expected_tick: float) -> None:
    """帶界是開區間上界：價格等於上界時落入下一帶"""
    assert tick_size(price) == expected_tick


@pytest.mark.unit
def test_tick_size_rejects_nonpositive() -> None:
    with pytest.raises(ValueError, match="price"):
        tick_size(0.0)


@pytest.mark.unit
@pytest.mark.parametrize(
    "reference, expected_limit, expected_pct",
    [
        # 790 × 1.1 = 869.00 恰好是 1 元跳動的整數倍 → 剛好 +10%
        (790.00, 869.00, 0.100000),
        # 27.90 × 1.1 = 30.69 → 0.05 跳動向下取整 → 30.65
        (27.90, 30.65, 0.098566),
        # 9.99 × 1.1 = 10.989 → 跨到 10~50 帶（0.05）→ 10.95
        # **最壞情況**：跨帶時折讓最大
        (9.99, 10.95, 0.096096),
        (45.50, 50.00, 0.098901),
        (91.00, 100.00, 0.098901),
        (910.00, 1000.00, 0.098901),
        (1200.00, 1320.00, 0.100000),
    ],
)
def test_limit_up_price_hand_computed(
    reference: float, expected_limit: float, expected_pct: float
) -> None:
    """全部手算，不是拿程式輸出反填"""
    price = limit_up_price(reference)

    assert price == pytest.approx(expected_limit)
    assert price / reference - 1 == pytest.approx(expected_pct, abs=1e-6)


@pytest.mark.unit
def test_worst_case_shortfall_is_under_half_a_percent() -> None:
    """
    掃過每個帶界附近，確認折讓的最壞情況。

    這決定了「用固定容差」需要多寬——實測最壞 0.39 pp，也就是第一版
    那個 0.4 pp 容差其實是對的，但它剛好壓在邊界上。精算就沒有這個問題。
    """
    worst = 0.0
    for upper, _ in TICK_BANDS[:-1]:
        for reference in (upper / 1.1 - 0.01, upper / 1.1, upper / 1.1 + 0.01,
                          upper - 0.01, upper, upper + 0.01):
            if reference <= 0:
                continue
            worst = max(worst, LIMIT_UP_PCT - (limit_up_price(reference) / reference - 1))

    assert worst < 0.005, f"最壞折讓 {worst:.4%} 超過 0.5 pp"


@pytest.mark.unit
def test_legacy_seven_percent_limit() -> None:
    """
    2015-06-01 之前是 ±7%。資料庫從 2015-01-05 開始，**前五個月落在
    舊制度內**。

    手算：100 × 1.07 = 107.00，跳動 0.50（100~500 帶）→ 漲停 107.00
    """
    assert limit_up_price(100.0, limit=LEGACY_LIMIT_UP_PCT) == pytest.approx(107.0)
    assert limit_up_price(100.0, limit=LIMIT_UP_PCT) == pytest.approx(110.0)


@pytest.mark.unit
def test_vectorised_matches_scalar() -> None:
    """
    純量版與矩陣版**不可漂移**。

    純量版給測試比對手算值，矩陣版給實際計算（逐格 Python 呼叫在
    330k 格上會慢兩個數量級）。兩份實作就是兩份可能不一致的規則。
    """
    rng = np.random.default_rng(20260916)
    prices = rng.uniform(5.0, 3000.0, size=(40, 12))
    frame = pd.DataFrame(prices, index=pd.date_range("2020-01-01", periods=40))

    vectorised = limit_up_price_frame(frame)

    for row in range(frame.shape[0]):
        for col in range(frame.shape[1]):
            assert vectorised.iat[row, col] == pytest.approx(
                limit_up_price(float(frame.iat[row, col]))
            )


@pytest.mark.unit
def test_vectorised_handles_missing_and_nonpositive() -> None:
    """缺值與非正價格回 NaN，不可拋錯也不可算出一個數字"""
    frame = pd.DataFrame(
        {"A": [np.nan, 0.0, -5.0, 100.0]},
        index=pd.date_range("2020-01-01", periods=4),
    )

    result = limit_up_price_frame(frame)

    assert result["A"].isna().tolist() == [True, True, True, False]
    assert result["A"].iloc[3] == pytest.approx(110.0)


# ══════════════════════════════════════════════════════════════
# limit_up_events
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_first_day_is_never_limit_up() -> None:
    """第一天沒有前收，`pct_change` 是 NaN，必須判 False"""
    closes = pd.DataFrame(
        {"A": [100.0, 110.0, 110.0]},
        index=pd.date_range("2020-01-01", periods=3),
    )

    events = limit_up_events(closes)

    assert events["A"].tolist() == [False, True, False]


@pytest.mark.unit
def test_events_do_not_mutate_input() -> None:
    """不就地改寫傳入的 DataFrame（CLAUDE.md 程式風格）"""
    closes = pd.DataFrame(
        {"A": [100.0, 110.0]}, index=pd.date_range("2020-01-01", periods=2)
    )
    before = closes.copy()

    limit_up_events(closes)

    pd.testing.assert_frame_equal(closes, before)


# ══════════════════════════════════════════════════════════════
# hit_within：標籤，區間左開右閉
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_hit_window_excludes_the_decision_day() -> None:
    """
    t 日收盤後決策，所以 **t 日的漲停不算**——那時它已經發生、買不到了。

    手算：A 在 index 2 漲停（100 → 110）。

        t=0   視窗 (0, 2] = index 1,2   含 index 2 的漲停   → True
        t=1   視窗 (1, 3] = index 2,3   含 index 2 的漲停   → True
        t=2   視窗 (2, 4] = index 3,4   沒有                → False
    """
    closes = pd.DataFrame(
        {"A": [100.0, 100.0, 110.0, 110.0, 110.0]},
        index=pd.date_range("2020-01-01", periods=5),
    )
    events = limit_up_events(closes)
    assert events["A"].tolist() == [False, False, True, False, False]

    label = hit_within(events, horizon=2)

    assert label["A"].tolist() == [True, True, False, False, False]


@pytest.mark.unit
def test_hit_window_length_is_respected() -> None:
    """
    視窗長度要精確。A 在 index 4 漲停。

        horizon=2  t=0 的視窗是 (0,2]，不含 index 4  → False
        horizon=4  t=0 的視窗是 (0,4]，含 index 4    → True
    """
    closes = pd.DataFrame(
        {"A": [100.0, 100.0, 100.0, 100.0, 110.0]},
        index=pd.date_range("2020-01-01", periods=5),
    )
    events = limit_up_events(closes)

    assert hit_within(events, horizon=2)["A"].iloc[0] is np.False_
    assert hit_within(events, horizon=4)["A"].iloc[0] is np.True_


@pytest.mark.unit
def test_hit_window_rejects_nonpositive_horizon() -> None:
    closes = pd.DataFrame(
        {"A": [100.0, 110.0]}, index=pd.date_range("2020-01-01", periods=2)
    )
    with pytest.raises(ValueError, match="horizon"):
        hit_within(limit_up_events(closes), horizon=0)


# ══════════════════════════════════════════════════════════════
# locked_all_day：買不到的漲停
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_locked_all_day_needs_both_limit_up_and_single_price() -> None:
    """
    高 = 低代表全天一個價位。漲停且一價到底 = 開盤就鎖上，掛單排不到。

    A  漲停且鎖死    → True
    B  漲停但有振幅  → False（買得到）
    C  沒漲停但一價  → False（一價到底不等於漲停）
    """
    index = pd.date_range("2020-01-01", periods=2)
    closes = pd.DataFrame(
        {"A": [100.0, 110.0], "B": [100.0, 110.0], "C": [100.0, 100.0]}, index=index
    )
    highs = pd.DataFrame(
        {"A": [100.0, 110.0], "B": [100.0, 111.0], "C": [100.0, 100.0]}, index=index
    )
    lows = pd.DataFrame(
        {"A": [100.0, 110.0], "B": [100.0, 105.0], "C": [100.0, 100.0]}, index=index
    )

    locked = locked_all_day(highs, lows, limit_up_events(closes))

    assert locked.iloc[1].tolist() == [True, False, False]
