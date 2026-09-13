#!/usr/bin/env python3
"""
Triple-barrier 標記測試

每個預期值都在註解裡手算，可自行驗算。**不可拿程式輸出反填**
（CLAUDE.md 禁令 10 / AGENTS.md 檢查項 24）。

核心規則（CLAUDE.md）：

    給定 T 日收盤決策，T+1 開盤進場
        上柵 = entry × (1 + target_pct)
        下柵 = entry × (1 − stop_pct)
        時間柵 = 進場後 horizon 個交易日

    y = +1  先觸及上柵
        −1  先觸及下柵
         0  時間柵到期都沒觸及

    · 用當日 high/low 判定觸發，不是 close
    · 同一根 K 同時觸及兩柵 → 保守判 −1
    · 進場價 = T+1 開盤價（不是 T 日收盤）
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from taiwan_quant.labeling.triple_barrier import (
    BarrierLabel,
    BarrierSpec,
    label_one,
    label_series,
)

pytestmark = pytest.mark.unit


def make_bars(rows: list[tuple[str, float, float, float, float]]) -> pd.DataFrame:
    """
    建立測試用日 K。

    rows: [(date, open, high, low, close), ...]
    """
    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close"])
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()


# ══════════════════════════════════════════════════════════════
# BarrierSpec：柵欄價位計算
# ══════════════════════════════════════════════════════════════


def test_spec_computes_barrier_prices() -> None:
    """
    手算：entry 100、target 8%、stop 4%
        上柵 = 100 × 1.08 = 108.0
        下柵 = 100 × 0.96 =  96.0
    """
    spec = BarrierSpec(entry_price=100.0, target_pct=0.08, stop_pct=0.04, horizon=5)
    assert spec.upper_price == pytest.approx(108.0)
    assert spec.lower_price == pytest.approx(96.0)


def test_spec_risk_reward() -> None:
    """
    手算：R:R = target_pct / stop_pct = 0.08 / 0.04 = 2.0
    """
    spec = BarrierSpec(entry_price=100.0, target_pct=0.08, stop_pct=0.04, horizon=5)
    assert spec.risk_reward == pytest.approx(2.0)


def test_spec_rejects_nonpositive_values() -> None:
    """邊界驗證：價格與柵欄寬度必須為正，horizon 至少 1"""
    with pytest.raises(ValueError, match="entry_price"):
        BarrierSpec(entry_price=0.0, target_pct=0.08, stop_pct=0.04, horizon=5)
    with pytest.raises(ValueError, match="target_pct"):
        BarrierSpec(entry_price=100.0, target_pct=0.0, stop_pct=0.04, horizon=5)
    with pytest.raises(ValueError, match="stop_pct"):
        BarrierSpec(entry_price=100.0, target_pct=0.08, stop_pct=-0.01, horizon=5)
    with pytest.raises(ValueError, match="horizon"):
        BarrierSpec(entry_price=100.0, target_pct=0.08, stop_pct=0.04, horizon=0)


# ══════════════════════════════════════════════════════════════
# label_one：單一標記，三種結局
# ══════════════════════════════════════════════════════════════


def test_hits_upper_barrier() -> None:
    """
    上柵先觸及 → y = +1

    決策日 D0（只提供特徵，不看它的價格）
    進場日 D1 開盤 100.0 → 上柵 108.0、下柵 96.0
        D1  high 101  low  99   → 未觸
        D2  high 105  low 100   → 未觸
        D3  high 109  low 104   → 109 >= 108，觸上柵 ✓
    預期：label +1、exit_price 108.0（柵欄價，不是當日 high）、holding_days 3
    """
    bars = make_bars([
        ("2026-01-05", 100.0, 100.0, 100.0, 100.0),   # D0 決策日
        ("2026-01-06", 100.0, 101.0,  99.0, 100.5),   # D1 進場
        ("2026-01-07", 100.5, 105.0, 100.0, 104.0),   # D2
        ("2026-01-08", 104.0, 109.0, 104.0, 108.5),   # D3 觸上柵
        ("2026-01-09", 108.5, 110.0, 108.0, 109.0),   # D4
    ])
    result = label_one(bars, decision_idx=0, target_pct=0.08, stop_pct=0.04, horizon=5)

    assert result.label == 1
    assert result.entry_price == pytest.approx(100.0)
    assert result.exit_price == pytest.approx(108.0)
    assert result.holding_days == 3
    assert result.exit_reason == "upper"


def test_hits_lower_barrier() -> None:
    """
    下柵先觸及 → y = −1

    進場 100.0 → 上柵 108.0、下柵 96.0
        D1  high 101  low  99   → 未觸
        D2  high 100  low  95.5 → 95.5 <= 96，觸下柵 ✓
    預期：label −1、exit_price 96.0、holding_days 2
    """
    bars = make_bars([
        ("2026-01-05", 100.0, 100.0, 100.0, 100.0),
        ("2026-01-06", 100.0, 101.0,  99.0,  99.5),
        ("2026-01-07",  99.5, 100.0,  95.5,  96.5),   # 觸下柵
        ("2026-01-08",  96.5,  98.0,  96.0,  97.0),
        ("2026-01-09",  97.0, 100.0,  96.5,  99.0),
    ])
    result = label_one(bars, decision_idx=0, target_pct=0.08, stop_pct=0.04, horizon=5)

    assert result.label == -1
    assert result.exit_price == pytest.approx(96.0)
    assert result.holding_days == 2
    assert result.exit_reason == "lower"


def test_time_barrier_expires() -> None:
    """
    時間柵到期，兩柵都沒觸及 → y = 0

    進場 100.0 → 上柵 108.0、下柵 96.0，horizon = 3
        D1 high 102 low 99   D2 high 103 low 98   D3 high 104 low 97
        全程在 (96, 108) 內
    預期：label 0、exit_price = D3 收盤 103.5、holding_days 3
    """
    bars = make_bars([
        ("2026-01-05", 100.0, 100.0, 100.0, 100.0),
        ("2026-01-06", 100.0, 102.0,  99.0, 101.0),
        ("2026-01-07", 101.0, 103.0,  98.0, 102.0),
        ("2026-01-08", 102.0, 104.0,  97.0, 103.5),   # 時間柵
        ("2026-01-09", 103.5, 110.0,  90.0, 105.0),   # 超出 horizon，不看
    ])
    result = label_one(bars, decision_idx=0, target_pct=0.08, stop_pct=0.04, horizon=3)

    assert result.label == 0
    assert result.exit_price == pytest.approx(103.5)
    assert result.holding_days == 3
    assert result.exit_reason == "time"


def test_same_bar_touches_both_barriers_is_conservative() -> None:
    """
    同一根 K 同時觸及兩柵 → 保守判 −1（CLAUDE.md 規則）

    進場 100.0 → 上柵 108.0、下柵 96.0
        D1  high 109  low 95   → 兩柵同時在區間內
    路徑未知，不可假設先漲後跌，保守取 −1。
    """
    bars = make_bars([
        ("2026-01-05", 100.0, 100.0, 100.0, 100.0),
        ("2026-01-06", 100.0, 109.0,  95.0, 100.0),   # 同時觸兩柵
        ("2026-01-07", 100.0, 101.0,  99.0, 100.0),
    ])
    result = label_one(bars, decision_idx=0, target_pct=0.08, stop_pct=0.04, horizon=5)

    assert result.label == -1
    assert result.exit_reason == "both"
    assert result.holding_days == 1


def test_entry_is_next_day_open_not_decision_close() -> None:
    """
    進場價必須是 T+1 開盤，不是 T 日收盤（CLAUDE.md 核心規則）。

    D0 收盤 100.0，D1 開盤 **95.0**（跳空）
    若誤用 D0 收盤當進場價：上柵 108.0
    正確用 D1 開盤 95.0：    上柵 95 × 1.08 = 102.6、下柵 95 × 0.96 = 91.2

    D2 high 103 → 103 >= 102.6，正確實作會判 +1
    若誤用 100 當 entry，103 < 108 不會觸發，結果不同。
    """
    bars = make_bars([
        ("2026-01-05", 100.0, 100.0, 100.0, 100.0),   # D0 收盤 100
        ("2026-01-06",  95.0,  96.0,  94.0,  95.5),   # D1 跳空開 95
        ("2026-01-07",  95.5, 103.0,  95.0, 102.0),   # D2 high 103
    ])
    result = label_one(bars, decision_idx=0, target_pct=0.08, stop_pct=0.04, horizon=5)

    assert result.entry_price == pytest.approx(95.0), "進場價應為 T+1 開盤"
    assert result.label == 1
    assert result.exit_price == pytest.approx(95.0 * 1.08)


def test_gap_through_barrier_fills_at_open() -> None:
    """
    跳空穿越柵欄時，成交價取開盤價而非柵欄價。

    進場 100.0 → 下柵 96.0
        D1 開盤直接跳到 90.0（跳空跌破下柵）

    保守做法：實際成交會在 90.0（停損單以市價成交），
    取 min(開盤價, 柵欄價) 作為賣出價，才不會高估停損執行品質。
    """
    bars = make_bars([
        ("2026-01-05", 100.0, 100.0, 100.0, 100.0),
        ("2026-01-06", 100.0, 100.0, 100.0, 100.0),   # D1 進場，正常
        ("2026-01-07",  90.0,  91.0,  89.0,  90.5),   # D2 跳空跌破下柵
    ])
    result = label_one(bars, decision_idx=0, target_pct=0.08, stop_pct=0.04, horizon=5)

    assert result.label == -1
    assert result.exit_price == pytest.approx(90.0), "跳空穿越應以開盤價成交，不可用柵欄價"


def test_insufficient_future_bars_returns_none() -> None:
    """
    未來 K 棒不足（決策日太靠近資料尾端）→ 回傳 None，不可猜測。

    只有 D0 與 D1，horizon = 5 但只剩 1 根 → 資訊不足
    """
    bars = make_bars([
        ("2026-01-05", 100.0, 100.0, 100.0, 100.0),
        ("2026-01-06", 100.0, 101.0,  99.0, 100.5),
    ])
    result = label_one(bars, decision_idx=0, target_pct=0.08, stop_pct=0.04, horizon=5)
    assert result is None


def test_no_next_bar_returns_none() -> None:
    """決策日是最後一根 K → 無法進場，回傳 None"""
    bars = make_bars([("2026-01-05", 100.0, 100.0, 100.0, 100.0)])
    assert label_one(bars, decision_idx=0, target_pct=0.08, stop_pct=0.04, horizon=5) is None


# ══════════════════════════════════════════════════════════════
# 報酬計算
# ══════════════════════════════════════════════════════════════


def test_return_pct_on_upper_hit() -> None:
    """
    手算：entry 100.0、exit 108.0
        報酬率 = 108 / 100 − 1 = 0.08
    注意：這是「毛報酬」，成本由回測層扣（禁令 3 的職責分離）
    """
    bars = make_bars([
        ("2026-01-05", 100.0, 100.0, 100.0, 100.0),
        ("2026-01-06", 100.0, 100.0, 100.0, 100.0),
        ("2026-01-07", 100.0, 109.0, 100.0, 108.5),
    ])
    result = label_one(bars, decision_idx=0, target_pct=0.08, stop_pct=0.04, horizon=5)
    assert result.gross_return == pytest.approx(0.08)


def test_return_pct_on_lower_hit() -> None:
    """
    手算：entry 100.0、exit 96.0
        報酬率 = 96 / 100 − 1 = −0.04
    """
    bars = make_bars([
        ("2026-01-05", 100.0, 100.0, 100.0, 100.0),
        ("2026-01-06", 100.0, 100.0, 100.0, 100.0),
        ("2026-01-07", 100.0, 100.0,  95.0,  96.0),
    ])
    result = label_one(bars, decision_idx=0, target_pct=0.08, stop_pct=0.04, horizon=5)
    assert result.gross_return == pytest.approx(-0.04)


# ══════════════════════════════════════════════════════════════
# label_series：批次標記
# ══════════════════════════════════════════════════════════════


def test_series_labels_every_feasible_decision_day() -> None:
    """
    批次標記：只有「未來 K 棒足夠」的決策日會被標記。

    10 根 K（idx 0..9）、horizon = 3、K 線全程在柵欄內（不會提前觸柵）：
        決策日 i 需要進場 idx i+1，時間柵走到 idx i+3
        i+3 <= 9  →  i <= 6
        idx 0..6 可標記 = 7 筆；idx 7..9 未來不足且未觸柵 → 留空
    預期 7 筆
    """
    rows = [
        (f"2026-01-{day:02d}", 100.0, 101.0, 99.0, 100.0)
        for day in range(5, 15)
    ]
    bars = make_bars(rows)
    labeled = label_series(bars, target_pct=0.08, stop_pct=0.04, horizon=3)

    assert len(labeled) == 7
    assert list(labeled.columns) == [
        "label",
        "entry_price",
        "exit_price",
        "gross_return",
        "holding_days",
        "exit_reason",
        "entry_date",
        "exit_date",
    ]


def test_series_index_is_decision_date_not_entry_date() -> None:
    """
    索引必須是**決策日**，不是進場日。

    這點關鍵：特徵是在決策日算的，label 要對齊決策日才能訓練。
    對齊錯就等於 look-ahead（禁令 1）。
    """
    rows = [
        (f"2026-01-{day:02d}", 100.0, 101.0, 99.0, 100.0)
        for day in range(5, 15)
    ]
    bars = make_bars(rows)
    labeled = label_series(bars, target_pct=0.08, stop_pct=0.04, horizon=3)

    assert labeled.index[0] == pd.Timestamp("2026-01-05")
    # 第一筆的進場日應為隔一根 K
    assert labeled.iloc[0]["entry_date"] == pd.Timestamp("2026-01-06")


def test_series_label_distribution_has_all_three_classes() -> None:
    """
    合成一段有漲有跌的資料，三類標籤都要出現。

    若只出現一類，代表柵欄寬度設定與資料波動不匹配，
    訓練會退化成常數預測。
    """
    rng = np.random.default_rng(42)
    price = 100.0
    rows = []
    for i in range(200):
        drift = 0.004 if i % 40 < 20 else -0.004      # 交替趨勢
        ret = drift + rng.normal(0, 0.02)
        new_price = price * (1 + ret)
        high = max(price, new_price) * 1.01
        low = min(price, new_price) * 0.99
        rows.append((f"2026-{1 + i // 28:02d}-{1 + i % 28:02d}", price, high, low, new_price))
        price = new_price

    bars = make_bars(rows)
    labeled = label_series(bars, target_pct=0.05, stop_pct=0.025, horizon=5)
    counts = labeled["label"].value_counts()

    assert set(counts.index) == {-1, 0, 1}, f"標籤分布不完整：{counts.to_dict()}"
    assert counts.min() >= 5, f"某類樣本過少：{counts.to_dict()}"


def test_series_does_not_mutate_input() -> None:
    """
    不可變性（CLAUDE.md 程式風格 / AGENTS.md 檢查項 21）。

    pandas 3.0 的 copy-on-write 會讓就地改寫直接壞掉，
    qlib-tw-trader 的 `_feature_selection()` 就是這樣炸的。
    """
    rows = [
        (f"2026-01-{day:02d}", 100.0, 101.0, 99.0, 100.0)
        for day in range(5, 15)
    ]
    bars = make_bars(rows)
    before = bars.copy(deep=True)

    label_series(bars, target_pct=0.08, stop_pct=0.04, horizon=3)

    pd.testing.assert_frame_equal(bars, before)


def test_series_validates_required_columns() -> None:
    """系統邊界驗證：缺少 OHLC 欄位必須明確報錯"""
    bad = pd.DataFrame(
        {"close": [100.0, 101.0]},
        index=pd.to_datetime(["2026-01-05", "2026-01-06"]),
    )
    with pytest.raises(ValueError, match="缺少必要欄位"):
        label_series(bad, target_pct=0.08, stop_pct=0.04, horizon=3)


def test_series_rejects_unsorted_index() -> None:
    """索引未排序會讓觸柵順序判斷錯誤，必須拒絕"""
    bars = make_bars([
        ("2026-01-06", 100.0, 101.0, 99.0, 100.0),
        ("2026-01-05", 100.0, 101.0, 99.0, 100.0),
    ])
    unsorted = bars.sort_index(ascending=False)
    with pytest.raises(ValueError, match="必須依日期升冪排序"):
        label_series(unsorted, target_pct=0.08, stop_pct=0.04, horizon=1)


# ══════════════════════════════════════════════════════════════
# 反 look-ahead：標記只能用決策日之後的價格
# ══════════════════════════════════════════════════════════════


def test_label_ignores_bars_beyond_horizon() -> None:
    """
    horizon 之外的 K 棒不得影響標記。

    horizon = 2，進場 100.0 → 上柵 108.0
        D1 high 101（未觸）
        D2 high 102（未觸）→ 時間柵到期，label 0
        D3 high 120（超出 horizon，**不可**讓它把 label 變成 +1）
    """
    bars = make_bars([
        ("2026-01-05", 100.0, 100.0, 100.0, 100.0),
        ("2026-01-06", 100.0, 101.0,  99.0, 100.0),
        ("2026-01-07", 100.0, 102.0,  99.0, 101.0),
        ("2026-01-08", 101.0, 120.0, 100.0, 119.0),   # horizon 外
    ])
    result = label_one(bars, decision_idx=0, target_pct=0.08, stop_pct=0.04, horizon=2)

    assert result.label == 0
    assert result.exit_reason == "time"
    assert result.exit_date == pd.Timestamp("2026-01-07")


def test_decision_day_own_prices_do_not_trigger_barrier() -> None:
    """
    決策日自己的 high/low 不可觸柵。

    決策日 D0 的 high 120（早就超過上柵），但進場是 D1，
    D0 的價格屬於「過去」，不構成出場事件。
        D1 開盤 100 → 上柵 108
        D1/D2 都在區間內 → label 0
    """
    bars = make_bars([
        ("2026-01-05", 100.0, 120.0,  80.0, 100.0),   # D0 振幅極大
        ("2026-01-06", 100.0, 101.0,  99.0, 100.0),
        ("2026-01-07", 100.0, 102.0,  98.0, 101.0),
    ])
    result = label_one(bars, decision_idx=0, target_pct=0.08, stop_pct=0.04, horizon=2)
    assert result.label == 0


# ══════════════════════════════════════════════════════════════
# BarrierLabel 結構
# ══════════════════════════════════════════════════════════════


def test_barrier_label_is_immutable() -> None:
    """標記結果不可變，避免下游誤改"""
    bars = make_bars([
        ("2026-01-05", 100.0, 100.0, 100.0, 100.0),
        ("2026-01-06", 100.0, 100.0, 100.0, 100.0),
        ("2026-01-07", 100.0, 109.0, 100.0, 108.5),
    ])
    result = label_one(bars, decision_idx=0, target_pct=0.08, stop_pct=0.04, horizon=5)
    with pytest.raises(Exception):
        result.label = 0  # type: ignore[misc]
