#!/usr/bin/env python3
"""
移動停損標記測試（路線 A）

## 為什麼要換掉 triple-barrier

樣本外驗證結論（`docs/需求規劃/202609/03_策略可行性最終結論.md`）：

    9 組策略全部輸給等權買進持有，差距達 85 個百分點。

機制原因很明確：**triple-barrier 的目標價把上檔封死了。**

    目標 +19.6%（40 日）→ 碰到就出場
    但這 14 個月市場漲了 262%

每次觸及目標就出場，就錯過後續續漲。強勢多頭裡，任何「設定目標價就走」
的策略都會系統性輸給單純持有。

## 路線 A 的解法

拿掉上柵，改成**移動停損**：

    初始停損 = 進場價 × (1 − trail_pct)
    每天更新 = max(現有停損, 期間最高價 × (1 − trail_pct))   ← 只升不降
    出場     = 盤中觸及停損線，或時間柵到期

上檔不封死 → 趨勢延續時能一路跟上；回檔超過 trail_pct → 鎖住已實現獲利。

這正是 Kimi 的建議（見 `來源原文/06_Kimi`）：

    改採「拉回分批買 + 紀律停損」的波段計畫即可。

## 與 triple-barrier 的共通點

進場仍是 **T+1 開盤**、仍用 high/low 判定觸發、跳空仍以開盤價成交、
資料不足仍回 None。這些規則不因為換標記方式而改變。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from taiwan_quant.labeling.trailing_stop import (
    TrailingExit,
    TrailingSpec,
    label_trailing,
    label_trailing_series,
)

pytestmark = pytest.mark.unit


def make_bars(rows: list[tuple[str, float, float, float, float]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close"])
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()


# ══════════════════════════════════════════════════════════════
# TrailingSpec
# ══════════════════════════════════════════════════════════════


def test_spec_initial_stop() -> None:
    """
    手算：進場 100、trail 5%
        初始停損 = 100 × 0.95 = 95.0
    """
    spec = TrailingSpec(entry_price=100.0, trail_pct=0.05, max_horizon=60)
    assert spec.initial_stop == pytest.approx(95.0)


def test_spec_stop_from_peak() -> None:
    """
    手算：最高價 120、trail 5%
        停損 = 120 × 0.95 = 114.0
    """
    spec = TrailingSpec(entry_price=100.0, trail_pct=0.05, max_horizon=60)
    assert spec.stop_from_peak(120.0) == pytest.approx(114.0)


def test_spec_rejects_bad_values() -> None:
    with pytest.raises(ValueError, match="entry_price"):
        TrailingSpec(entry_price=0.0, trail_pct=0.05, max_horizon=60)
    with pytest.raises(ValueError, match="trail_pct"):
        TrailingSpec(entry_price=100.0, trail_pct=0.0, max_horizon=60)
    with pytest.raises(ValueError, match="trail_pct"):
        TrailingSpec(entry_price=100.0, trail_pct=1.0, max_horizon=60)
    with pytest.raises(ValueError, match="max_horizon"):
        TrailingSpec(entry_price=100.0, trail_pct=0.05, max_horizon=0)


# ══════════════════════════════════════════════════════════════
# 核心行為：上檔不封死
# ══════════════════════════════════════════════════════════════


def test_upside_is_not_capped() -> None:
    """
    **這是路線 A 的全部重點。**

    構造：進場 100，之後連漲到 200，最後一天回檔觸及停損。
        trail 10% → 最高 200 時停損 = 180
        最後一天 low 175 → 觸發，出場 180

    triple-barrier 會在目標價（例如 +20% = 120）就出場，
    只拿到 20%；移動停損拿到 80%。
    """
    rows = [("2026-01-01", 100.0, 100.0, 100.0, 100.0)]   # D0 決策日
    price = 100.0
    for i in range(1, 21):                                  # D1~D20 連漲
        price *= 1.05
        rows.append((f"2026-02-{i:02d}", price, price, price * 0.99, price))
    # 最後一天回檔，觸及停損
    peak = price
    rows.append(("2026-03-01", peak, peak, peak * 0.85, peak * 0.86))

    result = label_trailing(make_bars(rows), decision_idx=0, trail_pct=0.10, max_horizon=60)

    assert result is not None
    assert result.exit_reason == "trailing_stop"
    assert result.gross_return > 0.5, (
        f"上檔被封死了：報酬僅 {result.gross_return:.2%}"
    )


def test_stop_ratchets_up_only() -> None:
    """
    停損只升不降。

    構造：進場 100 → 漲到 120 → 回落到 110（未觸停損 108）→ 再漲
        若停損跟著回落到 110×0.9 = 99，就失去保護已實現獲利的意義
    """
    rows = [
        ("2026-01-01", 100.0, 100.0, 100.0, 100.0),   # D0
        ("2026-01-02", 100.0, 100.0, 100.0, 100.0),   # D1 進場
        ("2026-01-03", 100.0, 120.0, 100.0, 120.0),   # 最高 120 → 停損 108
        ("2026-01-04", 120.0, 120.0, 110.0, 110.0),   # 回落但未觸 108
        ("2026-01-05", 110.0, 110.0, 107.0, 107.5),   # low 107 < 108 → 觸發
    ]
    result = label_trailing(make_bars(rows), decision_idx=0, trail_pct=0.10, max_horizon=60)

    assert result.exit_reason == "trailing_stop"
    assert result.exit_price == pytest.approx(108.0)
    assert result.highest_price == pytest.approx(120.0)


def test_initial_stop_applies_before_any_gain() -> None:
    """
    進場後立刻下跌，用初始停損（進場價 × (1 − trail)）。

    手算：進場 100、trail 5% → 停損 95
        D2 low 94 → 觸發，出場 95，報酬 −5%
    """
    rows = [
        ("2026-01-01", 100.0, 100.0, 100.0, 100.0),
        ("2026-01-02", 100.0, 100.0, 100.0, 100.0),   # 進場
        ("2026-01-03", 100.0, 100.0, 94.0, 95.0),
    ]
    result = label_trailing(make_bars(rows), decision_idx=0, trail_pct=0.05, max_horizon=60)

    assert result.exit_price == pytest.approx(95.0)
    assert result.gross_return == pytest.approx(-0.05)


# ══════════════════════════════════════════════════════════════
# 時間柵
# ══════════════════════════════════════════════════════════════


def test_time_barrier_exits_at_close() -> None:
    """
    時間柵到期未觸停損 → 以最後一根 K 收盤出場。

    手算：進場 100、max_horizon 3、全程在停損之上
        D3 收盤 108 → 報酬 +8%
    """
    rows = [
        ("2026-01-01", 100.0, 100.0, 100.0, 100.0),
        ("2026-01-02", 100.0, 102.0, 99.0, 101.0),
        ("2026-01-03", 101.0, 105.0, 100.0, 104.0),
        ("2026-01-04", 104.0, 109.0, 103.0, 108.0),
        ("2026-01-05", 108.0, 200.0, 50.0, 150.0),    # 超出 horizon，不看
    ]
    result = label_trailing(make_bars(rows), decision_idx=0, trail_pct=0.10, max_horizon=3)

    assert result.exit_reason == "time"
    assert result.exit_price == pytest.approx(108.0)
    assert result.gross_return == pytest.approx(0.08)
    assert result.holding_days == 3


def test_time_barrier_does_not_see_beyond_horizon() -> None:
    """horizon 之外的 K 棒不得影響結果"""
    rows = [
        ("2026-01-01", 100.0, 100.0, 100.0, 100.0),
        ("2026-01-02", 100.0, 101.0, 99.0, 100.0),
        ("2026-01-03", 100.0, 101.0, 99.0, 100.0),
        ("2026-01-04", 100.0, 100.0, 1.0, 5.0),       # horizon 外的崩盤
    ]
    result = label_trailing(make_bars(rows), decision_idx=0, trail_pct=0.10, max_horizon=2)
    assert result.exit_reason == "time"
    assert result.gross_return == pytest.approx(0.0)


# ══════════════════════════════════════════════════════════════
# 與 triple-barrier 共通的規則
# ══════════════════════════════════════════════════════════════


def test_entry_is_next_day_open() -> None:
    """
    進場價必須是 T+1 開盤，不是 T 日收盤。

    構造：D0 收盤 100、D1 跳空開 90
        正確 entry = 90 → 停損 = 90 × 0.95 = 85.5
        若誤用 100 → 停損 95，D2 low 88 會誤判觸發
    """
    rows = [
        ("2026-01-01", 100.0, 100.0, 100.0, 100.0),
        ("2026-01-02", 90.0, 92.0, 89.0, 91.0),        # 跳空開 90
        ("2026-01-03", 91.0, 93.0, 88.0, 92.0),        # low 88 > 85.5，不觸發
        ("2026-01-04", 92.0, 94.0, 91.0, 93.0),
    ]
    result = label_trailing(make_bars(rows), decision_idx=0, trail_pct=0.05, max_horizon=3)

    assert result.entry_price == pytest.approx(90.0)
    assert result.exit_reason == "time", "誤用 D0 收盤當進場價會提早觸發停損"


def test_gap_below_stop_fills_at_open() -> None:
    """
    跳空跌破停損線時以開盤價成交，不是停損價。

    構造：進場 100、停損 95，D2 開盤直接跳到 88
        用停損價 95 → 高估執行品質（市場沒有 95 的價格）
        用開盤價 88 → 實際成交價
    """
    rows = [
        ("2026-01-01", 100.0, 100.0, 100.0, 100.0),
        ("2026-01-02", 100.0, 100.0, 100.0, 100.0),
        ("2026-01-03", 88.0, 89.0, 87.0, 88.5),
    ]
    result = label_trailing(make_bars(rows), decision_idx=0, trail_pct=0.05, max_horizon=60)

    assert result.exit_price == pytest.approx(88.0)
    assert result.gross_return == pytest.approx(-0.12)


def test_decision_day_prices_do_not_trigger() -> None:
    """決策日自己的價格屬於過去，不構成出場事件"""
    rows = [
        ("2026-01-01", 100.0, 120.0, 50.0, 100.0),    # D0 振幅極大
        ("2026-01-02", 100.0, 101.0, 99.0, 100.0),
        ("2026-01-03", 100.0, 101.0, 99.0, 100.0),
    ]
    result = label_trailing(make_bars(rows), decision_idx=0, trail_pct=0.05, max_horizon=2)
    assert result.exit_reason == "time"


def test_insufficient_future_bars_returns_none() -> None:
    """
    未觸停損且可得 K 棒少於 horizon → 回 None。

    結果取決於還沒發生的交易日，不可猜測。
    """
    rows = [
        ("2026-01-01", 100.0, 100.0, 100.0, 100.0),
        ("2026-01-02", 100.0, 101.0, 99.0, 100.0),
    ]
    assert label_trailing(make_bars(rows), decision_idx=0, trail_pct=0.05, max_horizon=60) is None


def test_stop_hit_within_partial_window_is_determinate() -> None:
    """
    停損已在可得範圍內觸及 → 結果確定，不受後面缺幾根影響
    """
    rows = [
        ("2026-01-01", 100.0, 100.0, 100.0, 100.0),
        ("2026-01-02", 100.0, 100.0, 100.0, 100.0),
        ("2026-01-03", 100.0, 100.0, 90.0, 91.0),      # 觸停損 95
    ]
    result = label_trailing(make_bars(rows), decision_idx=0, trail_pct=0.05, max_horizon=60)
    assert result is not None
    assert result.exit_reason == "trailing_stop"


def test_no_next_bar_returns_none() -> None:
    rows = [("2026-01-01", 100.0, 100.0, 100.0, 100.0)]
    assert label_trailing(make_bars(rows), decision_idx=0, trail_pct=0.05, max_horizon=60) is None


# ══════════════════════════════════════════════════════════════
# 結果結構
# ══════════════════════════════════════════════════════════════


def test_result_is_immutable() -> None:
    rows = [
        ("2026-01-01", 100.0, 100.0, 100.0, 100.0),
        ("2026-01-02", 100.0, 100.0, 100.0, 100.0),
        ("2026-01-03", 100.0, 100.0, 90.0, 91.0),
    ]
    result = label_trailing(make_bars(rows), decision_idx=0, trail_pct=0.05, max_horizon=60)
    with pytest.raises(Exception):
        result.gross_return = 0.0  # type: ignore[misc]


def test_result_records_peak_for_audit() -> None:
    """
    要記錄期間最高價——否則無法回答「當時停損為什麼設在那個價位」。
    """
    rows = [
        ("2026-01-01", 100.0, 100.0, 100.0, 100.0),
        ("2026-01-02", 100.0, 100.0, 100.0, 100.0),
        ("2026-01-03", 100.0, 130.0, 100.0, 125.0),
        ("2026-01-04", 125.0, 125.0, 110.0, 112.0),   # 觸 130×0.9 = 117
    ]
    result = label_trailing(make_bars(rows), decision_idx=0, trail_pct=0.10, max_horizon=60)
    assert result.highest_price == pytest.approx(130.0)
    assert result.exit_price == pytest.approx(117.0)


# ══════════════════════════════════════════════════════════════
# 批次標記
# ══════════════════════════════════════════════════════════════


def test_series_index_is_decision_date() -> None:
    """索引必須是決策日，與 triple_barrier 一致"""
    rows = [
        (f"2026-01-{d:02d}", 100.0, 101.0, 99.0, 100.0)
        for d in range(1, 21)
    ]
    labeled = label_trailing_series(make_bars(rows), trail_pct=0.05, max_horizon=3)

    assert labeled.index[0] == pd.Timestamp("2026-01-01")
    assert labeled.iloc[0]["entry_date"] == pd.Timestamp("2026-01-02")


def test_series_columns() -> None:
    rows = [
        (f"2026-01-{d:02d}", 100.0, 101.0, 99.0, 100.0)
        for d in range(1, 21)
    ]
    labeled = label_trailing_series(make_bars(rows), trail_pct=0.05, max_horizon=3)
    assert list(labeled.columns) == [
        "label", "entry_price", "exit_price", "gross_return",
        "highest_price", "holding_days", "exit_reason",
        "entry_date", "exit_date",
    ]


def test_series_label_is_sign_of_return() -> None:
    """
    標籤是報酬的正負號。

    移動停損沒有固定目標，所以不是「先碰哪道柵欄」的三分類，
    而是「這筆賺還是賠」。報酬為 0 視為未獲利（保守）。
    """
    rows = [
        ("2026-01-01", 100.0, 100.0, 100.0, 100.0),
        ("2026-01-02", 100.0, 100.0, 100.0, 100.0),
        ("2026-01-03", 100.0, 130.0, 100.0, 125.0),
        ("2026-01-04", 125.0, 125.0, 110.0, 112.0),
    ]
    labeled = label_trailing_series(make_bars(rows), trail_pct=0.10, max_horizon=2)
    assert labeled.iloc[0]["label"] == 1


def test_series_does_not_mutate_input() -> None:
    rows = [
        (f"2026-01-{d:02d}", 100.0, 101.0, 99.0, 100.0)
        for d in range(1, 21)
    ]
    bars = make_bars(rows)
    before = bars.copy(deep=True)
    label_trailing_series(bars, trail_pct=0.05, max_horizon=3)
    pd.testing.assert_frame_equal(bars, before)


def test_series_validates_columns() -> None:
    bad = pd.DataFrame(
        {"close": [1.0, 2.0]}, index=pd.to_datetime(["2026-01-01", "2026-01-02"])
    )
    with pytest.raises(ValueError, match="缺少必要欄位"):
        label_trailing_series(bad, trail_pct=0.05, max_horizon=3)


def test_series_rejects_unsorted() -> None:
    rows = [
        ("2026-01-02", 100.0, 101.0, 99.0, 100.0),
        ("2026-01-01", 100.0, 101.0, 99.0, 100.0),
    ]
    bars = make_bars(rows).sort_index(ascending=False)
    with pytest.raises(ValueError, match="升冪"):
        label_trailing_series(bars, trail_pct=0.05, max_horizon=1)


# ══════════════════════════════════════════════════════════════
# 與 triple-barrier 的直接對比
# ══════════════════════════════════════════════════════════════


def test_beats_triple_barrier_in_strong_trend() -> None:
    """
    **驗證路線 A 的核心假設。**

    同一段強勢趨勢，移動停損的報酬必須明顯高於固定目標。

    構造：連漲 30 天（每天 +3%），最後回檔觸停損。
        triple-barrier 目標 +20% → 約第 7 天就出場，拿 20%
        移動停損 trail 10% → 一路跟到最後，拿遠超過 20%
    """
    from taiwan_quant.labeling.triple_barrier import label_one

    rows = [("2026-01-01", 100.0, 100.0, 100.0, 100.0)]
    price = 100.0
    for i in range(30):
        price *= 1.03
        rows.append((f"2026-{2 + i // 28:02d}-{1 + i % 28:02d}", price, price, price * 0.995, price))
    rows.append(("2026-04-01", price, price, price * 0.85, price * 0.86))

    bars = make_bars(rows)

    trailing = label_trailing(bars, decision_idx=0, trail_pct=0.10, max_horizon=60)
    fixed = label_one(bars, decision_idx=0, target_pct=0.20, stop_pct=0.10, horizon=60)

    # 固定目標的報酬被鎖在目標價附近。不是精確 20%——構造的資料每天
    # 跳空開高，觸柵當根以開盤價成交（正確的跳空處理），所以略高於目標。
    assert 0.20 <= fixed.gross_return < 0.25, (
        f"固定目標報酬 {fixed.gross_return:.2%} 應被鎖在目標價附近"
    )
    assert trailing.gross_return > fixed.gross_return * 2, (
        f"移動停損 {trailing.gross_return:.2%} 未明顯勝過固定目標 "
        f"{fixed.gross_return:.2%}"
    )
