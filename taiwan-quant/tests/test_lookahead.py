#!/usr/bin/env python3
"""
Look-ahead 物理截斷掃描器測試

CLAUDE.md 規格 16（來自 qlib-tw-trader 驗證的實測教訓）：

    look-ahead 掃描器不能用 qlib 的 `end_time` 做截斷，必須物理重新
    匯出只到 T 的資料集。

    實測：`D.features(["2330"], ["Ref($close,-3)/Ref($close,-1)-1"],
          end_time="2026-06-30")` 仍算得出 −0.023952...，與不截斷完全相同。
    `end_time` 只裁切輸出範圍，運算式仍會讀底層儲存的未來資料。

本掃描器的做法：
    1. 用完整資料算一次特徵
    2. 用 `bars.iloc[:i+1]`（**物理切掉**未來列）再算一次
    3. 比對第 i 列的值。只用過去資料的特徵必須完全相同

正控制組（步驟①的教訓）：
    沒有正控制組時，「全部通過」可能只是掃描器壞了。
    每次掃描都必須附帶一個「故意偷看未來」的特徵，確認它被抓到。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from taiwan_quant.validation.lookahead import (
    CHEATING_FEATURES,
    LookaheadError,
    ScanResult,
    TruncationReport,
    scan_builder,
    scan_features,
)

pytestmark = pytest.mark.unit


def make_bars(n: int = 60, seed: int = 42) -> pd.DataFrame:
    """產生有波動的日 K，避免特徵退化成常數而掩蓋洩漏"""
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.02, n))
    high = close * (1 + rng.uniform(0.001, 0.02, n))
    low = close * (1 - rng.uniform(0.001, 0.02, n))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    index = pd.date_range("2026-01-01", periods=n, freq="B", name="date")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close,
         "volume": rng.integers(1000, 9999, n)},
        index=index,
    )


# ── 受測特徵 ──────────────────────────────────────────────


def ma20(bars: pd.DataFrame) -> pd.Series:
    """合法：只用過去 20 天"""
    return bars["close"].rolling(20, min_periods=20).mean()


def ret1(bars: pd.DataFrame) -> pd.Series:
    """合法：昨收到今收"""
    return bars["close"].pct_change()


def zscore_whole_sample(bars: pd.DataFrame) -> pd.Series:
    """
    洩漏：用**整段樣本**的平均與標準差做標準化。

    這是最隱蔽的一種——程式裡沒有任何負數 shift，
    但 `mean()` / `std()` 吃了未來資料，所以每次資料變長，
    歷史上的值都會改變。靜態掃描抓不到，只有截斷測試抓得到。
    """
    close = bars["close"]
    return (close - close.mean()) / close.std()


def future_return(bars: pd.DataFrame) -> pd.Series:
    """洩漏：明天的報酬（明確作弊）"""
    return bars["close"].shift(-1) / bars["close"] - 1


def expanding_max(bars: pd.DataFrame) -> pd.Series:
    """合法：expanding 只看到目前為止"""
    return bars["close"].expanding().max()


def backward_fill_leak(bars: pd.DataFrame) -> pd.Series:
    """
    洩漏：用 bfill 補值等於把未來的值搬到過去。

    真實情境：籌碼資料有洞時用 bfill 補，看起來無害，實際是洩漏。
    """
    with_holes = bars["close"].copy()
    with_holes.iloc[::5] = np.nan
    return with_holes.bfill()


# ══════════════════════════════════════════════════════════════
# 合法特徵必須通過
# ══════════════════════════════════════════════════════════════


def test_rolling_mean_is_truncation_invariant() -> None:
    report = scan_features({"ma20": ma20}, make_bars())["ma20"]
    assert report.is_clean, report.describe()
    assert report.cuts_tested > 0


def test_pct_change_is_truncation_invariant() -> None:
    assert scan_features({"ret1": ret1}, make_bars())["ret1"].is_clean


def test_expanding_max_is_truncation_invariant() -> None:
    """expanding 只看到目前為止，是合法的"""
    assert scan_features({"expanding_max": expanding_max}, make_bars())["expanding_max"].is_clean


# ══════════════════════════════════════════════════════════════
# 洩漏特徵必須被抓到
# ══════════════════════════════════════════════════════════════


def test_detects_whole_sample_zscore() -> None:
    """
    全樣本標準化必須被抓到。

    這是本掃描器存在的主要理由——靜態掃描（找負數 shift）看不到它，
    因為程式裡沒有任何未來參照語法。
    """
    report = scan_features({"leak": zscore_whole_sample}, make_bars())["leak"]
    assert not report.is_clean, "全樣本標準化未被偵測"
    assert len(report.findings) > 0


def test_detects_explicit_future_shift() -> None:
    report = scan_features({"leak": future_return}, make_bars())["leak"]
    assert not report.is_clean


def test_detects_backward_fill() -> None:
    """bfill 把未來值搬到過去，必須被抓到"""
    report = scan_features({"leak": backward_fill_leak}, make_bars())["leak"]
    assert not report.is_clean


def test_finding_records_both_values() -> None:
    """findings 要記下完整值與截斷值，方便除錯"""
    report = scan_features({"leak": future_return}, make_bars())["leak"]
    finding = report.findings[0]
    assert finding.feature == "leak"
    assert finding.full_value != finding.truncated_value or (
        pd.isna(finding.full_value) != pd.isna(finding.truncated_value)
    )


# ══════════════════════════════════════════════════════════════
# NaN 處理（實測教訓：NaN 比較會靜默通過）
# ══════════════════════════════════════════════════════════════


def test_both_nan_counts_as_equal() -> None:
    """
    兩邊都是 NaN（視窗不足）→ 視為相同，不算 finding。

    ma20 在前 19 列必為 NaN，這是正常的。
    """
    bars = make_bars(30)
    report = scan_features({"ma20": ma20}, bars, cut_indices=[5, 10])["ma20"]
    assert report.is_clean


def test_one_side_nan_is_a_finding() -> None:
    """
    一邊 NaN 一邊有值 → **必須**算 finding。

    這正是洩漏的典型徵狀：完整資料算得出來，截斷後算不出來。
    絕不可因為 `nan == nan` 為 False 而漏判，也不可因為
    `nan != value` 的比較不拋錯就當成相同。
    """
    report = scan_features({"leak": future_return}, make_bars())["leak"]
    nan_findings = [
        f for f in report.findings
        if pd.isna(f.truncated_value) and not pd.isna(f.full_value)
    ]
    assert nan_findings, "未捕捉到『完整有值、截斷為 NaN』的洩漏"


# ══════════════════════════════════════════════════════════════
# 正控制組：掃描器自我驗證
# ══════════════════════════════════════════════════════════════


def test_scan_builder_runs_positive_control_by_default() -> None:
    """
    `scan_builder` 預設會跑正控制組，證明自己有鑑別力。

    步驟①的教訓：沒有正控制組時，「全部通過」可能只是掃描器壞了。
    """
    def builder(bars: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"ma20": ma20(bars), "ret1": ret1(bars)})

    result = scan_builder(builder, make_bars())
    assert result.positive_control_passed is True
    assert result.is_clean


def test_scan_builder_reports_leaking_column() -> None:
    def builder(bars: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"ok": ma20(bars), "bad": zscore_whole_sample(bars)})

    result = scan_builder(builder, make_bars())
    assert not result.is_clean
    assert result.leaking_features == ["bad"]
    assert result.clean_features == ["ok"]


def test_scan_builder_raises_when_positive_control_fails() -> None:
    """
    正控制組失敗代表掃描器本身沒鑑別力，此時必須拋錯，
    **不可**回報「全部通過」——那是最危險的假通過。
    """
    bars = make_bars(3)   # 太短，截斷測試無法產生有效切點
    def builder(b: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"ret1": ret1(b)})

    with pytest.raises(LookaheadError, match="正控制組"):
        scan_builder(builder, bars, cut_indices=[])


def test_cheating_features_catalog_is_all_detected() -> None:
    """內建的作弊特徵目錄每一個都必須被抓到，否則掃描器有盲點"""
    bars = make_bars()
    for name, fn in CHEATING_FEATURES.items():
        report = scan_features({name: fn}, bars)[name]
        assert not report.is_clean, f"作弊特徵 {name} 未被偵測"


# ══════════════════════════════════════════════════════════════
# 切點選擇
# ══════════════════════════════════════════════════════════════


def test_default_cut_indices_are_spread_out() -> None:
    """
    預設切點要分散在資料後段。

    只切一個點可能剛好躲過洩漏；只切最後一點則截斷等於沒截斷。
    """
    bars = make_bars(100)
    report = scan_features({"ma20": ma20}, bars)["ma20"]
    assert report.cuts_tested >= 3
    assert len(set(report.cut_dates)) == report.cuts_tested


def test_explicit_cut_indices_respected() -> None:
    bars = make_bars(60)
    report = scan_features({"ma20": ma20}, bars, cut_indices=[30, 40])["ma20"]
    assert report.cuts_tested == 2
    assert list(report.cut_dates) == [bars.index[30], bars.index[40]]


def test_rejects_out_of_range_cut_index() -> None:
    bars = make_bars(30)
    with pytest.raises(ValueError, match="切點"):
        scan_features({"ma20": ma20}, bars, cut_indices=[99])


def test_rejects_last_index_as_cut() -> None:
    """
    最後一列當切點沒有意義——截斷後與完整資料相同，永遠通過。
    允許它會製造假的安全感。
    """
    bars = make_bars(30)
    with pytest.raises(ValueError, match="切點"):
        scan_features({"ma20": ma20}, bars, cut_indices=[29])


# ══════════════════════════════════════════════════════════════
# 輸入驗證與不可變性
# ══════════════════════════════════════════════════════════════


def test_does_not_mutate_input_bars() -> None:
    bars = make_bars()
    before = bars.copy(deep=True)
    scan_features({"ma20": ma20, "leak": future_return}, bars)
    pd.testing.assert_frame_equal(bars, before)


def test_validates_required_columns() -> None:
    bad = pd.DataFrame({"close": [1.0, 2.0]}, index=pd.to_datetime(["2026-01-05", "2026-01-06"]))
    with pytest.raises(ValueError, match="缺少必要欄位"):
        scan_features({"ma20": ma20}, bad)


def test_rejects_unsorted_index() -> None:
    bars = make_bars(30).sort_index(ascending=False)
    with pytest.raises(ValueError, match="升冪"):
        scan_features({"ma20": ma20}, bars)


def test_report_is_immutable() -> None:
    report = scan_features({"ma20": ma20}, make_bars())["ma20"]
    with pytest.raises(Exception):
        report.feature = "x"  # type: ignore[misc]


def test_result_describe_is_human_readable() -> None:
    def builder(bars: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"ok": ma20(bars), "bad": future_return(bars)})

    text = scan_builder(builder, make_bars()).describe()
    assert "bad" in text
    assert "ok" in text
    assert "正控制組" in text


def test_tolerance_allows_float_noise() -> None:
    """
    浮點運算順序不同會有 1e-16 級誤差，不該當成洩漏。

    構造：合法特徵加上一個極小的、與資料長度無關的擾動。
    """
    def almost_ma20(bars: pd.DataFrame) -> pd.Series:
        return ma20(bars) + 1e-15

    assert scan_features({"f": almost_ma20}, make_bars())["f"].is_clean


def test_tiny_but_systematic_difference_is_caught() -> None:
    """
    誤差雖小但**隨資料長度變化**時必須被抓到。

    構造：加上 `len(bars) * 1e-6`，每次截斷長度不同 → 值就不同。
    這是真洩漏，不是浮點噪音。
    """
    def length_dependent(bars: pd.DataFrame) -> pd.Series:
        return ma20(bars) + len(bars) * 1e-6

    assert not scan_features({"f": length_dependent}, make_bars())["f"].is_clean
