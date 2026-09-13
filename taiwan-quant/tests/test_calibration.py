#!/usr/bin/env python3
"""
機率校準測試

問題：模型（或規則式基準）輸出的是**分數**，不是機率。

    momentum_score 用 tanh 映射到 [0.2, 0.8]
    LightGBM 的 predict_proba 也常常過度自信或過度保守

把未校準的分數當成 `P(+1)` 拿去跟進場門檻（40.22%）比較，
結論毫無意義——這會讓整個系統看起來有根據，實際上沒有。

解法：用歷史 triple-barrier 標籤做分箱校準。

    分數落在 [0.6, 0.7) 的歷史樣本中，實際有多少比例是 +1？
    → 那個比例才是這一箱的 P(+1)

三條設計原則：

1. **樣本不足的箱回 None**，不可用相鄰箱或整體基準硬補
   （那會讓罕見分數區間看起來有統計依據）
2. **校準必須只用訓練期資料**，不可用全樣本（否則是 look-ahead）
3. **時間柵到期（label 0）算 P(−1)**，與 `Candidate.expected_return`
   的保守假設一致
"""

from __future__ import annotations

import numpy as np
import pytest

from taiwan_quant.validation.calibration import (
    Calibrator,
    CalibrationError,
    fit_calibrator,
    reliability_report,
)

pytestmark = pytest.mark.unit


def make_samples(
    n_per_bin: int,
    hit_rates: dict[tuple[float, float], float],
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    構造已知命中率的樣本。

    hit_rates: {(分數下界, 分數上界): 該區間的實際 +1 比例}

    標籤必須**隨機**指派給該區間內的樣本，不可按位置。
    按位置指派（前 N 筆是 +1）會讓命中率與分數在區間內產生虛假關聯，
    導致分箱後的命中率偏離設定值。
    """
    rng = np.random.default_rng(seed)
    scores: list[float] = []
    labels: list[int] = []

    for (lo, hi), rate in hit_rates.items():
        n_hit = int(round(n_per_bin * rate))
        group_labels = np.array([1] * n_hit + [-1] * (n_per_bin - n_hit))
        rng.shuffle(group_labels)
        scores.extend(rng.uniform(lo, hi, n_per_bin))
        labels.extend(group_labels.tolist())

    return np.asarray(scores), np.asarray(labels)


# ══════════════════════════════════════════════════════════════
# 校準：分數 → 實際機率
# ══════════════════════════════════════════════════════════════


def test_recovers_known_hit_rate() -> None:
    """
    構造：分數 [0.6, 0.7) 的樣本中，實際 30% 是 +1。
    校準後查詢 0.65 應回 0.30（而不是 0.65）。

    這條說明校準在做什麼：分數 0.65 不代表 65% 機率。
    """
    # 2000 筆分 5 箱 = 每箱 400 筆。真實率 0.30 的二項標準差為
    # sqrt(0.3×0.7/400) = 0.023，所以 ±0.05 約為 2 個標準差，不會偶發失敗。
    scores, labels = make_samples(2000, {(0.6, 0.7): 0.30})
    calibrator = fit_calibrator(scores, labels, n_bins=5, min_samples_per_bin=30)

    assert calibrator.predict(0.65) == pytest.approx(0.30, abs=0.05)


def test_monotonic_scores_give_monotonic_probabilities() -> None:
    """分數越高、實際命中率越高時，校準後應保持單調"""
    scores, labels = make_samples(
        200,
        {(0.2, 0.4): 0.10, (0.4, 0.6): 0.30, (0.6, 0.8): 0.55},
    )
    calibrator = fit_calibrator(scores, labels, n_bins=3, min_samples_per_bin=30)

    low = calibrator.predict(0.3)
    mid = calibrator.predict(0.5)
    high = calibrator.predict(0.7)

    assert low < mid < high


def test_uncalibrated_score_is_not_probability() -> None:
    """
    核心情境：`momentum_score` 把分數映射到 [0.2, 0.8]，
    但實際命中率只有 ~27%（triple-barrier 的基礎勝率）。

    未校準時「P = 0.65」會通過 40.22% 門檻；
    校準後真實機率 0.27 應該被擋下。
    """
    scores, labels = make_samples(2000, {(0.55, 0.75): 0.27})
    calibrator = fit_calibrator(scores, labels, n_bins=4, min_samples_per_bin=30)

    calibrated = calibrator.predict(0.65)
    assert calibrated == pytest.approx(0.27, abs=0.05)
    assert calibrated < 0.4022, "校準後應低於進場門檻"


# ══════════════════════════════════════════════════════════════
# 樣本不足：回 None，不猜
# ══════════════════════════════════════════════════════════════


def test_sparse_bin_returns_none() -> None:
    """
    某分數區間樣本太少 → 回 None。

    **不可**用相鄰箱或整體基準硬補——那會讓罕見分數區間
    看起來有統計依據，實際沒有。
    """
    scores = np.concatenate([np.full(200, 0.3), np.full(3, 0.9)])
    labels = np.concatenate([np.full(200, -1), np.full(3, 1)])

    calibrator = fit_calibrator(scores, labels, n_bins=5, min_samples_per_bin=30)

    assert calibrator.predict(0.3) is not None
    assert calibrator.predict(0.9) is None, "樣本不足的箱不可給出機率"


def test_score_outside_training_range_returns_none() -> None:
    """
    分數超出訓練期見過的範圍 → 回 None，不外推。

    外推在金融資料上特別危險：極端分數往往出現在極端行情，
    而那正是歷史關係最可能失效的時候。
    """
    scores, labels = make_samples(200, {(0.4, 0.6): 0.3})
    calibrator = fit_calibrator(scores, labels, n_bins=4, min_samples_per_bin=30)

    assert calibrator.predict(0.05) is None
    assert calibrator.predict(0.99) is None


def test_requires_minimum_total_samples() -> None:
    scores = np.linspace(0.3, 0.7, 20)
    labels = np.where(scores > 0.5, 1, -1)
    with pytest.raises(CalibrationError, match="樣本不足"):
        fit_calibrator(scores, labels, n_bins=5, min_samples_per_bin=30)


# ══════════════════════════════════════════════════════════════
# 標籤處理
# ══════════════════════════════════════════════════════════════


def test_time_barrier_counts_as_not_hit() -> None:
    """
    時間柵到期（label 0）算 P(−1)，與 Candidate.expected_return 一致。

    構造：100 筆全是 label 0 → P(+1) = 0
    """
    scores = np.full(200, 0.5)
    labels = np.zeros(200, dtype=int)
    calibrator = fit_calibrator(scores, labels, n_bins=2, min_samples_per_bin=30)

    assert calibrator.predict(0.5) == pytest.approx(0.0)


def test_mixed_labels_counted_correctly() -> None:
    """
    手算：200 筆中 +1 有 50 筆、0 有 100 筆、−1 有 50 筆
        P(+1) = 50 / 200 = 0.25
    """
    scores = np.full(200, 0.5)
    labels = np.array([1] * 50 + [0] * 100 + [-1] * 50)
    calibrator = fit_calibrator(scores, labels, n_bins=2, min_samples_per_bin=30)

    assert calibrator.predict(0.5) == pytest.approx(0.25)


def test_rejects_invalid_labels() -> None:
    scores = np.full(100, 0.5)
    labels = np.full(100, 2)
    with pytest.raises(CalibrationError, match="標籤"):
        fit_calibrator(scores, labels, n_bins=2, min_samples_per_bin=10)


def test_rejects_mismatched_lengths() -> None:
    with pytest.raises(CalibrationError, match="長度"):
        fit_calibrator(np.zeros(10), np.zeros(5), n_bins=2, min_samples_per_bin=1)


def test_rejects_non_finite_scores() -> None:
    """NaN 分數會讓 digitize 給出無意義的箱號且不拋錯"""
    scores = np.array([0.5] * 99 + [np.nan])
    labels = np.zeros(100, dtype=int)
    with pytest.raises(CalibrationError, match="有限值"):
        fit_calibrator(scores, labels, n_bins=2, min_samples_per_bin=10)


# ══════════════════════════════════════════════════════════════
# 基礎資訊
# ══════════════════════════════════════════════════════════════


def test_reports_base_rate() -> None:
    """
    基礎勝率（整體 P(+1)）要能查得到——模型必須勝過它才有價值。

    手算：300 筆中 +1 有 90 筆 → 0.30
    """
    scores = np.linspace(0.3, 0.7, 300)
    labels = np.array([1] * 90 + [-1] * 210)
    calibrator = fit_calibrator(scores, labels, n_bins=5, min_samples_per_bin=30)

    assert calibrator.base_rate == pytest.approx(0.30)


def test_bins_cover_training_range() -> None:
    scores, labels = make_samples(200, {(0.3, 0.7): 0.3})
    calibrator = fit_calibrator(scores, labels, n_bins=4, min_samples_per_bin=30)

    assert calibrator.bins[0].lo == pytest.approx(scores.min())
    assert calibrator.bins[-1].hi == pytest.approx(scores.max())


def test_calibrator_is_immutable() -> None:
    scores, labels = make_samples(200, {(0.3, 0.7): 0.3})
    calibrator = fit_calibrator(scores, labels, n_bins=4, min_samples_per_bin=30)
    with pytest.raises(Exception):
        calibrator.base_rate = 0.5  # type: ignore[misc]


# ══════════════════════════════════════════════════════════════
# 可靠度報告
# ══════════════════════════════════════════════════════════════


def test_reliability_report_shows_score_vs_actual() -> None:
    """
    報告要並列「分數」與「實際機率」，讓過度自信一眼看得出來。
    """
    scores, labels = make_samples(200, {(0.6, 0.8): 0.25})
    calibrator = fit_calibrator(scores, labels, n_bins=3, min_samples_per_bin=30)

    text = reliability_report(calibrator)
    assert "分數區間" in text
    assert "實際 P(+1)" in text
    assert "樣本數" in text


def test_reliability_report_flags_overconfidence() -> None:
    """
    分數遠高於實際機率時要標示——這是最危險的失準方向。

    構造：分數約 0.7，實際命中率僅 0.2。
    """
    scores, labels = make_samples(200, {(0.65, 0.75): 0.20})
    calibrator = fit_calibrator(scores, labels, n_bins=2, min_samples_per_bin=30)

    assert "過度自信" in reliability_report(calibrator)


def test_reliability_report_marks_sparse_bins() -> None:
    scores = np.concatenate([np.full(200, 0.3), np.full(3, 0.9)])
    labels = np.concatenate([np.full(200, -1), np.full(3, 1)])
    calibrator = fit_calibrator(scores, labels, n_bins=5, min_samples_per_bin=30)

    assert "樣本不足" in reliability_report(calibrator)


# ══════════════════════════════════════════════════════════════
# 期望報酬校準（路線 A）
#
# triple-barrier 有固定目標，所以「命中率」就足以算期望值：
#     E[R] = p × target − (1 − p) × stop
#
# 移動停損沒有固定目標——每筆的實際報酬都不同（可能 +3%，也可能
# +112%）。此時命中率**不足以描述期望值**，必須直接校準「每箱的
# 平均實際報酬」。
#
# 這是路線 A 對校準層的唯一結構性改動。
# ══════════════════════════════════════════════════════════════

from taiwan_quant.validation.calibration import (  # noqa: E402
    ReturnCalibrator,
    fit_return_calibrator,
    return_reliability_report,
)


def make_return_samples(
    n_per_bin: int,
    mean_returns: dict[tuple[float, float], float],
    noise: float = 0.05,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """構造已知平均報酬的樣本"""
    rng = np.random.default_rng(seed)
    scores: list[float] = []
    returns: list[float] = []
    for (lo, hi), mean in mean_returns.items():
        scores.extend(rng.uniform(lo, hi, n_per_bin))
        returns.extend(rng.normal(mean, noise, n_per_bin))
    return np.asarray(scores), np.asarray(returns)


@pytest.mark.unit
def test_return_calibrator_recovers_mean_return() -> None:
    """
    構造：分數 [0.6, 0.7) 的樣本平均報酬 +8%。
    校準後查詢 0.65 應回約 0.08。

    2000 筆分 5 箱 = 每箱 400 筆，噪音 5% → 平均數標準誤
    = 0.05/sqrt(400) = 0.0025，±0.01 約 4 個標準誤。
    """
    scores, returns = make_return_samples(2000, {(0.6, 0.7): 0.08})
    cal = fit_return_calibrator(scores, returns, n_bins=5, min_samples_per_bin=30)

    assert cal.predict(0.65) == pytest.approx(0.08, abs=0.01)


@pytest.mark.unit
def test_return_calibrator_is_monotonic_when_signal_exists() -> None:
    scores, returns = make_return_samples(
        800, {(0.2, 0.4): -0.02, (0.4, 0.6): 0.03, (0.6, 0.8): 0.09}
    )
    cal = fit_return_calibrator(scores, returns, n_bins=3, min_samples_per_bin=30)

    assert cal.predict(0.3) < cal.predict(0.5) < cal.predict(0.7)


@pytest.mark.unit
def test_return_calibrator_reports_base_return() -> None:
    """整體平均報酬——策略必須勝過它才有選股價值"""
    scores, returns = make_return_samples(1000, {(0.3, 0.7): 0.05})
    cal = fit_return_calibrator(scores, returns, n_bins=4, min_samples_per_bin=30)
    assert cal.base_return == pytest.approx(0.05, abs=0.01)


@pytest.mark.unit
def test_return_calibrator_sparse_bin_returns_none() -> None:
    """樣本不足的箱回 None，不用相鄰箱硬補"""
    scores = np.concatenate([np.full(300, 0.3), np.full(3, 0.9)])
    returns = np.concatenate([np.full(300, 0.01), np.full(3, 0.50)])
    cal = fit_return_calibrator(scores, returns, n_bins=5, min_samples_per_bin=30)

    assert cal.predict(0.3) is not None
    assert cal.predict(0.9) is None


@pytest.mark.unit
def test_return_calibrator_outside_range_returns_none() -> None:
    scores, returns = make_return_samples(500, {(0.4, 0.6): 0.05})
    cal = fit_return_calibrator(scores, returns, n_bins=4, min_samples_per_bin=30)
    assert cal.predict(0.05) is None
    assert cal.predict(0.99) is None


@pytest.mark.unit
def test_return_calibrator_reports_dispersion() -> None:
    """
    每箱要帶報酬標準差——平均 +8% 但標準差 40% 與標準差 2%，
    對決策的意義完全不同。
    """
    scores, returns = make_return_samples(1000, {(0.5, 0.7): 0.08}, noise=0.20)
    cal = fit_return_calibrator(scores, returns, n_bins=4, min_samples_per_bin=30)

    usable = [b for b in cal.bins if b.n_samples >= 30]
    assert all(b.return_std > 0 for b in usable)
    assert usable[0].return_std == pytest.approx(0.20, abs=0.03)


@pytest.mark.unit
def test_return_calibrator_rejects_non_finite() -> None:
    scores = np.array([0.5] * 99 + [np.nan])
    returns = np.zeros(100)
    with pytest.raises(CalibrationError, match="有限值"):
        fit_return_calibrator(scores, returns, n_bins=2, min_samples_per_bin=10)

    with pytest.raises(CalibrationError, match="有限值"):
        fit_return_calibrator(
            np.full(100, 0.5), np.array([0.1] * 99 + [np.inf]),
            n_bins=2, min_samples_per_bin=10,
        )


@pytest.mark.unit
def test_return_calibrator_rejects_mismatched_lengths() -> None:
    with pytest.raises(CalibrationError, match="長度"):
        fit_return_calibrator(np.zeros(10), np.zeros(5), n_bins=2, min_samples_per_bin=1)


@pytest.mark.unit
def test_return_calibrator_is_immutable() -> None:
    scores, returns = make_return_samples(500, {(0.3, 0.7): 0.05})
    cal = fit_return_calibrator(scores, returns, n_bins=4, min_samples_per_bin=30)
    with pytest.raises(Exception):
        cal.base_return = 0.0  # type: ignore[misc]


@pytest.mark.unit
def test_return_reliability_report_shows_mean_and_dispersion() -> None:
    scores, returns = make_return_samples(800, {(0.4, 0.8): 0.06}, noise=0.15)
    cal = fit_return_calibrator(scores, returns, n_bins=4, min_samples_per_bin=30)

    text = return_reliability_report(cal)
    assert "分數區間" in text
    assert "平均報酬" in text
    assert "標準差" in text
    assert "樣本數" in text


@pytest.mark.unit
def test_return_reliability_report_flags_no_discrimination() -> None:
    """
    各箱平均報酬差異極小 → 標示「無鑑別力」。

    這是最重要的診斷：分數排序若與報酬無關，整個選股流程沒有意義。
    """
    scores, returns = make_return_samples(
        800, {(0.2, 0.4): 0.05, (0.4, 0.6): 0.051, (0.6, 0.8): 0.049}, noise=0.02
    )
    cal = fit_return_calibrator(scores, returns, n_bins=3, min_samples_per_bin=30)
    assert "無鑑別力" in return_reliability_report(cal)
