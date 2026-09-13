"""
逐期橫斷面 Rank IC 測試

## 為什麼需要這個工具

先前用「全部樣本池在一起算一個 Spearman」判定訊號有沒有排序能力，
那是**方法學錯誤**。實測差異：

    策略族      全域 Spearman   逐期 Rank IC
    動能突破      −0.0077        +0.0384
    籌碼跟隨      −0.0018        +0.0281
    均值回歸      −0.0132        −0.0369

全域版把橫斷面排序與時序變異混在一起——在一段所有股票都漲 20% 的期間，
相關係數被「哪幾天報酬高」主導，而不是「那天哪幾檔排得好」。

選股是**橫斷面**任務：今天這 150 檔裡，誰會比較強。
所以必須逐期算，再看跨期的平均與穩定度。

## 重疊必須修正

持有 60 日但每 5 日決策 → 連續 12 期的持有期重疊。
用 512 期算 t 值會嚴重高估顯著性，實際有效樣本只有 512/12 ≈ 42。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from taiwan_quant.validation.rank_ic import (
    RankICError,
    evaluate_rank_ic,
    period_rank_ic,
)


def period(scores: list[float], returns: list[float]) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(scores, float), np.asarray(returns, float)


# ══════════════════════════════════════════════════════════════
# 單期 Rank IC
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_perfect_ranking_gives_one() -> None:
    """分數排名與報酬排名完全一致 → IC = 1"""
    s, r = period([1, 2, 3, 4, 5], [0.01, 0.02, 0.03, 0.04, 0.05])
    assert period_rank_ic(s, r, min_names=5) == pytest.approx(1.0)


@pytest.mark.unit
def test_reversed_ranking_gives_minus_one() -> None:
    """完全相反 → IC = −1。**這是資訊，不是沒用**（反向操作即可）"""
    s, r = period([1, 2, 3, 4, 5], [0.05, 0.04, 0.03, 0.02, 0.01])
    assert period_rank_ic(s, r, min_names=5) == pytest.approx(-1.0)


@pytest.mark.unit
def test_uses_rank_not_magnitude() -> None:
    """
    用**排名**不是數值大小。

    一檔 +500% 的極端報酬不該主導整期的 IC——那正是 Pearson 的問題。
    """
    s, r = period([1, 2, 3, 4], [0.01, 0.02, 0.03, 5.00])
    assert period_rank_ic(s, r, min_names=4) == pytest.approx(1.0)


@pytest.mark.unit
def test_returns_none_when_too_few_names() -> None:
    """
    橫斷面太窄時回 `None`，不可回 0。

    3 檔算出來的 IC 是雜訊；回 0 會被平均進去，把真實的 IC 稀釋掉。
    """
    s, r = period([1, 2, 3], [0.01, 0.02, 0.03])
    assert period_rank_ic(s, r, min_names=10) is None


@pytest.mark.unit
def test_returns_none_when_scores_are_constant() -> None:
    """
    分數全部相同 → 沒有排序可言 → `None`。

    實測踩過：校準器只分 6 箱時，同一期有 10 檔並列同分。
    這種期數算出來的 IC 沒有意義。
    """
    s, r = period([0.5] * 12, list(np.linspace(0.01, 0.12, 12)))
    assert period_rank_ic(s, r) is None


@pytest.mark.unit
def test_ignores_non_finite_pairs() -> None:
    """NaN 的配對剔除，不可當成 0"""
    s = np.array([1, 2, 3, 4, 5, np.nan] + list(range(6, 16)), float)
    r = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
                 + [0.07 + 0.01 * i for i in range(10)], float)
    assert period_rank_ic(s, r) == pytest.approx(1.0)


@pytest.mark.unit
def test_rejects_mismatched_lengths() -> None:
    with pytest.raises(RankICError, match="長度"):
        period_rank_ic(np.zeros(5), np.zeros(3))


# ══════════════════════════════════════════════════════════════
# 跨期彙總
# ══════════════════════════════════════════════════════════════


def synthetic_periods(
    n_periods: int, n_names: int, ic: float, seed: int = 11
) -> dict[pd.Timestamp, tuple[np.ndarray, np.ndarray]]:
    """造出平均 Rank IC 約等於 `ic` 的多期資料"""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2019-01-01", periods=n_periods, freq="W-FRI")
    out = {}
    for d in dates:
        s = rng.normal(0, 1, n_names)
        noise = rng.normal(0, 1, n_names)
        r = ic * s + np.sqrt(max(1 - ic**2, 0.0)) * noise
        out[d] = (s, r)
    return out


@pytest.mark.unit
def test_mean_ic_recovers_planted_signal() -> None:
    data = synthetic_periods(200, 150, ic=0.10)
    report = evaluate_rank_ic(data, horizon=20, stride=5)
    assert report.mean_ic == pytest.approx(0.10, abs=0.02)


@pytest.mark.unit
def test_zero_signal_gives_insignificant_t() -> None:
    data = synthetic_periods(200, 150, ic=0.0)
    report = evaluate_rank_ic(data, horizon=20, stride=5)
    assert abs(report.t_stat) < 2.0
    assert not report.is_significant


@pytest.mark.unit
def test_overlap_correction_shrinks_t_stat() -> None:
    """
    重疊修正必須讓 t 值變小。

    持有 60 日、每 5 日決策 → 每 12 期才獨立。不修正會高估
    sqrt(12) ≈ 3.5 倍的顯著性。
    """
    data = synthetic_periods(240, 150, ic=0.05)
    naive = evaluate_rank_ic(data, horizon=5, stride=5)     # 無重疊
    overlapped = evaluate_rank_ic(data, horizon=60, stride=5)

    assert abs(overlapped.t_stat) < abs(naive.t_stat)
    assert overlapped.effective_periods == pytest.approx(naive.effective_periods / 12)


@pytest.mark.unit
def test_effective_periods_matches_hand_calculation() -> None:
    """240 期、持有 60 日、每 5 日決策 → 240 / 12 = 20 個有效期"""
    data = synthetic_periods(240, 150, ic=0.0)
    report = evaluate_rank_ic(data, horizon=60, stride=5)
    assert report.effective_periods == pytest.approx(20.0)


@pytest.mark.unit
def test_positive_rate() -> None:
    data = synthetic_periods(200, 150, ic=0.30)
    report = evaluate_rank_ic(data, horizon=20, stride=5)
    assert report.positive_rate > 0.8


@pytest.mark.unit
def test_by_year_breakdown() -> None:
    """
    逐年拆解。單一年份撐起全部結果是常見的假訊號。
    """
    data = synthetic_periods(300, 150, ic=0.10)
    report = evaluate_rank_ic(data, horizon=20, stride=5)
    assert len(report.by_year) >= 5
    assert all(isinstance(y, int) for y in report.by_year)


@pytest.mark.unit
def test_skips_periods_with_too_few_names() -> None:
    """
    橫斷面太窄的期數要被跳過，不是算出雜訊平均進去。
    """
    data = synthetic_periods(50, 150, ic=0.10)
    thin = {d: (s[:4], r[:4]) for d, (s, r) in list(data.items())[:20]}
    data.update(thin)

    report = evaluate_rank_ic(data, horizon=20, stride=5, min_names=10)
    assert report.n_periods == 30
    assert report.skipped == 20


@pytest.mark.unit
def test_raises_when_no_usable_period() -> None:
    data = {pd.Timestamp("2024-01-05"): (np.zeros(3), np.zeros(3))}
    with pytest.raises(RankICError, match="沒有"):
        evaluate_rank_ic(data, horizon=20, stride=5, min_names=10)


@pytest.mark.unit
def test_report_is_immutable() -> None:
    report = evaluate_rank_ic(synthetic_periods(50, 150, 0.1), horizon=20, stride=5)
    with pytest.raises(Exception):
        report.mean_ic = 0.0  # type: ignore[misc]


@pytest.mark.unit
def test_describe_mentions_verdict() -> None:
    report = evaluate_rank_ic(synthetic_periods(200, 150, 0.0), horizon=20, stride=5)
    text = report.describe()
    assert "Rank IC" in text
    assert "有效期數" in text
    assert "無法排除運氣" in text or "顯著" in text
