#!/usr/bin/env python3
"""
多重測試校正與 IC 監控測試

兩個都是 qlib-tw-trader 缺少、而實測證明必要的東西：

**PBO / Deflated Sharpe（CLAUDE.md 多重測試校正）**
    qlib-tw-trader 跑了 9 策略 × 7 hedge config ≈ 63 種組合，然後報告
    「最佳」那個的 Sharpe 1.724，卻沒做任何校正。63 組合裡挑最高值，
    必然被選擇偏誤污染。README 自己引了 Harvey/Liu/Zhu 的多重測試論文
    卻沒套用。

**valid/live IC 相關係數（規格 14）**
    實測 qlib-tw-trader：valid IC +0.0362、live IC −0.0263、
    相關係數 **−0.159**。意思是驗證期表現好的模型，樣本外反而略差——
    模型選擇機制方向相反。這比虧錢更嚴重，因為它代表調參數是在調噪音。

預期值全部手算或用解析解驗證。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from taiwan_quant.validation.stats import (
    ICMonitorResult,
    PBOResult,
    deflated_sharpe_ratio,
    ic_selection_health,
    probability_of_backtest_overfitting,
)

pytestmark = pytest.mark.unit


# ══════════════════════════════════════════════════════════════
# Deflated Sharpe Ratio
# ══════════════════════════════════════════════════════════════


def test_dsr_equals_sharpe_when_single_trial() -> None:
    """
    只試一組參數時沒有選擇偏誤，DSR 應接近原始 Sharpe 的顯著性。

    n_trials=1 時期望最大值調整項為 0，DSR 退化為單純的 Sharpe 檢定。
    """
    result = deflated_sharpe_ratio(
        observed_sharpe=1.0, n_trials=1, n_observations=100, sharpe_std=1.0
    )
    assert result.expected_max_sharpe == pytest.approx(0.0, abs=1e-9)
    assert result.deflated_sharpe > 0.5, "單一試驗且 Sharpe 為正，DSR 應偏高"


def test_dsr_penalizes_more_trials() -> None:
    """
    試越多組參數，同樣的 Sharpe 越不可信。

    qlib-tw-trader 的 63 組合就是這個情境。
    """
    few = deflated_sharpe_ratio(1.5, n_trials=1, n_observations=156, sharpe_std=1.0)
    many = deflated_sharpe_ratio(1.5, n_trials=63, n_observations=156, sharpe_std=1.0)
    assert many.deflated_sharpe < few.deflated_sharpe
    assert many.expected_max_sharpe > few.expected_max_sharpe


def test_dsr_expected_max_grows_with_log_trials() -> None:
    """
    期望最大 Sharpe 隨試驗數成長（約 sqrt(2·ln N)）。

    手算：N=63 時 sqrt(2·ln 63) ≈ sqrt(2×4.143) ≈ 2.878
    實際公式含 Euler-Mascheroni 修正，所以取寬鬆區間。
    """
    result = deflated_sharpe_ratio(1.5, n_trials=63, n_observations=156, sharpe_std=1.0)
    assert 1.5 < result.expected_max_sharpe < 3.5


def test_dsr_flags_qlib_tw_trader_case() -> None:
    """
    用 qlib-tw-trader 的實際數字：Sharpe 1.724、63 組合、156 週。

    這個案例應該被判為「不顯著」——README 自己報的 t-stat 是 1.89（< 2）。
    """
    result = deflated_sharpe_ratio(
        observed_sharpe=1.724, n_trials=63, n_observations=156, sharpe_std=1.0
    )
    assert not result.is_significant, (
        f"DSR = {result.deflated_sharpe:.3f} 判為顯著，但該案例 t-stat 僅 1.89"
    )


def test_dsr_significant_when_sharpe_far_above_expected_max() -> None:
    """Sharpe 遠高於期望最大值時應判為顯著"""
    result = deflated_sharpe_ratio(5.0, n_trials=10, n_observations=500, sharpe_std=1.0)
    assert result.is_significant


def test_dsr_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="n_trials"):
        deflated_sharpe_ratio(1.0, n_trials=0, n_observations=100, sharpe_std=1.0)
    with pytest.raises(ValueError, match="n_observations"):
        deflated_sharpe_ratio(1.0, n_trials=1, n_observations=1, sharpe_std=1.0)
    with pytest.raises(ValueError, match="sharpe_std"):
        deflated_sharpe_ratio(1.0, n_trials=1, n_observations=100, sharpe_std=0.0)


def test_dsr_is_immutable() -> None:
    result = deflated_sharpe_ratio(1.0, 1, 100, 1.0)
    with pytest.raises(Exception):
        result.deflated_sharpe = 0.0  # type: ignore[misc]


# ══════════════════════════════════════════════════════════════
# PBO（Probability of Backtest Overfitting）
# ══════════════════════════════════════════════════════════════


def test_pbo_is_low_when_is_ranking_predicts_oos() -> None:
    """
    樣本內排名能預測樣本外 → PBO 低（沒有過擬合）。

    構造：兩組策略，樣本內與樣本外排名完全一致。
    """
    is_returns = np.array([[0.01, 0.02, 0.03]] * 40)   # 策略 2 最好
    oos_returns = np.array([[0.01, 0.02, 0.03]] * 40)  # 樣本外也是策略 2 最好

    result = probability_of_backtest_overfitting(is_returns, oos_returns, n_splits=8)
    assert result.pbo < 0.5
    assert not result.is_overfit


def test_pbo_is_high_when_is_ranking_inverts_oos() -> None:
    """
    樣本內最佳策略在樣本外變最差 → PBO 高（嚴重過擬合）。

    這正是 qlib-tw-trader 的徵狀：valid IC 與 live IC 負相關。
    """
    rng = np.random.default_rng(0)
    n_obs = 60
    # 策略 0 樣本內好、樣本外差；策略 2 反之
    is_returns = np.column_stack([
        rng.normal(0.03, 0.01, n_obs),
        rng.normal(0.02, 0.01, n_obs),
        rng.normal(0.01, 0.01, n_obs),
    ])
    oos_returns = np.column_stack([
        rng.normal(0.01, 0.01, n_obs),
        rng.normal(0.02, 0.01, n_obs),
        rng.normal(0.03, 0.01, n_obs),
    ])

    result = probability_of_backtest_overfitting(is_returns, oos_returns, n_splits=10)
    assert result.pbo > 0.5, f"PBO = {result.pbo}，排名反轉應判為過擬合"
    assert result.is_overfit


def test_pbo_reports_median_oos_rank() -> None:
    """
    要回報樣本內最佳策略在樣本外的中位數排名分位。

    0.5 = 隨機（毫無預測力）；接近 1 = 樣本內排名有效。
    """
    is_returns = np.array([[0.01, 0.03]] * 40)
    oos_returns = np.array([[0.01, 0.03]] * 40)
    result = probability_of_backtest_overfitting(is_returns, oos_returns, n_splits=6)
    assert 0.0 <= result.median_oos_rank <= 1.0


def test_pbo_requires_multiple_strategies() -> None:
    """單一策略無法談「排名」，必須明確拒絕"""
    single = np.array([[0.01]] * 40)
    with pytest.raises(ValueError, match="至少 2 個策略"):
        probability_of_backtest_overfitting(single, single, n_splits=4)


def test_pbo_requires_matching_shapes() -> None:
    with pytest.raises(ValueError, match="形狀"):
        probability_of_backtest_overfitting(
            np.zeros((40, 3)), np.zeros((40, 2)), n_splits=4
        )


def test_pbo_requires_enough_observations() -> None:
    with pytest.raises(ValueError, match="觀測值不足"):
        probability_of_backtest_overfitting(
            np.zeros((3, 2)), np.zeros((3, 2)), n_splits=8
        )


def test_pbo_rejects_non_finite() -> None:
    """NaN 會讓 argmax 給出無意義結果且不拋錯，必須在邊界擋掉"""
    bad = np.array([[0.01, np.nan]] * 40)
    with pytest.raises(ValueError, match="有限值"):
        probability_of_backtest_overfitting(bad, bad, n_splits=4)


def test_pbo_threshold_is_half() -> None:
    """
    CLAUDE.md：PBO > 0.5 → 該策略族判定過擬合，不得進入 Top 3。

    門檻 0.5 的意義：樣本內最佳策略在樣本外落到後半段的機率超過一半，
    等於樣本內排名比丟硬幣還差。
    """
    result = PBOResult(pbo=0.51, median_oos_rank=0.4, n_splits=10, n_strategies=3)
    assert result.is_overfit
    assert not PBOResult(pbo=0.49, median_oos_rank=0.6, n_splits=10, n_strategies=3).is_overfit


# ══════════════════════════════════════════════════════════════
# IC 選擇健康度（規格 14）
# ══════════════════════════════════════════════════════════════


def test_ic_health_flags_negative_correlation() -> None:
    """
    valid IC 與 live IC 負相關 → 模型選擇機制失效。

    用 qlib-tw-trader 的實測數字：相關係數 −0.159。
    """
    valid = [0.05, 0.06, 0.07, 0.02, 0.01]
    live = [-0.03, -0.04, -0.05, 0.01, 0.02]   # 刻意反向

    result = ic_selection_health(valid, live)
    assert result.correlation < 0
    assert not result.is_healthy
    assert "失效" in result.verdict


def test_ic_health_accepts_positive_correlation() -> None:
    valid = [0.01, 0.02, 0.03, 0.04, 0.05]
    live = [0.01, 0.02, 0.03, 0.04, 0.05]
    result = ic_selection_health(valid, live)
    assert result.correlation == pytest.approx(1.0)
    assert result.is_healthy


def test_ic_health_flags_near_zero_correlation() -> None:
    """
    相關係數接近 0 也是失效——valid IC 對 live IC 沒有資訊，
    這時調參數就是在調噪音。
    """
    rng = np.random.default_rng(3)
    valid = list(rng.normal(0.03, 0.01, 40))
    live = list(rng.normal(0.0, 0.03, 40))
    result = ic_selection_health(valid, live)
    assert abs(result.correlation) < 0.3
    assert not result.is_healthy


def test_ic_health_computes_decay() -> None:
    """
    手算：平均 valid IC = 0.04、平均 live IC = 0.01
        衰減 = (0.04 − 0.01)/0.04 = 0.75  →  75%
    """
    result = ic_selection_health([0.04] * 5, [0.01] * 5)
    assert result.ic_decay == pytest.approx(0.75)


def test_ic_health_decay_over_one_when_live_negative() -> None:
    """
    live IC 為負時衰減會超過 100%——這個數字要如實呈現，不可截斷到 100%。

    手算：valid 0.0362、live −0.0263
        衰減 = (0.0362 − (−0.0263))/0.0362 = 0.0625/0.0362 = 1.7265  →  172.65%
    （qlib-tw-trader 實測報 172.6%，吻合）
    """
    result = ic_selection_health([0.0362] * 5, [-0.0263] * 5)
    assert result.ic_decay == pytest.approx(1.72651, rel=1e-4)


def test_ic_health_requires_matching_lengths() -> None:
    with pytest.raises(ValueError, match="長度"):
        ic_selection_health([0.01, 0.02], [0.01])


def test_ic_health_requires_enough_samples() -> None:
    """樣本太少算不出有意義的相關係數"""
    with pytest.raises(ValueError, match="樣本不足"):
        ic_selection_health([0.01, 0.02], [0.01, 0.02])


def test_ic_health_handles_zero_variance() -> None:
    """
    valid IC 完全相同 → 相關係數無定義，回 None 而非 0。

    回 0 會被誤讀成「有算出來，結果是沒有相關」。
    """
    result = ic_selection_health([0.03] * 10, list(np.linspace(0, 0.05, 10)))
    assert result.correlation is None
    assert not result.is_healthy


def test_ic_health_rejects_non_finite() -> None:
    with pytest.raises(ValueError, match="有限值"):
        ic_selection_health([0.01, float("nan"), 0.03, 0.04, 0.05], [0.01] * 5)


def test_ic_health_verdict_is_actionable() -> None:
    """
    verdict 要能直接貼進報告，講清楚「所以我該怎麼辦」。
    """
    bad = ic_selection_health([0.05, 0.06, 0.07, 0.02, 0.01], [-0.03, -0.04, -0.05, 0.01, 0.02])
    assert "調參數" in bad.verdict or "噪音" in bad.verdict


def test_ic_health_is_immutable() -> None:
    result = ic_selection_health([0.01, 0.02, 0.03, 0.04, 0.05], [0.01, 0.02, 0.03, 0.04, 0.05])
    with pytest.raises(Exception):
        result.correlation = 0.0  # type: ignore[misc]
