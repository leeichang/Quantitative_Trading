"""
多重測試校正與 IC 選擇健康度

兩個都是 qlib-tw-trader 缺少、而實測證明必要的東西。

## Deflated Sharpe Ratio（多重測試校正）

qlib-tw-trader 跑了 9 策略 × 7 hedge config ≈ 63 種組合，報告「最佳」
那個的 Sharpe 1.724，卻沒做任何校正。63 組合裡挑最高值，必然被選擇偏誤
污染——它的 README 自己引了 Harvey/Liu/Zhu 的多重測試論文卻沒套用。

DSR 的核心觀念：如果你試了 N 組參數，就算全部都沒有真實優勢，
最佳那組的 Sharpe 期望值也會是正的，而且隨 N 成長約 sqrt(2·ln N)。
要證明有優勢，觀測 Sharpe 必須顯著高於這個「運氣天花板」。

    Bailey & López de Prado (2014), "The Deflated Sharpe Ratio"

## PBO（Probability of Backtest Overfitting）

把樣本切成多組 IS/OOS，問一個直白的問題：
「樣本內表現最好的策略，在樣本外落到後半段的機率有多高？」

超過 0.5 表示樣本內排名比丟硬幣還差 → 該策略族判定過擬合。

    Bailey et al. (2015), "The Probability of Backtest Overfitting"

## valid/live IC 相關係數（CLAUDE.md 規格 14）

實測 qlib-tw-trader：valid IC +0.0362、live IC −0.0263、相關係數 **−0.159**。

意思是驗證期表現好的模型，樣本外反而略差——模型選擇機制方向相反。
**這比虧錢更嚴重**，因為它代表整個調參流程都在調噪音。
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

EULER_MASCHERONI = 0.5772156649015329

DSR_SIGNIFICANCE = 0.95
"""DSR 顯著門檻。低於此值視為「無法排除運氣」"""

PBO_THRESHOLD = 0.5
"""CLAUDE.md：PBO > 0.5 → 該策略族判定過擬合，不得進入 Top 3"""

IC_CORRELATION_MIN = 0.3
"""
valid/live IC 相關係數的健康門檻。

低於此值代表 valid IC 對 live IC 幾乎沒有資訊量，
用它選模型等於在調噪音。
"""

MIN_IC_SAMPLES = 5
"""算相關係數的最少樣本數"""

ZERO_VARIANCE_TOLERANCE = 1e-12
"""
判定「零變異」的容差。

重複相同浮點值算出的標準差是 1e-18 級的噪音而非精確 0，
用 `== 0` 判斷會漏掉，讓相關係數變成同樣是噪音的數字。
"""


def _normal_cdf(x: float) -> float:
    """標準常態 CDF"""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _normal_ppf(p: float) -> float:
    """
    標準常態分位數函數（反 CDF）。

    用 Acklam 的有理近似，精度約 1e-9，足夠本用途，
    且避免為此引入 scipy 依賴。
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"機率必須落在 (0, 1)，得到 {p}")

    a = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
    b = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00)

    p_low, p_high = 0.02425, 1.0 - 0.02425

    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )
    if p > p_high:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )

    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    )


# ══════════════════════════════════════════════════════════════
# Deflated Sharpe Ratio
# ══════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class DSRResult:
    """Deflated Sharpe Ratio 結果"""

    observed_sharpe: float
    expected_max_sharpe: float
    """N 組全無優勢時，最佳那組的 Sharpe 期望值——運氣的天花板"""

    deflated_sharpe: float
    """校正後的顯著性機率（0~1）"""

    n_trials: int
    n_observations: int

    @property
    def is_significant(self) -> bool:
        return self.deflated_sharpe >= DSR_SIGNIFICANCE

    def describe(self) -> str:
        return "\n".join([
            f"觀測 Sharpe        {self.observed_sharpe:.3f}",
            f"期望最大 Sharpe     {self.expected_max_sharpe:.3f}"
            f"（{self.n_trials} 組試驗的運氣天花板）",
            f"Deflated Sharpe    {self.deflated_sharpe:.4f}",
            f"判定              {'顯著' if self.is_significant else '無法排除運氣'}"
            f"（門檻 {DSR_SIGNIFICANCE}）",
        ])


def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_trials: int,
    n_observations: int,
    sharpe_std: float = 1.0,
) -> DSRResult:
    """
    計算 Deflated Sharpe Ratio。

    Args:
        observed_sharpe: 觀測到的（最佳）Sharpe
        n_trials: 試了幾組參數／策略。**這是關鍵——必須誠實填寫**
        n_observations: 報酬序列的觀測期數
        sharpe_std: 各試驗 Sharpe 的橫斷面標準差

    Returns:
        DSRResult

    Raises:
        ValueError: 參數不合法

    期望最大 Sharpe 用 Gumbel 極值分布近似：

        E[max] ≈ σ · [(1 − γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e))]

    其中 γ 是 Euler-Mascheroni 常數。N=1 時退化為 0（沒有選擇偏誤）。
    """
    if n_trials < 1:
        raise ValueError(f"n_trials 至少為 1，得到 {n_trials}")
    if n_observations < 2:
        raise ValueError(f"n_observations 至少為 2，得到 {n_observations}")
    if sharpe_std <= 0:
        raise ValueError(f"sharpe_std 必須為正，得到 {sharpe_std}")

    if n_trials == 1:
        expected_max = 0.0
    else:
        expected_max = sharpe_std * (
            (1.0 - EULER_MASCHERONI) * _normal_ppf(1.0 - 1.0 / n_trials)
            + EULER_MASCHERONI * _normal_ppf(1.0 - 1.0 / (n_trials * math.e))
        )

    # 以 Sharpe 估計量的標準誤做 z 檢定
    standard_error = math.sqrt(1.0 / (n_observations - 1))
    z = (observed_sharpe - expected_max) / standard_error

    return DSRResult(
        observed_sharpe=observed_sharpe,
        expected_max_sharpe=expected_max,
        deflated_sharpe=_normal_cdf(z),
        n_trials=n_trials,
        n_observations=n_observations,
    )


# ══════════════════════════════════════════════════════════════
# PBO
# ══════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class PBOResult:
    """PBO 結果"""

    pbo: float
    """樣本內最佳策略在樣本外落到後半段的機率"""

    median_oos_rank: float
    """樣本內最佳策略的樣本外排名分位中位數（0.5 = 隨機）"""

    n_splits: int
    n_strategies: int

    @property
    def is_overfit(self) -> bool:
        return self.pbo > PBO_THRESHOLD

    def describe(self) -> str:
        return "\n".join([
            f"PBO               {self.pbo:.3f}（門檻 {PBO_THRESHOLD}）",
            f"樣本外排名中位數    {self.median_oos_rank:.3f}（0.5 = 隨機）",
            f"切分 / 策略數      {self.n_splits} / {self.n_strategies}",
            f"判定              {'過擬合，不得進入 Top 3' if self.is_overfit else '未達過擬合門檻'}",
        ])


def probability_of_backtest_overfitting(
    is_returns: np.ndarray,
    oos_returns: np.ndarray,
    n_splits: int = 16,
) -> PBOResult:
    """
    計算 PBO。

    Args:
        is_returns: 樣本內報酬矩陣，shape (觀測期數, 策略數)
        oos_returns: 樣本外報酬矩陣，同 shape
        n_splits: 切分次數

    Returns:
        PBOResult

    Raises:
        ValueError: 形狀不符、策略數不足、觀測值不足或含非有限值

    做法：隨機切分觀測期，每次在 IS 段挑出最佳策略，看它在 OOS 段的
    排名分位。分位落在後半段（< 0.5）就記一次過擬合。
    """
    is_returns = np.asarray(is_returns, dtype=float)
    oos_returns = np.asarray(oos_returns, dtype=float)

    if is_returns.shape != oos_returns.shape:
        raise ValueError(
            f"IS 與 OOS 形狀必須相同：{is_returns.shape} vs {oos_returns.shape}"
        )
    if is_returns.ndim != 2 or is_returns.shape[1] < 2:
        raise ValueError("需要至少 2 個策略才能談排名")
    if is_returns.shape[0] < n_splits:
        raise ValueError(
            f"觀測值不足：{is_returns.shape[0]} 期 < {n_splits} 次切分"
        )
    if not (np.isfinite(is_returns).all() and np.isfinite(oos_returns).all()):
        raise ValueError(
            "報酬矩陣必須全為有限值。NaN 會讓 argmax 給出無意義結果且不拋錯。"
        )

    n_obs, n_strategies = is_returns.shape
    rng = np.random.default_rng(20260912)

    ranks: list[float] = []
    for _ in range(n_splits):
        # 隨機對半切，避免固定切點造成的偏誤
        order = rng.permutation(n_obs)
        half = n_obs // 2
        is_idx, oos_idx = order[:half], order[half:]

        is_scores = _sharpe_by_column(is_returns[is_idx])
        oos_scores = _sharpe_by_column(oos_returns[oos_idx])

        best = int(np.argmax(is_scores))
        # 樣本外排名分位：1.0 = 最佳、0.0 = 最差
        worse_count = int(np.sum(oos_scores < oos_scores[best]))
        ranks.append(worse_count / (n_strategies - 1))

    ranks_array = np.asarray(ranks)
    return PBOResult(
        pbo=float(np.mean(ranks_array < 0.5)),
        median_oos_rank=float(np.median(ranks_array)),
        n_splits=n_splits,
        n_strategies=n_strategies,
    )


def _sharpe_by_column(matrix: np.ndarray) -> np.ndarray:
    """逐欄算 Sharpe；零變異時以平均值代替（保持排序意義）"""
    mean = matrix.mean(axis=0)
    sd = matrix.std(axis=0, ddof=1) if matrix.shape[0] > 1 else np.zeros_like(mean)
    return np.where(sd > 0, mean / np.where(sd > 0, sd, 1.0), mean)


# ══════════════════════════════════════════════════════════════
# IC 選擇健康度（規格 14）
# ══════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ICMonitorResult:
    """valid/live IC 的關係診斷"""

    mean_valid_ic: float
    mean_live_ic: float
    ic_decay: float
    """(valid − live) / valid。live 為負時會超過 1.0，如實呈現不截斷"""

    correlation: float | None
    """valid IC 與 live IC 的相關係數；零變異時為 None"""

    n_samples: int

    @property
    def is_healthy(self) -> bool:
        """相關係數需達門檻才算模型選擇機制有效"""
        return self.correlation is not None and self.correlation >= IC_CORRELATION_MIN

    @property
    def verdict(self) -> str:
        if self.correlation is None:
            return (
                "valid IC 無變異，相關係數無定義——無法判斷模型選擇機制是否有效"
            )
        if self.correlation < 0:
            return (
                f"模型選擇機制**失效且方向相反**（相關係數 {self.correlation:+.3f}）："
                "驗證期表現好的模型樣本外反而較差。此時調參數是在調噪音，"
                "應先檢查特徵與標記，不要繼續最佳化。"
            )
        if self.correlation < IC_CORRELATION_MIN:
            return (
                f"模型選擇機制**失效**（相關係數 {self.correlation:+.3f} "
                f"< {IC_CORRELATION_MIN}）：valid IC 對 live IC 幾乎沒有資訊量，"
                "用它挑模型等於在調噪音。"
            )
        return (
            f"模型選擇機制有效（相關係數 {self.correlation:+.3f}）："
            "valid IC 對樣本外表現有預測力。"
        )

    def describe(self) -> str:
        corr = f"{self.correlation:+.4f}" if self.correlation is not None else "n/a"
        return "\n".join([
            f"平均 valid IC      {self.mean_valid_ic:+.4f}",
            f"平均 live IC       {self.mean_live_ic:+.4f}",
            f"IC 衰減            {self.ic_decay * 100:.1f}%",
            f"valid/live 相關    {corr}",
            f"樣本數             {self.n_samples}",
            "",
            self.verdict,
        ])


def ic_selection_health(
    valid_ic: Sequence[float],
    live_ic: Sequence[float],
) -> ICMonitorResult:
    """
    診斷 valid IC 對 live IC 的預測力（CLAUDE.md 規格 14）。

    Args:
        valid_ic: 各期的驗證期 IC
        live_ic: 各期的樣本外 IC

    Returns:
        ICMonitorResult

    Raises:
        ValueError: 長度不符、樣本不足或含非有限值

    實測參考（qlib-tw-trader）：
        valid IC +0.0362、live IC −0.0263、衰減 172.6%、相關係數 −0.159
        → 判定失效且方向相反
    """
    if len(valid_ic) != len(live_ic):
        raise ValueError(f"長度必須相同：{len(valid_ic)} vs {len(live_ic)}")
    if len(valid_ic) < MIN_IC_SAMPLES:
        raise ValueError(
            f"樣本不足：{len(valid_ic)} < {MIN_IC_SAMPLES}，算不出有意義的相關係數"
        )

    valid = np.asarray(valid_ic, dtype=float)
    live = np.asarray(live_ic, dtype=float)
    if not (np.isfinite(valid).all() and np.isfinite(live).all()):
        raise ValueError("IC 序列必須全為有限值")

    mean_valid = float(valid.mean())
    mean_live = float(live.mean())
    decay = (mean_valid - mean_live) / mean_valid if mean_valid != 0 else float("nan")

    # 用容差而非 `== 0`：重複相同浮點值算出的 std 是 1e-18 級的噪音而非精確 0，
    # 此時 corrcoef 會回一個同樣是噪音的相關係數（例如 4.3e-17），
    # 被誤讀成「算出來了，結果沒有相關」。
    if valid.std(ddof=1) < ZERO_VARIANCE_TOLERANCE or live.std(ddof=1) < ZERO_VARIANCE_TOLERANCE:
        correlation = None
    else:
        correlation = float(np.corrcoef(valid, live)[0, 1])

    return ICMonitorResult(
        mean_valid_ic=mean_valid,
        mean_live_ic=mean_live,
        ic_decay=decay,
        correlation=correlation,
        n_samples=len(valid),
    )
