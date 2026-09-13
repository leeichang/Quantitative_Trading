"""
機率校準

## 問題

模型與規則式基準輸出的是**分數**，不是機率。

```python
def momentum_score(features):
    ...
    return 0.5 + 0.3 * np.mean(parts)    # 映射到 [0.2, 0.8]
```

這個 `0.65` 不代表「65% 機率達標」，它只是一個排序用的數字。

把它拿去跟進場門檻比較：

```
P(+1) ≥ (stop + cost) / (target + stop) = 40.22%
0.65 ≥ 0.4022  →  通過
```

**結論毫無意義。** 系統看起來有根據，實際上沒有。這比沒有門檻更危險，
因為它製造了「已經做過風險計算」的假象。

LightGBM 的 `predict_proba` 同樣有這個問題——梯度提升樹在類別不平衡
時常常過度自信。

## 解法：分箱校準

用歷史 triple-barrier 標籤回答一個直白的問題：

> 分數落在 [0.6, 0.7) 的歷史樣本中，實際有多少比例是 +1？

那個比例才是這一箱的 `P(+1)`。

## 三條設計原則

1. **樣本不足的箱回 `None`**，不可用相鄰箱或整體基準硬補。
   硬補會讓罕見分數區間看起來有統計依據，實際沒有。

2. **超出訓練期分數範圍回 `None`**，不外推。
   外推在金融資料上特別危險：極端分數往往出現在極端行情，
   而那正是歷史關係最可能失效的時候。

3. **時間柵到期（label 0）算 P(−1)**，與 `Candidate.expected_return`
   的保守假設一致——到期時實際報酬介於兩柵之間，這裡取最差情況。

## 兩種校準器

```
Calibrator         分數 → P(+1)     固定目標（triple-barrier）
ReturnCalibrator   分數 → E[報酬]   無固定目標（移動停損，路線 A）
```

有固定目標時，命中率就足以算期望值：

    E[R] = p × target − (1 − p) × stop

移動停損沒有固定目標——每筆的實際報酬都不同（可能 +3%，也可能 +112%）。
此時命中率**不足以描述期望值**，必須直接校準「每箱的平均實際報酬」。

## 反 look-ahead

校準器**只能用訓練期資料擬合**。用全樣本擬合等於讓模型知道未來的
命中率分布（CLAUDE.md 禁令 1）。呼叫端負責切分。
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

import numpy as np

from taiwan_quant.config.costs import DEFAULT, Tier
from taiwan_quant.validation.binning import bin_edges, bin_mask, find_bin

VALID_LABELS = frozenset({-1, 0, 1})

DEFAULT_N_BINS = 10
DEFAULT_MIN_SAMPLES_PER_BIN = 30

OVERCONFIDENCE_GAP = 0.15
"""分數高於實際機率超過此值視為過度自信，報告中標示"""

DISCRIMINATION_MIN_SPREAD = DEFAULT.round_trip_rate(Tier.LARGE)
"""
最高箱與最低箱的平均報酬差距門檻（禁令 3：費率一律取自 config/costs.py）。

差距小於一趟來回成本時，即使排序方向正確也賺不回交易成本——
那等於沒有鑑別力。用成本當門檻而非拍一個 0.01，是因為這個數字
本來就是「值不值得交易」的自然分界。
"""


class CalibrationError(RuntimeError):
    """校準失敗"""


@dataclass(frozen=True)
class CalibrationBin:
    """單一分數區間的校準結果"""

    lo: float
    hi: float
    n_samples: int
    n_hits: int

    @property
    def empirical_prob(self) -> float:
        """實際 P(+1)"""
        return self.n_hits / self.n_samples if self.n_samples else 0.0

    @property
    def mid_score(self) -> float:
        return (self.lo + self.hi) / 2

    def is_usable(self, min_samples: int) -> bool:
        return self.n_samples >= min_samples


@dataclass(frozen=True)
class Calibrator:
    """
    分數 → 機率的對照表。

    不可變：校準器一旦擬合就不該被改動，否則前後段的預測不可比。
    """

    bins: tuple[CalibrationBin, ...]
    base_rate: float
    """整體 P(+1)。模型必須勝過它才有價值"""

    n_samples: int
    min_samples_per_bin: int
    score_min: float
    score_max: float

    def predict(self, score: float) -> float | None:
        """
        把分數轉成校準後的機率。

        Returns:
            機率；下列情況回 `None`（**不猜測**）：
              · 分數超出訓練期見過的範圍（不外推）
              · 該分數落入的箱樣本不足

        回 `None` 的候選應被視為「無法評估」而剔除，
        不是「機率很低」。
        """
        if not np.isfinite(score):
            return None
        if score < self.score_min or score > self.score_max:
            return None

        index = find_bin(score, self.bins)
        if index is None:
            return None

        bucket = self.bins[index]
        if not bucket.is_usable(self.min_samples_per_bin):
            return None
        return bucket.empirical_prob


def fit_calibrator(
    scores: np.ndarray,
    labels: np.ndarray,
    n_bins: int = DEFAULT_N_BINS,
    min_samples_per_bin: int = DEFAULT_MIN_SAMPLES_PER_BIN,
) -> Calibrator:
    """
    用歷史分數與 triple-barrier 標籤擬合校準器。

    Args:
        scores: 模型／基準輸出的分數
        labels: triple-barrier 標籤（+1 / 0 / −1）
        n_bins: 分箱數
        min_samples_per_bin: 每箱最少樣本，不足的箱預測時回 None

    Returns:
        Calibrator

    Raises:
        CalibrationError: 長度不符、標籤非法、含非有限值或總樣本不足

    **只能用訓練期資料擬合。** 用全樣本等於讓模型知道未來的命中率分布。
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)

    if len(scores) != len(labels):
        raise CalibrationError(f"長度必須相同：{len(scores)} vs {len(labels)}")
    if not np.isfinite(scores).all():
        raise CalibrationError(
            "分數必須全為有限值。NaN 會讓分箱給出無意義的結果且不拋錯。"
        )

    invalid = set(np.unique(labels)) - VALID_LABELS
    if invalid:
        raise CalibrationError(f"標籤只能是 −1 / 0 / +1，出現 {sorted(invalid)}")

    required = n_bins * min_samples_per_bin
    if len(scores) < required:
        raise CalibrationError(
            f"樣本不足：{len(scores)} < {required}"
            f"（{n_bins} 箱 × 每箱至少 {min_samples_per_bin} 筆）"
        )

    score_min = float(scores.min())
    score_max = float(scores.max())

    edges = bin_edges(score_min, score_max, n_bins)
    hits = labels == 1

    buckets: list[CalibrationBin] = []
    for i in range(n_bins):
        in_bin = bin_mask(scores, edges, i, n_bins)
        buckets.append(
            CalibrationBin(
                lo=float(edges[i]),
                hi=float(edges[i + 1]),
                n_samples=int(in_bin.sum()),
                n_hits=int(hits[in_bin].sum()),
            )
        )

    return Calibrator(
        bins=tuple(buckets),
        base_rate=float(hits.mean()),
        n_samples=len(scores),
        min_samples_per_bin=min_samples_per_bin,
        score_min=score_min,
        score_max=score_max,
    )


def reliability_report(calibrator: Calibrator) -> str:
    """
    產出可靠度報告：並列「分數」與「實際機率」。

    過度自信（分數遠高於實際機率）是最危險的失準方向——
    它會讓不該進場的標的通過門檻。報告會明確標示。
    """
    lines = [
        "=" * 72,
        "機率校準可靠度報告",
        "=" * 72,
        "",
        f"總樣本      {calibrator.n_samples:,}",
        f"基礎 P(+1)  {calibrator.base_rate:.4f}"
        f"（模型必須勝過這個才有價值）",
        f"分數範圍    [{calibrator.score_min:.4f}, {calibrator.score_max:.4f}]",
        f"每箱門檻    {calibrator.min_samples_per_bin} 筆",
        "",
        "─" * 72,
        f"{'分數區間':<22}{'樣本數':>8}{'實際 P(+1)':>12}{'落差':>10}  備註",
        "─" * 72,
    ]

    has_overconfidence = False
    has_sparse = False

    for bucket in calibrator.bins:
        usable = bucket.is_usable(calibrator.min_samples_per_bin)
        interval = f"[{bucket.lo:.3f}, {bucket.hi:.3f}]"

        if not usable:
            has_sparse = True
            lines.append(
                f"{interval:<22}{bucket.n_samples:>8}{'n/a':>12}{'':>10}  樣本不足"
            )
            continue

        gap = bucket.mid_score - bucket.empirical_prob
        note = ""
        if gap > OVERCONFIDENCE_GAP:
            note = "⚠️ 過度自信"
            has_overconfidence = True
        elif gap < -OVERCONFIDENCE_GAP:
            note = "過度保守"

        lines.append(
            f"{interval:<22}{bucket.n_samples:>8}"
            f"{bucket.empirical_prob:>12.4f}{gap:>+10.4f}  {note}"
        )

    lines.append("─" * 72)
    lines.append("")

    if has_overconfidence:
        lines.append(
            "⚠️  偵測到過度自信：分數明顯高於實際機率。"
        )
        lines.append(
            "    未校準時這些分數會通過進場門檻，實際勝率不足以覆蓋成本。"
        )
        lines.append("")

    if has_sparse:
        lines.append(
            "ⓘ  部分分數區間樣本不足，預測時會回 None（視為無法評估而剔除），"
        )
        lines.append("    不會用相鄰箱或整體基準硬補。")
        lines.append("")

    lines.append("=" * 72)
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
# 期望報酬校準（路線 A：移動停損）
# ══════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ReturnBin:
    """單一分數區間的報酬統計"""

    lo: float
    hi: float
    n_samples: int

    mean_return: float | None
    """該箱的平均實際毛報酬；無樣本時為 None"""

    return_std: float | None
    """
    該箱的報酬標準差；樣本 < 2 時為 None。

    必須一起報。平均 +8% 但標準差 40%，與平均 +8% 標準差 2%，
    對決策的意義完全不同——前者是賭，後者才是邊際優勢。
    """

    @property
    def mid_score(self) -> float:
        return (self.lo + self.hi) / 2

    def is_usable(self, min_samples: int) -> bool:
        return self.n_samples >= min_samples


@dataclass(frozen=True)
class ReturnCalibrator:
    """
    分數 → 期望毛報酬的對照表。

    「毛」是關鍵：這裡不扣成本。成本在 `ranking` 決定進場門檻時扣，
    在 `backtest` 彙總績效時扣。校準層扣一次、下游再扣一次會重複計算。
    """

    bins: tuple[ReturnBin, ...]

    base_return: float
    """整體平均毛報酬。策略必須勝過它，否則選股不如全買"""

    n_samples: int
    min_samples_per_bin: int
    score_min: float
    score_max: float

    def predict(self, score: float) -> float | None:
        """
        把分數轉成校準後的期望毛報酬。

        Returns:
            期望毛報酬；下列情況回 `None`（**不猜測**）：
              · 分數超出訓練期見過的範圍（不外推）
              · 該分數落入的箱樣本不足

        回 `None` 的候選應被視為「無法評估」而剔除，
        不是「期望報酬為 0」。
        """
        if not np.isfinite(score):
            return None
        if score < self.score_min or score > self.score_max:
            return None

        index = find_bin(score, self.bins)
        if index is None:
            return None

        bucket = self.bins[index]
        if not bucket.is_usable(self.min_samples_per_bin):
            return None
        return bucket.mean_return

    @property
    def usable_bins(self) -> tuple[ReturnBin, ...]:
        return tuple(b for b in self.bins if b.is_usable(self.min_samples_per_bin))

    @property
    def return_spread(self) -> float | None:
        """
        最高箱與最低箱的平均報酬差距。

        這是鑑別力的直接度量：差距 < 一趟來回成本，就算排序方向
        正確也賺不回成本。可用箱少於 2 時無法計算，回 None。
        """
        usable = self.usable_bins
        if len(usable) < 2:
            return None
        means = [b.mean_return for b in usable if b.mean_return is not None]
        if len(means) < 2:
            return None
        return max(means) - min(means)

    @property
    def has_discrimination(self) -> bool:
        """差距是否大到足以覆蓋交易成本"""
        spread = self.return_spread
        return spread is not None and spread >= DISCRIMINATION_MIN_SPREAD


def fit_return_calibrator(
    scores: np.ndarray,
    returns: np.ndarray,
    n_bins: int = DEFAULT_N_BINS,
    min_samples_per_bin: int = DEFAULT_MIN_SAMPLES_PER_BIN,
) -> ReturnCalibrator:
    """
    用歷史分數與**實際毛報酬**擬合期望報酬校準器。

    Args:
        scores: 模型／規則式基準輸出的分數
        returns: 對應的實際毛報酬率（`TrailingExit.gross_return`）
        n_bins: 分箱數
        min_samples_per_bin: 每箱最少樣本，不足的箱預測時回 None

    Returns:
        ReturnCalibrator

    Raises:
        CalibrationError: 長度不符、含非有限值或總樣本不足

    **只能用訓練期資料擬合。** 用全樣本等於讓模型知道未來的報酬分布。
    """
    scores = np.asarray(scores, dtype=float)
    returns = np.asarray(returns, dtype=float)

    if len(scores) != len(returns):
        raise CalibrationError(f"長度必須相同：{len(scores)} vs {len(returns)}")
    if not np.isfinite(scores).all():
        raise CalibrationError(
            "分數必須全為有限值。NaN 會讓分箱給出無意義的結果且不拋錯。"
        )
    if not np.isfinite(returns).all():
        raise CalibrationError(
            "報酬必須全為有限值。單一 inf 會讓整箱的平均變成 inf，"
            "而且比較運算不會拋錯——必須在邊界擋掉。"
        )

    required = n_bins * min_samples_per_bin
    if len(scores) < required:
        raise CalibrationError(
            f"樣本不足：{len(scores)} < {required}"
            f"（{n_bins} 箱 × 每箱至少 {min_samples_per_bin} 筆）"
        )

    score_min = float(scores.min())
    score_max = float(scores.max())
    edges = bin_edges(score_min, score_max, n_bins)

    buckets: list[ReturnBin] = []
    for i in range(n_bins):
        in_bin = bin_mask(scores, edges, i, n_bins)
        sample = returns[in_bin]
        buckets.append(
            ReturnBin(
                lo=float(edges[i]),
                hi=float(edges[i + 1]),
                n_samples=len(sample),
                mean_return=float(sample.mean()) if len(sample) else None,
                return_std=(
                    float(statistics.stdev(sample.tolist()))
                    if len(sample) >= 2
                    else None
                ),
            )
        )

    return ReturnCalibrator(
        bins=tuple(buckets),
        base_return=float(returns.mean()),
        n_samples=len(scores),
        min_samples_per_bin=min_samples_per_bin,
        score_min=score_min,
        score_max=score_max,
    )


def return_reliability_report(calibrator: ReturnCalibrator) -> str:
    """
    產出期望報酬校準報告。

    最重要的診斷是**鑑別力**：分數排序若與實際報酬無關，整個選股
    流程沒有意義。這裡用「最高箱 − 最低箱的平均報酬差距 vs 一趟
    來回成本」直接回答，而不是只印一張看起來很專業的表。
    """
    spread = calibrator.return_spread
    lines = [
        "=" * 78,
        "期望報酬校準報告（移動停損 / 無固定目標）",
        "=" * 78,
        "",
        f"總樣本      {calibrator.n_samples:,}",
        f"基礎平均報酬 {calibrator.base_return * 100:+.2f}%"
        f"（策略必須勝過它，否則選股不如全買）",
        f"分數範圍    [{calibrator.score_min:.4f}, {calibrator.score_max:.4f}]",
        f"每箱門檻    {calibrator.min_samples_per_bin} 筆",
        "",
        "─" * 78,
        f"{'分數區間':<22}{'樣本數':>8}{'平均報酬':>12}{'標準差':>12}"
        f"{'超額':>12}",
        "─" * 78,
    ]

    has_sparse = False
    for bucket in calibrator.bins:
        interval = f"[{bucket.lo:.3f}, {bucket.hi:.3f}]"
        if not bucket.is_usable(calibrator.min_samples_per_bin) or (
            bucket.mean_return is None
        ):
            has_sparse = True
            lines.append(
                f"{interval:<22}{bucket.n_samples:>8}{'n/a':>12}{'n/a':>12}"
                f"{'':>12}  樣本不足"
            )
            continue

        std = f"{bucket.return_std * 100:.2f}%" if bucket.return_std else "n/a"
        excess = bucket.mean_return - calibrator.base_return
        lines.append(
            f"{interval:<22}{bucket.n_samples:>8}"
            f"{bucket.mean_return * 100:>11.2f}%{std:>12}"
            f"{excess * 100:>+11.2f}%"
        )

    lines.append("─" * 78)
    lines.append("")

    if spread is None:
        lines.append("ⓘ  可用箱不足 2 個，無法判斷鑑別力。")
    else:
        lines.append(
            f"最高箱 − 最低箱  {spread * 100:.2f}%"
            f"｜一趟來回成本 {DISCRIMINATION_MIN_SPREAD * 100:.2f}%"
        )
        if calibrator.has_discrimination:
            lines.append("✓  分數對報酬有鑑別力，且差距大於交易成本。")
        else:
            lines.append("⚠️  無鑑別力：分數最高與最低箱的報酬差距小於一趟來回成本。")
            lines.append("    即使排序方向正確也賺不回成本，這個分數不該用來選股。")

    lines.append("")
    if has_sparse:
        lines.append(
            "ⓘ  部分分數區間樣本不足，預測時會回 None（視為無法評估而剔除），"
        )
        lines.append("    不會用相鄰箱或整體基準硬補。")
        lines.append("")

    lines.append("=" * 78)
    return "\n".join(lines)
