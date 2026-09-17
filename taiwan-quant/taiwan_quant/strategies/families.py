"""
三個策略族（D7）

    A. 動能突破    20/60 日動能 + 量能放大 + 站上均線
    B. 籌碼跟隨    外資／投信連續買超 + 融資使用率低檔
    C. 均值回歸    RSI 超賣 + 觸及布林下軌 + 未跌破長均線

三族共用同一套 triple-barrier 標記、同一套成本模型、同一套驗證流程。
唯一的差別是**分數怎麼算**——這個切分讓「哪一族有優勢」變成可比較的問題。

## 分數的意義

分數**不是機率**，是排序用的數字。校準（`validation/calibration.py`）會把
它映射成真實機率。所以絕對值不重要，重要的是：

    分數高的樣本，實際命中率必須比較高

因此每族都要有方向正確性的測試——方向錯了整個策略會反向操作，
而回測仍會跑出數字，這是最容易被忽略又最致命的錯誤。

## 值域

所有分數映射到 [0, 1]。有界是必要的：極端行情若讓分數跑到離群值，
校準分箱會被拉壞（大部分樣本擠在一箱）。

## 缺資料

回 `np.nan` 而不是猜。上游籌碼資料常常不完整，整批標的裡少數幾檔缺資料
是常態，不該讓整輪掃描中斷，也不該用 0 或均值硬補。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from taiwan_quant.features.chips import CHIPS_REQUIRED_COLUMNS, build_chips
from taiwan_quant.features.technical import build_technical

MIN_HISTORY = 200
"""最短歷史長度。最長的特徵視窗是 120 日動能，加上暖機留 200"""

PRICE_COLUMNS = ("open", "high", "low", "close", "volume")


def _squash(value: float, scale: float) -> float:
    """把無界的數值壓到 [0, 1]，0.5 為中性"""
    return float(0.5 + 0.5 * np.tanh(value * scale))


def _blend(parts: list[float], weights: list[float]) -> float:
    """加權平均；任一項為 NaN 則整體為 NaN（不可用其餘項硬補）"""
    if any(not np.isfinite(p) for p in parts):
        return float("nan")
    total = sum(weights)
    return float(sum(p * w for p, w in zip(parts, weights)) / total)


def _has_history(bars: pd.DataFrame) -> bool:
    return len(bars) >= MIN_HISTORY


def _has_columns(bars: pd.DataFrame, columns: tuple[str, ...]) -> bool:
    return set(columns).issubset(set(bars.columns))


# ══════════════════════════════════════════════════════════════
# A. 動能突破
# ══════════════════════════════════════════════════════════════


def momentum_breakout_score(bars: pd.DataFrame) -> float:
    """
    動能突破分數。

    組成：
        20 日動能      短期趨勢
        60 日動能      中期趨勢
        20 日均線偏離  目前是否站在均線之上
        20 日量能倍數  量價配合

    假設：近期強勢且帶量的標的，短期內續強的機率較高。

    Returns:
        [0, 1] 的分數；資料不足回 NaN
    """
    if not _has_history(bars) or not _has_columns(bars, PRICE_COLUMNS):
        return float("nan")

    return _momentum_from_row(build_technical(bars).iloc[-1])


def _momentum_from_row(row: pd.Series) -> float:
    """由單列技術特徵算動能分數。`momentum_breakout_score` 的核心"""
    return _blend(
        parts=[
            _squash(row["momentum_20"], 8.0),
            _squash(row["momentum_60"], 4.0),
            _squash(row["ma_ratio_20"], 20.0),
            _squash(row["volume_ratio_20"] - 1.0, 1.5),
        ],
        weights=[3.0, 2.0, 2.0, 1.0],
    )


# ══════════════════════════════════════════════════════════════
# B. 籌碼跟隨
# ══════════════════════════════════════════════════════════════


def chips_following_score(bars: pd.DataFrame) -> float:
    """
    籌碼跟隨分數。

    組成：
        外資 5/20 日買賣超比率   權重最高，動向最受關注
        投信 5 日買賣超比率      季底作帳行為明顯
        三大法人 20 日一致度      合力比單一法人有訊息量
        融資餘額相對均量（反向）  散戶追高是反向指標

    假設：法人持續且一致買超、而散戶融資未過熱的標的，續漲機率較高。

    Returns:
        [0, 1] 的分數；資料不足或缺籌碼欄位回 NaN
    """
    if not _has_history(bars):
        return float("nan")
    if not _has_columns(bars, PRICE_COLUMNS):
        return float("nan")
    if not _has_columns(bars, CHIPS_REQUIRED_COLUMNS):
        return float("nan")

    return _chips_from_row(build_chips(bars).iloc[-1])


def _chips_from_row(row: pd.Series) -> float:
    """由單列籌碼特徵算分數。`chips_following_score` 的核心"""
    # 一致度取絕對值（見 features/chips.py），要恢復方向才能當多空訊號
    strength_sign = np.sign(row["institution_strength"]) if np.isfinite(
        row["institution_strength"]
    ) else 0.0
    directional_agreement = row["institution_agreement_20"] * strength_sign

    return _blend(
        parts=[
            _squash(row["foreign_net_ratio_5"], 60.0),
            _squash(row["foreign_net_ratio_20"], 40.0),
            _squash(row["trust_net_ratio_5"], 120.0),
            _squash(directional_agreement, 2.5),
            # 融資相對均量越高分數越低 → 負號
            _squash(-np.log1p(max(row["margin_to_volume_20"], 0.0)), 0.35),
        ],
        weights=[3.0, 2.0, 1.5, 2.0, 1.5],
    )


# ══════════════════════════════════════════════════════════════
# C. 均值回歸
# ══════════════════════════════════════════════════════════════


def mean_reversion_score(bars: pd.DataFrame) -> float:
    """
    均值回歸分數。

    組成：
        RSI14（反向）           越低越超賣
        布林通道位置（反向）      越靠下軌越超賣
        20 日高低位置（反向）     越靠區間低點越超賣
        60 日均線偏離（正向）     **必須仍在長均線之上**

    最後一項是這一族與「接下墜的刀」的分界線：
    超賣只有在長期趨勢未破壞時才是機會，跌破長均線就是趨勢轉空。

    Returns:
        [0, 1] 的分數；資料不足回 NaN
    """
    if not _has_history(bars) or not _has_columns(bars, PRICE_COLUMNS):
        return float("nan")

    return _mean_reversion_from_row(build_technical(bars).iloc[-1])


def _mean_reversion_from_row(row: pd.Series) -> float:
    """由單列技術特徵算均值回歸分數。`mean_reversion_score` 的核心"""
    return _blend(
        parts=[
            # RSI 以 50 為中性，越低分數越高
            _squash(-(row["rsi_14"] - 50.0), 0.06),
            # 布林位置以 0.5 為中性，越低分數越高
            _squash(-(row["bollinger_position_20"] - 0.5), 3.0),
            _squash(-(row["high_low_position_20"] - 0.5), 3.0),
            # 長均線之上才給分——這條擋掉「接下墜的刀」
            _squash(row["ma_ratio_60"], 12.0),
        ],
        weights=[2.5, 2.0, 1.5, 3.0],
    )


# ══════════════════════════════════════════════════════════════
# 目錄
# ══════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class StrategyFamily:
    """一個策略族"""

    name: str
    score_fn: Callable[[pd.DataFrame], float]
    """單點分數：吃整段日 K，回最後一天的分數"""

    required_columns: tuple[str, ...]
    """
    需要的欄位。呼叫端可先檢查資料夠不夠，
    而不是跑到一半才發現缺欄位。
    """

    description: str

    publishable: bool = True
    """
    是否可進入推播路徑。

    `False` 代表**實測顯示它沒有優勢**，不是「還沒驗證」。族本身留在
    `STRATEGY_FAMILIES` 裡，因為回測與歷史結果的重現需要它——刪掉會讓
    先前的報告無法重跑（禁令 7、8）。

    推播腳本必須檢查這個欄位並硬拒絕。**不提供繞過開關**：要研究它就用
    診斷腳本，那些腳本直接取 `STRATEGY_FAMILIES`，不受此限制。
    """

    unpublishable_reason: str = ""
    """`publishable=False` 時的實測依據。空字串代表可推播"""

    feature_builder: Callable[[pd.DataFrame], pd.DataFrame] = build_technical
    row_scorer: Callable[[pd.Series], float] = _momentum_from_row
    """
    向量化路徑的兩個零件：特徵表怎麼建、單列怎麼算分。

    `score_fn` 每次呼叫都重建整張特徵表，只為了取最後一列。長歷史回測
    （11 年 × 586 檔 × 3 族）下那個固定開銷變成瓶頸——實測 4.0 ms/次，
    百萬次就是一小時。`score_series` 只建一次特徵表，逐列套用同一組運算。

    兩條路徑**必須給出完全相同的結果**，由
    `test_score_series_matches_score_fn_pointwise` 逐點比對。
    """

    def __post_init__(self) -> None:
        if not self.publishable and not self.unpublishable_reason.strip():
            raise ValueError(
                f"{self.name} 標記為不可推播，但沒有寫實測依據——"
                "停用一個策略族必須說明為什麼"
            )

    def score_series(self, bars: pd.DataFrame) -> pd.Series:
        """
        一次算出每一天的分數。

        Args:
            bars: 日 K（**不會被修改**）

        Returns:
            與 `bars` 同索引的分數序列。歷史不足或缺欄位的位置為 NaN。

        歷史不足的前 `MIN_HISTORY - 1` 天一律 NaN——與 `score_fn` 的
        `_has_history` 檢查一致，不可用不完整的視窗硬算。
        """
        empty = pd.Series(float("nan"), index=bars.index, dtype=float)
        if not _has_columns(bars, self.required_columns):
            return empty
        if len(bars) < MIN_HISTORY:
            return empty

        features = self.feature_builder(bars)
        scores = features.apply(self.row_scorer, axis=1).astype(float)
        scores.iloc[: MIN_HISTORY - 1] = float("nan")
        return scores.rename(None)


STRATEGY_FAMILIES: tuple[StrategyFamily, ...] = (
    StrategyFamily(
        name="動能突破",
        score_fn=momentum_breakout_score,
        required_columns=PRICE_COLUMNS,
        description="20/60 日動能 + 量能放大 + 站上均線",
        feature_builder=build_technical,
        row_scorer=_momentum_from_row,
    ),
    StrategyFamily(
        name="籌碼跟隨",
        score_fn=chips_following_score,
        required_columns=PRICE_COLUMNS + CHIPS_REQUIRED_COLUMNS,
        description="外資／投信連續買超 + 三大法人一致 + 融資未過熱",
        feature_builder=build_chips,
        row_scorer=_chips_from_row,
    ),
    StrategyFamily(
        name="均值回歸",
        score_fn=mean_reversion_score,
        required_columns=PRICE_COLUMNS,
        description="RSI 超賣 + 觸及布林下軌 + 未跌破 60 日均線",
        feature_builder=build_technical,
        row_scorer=_mean_reversion_from_row,
        publishable=False,
        unpublishable_reason=(
            "2026-09-17 對無資訊對照組實測：淨 +0.49%／趟，落在 200 組"
            "任意 4 特徵權重組合的第 10 百分位——**比 90% 的任意權重還差**，"
            "也低於持有全池的 +2.44%。"
            "依據 ../qlib-tw-trader/docs/原理說明/"
            "2026-09-17_手工分數對無資訊對照組的重算.md"
        ),
    ),
)
