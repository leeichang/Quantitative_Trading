"""重複模擬的穩健彙總。"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass


@dataclass(frozen=True)
class TrialSummary:
    """多次隨機模擬的中位數摘要。"""

    n_trials: int
    median_total_return: float
    median_max_drawdown: float


def summarize_trials(
    total_returns: list[float], max_drawdowns: list[float]
) -> TrialSummary:
    """以中位數彙總同一組設定的隨機模擬。"""
    if not total_returns:
        raise ValueError("trial 不可為空")
    if len(total_returns) != len(max_drawdowns):
        raise ValueError(
            f"報酬與回撤長度必須相同：{len(total_returns)} vs {len(max_drawdowns)}"
        )
    if not all(math.isfinite(value) for value in total_returns + max_drawdowns):
        raise ValueError("trial 數值必須全部有限")
    return TrialSummary(
        n_trials=len(total_returns),
        median_total_return=float(statistics.median(total_returns)),
        median_max_drawdown=float(statistics.median(max_drawdowns)),
    )
