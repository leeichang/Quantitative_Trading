"""因子選擇模組

支援三種方法：
1. none: Qlib 標準流程，不做選擇，依賴 LightGBM 內建機制
2. dedup: RD-Agent IC 去重複，移除高相關因子
3. ic_incremental: IC 增量選擇法（Greedy Forward Selection）

參考文獻：
- RD-Agent (Microsoft Research, 2025): https://arxiv.org/html/2505.15155v2
- Qlib (Microsoft): https://github.com/microsoft/qlib
"""

from src.services.factor_selection.base import (
    FactorSelectionResult,
    FactorSelector,
)
from src.services.factor_selection.ic_dedup import ICDeduplicator
from src.services.factor_selection.ic_incremental import ICIncrementalSelector
from src.services.factor_selection.robust import RobustFactorSelector

__all__ = [
    "FactorSelectionResult",
    "FactorSelector",
    "ICDeduplicator",
    "ICIncrementalSelector",
    "RobustFactorSelector",
]
