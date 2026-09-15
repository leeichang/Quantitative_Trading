"""
單一 fold 的訊號篩選

## 為什麼要有這個模組

門檻判定原本只寫在 `scripts/validate_oos_trailing.py` 的 walk-forward
迴圈裡。CPCV 要用**同一套門檻**跑不同的切分，兩邊各留一份的話，任務 E
調了門檻之後 CPCV 量到的就不是同一個東西了。

而且這種不一致**不會拋錯**——只會讓兩份報告悄悄講不同的故事。

⚠️ `validate_oos_trailing.py` 目前還留著自己的一份（那個檔案正在被改，
先不動它避免衝突）。**合併時要把它換成呼叫這裡**，否則 DRY 破口還在。

## 三道關卡

```
1. 校準器給得出預期報酬       predict() 回 None 就跳過（分數落在箱外）
2. 淨期望為正                 expected − round_trip_cost > 0
3. 優勢大於 edge_z 個標準誤    否則與 0 在統計上分不開
```

第 3 關是 `03_待辦與改進方向.md` 第 2 項的核心。實測（Codex 任務 E）：

```
籌碼跟隨 × 60 日｜訊號 44,319 筆，成交 102 筆   = 435 : 1
edge_z 1.0 → 2.5                              = 434 ~ 478 : 1
```

**拉高門檻幾乎沒有影響。** 60 日持有下期望報酬遠大於成本，幾乎所有候選
都過得了第 2 關；真正擋掉 99.8% 訊號的是組合層的槽位，不是這裡的門檻。

## 標準誤缺失一律淘汰

某一箱樣本不足、算不出標準誤時**必須淘汰**，不可當成 0。

把 None 當 0 會讓第 3 關自動通過，而且不會拋錯——這是本專案已經發生過
的 bug 型態（`derive_width` 回 NaN 導致 R:R 門檻靜默通過）。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

import pandas as pd

from taiwan_quant.config.costs import DEFAULT, Tier
from taiwan_quant.validation.binning import find_bin


class CandidateProtocol(Protocol):
    """訊號篩選只需要這三個欄位——不綁死 `Decision` 的其餘結構"""

    stock_id: str
    score: float
    gross_return: float


class CalibratorProtocol(Protocol):
    """
    `ReturnCalibrator` 的最小介面。

    箱索引用 `binning.find_bin(score, bins)` 自由函式查，不是校準器的
    方法——所以這裡要的是 `bins` 屬性而不是 `bin_index()`。
    """

    bins: tuple

    def predict(self, score: float) -> float | None: ...


def select_fold_signals(
    test_by_date: Mapping[pd.Timestamp, Sequence[CandidateProtocol]],
    calibrator: CalibratorProtocol,
    se_by_bin: Mapping[int, float | None],
    tiers: Mapping[str, Tier],
    edge_z: float,
) -> list[tuple[pd.Timestamp, CandidateProtocol, float]]:
    """
    篩出這個 fold 裡通過門檻的訊號。

    Args:
        test_by_date: 測試期每個決策日的候選清單
        calibrator: 已在訓練集擬合好的校準器
        se_by_bin: 每一箱的標準誤；`None` 代表樣本不足
        tiers: 每檔股票的流動性分層，查不到時視為 `Tier.MID`
        edge_z: 淨期望要超過幾個標準誤

    Returns:
        `(決策日, 候選, 淨期望報酬)`，依決策日與輸入順序排列

    **不在這裡挑 Top N，也不做持倉不重疊**——那是組合層的事
    （`backtest/portfolio_sim.py` 用槽位去消化）。這裡只回答
    「哪些訊號在統計上站得住」。

    第三個欄位回傳的是**淨期望報酬**不是原始分數：組合層要用它排序，
    用分數排會變成「模型多有信心」而不是「賺多少」。
    """
    picked: list[tuple[pd.Timestamp, CandidateProtocol, float]] = []

    for decision_date in sorted(test_by_date):
        for candidate in test_by_date[decision_date]:
            expected = calibrator.predict(candidate.score)
            if expected is None:
                continue

            cost = DEFAULT.round_trip_rate(tiers.get(candidate.stock_id, Tier.MID))
            net = expected - cost
            if net <= 0:
                continue

            bin_index = find_bin(candidate.score, calibrator.bins)
            se = se_by_bin.get(bin_index) if bin_index is not None else None
            if se is None or net < edge_z * se:
                continue

            picked.append((decision_date, candidate, net))

    return picked


def standard_errors_by_bin(calibrator_bins: Sequence) -> dict[int, float | None]:
    """
    每一箱的平均數標準誤 `std / sqrt(n)`。

    Args:
        calibrator_bins: `ReturnCalibrator.bins`

    Returns:
        箱索引 → 標準誤；標準差為 0 或缺值時回 `None`

    回 `None` 而不是 0，是為了讓 `select_fold_signals` 淘汰那一箱。
    標準差為 0 通常代表樣本太少或全部相同，不是「完全沒有不確定性」。
    """
    import numpy as np

    return {
        i: (float(b.return_std / np.sqrt(b.n_samples)) if b.return_std else None)
        for i, b in enumerate(calibrator_bins)
    }
