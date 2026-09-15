"""
CPCV 包裝層測試

## 這層在做什麼

上游的 `combinatorial_purged_splits` 吃的是「樣本數 + 標籤結束索引」。
本專案手上的是「決策日清單 + 持有交易日數 + 決策間隔」。這層負責翻譯，
以及把 15 條路徑的結果整理成可以寫進報告的分布。

## 為什麼不直接呼叫上游

翻譯錯了不會拋錯，只會讓 purge 少剔除幾期，結果變好看。
`label_end_indices` 必須有手算值測試——它是這層唯一會算錯的地方。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from taiwan_quant.validation.cpcv import (
    CPCVError,
    PathDistribution,
    cpcv_folds,
    label_end_indices,
    summarize_paths,
)


def dates(n: int) -> list[pd.Timestamp]:
    """n 個決策日，每 5 個交易日一次"""
    return list(pd.date_range("2019-01-04", periods=n, freq="5B"))


# ══════════════════════════════════════════════════════════════
# 單位翻譯：交易日 → 決策期索引
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_label_end_indices_matches_hand_calculation() -> None:
    """
    手算：持有 60 日、每 5 日決策 → 標籤跨 ceil(60/5) = 12 個決策期。

    20 期的情形：

    ```
    i       0   1   2  ...   7    8    9  ...  19
    到期   12  13  14  ...  19   19   19  ...  19
                                 ^
                                 超過尾端的一律夾到最後一期
    ```
    """
    ends = label_end_indices(n_periods=20, horizon=60, stride=5)

    assert ends[0] == 12
    assert ends[7] == 19
    assert ends[8] == 19       # min(20, 19)
    assert ends[19] == 19
    assert len(ends) == 20


@pytest.mark.unit
def test_label_end_uses_ceiling_not_floor() -> None:
    """
    持有 22 日、每 5 日決策 → 22/5 = 4.4，要進位成 5。

    捨去會讓第 5 期的標籤有一部分留在訓練集裡——那正是隔離要擋的。
    """
    ends = label_end_indices(n_periods=30, horizon=22, stride=5)

    assert ends[0] == 5


@pytest.mark.unit
def test_label_end_rejects_non_positive() -> None:
    with pytest.raises(CPCVError, match="必須為正"):
        label_end_indices(n_periods=20, horizon=0, stride=5)
    with pytest.raises(CPCVError, match="必須為正"):
        label_end_indices(n_periods=0, horizon=60, stride=5)


# ══════════════════════════════════════════════════════════════
# 切分
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_yields_binomial_number_of_paths() -> None:
    """C(6,2) = 15 條路徑，對照單一 walk-forward 的 1 條。"""
    folds = list(cpcv_folds(dates(370), horizon=60, stride=5))

    assert len(folds) == 15


@pytest.mark.unit
def test_folds_return_dates_not_indices() -> None:
    """
    呼叫端拿到的要是決策日，不是索引——索引還要自己換算就等著換錯。
    """
    decision_dates = dates(120)
    train, test = next(iter(cpcv_folds(decision_dates, horizon=60, stride=5)))

    assert all(isinstance(day, pd.Timestamp) for day in train[:3])
    assert all(day in decision_dates for day in test)
    assert not set(train) & set(test)


@pytest.mark.unit
def test_rejects_too_few_periods_for_the_split() -> None:
    """
    決策期數少於分組數時要明確拒絕，不可回傳空塊硬跑。
    """
    with pytest.raises(CPCVError):
        list(cpcv_folds(dates(4), horizon=60, stride=5, n_groups=6))


@pytest.mark.unit
def test_rejects_unsorted_dates() -> None:
    d = dates(120)
    with pytest.raises(CPCVError, match="升冪"):
        list(cpcv_folds(list(reversed(d)), horizon=60, stride=5))


# ══════════════════════════════════════════════════════════════
# 路徑分布
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_summarize_paths_matches_hand_calculation() -> None:
    """
    手算 [1, 2, 3, 4, 5]（單位：倍數報酬）：

    ```
    中位數   3.0
    全距     5 − 1 = 4.0
    p05      線性內插位置 0.05 × (5−1) = 0.2 → 1 + 0.2 × (2−1) = 1.2
    p95      位置 0.95 × 4 = 3.8 → 4 + 0.8 × (5−4) = 4.8
    ```
    """
    dist = summarize_paths([1.0, 2.0, 3.0, 4.0, 5.0])

    assert dist.n_paths == 5
    assert dist.median == pytest.approx(3.0)
    assert dist.spread == pytest.approx(4.0)
    assert dist.p05 == pytest.approx(1.2)
    assert dist.p95 == pytest.approx(4.8)


@pytest.mark.unit
def test_spread_is_the_acceptance_number_for_task_e() -> None:
    """
    任務 E 的驗收看的是全距。

    v6/v7 那次的擺盪是 −831 pp，驗收要求降到數十 pp——
    所以 `spread` 必須以百分點為單位直接可讀，不要藏在 describe() 裡。
    """
    dist = summarize_paths([0.15, 0.22, 0.31])

    assert dist.spread == pytest.approx(0.16)


@pytest.mark.unit
def test_summarize_rejects_empty() -> None:
    with pytest.raises(CPCVError, match="至少"):
        summarize_paths([])


@pytest.mark.unit
def test_describe_reports_the_caveat() -> None:
    """
    15 條路徑**不是** 15 個獨立樣本。這句話必須印在報告裡，
    否則下一個人會把它當成「證據多了 14 倍」。
    """
    text = summarize_paths([0.1, 0.2, 0.3]).describe()

    assert "獨立樣本" in text
    assert "否定" in text


@pytest.mark.unit
def test_distribution_is_immutable() -> None:
    """回傳新物件，不可就地改寫（pandas 3.0 copy-on-write）。"""
    dist = summarize_paths([0.1, 0.2])

    assert isinstance(dist, PathDistribution)
    with pytest.raises(AttributeError):
        dist.median = 0.0       # type: ignore[misc]
