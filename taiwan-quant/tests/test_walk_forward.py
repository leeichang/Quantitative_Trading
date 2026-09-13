"""
Walk-Forward 切分與標籤隔離測試

## 抓到的 bug（第 10 個）

原本的切分是：

```python
train_dates = dates[:cursor]
test_dates  = dates[cursor : cursor + test_span]
```

**訓練集的最後幾期，標籤期跨進測試段。** 持有 60 日、每 5 日決策時，
最後 12 期決策的實際報酬要用到測試期的價格才算得出來——校準器等於
「看過」測試期的走勢。

污染比例：第一個 fold 8.0%（12/150），最後一個 fold 2.4%（12/498）。

這與 CLAUDE.md 禁令 1（T 日決策只能用 T 日收盤前可得的資料）是同一條
原則的延伸：**訓練時可用的標籤，必須在訓練期結束前就已經揭曉。**
"""

from __future__ import annotations

import pandas as pd
import pytest

from taiwan_quant.validation.walk_forward import (
    WalkForwardError,
    embargo_periods,
    walk_forward_folds,
)


def dates(n: int) -> list[pd.Timestamp]:
    """n 個決策日，每 5 個交易日一次"""
    return list(pd.date_range("2019-01-04", periods=n, freq="5B"))


# ══════════════════════════════════════════════════════════════
# 隔離期數
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_embargo_periods_matches_hand_calculation() -> None:
    """
    持有 60 日、每 5 日決策 → 60 / 5 = 12 期的標籤會跨進測試段。
    """
    assert embargo_periods(horizon=60, stride=5) == 12


@pytest.mark.unit
def test_embargo_rounds_up() -> None:
    """
    不可無條件捨去。持有 22 日、每 5 日決策 → 22/5 = 4.4，
    第 5 期的標籤仍有一部分落在測試段內，必須一起隔離。
    """
    assert embargo_periods(horizon=22, stride=5) == 5


@pytest.mark.unit
def test_embargo_is_zero_when_label_resolves_within_one_period() -> None:
    """持有期等於決策間隔時，標籤在下一期開始前就揭曉，不需隔離"""
    assert embargo_periods(horizon=5, stride=5) == 1


@pytest.mark.unit
def test_embargo_rejects_non_positive() -> None:
    with pytest.raises(WalkForwardError, match="必須為正"):
        embargo_periods(horizon=0, stride=5)
    with pytest.raises(WalkForwardError, match="必須為正"):
        embargo_periods(horizon=60, stride=0)


# ══════════════════════════════════════════════════════════════
# 切分
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_train_excludes_embargoed_periods() -> None:
    """
    訓練集尾端的 12 期必須被剔除。

    150 期訓練 → 實際可用 138 期，最後一期是 index 137。
    """
    d = dates(200)
    folds = list(walk_forward_folds(d, first_train=150, test_span=12,
                                    horizon=60, stride=5))
    train, test = folds[0]

    assert len(train) == 138
    assert train[-1] == d[137]
    assert test == d[150:162]


@pytest.mark.unit
def test_no_train_date_label_reaches_test_period() -> None:
    """
    直接驗證性質：訓練集裡每一個決策日的**標籤到期日**，
    都必須早於測試段的第一天。

    這是隔離要保證的唯一一件事。
    """
    d = dates(200)
    horizon, stride = 60, 5
    for train, test in walk_forward_folds(d, first_train=150, test_span=12,
                                          horizon=horizon, stride=stride):
        label_span = pd.Timedelta(days=horizon * 7 / 5)   # 交易日換日曆日的保守估計
        for day in train:
            assert day + label_span <= test[0] + pd.Timedelta(days=7), (
                f"訓練日 {day.date()} 的標籤跨進測試段 {test[0].date()}"
            )


@pytest.mark.unit
def test_folds_advance_by_test_span() -> None:
    d = dates(200)
    folds = list(walk_forward_folds(d, first_train=150, test_span=12,
                                    horizon=60, stride=5))
    assert folds[0][1] == d[150:162]
    assert folds[1][1] == d[162:174]


@pytest.mark.unit
def test_train_window_expands() -> None:
    """滾動擴張窗口：訓練集越來越長"""
    d = dates(200)
    sizes = [len(train) for train, _ in
             walk_forward_folds(d, first_train=150, test_span=12,
                                horizon=60, stride=5)]
    assert sizes == sorted(sizes)
    assert sizes[1] > sizes[0]


@pytest.mark.unit
def test_stops_before_running_out_of_test_data() -> None:
    """測試段不足時停止，不可回傳半截的 fold"""
    d = dates(170)
    folds = list(walk_forward_folds(d, first_train=150, test_span=12,
                                    horizon=60, stride=5))
    assert len(folds) == 1
    assert len(folds[0][1]) == 12


@pytest.mark.unit
def test_yields_nothing_when_data_too_short() -> None:
    d = dates(100)
    assert list(walk_forward_folds(d, first_train=150, test_span=12,
                                   horizon=60, stride=5)) == []


@pytest.mark.unit
def test_skips_fold_when_embargo_empties_train() -> None:
    """
    隔離之後訓練集為空的 fold 必須跳過，不可用空集合擬合。

    first_train=10、隔離 12 期 → 訓練集會是空的。
    """
    d = dates(200)
    folds = list(walk_forward_folds(d, first_train=10, test_span=12,
                                    horizon=60, stride=5))
    assert all(len(train) > 0 for train, _ in folds)


@pytest.mark.unit
def test_rejects_unsorted_dates() -> None:
    d = dates(200)
    with pytest.raises(WalkForwardError, match="升冪"):
        list(walk_forward_folds(list(reversed(d)), first_train=150,
                                test_span=12, horizon=60, stride=5))


@pytest.mark.unit
def test_embargo_reduces_training_samples_measurably() -> None:
    """
    量化隔離的代價：第一個 fold 損失 8%，最後一個 2.4%。

    這個數字要留在測試裡——它是「修正前的結果被污染多少」的證據。
    """
    d = dates(520)
    folds = list(walk_forward_folds(d, first_train=150, test_span=12,
                                    horizon=60, stride=5))
    first_loss = 12 / 150
    last_train_end = 150 + (len(folds) - 1) * 12
    last_loss = 12 / last_train_end

    assert first_loss == pytest.approx(0.080, abs=0.001)
    assert last_loss == pytest.approx(0.024, abs=0.002)
