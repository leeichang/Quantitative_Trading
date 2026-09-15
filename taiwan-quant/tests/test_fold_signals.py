"""
單一 fold 的訊號篩選測試

## 為什麼把這段抽出來

門檻判定原本只寫在 `scripts/validate_oos_trailing.py` 的 walk-forward
迴圈裡。CPCV 要用同一套門檻跑不同的切分，兩邊各留一份的話，任務 E
調了門檻之後 CPCV 量到的就不是同一個東西了——而且**不會拋錯**，
只會讓兩份報告悄悄講不同的故事。

## 門檻的三道關卡

```
1. 校準器給得出預期報酬      predict() 回 None 就跳過（分數落在箱外）
2. 淨期望為正                expected − round_trip_cost > 0
3. 優勢大於 edge_z 個標準誤   否則與 0 在統計上分不開
```

第 3 關是 `03_待辦與改進方向.md` 第 2 項的核心：60 日持有下期望報酬遠
大於成本，**幾乎所有候選都過得了第 2 關**，門檻等於沒在篩選。
"""

from __future__ import annotations

import pandas as pd
import pytest

from taiwan_quant.config.costs import Tier
from taiwan_quant.validation.fold_signals import (
    CandidateProtocol,
    select_fold_signals,
)


class FakeCandidate:
    """最小的 Decision 替身——只要有這三個屬性就夠"""

    def __init__(self, stock_id: str, score: float, gross_return: float) -> None:
        self.stock_id = stock_id
        self.score = score
        self.gross_return = gross_return


class FakeBin:
    """`binning.Bin` 協定要的三個欄位"""

    def __init__(self, lo: float, hi: float) -> None:
        self.lo = lo
        self.hi = hi
        self.n_samples = 100


class FakeCalibrator:
    """
    回傳預設好的預期報酬。

    箱索引是靠 `binning.find_bin(score, bins)` 查的，不是校準器的方法，
    所以這裡要提供真的 `bins`——自己造一個 `bin_index()` 會繞過那條路徑，
    測不到實際會走的程式碼。

    ```
    箱 0  [0.0, 0.3)   ← 分數 0.1
    箱 1  [0.3, 0.7)   ← 分數 0.5
    箱 2  [0.7, 1.0]   ← 分數 0.9
    ```
    """

    def __init__(self, predictions: dict[float, float | None]) -> None:
        self._predictions = predictions
        self.bins = tuple(
            FakeBin(lo, hi) for lo, hi in ((0.0, 0.3), (0.3, 0.7), (0.7, 1.0))
        )

    def predict(self, score: float) -> float | None:
        return self._predictions.get(score)


def candidates() -> dict[pd.Timestamp, list[CandidateProtocol]]:
    day = pd.Timestamp("2024-01-02")
    return {
        day: [
            FakeCandidate("2330", score=0.9, gross_return=0.20),
            FakeCandidate("2317", score=0.5, gross_return=0.05),
            FakeCandidate("1301", score=0.1, gross_return=-0.03),
        ]
    }


# ══════════════════════════════════════════════════════════════
# 三道關卡，逐一驗
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_rejects_when_calibrator_has_no_prediction() -> None:
    """分數落在箱外時 predict() 回 None，該筆直接跳過，不可當成 0。"""
    calibrator = FakeCalibrator(predictions={0.9: None, 0.5: None, 0.1: None})

    picked = select_fold_signals(
        candidates(), calibrator, se_by_bin={0: 0.01, 1: 0.01, 2: 0.01},
        tiers={}, edge_z=1.0,
    )

    assert picked == []


@pytest.mark.unit
def test_rejects_when_net_expectation_is_not_positive() -> None:
    """
    手算：MID 級來回成本 1.271%。

    預期毛報酬 1.0% → 淨 −0.271% → 淘汰
    預期毛報酬 5.0% → 淨 +3.729% → 通過第 2 關
    """
    calibrator = FakeCalibrator(predictions={0.9: 0.05, 0.5: 0.01, 0.1: 0.01})

    picked = select_fold_signals(
        candidates(), calibrator, se_by_bin={0: 0.001, 1: 0.001, 2: 0.001},
        tiers={}, edge_z=1.0,
    )

    assert [item[1].stock_id for item in picked] == ["2330"]


@pytest.mark.unit
def test_rejects_when_edge_is_within_one_standard_error() -> None:
    """
    淨期望 +3.729%，標準誤 5% → 3.729% < 1.0 × 5% → 淘汰。

    這一關是「統計上分不開 0」的判定，不是報酬高低的判定。
    """
    calibrator = FakeCalibrator(predictions={0.9: 0.05, 0.5: 0.0, 0.1: 0.0})

    picked = select_fold_signals(
        candidates(), calibrator, se_by_bin={0: 0.05, 1: 0.05, 2: 0.05},
        tiers={}, edge_z=1.0,
    )

    assert picked == []


@pytest.mark.unit
def test_raising_edge_z_tightens_the_gate() -> None:
    """
    同一筆候選（淨 +3.729%、標準誤 2%）：

    ```
    edge_z = 1.0   門檻 2.0%   通過
    edge_z = 2.0   門檻 4.0%   淘汰
    ```

    任務 E 的方案 E2 就是拉這個數字。實測拉到 2.5 對訊號／成交比幾乎
    沒有影響（435:1 → 434~478:1），因為擋掉訊號的是槽位不是門檻。
    """
    calibrator = FakeCalibrator(predictions={0.9: 0.05, 0.5: 0.0, 0.1: 0.0})
    se = {0: 0.02, 1: 0.02, 2: 0.02}

    loose = select_fold_signals(candidates(), calibrator, se, tiers={}, edge_z=1.0)
    strict = select_fold_signals(candidates(), calibrator, se, tiers={}, edge_z=2.0)

    assert len(loose) == 1
    assert strict == []


@pytest.mark.unit
def test_missing_standard_error_is_rejected_not_defaulted() -> None:
    """
    某一箱樣本不足、算不出標準誤時必須淘汰。

    **這是曾經的 bug 型態**：把 None 當成 0 會讓門檻自動通過，
    而且不會拋錯——只會讓結果變好看。
    """
    calibrator = FakeCalibrator(predictions={0.9: 0.05, 0.5: 0.0, 0.1: 0.0})

    picked = select_fold_signals(
        candidates(), calibrator, se_by_bin={0: 0.001, 1: 0.001, 2: None},
        tiers={}, edge_z=1.0,
    )

    assert picked == []


@pytest.mark.unit
def test_cost_uses_the_tier_of_each_stock() -> None:
    """
    成本必須逐檔查分層，不可全用 MID（禁令 3、4）。

    手算：LARGE 來回 1.0710%、MID 1.2710%，差 0.2 pp。
    預期毛報酬 1.2%、標準誤 0.1%、edge_z = 1.0：

    ```
    LARGE   1.2% − 1.0710% = +0.129%  >  0.1%   通過
    MID     1.2% − 1.2710% = −0.071%  ≤  0      第 2 關就淘汰
    ```

    同樣一筆候選，分層查錯就會多出一筆不該有的訊號。
    """
    calibrator = FakeCalibrator(predictions={0.9: 0.012, 0.5: 0.0, 0.1: 0.0})
    se = {0: 0.001, 1: 0.001, 2: 0.001}

    as_large = select_fold_signals(
        candidates(), calibrator, se, tiers={"2330": Tier.LARGE}, edge_z=1.0
    )
    as_mid = select_fold_signals(
        candidates(), calibrator, se, tiers={"2330": Tier.MID}, edge_z=1.0
    )

    assert len(as_large) == 1
    assert as_mid == []


@pytest.mark.unit
def test_returns_net_expectation_not_raw_score() -> None:
    """
    回傳的第三個欄位是**淨期望報酬**，組合層要用它排序。
    回傳原始分數的話，排序會變成「模型多有信心」而不是「賺多少」。
    """
    calibrator = FakeCalibrator(predictions={0.9: 0.05, 0.5: 0.0, 0.1: 0.0})

    picked = select_fold_signals(
        candidates(), calibrator, se_by_bin={0: 0.001, 1: 0.001, 2: 0.001},
        tiers={}, edge_z=1.0,
    )

    _, _, net = picked[0]
    assert net == pytest.approx(0.05 - 0.01271, abs=1e-6)


@pytest.mark.unit
def test_empty_test_dates_yield_nothing() -> None:
    calibrator = FakeCalibrator(predictions={})

    assert select_fold_signals({}, calibrator, {}, tiers={}, edge_z=1.0) == []
