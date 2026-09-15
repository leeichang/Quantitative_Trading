"""
平手排序測試

## 這個模組要修的 bug（第 12 個）

三個地方都用同一個排序鍵：

```python
sorted(candidates, key=lambda s: (-s.rank_score, s.stock_id))
#                                               ^^^^^^^^^^^ bug
```

看起來無害，實測不是。動能突破在開發集有 33,900 筆通過門檻的候選，
但**只有 12 個相異 `rank_score`**：

```
CALIBRATION_BINS = 6          ReturnCalibrator.predict() 回傳箱平均值
× 2 個流動性分層（成本不同）
= 12 個相異值
```

從 ~150 檔選 Top 3，第 3 名平均有 **3.8 檔同分**：

```
2019-06-27  候選 146｜與第 3 名同分 6 檔｜選中 ['2303', '2379', '1102']
2021-02-18  候選 144｜與第 3 名同分 6 檔｜選中 ['2301', '2303', '2344']
```

**Top 3 裡通常有 1~2 個位置是股票代號決定的，不是訊號決定的。**

## 為什麼字母序是最糟的選擇

| | 與報酬相關？ | 可重現？ | 可量測任意性？ |
|---|---|---|---|
| 股票代號 | 否，但**與公司特性系統性相關**（低號＝老牌大型股） | 是 | **否** |
| 確定性抖動 | 否 | 是 | **是**（換 seed 重跑） |

代號不只是無資訊，它是**有偏**的：台股代號與上市年份、產業、規模都相關，
等於在排序裡偷偷塞進一個未宣告的因子。

而且它讓任意性**無法量測**。抖動加 seed 之後，換個 seed 重跑就知道
有多少結果是平手運氣——這正是 CPCV 想回答的問題。

## 為什麼不用原始分數當次要鍵

`Signal` 身上沒有原始分數，只有經過分箱的 `rank_score`。加欄位要改所有
建構點，影響面太大。而且用原始分數等於假設「校準器分不出來的差異，
原始分數分得出來」——那是沒有證據的。
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from taiwan_quant.ranking.tie_break import (
    DEFAULT_TIE_SEED,
    deterministic_jitter,
    ordering_key,
)


class FakeSignal:
    def __init__(self, stock_id: str, rank_score: float) -> None:
        self.stock_id = stock_id
        self.rank_score = rank_score


# ══════════════════════════════════════════════════════════════
# 抖動本身
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_jitter_is_stable_for_the_same_inputs() -> None:
    """同一組 (代號, seed) 每次都要給同一個值，否則回測不可重現。"""
    first = deterministic_jitter("2330", seed=42)
    second = deterministic_jitter("2330", seed=42)

    assert first == second


@pytest.mark.unit
def test_jitter_stays_in_the_unit_interval() -> None:
    values = [deterministic_jitter(f"{i:04d}", seed=7) for i in range(200)]

    assert all(0.0 <= v < 1.0 for v in values)


@pytest.mark.unit
def test_different_seeds_give_different_orderings() -> None:
    """
    換 seed 要真的換排序——否則「量測平手任意性」這件事做不到。
    """
    ids = [f"{i:04d}" for i in range(50)]
    by_seed_1 = sorted(ids, key=lambda s: deterministic_jitter(s, seed=1))
    by_seed_2 = sorted(ids, key=lambda s: deterministic_jitter(s, seed=2))

    assert by_seed_1 != by_seed_2


@pytest.mark.unit
def test_jitter_survives_a_new_process() -> None:
    """
    **必須用穩定雜湊，不可用內建 `hash()`。**

    CPython 的 `hash(str)` 每個行程都會換 salt（PYTHONHASHSEED），
    同一份程式今天跑跟明天跑會得到不同的排序——回測就不可重現了。

    這個測試真的另開一個行程來驗。
    """
    code = (
        "import sys; sys.path.insert(0, '.');"
        "from taiwan_quant.ranking.tie_break import deterministic_jitter;"
        "print(repr(deterministic_jitter('2330', seed=42)))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )

    assert float(result.stdout.strip()) == deterministic_jitter("2330", seed=42)


# ══════════════════════════════════════════════════════════════
# 沒有字母序偏差
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_low_stock_codes_do_not_win_more_than_chance() -> None:
    """
    全部同分時，低號碼的中選率應該接近 50%，不是 100%。

    實測舊行為：95% 的成交集中在代號 < 2500，而候選只有 49%。
    台股代號與上市年份、產業、規模相關——用它排序等於在模型裡偷偷
    塞進一個沒有宣告的因子。
    """
    ids = [f"{1000 + i * 7:04d}" for i in range(400)]
    half = len(ids) // 2
    low_codes = set(ids[:half])

    winners = sorted(ids, key=lambda s: deterministic_jitter(s, DEFAULT_TIE_SEED))[:100]
    low_share = sum(1 for sid in winners if sid in low_codes) / len(winners)

    assert 0.35 < low_share < 0.65, f"低號碼中選率 {low_share:.0%}，偏離隨機"


# ══════════════════════════════════════════════════════════════
# 排序鍵
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_rank_score_still_dominates() -> None:
    """
    抖動只在平手時作用。分數高的永遠排前面——這條不能被破壞。
    """
    weak = FakeSignal("0001", rank_score=0.01)
    strong = FakeSignal("9999", rank_score=0.09)

    ordered = sorted([weak, strong], key=ordering_key)

    assert [s.stock_id for s in ordered] == ["9999", "0001"]


@pytest.mark.unit
def test_ties_are_broken_by_jitter_not_by_code() -> None:
    """
    同分時的順序要跟著抖動走，不是跟著代號走。

    取一組同分候選，確認排序**不等於**代號升冪。
    """
    tied = [FakeSignal(f"{1000 + i:04d}", rank_score=0.05) for i in range(20)]

    ordered = [s.stock_id for s in sorted(tied, key=ordering_key)]

    assert ordered != sorted(ordered)


@pytest.mark.unit
def test_ordering_is_reproducible() -> None:
    """同一組輸入、同一個 seed，排序每次都一樣。"""
    tied = [FakeSignal(f"{2000 + i:04d}", rank_score=0.05) for i in range(30)]

    first = [s.stock_id for s in sorted(tied, key=ordering_key)]
    second = [s.stock_id for s in sorted(tied, key=ordering_key)]

    assert first == second


@pytest.mark.unit
def test_seed_can_be_varied_to_measure_tie_luck() -> None:
    """
    換 seed 要換出不同的 Top 3——這是量測「多少結果來自平手運氣」的手段。
    """
    tied = [FakeSignal(f"{3000 + i:04d}", rank_score=0.05) for i in range(40)]

    top_a = [s.stock_id for s in sorted(tied, key=lambda s: ordering_key(s, seed=1))][:3]
    top_b = [s.stock_id for s in sorted(tied, key=lambda s: ordering_key(s, seed=2))][:3]

    assert top_a != top_b
