"""
`scripts/diagnose_path_dependence.py` 的擾動函式測試

`perturb` 是整個對照實驗的核心：若它意外修改了傳入的標的池，
後續每個擾動會疊加在前一個上面，**實驗就從「同一基準的 N 個擾動」
變成「一條隨機漫步」**，而擺盪幅度會被高估。

所以這裡逐項斷言：不修改輸入、種子可重現、不同種子不同結果、
`seed=0` 是基準。
"""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.diagnose_path_dependence import DiagnosticError, _spread, perturb


def _members(n_dates: int = 4, n_names: int = 20) -> dict[pd.Timestamp, set[str]]:
    days = pd.bdate_range("2020-01-01", periods=n_dates)
    return {
        day: {f"{1000 + i:04d}" for i in range(n_names)}
        for day in days
    }


def test_perturb_does_not_mutate_the_input():
    """**最重要的一條。** 就地修改會讓擾動互相疊加"""
    original = _members()
    snapshot = {day: set(names) for day, names in original.items()}

    perturb(original, drop=3, seed=1)

    assert original == snapshot


def test_perturb_drops_exactly_the_requested_count():
    original = _members(n_names=20)

    result = perturb(original, drop=3, seed=1)

    for day, names in result.items():
        assert len(names) == 17, f"{day} 應剩 17 檔，得到 {len(names)}"
        assert names < original[day], "擾動後應是原集合的真子集"


def test_seed_zero_is_the_untouched_baseline():
    """
    基準也走同一條程式路徑，只是不移除任何東西——避免「基準走了
    不同的路所以不可比」這種混淆。
    """
    original = _members()

    result = perturb(original, drop=3, seed=0)

    assert result == original
    assert result is not original, "仍應回副本，不可回傳同一個物件"


def test_drop_zero_returns_the_baseline_regardless_of_seed():
    original = _members()

    assert perturb(original, drop=0, seed=7) == original


def test_same_seed_reproduces_the_same_perturbation():
    original = _members()

    first = perturb(original, drop=4, seed=11)
    second = perturb(original, drop=4, seed=11)

    assert first == second


def test_different_seeds_give_different_perturbations():
    original = _members(n_names=60)

    first = perturb(original, drop=5, seed=1)
    second = perturb(original, drop=5, seed=2)

    assert first != second


def test_each_date_is_perturbed_independently():
    """
    v6→v7 的名單變化逐日不同，所以擾動也要逐日獨立。
    每天移掉同一批會變成「縮小標的池」，那是另一個實驗。
    """
    original = _members(n_dates=6, n_names=40)

    result = perturb(original, drop=4, seed=3)
    removed = [original[day] - result[day] for day in sorted(result)]

    assert len({frozenset(r) for r in removed}) > 1, (
        "各決策日移掉的應該不完全相同"
    )


def test_a_date_with_too_few_names_is_left_alone():
    """名單比 drop 還短時不可清空——那會讓該日毫無候選"""
    day = pd.Timestamp("2020-01-01")
    original = {day: {"1000", "1001"}}

    result = perturb(original, drop=5, seed=1)

    assert result[day] == {"1000", "1001"}


def test_negative_drop_fails_fast():
    with pytest.raises(DiagnosticError, match="drop"):
        perturb(_members(), drop=-1, seed=1)


# ══════════════════════════════════════════════════════════════
# _spread
# ══════════════════════════════════════════════════════════════


def test_spread_reports_the_range_in_percentage_points():
    """手算：0.10 到 0.55 的全距是 45 個百分點"""
    result = _spread([0.10, 0.30, 0.55])

    assert result["min"] == pytest.approx(0.10)
    assert result["max"] == pytest.approx(0.55)
    assert result["range_pp"] == pytest.approx(45.0)
    assert result["median"] == pytest.approx(0.30)


def test_spread_of_a_single_value_has_zero_std():
    result = _spread([0.42])

    assert result["range_pp"] == pytest.approx(0.0)
    assert result["std_pp"] == pytest.approx(0.0)


# ══════════════════════════════════════════════════════════════
# 換倉節點對日曆的依賴（本腳本第一版的 bug）
# ══════════════════════════════════════════════════════════════


def test_rebalance_nodes_follow_whichever_calendar_is_passed():
    """
    **本腳本第一版的 bug。**

    `select_periodic_rebalances` 的節點是
    `range(0, len(calendar), rebalance_every)`，所以**傳進去的是哪一份
    日曆決定了節點落在哪裡**。第一版傳了完整 2197 天日曆而非
    `oos_calendar`，節點散佈到暖機期之前，動能突破跑出 −5.38%
    （參考量級 +300%）。

    成交筆數幾乎沒變（56 vs 57），所以任何「筆數合理」的檢查都抓不到——
    錯的是**日期**。這個測試直接釘住那件事。
    """
    from taiwan_quant.backtest.portfolio_sim import Signal
    from taiwan_quant.config.costs import Tier
    from taiwan_quant.validation.thresholds import select_periodic_rebalances

    full = list(pd.bdate_range("2020-01-01", periods=60))
    sliced = full[30:]

    def lookup(_stock_id: str, _day: pd.Timestamp) -> float:
        return 100.0

    signals = [
        Signal(
            decision_date=day,
            exit_date=full[min(index + 10, len(full) - 1)],
            stock_id=f"{1000 + rank:04d}",
            gross_return=0.01,
            rank_score=1.0 - rank / 5,
            tier=Tier.LARGE,
        )
        for index, day in enumerate(full)
        for rank in range(5)
    ]

    on_full = select_periodic_rebalances(signals, full, lookup, 10, 3)
    on_sliced = select_periodic_rebalances(signals, sliced, lookup, 10, 3)

    entries_full = {s.decision_date for s in on_full.signals}
    entries_sliced = {s.decision_date for s in on_sliced.signals}

    assert entries_full != entries_sliced, (
        "節點必須隨傳入日曆改變，否則這個 bug 無法被偵測"
    )
