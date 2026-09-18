"""
`scripts/record_forward_events.py` 的測試

兩件事會靜默出錯：

1. **`spike_names` 的排序**——若沒有依分數降冪，`rank` 欄位就沒有意義，
   而 `N_POSITIONS` 截斷會砍掉最強的訊號而不是最弱的。
2. **虛無抽樣的決定性**——種子綁決策日。若不決定性，同一天重跑會抽到
   不同名單，而 `INSERT OR IGNORE` 會讓第一次的結果永久固定、
   第二次靜默消失，帳本就無法重現。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.record_forward_events import (
    MARGIN_WINDOW,
    RecordError,
    spike_names,
)


def _frames(n_days: int = 120, n_names: int = 4):
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    cols = [f"S{i}" for i in range(n_names)]
    closes = pd.DataFrame(100.0, index=idx, columns=cols)
    # 融資餘額平穩，最後一天讓 S2 > S0 > 其餘
    margin = pd.DataFrame(1000.0, index=idx, columns=cols)
    return closes, margin, idx, cols


def test_spike_names_sorts_by_score_descending():
    """
    **rank 的意義靠這條。** N_POSITIONS 截斷取前面，所以排序錯了會砍掉
    最強的訊號。
    """
    closes, margin, idx, cols = _frames()
    last = idx[-1]
    margin.loc[last, "S0"] = 1000.0 + 300.0
    margin.loc[last, "S2"] = 1000.0 + 900.0

    # min_observations 設小，讓門檻在最後一天已生效
    import scripts.record_forward_events as mod

    original = mod.MIN_OBSERVATIONS
    mod.MIN_OBSERVATIONS = 50
    try:
        fired = spike_names(closes, margin, last)
    finally:
        mod.MIN_OBSERVATIONS = original

    assert fired[:2] == ["S2", "S0"], f"應依分數降冪，得到 {fired}"


def test_spike_names_returns_empty_when_nothing_crosses():
    """事件型策略常常空手。回空清單，不可回全部或拋錯"""
    closes, margin, idx, _ = _frames()

    import scripts.record_forward_events as mod

    original = mod.MIN_OBSERVATIONS
    mod.MIN_OBSERVATIONS = 50
    try:
        fired = spike_names(closes, margin, idx[-1])
    finally:
        mod.MIN_OBSERVATIONS = original

    assert fired == []


def test_spike_names_rejects_a_date_outside_the_index():
    closes, margin, _, _ = _frames()
    with pytest.raises(RecordError, match="不在價格索引"):
        spike_names(closes, margin, pd.Timestamp("1999-01-01"))


def test_margin_window_is_long_enough_to_be_a_baseline():
    """視窗太短會讓「暴增」變成「昨天剛好低」"""
    assert MARGIN_WINDOW >= 20


# ══════════════════════════════════════════════════════════════
# 虛無抽樣的決定性
# ══════════════════════════════════════════════════════════════


def _draw(pool: list[str], as_of, count: int) -> list[str]:
    """複製腳本裡的抽樣邏輯,用來驗決定性"""
    rng = np.random.default_rng(int(as_of.strftime("%Y%m%d")))
    return [pool[i] for i in rng.choice(len(pool), size=count, replace=False)]


def test_null_draw_is_reproducible_for_the_same_date():
    """
    同一天重跑必須抽到同一組。

    `INSERT OR IGNORE` 讓第一次寫入的內容永久固定,若抽樣不決定性,
    第二次跑的名單會靜默消失,帳本就無法重現。
    """
    import datetime

    pool = [f"{1000 + i:04d}" for i in range(50)]
    day = datetime.date(2026, 9, 11)

    assert _draw(pool, day, 5) == _draw(pool, day, 5)


def test_null_draw_differs_across_dates():
    import datetime

    pool = [f"{1000 + i:04d}" for i in range(50)]

    first = _draw(pool, datetime.date(2026, 9, 11), 5)
    second = _draw(pool, datetime.date(2026, 9, 14), 5)

    assert first != second


def test_null_draw_returns_the_requested_count_without_repeats():
    import datetime

    pool = [f"{1000 + i:04d}" for i in range(50)]
    drawn = _draw(pool, datetime.date(2026, 9, 11), 7)

    assert len(drawn) == 7
    assert len(set(drawn)) == 7
