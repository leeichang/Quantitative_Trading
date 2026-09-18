"""OOS 守門與對照組的可重現性測試。"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.validate_oos_momentum import randomized_candidates


@pytest.mark.unit
def test_randomized_candidates_ignore_set_construction_order() -> None:
    left = set(["2330", "2317", "2454", "2303", "2881"])
    right = set(reversed(["2330", "2317", "2454", "2303", "2881"]))

    first = randomized_candidates(left, np.random.default_rng(20260916))
    second = randomized_candidates(right, np.random.default_rng(20260916))

    assert first == second
