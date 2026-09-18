"""雙 sleeve 的資金權重與 Sharpe 計算。"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.diagnose_sleeve_blends import summarize_sleeve_pair


@pytest.mark.unit
def test_sleeve_pair_uses_equal_capital_weight_and_net_inputs() -> None:
    result = summarize_sleeve_pair([0.10, -0.02, 0.04], [-0.02, 0.06, 0.02])
    expected = np.asarray([0.04, 0.02, 0.03])

    assert result["blend_net_per_trip"] == pytest.approx(expected.mean())
    assert result["blend_net_sharpe"] is not None
    assert result["periods"] == 3
