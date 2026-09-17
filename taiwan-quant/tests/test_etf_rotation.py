"""
ETF 輪動的測試

## 為什麼這個模組一定要有測試

第一版的輪動邏輯寫在臨時腳本裡，跑出「年化 81.8%」。那不是發現，
是索引錯位：

```python
cl = cl.dropna(how="any")     # cl 被過濾了
cl.iloc[i] / op.iloc[i + 1]   # 但 op 沒有 → iloc 位置對應到不同日期
```

修正後是 12.5%。**`iloc` 錯位不會拋錯，只會安靜地算錯**，而數字看起來
太好時人會傾向相信它。這裡的第一個測試就是釘住那件事。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from taiwan_quant.ranking.etf_rotation import (
    RotationError,
    aligned_views,
    holding_return,
    top_k,
    trailing_momentum,
)


def frame(data: dict[str, list[float]], days: int | None = None) -> pd.DataFrame:
    n = days or len(next(iter(data.values())))
    return pd.DataFrame(data, index=pd.date_range("2020-01-01", periods=n, freq="D"))


# ══════════════════════════════════════════════════════════════
# aligned_views：那個 bug 的防線
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_aligned_views_drops_dates_missing_in_any_frame() -> None:
    """
    任一矩陣缺值的日期，全部矩陣都要拿掉——否則 `iloc[i]` 在兩者
    指向不同日期。

    手算：收盤在 index 1 缺值，開盤完整
        → 共同索引是 {0, 2}，兩個矩陣都只剩 2 列
        → aligned_closes.iloc[1] 與 aligned_opens.iloc[1] 都是原本的 index 2
    """
    closes = frame({"A": [10.0, np.nan, 12.0], "B": [20.0, 21.0, 22.0]})
    opens = frame({"A": [10.0, 11.0, 11.5], "B": [20.0, 20.5, 21.5]})

    aligned_closes, aligned_opens = aligned_views(closes, opens)

    assert len(aligned_closes) == 2
    assert aligned_closes.index.equals(aligned_opens.index)
    assert aligned_closes.index[1] == closes.index[2]
    assert aligned_opens.iloc[1]["A"] == pytest.approx(11.5)


@pytest.mark.unit
def test_aligned_views_keeps_only_common_columns_sorted() -> None:
    """欄位也要對齊，而且順序固定——不固定就無法重現"""
    closes = frame({"B": [1.0, 2.0], "A": [3.0, 4.0], "C": [5.0, 6.0]})
    opens = frame({"A": [3.0, 3.5], "B": [1.0, 1.5]})

    aligned_closes, aligned_opens = aligned_views(closes, opens)

    assert list(aligned_closes.columns) == ["A", "B"]
    assert list(aligned_opens.columns) == ["A", "B"]


@pytest.mark.unit
def test_aligned_views_rejects_empty_overlap() -> None:
    closes = frame({"A": [1.0, np.nan], "B": [1.0, 1.0]}).iloc[1:]
    opens = frame({"A": [1.0, 1.0], "B": [1.0, 1.0]}).iloc[:1]
    with pytest.raises(RotationError, match="共同"):
        aligned_views(closes, opens)


@pytest.mark.unit
def test_aligned_views_rejects_single_asset() -> None:
    """只剩一檔就沒有排序可言，寧可拋錯"""
    closes = frame({"A": [1.0, 2.0]})
    with pytest.raises(RotationError, match="無法排序"):
        aligned_views(closes)


# ══════════════════════════════════════════════════════════════
# trailing_momentum：時點紀律
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_trailing_momentum_hand_computed() -> None:
    """
    手算，lookback=2，position=2：

        A  10 → 12   12/10 − 1 = +0.20
        B  20 → 19   19/20 − 1 = −0.05
    """
    closes = frame({"A": [10.0, 11.0, 12.0], "B": [20.0, 21.0, 19.0]})

    momentum = trailing_momentum(closes, position=2, lookback=2)

    assert momentum["A"] == pytest.approx(0.20)
    assert momentum["B"] == pytest.approx(-0.05)


@pytest.mark.unit
def test_trailing_momentum_ignores_future_prices() -> None:
    """禁令 1：決策日之後的價格不可影響分數"""
    closes = frame({"A": [10.0, 11.0, 12.0, 99.0], "B": [20.0, 21.0, 19.0, 1.0]})

    truncated = trailing_momentum(closes.iloc[:3], position=2, lookback=2)
    full = trailing_momentum(closes, position=2, lookback=2)

    pd.testing.assert_series_equal(truncated, full)


@pytest.mark.unit
def test_trailing_momentum_requires_full_warmup() -> None:
    """
    暖機不足要拋錯，不可用不完整的視窗硬算。

    `position=1, lookback=2` 會退到 `iloc[-1]`——**Python 的負索引會
    silently 取到序列最後一天**，也就是未來。那是最惡劣的一種 look-ahead，
    所以必須擋在門口。
    """
    closes = frame({"A": [10.0, 11.0, 12.0], "B": [20.0, 21.0, 19.0]})
    with pytest.raises(RotationError, match="不足"):
        trailing_momentum(closes, position=1, lookback=2)


# ══════════════════════════════════════════════════════════════
# top_k
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_top_k_orders_by_score_descending() -> None:
    scores = pd.Series({"A": 0.1, "B": 0.3, "C": 0.2})
    assert top_k(scores, 2) == ["B", "C"]


@pytest.mark.unit
def test_top_k_breaks_ties_by_stock_id() -> None:
    """平手用代號，結果必須可重現"""
    scores = pd.Series({"0056": 0.2, "0050": 0.2, "0051": 0.1})
    assert top_k(scores, 2) == ["0050", "0056"]


@pytest.mark.unit
def test_top_k_skips_missing_scores() -> None:
    """缺分數的不可入選——不是當成 0，是根本不參加"""
    scores = pd.Series({"A": np.nan, "B": -0.5, "C": 0.1})
    assert top_k(scores, 3) == ["C", "B"]


@pytest.mark.unit
def test_top_k_rejects_nonpositive() -> None:
    with pytest.raises(RotationError, match="k"):
        top_k(pd.Series({"A": 1.0}), 0)


# ══════════════════════════════════════════════════════════════
# holding_return
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_holding_return_enters_next_open_exits_after_horizon() -> None:
    """
    手算，position=0、horizon=1：

        進場 = opens.iloc[1]     出場 = closes.iloc[2]
        A  開 11 → 收 13   13/11 − 1 = +0.181818
        B  開 21 → 收 19   19/21 − 1 = −0.095238
        等權 = (+0.181818 − 0.095238) / 2 = +0.043290
    """
    closes = frame({"A": [10.0, 12.0, 13.0], "B": [20.0, 22.0, 19.0]})
    opens = frame({"A": [10.0, 11.0, 12.5], "B": [20.0, 21.0, 20.0]})

    result = holding_return(closes, opens, position=0, horizon=1, picks=["A", "B"])

    assert result == pytest.approx((13 / 11 - 1 + 19 / 21 - 1) / 2)
    assert result == pytest.approx(0.043290, abs=1e-6)


@pytest.mark.unit
def test_holding_return_uses_close_not_open_for_exit() -> None:
    """
    出場是收盤，不是開盤。兩者差異在這組資料上是 13 vs 12.5，
    若寫錯會系統性偏低或偏高。
    """
    closes = frame({"A": [10.0, 12.0, 13.0], "B": [20.0, 22.0, 19.0]})
    opens = frame({"A": [10.0, 11.0, 12.5], "B": [20.0, 21.0, 20.0]})

    result = holding_return(closes, opens, position=0, horizon=1, picks=["A"])

    assert result == pytest.approx(13 / 11 - 1)
    assert result != pytest.approx(12.5 / 11 - 1)


@pytest.mark.unit
def test_holding_return_rejects_insufficient_horizon() -> None:
    """視野不足要拋錯，不可讓負索引繞回序列尾端"""
    closes = frame({"A": [10.0, 12.0], "B": [20.0, 22.0]})
    opens = frame({"A": [10.0, 11.0], "B": [20.0, 21.0]})
    with pytest.raises(RotationError, match="視野不足"):
        holding_return(closes, opens, position=0, horizon=5, picks=["A"])


@pytest.mark.unit
def test_holding_return_rejects_empty_picks() -> None:
    closes = frame({"A": [10.0, 12.0, 13.0], "B": [20.0, 22.0, 19.0]})
    opens = frame({"A": [10.0, 11.0, 12.5], "B": [20.0, 21.0, 20.0]})
    with pytest.raises(RotationError, match="picks"):
        holding_return(closes, opens, position=0, horizon=1, picks=[])


# ══════════════════════════════════════════════════════════════
# 候選必須跑在同一個決策網格上
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_candidates_with_different_lookbacks_share_one_grid() -> None:
    """
    不同回看期的候選必須有共同的決策日，否則無法比較。

    第一版用 `start = max(lookback, 1)`，於是回看 20 的決策日是
    20, 20+H, ... 而回看 250 的是 250, 250+H, ...——**只在 (250−20)
    是 H 的倍數時才重合**。H=20 時 230/20 = 11.5，交集為空，
    walk-forward 一律回報「測試期不足」。

    修法是所有候選共用 `max(LOOKBACK_GRID)` 當起點。
    """
    from scripts.validate_etf_rotation import LOOKBACK_GRID, candidate_returns

    rng = np.random.default_rng(20260917)
    n = 900
    data = {
        sid: list(100.0 * np.cumprod(1 + rng.normal(0.0002, 0.01, n)))
        for sid in ("0050", "0056")
    }
    closes = frame(data, days=n)
    opens = closes.shift(1).bfill()

    grids = {
        lb: set(candidate_returns(closes, opens, lb, horizon=20)[0])
        for lb in LOOKBACK_GRID
    }
    common = set.intersection(*grids.values())

    assert len(common) > 20, (
        f"不同回看期的共同決策日只有 {len(common)} 個——網格沒有對齊"
    )
    # 而且每個回看期的決策日集合應該完全相同（共用起點與步長）
    assert all(g == grids[LOOKBACK_GRID[0]] for g in grids.values())


@pytest.mark.unit
def test_candidate_returns_rejects_start_shorter_than_lookback() -> None:
    """起點不足回看期要拋錯，不可讓 iloc 負索引繞到序列尾端"""
    from scripts.validate_etf_rotation import candidate_returns

    closes = frame({"0050": [100.0] * 60, "0056": [50.0] * 60}, days=60)
    opens = closes.copy()
    with pytest.raises(ValueError, match="不足回看"):
        candidate_returns(closes, opens, lookback=50, horizon=5, start=10)
