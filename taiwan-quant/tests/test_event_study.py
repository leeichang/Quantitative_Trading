"""
事件研究的測試

承重的三條：

1. **forward 報酬是 T+1 開盤進、T+H 收盤出**——手算驗證，不是拿另一條
   程式路徑比對。2026-09-18 之前主線有三個檔案多算一天，而測試也照同
   一個誤解寫，所以測試通過而行為是錯的。
2. **觀測單位是「日」不是「股票-事件」**——若把每筆事件當獨立樣本，
   n 會虛增、SE 虛減，什麼都會顯著。
3. **分位門檻只能用 ≤ t 的資料**（禁令 1）——用全期分布是 look-ahead。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from taiwan_quant.validation.event_study import (
    EventStudyError,
    capacity_economics,
    event_study,
    expanding_quantile_mask,
    forward_returns,
    per_date_mean,
)


def _frame(values: list[list[float]], n_cols: int = 2) -> pd.DataFrame:
    idx = pd.bdate_range("2020-01-01", periods=len(values))
    cols = [f"S{i}" for i in range(n_cols)]
    return pd.DataFrame(values, index=idx, columns=cols)


# ══════════════════════════════════════════════════════════════
# forward_returns
# ══════════════════════════════════════════════════════════════


def test_forward_return_enters_at_next_open_and_exits_at_t_plus_h_close():
    """
    手算：opens[1] = 11、closes[1] = 12，H=1 時 forward[0] = 12/11 − 1。

    多算一天會得到 closes[2]/opens[1] = 13/11 − 1 = +18.18%，差一倍。
    """
    opens = _frame([[10.0, 10.0], [11.0, 11.0], [12.0, 12.0], [13.0, 13.0]])
    closes = _frame([[10.5, 10.5], [12.0, 12.0], [13.0, 13.0], [14.0, 14.0]])

    forward = forward_returns(opens, closes, holding_days=1)

    assert forward.iloc[0, 0] == pytest.approx(12.0 / 11.0 - 1)
    assert forward.iloc[0, 0] == pytest.approx(0.090909, abs=1e-6)
    assert forward.iloc[0, 0] != pytest.approx(13.0 / 11.0 - 1)


def test_forward_return_tail_is_nan_when_the_exit_is_past_the_end():
    opens = _frame([[10.0, 10.0], [11.0, 11.0], [12.0, 12.0]])
    closes = _frame([[10.5, 10.5], [12.0, 12.0], [13.0, 13.0]])

    forward = forward_returns(opens, closes, holding_days=2)

    assert np.isnan(forward.iloc[-1, 0])
    assert np.isnan(forward.iloc[-2, 0])


def test_forward_return_rejects_a_zero_horizon():
    frame = _frame([[10.0, 10.0], [11.0, 11.0]])
    with pytest.raises(ValueError, match="holding_days"):
        forward_returns(frame, frame, holding_days=0)


def test_forward_return_rejects_mismatched_columns():
    """
    契約由 `data/integrity.forward_returns` 定義，拋的是 `ValueError`。

    本模組原本自己寫了一份（拋 `EventStudyError`、訊息寫「形狀」），
    與 Codex 在工作單 09 建的 canonical 版本重複。已刪除改成 re-export，
    所以這裡改斷言 canonical 的契約。
    """
    opens = _frame([[10.0, 10.0], [11.0, 11.0]])
    closes = _frame([[10.0], [11.0]], n_cols=1)
    with pytest.raises(ValueError, match="columns"):
        forward_returns(opens, closes, holding_days=1)


# ══════════════════════════════════════════════════════════════
# per_date_mean
# ══════════════════════════════════════════════════════════════


def test_per_date_mean_averages_only_the_masked_names():
    returns = _frame([[0.10, 0.20], [0.30, 0.40]])
    mask = _frame([[1.0, 0.0], [1.0, 1.0]])

    result = per_date_mean(returns, mask)

    assert result.iloc[0] == pytest.approx(0.10)
    assert result.iloc[1] == pytest.approx(0.35)


def test_per_date_mean_is_nan_when_nothing_is_masked_that_day():
    """當天沒有事件就回 NaN，由呼叫端丟掉——不要靜默當成 0"""
    returns = _frame([[0.10, 0.20], [0.30, 0.40]])
    mask = _frame([[0.0, 0.0], [1.0, 1.0]])

    result = per_date_mean(returns, mask)

    assert np.isnan(result.iloc[0])
    assert result.iloc[1] == pytest.approx(0.35)


def test_per_date_mean_skips_missing_returns():
    returns = _frame([[0.10, float("nan")], [0.30, 0.40]])
    mask = _frame([[1.0, 1.0], [1.0, 1.0]])

    result = per_date_mean(returns, mask)

    assert result.iloc[0] == pytest.approx(0.10)


# ══════════════════════════════════════════════════════════════
# event_study
# ══════════════════════════════════════════════════════════════


def test_event_study_pairs_against_the_same_day_universe_mean():
    """
    手算。第 0 天：事件組只有 S0（+10%），全池平均 (10+20)/2 = 15%，
    配對差異 = 10 − 15 = −5%。
    """
    returns = _frame([[0.10, 0.20], [0.30, 0.10]])
    events = _frame([[1.0, 0.0], [1.0, 0.0]])
    universe = _frame([[1.0, 1.0], [1.0, 1.0]])

    result = event_study(
        event="測試", horizon=1, returns=returns,
        event_mask=events, universe_mask=universe,
    )

    assert result.n_dates == 2
    assert result.paired[0] == pytest.approx(-0.05)
    # 第 1 天：30 − 20 = +10%
    assert result.paired[1] == pytest.approx(+0.10)
    assert result.excess == pytest.approx(0.025)


def test_event_study_counts_dates_not_stock_events():
    """
    **承重測試。** 一天 5 檔事件、共 2 天 → n_dates = 2，不是 10。

    若把每筆事件當樣本，SE 會虛減 √5 倍，什麼都會顯著。
    """
    returns = _frame([[0.1] * 5, [0.2] * 5], n_cols=5)
    events = _frame([[1.0] * 5, [1.0] * 5], n_cols=5)
    universe = _frame([[1.0] * 5, [1.0] * 5], n_cols=5)

    result = event_study(
        event="測試", horizon=1, returns=returns,
        event_mask=events, universe_mask=universe,
    )

    assert result.n_events == 10
    assert result.n_dates == 2
    assert len(result.paired) == 2
    assert result.events_per_date == pytest.approx(5.0)


def test_event_study_ignores_events_outside_the_universe():
    """禁令 2：事件發生在當時不在標的池的股票上，不可採計"""
    returns = _frame([[0.10, 0.90], [0.30, 0.90]])
    events = _frame([[1.0, 1.0], [1.0, 1.0]])
    universe = _frame([[1.0, 0.0], [1.0, 0.0]])

    result = event_study(
        event="測試", horizon=1, returns=returns,
        event_mask=events, universe_mask=universe,
    )

    assert result.n_events == 2
    # S1 完全被排除，所以事件組與全池都只有 S0 → 配對差異為零
    assert result.paired == pytest.approx((0.0, 0.0))


def test_event_study_drops_days_where_either_side_is_missing():
    """配對必須同日。少了任一邊就丟掉那天，不可跨日相減"""
    returns = _frame([[0.10, 0.20], [float("nan"), float("nan")], [0.30, 0.10]])
    events = _frame([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    universe = _frame([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])

    result = event_study(
        event="測試", horizon=1, returns=returns,
        event_mask=events, universe_mask=universe,
    )

    assert result.n_dates == 2


def test_event_study_standard_error_shrinks_with_more_dates():
    """SE 應隨日數以 1/√n 收斂——這是事件法換來檢定力的來源"""
    rng = np.random.default_rng(0)

    def build(n_days):
        idx = pd.bdate_range("2020-01-01", periods=n_days)
        cols = ["S0", "S1"]
        r = pd.DataFrame(rng.normal(0.02, 0.10, (n_days, 2)), idx, cols)
        e = pd.DataFrame([[1.0, 0.0]] * n_days, idx, cols)
        u = pd.DataFrame(1.0, idx, cols)
        return event_study(event="t", horizon=1, returns=r,
                           event_mask=e, universe_mask=u)

    small = build(40)
    large = build(400)

    ratio = small.standard_error / large.standard_error
    assert ratio == pytest.approx(np.sqrt(10), rel=0.4), (
        f"SE 比應接近 √10 = 3.16，得到 {ratio:.2f}"
    )


def test_capacity_economics_subtracts_same_day_fill_costs() -> None:
    """手算兩日：毛配對為 -5%、+5%，成本 1%、2%，淨為 -6%、+3%。"""
    returns = _frame([[0.10, 0.20], [0.30, 0.40]])
    fills = _frame([[1.0, 0.0], [0.0, 1.0]])
    universe = _frame([[1.0, 1.0], [1.0, 1.0]])
    costs = _frame([[0.01, 0.99], [0.99, 0.02]])

    result = capacity_economics(
        returns=returns,
        fills=fills,
        universe_mask=universe,
        cost_rates=costs,
    )

    assert result.gross_paired == pytest.approx((-0.05, 0.05))
    assert result.net_paired == pytest.approx((-0.06, 0.03))
    assert result.gross_excess == pytest.approx(0.0)
    assert result.cost_per_trip == pytest.approx(0.015)
    assert result.net_per_trip == pytest.approx(-0.015)
    assert result.net_standard_deviation == pytest.approx(np.std([-0.06, 0.03], ddof=1))


# ══════════════════════════════════════════════════════════════
# expanding_quantile_mask（禁令 1）
# ══════════════════════════════════════════════════════════════


def test_expanding_quantile_uses_only_past_data():
    """
    **禁令 1。** 前面幾天還沒有足夠歷史，門檻必須是 NaN → 不標事件。

    用全期分布會讓第 0 天就知道整段歷史的值域。
    """
    values = _frame([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [100.0, 200.0]])

    mask = expanding_quantile_mask(
        values, quantile=0.5, refresh_every=1, min_observations=4
    )

    # 第 0 天沒有任何歷史 → 不可能標出事件
    assert not mask.iloc[0].any()


def test_expanding_quantile_flags_values_above_the_trailing_threshold():
    values = _frame([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [100.0, 200.0]])

    mask = expanding_quantile_mask(
        values, quantile=0.5, refresh_every=1, min_observations=2
    )

    # 最後一天的值遠高於之前的中位數，必須被標出
    assert mask.iloc[-1].all()


def test_expanding_quantile_rejects_an_out_of_range_quantile():
    values = _frame([[1.0, 2.0], [3.0, 4.0]])
    for bad in (0.0, 1.0, -0.5, 1.5):
        with pytest.raises(EventStudyError, match="分位"):
            expanding_quantile_mask(
                values, quantile=bad, refresh_every=1, min_observations=1
            )


def test_expanding_quantile_rejects_a_zero_refresh_interval():
    values = _frame([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(EventStudyError, match="refresh_every"):
        expanding_quantile_mask(
            values, quantile=0.5, refresh_every=0, min_observations=1
        )


# ══════════════════════════════════════════════════════════════
# capacity_constrained_fills
# ══════════════════════════════════════════════════════════════


def _grid(rows: list[list[float]], cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        rows, index=pd.bdate_range("2020-01-01", periods=len(rows)), columns=cols
    )


def test_capacity_takes_everything_when_slots_are_ample():
    """位子夠多時不該有任何損失——否則模擬本身有 bug"""
    from taiwan_quant.validation.event_study import capacity_constrained_fills

    events = _grid([[1.0, 1.0], [1.0, 1.0]], ["A", "B"])
    scores = _grid([[0.9, 0.5], [0.9, 0.5]], ["A", "B"])

    result = capacity_constrained_fills(
        events, scores, n_slots=10, holding_days=1
    )

    assert result.n_offered == 4
    assert result.n_filled == 4
    assert result.capture_rate == pytest.approx(1.0)


def test_capacity_blocks_when_slots_are_full():
    """
    **承重測試。** 1 個位子、持有 3 日：第 0 天吃 1 檔，
    第 1、2 天全部擋掉，第 3 天位子釋放才能再吃。
    """
    from taiwan_quant.validation.event_study import capacity_constrained_fills

    events = _grid([[1.0, 1.0]] * 4, ["A", "B"])
    scores = _grid([[0.9, 0.5]] * 4, ["A", "B"])

    result = capacity_constrained_fills(
        events, scores, n_slots=1, holding_days=3
    )

    assert result.n_offered == 8
    assert result.n_filled == 2, "只有第 0 天與第 3 天能開倉"
    assert result.n_blocked_no_slot > 0
    assert bool(result.fills.iloc[0]["A"])
    assert not result.fills.iloc[1].any()
    assert not result.fills.iloc[2].any()
    assert result.fills.iloc[3].any()


def test_capacity_prefers_the_higher_score_when_slots_are_scarce():
    """位子不足時取分數高的。排序錯了會系統性吃到較差的事件"""
    from taiwan_quant.validation.event_study import capacity_constrained_fills

    events = _grid([[1.0, 1.0]], ["A", "B"])
    scores = _grid([[0.1, 0.9]], ["A", "B"])

    result = capacity_constrained_fills(
        events, scores, n_slots=1, holding_days=1
    )

    assert bool(result.fills.iloc[0]["B"])
    assert not bool(result.fills.iloc[0]["A"])


def test_capacity_does_not_double_buy_a_held_name():
    """已在持倉的再觸發是加碼不是分散，要跳過並記錄"""
    from taiwan_quant.validation.event_study import capacity_constrained_fills

    events = _grid([[1.0], [1.0], [1.0]], ["A"])
    scores = _grid([[0.9], [0.9], [0.9]], ["A"])

    result = capacity_constrained_fills(
        events, scores, n_slots=5, holding_days=2
    )

    assert result.n_filled == 2, "第 0 天買，第 1 天跳過，第 2 天釋放後再買"
    assert result.n_blocked_held == 1


def test_capacity_never_exceeds_the_slot_count():
    """任何一天的同時持倉都不可超過位子數"""
    from taiwan_quant.validation.event_study import capacity_constrained_fills

    rng = np.random.default_rng(3)
    cols = [f"S{i}" for i in range(12)]
    events = _grid(rng.integers(0, 2, (60, 12)).astype(float).tolist(), cols)
    scores = _grid(rng.random((60, 12)).tolist(), cols)
    slots, hold = 4, 5

    result = capacity_constrained_fills(
        events, scores, n_slots=slots, holding_days=hold
    )

    concurrent = result.fills.astype(int).rolling(hold, min_periods=1).sum().sum(axis=1)
    assert concurrent.max() <= slots, f"同時持倉 {concurrent.max()} 超過 {slots}"


@pytest.mark.parametrize("slots, hold", [(0, 1), (-1, 1), (1, 0), (1, -1)])
def test_capacity_rejects_invalid_parameters(slots, hold):
    from taiwan_quant.validation.event_study import capacity_constrained_fills

    events = _grid([[1.0]], ["A"])
    with pytest.raises(EventStudyError):
        capacity_constrained_fills(
            events, events, n_slots=slots, holding_days=hold
        )


def test_capacity_allow_repeat_opens_a_second_lot_in_the_same_name():
    """
    `allow_repeat=True` 時同一檔重複觸發會另開位子（加碼），
    而不是跳過。實測 H=40 時 62.6% 的事件是這種重複。

    ⚠️ 代價是集中度：同一檔可能佔多個位子。
    """
    from taiwan_quant.validation.event_study import capacity_constrained_fills

    events = _grid([[1.0], [1.0], [1.0]], ["A"])
    scores = _grid([[0.9], [0.9], [0.9]], ["A"])

    skip = capacity_constrained_fills(
        events, scores, n_slots=5, holding_days=3, allow_repeat=False
    )
    repeat = capacity_constrained_fills(
        events, scores, n_slots=5, holding_days=3, allow_repeat=True
    )

    assert skip.n_filled == 1, "跳過模式：買一次後被自己擋住"
    assert skip.n_blocked_held == 2
    assert repeat.n_filled == 3, "加碼模式：三次都開倉"
    assert repeat.n_blocked_held == 0


def test_capacity_allow_repeat_still_respects_the_slot_cap():
    """加碼不可以突破位子上限——否則就不是容量受限的模擬了"""
    from taiwan_quant.validation.event_study import capacity_constrained_fills

    events = _grid([[1.0], [1.0], [1.0], [1.0]], ["A"])
    scores = _grid([[0.9], [0.9], [0.9], [0.9]], ["A"])

    result = capacity_constrained_fills(
        events, scores, n_slots=2, holding_days=10, allow_repeat=True
    )

    assert result.n_filled == 2
    assert result.n_blocked_no_slot == 2
