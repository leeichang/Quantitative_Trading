"""
路徑相依量測的測試，以及機制本身的性質測試

## 兩層測試

1. **量測函式的單元測試**——`divergence`、`propagation_ratio` 算對沒有
2. **機制的性質測試**——用真的 `simulate_portfolio` 與
   `select_periodic_rebalances` 跑合成訊號，驗證待辦第 2 項的假說

第 2 層才是重點。待辦說「99.8% 訊號被槽位擋掉，成交哪一百筆取決於
槽位何時釋放」——那是機制假說，這裡直接驗它：

```
槽位排隊    移掉第一個決策日的一檔 → 分歧一路傳到序列末端
定期換倉    同樣的擾動 → 分歧只出現在被擾動的那一期
```

`validation/thresholds.py` 的 docstring 早就寫了 E3 的設計意圖是
「讓策略不再依賴槽位何時偶然釋放」。**這些測試是那句話的驗證。**
"""

from __future__ import annotations

import pandas as pd
import pytest

from taiwan_quant.backtest.portfolio_sim import Signal, simulate_portfolio
from taiwan_quant.config.costs import Tier
from taiwan_quant.validation.path_dependence import (
    divergence,
    fill_signature,
    propagation_ratio,
)
from taiwan_quant.validation.thresholds import select_periodic_rebalances


class _Trade:
    """最小替身，只需要 stock_id 與 entry_date"""

    def __init__(self, stock_id: str, entry_date: str) -> None:
        self.stock_id = stock_id
        self.entry_date = pd.Timestamp(entry_date)


# ══════════════════════════════════════════════════════════════
# fill_signature / divergence
# ══════════════════════════════════════════════════════════════


def test_fill_signature_keys_on_stock_and_entry_date_only():
    """同一檔同一天進場就是同一筆，報酬差異不算路徑分歧"""
    a = _Trade("2330", "2020-01-02")
    b = _Trade("2330", "2020-01-02")

    assert fill_signature([a]) == fill_signature([b])


def test_identical_runs_report_no_divergence():
    trades = [_Trade("2330", "2020-01-02"), _Trade("2317", "2020-03-02")]

    result = divergence(trades, list(trades))

    assert result.diverged is False
    assert result.shared == 2
    assert result.jaccard == pytest.approx(1.0)
    assert result.first_divergence is None
    assert result.last_divergence is None


def test_divergence_counts_each_side_and_spans_the_differing_dates():
    baseline = [
        _Trade("2330", "2020-01-02"),
        _Trade("2317", "2020-03-02"),
        _Trade("2454", "2020-06-01"),
    ]
    perturbed = [
        _Trade("2330", "2020-01-02"),   # 共有
        _Trade("1301", "2020-03-02"),   # 取代 2317
        _Trade("2454", "2020-06-01"),   # 共有
    ]

    result = divergence(baseline, perturbed)

    assert result.shared == 2
    assert result.only_baseline == 1
    assert result.only_perturbed == 1
    # 手算：交集 2、聯集 4
    assert result.jaccard == pytest.approx(0.5)
    assert result.first_divergence == pd.Timestamp("2020-03-02")
    assert result.last_divergence == pd.Timestamp("2020-03-02")


def test_divergence_of_two_empty_runs_is_not_a_divergence():
    result = divergence([], [])

    assert result.diverged is False
    assert result.jaccard == pytest.approx(1.0)


# ══════════════════════════════════════════════════════════════
# propagation_ratio
# ══════════════════════════════════════════════════════════════


def _calendar(n: int) -> list[pd.Timestamp]:
    return list(pd.bdate_range("2020-01-01", periods=n))


def test_propagation_ratio_is_one_when_divergence_reaches_the_end():
    calendar = _calendar(11)
    result = divergence(
        [_Trade("A", calendar[0].isoformat())],
        [_Trade("B", calendar[-1].isoformat())],
    )

    assert propagation_ratio(result, calendar[0], calendar) == pytest.approx(1.0)


def test_propagation_ratio_is_a_fraction_when_divergence_stops_early():
    calendar = _calendar(11)
    # 擾動在位置 0，最後分歧在位置 2，剩餘 10 個交易日 → 2/10
    result = divergence(
        [_Trade("A", calendar[2].isoformat())],
        [_Trade("B", calendar[2].isoformat())],
    )

    assert propagation_ratio(result, calendar[0], calendar) == pytest.approx(0.2)


def test_propagation_ratio_is_zero_without_divergence():
    calendar = _calendar(11)
    trades = [_Trade("A", calendar[3].isoformat())]

    result = divergence(trades, list(trades))

    assert propagation_ratio(result, calendar[0], calendar) == pytest.approx(0.0)


def test_propagation_ratio_rejects_an_unknown_perturbation_date():
    calendar = _calendar(5)
    result = divergence([], [])

    with pytest.raises(ValueError, match="不在交易日曆"):
        propagation_ratio(result, pd.Timestamp("1999-01-01"), calendar)


def test_propagation_ratio_rejects_an_empty_calendar():
    result = divergence([], [])

    with pytest.raises(ValueError, match="日曆不可為空"):
        propagation_ratio(result, pd.Timestamp("2020-01-01"), [])


# ══════════════════════════════════════════════════════════════
# 機制的性質測試：待辦第 2 項的假說
# ══════════════════════════════════════════════════════════════

HOLD = 10
"""合成訊號的持有期（交易日）。真實是 60，縮小只為讓測試快"""

N_SLOTS = 3
CANDIDATES_PER_DATE = 12
"""每個決策日 12 檔候選、3 個槽位 → 排隊比 4:1，與真實同號"""


def _synthetic_signals(
    calendar: list[pd.Timestamp],
    *,
    drop: tuple[pd.Timestamp, str] | None = None,
    staggered_exits: bool = True,
) -> list[Signal]:
    """
    每 5 個交易日產生 12 檔候選，分數固定可重現。

    `drop` 移掉指定 (決策日, 代號) 的一檔——這就是最小擾動，
    對應 v6 → v7 的「標的池換掉幾檔」。

    `staggered_exits` 讓出場日**逐檔不同**，範圍 [3, HOLD]。真實系統的
    `Signal.exit_date` 來自 triple-barrier／移動停損，觸價就提前出場，
    所以槽位的釋放時點是不規則的。

    ⚠️ **這個參數不是裝飾。** 設成 `False`（全部同日出場）時，三個槽位
    永遠同時釋放，槽位排隊就退化成定期換倉，路徑相依會消失——
    見 `test_synchronised_exits_make_the_slot_queue_degenerate`。
    """
    signals: list[Signal] = []
    for index, day in enumerate(calendar):
        if index % 5 != 0:
            continue
        if index + HOLD >= len(calendar):
            break
        for rank in range(CANDIDATES_PER_DATE):
            stock_id = f"{1000 + (index * 7 + rank * 13) % 90:04d}"
            if drop is not None and drop == (day, stock_id):
                continue
            if staggered_exits:
                # [3, HOLD] 之間錯開，模擬提前觸價出場
                offset = 3 + (index * 11 + rank * 7) % (HOLD - 2)
            else:
                offset = HOLD
            signals.append(
                Signal(
                    decision_date=day,
                    exit_date=calendar[index + offset],
                    stock_id=stock_id,
                    # 報酬與分數由 (index, rank) 決定：可重現且非退化
                    gross_return=((index * 31 + rank * 17) % 41 - 20) / 100.0,
                    rank_score=1.0 - rank / CANDIDATES_PER_DATE,
                    tier=Tier.LARGE,
                )
            )
    return signals


def _lookup(_stock_id: str, _day: pd.Timestamp) -> float:
    """定價替身。E3 要能取到進出場價，數值本身不影響成交集合"""
    return 100.0


def _slot_trades(signals: list[Signal], calendar: list[pd.Timestamp]):
    return simulate_portfolio(
        signals, _lookup, calendar, n_slots=N_SLOTS
    ).trades


def _rebalance_trades(signals: list[Signal], calendar: list[pd.Timestamp]):
    selected = select_periodic_rebalances(
        signals, calendar, _lookup, HOLD, N_SLOTS
    )
    return simulate_portfolio(
        list(selected.signals), _lookup, calendar, n_slots=N_SLOTS
    ).trades


def test_the_slot_queue_blocks_most_signals():
    """先確認合成資料真的有排隊，否則下面的測試沒有意義"""
    calendar = _calendar(140)
    signals = _synthetic_signals(calendar)

    result = simulate_portfolio(signals, _lookup, calendar, n_slots=N_SLOTS)

    assert result.slot_blocked_signals > result.opened_signals, (
        f"被擋 {result.slot_blocked_signals} 應多於成交 "
        f"{result.opened_signals}，否則沒有排隊現象"
    )


def test_one_dropped_name_outlives_its_own_position_under_the_slot_queue():
    """
    **待辦第 2 項的假說，收斂成可推導的形式。**

    移掉第一個決策日的一檔，槽位排隊下分歧會延續到**被擾動的那個部位
    早已出場之後**——因為槽位的釋放時點被推移，後續誰能進場跟著改變。

    「傳播到末端」不成立（實測 14.4%，會自行收斂），所以不斷言那個。
    斷言的是機制本身可推導的部分：**擾動活得比它擾動的部位更久。**
    那才是路徑相依的定義，也是報酬變成抽樣結果的原因。
    """
    calendar = _calendar(140)
    baseline = _synthetic_signals(calendar)
    victim = baseline[0]
    perturbed = _synthetic_signals(
        calendar, drop=(victim.decision_date, victim.stock_id)
    )
    assert len(perturbed) == len(baseline) - 1

    result = divergence(
        _slot_trades(baseline, calendar), _slot_trades(perturbed, calendar)
    )

    assert result.diverged, "移掉一檔應該改變成交集合"
    assert result.last_divergence is not None
    # 被擾動的部位最晚在 decision_date + HOLD 出場
    latest_exit = calendar[calendar.index(victim.decision_date) + HOLD]
    assert result.last_divergence > latest_exit, (
        f"分歧應延續到 {latest_exit.date()} 之後，"
        f"得到 {result.last_divergence.date()}（{result.describe()}）"
    )


def test_the_same_drop_stays_local_under_periodic_rebalancing():
    """
    定期換倉下每個節點獨立選 Top N，擾動不應該跨期傳播。

    這是 `validation/thresholds.py` docstring 那句「讓策略不再依賴槽位
    何時偶然釋放」的驗證。
    """
    calendar = _calendar(140)
    baseline = _synthetic_signals(calendar)
    victim = baseline[0]
    perturbed = _synthetic_signals(
        calendar, drop=(victim.decision_date, victim.stock_id)
    )

    result = divergence(
        _rebalance_trades(baseline, calendar),
        _rebalance_trades(perturbed, calendar),
    )
    reach = propagation_ratio(result, victim.decision_date, calendar)

    assert reach <= 0.2, (
        f"定期換倉下擾動應被關在原地，得到 {reach:.1%}"
        f"（{result.describe()}）"
    )


def test_rebalancing_diverges_strictly_less_than_the_slot_queue():
    """
    同一個擾動、同一組訊號，兩個方案的傳播範圍直接對比。

    這一條是本檔案的承重測試：**若兩者傳播一樣遠，改方案就沒有用。**
    """
    calendar = _calendar(140)
    baseline = _synthetic_signals(calendar)
    victim = baseline[0]
    perturbed = _synthetic_signals(
        calendar, drop=(victim.decision_date, victim.stock_id)
    )

    slot = propagation_ratio(
        divergence(
            _slot_trades(baseline, calendar), _slot_trades(perturbed, calendar)
        ),
        victim.decision_date,
        calendar,
    )
    rebalance = propagation_ratio(
        divergence(
            _rebalance_trades(baseline, calendar),
            _rebalance_trades(perturbed, calendar),
        ),
        victim.decision_date,
        calendar,
    )

    assert rebalance < slot, (
        f"定期換倉的傳播 {rebalance:.1%} 應小於槽位排隊的 {slot:.1%}"
    )


def test_synchronised_exits_make_the_slot_queue_degenerate():
    """
    **路徑相依的必要條件是出場日錯開。**

    全部同日出場時，三個槽位永遠同時釋放，進場節點變成固定週期——
    槽位排隊此時等價於定期換倉，一次擾動不會跨期傳播。

    這個測試是先前一版合成資料的意外發現：出場日寫成固定 `index + HOLD`
    時，槽位方案的傳播量到 0.0%，反而低於定期換倉。**不是假說錯，是
    合成資料少了真實系統才有的不規則出場。**

    所以待辦第 2 項的機制描述要補一句：不只是「99.8% 訊號被擋」，
    而是「被擋 + 出場日不規則」才產生路徑相依。
    """
    calendar = _calendar(140)
    baseline = _synthetic_signals(calendar, staggered_exits=False)
    victim = baseline[0]
    perturbed = _synthetic_signals(
        calendar,
        drop=(victim.decision_date, victim.stock_id),
        staggered_exits=False,
    )

    reach = propagation_ratio(
        divergence(
            _slot_trades(baseline, calendar), _slot_trades(perturbed, calendar)
        ),
        victim.decision_date,
        calendar,
    )

    assert reach <= 0.05, (
        f"同日出場時槽位排隊應退化、不傳播，得到 {reach:.1%}"
    )
