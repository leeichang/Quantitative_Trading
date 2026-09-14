"""
多槽位組合模擬測試

`backtest/engine.py` 把各期報酬依序複合，結構上只容得下**一個**持倉。
但 CLAUDE.md 的設計是 Top 3 同時持有——用依序複合去評價它，會把
「同時持有 3 檔、幾乎全時間在市場」誤算成「一次一檔、七成時間空手」。

實測：移動停損版 OOS 曝險只有 23~39%，而對照組買進持有是 100%。
在多頭裡這個差距本身就足以輸掉，與選股能力無關。

比對手算值，不拿程式輸出反填預期值。
"""

from __future__ import annotations

import pandas as pd
import pytest

from taiwan_quant.backtest.portfolio_sim import (
    Signal,
    simulate_portfolio,
)
from taiwan_quant.config.costs import GROSS, Tier

ZERO_COST = GROSS
"""無成本情境（config/costs.py 的既有實例），方便手算"""


def calendar(n: int, start: str = "2024-01-01") -> list[pd.Timestamp]:
    return list(pd.date_range(start, periods=n, freq="B"))


def flat_prices(stock_ids: list[str], cal: list[pd.Timestamp], value: float = 100.0):
    """所有標的每天都是同一個價格——讓逐日市值標記可預測"""
    table = {
        sid: pd.Series([value] * len(cal), index=pd.DatetimeIndex(cal))
        for sid in stock_ids
    }

    def lookup(stock_id: str, day: pd.Timestamp) -> float | None:
        series = table.get(stock_id)
        if series is None or day not in series.index:
            return None
        return float(series.loc[day])

    return lookup


def linear_prices(stock_ids: list[str], cal: list[pd.Timestamp], daily: float = 0.01):
    """每日固定漲幅，讓權益曲線可手算"""
    table = {
        sid: pd.Series(
            [100.0 * (1 + daily) ** i for i in range(len(cal))],
            index=pd.DatetimeIndex(cal),
        )
        for sid in stock_ids
    }

    def lookup(stock_id: str, day: pd.Timestamp) -> float | None:
        series = table.get(stock_id)
        if series is None or day not in series.index:
            return None
        return float(series.loc[day])

    return lookup


def signal(
    decision_date: pd.Timestamp,
    exit_date: pd.Timestamp,
    stock_id: str = "A",
    gross_return: float = 0.10,
    rank_score: float = 1.0,
    tier: Tier = Tier.LARGE,
) -> Signal:
    return Signal(
        decision_date=decision_date,
        exit_date=exit_date,
        stock_id=stock_id,
        gross_return=gross_return,
        rank_score=rank_score,
        tier=tier,
    )


# ══════════════════════════════════════════════════════════════
# 基本權益演進
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_equity_starts_at_one() -> None:
    cal = calendar(10)
    result = simulate_portfolio([], flat_prices(["A"], cal), cal, n_slots=3, cost=ZERO_COST)

    assert result.equity.iloc[0] == pytest.approx(1.0)
    assert result.total_return == pytest.approx(0.0)


@pytest.mark.unit
def test_single_slot_single_trade_matches_gross_return() -> None:
    """
    1 個槽位、1 筆 +10% 的交易、無成本 → 期末權益 1.10

    全部資金押在唯一的槽位上，所以毛報酬就是總報酬。
    """
    cal = calendar(10)
    sig = signal(cal[1], cal[5], gross_return=0.10)
    result = simulate_portfolio(
        [sig], flat_prices(["A"], cal), cal, n_slots=1, cost=ZERO_COST
    )

    assert result.total_return == pytest.approx(0.10)
    assert result.n_trades == 1


@pytest.mark.unit
def test_three_slots_one_trade_dilutes_return() -> None:
    """
    3 個槽位但只有 1 筆交易 → 只有 1/3 的資金在市場。

    手算：1 + 0.10 / 3 = 1.0333
    """
    cal = calendar(10)
    sig = signal(cal[1], cal[5], gross_return=0.10)
    result = simulate_portfolio(
        [sig], flat_prices(["A"], cal), cal, n_slots=3, cost=ZERO_COST
    )

    assert result.total_return == pytest.approx(0.10 / 3, abs=1e-9)


@pytest.mark.unit
def test_three_concurrent_trades_use_full_capital() -> None:
    """
    3 個槽位全滿、每筆 +10% → 總報酬 +10%（不是 +30%）。

    這是多槽位的重點：同時持有不會放大報酬，但會**提高曝險**。
    """
    cal = calendar(10)
    sigs = [
        signal(cal[1], cal[5], stock_id=s, gross_return=0.10)
        for s in ("A", "B", "C")
    ]
    result = simulate_portfolio(
        sigs, flat_prices(["A", "B", "C"], cal), cal, n_slots=3, cost=ZERO_COST
    )

    assert result.total_return == pytest.approx(0.10, abs=1e-9)
    assert result.n_trades == 3


@pytest.mark.unit
def test_sequential_trades_compound() -> None:
    """
    同一槽位連做兩筆 +10% → 1.10 × 1.10 = 1.21
    """
    cal = calendar(20)
    sigs = [
        signal(cal[1], cal[5], stock_id="A", gross_return=0.10),
        signal(cal[6], cal[10], stock_id="A", gross_return=0.10),
    ]
    result = simulate_portfolio(
        sigs, flat_prices(["A"], cal), cal, n_slots=1, cost=ZERO_COST
    )

    assert result.total_return == pytest.approx(0.21, abs=1e-9)


# ══════════════════════════════════════════════════════════════
# 槽位限制
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_never_exceeds_slot_count() -> None:
    cal = calendar(20)
    sigs = [
        signal(cal[1], cal[10], stock_id=s, gross_return=0.05)
        for s in ("A", "B", "C", "D", "E")
    ]
    result = simulate_portfolio(
        sigs, flat_prices(list("ABCDE"), cal), cal, n_slots=3, cost=ZERO_COST
    )

    assert result.n_trades == 3
    assert result.max_concurrent == 3


@pytest.mark.unit
def test_picks_highest_rank_score_first() -> None:
    """槽位不足時取排名最高的"""
    cal = calendar(20)
    sigs = [
        signal(cal[1], cal[10], stock_id="低", gross_return=0.05, rank_score=0.1),
        signal(cal[1], cal[10], stock_id="高", gross_return=0.05, rank_score=0.9),
    ]
    result = simulate_portfolio(
        sigs, flat_prices(["低", "高"], cal), cal, n_slots=1, cost=ZERO_COST
    )

    assert [t.stock_id for t in result.trades] == ["高"]


@pytest.mark.unit
def test_does_not_hold_same_stock_twice() -> None:
    """同一檔已在持倉時不再開新倉——那會變成加碼，不是分散"""
    cal = calendar(20)
    sigs = [
        signal(cal[1], cal[10], stock_id="A", gross_return=0.05),
        signal(cal[2], cal[11], stock_id="A", gross_return=0.05),
    ]
    result = simulate_portfolio(
        sigs, flat_prices(["A"], cal), cal, n_slots=3, cost=ZERO_COST
    )

    assert result.n_trades == 1


@pytest.mark.unit
def test_slot_reopens_after_exit() -> None:
    cal = calendar(20)
    sigs = [
        signal(cal[1], cal[5], stock_id="A", gross_return=0.05),
        signal(cal[6], cal[10], stock_id="B", gross_return=0.05),
    ]
    result = simulate_portfolio(
        sigs, flat_prices(["A", "B"], cal), cal, n_slots=1, cost=ZERO_COST
    )

    assert result.n_trades == 2
    assert result.max_concurrent == 1


# ══════════════════════════════════════════════════════════════
# 成本
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_round_trip_cost_charged_once() -> None:
    """
    成本一趟來回只扣一次。

    手算（1 槽、預設成本 1.071%）：1 + 0.10 − 0.01071 = 1.08929
    """
    from taiwan_quant.config.costs import DEFAULT

    cal = calendar(10)
    sig = signal(cal[1], cal[5], gross_return=0.10)
    result = simulate_portfolio(
        [sig], flat_prices(["A"], cal), cal, n_slots=1, cost=DEFAULT
    )

    assert result.total_return == pytest.approx(
        0.10 - DEFAULT.round_trip_rate(Tier.LARGE), abs=1e-9
    )


@pytest.mark.unit
def test_turnover_and_annual_cost_drag_are_recomputed_from_trades() -> None:
    """
    10 個交易日內完成兩次全額換倉：
      第二筆可配置資金是扣除第一筆成本後的 1-cost。
      單邊總換手 = 1 + (1-cost)，再除以兩週。
    """
    from taiwan_quant.config.costs import DEFAULT

    cal = calendar(10)
    sigs = [
        signal(cal[0], cal[4], stock_id="A", gross_return=0.0),
        signal(cal[4], cal[9], stock_id="B", gross_return=0.0),
    ]
    result = simulate_portfolio(
        sigs, flat_prices(["A", "B"], cal), cal, n_slots=1, cost=DEFAULT
    )

    expected_cost = DEFAULT.round_trip_rate(Tier.LARGE)
    traded = 2.0 - expected_cost
    assert result.weekly_turnover == pytest.approx(traded / 2.0)
    assert result.annualized_cost_drag == pytest.approx(
        traded * expected_cost * 252 / 10
    )


# ══════════════════════════════════════════════════════════════
# 曝險與逐日市值
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_exposure_matches_hand_calculation() -> None:
    """
    3 個槽位、只有 1 個槽位持有 10 天、日曆 20 天。

    曝險 = 持倉槽位日 / (槽位數 × 日曆日)
         = 10 / (3 × 20) = 16.67%
    """
    cal = calendar(20)
    sig = signal(cal[0], cal[10], gross_return=0.05)
    result = simulate_portfolio(
        [sig], flat_prices(["A"], cal), cal, n_slots=3, cost=ZERO_COST
    )

    assert result.exposure == pytest.approx(10 / 60, abs=1e-9)


@pytest.mark.unit
def test_full_exposure_when_all_slots_always_filled() -> None:
    """
    出場日**當天**就不算持倉了（當天平倉、槽位釋放），所以要拿到
    100% 曝險，出場日必須落在日曆之後。
    """
    cal = calendar(11)
    after_end = cal[-1] + pd.Timedelta(days=30)
    sigs = [
        signal(cal[0], after_end, stock_id=s, gross_return=0.05)
        for s in ("A", "B", "C")
    ]
    result = simulate_portfolio(
        sigs, flat_prices(["A", "B", "C"], cal), cal, n_slots=3, cost=ZERO_COST
    )

    assert result.exposure == pytest.approx(1.0, abs=1e-9)


@pytest.mark.unit
def test_exit_day_releases_slot_same_day() -> None:
    """
    出場日當天槽位就能被下一筆接上——不必空等一天。

    11 天日曆、持有日曆第 0~9 天（第 10 天平倉）→ 曝險 10/11。
    """
    cal = calendar(11)
    sigs = [
        signal(cal[0], cal[10], stock_id=s, gross_return=0.05)
        for s in ("A", "B", "C")
    ]
    result = simulate_portfolio(
        sigs, flat_prices(["A", "B", "C"], cal), cal, n_slots=3, cost=ZERO_COST
    )

    assert result.exposure == pytest.approx(10 / 11, abs=1e-9)
    assert result.n_trades == 3


@pytest.mark.unit
def test_equity_marked_to_market_daily() -> None:
    """
    逐日標記市值，不是只在出場日跳一次。

    每日 +1%、1 槽全押、持有第 0~5 天：
        第 3 天的權益 = 1.01³ = 1.030301
    """
    cal = calendar(10)
    sig = signal(cal[0], cal[5], gross_return=0.05)
    result = simulate_portfolio(
        [sig], linear_prices(["A"], cal, daily=0.01), cal, n_slots=1, cost=ZERO_COST
    )

    assert result.equity.iloc[3] == pytest.approx(1.01**3, abs=1e-6)


@pytest.mark.unit
def test_max_drawdown_from_daily_equity() -> None:
    """
    逐日回撤才抓得到期間內的低點。

    價格 100 → 120 → 90（持有全程），回撤 = 1 − 90/120 = 25%
    """
    cal = calendar(4)
    prices = pd.Series([100.0, 120.0, 90.0, 95.0], index=pd.DatetimeIndex(cal))

    def lookup(stock_id: str, day: pd.Timestamp) -> float | None:
        return float(prices.loc[day]) if day in prices.index else None

    sig = signal(cal[0], cal[3], gross_return=-0.05)
    result = simulate_portfolio([sig], lookup, cal, n_slots=1, cost=ZERO_COST)

    assert result.max_drawdown == pytest.approx(0.25, abs=1e-9)


# ══════════════════════════════════════════════════════════════
# 反 look-ahead 與輸入驗證
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_signal_not_acted_before_its_decision_date() -> None:
    """
    決策日之前不得進場。

    訊號在第 5 天，第 0~4 天權益必須紋風不動。
    """
    cal = calendar(10)
    sig = signal(cal[5], cal[8], gross_return=0.50)
    result = simulate_portfolio(
        [sig], linear_prices(["A"], cal, daily=0.05), cal, n_slots=1, cost=ZERO_COST
    )

    for i in range(5):
        assert result.equity.iloc[i] == pytest.approx(1.0)


@pytest.mark.unit
def test_signal_outside_calendar_is_ignored() -> None:
    cal = calendar(10)
    sig = signal(pd.Timestamp("2030-01-01"), pd.Timestamp("2030-02-01"))
    result = simulate_portfolio(
        [sig], flat_prices(["A"], cal), cal, n_slots=1, cost=ZERO_COST
    )

    assert result.n_trades == 0


@pytest.mark.unit
def test_rejects_non_positive_slots() -> None:
    cal = calendar(10)
    with pytest.raises(ValueError, match="n_slots"):
        simulate_portfolio([], flat_prices(["A"], cal), cal, n_slots=0, cost=ZERO_COST)


@pytest.mark.unit
def test_rejects_empty_calendar() -> None:
    with pytest.raises(ValueError, match="日曆"):
        simulate_portfolio([], flat_prices(["A"], []), [], n_slots=3, cost=ZERO_COST)


@pytest.mark.unit
def test_rejects_exit_before_decision() -> None:
    cal = calendar(10)
    with pytest.raises(ValueError, match="exit_date"):
        Signal(
            decision_date=cal[5],
            exit_date=cal[1],
            stock_id="A",
            gross_return=0.1,
            rank_score=1.0,
            tier=Tier.LARGE,
        )


@pytest.mark.unit
def test_result_is_immutable() -> None:
    cal = calendar(10)
    result = simulate_portfolio([], flat_prices(["A"], cal), cal, n_slots=1, cost=ZERO_COST)
    with pytest.raises(Exception):
        result.total_return = 1.0  # type: ignore[misc]
