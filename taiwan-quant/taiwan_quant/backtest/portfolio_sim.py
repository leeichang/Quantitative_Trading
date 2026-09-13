"""
多槽位組合模擬

## 為什麼需要它（`engine.py` 不夠用）

`engine.py` 把各期報酬**依序複合**，結構上只容得下一個持倉：

    週 1 報酬 × 週 2 報酬 × ...

但 CLAUDE.md 的設計是 **Top 3 同時持有**。用依序複合去評價它，等於
把「3 檔並行、幾乎全時間在市場」硬套成「一次一檔、大部分時間空手」。

實測的後果很嚴重：

    移動停損版 OOS 曝險 23~39%，對照組買進持有 100%

在 +240% 的多頭裡，光是這個曝險差距就足以輸掉，**與選股能力無關**。
不修掉它，任何「策略輸給買進持有」的結論都不成立。

## 模型

    槽位     固定 n 個（預設 3，對應 Top 3）
    配置     開倉時撥出「當下權益 / n」，受現金餘額限制
    出場     到標記給定的出場日，槽位釋放、現金回收
    現金     空手期間報酬為 0（不假設有貨幣市場收益）
    成本     一趟來回在出場時一次扣除

權益**逐日標記市值**——只在出場日跳一次會低估最大回撤，而 MaxDD 正是
跟買進持有對比時最關鍵的風險指標。

## 邊界

本模組吃的是**已實現的訊號**（`Signal`），出場日與毛報酬都由 `labeling`
決定。它不做選股也不做標記，所以不可能偷看未來——它只看得到已經發生的
結果，以及每個訊號自己的決策日。

`engine.py` 保留：它算換手率與成本敏感度的介面已經被規格 13、15 綁死，
而且單一持倉的視角在檢查「單筆交易品質」時仍然有用。兩者職責不同：

    engine.py        單筆交易的品質（報酬、成本、換手）
    portfolio_sim.py 整個組合的表現（權益曲線、曝險、回撤）
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from taiwan_quant.config.costs import DEFAULT, CostModel, Tier

TRADING_DAYS_PER_YEAR = 252.0

PriceLookup = Callable[[str, pd.Timestamp], float | None]
"""(股票代號, 日期) → 當日價格；查不到回 None"""


@dataclass(frozen=True)
class Signal:
    """
    一筆已實現的交易訊號。

    `exit_date` 與 `gross_return` 由 `labeling` 給——模擬器不重算出場，
    否則同一套規則會有兩份實作。
    """

    decision_date: pd.Timestamp
    """訊號產生日。**進場不得早於這一天**"""

    exit_date: pd.Timestamp
    stock_id: str
    gross_return: float
    rank_score: float
    """排名依據；槽位不足時取高的"""

    tier: Tier

    def __post_init__(self) -> None:
        if self.exit_date < self.decision_date:
            raise ValueError(
                f"exit_date {self.exit_date} 不可早於 decision_date {self.decision_date}"
            )
        if not math.isfinite(self.gross_return):
            raise ValueError(f"gross_return 必須為有限值，得到 {self.gross_return}")


@dataclass(frozen=True)
class ClosedTrade:
    """已平倉的交易紀錄（稽核用，禁令 7、8）"""

    stock_id: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    allocation: float
    """開倉時撥出的權益比例（以期初權益為 1.0 計）"""

    gross_return: float
    cost_rate: float

    @property
    def net_return(self) -> float:
        return self.gross_return - self.cost_rate


@dataclass(frozen=True)
class PortfolioSimResult:
    """組合模擬結果"""

    equity: pd.Series
    """逐日權益曲線，起點 1.0"""

    trades: tuple[ClosedTrade, ...]

    total_return: float
    annualized_return: float
    sharpe: float | None
    """年化 Sharpe（無風險利率視為 0）；樣本不足或零變異時為 None"""

    max_drawdown: float

    exposure: float
    """
    曝險：持倉槽位日 / (槽位數 × 日曆日)。

    **這是跟買進持有對比時的關鍵欄位。** 買進持有的曝險是 100%；
    曝險 30% 的策略即使每筆都贏，累積報酬仍然可能大幅落後。
    """

    max_concurrent: int
    n_slots: int

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    def describe(self) -> str:
        sharpe = f"{self.sharpe:.3f}" if self.sharpe is not None else "n/a"
        return "\n".join([
            f"槽位            {self.n_slots}（最高同時持有 {self.max_concurrent}）",
            f"交易筆數        {self.n_trades}",
            f"總報酬          {self.total_return * 100:+.2f}%",
            f"年化報酬        {self.annualized_return * 100:+.2f}%",
            f"Sharpe          {sharpe}",
            f"最大回撤        {self.max_drawdown * 100:.2f}%",
            f"曝險            {self.exposure * 100:.1f}%",
        ])


@dataclass
class _OpenPosition:
    """模擬進行中的持倉（內部可變狀態，不外露）"""

    stock_id: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    allocation: float
    entry_price: float
    gross_return: float
    cost_rate: float


def _annualize(cumulative: float, n_days: int) -> float:
    """以實際交易日數年化"""
    if n_days <= 0:
        return 0.0
    growth = 1.0 + cumulative
    if growth <= 0:
        return -1.0
    return growth ** (TRADING_DAYS_PER_YEAR / n_days) - 1.0


def _sharpe(daily_returns: list[float]) -> float | None:
    """年化 Sharpe。樣本 < 2 或零變異時回 None（不可回 0 或 inf）"""
    if len(daily_returns) < 2:
        return None
    sd = statistics.stdev(daily_returns)
    if sd == 0:
        return None
    return statistics.fmean(daily_returns) / sd * math.sqrt(TRADING_DAYS_PER_YEAR)


def _max_drawdown(equity: list[float]) -> float:
    peak = equity[0] if equity else 1.0
    worst = 0.0
    for value in equity:
        peak = max(peak, value)
        if peak > 0:
            worst = max(worst, (peak - value) / peak)
    return worst


def simulate_portfolio(
    signals: list[Signal],
    price_lookup: PriceLookup,
    calendar: list[pd.Timestamp],
    n_slots: int = 3,
    cost: CostModel = DEFAULT,
) -> PortfolioSimResult:
    """
    模擬多槽位組合。

    Args:
        signals: 已實現的交易訊號（可亂序）
        price_lookup: (代號, 日期) → 價格，用於逐日標記市值
        calendar: 模擬用的交易日曆（升冪）
        n_slots: 同時可持有的檔數
        cost: 成本模型（一律來自 config/costs.py）

    Returns:
        PortfolioSimResult

    Raises:
        ValueError: n_slots < 1 或日曆為空

    決策日不在日曆內的訊號會被忽略——那代表它落在模擬區間之外。
    """
    if n_slots < 1:
        raise ValueError(f"n_slots 至少為 1，得到 {n_slots}")
    if not calendar:
        raise ValueError("交易日曆不可為空")

    by_date: dict[pd.Timestamp, list[Signal]] = {}
    for s in signals:
        by_date.setdefault(s.decision_date, []).append(s)
    for batch in by_date.values():
        # 排名高的先進場；同分時用代號確保可重現（禁令 7、8）
        batch.sort(key=lambda s: (-s.rank_score, s.stock_id))

    cash = 1.0
    open_positions: list[_OpenPosition] = []
    closed: list[ClosedTrade] = []
    equity_values: list[float] = []
    slot_days = 0
    max_concurrent = 0

    for day in calendar:
        # ── 1. 出場（先釋放槽位，同一天才可能有新倉接上） ──
        still_open: list[_OpenPosition] = []
        for p in open_positions:
            if p.exit_date <= day:
                cash += p.allocation * (1.0 + p.gross_return - p.cost_rate)
                closed.append(ClosedTrade(
                    stock_id=p.stock_id,
                    entry_date=p.entry_date,
                    exit_date=day,
                    allocation=p.allocation,
                    gross_return=p.gross_return,
                    cost_rate=p.cost_rate,
                ))
            else:
                still_open.append(p)
        open_positions = still_open

        # ── 2. 進場 ──
        for s in by_date.get(day, []):
            if len(open_positions) >= n_slots:
                break
            if any(p.stock_id == s.stock_id for p in open_positions):
                # 同一檔已在持倉 → 再開一次是加碼，不是分散
                continue

            entry_price = price_lookup(s.stock_id, day)
            if entry_price is None or entry_price <= 0:
                continue

            held_value = sum(
                p.allocation * (price_lookup(p.stock_id, day) or p.entry_price)
                / p.entry_price
                for p in open_positions
            )
            target = (cash + held_value) / n_slots
            allocation = min(target, cash)
            if allocation <= 0:
                continue

            cash -= allocation
            open_positions.append(_OpenPosition(
                stock_id=s.stock_id,
                entry_date=day,
                exit_date=s.exit_date,
                allocation=allocation,
                entry_price=entry_price,
                gross_return=s.gross_return,
                cost_rate=cost.round_trip_rate(s.tier),
            ))

        # ── 3. 逐日標記市值 ──
        held_value = 0.0
        for p in open_positions:
            price = price_lookup(p.stock_id, day)
            ratio = price / p.entry_price if price and p.entry_price > 0 else 1.0
            held_value += p.allocation * ratio

        equity_values.append(cash + held_value)
        slot_days += len(open_positions)
        max_concurrent = max(max_concurrent, len(open_positions))

    # 日曆結束仍未出場的部位按最後市值結算，避免權益憑空消失
    equity = pd.Series(equity_values, index=pd.DatetimeIndex(calendar))
    total_return = float(equity.iloc[-1]) - 1.0

    daily_returns = [
        equity_values[i] / equity_values[i - 1] - 1.0
        for i in range(1, len(equity_values))
        if equity_values[i - 1] > 0
    ]

    return PortfolioSimResult(
        equity=equity,
        trades=tuple(closed),
        total_return=total_return,
        annualized_return=_annualize(total_return, len(calendar)),
        sharpe=_sharpe(daily_returns),
        max_drawdown=_max_drawdown(equity_values),
        exposure=slot_days / (n_slots * len(calendar)),
        max_concurrent=max_concurrent,
        n_slots=n_slots,
    )
