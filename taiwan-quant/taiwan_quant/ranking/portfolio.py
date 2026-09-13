"""
Top 3 選股與部位規模（triple-barrier 版）

依據 D4 與 CLAUDE.md 的投組約束。

投組約束與部位規模在 `constraints.py`，與移動停損版共用。
本模組只負責 triple-barrier 特有的部分：**動態進場門檻**。

## 進場門檻是動態的（D7 修訂）

    P(+1) ≥ (stop_pct + round_trip_cost) / (target_pct + stop_pct)

不是固定的 0.55。門檻隨每檔標的的柵欄寬度與流動性分層計算——
中型股滑價高，門檻自然就高（要更確定才值得做）。

以 D7 修訂後的實測參數（target 6.20%、stop 2.38%、0050）算出 **40.22%**，
這就是 CLAUDE.md 的紅線。

## ⚠️ 實測結論：這條路線在 OOS 輸給買進持有

見 `../docs/需求規劃/202609/03_策略可行性最終結論.md`：9 組策略全部輸給
等權買進持有，機制原因是**目標價把上檔封死**。路線 A 的
`trailing_portfolio.py` 是對此的回應。

本模組保留，因為它仍是有效的對照組——沒有對照就無法證明移動停損真的
比較好。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from taiwan_quant.config.costs import DEFAULT, CostModel, Tier
from taiwan_quant.ranking.constraints import (
    DEFENSIVE_BETA_MAX,
    HIGH_VOLATILITY_PERCENTILE,
    MAX_CORRELATION,
    MAX_HIGH_VOLATILITY,
    MAX_POSITION_PCT,
    MAX_SAME_INDUSTRY,
    RISK_PER_TRADE,
    TOP_N,
    RejectReason,
    Rejection,
    greedy_pick,
    position_shares,
    promote_defensive,
)

__all__ = [
    "DEFENSIVE_BETA_MAX",
    "HIGH_VOLATILITY_PERCENTILE",
    "MAX_CORRELATION",
    "MAX_HIGH_VOLATILITY",
    "MAX_POSITION_PCT",
    "MAX_SAME_INDUSTRY",
    "MIN_RISK_REWARD",
    "RISK_PER_TRADE",
    "TOP_N",
    "Candidate",
    "PortfolioResult",
    "Position",
    "RejectReason",
    "Rejection",
    "entry_threshold",
    "position_shares",
    "select_portfolio",
]

MIN_RISK_REWARD = 2.0
"""R:R 門檻。不為湊檔數放寬"""


@dataclass(frozen=True)
class Candidate:
    """一個候選標的（由模型與柵欄推導產出）"""

    stock_id: str
    prob_up: float
    """模型預測的 P(+1)"""

    target_pct: float
    stop_pct: float
    entry_price: float
    tier: Tier
    industry: str
    volatility_pct: float
    """ATR 在標的池中的分位（0~1）"""

    beta: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.prob_up <= 1.0:
            raise ValueError(f"prob_up 必須落在 [0, 1]，得到 {self.prob_up}")
        if self.target_pct <= 0:
            raise ValueError(f"target_pct 必須為正，得到 {self.target_pct}")
        if self.stop_pct <= 0:
            raise ValueError(f"stop_pct 必須為正，得到 {self.stop_pct}")
        if self.entry_price <= 0:
            raise ValueError(f"entry_price 必須為正，得到 {self.entry_price}")

    @property
    def risk_reward(self) -> float:
        return self.target_pct / self.stop_pct

    @property
    def is_high_volatility(self) -> bool:
        return self.volatility_pct > HIGH_VOLATILITY_PERCENTILE

    @property
    def is_defensive(self) -> bool:
        return self.beta < DEFENSIVE_BETA_MAX

    def expected_return(self, cost: CostModel) -> float:
        """
        期望淨報酬（扣一趟來回成本）。

            P(+1) × target − P(−1) × stop − cost

        把「時間柵到期」的情況併入 P(−1) 是保守處理：到期時實際報酬
        介於兩柵之間，這裡假設最差情況。
        """
        gross = self.prob_up * self.target_pct - (1 - self.prob_up) * self.stop_pct
        return gross - cost.round_trip_rate(self.tier)

    @property
    def target_price(self) -> float:
        return self.entry_price * (1 + self.target_pct)

    @property
    def stop_price(self) -> float:
        return self.entry_price * (1 - self.stop_pct)


@dataclass(frozen=True)
class Position:
    """入選的部位"""

    candidate: Candidate
    shares: int
    position_value: float
    capital_pct: float
    expected_return: float
    risk_reward: float


@dataclass(frozen=True)
class PortfolioResult:
    """選股結果"""

    positions: tuple[Position, ...]
    rejected: tuple[Rejection, ...]
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def has_defensive(self) -> bool:
        return any(p.candidate.is_defensive for p in self.positions)

    @property
    def total_capital_pct(self) -> float:
        return sum(p.capital_pct for p in self.positions)

    def describe(self) -> str:
        if not self.positions:
            lines = ["本週無符合條件的標的。"]
            lines.extend(f"  ⚠️  {w}" for w in self.warnings)
            return "\n".join(lines)

        medals = ["🥇", "🥈", "🥉"]
        lines: list[str] = []
        for i, p in enumerate(self.positions):
            c = p.candidate
            lines.extend([
                f"{medals[i] if i < len(medals) else '  '} {c.stock_id}",
                f"   進場區間  {c.entry_price:,.0f}",
                f"   目標價    {c.target_price:,.0f}  (+{c.target_pct * 100:.2f}%)",
                f"   失效價    {c.stop_price:,.0f}  (−{c.stop_pct * 100:.2f}%)",
                f"   R:R       1 : {p.risk_reward:.2f}",
                f"   P(先達標)  {c.prob_up * 100:.1f}%",
                f"   期望淨報酬 {p.expected_return * 100:+.2f}%",
                f"   建議部位  約 {p.capital_pct * 100:.1f}% 資金"
                f"（零股 {p.shares:,} 股）",
                f"   ⓘ 零股交易，價差較整張寬，成本已含"
                f"{'0.3%' if c.tier is Tier.LARGE else '0.4%'} 滑價",
                "",
            ])

        lines.append(f"合計配置 {self.total_capital_pct * 100:.1f}% 資金")
        if self.warnings:
            lines.append("")
            lines.extend(f"⚠️  {w}" for w in self.warnings)
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
# 動態進場門檻
# ══════════════════════════════════════════════════════════════


def entry_threshold(
    target_pct: float,
    stop_pct: float,
    cost: CostModel = DEFAULT,
    tier: Tier = Tier.LARGE,
) -> float:
    """
    含成本的損益兩平勝率。

        p × target − (1 − p) × stop − cost = 0
        p = (stop + cost) / (target + stop)

    Args:
        target_pct / stop_pct: 柵欄寬度
        cost: 成本模型
        tier: 流動性分層

    Returns:
        P(+1) 的最低要求

    實測（D7 修訂後參數）：
        target 6.20%、stop 2.38%、0050 成本 1.071% → **40.22%**
    """
    if target_pct <= 0:
        raise ValueError(f"target_pct 必須為正，得到 {target_pct}")
    if stop_pct <= 0:
        raise ValueError(f"stop_pct 必須為正，得到 {stop_pct}")

    return (stop_pct + cost.round_trip_rate(tier)) / (target_pct + stop_pct)


# ══════════════════════════════════════════════════════════════
# 選股
# ══════════════════════════════════════════════════════════════


def _prefilter(
    candidates: list[Candidate],
    capital: float,
    cost: CostModel,
) -> tuple[list[Candidate], list[Rejection]]:
    """套用與投組無關的個別條件：進場門檻、R:R、部位規模"""
    passed: list[Candidate] = []
    rejected: list[Rejection] = []

    for c in candidates:
        threshold = entry_threshold(c.target_pct, c.stop_pct, cost, c.tier)
        if c.prob_up < threshold:
            rejected.append(Rejection(
                c.stock_id,
                RejectReason.BELOW_ENTRY_THRESHOLD,
                f"P(+1) {c.prob_up:.1%} < 門檻 {threshold:.2%}"
                f"（target {c.target_pct:.2%}、stop {c.stop_pct:.2%}、"
                f"成本 {cost.round_trip_rate(c.tier):.3%}）",
            ))
            continue

        if c.risk_reward < MIN_RISK_REWARD:
            rejected.append(Rejection(
                c.stock_id,
                RejectReason.BELOW_RISK_REWARD,
                f"R:R {c.risk_reward:.2f} < {MIN_RISK_REWARD}。風險紀律不為湊檔數放寬",
            ))
            continue

        if position_shares(capital, c.entry_price, c.stop_pct) < 1:
            rejected.append(Rejection(
                c.stock_id,
                RejectReason.POSITION_TOO_SMALL,
                f"建議股數 < 1（單股 {c.entry_price:,.0f} 元，"
                f"資金上限 {capital * MAX_POSITION_PCT:,.0f} 元）",
            ))
            continue

        passed.append(c)

    return passed, rejected


def select_portfolio(
    candidates: list[Candidate],
    capital: float,
    correlations: dict[tuple[str, str], float],
    cost: CostModel = DEFAULT,
) -> PortfolioResult:
    """
    從候選中挑出最多 3 檔並計算部位規模。

    Args:
        candidates: 候選標的
        capital: 總資金
        correlations: {(stock_a, stock_b): 60 日報酬相關係數}，順序無關
        cost: 成本模型

    Returns:
        PortfolioResult（含入選部位、剔除原因、警告）

    Raises:
        ValueError: capital 非正

    流程：
        1. 個別過濾（進場門檻、R:R、部位規模）
        2. 依期望淨報酬排序
        3. 貪婪挑選並套用投組約束
        4. 確保至少 1 支 defensive（必要時替換）
        5. 計算部位規模

    寧可少推幾檔也不違反約束——湊滿 3 檔不是目標。
    """
    if capital <= 0:
        raise ValueError(f"capital 必須為正，得到 {capital}")

    warnings: list[str] = []

    if not candidates:
        return PortfolioResult(
            positions=(),
            rejected=(),
            warnings=("本週無任何候選標的進入排名",),
        )

    passed, rejected = _prefilter(candidates, capital, cost)

    if not passed:
        warnings.append("所有候選都未通過進場門檻或風險紀律，本週不推播")
        return PortfolioResult(
            positions=(), rejected=tuple(rejected), warnings=tuple(warnings)
        )

    # 依期望淨報酬排序；同值時用代號確保可重現（禁令 7、8）
    ordered = sorted(passed, key=lambda c: (-c.expected_return(cost), c.stock_id))

    picked, pick_rejections = greedy_pick(ordered, correlations)
    rejected.extend(pick_rejections)

    promoted = promote_defensive(picked, ordered, correlations)
    if promoted is not None:
        picked = promoted
    else:
        warnings.append(
            "組合中沒有 defensive（低 beta）標的，"
            "全部集中在高 beta——市場回檔時三檔會同步下跌"
        )

    # 維持期望值排序
    picked.sort(key=lambda c: (-c.expected_return(cost), c.stock_id))

    positions: list[Position] = []
    for c in picked:
        shares = position_shares(capital, c.entry_price, c.stop_pct)
        value = shares * c.entry_price
        positions.append(Position(
            candidate=c,
            shares=shares,
            position_value=value,
            capital_pct=value / capital,
            expected_return=c.expected_return(cost),
            risk_reward=c.risk_reward,
        ))

    if len(positions) < TOP_N:
        warnings.append(
            f"僅選出 {len(positions)} 檔（目標 {TOP_N} 檔）："
            "候選不足以在滿足投組約束下湊滿。寧可少推也不違反風控"
        )

    return PortfolioResult(
        positions=tuple(positions),
        rejected=tuple(rejected),
        warnings=tuple(warnings),
    )
