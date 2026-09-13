"""
投組約束與部位規模（選股法共用）

## 為什麼抽出來

系統有兩種選股法：

    portfolio.py            triple-barrier   門檻 P(+1) ≥ (stop+cost)/(target+stop)
    trailing_portfolio.py   移動停損         門檻 E[毛報酬] − cost > 0

**只有個別過濾條件不同**，投組約束（產業／波動／相關／defensive）與
部位規模完全一樣——它們管的是「三檔擺在一起會不會出事」，跟怎麼標記
無關。寫兩份再祈禱它們同步，遲早有一份會落後。

## 約束清單（CLAUDE.md）

    最多 2 支同產業
    最多 1 支高波動（ATR 分位 > 80%）
    至少 1 支 defensive（低 beta）
    三檔之間 60 日報酬相關係數 < 0.7

## 核心設計原則

**寧可少推幾檔，也不違反風控約束。** 湊滿 3 檔不是目標。

ChatGPT 來源的原話（見 `來源原文/02_ChatGPT`）：

    Top 3 若全部是 AI / 半導體 = Factor concentration，風險很高。

## 為什麼用 beta 當 defensive 代理

CLAUDE.md 原文寫「至少 1 支 defensive / 高股息 / 低 beta」。

實測上游資料 `stock_daily_per`（含殖利率）覆蓋率僅 4.3%（FinMind 免費額度
撞牆），**高股息無法使用**。beta 可從價格資料直接算，所以用它當代理。

拿到完整殖利率資料後應改為「低 beta **或** 高股息」。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

# ── 投組約束 ──────────────────────────────────────────────

TOP_N = 3
MAX_SAME_INDUSTRY = 2
MAX_HIGH_VOLATILITY = 1
MAX_CORRELATION = 0.7

HIGH_VOLATILITY_PERCENTILE = 0.80
"""ATR 分位超過此值視為高波動"""

DEFENSIVE_BETA_MAX = 1.0
"""beta 低於此值視為 defensive"""

# ── 部位規模（D4） ────────────────────────────────────────

RISK_PER_TRADE = 0.01
"""單筆風險上限：總資金 1%"""

MAX_POSITION_PCT = 0.33
"""單檔資金上限：總資金 33%（3 檔不超過 100%）"""


class RejectReason(str, Enum):
    """候選被剔除的原因。每筆剔除都要記錄，否則無法稽核（禁令 7、8）"""

    BELOW_ENTRY_THRESHOLD = "below_entry_threshold"
    BELOW_RISK_REWARD = "below_risk_reward"
    STATISTICALLY_INSIGNIFICANT = "statistically_insignificant"
    INDUSTRY_LIMIT = "industry_limit"
    VOLATILITY_LIMIT = "volatility_limit"
    CORRELATION_LIMIT = "correlation_limit"
    CORRELATION_UNKNOWN = "correlation_unknown"
    POSITION_TOO_SMALL = "position_too_small"
    TOP_N_REACHED = "top_n_reached"


@dataclass(frozen=True)
class Rejection:
    """剔除紀錄"""

    stock_id: str
    reason: RejectReason
    detail: str


class PortfolioCandidate(Protocol):
    """
    投組約束只需要這四項。

    刻意不要求 target_pct / prob_up——那些是 triple-barrier 才有的概念，
    放進 Protocol 會讓移動停損版被迫填假值。
    """

    stock_id: str
    industry: str

    @property
    def is_high_volatility(self) -> bool: ...

    @property
    def is_defensive(self) -> bool: ...


def correlation(
    correlations: dict[tuple[str, str], float], a: str, b: str
) -> float | None:
    """查兩檔的相關係數；順序無關。查不到回 None"""
    if (a, b) in correlations:
        return correlations[(a, b)]
    if (b, a) in correlations:
        return correlations[(b, a)]
    return None


def violates_constraints[T: PortfolioCandidate](
    candidate: T,
    picked: list[T],
    correlations: dict[tuple[str, str], float],
) -> Rejection | None:
    """檢查加入 candidate 是否違反投組約束；沒問題回 None"""
    same_industry = sum(1 for p in picked if p.industry == candidate.industry)
    if same_industry >= MAX_SAME_INDUSTRY:
        return Rejection(
            candidate.stock_id,
            RejectReason.INDUSTRY_LIMIT,
            f"同產業（{candidate.industry}）已有 {same_industry} 檔，"
            f"上限 {MAX_SAME_INDUSTRY}",
        )

    if candidate.is_high_volatility:
        high_vol = sum(1 for p in picked if p.is_high_volatility)
        if high_vol >= MAX_HIGH_VOLATILITY:
            return Rejection(
                candidate.stock_id,
                RejectReason.VOLATILITY_LIMIT,
                f"高波動標的已有 {high_vol} 檔，上限 {MAX_HIGH_VOLATILITY}",
            )

    for existing in picked:
        rho = correlation(correlations, candidate.stock_id, existing.stock_id)
        if rho is None:
            return Rejection(
                candidate.stock_id,
                RejectReason.CORRELATION_UNKNOWN,
                f"缺少與 {existing.stock_id} 的相關係數資料。"
                "保守拒絕——當成 0 等於假設無關，可能讓高度相關的標的同時入選",
            )
        if abs(rho) >= MAX_CORRELATION:
            return Rejection(
                candidate.stock_id,
                RejectReason.CORRELATION_LIMIT,
                f"與 {existing.stock_id} 相關係數 {rho:.2f} ≥ {MAX_CORRELATION}",
            )

    return None


def greedy_pick[T: PortfolioCandidate](
    ordered: list[T],
    correlations: dict[tuple[str, str], float],
) -> tuple[list[T], list[Rejection]]:
    """依序貪婪挑選，記錄每筆剔除原因。`ordered` 須已依期望值排序"""
    picked: list[T] = []
    rejected: list[Rejection] = []

    for c in ordered:
        if len(picked) >= TOP_N:
            rejected.append(
                Rejection(c.stock_id, RejectReason.TOP_N_REACHED, f"已選滿 {TOP_N} 檔")
            )
            continue

        violation = violates_constraints(c, picked, correlations)
        if violation is not None:
            rejected.append(violation)
            continue

        picked.append(c)

    return picked, rejected


def promote_defensive[T: PortfolioCandidate](
    picked: list[T],
    ordered: list[T],
    correlations: dict[tuple[str, str], float],
) -> list[T] | None:
    """
    嘗試把一檔 defensive 換進組合。

    貪婪選法依期望值排序取前 N，可能全部是高 beta。CLAUDE.md 要求
    至少 1 支 defensive，所以要主動替換。

    做法：從期望值最低的持股開始，逐一嘗試換成最佳的 defensive 候選，
    換完後仍須滿足所有約束。換不成回 None。
    """
    if any(c.is_defensive for c in picked):
        return picked

    taken = {p.stock_id for p in picked}
    defensive_pool = [c for c in ordered if c.is_defensive and c.stock_id not in taken]
    if not defensive_pool:
        return None

    # 從期望值最低的持股開始換（犧牲最小）
    for drop_idx in range(len(picked) - 1, -1, -1):
        remaining = [c for i, c in enumerate(picked) if i != drop_idx]
        for defensive in defensive_pool:
            if violates_constraints(defensive, remaining, correlations) is None:
                return remaining + [defensive]

    return None


def position_shares(capital: float, entry_price: float, stop_pct: float) -> int:
    """
    建議股數 = floor(min(風險倒推, 資金上限))

        風險倒推 = 總資金 × 1% ÷ (entry × stop_pct)
        資金上限 = 總資金 × 33% ÷ entry

    Args:
        capital: 總資金
        entry_price: 進場價
        stop_pct: 停損幅度（移動停損版傳 trail_pct，即初始停損距離）

    Returns:
        建議股數（**零股，不是張數**）。算出 < 1 股時回 0。

    為什麼要第二道 33% 上限：停損幅度只有 2% 時，1% 風險倒推出的部位
    會是資金的 50%，三檔就爆到 150%。

    為什麼是零股：總資金 40 萬買不起一張台積電（2,430 × 1,000 = 243 萬）。
    """
    if capital <= 0:
        raise ValueError(f"capital 必須為正，得到 {capital}")
    if entry_price <= 0:
        raise ValueError(f"entry_price 必須為正，得到 {entry_price}")
    if stop_pct <= 0:
        raise ValueError(f"stop_pct 必須為正，得到 {stop_pct}")

    risk_based = capital * RISK_PER_TRADE / (entry_price * stop_pct)
    cap_based = capital * MAX_POSITION_PCT / entry_price
    return int(min(risk_based, cap_based))
