"""
Top 3 選股（移動停損版，路線 A）

投組約束與部位規模在 `constraints.py`，與 triple-barrier 版共用。
本模組只負責移動停損特有的部分：**沒有目標價時，門檻怎麼定**。

## 為什麼門檻要改

triple-barrier 的門檻建立在「目標固定」之上：

    P(+1) ≥ (stop + cost) / (target + stop)

移動停損沒有 `target`，這條式子連寫都寫不出來。取而代之的是直接
比較期望值與成本：

    E[毛報酬] − 一趟來回成本 > 0

期望值由 `validation.calibration.ReturnCalibrator` 給——那是分數所在
分箱的**歷史平均實際報酬**，不是命中率換算的。

## 第二道門檻：優勢要大於雜訊

光是「淨期望為正」不夠。OOS 報告已經指出：

    獨立交易期只有 6~15 個，統計結論極不穩定。

所以再加一道：

    E[毛報酬] − 成本  ≥  z × 標準差 / sqrt(樣本數)

也就是**優勢至少要大於一個標準誤**。這不是拍腦袋的安全邊際，而是
「這個估計值本身的誤差有多大」的直接量測。樣本少、離散大時自動變嚴，
樣本多、穩定時自動放寬。

離散度未知（樣本 < 2）時保守拒絕——當成 0 等於假設「這個估計完全
沒有誤差」，那是最危險的假設。

## 沒有 R:R

移動停損沒有目標價，所以沒有風報比可算。

**不得拿「期望報酬 / trail_pct」硬充 R:R**——那是兩個不同的量
（期望值 vs 最好情況），混用會讓門檻的意義說不清楚。風險紀律由上面
那道統計顯著性門檻承擔。

## 輸出形式

使用者要的是「下週買進、持續持有 3 個月」，所以推播內容是：

    進場參考價 + 移動停損幅度 + 最長持有期

**沒有目標價。** 輸出裡若還出現目標價，代表上檔又被封死了——那正是
OOS 輸給買進持有的機制原因。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from taiwan_quant.config.costs import DEFAULT, CostModel, Tier
from taiwan_quant.ranking.constraints import (
    DEFENSIVE_BETA_MAX,
    HIGH_VOLATILITY_PERCENTILE,
    MAX_POSITION_PCT,
    TOP_N,
    RejectReason,
    Rejection,
    greedy_pick,
    position_shares,
    promote_defensive,
)

ENTRY_TRANCHES = 3
"""
分批進場的批數。

使用者要的是「拉回分批買 + 移動停損」，這是前半。

⚠️ **回測沒有模擬分批**——標記層假設 T+1 開盤一次買足。分批是執行面的
建議，它的績效沒有被驗證過。要驗證得做盤中撮合模擬，那是另一個量級的
工作，而且需要分價量資料。
"""

TRADING_DAYS_PER_MONTH = 21
"""台股一個月約 21 個交易日，用於把持有期換算成使用者看得懂的月數"""

DEFAULT_EDGE_Z = 1.0
"""
優勢須大於幾個標準誤。

1.0 是相對寬鬆的設定（約 84% 單尾信心）。用 1.96 會嚴到幾乎選不出東西，
在目前樣本量下等於永遠不推播；1.0 保留可操作性，同時擋掉最明顯的雜訊。
這個值應在訓練期掃描後定案，不是最終答案。
"""


@dataclass(frozen=True)
class EntryTranche:
    """分批進場的其中一批"""

    price: float
    weight: float
    """佔總部位的比例"""

    shares: int
    """該批股數；未指定總股數時為 0"""


@dataclass(frozen=True)
class TrailingCandidate:
    """
    一個候選標的（由移動停損標記與期望報酬校準產出）。

    `expected_gross_return` / `return_std` / `n_samples` 三者都來自
    `ReturnCalibrator` 的同一個分箱——分開傳會讓下游無從檢查它們是否一致。
    """

    stock_id: str

    expected_gross_return: float
    """該分數所在分箱的歷史平均實際毛報酬"""

    return_std: float | None
    """該分箱的報酬標準差；樣本 < 2 時為 None"""

    n_samples: int
    """該分箱的樣本數"""

    trail_pct: float
    """移動停損幅度，由 `labeling.trail_width.derive_trail_width` 推導"""

    entry_price: float
    tier: Tier
    industry: str

    volatility_pct: float
    """ATR 在標的池中的分位（0~1）"""

    beta: float

    max_horizon: int
    """最長持有交易日數"""

    def __post_init__(self) -> None:
        if not math.isfinite(self.expected_gross_return):
            raise ValueError(
                f"expected_gross_return 必須為有限值，得到 {self.expected_gross_return}。"
                "NaN 不會讓門檻比較拋錯，而是讓比較永遠為 False——靜默通過。"
            )
        if not 0.0 < self.trail_pct < 1.0:
            raise ValueError(f"trail_pct 必須落在 (0, 1)，得到 {self.trail_pct}")
        if self.entry_price <= 0:
            raise ValueError(f"entry_price 必須為正，得到 {self.entry_price}")
        if self.n_samples < 0:
            raise ValueError(f"n_samples 不可為負，得到 {self.n_samples}")
        if self.max_horizon < 1:
            raise ValueError(f"max_horizon 至少為 1，得到 {self.max_horizon}")

    # ── 投組約束需要的介面（PortfolioCandidate Protocol） ──

    @property
    def is_high_volatility(self) -> bool:
        return self.volatility_pct > HIGH_VOLATILITY_PERCENTILE

    @property
    def is_defensive(self) -> bool:
        return self.beta < DEFENSIVE_BETA_MAX

    # ── 期望值與顯著性 ──

    def expected_net_return(self, cost: CostModel) -> float:
        """期望毛報酬扣掉一趟來回成本"""
        return self.expected_gross_return - cost.round_trip_rate(self.tier)

    @property
    def standard_error(self) -> float | None:
        """
        期望報酬估計值的標準誤 = 標準差 / sqrt(樣本數)。

        離散度未知或樣本 < 2 時回 `None`（**不是 0**）。0 代表
        「算出來剛好沒有誤差」，None 代表「無法評估」——下游必須
        分得出這兩者。
        """
        if self.return_std is None or self.n_samples < 2:
            return None
        if not math.isfinite(self.return_std) or self.return_std < 0:
            return None
        return self.return_std / math.sqrt(self.n_samples)

    def has_significant_edge(self, cost: CostModel, z: float = DEFAULT_EDGE_Z) -> bool:
        """優勢是否大於 z 個標準誤"""
        se = self.standard_error
        if se is None:
            return False
        return self.expected_net_return(cost) >= z * se

    # ── 價格 ──

    @property
    def initial_stop_price(self) -> float:
        """
        進場時的停損價。

        之後會隨期間最高價上移（只升不降），所以這只是起點，不是失效價。
        """
        return self.entry_price * (1 - self.trail_pct)

    @property
    def holding_months(self) -> float:
        return self.max_horizon / TRADING_DAYS_PER_MONTH

    def entry_ladder(self, total_shares: int = 0) -> tuple[EntryTranche, ...]:
        """
        拉回分批買的價格階梯。

        Args:
            total_shares: 建議總股數；給 0 時各批 shares 為 0（只看價格）

        Returns:
            由高到低的三批，權重等分。

        價格分佈在 `[初始停損, 進場價]` 之間，但**最低一批仍高於停損**：

            第 i 批價格 = 進場價 × (1 − i × trail_pct / 批數)

        最低批在 `進場價 × (1 − (批數−1)/批數 × trail_pct)`，
        距停損還有 `trail_pct / 批數` 的空間。

        **不可在停損之下加碼**——那等於一邊說「跌破就走」一邊在跌破後
        買更多，兩個決定互相矛盾。

        移動停損線本身不受分批影響（它錨定在期間最高價），分批只降低
        平均成本。
        """
        step = self.trail_pct / ENTRY_TRANCHES
        weight = 1.0 / ENTRY_TRANCHES
        # 整數股數用 floor，餘數留給第一批（現價那批最可能成交）
        per_tranche = total_shares // ENTRY_TRANCHES
        remainder = total_shares - per_tranche * ENTRY_TRANCHES

        return tuple(
            EntryTranche(
                price=self.entry_price * (1 - i * step),
                weight=weight,
                shares=per_tranche + (remainder if i == 0 else 0),
            )
            for i in range(ENTRY_TRANCHES)
        )


@dataclass(frozen=True)
class TrailingPosition:
    """入選的部位"""

    candidate: TrailingCandidate
    shares: int
    position_value: float
    capital_pct: float
    expected_net_return: float


@dataclass(frozen=True)
class TrailingPortfolioResult:
    """選股結果"""

    positions: tuple[TrailingPosition, ...]
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
            std = (
                f"±{c.return_std * 100:.2f}%"
                if c.return_std is not None
                else "離散度未知"
            )
            lines.extend([
                f"{medals[i] if i < len(medals) else '  '} {c.stock_id}",
                f"   進場參考  {c.entry_price:,.0f}",
                f"   移動停損  −{c.trail_pct * 100:.2f}%"
                f"（隨期間最高價上移，只升不降）",
                f"   初始停損價 {c.initial_stop_price:,.0f}",
                "   分批進場  "
                + "、".join(
                    f"{t.price:,.0f}×{t.shares:,}股"
                    for t in c.entry_ladder(p.shares)
                ),
                f"   最長持有  {c.max_horizon} 個交易日"
                f"（約 {c.holding_months:.0f} 個月）",
                f"   期望毛報酬 {c.expected_gross_return * 100:+.2f}%  {std}",
                f"   期望淨報酬 {p.expected_net_return * 100:+.2f}%",
                f"   建議部位  約 {p.capital_pct * 100:.1f}% 資金"
                f"（零股 {p.shares:,} 股）",
                f"   ⓘ 零股交易，價差較整張寬，成本已含"
                f"{'0.3%' if c.tier is Tier.LARGE else '0.4%'} 滑價",
                "",
            ])

        lines.append(f"合計配置 {self.total_capital_pct * 100:.1f}% 資金")
        lines.append(
            "ⓘ 沒有目標價是刻意的：設定目標就走，強勢多頭裡會系統性輸給單純持有。"
        )
        lines.append(
            "ⓘ 分批進場為執行面建議，**回測未模擬分批**（標記層假設一次買足）。"
        )

        # 分箱校準只給有限個離散值。多檔落在同一箱時期望報酬完全相同，
        # 排名順序其實是任意的——不講清楚，讀者會以為 🥇 比 🥈 更有把握。
        expected_values = [p.candidate.expected_gross_return for p in self.positions]
        if len(set(expected_values)) < len(expected_values):
            lines.append(
                "ⓘ 部分標的期望報酬相同（落在校準器的同一個分數箱），"
                "它們之間的**排序為任意**，不代表把握度高低。"
            )
        if self.warnings:
            lines.append("")
            lines.extend(f"⚠️  {w}" for w in self.warnings)
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
# 選股
# ══════════════════════════════════════════════════════════════


def _prefilter(
    candidates: list[TrailingCandidate],
    capital: float,
    cost: CostModel,
    edge_z: float,
) -> tuple[list[TrailingCandidate], list[Rejection]]:
    """套用與投組無關的個別條件：期望淨報酬、統計顯著性、部位規模"""
    passed: list[TrailingCandidate] = []
    rejected: list[Rejection] = []

    for c in candidates:
        net = c.expected_net_return(cost)
        if net <= 0:
            rejected.append(Rejection(
                c.stock_id,
                RejectReason.BELOW_ENTRY_THRESHOLD,
                f"期望淨報酬 {net:+.2%} ≤ 0"
                f"（毛 {c.expected_gross_return:+.2%}、"
                f"成本 {cost.round_trip_rate(c.tier):.3%}）",
            ))
            continue

        if not c.has_significant_edge(cost, edge_z):
            se = c.standard_error
            detail = (
                f"優勢 {net:+.2%} < {edge_z:.1f} × 標準誤 {se:.2%}"
                f"（標準差 {c.return_std:.2%}、樣本 {c.n_samples}）"
                if se is not None
                else f"離散度未知（樣本 {c.n_samples}），無法判斷優勢是否顯著"
            )
            rejected.append(Rejection(
                c.stock_id, RejectReason.STATISTICALLY_INSIGNIFICANT, detail
            ))
            continue

        if position_shares(capital, c.entry_price, c.trail_pct) < 1:
            rejected.append(Rejection(
                c.stock_id,
                RejectReason.POSITION_TOO_SMALL,
                f"建議股數 < 1（單股 {c.entry_price:,.0f} 元，"
                f"資金上限 {capital * MAX_POSITION_PCT:,.0f} 元）",
            ))
            continue

        passed.append(c)

    return passed, rejected


def select_trailing_portfolio(
    candidates: list[TrailingCandidate],
    capital: float,
    correlations: dict[tuple[str, str], float],
    cost: CostModel = DEFAULT,
    edge_z: float = DEFAULT_EDGE_Z,
) -> TrailingPortfolioResult:
    """
    從候選中挑出最多 3 檔並計算部位規模。

    Args:
        candidates: 候選標的
        capital: 總資金
        correlations: {(stock_a, stock_b): 60 日報酬相關係數}，順序無關
        cost: 成本模型
        edge_z: 優勢須大於幾個標準誤

    Returns:
        TrailingPortfolioResult（含入選部位、剔除原因、警告）

    Raises:
        ValueError: capital 非正

    流程與 triple-barrier 版相同，只有第 1 步的過濾條件不同：
        1. 個別過濾（期望淨報酬 > 0、優勢 ≥ z 個標準誤、部位規模）
        2. 依期望淨報酬排序
        3. 貪婪挑選並套用投組約束
        4. 確保至少 1 支 defensive（必要時替換）
        5. 計算部位規模
    """
    if capital <= 0:
        raise ValueError(f"capital 必須為正，得到 {capital}")

    warnings: list[str] = []

    if not candidates:
        return TrailingPortfolioResult(
            positions=(),
            rejected=(),
            warnings=("本週無任何候選標的進入排名",),
        )

    passed, rejected = _prefilter(candidates, capital, cost, edge_z)

    if not passed:
        warnings.append("所有候選的期望淨報酬都不顯著為正，本週不推播")
        return TrailingPortfolioResult(
            positions=(), rejected=tuple(rejected), warnings=tuple(warnings)
        )

    # 依期望淨報酬排序；同值時用代號確保可重現（禁令 7、8）
    ordered = sorted(passed, key=lambda c: (-c.expected_net_return(cost), c.stock_id))

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

    picked.sort(key=lambda c: (-c.expected_net_return(cost), c.stock_id))

    positions: list[TrailingPosition] = []
    for c in picked:
        shares = position_shares(capital, c.entry_price, c.trail_pct)
        value = shares * c.entry_price
        positions.append(TrailingPosition(
            candidate=c,
            shares=shares,
            position_value=value,
            capital_pct=value / capital,
            expected_net_return=c.expected_net_return(cost),
        ))

    if len(positions) < TOP_N:
        warnings.append(
            f"僅選出 {len(positions)} 檔（目標 {TOP_N} 檔）："
            "候選不足以在滿足投組約束下湊滿。寧可少推也不違反風控"
        )

    return TrailingPortfolioResult(
        positions=tuple(positions),
        rejected=tuple(rejected),
        warnings=tuple(warnings),
    )
