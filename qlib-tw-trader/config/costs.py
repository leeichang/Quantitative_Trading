"""
台股交易成本模型 —— 單一來源

依據 docs/需求規劃/202609/01_決策紀錄.md 的 D3、D4。

**禁止在其他地方重寫費率。**所有回測、模擬、訊號產出都必須呼叫本模組。
理由：qlib-tw-trader 原本把成本寫在 scripts/evaluate_models.py，而
src/services/walk_forward_backtester.py 完全沒有成本，導致 API 與
Dashboard 顯示的是毛報酬。兩套數字不一致且沒人發現，正是散落費率的後果。

成本組成（台股現股做多，一趟來回）：

    買進   手續費 max(金額 × 0.1425% × 折扣, 20) + 金額 × 滑價
    賣出   手續費 max(金額 × 0.1425% × 折扣, 20) + 金額 × 0.3%（證交稅）
                                                 + 金額 × 滑價

    6 折、忽略最低手續費、滑價 0.3% 時，一趟來回約 0.771%

滑價分層的理由（D4）：總資金 40 萬買不起一張台積電（2,430 × 1,000 =
243 萬），必須做盤中零股。零股價差明顯比整張寬，因此滑價不能沿用
整張常見的 0.1%。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# ══════════════════════════════════════════════════════════════
# 費率常數
# ══════════════════════════════════════════════════════════════

FEE_RATE = 0.001425
"""券商手續費基本費率（法定上限 0.1425%）"""

FEE_DISCOUNT_DEFAULT = 0.6
"""預設折扣：網路下單 6 折。敏感度分析另跑 1.0（無折扣）"""

TAX_RATE = 0.003
"""證交稅 0.3%，僅賣出收取，不可折扣"""

MIN_FEE = 20.0
"""最低手續費 20 元。部位低於約 23,400 元（6 折）時會被這條咬住"""


class Tier(str, Enum):
    """流動性分層，決定滑價"""

    LARGE = "0050"
    """台灣 50 成分股，大型股零股"""

    MID = "0051"
    """中型 100 成分股，零股價差更寬"""


SLIPPAGE: dict[Tier, float] = {
    Tier.LARGE: 0.003,
    Tier.MID: 0.004,
}
"""零股滑價，依流動性分層（D3）"""


# ══════════════════════════════════════════════════════════════
# 成本計算
# ══════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class CostModel:
    """
    不可變的成本模型。

    建立不同折扣的實例來做敏感度分析：

        base = CostModel()                  # 6 折
        worst = CostModel(fee_discount=1.0) # 無折扣
    """

    fee_discount: float = FEE_DISCOUNT_DEFAULT
    apply_min_fee: bool = True
    apply_slippage: bool = True

    def __post_init__(self) -> None:
        if not 0.0 < self.fee_discount <= 1.0:
            raise ValueError(f"fee_discount 必須落在 (0, 1]，得到 {self.fee_discount}")

    # ---- 單邊成本 ----

    def commission(self, amount: float) -> float:
        """手續費（含最低收費）"""
        raw = abs(amount) * FEE_RATE * self.fee_discount
        return max(raw, MIN_FEE) if self.apply_min_fee else raw

    def slippage(self, amount: float, tier: Tier = Tier.LARGE) -> float:
        """滑價"""
        if not self.apply_slippage:
            return 0.0
        return abs(amount) * SLIPPAGE[tier]

    def buy_cost(self, amount: float, tier: Tier = Tier.LARGE) -> float:
        """買進單邊總成本（絕對金額）"""
        return self.commission(amount) + self.slippage(amount, tier)

    def sell_cost(self, amount: float, tier: Tier = Tier.LARGE) -> float:
        """賣出單邊總成本（絕對金額，含證交稅）"""
        return (
            self.commission(amount)
            + abs(amount) * TAX_RATE
            + self.slippage(amount, tier)
        )

    def trade_cost(
        self, amount: float, is_sell: bool, tier: Tier = Tier.LARGE
    ) -> float:
        """單筆交易成本。介面對齊 scripts/evaluate_models.py 的 calc_trade_cost"""
        return (
            self.sell_cost(amount, tier) if is_sell else self.buy_cost(amount, tier)
        )

    # ---- 比率形式（回測用） ----

    def round_trip_rate(self, tier: Tier = Tier.LARGE) -> float:
        """
        一趟來回的成本率（忽略最低手續費）。

        回測扣成本時用比率比用絕對金額方便，但要記得這個數字
        在小額部位會低估（最低手續費 20 元的影響）。
        """
        fee = FEE_RATE * self.fee_discount
        slip = SLIPPAGE[tier] if self.apply_slippage else 0.0
        return fee * 2 + TAX_RATE + slip * 2

    def one_way_rate(self, is_sell: bool, tier: Tier = Tier.LARGE) -> float:
        """單邊成本率（忽略最低手續費）"""
        fee = FEE_RATE * self.fee_discount
        slip = SLIPPAGE[tier] if self.apply_slippage else 0.0
        return fee + slip + (TAX_RATE if is_sell else 0.0)

    def min_amount_above_min_fee(self) -> float:
        """
        最低手續費不再生效的部位金額門檻。

        低於此金額的交易，實際費率高於名目費率。
        """
        return MIN_FEE / (FEE_RATE * self.fee_discount)


# ══════════════════════════════════════════════════════════════
# 預設實例與敏感度組合（D3：回測必須雙跑）
# ══════════════════════════════════════════════════════════════

DEFAULT = CostModel()
"""預設：6 折 + 最低手續費 + 滑價"""

NO_DISCOUNT = CostModel(fee_discount=1.0)
"""敏感度上界：無折扣"""

GROSS = CostModel(apply_min_fee=False, apply_slippage=False, fee_discount=1.0)
"""僅供對照：不含滑價、不含最低手續費（近似 qlib-tw-trader 原本 scripts/ 的模型）"""

SENSITIVITY_SET: dict[str, CostModel] = {
    "毛報酬（無成本）": CostModel(fee_discount=1.0),  # 呼叫端可選擇完全不扣
    "6折+滑價（預設）": DEFAULT,
    "無折扣+滑價": NO_DISCOUNT,
    "原專案模型（無滑價）": GROSS,
}


# ══════════════════════════════════════════════════════════════
# 換手率 → 年化成本拖累
# ══════════════════════════════════════════════════════════════


def annual_cost_drag(
    weekly_turnover: float,
    cost: CostModel = DEFAULT,
    tier: Tier = Tier.LARGE,
    weeks_per_year: int = 52,
) -> float:
    """
    由週換手率推算年化成本拖累。

    Args:
        weekly_turnover: 週換手率（0.099 表示每週換掉 9.9% 部位）
        cost: 成本模型
        tier: 流動性分層
        weeks_per_year: 一年週數

    Returns:
        年化成本拖累（0.031 表示每年吃掉 3.1% 報酬）

    換手率定義：每週被替換掉的部位比例（單邊）。一個部位被換掉
    需要「賣舊 + 買新」，因此年化來回次數 = 週換手率 × 週數。
    """
    annual_round_trips = weekly_turnover * weeks_per_year
    return annual_round_trips * cost.round_trip_rate(tier)


def sharpe_after_cost(
    gross_annual_return: float,
    gross_sharpe: float,
    weekly_turnover: float,
    cost: CostModel = DEFAULT,
    tier: Tier = Tier.LARGE,
) -> tuple[float, float, float]:
    """
    估算扣成本後的年化報酬與 Sharpe。

    Returns:
        (年化成本拖累, 淨年化報酬, 淨 Sharpe)

    假設：成本只降低報酬、不改變波動度。這是近似——實務上
    成本本身幾乎無波動，所以此假設偏樂觀但差距不大。
    """
    if gross_sharpe <= 0:
        raise ValueError("gross_sharpe 必須為正才能反推波動度")

    drag = annual_cost_drag(weekly_turnover, cost, tier)
    net_return = gross_annual_return - drag
    implied_vol = gross_annual_return / gross_sharpe
    net_sharpe = net_return / implied_vol
    return drag, net_return, net_sharpe
