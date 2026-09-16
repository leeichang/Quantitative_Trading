"""
台股交易成本模型 —— 單一來源

依據 ../docs/需求規劃/202609/01_決策紀錄.md 的 D3、D4。

**禁止在其他地方重寫費率（CLAUDE.md 禁令 3）。**所有回測、模擬、訊號產出
都必須呼叫本模組。

理由（實測教訓）：qlib-tw-trader 把成本寫在 `scripts/evaluate_models.py`，
而 `src/services/walk_forward_backtester.py` 完全沒有成本，導致 API 與
Dashboard 顯示毛報酬、離線腳本顯示淨報酬，兩套數字不一致且長期無人察覺。

成本組成（台股現股做多，一趟來回）：

    買進   max(金額 × 0.1425% × 折扣, 20) + 金額 × 滑價
    賣出   max(金額 × 0.1425% × 折扣, 20) + 金額 × 0.3%（證交稅）
                                          + 金額 × 滑價

    6 折、忽略最低手續費、滑價 0.3% 時，一趟來回 = 1.071%

為什麼滑價要分層且不能用整張常見的 0.1%（D4）：
總資金 40 萬買不起一張台積電（2,430 × 1,000 = 243 萬），必須做盤中零股。
零股價差明顯比整張寬。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# ══════════════════════════════════════════════════════════════
# 費率常數
# ══════════════════════════════════════════════════════════════

FEE_RATE: float = 0.001425
"""券商手續費基本費率（法定上限 0.1425%）"""

FEE_DISCOUNT_DEFAULT: float = 0.6
"""預設折扣：網路下單 6 折。敏感度分析另跑 1.0（無折扣）"""

TAX_RATE: float = 0.003
"""證交稅 0.3%，僅賣出收取，不可折扣"""

MIN_FEE: float = 20.0
"""最低手續費 20 元。部位低於約 23,392 元（6 折）時會被這條咬住"""


class Tier(str, Enum):
    """
    流動性分層，決定滑價。

    兩個維度：**標的類別**（0050 成分股 / 中型 100 / ETF）× **交易單位**
    （零股 / 整股）。原本只有前兩個成員，都隱含「零股」。
    """

    LARGE = "0050"
    """台灣 50 成分股，大型股**零股**（CLAUDE.md 禁令 4：0.3%）"""

    MID = "0051"
    """中型 100 成分股，**零股**價差更寬（CLAUDE.md 禁令 4：0.4%）"""

    LARGE_WHOLE = "0050-lot"
    """台灣 50 成分股，**整股**"""

    MID_WHOLE = "0051-lot"
    """中型 100 成分股，**整股**"""

    ETF_ODD = "etf-odd"
    """ETF 本身（非成分股），**零股**"""

    ETF_WHOLE = "etf-lot"
    """ETF 本身，**整股**"""


SLIPPAGE: dict[Tier, float] = {
    # 零股，D3/D4 明文規定，不可修改
    Tier.LARGE: 0.003,
    Tier.MID: 0.004,
    # 整股。實證依據：台股跳動單位隱含的半價差中位數 0.0937%
    # （490 筆實際持倉，股價 0~50 元 0.0973%、50~100 元 0.0708%、
    # 100~500 元 0.1411%），且 4 萬元部位對當日成交額中位僅 98.9 ppm，
    # 市場衝擊近 0。取 0.1% 略為保守。
    Tier.LARGE_WHOLE: 0.001,
    Tier.MID_WHOLE: 0.001,
    # ETF。實證 2024 年起收盤價最小跳動：0050 為 0.01（半價差 0.0037%）、
    # 0056 為 0.01（0.0133%），比股票（2330 tick 1.0、1303 tick 0.05）
    # 細 4~10 倍。零股取 0.1%（跳動下限的 25 倍）、整股 0.05%（13 倍），
    # 都遠高於跳動下限，是保守處理。
    Tier.ETF_ODD: 0.001,
    Tier.ETF_WHOLE: 0.0005,
}
"""
滑價，依標的類別與交易單位分層。

⚠️ `LARGE` 與 `MID`（零股）的數字來自 CLAUDE.md 禁令 4，是明文規格，
**只能新增成員，不可修改既有值**。
"""

LOT_SIZE: int = 1000
"""台股一張 = 1,000 股"""


def lot_size() -> int:
    """一張的股數。包成函式是為了讓呼叫端不要各自寫死 1000"""
    return LOT_SIZE


def resolve_tier(
    price: float, amount: float, *, large: bool = True, is_etf: bool = False
) -> Tier:
    """
    依股價與部位金額決定分層。

    Args:
        price: 每股價格
        amount: 這一檔要投入的金額
        large: 是否為 0050 成分股（`False` 視為中型 100）。ETF 時忽略
        is_etf: 標的本身是否為 ETF（不是「ETF 的成分股」）

    Returns:
        對應的 `Tier`

    Raises:
        ValueError: `price` 或 `amount` 非正

    **買得起一整張就用整股滑價，否則用零股。** 這是 D4 零股假設缺的維度：
    40 萬買不起一張台積電是真的，但 265 檔市值池裡有 153 檔（58%）在
    40 萬 × 33% 的上限下買得起整張。
    """
    if price <= 0:
        raise ValueError(f"price 必須為正，得到 {price}")
    if amount <= 0:
        raise ValueError(f"amount 必須為正，得到 {amount}")

    whole = amount >= price * LOT_SIZE
    if is_etf:
        return Tier.ETF_WHOLE if whole else Tier.ETF_ODD
    if large:
        return Tier.LARGE_WHOLE if whole else Tier.LARGE
    return Tier.MID_WHOLE if whole else Tier.MID


# ══════════════════════════════════════════════════════════════
# 成本模型
# ══════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class CostModel:
    """
    不可變的成本模型。

    建立不同折扣的實例來做敏感度分析（CLAUDE.md 規格 15 要求至少三檔並列）：

        gross = CostModel(apply_fee=False, apply_tax=False, apply_slippage=False)
        base  = CostModel()                    # 6 折 + 滑價
        worst = CostModel(fee_discount=1.0)    # 無折扣 + 滑價
    """

    fee_discount: float = FEE_DISCOUNT_DEFAULT
    apply_fee: bool = True
    apply_tax: bool = True
    apply_min_fee: bool = True
    apply_slippage: bool = True

    def __post_init__(self) -> None:
        if not 0.0 < self.fee_discount <= 1.0:
            raise ValueError(f"fee_discount 必須落在 (0, 1]，得到 {self.fee_discount}")

    # ---- 單邊絕對金額 ----

    def commission(self, amount: float) -> float:
        """手續費（含最低收費）"""
        if not self.apply_fee:
            return 0.0
        raw = abs(amount) * FEE_RATE * self.fee_discount
        return max(raw, MIN_FEE) if self.apply_min_fee else raw

    def tax(self, amount: float) -> float:
        """證交稅（呼叫端負責只在賣出時使用）"""
        return abs(amount) * TAX_RATE if self.apply_tax else 0.0

    def slippage(self, amount: float, tier: Tier = Tier.LARGE) -> float:
        """滑價"""
        return abs(amount) * SLIPPAGE[tier] if self.apply_slippage else 0.0

    def buy_cost(self, amount: float, tier: Tier = Tier.LARGE) -> float:
        """買進單邊總成本（絕對金額，不含證交稅）"""
        return self.commission(amount) + self.slippage(amount, tier)

    def sell_cost(self, amount: float, tier: Tier = Tier.LARGE) -> float:
        """賣出單邊總成本（絕對金額，含證交稅）"""
        return self.commission(amount) + self.tax(amount) + self.slippage(amount, tier)

    def round_trip_cost(self, amount: float, tier: Tier = Tier.LARGE) -> float:
        """一趟來回總成本（絕對金額）"""
        return self.buy_cost(amount, tier) + self.sell_cost(amount, tier)

    # ---- 比率形式（回測用） ----

    def one_way_rate(self, is_sell: bool, tier: Tier = Tier.LARGE) -> float:
        """
        單邊成本率（**忽略最低手續費**）。

        注意：小額部位用比率會低估成本。部位金額已知時請用
        `buy_cost` / `sell_cost` 的絕對金額版本。
        """
        fee = FEE_RATE * self.fee_discount if self.apply_fee else 0.0
        slip = SLIPPAGE[tier] if self.apply_slippage else 0.0
        tax = TAX_RATE if (is_sell and self.apply_tax) else 0.0
        return fee + slip + tax

    def round_trip_rate(self, tier: Tier = Tier.LARGE) -> float:
        """一趟來回成本率（忽略最低手續費）"""
        return self.one_way_rate(False, tier) + self.one_way_rate(True, tier)

    def min_amount_above_min_fee(self) -> float:
        """
        最低手續費不再生效的部位金額門檻。

        低於此金額的交易，實際費率高於名目費率。
        以 40 萬資金做零股三檔，這種小額單會常出現。
        """
        if not self.apply_fee or not self.apply_min_fee:
            return 0.0
        return MIN_FEE / (FEE_RATE * self.fee_discount)


# ══════════════════════════════════════════════════════════════
# 預設實例與敏感度組合（CLAUDE.md 規格 15）
# ══════════════════════════════════════════════════════════════

GROSS = CostModel(apply_fee=False, apply_tax=False, apply_slippage=False)
"""毛報酬：完全不扣成本。只作為對照，不可當結論"""

FEE_TAX_ONLY = CostModel(apply_slippage=False)
"""只含手續費與證交稅、不含滑價。用來重現 qlib-tw-trader 的成本模型"""

DEFAULT = CostModel()
"""預設：6 折 + 最低手續費 + 滑價"""

NO_DISCOUNT = CostModel(fee_discount=1.0)
"""敏感度上界：無折扣 + 滑價"""

SENSITIVITY_SET: dict[str, CostModel] = {
    "無成本（毛報酬）": GROSS,
    "僅手續費+證交稅（無滑價）": FEE_TAX_ONLY,
    "6折+滑價（預設）": DEFAULT,
    "無折扣+滑價": NO_DISCOUNT,
}
"""回測報告必須並列的成本情境（規格 15）"""


# ══════════════════════════════════════════════════════════════
# 換手率 → 年化成本拖累（CLAUDE.md 規格 13）
# ══════════════════════════════════════════════════════════════

WEEKS_PER_YEAR: int = 52


def annual_cost_drag(
    weekly_turnover: float,
    cost: CostModel = DEFAULT,
    tier: Tier = Tier.LARGE,
    weeks_per_year: int = WEEKS_PER_YEAR,
) -> float:
    """
    由週換手率推算年化成本拖累。

    換手率定義：每週被替換掉的部位比例（單邊）。一個部位被換掉需要
    「賣舊 + 買新」，因此年化來回次數 = 週換手率 × 週數。

    Args:
        weekly_turnover: 週換手率（0.099 表示每週換掉 9.9% 部位）
        cost: 成本模型
        tier: 流動性分層
        weeks_per_year: 一年週數

    Returns:
        年化成本拖累（0.0551 表示每年吃掉 5.51% 報酬）

    實測參考（qlib-tw-trader 驗證）：
        週換手 9.9%（HoldDrop）   → 年化拖累   5.51%
        週換手 271.5%（日度調倉） → 年化拖累 115.56%
    """
    if weekly_turnover < 0:
        raise ValueError(f"weekly_turnover 不可為負，得到 {weekly_turnover}")
    return weekly_turnover * weeks_per_year * cost.round_trip_rate(tier)


def sharpe_after_cost(
    gross_annual_return: float,
    gross_sharpe: float,
    weekly_turnover: float,
    cost: CostModel = DEFAULT,
    tier: Tier = Tier.LARGE,
) -> tuple[float, float, float]:
    """
    解析估算扣成本後的年化報酬與 Sharpe。

    假設成本只降低報酬、不改變波動度。這是近似——成本本身幾乎無波動，
    所以偏樂觀但差距不大。**有實際回測資料時一律用實測，不要用這個。**

    Returns:
        (年化成本拖累, 淨年化報酬, 淨 Sharpe)
    """
    if gross_sharpe <= 0:
        raise ValueError("gross_sharpe 必須為正才能反推波動度")

    drag = annual_cost_drag(weekly_turnover, cost, tier)
    net_return = gross_annual_return - drag
    implied_vol = gross_annual_return / gross_sharpe
    return drag, net_return, net_return / implied_vol
