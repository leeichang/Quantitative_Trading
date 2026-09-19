"""
依跳動單位與部位規模估滑價，取代兩段式的整股／零股分層

## 為什麼要換

`config/costs.py` 的 `SLIPPAGE` 是兩段式：

```
零股   LARGE 0.3% / MID 0.4%   ← CLAUDE.md 禁令 4，明文規格
整股   0.1%                    ← 實證：490 筆實際持倉，半價差中位數 0.0937%
```

**整股那個有量測支撐，零股那個沒有。** 而它是總成本裡最大的一項：
4 萬元零股單的來回成本 1.071% 裡，滑價佔 0.600%，手續費 0.171%、
證交稅 0.300%。

2026-09-19 實測跳動單位隱含的半價差，發現 0.3% 對每一個價位都過高：

```
股價        跳動單位   半價差   vs 規格 0.3%
  46.30      0.05    0.054%     5.6 倍
 234.50      0.50    0.107%     2.8 倍
2410.00      5.00    0.104%     2.9 倍
5765.00      5.00    0.043%     6.9 倍
```

而結論完全靠這個數字：同一個訊號在 0.20% 滑價下區間不含零，
在 0.30% 下含零。**0.10 pp 決定「有沒有東西」。**

## 兩段式錯在哪

它把 2330（一張 241 萬）和 2884（一張 4.6 萬）當同一類。40 萬資金下
2330 永遠是零股——**那與資金無關，是股價的問題**。把「買不起一張」
和「流動性差」混為一談，等於用價格高低代理流動性。

## 新模型

```
滑價 = 半價差 × 零股倍數 + 市場衝擊
```

- **半價差** `tick_size(price) / 2 / price`，完全由跳動單位決定，可算不必假設
- **零股倍數** 盤中零股有獨立委託簿，比整股簿薄，價差寬於跳動單位。
  **這是唯一沒有量測的參數**，所以它是顯式輸入而不是內建常數
- **市場衝擊** 部位佔當日成交額的比例。既有註解實測「4 萬元部位對當日
  成交額中位僅 98.9 ppm」，所以 40 萬規模下近 0——但仍然算，
  因為冷門股會不同

## 這個模組不取代 costs.py

`costs.SLIPPAGE` 保持原值不動（禁令 4 是明文規格，而且舊報告要能重現）。
本模組是**並列的第二個估計**，用途是量化「結論對那個未量測的參數有多敏感」。

要把它接進正式成本模型，需要先有零股倍數的實測依據——
TWSE 每日行情有「最後揭示買價／賣價」，那是可量測的，但尚未回補入庫。
"""

from __future__ import annotations

from dataclasses import dataclass

from taiwan_quant.labeling.limit_up import tick_size

ODD_LOT_MULTIPLIER_DEFAULT = 2.0
"""
零股價差相對跳動單位的倍數。**沒有實證依據，所以是顯式參數。**

取 2.0 不是量測結果，是一個明示的保守佔位值：整股的實證半價差
（0.0937%）與跳動單位隱含值幾乎相同，代表整股倍數約 1.0；零股簿更薄
所以應大於 1.0，但大多少沒人量過。

⚠️ **任何用到它的結論都要一起報敏感度**（例如 1.0 / 2.0 / 3.0 三檔），
不可以只報一個值。
"""

IMPACT_COEFFICIENT = 0.1
"""
市場衝擊係數：`impact = coefficient × (部位金額 / 當日成交額)`。

線性近似，且係數也沒有實證依據。之所以仍然保留這一項，是因為它在
冷門股上不可忽略——40 萬買一檔日成交額 200 萬的股票是 20%，
而不是大型股的 98.9 ppm。

⚠️ 同樣要報敏感度。
"""


class SlippageError(ValueError):
    """輸入不合法。"""


@dataclass(frozen=True)
class SlippageEstimate:
    """單邊滑價的分解，讓每一項的來源可追。"""

    half_spread: float
    """跳動單位隱含的半價差。**可算，不是假設**"""

    odd_lot_premium: float
    """零股簿較薄的加成。**未量測**"""

    market_impact: float
    """部位相對當日成交額。線性近似"""

    @property
    def total(self) -> float:
        return self.half_spread + self.odd_lot_premium + self.market_impact

    def describe(self) -> str:
        return (
            f"半價差 {self.half_spread:.4%}"
            f" + 零股加成 {self.odd_lot_premium:.4%}"
            f" + 衝擊 {self.market_impact:.4%}"
            f" = {self.total:.4%}"
        )


def half_spread(price: float) -> float:
    """
    跳動單位隱含的半價差。

    這是價差的**下限**：實際委託簿可能更寬，但不會比一個跳動單位更窄。

    Raises:
        SlippageError: 價格非正
    """
    if not price > 0:
        raise SlippageError(f"價格必須為正，得到 {price}")
    return (tick_size(price) / 2.0) / price


def market_impact(
    amount: float,
    daily_turnover: float,
    *,
    coefficient: float = IMPACT_COEFFICIENT,
) -> float:
    """
    部位相對當日成交額的線性衝擊。

    `daily_turnover <= 0`（當天無成交）時回 `inf`——**不是 0**。
    買不到的東西不該被估成零成本，呼叫端要自己決定跳過。

    Raises:
        SlippageError: 金額為負或係數為負
    """
    if amount < 0:
        raise SlippageError(f"部位金額不可為負，得到 {amount}")
    if coefficient < 0:
        raise SlippageError(f"衝擊係數不可為負，得到 {coefficient}")
    if daily_turnover <= 0:
        return float("inf")
    return coefficient * (amount / daily_turnover)


def estimate(
    *,
    price: float,
    amount: float,
    daily_turnover: float,
    whole_lots: bool,
    odd_lot_multiplier: float = ODD_LOT_MULTIPLIER_DEFAULT,
    impact_coefficient: float = IMPACT_COEFFICIENT,
) -> SlippageEstimate:
    """
    單邊滑價估計。

    Args:
        price: 成交價（未還原——跳動單位看的是實際報價）
        amount: 部位金額
        daily_turnover: 當日成交額（金額，不是股數）
        whole_lots: 是否買得起整張。`True` 時零股加成為 0
        odd_lot_multiplier: 見 `ODD_LOT_MULTIPLIER_DEFAULT`。**要報敏感度**
        impact_coefficient: 見 `IMPACT_COEFFICIENT`

    Raises:
        SlippageError: 倍數小於 1（零股不可能比整股便宜）
    """
    if odd_lot_multiplier < 1.0:
        raise SlippageError(
            f"零股倍數不可小於 1——零股簿比整股薄，不會更便宜，"
            f"得到 {odd_lot_multiplier}"
        )
    base = half_spread(price)
    premium = 0.0 if whole_lots else base * (odd_lot_multiplier - 1.0)
    return SlippageEstimate(
        half_spread=base,
        odd_lot_premium=premium,
        market_impact=market_impact(
            amount, daily_turnover, coefficient=impact_coefficient
        ),
    )


def affordable_whole_lots(price: float, amount: float, lot_size: int = 1000) -> bool:
    """
    這筆金額買不買得起一整張。

    **這是算術，不是流動性判斷。** 2330 一張 241 萬，40 萬永遠買不起——
    那與 2330 的流動性無關（它是台股最大最活絡的標的）。
    兩段式滑價把這兩件事混在一起，這個函式把它們分開。
    """
    if not price > 0:
        raise SlippageError(f"價格必須為正，得到 {price}")
    return amount >= price * lot_size
