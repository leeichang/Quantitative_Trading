"""
整股／零股與 ETF 的成本分層測試

## 為什麼要加這一層

`validate_oos_momentum.py` 第一版把滑價寫成 `SLIPPAGE_ONE_WAY = 0.001`
直接放在腳本裡——**那違反 CLAUDE.md 禁令 3**（一律呼叫 config/costs.py，
禁止他處重寫費率）。這個模組把那個數字搬回單一來源，並補上它缺的維度。

原本的 `Tier` 只有兩個成員，而且混了兩件事：

```
Tier.LARGE = "0050"   0050 成分股、**零股**、0.3%
Tier.MID   = "0051"   中型100成分股、**零股**、0.4%
```

零股假設來自 D4：40 萬買不起一張台積電（243 萬）。但實測 265 檔市值池
裡有 **153 檔（58%）**在 40 萬 × 33% 上限下買得起整張，整股滑價實證
只有 0.094%（跳動單位半價差中位數，490 筆實際持倉）。

## ETF 是第三類

實證 2024 年起的收盤價最小跳動：

```
0050 (ETF)   tick 0.0100   半價差 0.0037%
0056 (ETF)   tick 0.0100   半價差 0.0133%
2330 (股票)  tick 1.0000   半價差 0.0463%
1303 (股票)  tick 0.0500   半價差 0.0503%
```

**ETF 的跳動單位比股票細 4~10 倍。** 但 4 萬元買 0050 一張要 107,700，
所以實務上仍是零股——ETF 零股的價差比整股寬，取 0.1%（跳動下限的 25 倍）
是保守的。

## 不動禁令 4 的數字

`Tier.LARGE` 0.3% 與 `Tier.MID` 0.4% **原封不動**。那是 CLAUDE.md 禁令 4
明文規定的，只能新增成員，不能改既有的。
"""

from __future__ import annotations

import pytest

from taiwan_quant.config.costs import (
    DEFAULT,
    SLIPPAGE,
    Tier,
    lot_size,
    resolve_tier,
)


# ══════════════════════════════════════════════════════════════
# 既有分層不可變動（禁令 4）
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_odd_lot_constituent_slippage_is_unchanged() -> None:
    """
    CLAUDE.md 禁令 4：零股分層，0050 為 0.3%、0051 為 0.4%。

    這兩個數字是明文規格，只能新增成員，不可修改。
    """
    assert SLIPPAGE[Tier.LARGE] == 0.003
    assert SLIPPAGE[Tier.MID] == 0.004


@pytest.mark.unit
def test_existing_round_trip_rates_are_unchanged() -> None:
    """手算：6 折手續費 0.0855%×2 + 稅 0.3% + 滑價 0.3%×2 = 1.071%"""
    assert DEFAULT.round_trip_rate(Tier.LARGE) == pytest.approx(0.01071)
    assert DEFAULT.round_trip_rate(Tier.MID) == pytest.approx(0.01271)


# ══════════════════════════════════════════════════════════════
# 新增：整股與 ETF
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_whole_lot_slippage_comes_from_tick_measurement() -> None:
    """
    整股滑價 0.1%。依據：台股跳動單位隱含的半價差中位數 0.0937%
    （490 筆實際持倉），取 0.1% 略為保守。
    """
    assert SLIPPAGE[Tier.LARGE_WHOLE] == 0.001
    assert SLIPPAGE[Tier.MID_WHOLE] == 0.001


@pytest.mark.unit
def test_whole_lot_round_trip_matches_hand_calculation() -> None:
    """手算：0.0855%×2 + 0.3% + 0.1%×2 = 0.671%"""
    assert DEFAULT.round_trip_rate(Tier.LARGE_WHOLE) == pytest.approx(0.00671)


@pytest.mark.unit
def test_etf_slippage_is_tighter_than_stocks_but_not_free() -> None:
    """
    ETF 跳動單位比股票細 4~10 倍（實證 0050 tick 0.01、半價差 0.0037%），
    但零股價差仍比整股寬。取 0.1%（跳動下限的 25 倍）保守處理。

    整股 ETF 0.05%——仍是實證半價差的 13 倍。
    """
    assert SLIPPAGE[Tier.ETF_ODD] == 0.001
    assert SLIPPAGE[Tier.ETF_WHOLE] == 0.0005
    assert SLIPPAGE[Tier.ETF_WHOLE] < SLIPPAGE[Tier.ETF_ODD]
    assert SLIPPAGE[Tier.ETF_ODD] < SLIPPAGE[Tier.LARGE]


# ══════════════════════════════════════════════════════════════
# 整股／零股的判定
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_lot_size_is_one_thousand_shares() -> None:
    """台股一張 = 1,000 股"""
    assert lot_size() == 1000


@pytest.mark.unit
def test_resolve_tier_picks_whole_lot_when_affordable() -> None:
    """
    手算：股價 100 元 → 一張 100,000 元。

    部位 132,000 元（40萬 × 33%）→ 買得起整張 → 整股分層
    部位  40,000 元（40萬 / 10 檔）→ 買不起 → 零股分層
    """
    assert resolve_tier(
        actual_price=100.0, adjusted_price=80.0, amount=132_000
    ) is Tier.LARGE_WHOLE
    assert resolve_tier(
        actual_price=100.0, adjusted_price=80.0, amount=40_000
    ) is Tier.LARGE


@pytest.mark.unit
def test_resolve_tier_uses_actual_not_adjusted_price_for_affordability() -> None:
    """
    工作單 J 的手算案例：實際價 42 元，一張需 42,000；40,000 買不起。

    還原價 38 元若被誤用會判成買得起整張，因此結果必須是零股分層。
    """
    assert resolve_tier(
        actual_price=42.0,
        adjusted_price=38.0,
        amount=40_000,
    ) is Tier.LARGE


@pytest.mark.unit
def test_resolve_tier_rejects_missing_actual_price() -> None:
    """缺實際價不可靜默退回還原價。"""
    with pytest.raises(ValueError, match="actual_price"):
        resolve_tier(
            actual_price=None,
            adjusted_price=38.0,
            amount=40_000,
        )


@pytest.mark.unit
def test_resolve_tier_boundary_is_exactly_one_lot() -> None:
    """
    手算：股價 40 元 → 一張 40,000 元。

    部位剛好 40,000 → 買得起整張（含）
    部位 39,999 → 買不起
    """
    assert resolve_tier(
        actual_price=40.0, adjusted_price=35.0, amount=40_000
    ) is Tier.LARGE_WHOLE
    assert resolve_tier(
        actual_price=40.0, adjusted_price=35.0, amount=39_999
    ) is Tier.LARGE


@pytest.mark.unit
def test_resolve_tier_honours_the_mid_tier_request() -> None:
    """中型股的零股滑價是 0.4%，不可誤用 0.3%"""
    assert resolve_tier(
        actual_price=100.0, adjusted_price=80.0, amount=40_000, large=False
    ) is Tier.MID
    assert resolve_tier(
        actual_price=100.0, adjusted_price=80.0, amount=132_000, large=False
    ) is Tier.MID_WHOLE


@pytest.mark.unit
def test_resolve_tier_routes_etfs_separately() -> None:
    """
    手算：0050 約 107.70 元 → 一張 107,700 元。

    部位 40,000 → ETF 零股
    部位 132,000 → ETF 整股
    """
    assert resolve_tier(
        actual_price=107.70, adjusted_price=100.0,
        amount=40_000, is_etf=True,
    ) is Tier.ETF_ODD
    assert resolve_tier(
        actual_price=107.70, adjusted_price=100.0,
        amount=132_000, is_etf=True,
    ) is Tier.ETF_WHOLE


@pytest.mark.unit
def test_resolve_tier_rejects_invalid_inputs() -> None:
    """系統邊界要驗證輸入，不可靜默回一個分層"""
    with pytest.raises(ValueError, match="必須為正"):
        resolve_tier(actual_price=0.0, adjusted_price=1.0, amount=40_000)
    with pytest.raises(ValueError, match="必須為正"):
        resolve_tier(actual_price=100.0, adjusted_price=80.0, amount=0)


# ══════════════════════════════════════════════════════════════
# 這一層真正要防的事
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_every_tier_has_a_slippage_rate() -> None:
    """
    新增分層卻忘記填滑價時，`round_trip_rate` 會 KeyError 而不是
    靜默用 0——但更好的是這個測試在加成員時就擋下來。
    """
    for tier in Tier:
        assert tier in SLIPPAGE, f"{tier} 沒有對應的滑價"
        assert 0 < SLIPPAGE[tier] < 0.05


@pytest.mark.unit
def test_whole_lot_is_always_cheaper_than_odd_lot() -> None:
    """整股不可能比零股貴。這條錯了代表分層設定反了。"""
    assert SLIPPAGE[Tier.LARGE_WHOLE] < SLIPPAGE[Tier.LARGE]
    assert SLIPPAGE[Tier.MID_WHOLE] < SLIPPAGE[Tier.MID]
    assert SLIPPAGE[Tier.ETF_WHOLE] < SLIPPAGE[Tier.ETF_ODD]
