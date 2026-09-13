"""
移動停損選股測試（路線 A）

與 triple-barrier 版的差別只在**個別過濾條件**：

    triple-barrier   P(+1) ≥ (stop + cost) / (target + stop)
    移動停損         E[毛報酬] − cost > 0，且優勢大於一個標準誤

投組約束（產業／波動／相關／defensive）與部位規模完全共用。
"""

from __future__ import annotations

import math

import pytest

from taiwan_quant.config.costs import DEFAULT, Tier
from taiwan_quant.ranking.constraints import MAX_POSITION_PCT, RejectReason
from taiwan_quant.ranking.trailing_portfolio import (
    TrailingCandidate,
    select_trailing_portfolio,
)

CAPITAL = 400_000.0
ROUND_TRIP = DEFAULT.round_trip_rate(Tier.LARGE)


def make_candidate(
    stock_id: str,
    expected_gross_return: float = 0.18,
    return_std: float = 0.20,
    n_samples: int = 400,
    trail_pct: float = 0.12,
    entry_price: float = 200.0,
    tier: Tier = Tier.LARGE,
    industry: str = "半導體",
    volatility_pct: float = 0.50,
    beta: float = 0.9,
    max_horizon: int = 60,
) -> TrailingCandidate:
    return TrailingCandidate(
        stock_id=stock_id,
        expected_gross_return=expected_gross_return,
        return_std=return_std,
        n_samples=n_samples,
        trail_pct=trail_pct,
        entry_price=entry_price,
        tier=tier,
        industry=industry,
        volatility_pct=volatility_pct,
        beta=beta,
        max_horizon=max_horizon,
    )


def no_correlation(ids: list[str]) -> dict[tuple[str, str], float]:
    """所有配對都不相關，讓測試專注在單一變因"""
    return {(a, b): 0.0 for a in ids for b in ids if a != b}


# ══════════════════════════════════════════════════════════════
# 期望報酬與統計顯著性
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_expected_net_return_deducts_cost() -> None:
    """手算：0.18 − 1.071% = 0.16929"""
    c = make_candidate("2330", expected_gross_return=0.18)
    assert c.expected_net_return(DEFAULT) == pytest.approx(0.18 - ROUND_TRIP)


@pytest.mark.unit
def test_standard_error_matches_hand_calculation() -> None:
    """標準誤 = 標準差 / sqrt(n) = 0.20 / sqrt(400) = 0.01"""
    c = make_candidate("2330", return_std=0.20, n_samples=400)
    assert c.standard_error == pytest.approx(0.01, abs=1e-12)


@pytest.mark.unit
def test_standard_error_is_none_when_sample_too_small() -> None:
    """樣本 < 2 無法算離散度 → None（不是 0）"""
    assert make_candidate("2330", n_samples=1).standard_error is None
    assert make_candidate("2330", return_std=None).standard_error is None


# ══════════════════════════════════════════════════════════════
# 進場門檻
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_rejects_when_expected_return_below_cost() -> None:
    """期望毛報酬賺不回一趟來回成本 → 剔除"""
    below = make_candidate("2330", expected_gross_return=ROUND_TRIP * 0.5)
    result = select_trailing_portfolio([below], CAPITAL, no_correlation(["2330"]))

    assert result.positions == ()
    assert result.rejected[0].reason is RejectReason.BELOW_ENTRY_THRESHOLD
    assert "成本" in result.rejected[0].detail


@pytest.mark.unit
def test_rejects_when_edge_is_within_one_standard_error() -> None:
    """
    期望淨報酬為正，但小於一個標準誤 → 剔除。

    構造：毛報酬 2%、成本 1.071% → 淨 0.929%
          標準差 40%、樣本 100 → 標準誤 4%

    0.929% < 4%：這個「優勢」與 0 在統計上分不開。
    OOS 報告已經指出樣本極少會讓結論不穩，這道門檻就是對症下藥。
    """
    noisy = make_candidate(
        "2330", expected_gross_return=0.02, return_std=0.40, n_samples=100
    )
    result = select_trailing_portfolio([noisy], CAPITAL, no_correlation(["2330"]))

    assert result.positions == ()
    assert result.rejected[0].reason is RejectReason.STATISTICALLY_INSIGNIFICANT


@pytest.mark.unit
def test_accepts_when_edge_clears_standard_error() -> None:
    """毛 18%、成本 1.071% → 淨 16.9%；標準誤 1% → 通過"""
    solid = make_candidate("2330", expected_gross_return=0.18)
    result = select_trailing_portfolio([solid], CAPITAL, no_correlation(["2330"]))

    assert len(result.positions) == 1
    assert result.positions[0].candidate.stock_id == "2330"


@pytest.mark.unit
def test_rejects_when_dispersion_unknown() -> None:
    """
    沒有離散度就無法判斷優勢是否顯著 → 保守拒絕。

    當成 0 等於假設「這個估計完全沒有誤差」，那是最危險的假設。
    """
    unknown = make_candidate("2330", return_std=None, n_samples=1)
    result = select_trailing_portfolio([unknown], CAPITAL, no_correlation(["2330"]))

    assert result.positions == ()
    assert result.rejected[0].reason is RejectReason.STATISTICALLY_INSIGNIFICANT


@pytest.mark.unit
def test_no_fixed_target_means_no_risk_reward_filter() -> None:
    """
    移動停損沒有目標價，所以沒有 R:R 可算。

    不得偷偷拿「期望報酬 / trail_pct」硬充 R:R——那是兩個不同的量
    （期望值 vs 最好情況），混用會讓門檻的意義說不清楚。

    構造：期望報酬 3%、停損 12%，「R:R」只有 0.25，遠低於 2.0，
    但優勢顯著（標準誤 0.5%），應該通過。
    """
    low_ratio = make_candidate(
        "2330", expected_gross_return=0.03, return_std=0.10, n_samples=400,
        trail_pct=0.12,
    )
    result = select_trailing_portfolio([low_ratio], CAPITAL, no_correlation(["2330"]))

    assert len(result.positions) == 1


# ══════════════════════════════════════════════════════════════
# 投組約束（與 triple-barrier 版共用）
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_industry_limit_still_applies() -> None:
    ids = ["A", "B", "C"]
    cands = [
        make_candidate("A", expected_gross_return=0.30, industry="半導體"),
        make_candidate("B", expected_gross_return=0.25, industry="半導體"),
        make_candidate("C", expected_gross_return=0.20, industry="半導體"),
    ]
    result = select_trailing_portfolio(cands, CAPITAL, no_correlation(ids))

    assert len(result.positions) == 2
    assert any(r.reason is RejectReason.INDUSTRY_LIMIT for r in result.rejected)


@pytest.mark.unit
def test_correlation_limit_still_applies() -> None:
    cands = [
        make_candidate("A", expected_gross_return=0.30, industry="半導體"),
        make_candidate("B", expected_gross_return=0.25, industry="金融"),
    ]
    result = select_trailing_portfolio(
        cands, CAPITAL, {("A", "B"): 0.85}
    )

    assert len(result.positions) == 1
    assert result.rejected[0].reason is RejectReason.CORRELATION_LIMIT


@pytest.mark.unit
def test_warns_when_no_defensive() -> None:
    ids = ["A", "B"]
    cands = [
        make_candidate("A", expected_gross_return=0.30, industry="半導體", beta=1.4),
        make_candidate("B", expected_gross_return=0.25, industry="金融", beta=1.5),
    ]
    result = select_trailing_portfolio(cands, CAPITAL, no_correlation(ids))

    assert any("defensive" in w for w in result.warnings)


@pytest.mark.unit
def test_position_uses_trail_pct_as_stop() -> None:
    """
    部位規模的風險倒推用 trail_pct（初始停損距離）。

    手算：min(400000 × 1% / (200 × 0.12), 400000 × 33% / 200)
        = min(4000 / 24, 660) = min(166.67, 660) = 166 股
    """
    c = make_candidate("2330", trail_pct=0.12, entry_price=200.0)
    result = select_trailing_portfolio([c], CAPITAL, no_correlation(["2330"]))

    assert result.positions[0].shares == 166


@pytest.mark.unit
def test_rejects_when_position_below_one_share() -> None:
    pricey = make_candidate("2330", entry_price=CAPITAL * MAX_POSITION_PCT + 1)
    result = select_trailing_portfolio([pricey], CAPITAL, no_correlation(["2330"]))

    assert result.positions == ()
    assert result.rejected[0].reason is RejectReason.POSITION_TOO_SMALL


# ══════════════════════════════════════════════════════════════
# 輸出格式（使用者可見）
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_describe_has_no_target_price() -> None:
    """
    路線 A 的核心：**沒有目標價**。

    輸出裡若還出現「目標價」，代表上檔又被封死了——那正是 OOS
    輸給買進持有的機制原因。
    """
    c = make_candidate("2330")
    text = select_trailing_portfolio([c], CAPITAL, no_correlation(["2330"])).describe()

    # 守的是「沒有目標價欄位」，不是「沒有這三個字」——
    # 結尾的說明段落本來就要解釋為什麼刻意不給目標價。
    labels = [line.split()[0] for line in text.splitlines() if line.startswith("   ")]
    assert "目標價" not in labels
    assert "失效價" not in labels

    assert "移動停損" in labels
    assert "只升不降" in text


@pytest.mark.unit
def test_describe_states_holding_period_in_months() -> None:
    """使用者要的是「下週買進、持續持有 3 個月」"""
    c = make_candidate("2330", max_horizon=60)
    text = select_trailing_portfolio([c], CAPITAL, no_correlation(["2330"])).describe()

    assert "60 個交易日" in text
    assert "約 3 個月" in text


@pytest.mark.unit
def test_describe_shows_dispersion_not_just_mean() -> None:
    """平均 +18% 標準差 20%，與平均 +18% 標準差 2%，意義完全不同"""
    c = make_candidate("2330", expected_gross_return=0.18, return_std=0.20)
    text = select_trailing_portfolio([c], CAPITAL, no_correlation(["2330"])).describe()

    assert "18.00%" in text
    assert "20.00%" in text


@pytest.mark.unit
def test_describe_shows_initial_stop_price() -> None:
    """初始停損價 = 200 × (1 − 0.12) = 176"""
    c = make_candidate("2330", entry_price=200.0, trail_pct=0.12)
    text = select_trailing_portfolio([c], CAPITAL, no_correlation(["2330"])).describe()

    assert "176" in text


@pytest.mark.unit
def test_describe_when_nothing_qualifies() -> None:
    weak = make_candidate("2330", expected_gross_return=0.0)
    text = select_trailing_portfolio([weak], CAPITAL, no_correlation(["2330"])).describe()

    assert "無符合條件" in text


# ══════════════════════════════════════════════════════════════
# 不可變性與輸入驗證
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_candidate_is_immutable() -> None:
    c = make_candidate("2330")
    with pytest.raises(Exception):
        c.trail_pct = 0.5  # type: ignore[misc]


@pytest.mark.unit
def test_candidate_rejects_invalid_trail_pct() -> None:
    with pytest.raises(ValueError, match="trail_pct"):
        make_candidate("2330", trail_pct=0.0)
    with pytest.raises(ValueError, match="trail_pct"):
        make_candidate("2330", trail_pct=1.0)


@pytest.mark.unit
def test_candidate_rejects_non_finite_expected_return() -> None:
    with pytest.raises(ValueError, match="有限值"):
        make_candidate("2330", expected_gross_return=math.nan)


@pytest.mark.unit
def test_rejects_non_positive_capital() -> None:
    with pytest.raises(ValueError, match="capital"):
        select_trailing_portfolio([make_candidate("2330")], 0.0, {})


@pytest.mark.unit
def test_empty_candidates_returns_warning() -> None:
    result = select_trailing_portfolio([], CAPITAL, {})
    assert result.positions == ()
    assert result.warnings


# ══════════════════════════════════════════════════════════════
# 拉回分批買（使用者要求的「拉回分批買 + 移動停損」的前半）
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_entry_ladder_has_three_tranches() -> None:
    c = make_candidate("2330", entry_price=200.0, trail_pct=0.12)
    assert len(c.entry_ladder()) == 3


@pytest.mark.unit
def test_entry_ladder_matches_hand_calculation() -> None:
    """
    進場 200、移動停損 12% → 初始停損 176。

    三批分佈在 [停損, 進場] 之間，第一批在現價、最後一批仍高於停損：

        第 1 批  200 × (1 − 0 × 0.12/3) = 200.00
        第 2 批  200 × (1 − 1 × 0.12/3) = 192.00
        第 3 批  200 × (1 − 2 × 0.12/3) = 184.00

    最後一批 184 > 停損 176 —— 不可在停損之下加碼。
    """
    c = make_candidate("2330", entry_price=200.0, trail_pct=0.12)
    prices = [t.price for t in c.entry_ladder()]

    assert prices == pytest.approx([200.0, 192.0, 184.0])
    assert prices[-1] > c.initial_stop_price


@pytest.mark.unit
def test_entry_ladder_weights_sum_to_one() -> None:
    c = make_candidate("2330")
    assert sum(t.weight for t in c.entry_ladder()) == pytest.approx(1.0)


@pytest.mark.unit
def test_entry_ladder_shares_sum_to_position() -> None:
    """各批股數加總不可超過建議部位"""
    c = make_candidate("2330", entry_price=200.0, trail_pct=0.12)
    result = select_trailing_portfolio([c], CAPITAL, no_correlation(["2330"]))
    position = result.positions[0]

    ladder = c.entry_ladder(total_shares=position.shares)
    assert sum(t.shares for t in ladder) <= position.shares


@pytest.mark.unit
def test_entry_ladder_never_below_stop() -> None:
    """任何 trail_pct 下，最低一批都必須高於初始停損"""
    for trail in (0.05, 0.12, 0.25, 0.30):
        c = make_candidate("2330", entry_price=200.0, trail_pct=trail)
        assert c.entry_ladder()[-1].price > c.initial_stop_price


@pytest.mark.unit
def test_describe_shows_entry_ladder() -> None:
    c = make_candidate("2330", entry_price=200.0, trail_pct=0.12)
    text = select_trailing_portfolio([c], CAPITAL, no_correlation(["2330"])).describe()

    assert "分批進場" in text
    assert "192" in text
    assert "184" in text


@pytest.mark.unit
def test_describe_discloses_ladder_is_not_backtested() -> None:
    """
    回測假設 T+1 開盤一次買足，**沒有模擬分批**。

    不寫清楚，讀者會以為分批的績效也被驗證過了。
    """
    c = make_candidate("2330")
    text = select_trailing_portfolio([c], CAPITAL, no_correlation(["2330"])).describe()

    assert "回測未模擬" in text


# ══════════════════════════════════════════════════════════════
# 分箱校準的離散性（必須揭露）
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_describe_warns_when_picks_share_expected_return() -> None:
    """
    分箱校準只給有限個離散值。多檔落在同一箱時期望報酬完全相同，
    **它們之間的排序是任意的**（目前用代號決勝）。

    不揭露的話，讀者會以為 🥇 比 🥈 更有把握——那是假的。
    """
    ids = ["A", "B", "C"]
    cands = [
        make_candidate(s, expected_gross_return=0.18, industry=ind)
        for s, ind in zip(ids, ("半導體", "金融", "食品"))
    ]
    text = select_trailing_portfolio(cands, CAPITAL, no_correlation(ids)).describe()

    assert "排序為任意" in text


@pytest.mark.unit
def test_describe_no_warning_when_expected_returns_differ() -> None:
    ids = ["A", "B"]
    cands = [
        make_candidate("A", expected_gross_return=0.20, industry="半導體"),
        make_candidate("B", expected_gross_return=0.14, industry="金融"),
    ]
    text = select_trailing_portfolio(cands, CAPITAL, no_correlation(ids)).describe()

    assert "排序為任意" not in text
