#!/usr/bin/env python3
"""
Top 3 選股與部位規模測試

依據 D4 與 CLAUDE.md 的投組約束：

    最多 2 支同產業
    最多 1 支高波動（ATR 分位 > 80%）
    至少 1 支 defensive（低 beta）
    三檔之間 60 日報酬相關係數 < 0.7

    建議股數 = floor(min(
        總資金 × 1% ÷ (entry − stop),     # 風險倒推
        總資金 × 33% ÷ entry              # 單檔資金上限
    ))
    若 < 1 股 → 不推播此檔

進場門檻是**動態**的（D7 修訂）：

    P(+1) ≥ (stop_pct + round_trip_cost) / (target_pct + stop_pct)

不是固定的 0.55——門檻隨每檔標的的柵欄寬度與流動性分層計算。

核心設計原則：**寧可少推幾檔，也不違反風控約束。**
湊滿 3 檔不是目標，「這 3 檔的風險/報酬比最好」才是。
"""

from __future__ import annotations

import numpy as np
import pytest

from taiwan_quant.config.costs import DEFAULT, Tier
from taiwan_quant.ranking.portfolio import (
    Candidate,
    PortfolioResult,
    RejectReason,
    entry_threshold,
    position_shares,
    select_portfolio,
)

pytestmark = pytest.mark.unit

CAPITAL = 400_000.0


def cand(
    stock_id: str,
    prob_up: float = 0.60,
    target_pct: float = 0.062,
    stop_pct: float = 0.0238,
    entry_price: float = 100.0,
    tier: str = "0050",
    industry: str = "半導體",
    volatility_pct: float = 0.5,
    beta: float = 0.9,
) -> Candidate:
    return Candidate(
        stock_id=stock_id,
        prob_up=prob_up,
        target_pct=target_pct,
        stop_pct=stop_pct,
        entry_price=entry_price,
        tier=Tier(tier),
        industry=industry,
        volatility_pct=volatility_pct,
        beta=beta,
    )


def uncorrelated(ids: list[str], rho: float = 0.1) -> dict[tuple[str, str], float]:
    """產生低相關矩陣"""
    return {
        (a, b): rho
        for i, a in enumerate(ids)
        for b in ids[i + 1:]
    }


# ══════════════════════════════════════════════════════════════
# 動態進場門檻
# ══════════════════════════════════════════════════════════════


def test_entry_threshold_from_barrier_and_cost() -> None:
    """
    手算（D7 修訂後的實測參數）：
        target 6.20%、stop 2.38%、0050 來回成本 1.071%

        門檻 = (0.0238 + 0.01071) / (0.0620 + 0.0238)
             = 0.03451 / 0.0858
             = 0.402214...   →  40.22%

    這正是 CLAUDE.md 的新紅線。
    """
    threshold = entry_threshold(
        target_pct=0.0620, stop_pct=0.0238, cost=DEFAULT, tier=Tier.LARGE
    )
    assert threshold == pytest.approx(0.402214, abs=1e-5)


def test_entry_threshold_without_cost_is_breakeven() -> None:
    """
    不計成本時退化為損益兩平勝率：
        stop / (target + stop) = 0.0238 / 0.0858 = 0.277389...
    """
    from taiwan_quant.config.costs import GROSS

    threshold = entry_threshold(0.0620, 0.0238, cost=GROSS, tier=Tier.LARGE)
    assert threshold == pytest.approx(0.0238 / 0.0858, abs=1e-9)


def test_entry_threshold_higher_for_mid_tier() -> None:
    """中型股滑價較高 → 門檻較高（要更確定才值得做）"""
    large = entry_threshold(0.0620, 0.0238, DEFAULT, Tier.LARGE)
    mid = entry_threshold(0.0620, 0.0238, DEFAULT, Tier.MID)
    assert mid > large


def test_entry_threshold_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="target_pct"):
        entry_threshold(0.0, 0.02, DEFAULT, Tier.LARGE)
    with pytest.raises(ValueError, match="stop_pct"):
        entry_threshold(0.06, 0.0, DEFAULT, Tier.LARGE)


# ══════════════════════════════════════════════════════════════
# 部位規模：兩道上限取小（D4）
# ══════════════════════════════════════════════════════════════


def test_position_shares_risk_based() -> None:
    """
    風險倒推主導的情況。

    手算：資金 40 萬、entry 2430、stop 2.38%
        每股風險 = 2430 × 0.0238 = 57.834 元
        風險倒推 = 400000 × 0.01 / 57.834 = 4000 / 57.834 = 69.16 股
        資金上限 = 400000 × 0.33 / 2430 = 132000 / 2430 = 54.32 股
        取小 → 54 股（資金上限主導）
    """
    shares = position_shares(CAPITAL, entry_price=2430.0, stop_pct=0.0238)
    assert shares == 54


def test_position_shares_capital_cap_binds_on_tight_stop() -> None:
    """
    停損很窄時，1% 風險倒推會爆掉資金上限——這正是第二道上限存在的理由。

    手算：entry 100、stop 1%
        每股風險 = 1.0 元
        風險倒推 = 400000 × 0.01 / 1.0 = 4,000 股 → 部位 40 萬（100% 資金！）
        資金上限 = 400000 × 0.33 / 100 = 1,320 股 → 部位 13.2 萬（33%）
        取小 → 1,320 股
    """
    shares = position_shares(CAPITAL, entry_price=100.0, stop_pct=0.01)
    assert shares == 1320
    assert shares * 100.0 == pytest.approx(CAPITAL * 0.33)


def test_position_shares_risk_cap_binds_on_wide_stop() -> None:
    """
    停損很寬時，風險倒推主導。

    手算：entry 100、stop 10%
        每股風險 = 10 元
        風險倒推 = 4000 / 10 = 400 股 → 部位 4 萬（10% 資金）
        資金上限 = 1,320 股
        取小 → 400 股
    """
    assert position_shares(CAPITAL, entry_price=100.0, stop_pct=0.10) == 400


def test_position_shares_floors_to_whole_share() -> None:
    """
    零股以「股」為單位，必須向下取整。

    手算：entry 2430、stop 2.38%、資金上限主導 = 54.32 股 → 54 股
    向上取整會超出資金上限。
    """
    assert position_shares(CAPITAL, 2430.0, 0.0238) == 54


def test_position_shares_zero_when_too_expensive() -> None:
    """
    單股價格高到連 1 股都超出資金上限 → 回 0（不推播此檔）。

    手算：entry 200,000 元、資金上限 = 400000 × 0.33 = 132,000
        132,000 / 200,000 = 0.66 股 → floor → 0
    """
    assert position_shares(CAPITAL, entry_price=200_000.0, stop_pct=0.05) == 0


def test_position_shares_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="capital"):
        position_shares(0.0, 100.0, 0.02)
    with pytest.raises(ValueError, match="entry_price"):
        position_shares(CAPITAL, 0.0, 0.02)
    with pytest.raises(ValueError, match="stop_pct"):
        position_shares(CAPITAL, 100.0, 0.0)


def test_position_shares_never_exceeds_caps() -> None:
    """隨機參數下兩道上限都不可被突破"""
    rng = np.random.default_rng(5)
    for _ in range(200):
        entry = float(rng.uniform(10, 3000))
        stop = float(rng.uniform(0.005, 0.15))
        shares = position_shares(CAPITAL, entry, stop)
        assert shares * entry <= CAPITAL * 0.33 + 1e-6
        assert shares * entry * stop <= CAPITAL * 0.01 + 1e-6


# ══════════════════════════════════════════════════════════════
# 進場門檻過濾
# ══════════════════════════════════════════════════════════════


def test_rejects_candidate_below_entry_threshold() -> None:
    """
    P(+1) 低於動態門檻（40.22%）→ 剔除。

    這是 D7 修訂的紅線在選股層的落實。
    """
    weak = cand("A", prob_up=0.35)
    result = select_portfolio([weak], CAPITAL, correlations={})
    assert not result.positions
    assert result.rejected[0].reason is RejectReason.BELOW_ENTRY_THRESHOLD


def test_accepts_candidate_above_entry_threshold() -> None:
    result = select_portfolio([cand("A", prob_up=0.60)], CAPITAL, correlations={})
    assert [p.candidate.stock_id for p in result.positions] == ["A"]


def test_rejects_candidate_below_min_risk_reward() -> None:
    """
    R:R < 2.0 → 剔除。風險紀律不為湊檔數放寬。

    手算：target 3%、stop 2% → R:R = 1.5 < 2.0
    """
    poor_rr = cand("A", target_pct=0.03, stop_pct=0.02, prob_up=0.95)
    result = select_portfolio([poor_rr], CAPITAL, correlations={})
    assert result.rejected[0].reason is RejectReason.BELOW_RISK_REWARD


# ══════════════════════════════════════════════════════════════
# 投組約束
# ══════════════════════════════════════════════════════════════


def test_at_most_two_same_industry() -> None:
    """
    同產業最多 2 支。

    構造：4 檔全是半導體，品質遞減。應只選前 2 檔。
    """
    ids = ["A", "B", "C", "D"]
    candidates = [
        cand(sid, prob_up=p, industry="半導體", beta=0.8)
        for sid, p in zip(ids, [0.75, 0.70, 0.65, 0.60])
    ]
    result = select_portfolio(candidates, CAPITAL, correlations=uncorrelated(ids))

    assert [p.candidate.stock_id for p in result.positions] == ["A", "B"]
    assert any(r.reason is RejectReason.INDUSTRY_LIMIT for r in result.rejected)


def test_at_most_one_high_volatility() -> None:
    """
    高波動（ATR 分位 > 80%）最多 1 支。

    構造：3 檔都是高波動、不同產業。應只選 1 檔。
    """
    ids = ["A", "B", "C"]
    candidates = [
        cand(sid, prob_up=p, industry=ind, volatility_pct=0.95, beta=0.8)
        for sid, p, ind in zip(ids, [0.75, 0.70, 0.65], ["半導體", "金融", "航運"])
    ]
    result = select_portfolio(candidates, CAPITAL, correlations=uncorrelated(ids))

    assert len(result.positions) == 1
    assert any(r.reason is RejectReason.VOLATILITY_LIMIT for r in result.rejected)


def test_correlation_limit_blocks_similar_pair() -> None:
    """
    三檔之間 60 日報酬相關係數必須 < 0.7。

    構造：A 與 B 高度相關（0.85），不同產業所以產業限制擋不住。
    B 應被相關性擋掉。
    """
    candidates = [
        cand("A", prob_up=0.75, industry="半導體", beta=0.8),
        cand("B", prob_up=0.70, industry="電子零件", beta=0.8),
        cand("C", prob_up=0.65, industry="金融", beta=0.8),
    ]
    correlations = {("A", "B"): 0.85, ("A", "C"): 0.2, ("B", "C"): 0.2}

    result = select_portfolio(candidates, CAPITAL, correlations=correlations)

    assert [p.candidate.stock_id for p in result.positions] == ["A", "C"]
    assert any(r.reason is RejectReason.CORRELATION_LIMIT for r in result.rejected)


def test_correlation_lookup_is_order_independent() -> None:
    """相關係數字典用 (A,B) 或 (B,A) 查都要找得到"""
    candidates = [
        cand("A", prob_up=0.75, industry="半導體", beta=0.8),
        cand("B", prob_up=0.70, industry="金融", beta=0.8),
    ]
    result = select_portfolio(candidates, CAPITAL, correlations={("B", "A"): 0.9})
    assert len(result.positions) == 1


def test_missing_correlation_is_rejected_conservatively() -> None:
    """
    缺相關係數資料時**保守拒絕**，不可當成 0。

    當成 0 等於假設「無關」，可能讓兩檔高度相關的標的同時入選，
    而使用者以為已經分散了。
    """
    candidates = [
        cand("A", prob_up=0.75, industry="半導體", beta=0.8),
        cand("B", prob_up=0.70, industry="金融", beta=0.8),
    ]
    result = select_portfolio(candidates, CAPITAL, correlations={})

    assert len(result.positions) == 1
    assert any(r.reason is RejectReason.CORRELATION_UNKNOWN for r in result.rejected)


def test_top_n_limit() -> None:
    """最多 3 檔，即使有更多合格候選"""
    ids = [f"S{i}" for i in range(6)]
    candidates = [
        cand(sid, prob_up=0.70 - i * 0.01, industry=f"產業{i}", beta=0.8)
        for i, sid in enumerate(ids)
    ]
    result = select_portfolio(candidates, CAPITAL, correlations=uncorrelated(ids))
    assert len(result.positions) == 3


# ══════════════════════════════════════════════════════════════
# 至少 1 支 defensive
# ══════════════════════════════════════════════════════════════


def test_requires_at_least_one_defensive() -> None:
    """
    最終組合必須含至少 1 支低 beta 標的。

    構造：3 檔全是高 beta（1.5）。應退回較少檔數並警告。
    """
    ids = ["A", "B", "C"]
    candidates = [
        cand(sid, prob_up=p, industry=ind, beta=1.5)
        for sid, p, ind in zip(ids, [0.75, 0.70, 0.65], ["半導體", "金融", "航運"])
    ]
    result = select_portfolio(candidates, CAPITAL, correlations=uncorrelated(ids))

    assert not result.has_defensive
    assert any("defensive" in w for w in result.warnings)


def test_defensive_is_promoted_into_portfolio() -> None:
    """
    有 defensive 候選時應被納入，即使 P(+1) 排名較後。

    構造：前兩名高 beta、第三名低 beta。
    貪婪選法會選前三名（全高 beta）；正確實作要把 defensive 換進來。
    """
    ids = ["A", "B", "C", "D"]
    candidates = [
        cand("A", prob_up=0.75, industry="半導體", beta=1.5),
        cand("B", prob_up=0.72, industry="金融", beta=1.4),
        cand("C", prob_up=0.70, industry="航運", beta=1.3),
        cand("D", prob_up=0.60, industry="食品", beta=0.6),   # defensive
    ]
    result = select_portfolio(candidates, CAPITAL, correlations=uncorrelated(ids))

    picked = [p.candidate.stock_id for p in result.positions]
    assert "D" in picked, f"defensive 未被納入：{picked}"
    assert result.has_defensive


def test_no_warning_when_defensive_present() -> None:
    candidates = [
        cand("A", prob_up=0.75, industry="半導體", beta=0.7),
        cand("B", prob_up=0.70, industry="金融", beta=0.6),
    ]
    result = select_portfolio(
        candidates, CAPITAL, correlations={("A", "B"): 0.2}
    )
    assert result.has_defensive
    assert not any("defensive" in w for w in result.warnings)


# ══════════════════════════════════════════════════════════════
# 部位規模過濾
# ══════════════════════════════════════════════════════════════


def test_drops_candidate_when_shares_below_one() -> None:
    """
    算出 < 1 股 → 不推播（D4）。

    構造：單股 20 萬，超出 33% 資金上限（13.2 萬）。
    """
    expensive = cand("A", entry_price=200_000.0, prob_up=0.70)
    result = select_portfolio([expensive], CAPITAL, correlations={})

    assert not result.positions
    assert result.rejected[0].reason is RejectReason.POSITION_TOO_SMALL


def test_position_reports_capital_percentage() -> None:
    """
    推播要用「佔總資金 %」表達（D4）。

    手算：entry 2430、54 股 → 131,220 元 → 32.8% 資金
    """
    result = select_portfolio(
        [cand("A", entry_price=2430.0, prob_up=0.70)], CAPITAL, correlations={}
    )
    position = result.positions[0]

    assert position.shares == 54
    assert position.position_value == pytest.approx(54 * 2430.0)
    assert position.capital_pct == pytest.approx(54 * 2430.0 / CAPITAL)
    assert 0.32 < position.capital_pct < 0.33


def test_total_capital_within_100_percent() -> None:
    """三檔合計不可超過 100% 資金（33% × 3 = 99%）"""
    ids = ["A", "B", "C"]
    candidates = [
        cand(sid, prob_up=0.70, industry=ind, entry_price=100.0, stop_pct=0.01, beta=0.8)
        for sid, ind in zip(ids, ["半導體", "金融", "航運"])
    ]
    result = select_portfolio(candidates, CAPITAL, correlations=uncorrelated(ids))
    assert result.total_capital_pct <= 1.0


# ══════════════════════════════════════════════════════════════
# 期望值與排序
# ══════════════════════════════════════════════════════════════


def test_expected_return_net_of_cost() -> None:
    """
    手算：P(+1)=0.60、target 6.20%、stop 2.38%、成本 1.071%
        期望毛 = 0.60 × 0.0620 − 0.40 × 0.0238 = 0.0372 − 0.00952 = 0.02768
        期望淨 = 0.02768 − 0.01071 = 0.01697
    """
    result = select_portfolio([cand("A", prob_up=0.60)], CAPITAL, correlations={})
    assert result.positions[0].expected_return == pytest.approx(0.01697, abs=1e-6)


def test_ranked_by_expected_return_not_probability() -> None:
    """
    排序依**期望淨報酬**，不是 P(+1)。

    構造：A 勝率高但 R:R 差、B 勝率略低但 R:R 好。
        A: p=0.70、target 5%、stop 2.5% → 毛 0.70×0.05 − 0.30×0.025 = 0.0275
        B: p=0.65、target 9%、stop 3.0% → 毛 0.65×0.09 − 0.35×0.030 = 0.0480
    B 的期望值較高，應排前面。
    """
    candidates = [
        cand("A", prob_up=0.70, target_pct=0.05, stop_pct=0.025, industry="半導體", beta=0.8),
        cand("B", prob_up=0.65, target_pct=0.09, stop_pct=0.030, industry="金融", beta=0.8),
    ]
    result = select_portfolio(
        candidates, CAPITAL, correlations={("A", "B"): 0.2}
    )
    assert [p.candidate.stock_id for p in result.positions] == ["B", "A"]


def test_risk_reward_reported() -> None:
    """
    手算：target 6.20% / stop 2.38% = 2.605
    """
    result = select_portfolio([cand("A")], CAPITAL, correlations={})
    assert result.positions[0].risk_reward == pytest.approx(0.062 / 0.0238, abs=1e-6)


# ══════════════════════════════════════════════════════════════
# 可稽核性（禁令 7、8）
# ══════════════════════════════════════════════════════════════


def test_every_rejection_has_a_reason() -> None:
    """
    每個被剔除的候選都要記錄原因。

    沒有這個就無法回答「為什麼 2026-09-12 沒有推薦台積電」。
    """
    ids = ["A", "B", "C", "D", "E"]
    candidates = [
        cand("A", prob_up=0.75, industry="半導體", beta=0.8),
        cand("B", prob_up=0.72, industry="半導體", beta=0.8),
        cand("C", prob_up=0.70, industry="半導體", beta=0.8),   # 產業超限
        cand("D", prob_up=0.30, industry="金融", beta=0.8),      # 門檻不足
        cand("E", prob_up=0.70, industry="航運", entry_price=1e6, beta=0.8),  # 部位過小
    ]
    result = select_portfolio(candidates, CAPITAL, correlations=uncorrelated(ids))

    evaluated = {p.candidate.stock_id for p in result.positions} | {
        r.stock_id for r in result.rejected
    }
    assert evaluated == set(ids), "有候選既沒入選也沒記錄剔除原因"
    assert all(r.detail for r in result.rejected), "剔除原因缺少說明文字"


def test_result_is_immutable() -> None:
    result = select_portfolio([cand("A")], CAPITAL, correlations={})
    with pytest.raises(Exception):
        result.positions = ()  # type: ignore[misc]


def test_describe_is_human_readable() -> None:
    ids = ["A", "B"]
    candidates = [
        cand("A", prob_up=0.70, industry="半導體", beta=0.7),
        cand("B", prob_up=0.65, industry="金融", beta=0.6),
    ]
    text = select_portfolio(
        candidates, CAPITAL, correlations=uncorrelated(ids)
    ).describe()

    assert "進場區間" in text
    assert "目標價" in text
    assert "失效價" in text
    assert "零股" in text


# ══════════════════════════════════════════════════════════════
# 輸入驗證
# ══════════════════════════════════════════════════════════════


def test_empty_candidates_returns_empty_result() -> None:
    """
    沒有候選 → 回空結果並警告，不可拋錯。

    「這週沒有值得買的」是合法且重要的結論。
    """
    result = select_portfolio([], CAPITAL, correlations={})
    assert not result.positions
    assert any("候選" in w for w in result.warnings)


def test_candidate_validates_probability_range() -> None:
    with pytest.raises(ValueError, match="prob_up"):
        cand("A", prob_up=1.5)


def test_candidate_validates_positive_barriers() -> None:
    with pytest.raises(ValueError, match="target_pct"):
        cand("A", target_pct=0.0)
    with pytest.raises(ValueError, match="stop_pct"):
        cand("A", stop_pct=-0.01)


def test_rejects_nonpositive_capital() -> None:
    with pytest.raises(ValueError, match="capital"):
        select_portfolio([cand("A")], 0.0, correlations={})
