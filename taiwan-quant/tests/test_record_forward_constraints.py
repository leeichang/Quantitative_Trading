"""
前推帳本的投組約束測試

## 為什麼兩版並存

`momentum_top10_h40@v1` 的第一筆紀錄（2026-09-11）有 **6 檔金融保險**，
違反 CLAUDE.md「同產業 ≤ 2」。帳本不可改寫，所以它留著當對照組，
約束版走新的 `strategy_version`。

**兩版在同一段未來上被直接比較**，比在已用掉的區間上重跑可信得多。

## 為什麼不放寬約束的數字

開發集實測（`reports/constraint_cost_dev.json`，36 期）：

```
全部嚴格 vs baseline   配對差異 −1.75 pp/趟   標準誤 1.58 pp   t = −1.11
```

2 SE = 3.16 pp，而優勢本身只有 3.28 pp——**這個檢定只能偵測「約束把整個
優勢消滅」**，再小的代價都看不見。沒有證據支持放寬，所以維持原規格。
"""

from __future__ import annotations

import json

import pytest

from taiwan_quant.ranking.constraints import ConstraintLimits
from taiwan_quant.ranking.portfolio_features import build_candidates

from scripts.record_forward import (
    CLAUDE_MD_LIMITS,
    N_POSITIONS,
    VARIANTS,
    Variant,
    select,
)


def candidates_with(industries: dict[str, str], **overrides: dict) -> list:
    """所有候選低波動、低 beta，讓測試只檢驗指定的那條約束"""
    ordered = list(industries)
    return build_candidates(
        ordered=ordered,
        scores={sid: 1.0 - i * 0.01 for i, sid in enumerate(ordered)},
        industries=industries,
        atr_pct=overrides.get("atr_pct", {sid: 0.1 for sid in ordered}),
        beta_map=overrides.get("beta_map", {sid: 0.5 for sid in ordered}),
    )


def zero_correlations(ids: list[str]) -> dict[tuple[str, str], float]:
    """全部兩兩無關——相關約束不可在其他測試裡誤觸發"""
    return {
        (a, b): 0.0 for i, a in enumerate(ids) for b in ids[i + 1:]
    }


# ══════════════════════════════════════════════════════════════
# 兩個版本的定義
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_two_variants_with_distinct_versions() -> None:
    """
    版本字串必須相異，否則唯一鍵碰撞、第二版會被 `INSERT OR IGNORE`
    靜默丟掉——`edge_z` 那個 bug 就是這樣發生的。
    """
    versions = [v.strategy_version for v in VARIANTS]
    assert len(versions) == len(set(versions))
    assert [v.limits is None for v in VARIANTS] == [True, False], (
        "第一版必須是無約束的對照組，第二版才是約束版"
    )


@pytest.mark.unit
def test_constrained_variant_uses_claude_md_numbers_verbatim() -> None:
    """
    約束版的數字必須與 CLAUDE.md 完全一致。

    放寬任何一個都是「用回測結果放寬風控規格」。這個測試存在的目的就是
    讓那件事不能安靜地發生。
    """
    assert CLAUDE_MD_LIMITS.max_same_industry == 2
    assert CLAUDE_MD_LIMITS.max_high_volatility == 1
    assert CLAUDE_MD_LIMITS.max_correlation == 0.7
    assert CLAUDE_MD_LIMITS.require_defensive is True
    assert CLAUDE_MD_LIMITS.top_n == N_POSITIONS


@pytest.mark.unit
def test_params_json_records_the_constraint_settings() -> None:
    """
    禁令 8：約束設定也是回測參數。

    不存的話，事後看到「這一版少了 4 檔金融」會無法判斷是約束造成的
    還是訊號變了。
    """
    constrained = next(v for v in VARIANTS if v.limits is not None)
    params = json.loads(constrained.params_json)

    assert params["portfolio_constraints"]["max_same_industry"] == 2
    assert params["portfolio_constraints"]["max_correlation"] == 0.7
    # 高波動與 defensive 的判定門檻也要存——換了門檻等於換了約束
    assert "high_volatility_percentile" in params["portfolio_constraints"]
    assert "defensive_beta_max" in params["portfolio_constraints"]
    assert "beta_benchmark" in params["portfolio_constraints"]

    control = next(v for v in VARIANTS if v.limits is None)
    assert json.loads(control.params_json)["portfolio_constraints"] is None


# ══════════════════════════════════════════════════════════════
# select()
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_no_limits_keeps_pure_ranking() -> None:
    """對照組必須與舊行為完全一致，否則它就不是對照組了"""
    industries = {f"S{i}": f"產業{i}" for i in range(15)}
    picked = select(candidates_with(industries), {}, None)

    assert picked == [f"S{i}" for i in range(N_POSITIONS)]


@pytest.mark.unit
def test_industry_cap_replaces_with_lower_ranked_names() -> None:
    """
    重現第一筆帳本的形狀：6 檔金融 + 夠多其他產業。

    約束應該保留前 2 檔金融，其餘用較低排名的其他產業補上——
    **不是整批丟掉金融股**。
    """
    industries = {
        **{f"F{i}": "金融保險" for i in range(6)},
        **{f"O{i}": f"產業{i}" for i in range(9)},
    }
    ordered = list(industries)
    picked = select(
        candidates_with(industries),
        zero_correlations(ordered),
        ConstraintLimits(top_n=N_POSITIONS, max_same_industry=2,
                         max_high_volatility=N_POSITIONS,
                         max_correlation=1.0, require_defensive=False),
    )

    assert len(picked) == N_POSITIONS
    assert [p for p in picked if p.startswith("F")] == ["F0", "F1"]
    assert picked[:3] == ["F0", "F1", "O0"], "高分的金融股不可被整批丟掉"


@pytest.mark.unit
def test_short_of_n_is_allowed() -> None:
    """
    約束擋太多時湊不滿 N 是**刻意的**。

    CLAUDE.md：「寧可少推幾檔，也不違反風控約束。」湊滿 10 檔不是目標。
    """
    industries = {f"F{i}": "金融保險" for i in range(6)}
    ordered = list(industries)
    picked = select(
        candidates_with(industries),
        zero_correlations(ordered),
        ConstraintLimits(top_n=N_POSITIONS, max_same_industry=2,
                         max_high_volatility=N_POSITIONS,
                         max_correlation=1.0, require_defensive=False),
    )

    assert picked == ["F0", "F1"]


@pytest.mark.unit
def test_correlation_cap_rejects_co_moving_names() -> None:
    """相關係數 ≥ 0.7 的配對不可同時入選"""
    industries = {f"S{i}": f"產業{i}" for i in range(4)}
    ordered = list(industries)
    correlations = zero_correlations(ordered) | {("S0", "S1"): 0.95}
    picked = select(
        candidates_with(industries),
        correlations,
        ConstraintLimits(top_n=N_POSITIONS, max_same_industry=N_POSITIONS,
                         max_high_volatility=N_POSITIONS,
                         max_correlation=0.7, require_defensive=False),
    )

    assert "S1" not in picked
    assert picked == ["S0", "S2", "S3"]


@pytest.mark.unit
def test_missing_correlation_is_rejected_not_assumed_zero() -> None:
    """
    缺相關係數一律拒絕。

    填 0 等於宣稱「這兩檔無關」——那正是約束要防的事。
    """
    industries = {f"S{i}": f"產業{i}" for i in range(3)}
    picked = select(
        candidates_with(industries),
        {},  # 完全沒有相關係數資料
        ConstraintLimits(top_n=N_POSITIONS, max_same_industry=N_POSITIONS,
                         max_high_volatility=N_POSITIONS,
                         max_correlation=0.7, require_defensive=False),
    )

    assert picked == ["S0"], "第一檔沒有比較對象所以可以進；之後每一檔都缺資料"


@pytest.mark.unit
def test_volatility_cap_counts_only_high_volatility_names() -> None:
    """ATR 分位 > 0.80 才佔用高波動額度；低波動的不受限"""
    industries = {f"S{i}": f"產業{i}" for i in range(5)}
    ordered = list(industries)
    picked = select(
        candidates_with(
            industries,
            atr_pct={"S0": 0.95, "S1": 0.90, "S2": 0.5, "S3": 0.5, "S4": 0.5},
        ),
        zero_correlations(ordered),
        ConstraintLimits(top_n=N_POSITIONS, max_same_industry=N_POSITIONS,
                         max_high_volatility=1,
                         max_correlation=1.0, require_defensive=False),
    )

    assert picked == ["S0", "S2", "S3", "S4"], "S1 是第二檔高波動，應被擋"
