"""
無資訊對照組的測試

## 這個模組要擋住什麼

2026-09-17 實測證明 CLAUDE.md 原本的「隨機進場」對照組太弱：一個標籤
被打亂、證明沒有預測資訊的模型，仍然顯著打敗它（+1.88%／趟，t = 2.32）。

原因是**用任意函數取前 N 檔 ≠ 均勻隨機抽 N 檔**——任意函數繼承一個
因子傾斜。所以對照組必須保留策略的結構（加權特徵混合、集中選股、
時間上持續），只把手挑權重換成任意權重。

**最容易做錯的是讓權重每期重抽**，那會失去持續性，退化回太弱的門檻。
`test_weights_are_constant_across_dates` 釘住它。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from taiwan_quant.validation.uninformed import (
    DEFAULT_DRAWS,
    NullDistribution,
    UninformedError,
    cross_sectional_ranks,
    draw_weights,
    random_weight_scores,
)


# ══════════════════════════════════════════════════════════════
# 分位：尺度無關
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_ranks_are_scale_free() -> None:
    """
    兩個量級差 100 倍的特徵，分位必須相同——否則加權時量級大的那個
    主導，而那是量級的效果不是權重的。

    手算：三檔，同一個順序
        momentum  0.01, 0.02, 0.03   →  1/3, 2/3, 1
        rsi         30,   50,   70   →  1/3, 2/3, 1
    """
    features = pd.DataFrame(
        {"momentum": [0.01, 0.02, 0.03], "rsi": [30.0, 50.0, 70.0]},
        index=["A", "B", "C"],
    )

    ranks = cross_sectional_ranks(features)

    assert ranks["momentum"].tolist() == [1 / 3, 2 / 3, 1.0]
    assert ranks["rsi"].tolist() == [1 / 3, 2 / 3, 1.0]


@pytest.mark.unit
def test_ranks_reject_single_candidate() -> None:
    with pytest.raises(UninformedError, match="無從排序"):
        cross_sectional_ranks(pd.DataFrame({"a": [1.0]}, index=["A"]))


# ══════════════════════════════════════════════════════════════
# 加權：缺值填 0.5 而非 0
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_missing_feature_uses_median_rank_not_zero() -> None:
    """
    缺值代入 0.5（當日分位中位），**不是 0**。

    0 代表「該特徵最低」，會讓缺籌碼資料的標的被系統性壓低——那是一個
    沒有宣告的篩選條件，而且它會讓對照組偏向有完整資料的大型股。

    手算：權重 [1.0, 1.0]
        A  ranks (1.0, 1.0)      → 2.0
        B  ranks (0.5, NaN→0.5)  → 1.0
        若缺值填 0 則 B = 0.5，差一倍
    """
    ranks = pd.DataFrame(
        {"a": [1.0, 0.5], "b": [1.0, np.nan]}, index=["A", "B"]
    )

    scores = random_weight_scores(ranks, np.array([1.0, 1.0]))

    assert scores["A"] == pytest.approx(2.0)
    assert scores["B"] == pytest.approx(1.0)


@pytest.mark.unit
def test_all_missing_stays_missing() -> None:
    """全特徵皆缺的標的不可拿到一個分數——它應該根本不參加"""
    ranks = pd.DataFrame(
        {"a": [1.0, np.nan], "b": [1.0, np.nan]}, index=["A", "B"]
    )

    scores = random_weight_scores(ranks, np.array([1.0, 1.0]))

    assert scores["A"] == pytest.approx(2.0)
    assert np.isnan(scores["B"])


@pytest.mark.unit
def test_weight_length_must_match() -> None:
    ranks = pd.DataFrame({"a": [1.0, 0.5]}, index=["A", "B"])
    with pytest.raises(UninformedError, match="不符"):
        random_weight_scores(ranks, np.array([1.0, 1.0]))


# ══════════════════════════════════════════════════════════════
# 權重：可重現、有正有負、時間上固定
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_weights_are_reproducible() -> None:
    """禁令 7：同一個種子必須給出同一組權重"""
    first = draw_weights(32, n_draws=10, seed=42)
    second = draw_weights(32, n_draws=10, seed=42)

    assert np.array_equal(first, second)
    assert first.shape == (10, 32)


@pytest.mark.unit
def test_weights_span_both_signs() -> None:
    """
    有正有負，所以對照組不預設「特徵越高越好」。

    手挑權重全為正（`families._blend` 的 weights），但那是手挑的一部分，
    對照組不該繼承那個先驗——繼承了就等於偷偷給對照組一半的答案。
    """
    weights = draw_weights(32, n_draws=50, seed=1)

    assert (weights > 0).any() and (weights < 0).any()
    assert abs(float(weights.mean())) < 0.2, "應該大致以 0 為中心"


@pytest.mark.unit
def test_weights_are_constant_across_dates() -> None:
    """
    一次抽樣 = 一組權重，套用到所有決策日。

    手工分數的權重不隨時間變，所以對照組也不能變。每期重抽會讓選股
    失去時間上的持續性，退化回「隨機選股」——正是被證明太弱的門檻。

    這裡驗的是 API 的形狀：`draw_weights` 回傳 `(n_draws, n_features)`，
    也就是每次抽樣**一組**權重，不是每期一組。
    """
    weights = draw_weights(32, n_draws=7, seed=3)

    assert weights.ndim == 2
    assert weights.shape == (7, 32), "第一維是抽樣次數，不是決策日數"


@pytest.mark.unit
def test_draw_weights_rejects_nonpositive() -> None:
    with pytest.raises(UninformedError, match="n_features"):
        draw_weights(0)
    with pytest.raises(UninformedError, match="n_draws"):
        draw_weights(32, n_draws=0)


# ══════════════════════════════════════════════════════════════
# 不同權重要選出不同名字（否則對照組沒有變異）
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_different_weights_pick_different_names() -> None:
    """
    對照組要有變異，否則「分位」沒有意義。

    兩組相反的權重在同一份特徵上必須選出不同的前 2 名。
    """
    ranks = pd.DataFrame(
        {
            "momentum": [1.0, 0.75, 0.5, 0.25],
            "volatility": [0.25, 0.5, 0.75, 1.0],
        },
        index=["A", "B", "C", "D"],
    )

    favour_momentum = random_weight_scores(ranks, np.array([1.0, 0.0]))
    favour_volatility = random_weight_scores(ranks, np.array([0.0, 1.0]))

    top_momentum = set(favour_momentum.nlargest(2).index)
    top_volatility = set(favour_volatility.nlargest(2).index)

    assert top_momentum == {"A", "B"}
    assert top_volatility == {"C", "D"}
    assert top_momentum.isdisjoint(top_volatility)


# ══════════════════════════════════════════════════════════════
# NullDistribution
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_percentile_of_observed_hand_computed() -> None:
    """手算：10 次抽樣，真策略贏 7 次 → 第 70 百分位"""
    null = NullDistribution(
        draws=(0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.09, 0.10, 0.11),
        observed=0.08,
    )

    assert null.percentile_of_observed() == pytest.approx(0.7)


@pytest.mark.unit
def test_percentile_rejects_empty_draws() -> None:
    with pytest.raises(UninformedError, match="沒有抽樣"):
        NullDistribution(draws=(), observed=0.01).percentile_of_observed()


@pytest.mark.unit
def test_describe_states_the_caveat() -> None:
    """
    摘要必須帶上「抽樣不是獨立策略」的警語。

    與 CPCV 的 15 條路徑同一個問題：共用同一份歷史，用來算分位可以，
    用來宣稱顯著不行。
    """
    lines = NullDistribution(draws=(0.01, 0.02, 0.03), observed=0.04).describe()

    assert any("不是獨立策略" in line for line in lines)
    assert any("百分位" in line for line in lines)


@pytest.mark.unit
def test_default_draws_is_at_least_the_claude_md_baseline() -> None:
    """CLAUDE.md 的隨機對照組是 100 次；這裡不可更少"""
    assert DEFAULT_DRAWS >= 100


# ══════════════════════════════════════════════════════════════
# 稀疏度必須與被比較的策略一致
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_sparse_weights_have_exactly_k_nonzero() -> None:
    """
    `sparsity=k` 每組只能有 k 個非零權重。

    手工分數用 4~6 個特徵（動能 4、籌碼 6、均值回歸 4），所以對照組
    的稀疏度要匹配。
    """
    weights = draw_weights(32, n_draws=20, seed=5, sparsity=4)

    assert weights.shape == (32,)[0:0] + (20, 32)
    nonzero = (weights != 0).sum(axis=1)
    assert set(nonzero.tolist()) == {4}


@pytest.mark.unit
def test_sparse_draws_choose_different_feature_subsets() -> None:
    """不同抽樣要選到不同的特徵子集，否則對照組沒有變異"""
    weights = draw_weights(32, n_draws=30, seed=6, sparsity=4)

    subsets = {frozenset(np.flatnonzero(row).tolist()) for row in weights}
    assert len(subsets) > 10, f"30 組只選出 {len(subsets)} 種子集"


@pytest.mark.unit
def test_dense_null_is_weaker_than_sparse_by_construction() -> None:
    """
    密集對照組的分數比稀疏的更接近雜訊——這是它太弱的機制。

    32 個獨立分位特徵、隨機正負權重，加總會互相抵銷，與任一單一特徵
    的相關性遠低於只用 4 個特徵的組合。

    手算不可行，所以用合成資料量相關性：稀疏組合與其第一個非零特徵的
    相關性必須明顯高於密集組合與同一特徵的相關性。
    """
    rng = np.random.default_rng(11)
    ranks = pd.DataFrame(
        rng.uniform(0, 1, size=(400, 32)),
        columns=[f"f{i}" for i in range(32)],
        index=[f"S{i}" for i in range(400)],
    )
    dense = draw_weights(32, n_draws=40, seed=7)
    sparse = draw_weights(32, n_draws=40, seed=7, sparsity=4)

    def mean_abs_top_corr(weights: np.ndarray) -> float:
        values = []
        for row in weights:
            scores = random_weight_scores(ranks, row)
            leading = int(np.argmax(np.abs(row)))
            values.append(abs(scores.corr(ranks.iloc[:, leading])))
        return float(np.mean(values))

    assert mean_abs_top_corr(sparse) > mean_abs_top_corr(dense) * 1.5


@pytest.mark.unit
def test_sparsity_out_of_range_is_rejected() -> None:
    with pytest.raises(UninformedError, match="sparsity"):
        draw_weights(32, sparsity=0)
    with pytest.raises(UninformedError, match="sparsity"):
        draw_weights(32, sparsity=33)
