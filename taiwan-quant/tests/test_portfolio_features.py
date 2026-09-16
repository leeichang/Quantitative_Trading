"""
投組約束輸入的測試

## 為什麼每個值都手算

CLAUDE.md：「特徵與標記的測試必須**比對手算值**，不可拿程式輸出反填
預期值。」所以合成資料刻意選成閉式解：

```
beta          個股報酬 = 2 × 市場報酬   →   Cov(2m,m)/Var(m) = 2.0 恰好
correlation   同上                      →   corr = 1.0 恰好
ATR           3 根 K、period=2          →   可以逐項寫出 True Range
```
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from taiwan_quant.ranking.constraints import (
    ConstraintLimits,
    DEFAULT_LIMITS,
    RejectReason,
    greedy_pick,
)
from taiwan_quant.ranking.portfolio_features import (
    UNCLASSIFIED,
    PortfolioFeatureError,
    RankedCandidate,
    atr_percentiles,
    betas,
    build_candidates,
    load_industries,
    return_correlations,
)

# ══════════════════════════════════════════════════════════════
# ATR 分位
# ══════════════════════════════════════════════════════════════


def _bars(rows: list[tuple[float, float, float]]) -> pd.DataFrame:
    """(high, low, close) 轉成日 K，index 為連續交易日"""
    index = pd.date_range("2020-01-01", periods=len(rows), freq="D")
    return pd.DataFrame(rows, columns=["high", "low", "close"], index=index)


@pytest.mark.unit
def test_atr_percentile_normalises_by_price() -> None:
    """
    ATR 必須先除以股價才能做橫斷面比較。

    手算，period=2：

        高波動 A   TR = [NaN, 1, 2]        ATR(2) = 1.5   close 11
                   ATR% = 1.5 / 11 = 0.13636
        低波動 B   TR = [NaN, 0.1, 0.1]    ATR(2) = 0.1   close 10
                   ATR% = 0.1 / 10 = 0.01

    B 的**絕對**股價與 A 接近，所以這題不是在測正規化。真正的檢查是
    下一個測試：股價差 10 倍但波動比例相同時，分位必須相同。
    """
    by_stock = {
        "A": _bars([(10, 9, 10), (11, 10, 11), (12, 10, 11)]),
        "B": _bars([(10, 9.9, 10), (10.1, 10, 10), (10.1, 10, 10)]),
    }
    day = by_stock["A"].index[-1]

    result = atr_percentiles(by_stock, day, ("A", "B"), period=2)

    assert result == {"B": 0.5, "A": 1.0}


@pytest.mark.unit
def test_atr_percentile_ties_when_volatility_ratio_matches() -> None:
    """
    股價 790 元與 27.9 元、波動比例相同 → 分位必須相同。

    若忘記除以股價，高價股的 ATR 絕對值天生大一個數量級，分位就變成
    在排序股價。這正是第一筆帳本裡 1560（790 元）與 2801（27.9 元）
    並存時會踩到的坑。
    """
    # 兩檔的 high/low/close 都是同一組數字乘以價格倍率 → ATR% 完全相同
    pattern = [(1.00, 0.98, 1.00), (1.02, 1.00, 1.01), (1.03, 1.00, 1.01)]
    by_stock = {
        "HIGH_PRICE": _bars([(h * 790, lo * 790, c * 790) for h, lo, c in pattern]),
        "LOW_PRICE": _bars([(h * 27.9, lo * 27.9, c * 27.9) for h, lo, c in pattern]),
    }
    day = by_stock["HIGH_PRICE"].index[-1]

    result = atr_percentiles(by_stock, day, ("HIGH_PRICE", "LOW_PRICE"), period=2)

    assert result["HIGH_PRICE"] == pytest.approx(result["LOW_PRICE"])


@pytest.mark.unit
def test_atr_percentile_ignores_bars_after_decision_day() -> None:
    """禁令 1：決策日之後的 K 線不可影響分位"""
    quiet_then_wild = _bars(
        [(10, 9.9, 10), (10.1, 10, 10), (10.1, 10, 10), (50, 5, 10)]
    )
    steady = _bars([(10, 9.9, 10), (10.1, 10, 10), (10.1, 10, 10), (10.1, 10, 10)])
    by_stock = {"A": quiet_then_wild, "B": steady}
    day = quiet_then_wild.index[2]  # 第 4 根暴漲 K 在決策日之後

    result = atr_percentiles(by_stock, day, ("A", "B"), period=2)

    # 截斷到 day 為止，兩檔的 K 線完全相同 → 分位必須相同
    assert result["A"] == pytest.approx(result["B"])


# ══════════════════════════════════════════════════════════════
# beta
# ══════════════════════════════════════════════════════════════


def _closes_from_returns(columns: dict[str, np.ndarray]) -> pd.DataFrame:
    """報酬序列轉累積價格，起始價 100"""
    n = len(next(iter(columns.values())))
    index = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    return pd.DataFrame(
        {name: 100.0 * np.cumprod(np.r_[1.0, 1.0 + r]) for name, r in columns.items()},
        index=index,
    )


@pytest.mark.unit
def test_beta_is_exactly_two_when_returns_are_doubled() -> None:
    """
    個股報酬 = 2 × 市場報酬 → beta = Cov(2m, m) / Var(m) = 2·Var(m)/Var(m) = 2.0

    這是閉式解，不是程式輸出反填。
    """
    rng = np.random.default_rng(20260916)
    market = rng.normal(0.0, 0.01, 200)
    closes = _closes_from_returns(
        {"0050": market, "DOUBLE": 2.0 * market, "HALF": 0.5 * market}
    )
    day = closes.index[-1]

    result = betas(closes, day, ("DOUBLE", "HALF"))

    assert result["DOUBLE"] == pytest.approx(2.0)
    assert result["HALF"] == pytest.approx(0.5)


@pytest.mark.unit
def test_beta_requires_benchmark_in_frame() -> None:
    """
    基準缺失要拋錯，不可靜默回空。

    靜默回空會讓每一檔的 beta 都變成缺值，`build_candidates` 把缺值填
    `inf`，結果「至少 1 支 defensive」永遠無法滿足——而且沒有任何訊息
    說明原因。
    """
    closes = _closes_from_returns({"2330": np.full(200, 0.001)})
    with pytest.raises(PortfolioFeatureError, match="0050"):
        betas(closes, closes.index[-1], ("2330",))


@pytest.mark.unit
def test_beta_skips_names_with_too_little_overlap() -> None:
    """共同交易日不足的不回報，而不是用少數點硬算一個不穩定的 beta"""
    rng = np.random.default_rng(1)
    market = rng.normal(0.0, 0.01, 200)
    closes = _closes_from_returns({"0050": market, "NEW": 2.0 * market})
    closes.loc[closes.index[:150], "NEW"] = np.nan  # 只剩 50 筆有效

    result = betas(closes, closes.index[-1], ("NEW",))

    assert "NEW" not in result


# ══════════════════════════════════════════════════════════════
# 相關係數
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_correlation_is_one_for_proportional_returns() -> None:
    """報酬成正比 → 相關係數恰好 1.0；只存一個方向"""
    rng = np.random.default_rng(7)
    base = rng.normal(0.0, 0.01, 100)
    closes = _closes_from_returns({"A": base, "B": 3.0 * base})
    day = closes.index[-1]

    result = return_correlations(closes, day, ("A", "B"))

    assert result == pytest.approx({("A", "B"): 1.0})
    assert ("B", "A") not in result


@pytest.mark.unit
def test_correlation_ignores_prices_after_decision_day() -> None:
    """禁令 1：決策日之後的價格不可影響相關係數"""
    rng = np.random.default_rng(11)
    base = rng.normal(0.0, 0.01, 100)
    independent = rng.normal(0.0, 0.01, 100)
    closes = _closes_from_returns({"A": base, "B": 3.0 * base})
    day = closes.index[80]

    truncated = return_correlations(closes, day, ("A", "B"))
    # 決策日之後改成完全無關，結果必須不變
    closes.loc[closes.index[81:], "B"] = _closes_from_returns(
        {"B": independent}
    )["B"].values[81:]
    after = return_correlations(closes, day, ("A", "B"))

    assert truncated[("A", "B")] == pytest.approx(after[("A", "B")])


# ══════════════════════════════════════════════════════════════
# 產業
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_load_industries_omits_blank_classification(tmp_path: Path) -> None:
    """
    缺分類的股票不進回傳值。

    在這裡填 "未分類" 會讓它變成一個真的產業，然後所有缺分類的股票
    互相擠掉名額——那是資料缺失造成的選股差異，不是風控決定。
    缺值怎麼處理由 `build_candidates` 明確負責。
    """
    import sqlite3

    db = tmp_path / "master.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE stock_master (stock_id TEXT, industry TEXT)")
    con.executemany(
        "INSERT INTO stock_master VALUES (?, ?)",
        [("2801", "金融保險"), ("9999", None), ("8888", "  "), ("1303", "塑膠工業")],
    )
    con.commit()
    con.close()

    assert load_industries(db) == {"2801": "金融保險", "1303": "塑膠工業"}


# ══════════════════════════════════════════════════════════════
# 候選組裝：缺值方向
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_missing_values_never_earn_an_advantage() -> None:
    """
    ATR 缺 → 0.0（低波動，不佔高波動額度）
    beta 缺 → inf（非 defensive，不能用來滿足 defensive 要求）

    兩者方向相反，但規則一致：**缺值不得取得好處**。

    反過來各自會怎麼錯：
        ATR 缺值填 1.0  → 被當高波動而佔掉額度（過度懲罰，且憑空）
        beta 缺值填 0.0 → 假裝是 defensive，放過真正的集中風險
    """
    candidates = build_candidates(
        ordered=["A"],
        scores={"A": 0.9},
        industries={},
        atr_pct={},
        beta_map={},
    )

    only = candidates[0]
    assert only.industry == UNCLASSIFIED
    assert only.atr_percentile == 0.0
    assert only.is_high_volatility is False
    assert only.beta == float("inf")
    assert only.is_defensive is False


@pytest.mark.unit
def test_build_candidates_preserves_order() -> None:
    """
    順序必須原樣保留——`greedy_pick` 依序貪婪挑選，重排等於改變選股結果。
    """
    ordered = ["C", "A", "B"]
    result = build_candidates(
        ordered=ordered,
        scores={"A": 0.1, "B": 0.2, "C": 0.3},
        industries={"A": "金融保險", "B": "金融保險", "C": "塑膠工業"},
        atr_pct={"A": 0.5, "B": 0.9, "C": 0.1},
        beta_map={"A": 0.8, "B": 1.5, "C": 0.9},
    )

    assert [c.stock_id for c in result] == ordered


# ══════════════════════════════════════════════════════════════
# ConstraintLimits
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_default_limits_match_claude_md() -> None:
    """
    預設值必須是 CLAUDE.md 的原始規格。

    放寬是有代價的決定，必須明確傳入新的 `ConstraintLimits` 並在文件裡
    寫明依據——不可靜默改預設值讓回測看起來變好。
    """
    assert DEFAULT_LIMITS == ConstraintLimits(
        top_n=3,
        max_same_industry=2,
        max_high_volatility=1,
        max_correlation=0.7,
        require_defensive=True,
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"top_n": 0}, "top_n"),
        ({"max_same_industry": 0}, "max_same_industry"),
        ({"max_high_volatility": -1}, "max_high_volatility"),
        ({"max_correlation": 0.0}, "max_correlation"),
        ({"max_correlation": 1.5}, "max_correlation"),
    ],
)
def test_limits_reject_nonsense(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        ConstraintLimits(**kwargs)


@pytest.mark.unit
def test_industry_cap_scales_with_top_n() -> None:
    """
    同一組候選、同一個 `max_same_industry=2`，N 從 3 變 10 時被擋掉的
    數量完全不同——這就是「Top 3 的約束不能無腦套到 N=10」的具體形狀。

    6 檔金融 + 4 檔非金融，依分數交錯排列（金融分數較高）。
    """
    financials = [f"F{i}" for i in range(6)]
    others = [f"O{i}" for i in range(4)]
    ordered = financials + others
    candidates = build_candidates(
        ordered=ordered,
        scores={sid: 1.0 - i * 0.01 for i, sid in enumerate(ordered)},
        industries={
            **{sid: "金融保險" for sid in financials},
            **{sid: f"產業{i}" for i, sid in enumerate(others)},
        },
        atr_pct={sid: 0.1 for sid in ordered},
        beta_map={sid: 0.5 for sid in ordered},
    )
    # 全部兩兩相關 0.0，讓這個測試只檢驗產業約束
    correlations = {
        (a, b): 0.0
        for i, a in enumerate(ordered)
        for b in ordered[i + 1:]
    }

    picked_3, _ = greedy_pick(candidates, correlations, ConstraintLimits(top_n=3))
    picked_10, _ = greedy_pick(candidates, correlations, ConstraintLimits(top_n=10))

    # N=3：前 2 檔金融 + 第 1 檔非金融，只損失 1 個名次
    assert [c.stock_id for c in picked_3] == ["F0", "F1", "O0"]
    # N=10：4 檔金融被擋掉，而候選池只剩 4 檔非金融 → 湊不滿 10 檔
    assert [c.stock_id for c in picked_10] == ["F0", "F1", "O0", "O1", "O2", "O3"]
    assert len(picked_10) == 6, "湊不滿 N 是刻意的：寧可少推，不違反風控"
