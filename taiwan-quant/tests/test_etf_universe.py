"""
ETF 納入標的池的測試

## 為什麼要做這件事

`08_動能突破N10的樣本外結果.md`：策略在 OOS 拿到 +151.65%，超出隨機
10 檔的 95% 分位（+129.57%），但**大幅輸給 0050 買進持有（+236.44%）**。

直接原因是結構性的，不是模型不行：

```
標的池   market_cap 時點池前 150 名（個股）
0050     不在池內 → 策略永遠不可能選它
```

2024-2026 是 AI 集中行情，0050 的權重高度集中在台積電，等於單押最強
的那一檔。**如果那個區間的最佳決策是「買 0050 不動」，策略在設計上
就沒有機會做出那個決策。**

這個模組讓 ETF 成為候選，然後讓策略自己決定要不要買。

## 為什麼不直接寫進 universe_history

`universe_history` 是市值排名的季度快照。ETF 沒有「市值排名」的意義，
硬塞進去會讓那張表的語意變混（它的 `metric` 欄位是市值）。

ETF 是**結構上不同的一類**：它永遠可交易、永遠在池內、成本分層不同
（見 `config/costs.py` 的 `Tier.ETF_ODD` / `ETF_WHOLE`）。所以用一個
獨立的常數表達，而不是混進排名快照。

## ⚠️ 加了 ETF 就需要新的 OOS 區間

2024-01 ~ 2026-08 已經在 `momentum_top10_h40@oos-2026-09-16` 用掉了
（禁令 6）。**這個改動不可以在那個區間上重跑驗證。**
"""

from __future__ import annotations

import pytest

from taiwan_quant.config.costs import Tier, resolve_tier
from taiwan_quant.data.etf_universe import (
    ETF_CANDIDATES,
    is_etf,
    merge_etf_candidates,
)


# ══════════════════════════════════════════════════════════════
# ETF 名單
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_etf_candidates_are_the_three_required_benchmarks() -> None:
    """
    CLAUDE.md 要求每份報告並列 0050 買進持有，而 0051／0056 是既有的
    對照組（`validation/benchmarks.py` 的 `ETF_BENCHMARKS`）。

    把對照組加進候選是最小的改動：它們的資料本來就在，而且已經被當成
    「策略要打敗的東西」——現在讓策略可以直接持有它們。
    """
    assert ETF_CANDIDATES == ("0050", "0051", "0056")


@pytest.mark.unit
def test_is_etf_recognises_only_the_listed_codes() -> None:
    """
    用白名單而不是「代號開頭 00」的規則。

    台股有數百檔 ETF，多數流動性遠不如這三檔，而且滑價分層沒有實證依據。
    白名單讓「哪些 ETF 可交易」成為明確的決定，不是代號的副作用。
    """
    assert is_etf("0050")
    assert is_etf("0056")
    assert not is_etf("2330")
    assert not is_etf("00878")      # 存在但未納入，不可靠代號規則誤判


# ══════════════════════════════════════════════════════════════
# 併入標的池
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_merge_appends_etfs_without_touching_the_ranked_names() -> None:
    """
    手算：市值池 3 檔 + 3 檔 ETF = 6 檔，而且**排名順序不變**。

    ETF 接在後面而不是插進排名裡——排名是市值的，ETF 沒有那個維度。
    """
    ranked = ("2330", "2317", "2454")

    merged = merge_etf_candidates(ranked)

    assert merged[:3] == ranked
    assert set(merged[3:]) == set(ETF_CANDIDATES)
    assert len(merged) == 6


@pytest.mark.unit
def test_merge_does_not_duplicate_when_an_etf_is_already_ranked() -> None:
    """
    0050 理論上不會出現在市值池（它不是個股），但若資料有異常，
    併入不可產生重複——重複會讓同一檔被算兩次權重。
    """
    ranked = ("2330", "0050")

    merged = merge_etf_candidates(ranked)

    assert len(merged) == len(set(merged))
    assert merged.count("0050") == 1


@pytest.mark.unit
def test_merge_preserves_input_when_disabled() -> None:
    """
    `include=False` 時必須逐筆不變——這是為了讓既有結果可重現。

    OOS 區間 2024-01 ~ 2026-08 已用掉，那次沒有 ETF。要重現它就必須
    能關掉這個功能。
    """
    ranked = ("2330", "2317")

    assert merge_etf_candidates(ranked, include=False) == ranked


@pytest.mark.unit
def test_merge_rejects_non_sequence() -> None:
    with pytest.raises(TypeError):
        merge_etf_candidates("2330")        # type: ignore[arg-type]


# ══════════════════════════════════════════════════════════════
# 與成本分層的接合
# ══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_etf_positions_route_to_the_etf_cost_tier() -> None:
    """
    手算：0050 約 107.70 元 → 一張 107,700 元。

    N=10 時每檔 40,000 元 → 買不起整張 → ETF 零股（0.1%）
    N=3 時每檔 133,333 元 → 買得起 → ETF 整股（0.05%）

    **不可誤用 `Tier.LARGE`（0.3%）**——那是 0050「成分股」的零股滑價，
    不是 0050 本身。ETF 的跳動單位細 10 倍，用 0.3% 會高估三倍。
    """
    assert resolve_tier(price=107.70, amount=40_000,
                        is_etf=is_etf("0050")) is Tier.ETF_ODD
    assert resolve_tier(price=107.70, amount=133_333,
                        is_etf=is_etf("0050")) is Tier.ETF_WHOLE
    # 個股走原本的分層
    assert resolve_tier(price=107.70, amount=40_000,
                        is_etf=is_etf("2330")) is Tier.LARGE
