"""
ETF 候選標的

## 為什麼要有這個模組

`08_動能突破N10的樣本外結果.md` 的核心發現：策略在樣本外**大幅輸給
0050 買進持有**，Sharpe 也輸。

⚠️ 那份文件的數字已被兩次修正取代，而本模組的存在理由不受影響——
「輸 0050」在每個版本都成立：

```
              原文        本模組的逐檔成本更正後
動能突破     +151.65%    +138.83%   Sharpe 1.08
0050        +236.44%    +242.15%   Sharpe 1.90   ← 每個版本都贏
隨機 p95    +129.57%    +120.97%   ← 2026-09-18 撤回，弱虛無
```

原文還寫「超出隨機 10 檔的 95% 分位」。那個門檻來自弱虛無，
規格已改成嚴格虛無，該句降級為「量不出差別」。見
../../../qlib-tw-trader/docs/原理說明/2026-09-18_超過隨機95分位這句話該退休.md

直接原因是結構性的，不是模型不行：

```
標的池   market_cap 時點池前 150 名（個股）
0050     不在池內 → 策略永遠不可能選它
```

2024-2026 是 AI 集中行情，0050 的權重高度集中在台積電，等於單押最強的
那一檔。**如果那個區間的最佳決策是「買 0050 不動」，策略在設計上就沒有
機會做出那個決策。**

這個模組讓 ETF 成為候選，然後讓策略自己決定。

## 為什麼不寫進 universe_history

那張表是市值排名的季度快照，`metric` 欄位是市值。ETF 沒有「市值排名」
的意義，硬塞會讓表的語意變混。

ETF 是**結構上不同的一類**：

```
個股   排名進出、可能下市、零股滑價 0.3~0.4%（禁令 4）
ETF    永遠可交易、跳動單位細 10 倍、滑價 0.05~0.1%（實證）
```

所以用獨立常數表達，不混進排名快照。

## 為什麼用白名單而不是代號規則

台股有數百檔 ETF。用「代號開頭 00」會一次拉進大量流動性不足的標的，
而且它們的滑價分層沒有實證依據。

白名單讓「哪些 ETF 可交易」成為**明確的決定**，不是代號的副作用。
要加新的 ETF 就必須先實證它的跳動單位與成交量。

## ⚠️ 加了 ETF 需要新的 OOS 區間

2024-01 ~ 2026-08 已經在 `momentum_top10_h40@oos-2026-09-16` 用掉了
（禁令 6）。**這個改動不可以在那個區間上重跑驗證**——那會變成「看過
結果之後調整設計再重跑」，正是禁令 6 要防的事。
"""

from __future__ import annotations

from collections.abc import Sequence

ETF_CANDIDATES: tuple[str, ...] = ("0050", "0051", "0056")
"""
可納入候選的 ETF。

就是 `validation/benchmarks.py` 的 `ETF_BENCHMARKS` 那三檔——它們的資料
本來就在，而且已經被當成「策略要打敗的東西」。現在讓策略可以直接持有。

要新增必須先實證：跳動單位、成交量、以及對應的滑價分層。
"""

_ETF_SET = frozenset(ETF_CANDIDATES)


def is_etf(stock_id: str) -> bool:
    """
    是否為納入名單的 ETF。

    Args:
        stock_id: 標的代號

    Returns:
        在白名單內回 `True`

    **不用代號規則判定。** `00878` 是 ETF 但不在名單內，回 `False`——
    它的滑價分層沒有實證依據，誤判成 ETF 會用錯成本。
    """
    return stock_id in _ETF_SET


def merge_etf_candidates(
    ranked: Sequence[str], *, include: bool = True
) -> tuple[str, ...]:
    """
    把 ETF 接在排名標的池後面。

    Args:
        ranked: 依市值排名的個股代號（順序有意義）
        include: `False` 時逐筆回傳原輸入，用於重現沒有 ETF 的舊結果

    Returns:
        併入後的候選清單

    Raises:
        TypeError: `ranked` 不是序列

    **ETF 接在後面而不是插進排名裡**——排名是市值的，ETF 沒有那個維度。
    已存在的代號不重複加入；重複會讓同一檔被算兩次權重。
    """
    if isinstance(ranked, str) or not isinstance(ranked, Sequence):
        raise TypeError(f"ranked 必須是字串序列，得到 {type(ranked).__name__}")
    if not include:
        return tuple(ranked)
    existing = set(ranked)
    return tuple(ranked) + tuple(
        sid for sid in ETF_CANDIDATES if sid not in existing
    )
