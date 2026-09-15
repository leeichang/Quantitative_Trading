"""
外部驗證工具（來自 HKUDS/Vibe-Trading 的 quantlib，MIT）

## 為什麼引入外部程式碼

兩個問題自己重寫的成本遠高於借用：

```
CPCV        combinatorial purged cross-validation
PBO / DSR   多重測試校正（CLAUDE.md 要求但一直沒實作）
```

## CPCV 解決的是哪個問題

`03_待辦與改進方向.md` 開頭那張表：v6 → v7 只補了 23 期缺漏的市值快照，
策略參數完全沒動，均值回歸 × 60 從 +954.87% 掉到 +123.93%（−831 pp），
而等權對照只動了 13 pp。

當時的結論「不是 bug，是路徑相依」是對的，但**沒有辦法量化它**——
單一 walk-forward 只給一條路徑，你無從知道那條路徑有多大代表性。

CPCV 用同一份資料生出 C(n_groups, n_test_groups) 條路徑，給出報酬的
**分布**而不是點估計。實測本專案規模（370 個決策期、標籤跨度 12 期）
得到 15 條路徑。

⚠️ **15 條路徑不是 15 個獨立樣本。** 它們共用同一份歷史，只是切法不同。
它回答的是「這條策略對切分方式有多敏感」，不是「多了 14 倍證據」。
**用它否定比用它肯定可靠得多。**

## 明確不要用的

| 函式 | 為什麼 |
|---|---|
| `purged_kfold_splits` | 會拿測試段**之後**的資料訓練，違反禁令 5 |
| `timeseries.bootstrap_sharpe` | 逐點 IID 重抽，對自相關報酬會低估區間寬度（所以沒有引入 timeseries.py） |

## 維護方式

`crossvalidation.py` 與 `multipletesting.py` **原封不動**，只在檔頭加了
來源註記。要升級時直接 diff 上游。

本專案的手算值測試在 `tests/test_external_validation.py`——
**不可拿上游的測試當作本專案的測試**（禁令 10）。
"""

from __future__ import annotations

SOURCE_REPO = "https://github.com/HKUDS/Vibe-Trading"
SOURCE_COMMIT = "a5b79422f9f0c6b23512159dfefc8031498df5b8"
"""升級時拿這個 commit 跟上游做 diff"""
