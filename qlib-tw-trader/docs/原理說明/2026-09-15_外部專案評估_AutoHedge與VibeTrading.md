# 外部專案評估：AutoHedge 與 Vibe-Trading

- 日期：2026-09-15
- 來源：`https://github.com/The-Swarm-Corporation/AutoHedge`、`https://github.com/HKUDS/Vibe-Trading`
- clone 位置：`external/`（已加入 `.gitignore`，不納入版控）

---

## 結論先講

| 專案 | 判定 | 理由 |
|---|---|---|
| AutoHedge | **不採用** | 解決的不是我的問題。633 行、零測試、LLM 下單、加密貨幣 |
| Vibe-Trading 的 `quantlib` | **採用三個模組** | 直接命中我最大的兩個問題，而且實測可用 |
| Vibe-Trading 其餘部分 | 不採用 | Web 前端 / Docker / 多市場引擎，與本專案無關 |

**最重要的一句**：Vibe-Trading 的 `combinatorial_purged_splits`（CPCV）
把「重跑一次結果就擺盪數百個百分點」從**缺陷**變成**可量測的統計量**。

---

## 為什麼 AutoHedge 不適用

```
autohedge/  633 行   prompts.py 202 行、cli.py 230 行
測試        0 個
架構        Director / Quant / Risk / Execution 四個 LLM agent 互相對話
標的        Solana，Coinbase 開發中
```

它是「用 LLM 產生交易論點並自動下單」。我的系統**明確不下單**
（使用者需求：只提投資建議），而且我的問題全部是統計問題：

```
訊號/成交比 435:1     ← 門檻設計問題
重跑擺盪數百 pp        ← 估計量變異問題
有效樣本 15~30        ← 統計力問題
OOS 被看過 7 次        ← 實驗紀律問題
```

**沒有一個是「缺少 LLM agent」造成的。** 加一層 LLM 在上面，只會讓不
穩定的估計量多一層不可重現的包裝。

⚠️ 它的 `logs/trades_*.csv` 是作者的實際交易紀錄，不是回測結果——
沒有任何樣本外驗證、成本模型或統計檢定。

---

## Vibe-Trading 的 `quantlib`：可以直接用

```
agent/src/quantlib/     12,076 行
agent/tests/quantlib/    8,053 行測試
授權                     MIT
依賴                     numpy / pandas / scipy  ← 與本專案完全相同
```

模組自己的說明寫得很清楚，風格與本專案一致（Args / Returns / Raises、
型別註記、邊界驗證、解釋「為什麼」而不只是「做什麼」）。

### 1. `crossvalidation.py` — 解決「路徑相依」與「樣本太少」

實測（用本專案的實際規模：370 個決策期、標籤跨度 12 期）：

```
CPCV：15 條路徑（單一 walk-forward 只有 1 條）
  訓練 231｜測試 123｜purged 12｜embargoed 4
  訓練/測試重疊：False
```

**這正是 `03_待辦與改進方向.md` 開頭那張表的解法。**

v6 → v7 只補了 23 期缺漏的市值快照，策略參數完全沒動，結果：

```
均值回歸 × 60    +954.87% → +123.93%   −831 pp
均值回歸 × 120   +294.62% → +802.42%   +508 pp
```

我當時的結論是「不是 bug，是路徑相依」——對，但**沒有辦法量化它**。
單一 walk-forward 只給一條路徑，你無從知道那條路徑有多大代表性。

CPCV 用同一份資料生出 15 條路徑，直接給出報酬的**分布**而不是點估計。
「擺盪數百 pp」會變成一個可以寫進報告的區間，不是重跑才發現的意外。

另外兩個函式也用得上：

| 函式 | 用途 |
|---|---|
| `purged_walk_forward_splits` | 實測 4 folds、每 fold 剛好 purge 12 期，**與本專案的 `embargo_periods(60, 5) = 12` 完全一致** |
| `detect_boundary_leakage` | 邊界洩漏偵測，可當作既有 `walk_forward.py` 的交叉驗證 |

⚠️ 它的 `purged_kfold_splits` **不要用**。K-fold 會拿測試段**之後**的資料
訓練，對估計泛化能力是對的，但違反本專案禁令 5（滾動訓練，只用預測期
之前的資料）。模組自己的 docstring 也點明了這件事。

### 2. `multipletesting.py` — 解決「掃了 N 組沒做校正」

CLAUDE.md 要求「掃 N 組參數時必須回報 DSR 或 PBO，PBO > 0.5 判定過擬合」。
**這條我一直沒實作。** 他們有，而且實測會分辨：

```
PBO（20 組純雜訊策略）      0.329
PBO（其中 1 組有真優勢）    0.029   ← 明顯下降，有鑑別力
```

DSR 用本專案的真實數字跑：

```python
deflated_sharpe_ratio(0.9, n_trials=20, trial_sharpe_std=0.5, n_observations=30)

observed_sharpe          0.90
expected_maximum_sharpe  0.95     ← 純靠運氣、掃 20 組能拿到的最佳值
deflated_sharpe_ratio    0.41
survives                 False
```

**讀法**：掃 20 組參數時，光靠運氣就能期望拿到 Sharpe 0.95。
我最好的組合是 0.90——**比運氣還低**。

這與既有結論（「不建議推播」）一致，但現在有一個能寫進報告的統計量，
而不只是「DSR 0.1015 未達顯著」這種自己算的數字。

### 3. 順帶：`impact.py`（281 行）市場衝擊模型

目前的滑價是固定分層（0050 = 0.3%、0051 = 0.4%）。他們有依成交量的
衝擊模型。**優先度低**——40 萬資金做零股，衝擊成本可以忽略，固定分層
已經夠保守。列在這裡是為了記錄評估過。

---

## 不能用的部分：他們的 bootstrap 是 IID

任務 F 要的是 block bootstrap。他們的 `timeseries.bootstrap_sharpe` 是
**逐點重抽**：

```python
bootstrap_stats[i] = float(statistic_func(sample[rng.integers(0, n, size=n)]))
#                                                ^^^^^^^^^^^^^^^^^^^^^^^^^^
#                                                每一筆獨立抽，打散了時序
```

報酬序列有自相關，逐點重抽會**低估**信賴區間寬度——區間看起來比實際窄，
結論比實際有把握。這正是任務 F 要避開的錯誤。

**任務 F 仍然要自己寫**，區塊長度取一個持有期。可以參考他們的
`bootstrap_statistic` 介面設計（回傳 `point_estimate` / `ci_lower` /
`ci_upper` / `bootstrap_std` 的 dict），但重抽邏輯要換成區塊。

---

## 對四個問題的實際影響

| 問題 | Vibe-Trading 有幫助嗎 | 說明 |
|---|---|---|
| 訊號/成交比 435:1 | ❌ 沒有 | 這是本專案的門檻設計問題，外部程式碼幫不上 |
| 重跑擺盪數百 pp | ✅ **直接解決** | CPCV 把它變成可量測的分布 |
| 有效樣本 15~30 | ✅ **部分解決** | 15 條路徑 ≠ 15 倍獨立樣本，但比 1 條有資訊量得多 |
| OOS 被看過 7 次 | ✅ **部分解決** | PBO / DSR 能量化「看了 7 次」的代價 |
| 真實 0050/0051 成分股 | ❌ 沒有 | 他們做的是 A 股 / 美股 / 加密貨幣 |

⚠️ **CPCV 的 15 條路徑不是 15 個獨立樣本。** 它們共用同一份歷史，只是
切法不同。它給的是「這條策略對切分方式有多敏感」，不是「多了 14 倍證據」。
用它來**否定**（變異太大 → 結論不可靠）比用來**肯定**可靠得多。

---

## 建議怎麼接

**不要整包 vendor 進來。** 12,076 行裡我只需要三個模組，而且要保留
自己的測試（比對手算值，禁令 10）。

```
1. 複製 crossvalidation.py + multipletesting.py 到
   taiwan_quant/validation/external/，保留 MIT 授權標頭與來源註記
2. 為「本專案實際會呼叫的函式」寫自己的測試，比對手算值——
   不可拿他們的測試當作我的測試
3. 先用 purged_walk_forward_splits 與既有 walk_forward.py 交叉比對，
   兩邊 fold 切法應一致（實測 purge 都是 12 期）。對不上就是有一邊錯了
4. 再接 CPCV 與 PBO
```

第 3 步是最便宜的檢查：**兩套獨立實作的切分結果應該一致。**
不一致的話，就抓到一個我自己看不出來的 bug。

---

## 這次評估本身的教訓

我原本的任務 F 是「自己寫 block bootstrap 估信賴區間」。查完發現：

- 他們的 bootstrap **不能用**（IID，方向錯了）
- 但他們的 **CPCV 比 bootstrap 更適合我的問題**

bootstrap 是在「一條權益曲線」上估變異；CPCV 是在「切分方式」上估變異。
我真正的問題是**後者**——v6/v7 那張表證明了，變的不是隨機性，是資料
完整性造成的路徑改變。

**任務 F 的優先度應該下調，CPCV 上調。** 這個結論是查外部專案查出來的，
不是原本規劃裡有的。

---

⚠️ 本文件為工程評估，不構成投資建議。
