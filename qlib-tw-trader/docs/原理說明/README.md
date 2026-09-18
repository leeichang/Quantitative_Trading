# 原理說明索引

每份文件說明「為什麼這樣設計」，不只是「做了什麼」。檔名含日期。

---

## 2026-09-12

依閱讀順序排列（後面的會引用前面的）：

| # | 文件 | 對應程式 | 核心問題 |
|---|---|---|---|
| 1 | [交易成本模型原理](2026-09-12_交易成本模型原理.md) | `taiwan-quant/taiwan_quant/config/costs.py` | 一個不下單的系統為什麼需要成本模型？ |
| 2 | [Triple-Barrier 標記原理](2026-09-12_Triple-Barrier標記原理.md) | `labeling/triple_barrier.py` | 為什麼「預測股價」是個爛問題？ |
| 3 | [柵欄寬度推導原理](2026-09-12_柵欄寬度推導原理.md) | `labeling/barrier_width.py` | 目標價與停損價為什麼不可以拍一個數字？ |
| 4 | [Look-ahead 物理截斷掃描原理](2026-09-12_Look-ahead物理截斷掃描原理.md) | `validation/lookahead.py` | 為什麼靜態掃描抓不到最隱蔽的洩漏？ |
| 5 | [標的池與 Survivorship 原理](2026-09-12_標的池與Survivorship原理.md) | `data/constituents.py`、`universe_history.py`、`market_cap.py` | 用今天的名單回測歷史錯在哪？ |
| 6 | [特徵工程原理](2026-09-12_特徵工程原理.md) | `features/technical.py`、`chips.py` | 為什麼只做 32 個特徵而不是 303 個？ |
| 7 | [週頻回測與換手率原理](2026-09-12_週頻回測與換手率原理.md) | `backtest/engine.py` | 為什麼換手率必須是一級輸出？ |
| 8 | [多重測試校正與 IC 監控原理](2026-09-12_多重測試校正與IC監控原理.md) | `validation/stats.py` | 試越多組參數，最好的那組為什麼越不可信？ |
| 9 | [機率校準原理](2026-09-12_機率校準原理.md) | `validation/calibration.py` | **模型輸出的分數為什麼不是機率？** |
| 10 | [選股與部位規模原理](2026-09-12_選股與部位規模原理.md) | `ranking/portfolio.py` | 為什麼湊滿 3 檔不是目標？ |
| 11 | [Telegram 推播與端到端管線原理](2026-09-12_Telegram推播與端到端管線原理.md) | `notify/telegram.py`、`scripts/weekly_plan.py` | 整條管線串起來會發生什麼？ |
| 12 | [策略族與持有期診斷原理](2026-09-12_策略族與持有期診斷原理.md) | `strategies/families.py`、`scripts/diagnose_families.py` | 換策略族有用，還是拉長持有期有用？ |
| 13 | [移動停損與多槽位組合原理](2026-09-12_移動停損與多槽位組合原理.md) | `labeling/trailing_stop.py`、`trail_width.py`、`backtest/portfolio_sim.py` | **拿掉目標價之後，整套系統要跟著改什麼？** |

---

## 2026-09-13

| # | 文件 | 對應程式 | 核心問題 |
|---|---|---|---|
| 14 | [長歷史資料回補原理](2026-09-13_長歷史資料回補原理.md) | `data/twse_history.py`、`scripts/backfill_finmind_history.py`、`loader.load_universe_at` | **資料只有 1.5 年 OOS，怎麼拉到 7.3 年？** |
| 15 | [對照組與標的池代理原理](2026-09-13_對照組與標的池代理原理.md) | `validation/benchmarks.py`、`data/adjustment.py`、`scripts/build_universe_marketcap.py` | **怎麼驗證一個「代理」有多準？** |

> 第 14 篇推翻了第 13 篇的樂觀讀法：資料從 915 個交易日拉到 2,850 個之後，
> 移動停損在 20 日持有下是 **−68.91%**，而先前在 1.56 年多頭裡是 +372.79%。
> **那個正報酬是環境的產物，不是策略的能力。**
>
> 過程中抓到一個會污染八年資料的解析 bug（T86 欄位 2018 年從 16 欄變 19 欄，
> 用固定索引解析時 97 檔只有 1 檔吻合且不拋錯），並解掉 survivorship
> （2015 年標的池有 80/150 檔今天已不在，含日月光、矽品）。
>
> 第 15 篇補上 CLAUDE.md 明訂卻缺席五輪的 **0050 對照組**，並在過程中發現
> 還原因子的缺漏處理有 bug——0050 因 2025 年的 1:4 分割算出 84.20% 的假回撤。
> **修正後所有先前的回撤數字都要重看。** 標的池代理也從命中 72.7% 提升到 94.7%。

## 2026-09-14

| 文件 | 對應程式 | 核心問題 |
|---|---|---|
| [OOS 凍結與前推帳本原理](2026-09-14_OOS凍結與前推帳本原理.md) | `validation/walk_forward.py`、`forward_predictions.py`、`data/calendar.py`、`scripts/record_forward.py` | 為什麼 OOS 只能用一次，而前推帳本是唯一乾淨的證據來源？（禁令 6） |

## 2026-09-15

| 文件 | 對應程式 | 核心問題 |
|---|---|---|
| [CPCV 路徑分布與多重測試校正](2026-09-15_CPCV路徑分布與多重測試校正.md) | `validation/cpcv.py`、`fold_signals.py`、`validation/external/`、`scripts/diagnose_cpcv.py` | 單一回測曲線為什麼不夠？PBO 與 DSR 各回答什麼問題？ |
| [平手排序偏差與 E3 的 CPCV 結果](2026-09-15_平手排序偏差與E3的CPCV結果.md) | `ranking/tie_break.py`、`scripts/diagnose_cpcv.py --scheme E3` | 分數相同時的排序方式，會不會自己造出優勢？ |
| [外部專案評估：AutoHedge 與 Vibe-Trading](2026-09-15_外部專案評估_AutoHedge與VibeTrading.md) | — | 現成專案能不能直接用？缺的是哪一塊？ |

## 2026-09-16

| 文件 | 對應程式 | 核心問題 |
|---|---|---|
| [下市股最後有價日結算與偏差量測](2026-09-16_下市股最後有價日結算與偏差量測.md) | `validation/delisting.py`、`scripts/diagnose_delistings.py` | 固定期末價缺值時，如何區分下市與停牌並避免靜默剔除？ |
| [漲停預測力與成本結構](2026-09-16_漲停預測力與成本結構.md) | `scripts/diagnose_limit_up.py` | 「每週抓一檔漲停，40 週複利 45 倍」——訊號只在極端值有資訊，中段比隨機還差 |
| [漲停事件驅動與原始漲停率的更正](2026-09-16_漲停事件驅動與原始漲停率的更正.md) | `labeling/limit_up.py`、`scripts/diagnose_limit_up_events.py` | 漲停價如何依台股跳動單位計算？原始漲停率先前算錯在哪？ |
| [箱內排序能力與優勢集中度](2026-09-16_箱內排序能力與優勢集中度.md) | `scripts/diagnose_within_bin_ic.py` | 分數在同一個機率箱內還有排序能力嗎？ |
| [最佳箱的逐年穩定度](2026-09-16_最佳箱的逐年穩定度.md) | `scripts/diagnose_bin_stability.py` | 「最好的那一箱」每年都是同一箱嗎？ |
| [持有期 120 日的實測結果](2026-09-16_持有期120日的實測結果.md) | `scripts/diagnose_families.py` | 成本最低的持有期，報酬撐得住嗎？ |
| [投組約束的代價與檢定力極限](2026-09-16_投組約束的代價與檢定力極限.md) | `ranking/portfolio_features.py`、`ranking/constraints.py`、`scripts/diagnose_constraints.py` | 產業／ATR／相關性約束要付多少代價？**這個檢定只能偵測「約束把整個優勢消滅」** |
| [成本分層必須使用實際價格](2026-09-16_成本分層必須使用實際價格.md) | `config/costs.py` | 還原價錨定在最新日，用它分層會分錯級距 |
| [成本地板與可負擔性用錯價格](2026-09-16_成本地板與可負擔性用錯價格.md) | `data/loader.py`、`scripts/diagnose_cost_floor.py` | 40 萬資金的成本地板在哪？可負擔性為什麼必須用 `raw_*`？ |
| [ETF 納入標的池與逐檔成本更正](2026-09-16_ETF納入與逐檔成本更正.md) | `data/etf_universe.py`、`config/costs.py`、`scripts/validate_oos_momentum.py` | 策略選不到 0050，把 ETF 放進池裡會怎樣？逐檔成本改了哪些數字？ |
| [前推帳本改記動能突破 N=10](2026-09-16_前推帳本改記動能突破.md) | `forward_predictions.py`、`scripts/record_forward.py` | 為什麼帳本要記證據最完整的那一組，而這不是背書？ |

## 2026-09-17

| 文件 | 對應程式 | 核心問題 |
|---|---|---|
| [交易日缺口驗證與守門](2026-09-17_交易日缺口驗證與守門.md) | `data/integrity.py`、`scripts/diagnose_data_integrity.py` | 無價日是資料漏列還是實際停牌？回測如何避免靜默縮小候選池？ |
| [單一價格框架](2026-09-17_單一價格框架.md) | `data/loader.py` | 還原價與實際成交價如何共用同一索引並避免雙次載入漂移？ |
| [守門範圍與下市豁免](2026-09-17_守門範圍與下市豁免.md) | `data/integrity.py`、`validation/delisting.py` | 為什麼只守實際成交的前 N 檔，且下市與暫停必須分流？ |
| [新聞在落地當天就被定價，與 ETF 的成本區間](2026-09-17_新聞已被定價與ETF成本區間.md) | — | 「新聞的影響比任何因素都大，能收集新聞嗎？」 |
| [ETF 輪動的否定](2026-09-17_ETF輪動的否定.md) | `ranking/etf_rotation.py`、`scripts/validate_etf_rotation.py` | 流動性、PBO、與「選參數」本身沒有資訊 |
| [本金要多少才夠？](2026-09-17_本金與成本地板.md) | `scripts/diagnose_cost_floor.py` | 「40 萬難以打敗買進持有，那最少要多少本金？」——**本金不是瓶頸** |
| [LightGBM baseline 與對照組的錯誤](2026-09-17_LightGBM_baseline與對照組的錯誤.md) | `models/lgbm_baseline.py`、`scripts/validate_lgbm_baseline.py` | ⚠️ 本文的主要結論已被 2026-09-18 的持有期修正推翻，數字以 09-18 兩篇為準 |
| [用新門檻重算手工分數](2026-09-17_手工分數對無資訊對照組的重算.md) | `validation/uninformed.py`、`scripts/validate_hand_scores_vs_uninformed.py` | ⚠️ 數字已被 2026-09-18 的持有期修正更正（方向與順序不變） |

## 2026-09-18

| 文件 | 對應程式 | 核心問題 |
|---|---|---|
| [持有期多算一天，推翻了我自己關於對照組的結論](2026-09-18_持有期多算一天推翻了對照組的結論.md) | `models/lgbm_baseline.py`、`scripts/validate_lgbm_baseline.py`、`scripts/validate_hand_scores_vs_uninformed.py` | `shift(-1-H)` 多持有一天。**「對照組太弱」與「73% 是因子傾斜」都不成立** |
| [持有期慣例的清查](2026-09-18_持有期慣例的清查.md) | `data/integrity.py`、六支算 forward 的腳本 | 一個宣稱有單一來源的註解，和一個生產碼零呼叫者的守門 |
| [「超過隨機 95% 分位」這句話該退休](2026-09-18_超過隨機95分位這句話該退休.md) | `validation/uninformed.py`、`taiwan-quant/CLAUDE.md` | 換成嚴格虛無後，兩種搬法給出相反判定——**結論不能重算，只能撤回** |
| [信賴區間讓累積報酬失去意義](2026-09-18_信賴區間讓累積報酬失去意義.md) | `validation/bootstrap.py`、`scripts/diagnose_bootstrap_intervals.py` | 樣本外 +138.83% 的 95% 區間是 [−13.3%, +587.2%]，**七個基準全落在裡面** |
| [門檻形同虛設與路徑相依的根因](2026-09-18_門檻形同虛設與路徑相依的根因.md) | `validation/path_dependence.py`、`backtest/portfolio_sim.py`、`validation/thresholds.py` | 門檻站在瓶頸下游（訊號 ×18 而報酬不變）；路徑相依需要**出場日錯開** |
| [融資暴增之後的超額報酬](2026-09-18_融資暴增之後的超額報酬.md) | `validation/event_study.py`、`scripts/diagnose_event_reactions.py` | **第一個通過多重測試的正向結果**：+1.709%／趟、t=5.21、不是動能；但只有 H=40 淨值為正 |

## 2026-09-19

| 文件 | 對應程式 | 核心問題 |
|---|---|---|
| [訊號是真的但 40 萬吃不到](2026-09-19_訊號是真的但40萬吃不到.md) | `validation/event_study.capacity_constrained_fills` | 容量、加碼、持有期、新聞四條收割路徑全部失敗；**真正的約束是零股成本** |

---

> 09-19 那篇把 09-18 的正向結果收窄了：訊號是真的（t=5.21），
> 但 40 萬只能持有 20 檔，而 20 檔的淨值是負的。
>
> **唯一還沒攻過的是成本結構本身。** 同一個訊號在整股成本（0.671%）
> 下的區間不含零，在零股成本（1.081%）下含零——變的不是訊號，
> 是「40 萬 ÷ 10 檔 = 被迫零股」。

---

> 09-18 三篇是同一件事的三個層次：**bug、範圍、以及引用。**
>
> 持有期多算一天（bug）能活 28 天，因為 forward 報酬有六份實作（範圍）；
> 而一個被取代的門檻能繼續當證據用兩天，因為它散落在 11 處引用裡
> （引用）。**三者都是「沒有單一來源」的不同表現。**
>
> ⚠️ 09-17 的 LightGBM 與手工分數兩篇，數字已被 09-18 更正。
> 兩篇都保留原文並加註（禁令 9），但**引用時以 09-18 為準**。

---

> 第 9 篇是本日最重要的發現：同一條管線，校準前推播 3 檔（顯示 P=64.6%），
> 校準後推播 **0 檔**（最高真實機率僅 22.06%，門檻 40.22%）。
>
> 第 13 篇是第二重要的：回測框架的單槽位限制，讓策略的曝險只有 23~39%，
> 而對照組買進持有是 100%。**「策略輸給買進持有」有相當部分是在測框架，
> 不是在測策略。** 修掉之後跑贏隨機從 3/9 變成 8/9。

---

## 貫穿全部文件的三個教訓

這些不是理論，是實測踩出來的。

### 1. NaN 不會讓比較拋錯，它會讓比較「看起來通過」

```python
float('nan') < 2.0     # False  → 門檻靜默放行
float('nan') != float('nan')   # True → 兩邊都 NaN 被誤判為「不同」
float('nan') == 100.0  # False → 一邊沒值被誤判為「相同」
```

踩到三次：

| 次數 | 位置 | 症狀 |
|---|---|---|
| 1 | `barrier_width.derive_width()` | 一列 OHLC 為 NULL → `np.quantile` 回 NaN → `NaN < 2.0` 為 False → **R:R 門檻靜默放行** |
| 2 | `lookahead._values_differ()` | 若只寫 `full != trunc`，洩漏的典型徵狀（一邊 NaN）會漏判 |
| 3 | `stats.ic_selection_health()` | 浮點噪音讓 `std == 0` 判斷失效 → 相關係數變成 4.3e-17 的噪音 |

**對策**：任何門檻比較前先 `np.isfinite()`；任何「零」的判斷用容差不用精確相等。

### 2. 「無法計算」與「算出來很差」必須區分

| 情境 | 錯誤做法 | 正確做法 |
|---|---|---|
| 樣本不足算不出 Sharpe | 回 0 | 回 `None` |
| valid IC 無變異 | 回 0 | 回 `None` |
| 未來 K 棒不足無法標記 | 猜一個標籤 | 回 `None` |
| 解析出的成分股少一半 | 回半套資料 | 拋錯 |

回一個看起來像答案的數字，會讓下游做出錯誤決策，而且沒有任何警告。

### 3. 沒有正控制組的「全部通過」不可採信

若掃描器本身壞了（regex 寫錯、函式沒被呼叫、切點清單是空的），結果都是「全部通過」。

這是**最危險的假通過**——它給了虛假的安全感。

所以每個驗證工具都內建故意失敗的案例：

- `lookahead.py` 有 4 個作弊特徵，抓不到就拋 `LookaheadError`
- `test_lookahead_audit.py` 有合成作弊公式 + label 正控制組
- `test_lookahead_truncation.py` 的 label 截斷後必須變 `None`

---

## 決策與驗證紀錄

原理說明的上游依據：

| 文件 | 內容 |
|---|---|
| `docs/需求規劃/202609/00_總整合與最佳建議.md` | 七家 AI 意見匯總與最佳建議 |
| `docs/需求規劃/202609/01_決策紀錄.md` | D1~D7 決策 + D7 修訂 |
| `docs/需求規劃/202609/評估_qlib-tw-trader.md` | 採用評估（結論：部分沿用） |
| `docs/需求規劃/202609/02_驗證結果_qlib-tw-trader.md` | 兩步實測（look-ahead 通過、成本紅線觸發） |
| `docs/需求規劃/202609/03_策略可行性最終結論.md` | triple-barrier 樣本外：9/9 輸給買進持有 |
| `docs/需求規劃/202609/04_路線A驗證結果.md` | 移動停損 + 多槽位：8/9 贏隨機、3/9 贏買進持有，但不穩 |
| `docs/需求規劃/202609/05_長歷史驗證結果.md` | 資料拉到 2015 後重跑：先前的正報酬是多頭環境的產物 |
| `docs/需求規劃/202609/06_對照組補齊與資料修正.md` | 0050 對照補齊、還原因子 bug 修正、標的池代理 94.7% |
| `taiwan-quant/CLAUDE.md` | 12 條禁令 + 4 條實測逼出的規格 |
| `taiwan-quant/AGENTS.md` | Codex reviewer 的 30 項審查清單 |
| `taiwan-quant/docs/建置進度.md` | 模組完成度與已知限制 |

---

⚠️ 全部文件皆為工程原理說明，不構成投資建議。
