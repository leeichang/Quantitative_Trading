# Codex 工作單：兩套持有期慣例活了 28 天，因為 forward 報酬有六份實作

- 日期：2026-09-18
- 分支：`codex/oos-freeze`（**不要開新分支，不要動 main**）
- 前一張：`08_Codex工作單_守門範圍與下市豁免.md`
- 起點：`f441bc9 fix: holding period was one day too long, which invented my benchmark finding`
- 目前測試數：901 passed

⚠️ **先確認你能看到 `f441bc9`。** 這個 commit 本機領先遠端一個，
若還沒推送，你 `git pull` 不會拿到持有期修正，下面所有數字都對不上。
開工前先 `git log --oneline -1` 核對 SHA。

---

## 第零部分：任務 P 你做對的部分

### 你找到了主線的 off-by-one，而且方向判斷正確

我在三個檔案寫 `closes.shift(-1 - horizon)`，持有期變成 H+1 天。
你在修守門時發現 `data/integrity.holding_dates` 的
`decision_index + holding_days` 與主線不一致，**並且指出是主線錯**。

那個判斷需要勇氣，因為主線那三個檔案有通過的測試。

**而測試通過恰恰是因為它錯了**——程式與測試是我同時寫的，兩邊照同一個
誤解，所以測試驗證的是我的理解，不是行為：

```python
# 修正前的 tests/test_lgbm_baseline.py，斷言 closes[2] / opens[1]
assert value == pytest.approx(13.0 / 11.0 - 1)   # T+1+horizon，錯
```

這一天的差異推翻了我 09-17 的主要發現。完整更正見
`../qlib-tw-trader/docs/原理說明/2026-09-18_持有期多算一天推翻了對照組的結論.md`。

### 守門範圍收到實際遞補者是正確的設計

`select_holding_positions`（`taiwan_quant/data/integrity.py:154`）用
`MISSING_ENTRY` 跳過、`SUSPENDED_OR_MISSING` 才拋錯，並把下市判斷
委派給 `validation/delisting.py` 而不是寫第二份。那是對的。

### 獨立驗證：兩支不同 agent 寫的腳本現在對得上

```
你的 diagnose_constraints baseline（task08_constraints_dev.json）   3.0863%／趟
主線的 validate_hand_scores 動能突破                                3.09%／趟
```

修正之前是 3.09% vs 3.48%。那個 0.39 pp 落差就是多算的那一天。
**兩份獨立實作對同一個量收斂，比任何單一腳本的自我一致都有說服力。**

---

## 第一部分：根因——六份實作

慣例已經統一成 `shift(-horizon)`。但實作沒有統一：

```
taiwan_quant/models/lgbm_baseline.py:211            closes.shift(-horizon) / opens.shift(-1) - 1
scripts/diagnose_constraints.py:328                 closes.shift(-HOLDING_DAYS) / opens.shift(-1) - 1
scripts/diagnose_cost_floor.py:363                  closes.shift(-horizon) / opens.shift(-1) - 1
scripts/diagnose_position_costs.py:127              adjusted_closes.shift(-HOLDING_DAYS) / adjusted_opens.shift(-1) - 1.0
scripts/validate_hand_scores_vs_uninformed.py:190   closes.shift(-HOLDING_DAYS) / opens.shift(-1) - 1
scripts/validate_lgbm_baseline.py:205               closes.shift(-HOLDING_DAYS) / opens.shift(-1) - 1
```

**六份。兩套慣例能共存 28 天，就是因為沒有單一來源可以違反。**

### 最刺眼的一點

其中兩個檔案的註解**宣稱**有單一來源，程式碼卻自己算：

```
scripts/validate_lgbm_baseline.py:204            # data/integrity.holding_dates 都不一致。
scripts/validate_hand_scores_vs_uninformed.py:189  # data/integrity.holding_dates 都不一致。
```

這兩支腳本**沒有 import 任何 integrity 函式**。註解記錄了正確的意圖，
程式碼沒有執行它。註解不會被測試，所以它可以永遠是對的而程式永遠是錯的。

### `data/integrity` 的實際覆蓋率

30 支腳本裡只有 3 支用到它（不是我先前口頭說的 5 支）：

```
函式                                 生產碼呼叫者
holding_dates                        diagnose_data_integrity.py:123
                                     diagnose_delistings.py:200
                                     （integrity.py 內部 2 處）
complete_holding_decision_dates      diagnose_data_integrity.py:113
                                     diagnose_delistings.py:140
                                     validate_oos_momentum.py:181
select_holding_positions             validate_oos_momentum.py:210, 299, 323
assert_holding_price_completeness     ← 生產碼零呼叫者
```

---

## 第二部分：任務

### 任務 V：把 forward 報酬收成單一 helper（最高優先）

在 `taiwan_quant/data/integrity.py` 旁邊——或另開一個模組，你決定——
提供一個函式，讓六個呼叫點全部改用它。

**契約必須包含的東西：**

```
輸入   opens、closes（同索引同欄位的寬表）、holding_days
輸出   與輸入同索引的 forward 報酬框（或 stack 後的 Series，二選一，寫清楚）
定義   T+1 開盤進、T+H 收盤出
       forward[T] = closes[T + holding_days] / opens[T + 1] - 1
```

**三個必須處理的細節，不要靜默決定：**

1. **`holding_days` 的下界。** `holding_days < 1` 要拋錯，不要回 NaN。
   `holding_days = 0` 在數學上是 `closes[T] / opens[T+1]`，那是未來價
   除以更未來的價，無意義。`models/lgbm_baseline.py` 已有 `BaselineError`
   的先例。

2. **adjusted vs raw。** `diagnose_position_costs.py:127` 用
   `adjusted_*`，其餘五處用 `opens`/`closes`。helper 不該猜——
   由呼叫端傳入哪一組框。但**要在 docstring 寫明**：報酬算在還原價上，
   可負擔性算在 `raw_*` 上（這是 09-16 踩過的坑）。

3. **尾端不足 H 天的處理。** 現在六處都靠 `shift` 自然產生 NaN，
   然後各自 `dropna`。helper 應該保持這個行為（回 NaN 而不是截斷），
   **但呼叫端要顯式篩掉**——最好是改用
   `complete_holding_decision_dates`，那已經是單一來源。

**驗收條件：**

```
□ 六個呼叫點全部改完，rg 不再有第二份 shift 算式
□ 新增一個測試，用手算的小框直接驗數字（不要只比對兩條路徑一致——
  那是我上次犯的錯，兩邊照同一個誤解會一起通過）
□ 新增一個測試，斷言出場日等於 holding_dates 回傳的第二個值
□ 六支腳本各跑一次，輸出附在交回報告裡
□ 數字如有變化，逐項列出變多少（預期為零，若非零就是有第七套慣例）
```

⚠️ **這一項可能發現新 bug。** `diagnose_cost_floor.py:363` 在
`forward_cache[horizon]` 裡對多個 horizon 迴圈——若某個 horizon 的
邊界處理與其他不同，統一時會浮出來。**浮出來就報，不要順手修掉**
再說「已統一」。

### 任務 W：任務 R 收尾——三份 stale 報告，不是兩份

先修正工作單 08 的說法。受 task M 的 `build_dataset` 左連接影響、
且**尚未重跑**的有三份：

```
reports/cost_floor_dev.json        09-17 09:40   diagnose_cost_floor.py:348 呼叫 build_dataset
reports/limit_up_events_dev.json   09-17 09:46   diagnose_limit_up_events.py:417 呼叫 build_dataset
reports/threshold_sweep_dev.json   09-16 08:22   evaluate_threshold_schemes.py:114 呼叫 build_dataset
```

第三份是工作單 08 漏掉的。它是全 `reports/` 最舊的檔案。

已完成的部分，供你核對：

```
task08_constraints_dev.json        09-18 08:57   你跑的
task08_data_integrity_dev.json     09-18 08:57   你跑的
task08_delistings_dev.json         09-18 08:57   你跑的
task08_position_costs_dev.json     09-18 08:57   你跑的
hand_vs_uninformed_dev.json        09-18 09:07   我跑的（持有期修正後）
lgbm_baseline_dev.json             09-18 09:15   我跑的（持有期修正後）
etf_rotation_dev.json              09-17 11:11   豁免（實測 0 天缺口）
```

#### ⚠️ 附帶問題：你用新檔名寫，舊檔名留著

```
constraint_cost_dev.json     09-16 22:25   ← 舊，已被 task08_constraints_dev.json 取代
delisting_policy_dev.json    09-17 09:33   ← 舊，已被 task08_delistings_dev.json 取代
position_costs_dev.json      09-17 09:33   ← 舊，已被 task08_position_costs_dev.json 取代
```

`reports/` 現在同時有兩代同一個量測。**讀 `constraint_cost_dev.json`
會拿到 task M 修正前的數字，而檔名沒有任何提示。**

禁令 9 說不可就地改寫已發佈的報告數字。它沒說不可以加註。三個選項：

```
A. 舊檔加一個 superseded_by 欄位，指向新檔（推薦：保留歷史且可機讀）
B. 舊檔移到 reports/superseded/ 子目錄
C. 新檔改回原檔名，舊檔改名加 _pre_task08 後綴
```

**選一個，說為什麼，然後對全部三組一致套用。** 不要三組用三種做法。

### 任務 X：任務 S 的題目已作廢，要重寫

工作單 08 問「為什麼標籤打亂比隨機權重嚴格」，引用打亂 = +2.94%／趟
是最嚴格的對照組。

**持有期修正後順序反了：**

```
對照組構造              41 天（工作單 08 引用）   40 天（正確）
標籤打亂                  +2.94%   最嚴格          +0.56%   最寬鬆
稀疏隨機權重              +1.58%                  +0.86% ~ +1.14%   最嚴格
純隨機選股                +1.05%                  +0.57%
```

**原題目沒有答案，因為打亂不嚴格。** 正確的題目是：

> 為什麼**稀疏隨機權重**比標籤打亂與純隨機選股都嚴格 0.3 ~ 0.6 pp？

我有一個未驗證的機制假說，寫在
`2026-09-18_持有期多算一天推翻了對照組的結論.md` 的「為什麼那一天的
差異這麼大」一節：打亂的模型在 `forward` 的橫斷面排名上訓練，改變
`forward` 一天就改變標籤、改變擬合出的函數、改變它選的名字；而隨機
選股與 `forward` 無關。

**那一節明確標了 `⚠️ 這是機制推論，不是實測`。** 要確認需要比對兩個
版本各自選出的股票名單。

```
□ 若你做這個測試：比對 41 天版與 40 天版各自的前 10 名單重疊度
□ 若假說成立：說明它如何導出「稀疏隨機權重最嚴格」
□ 若假說不成立：說清楚，並更正那一節（加註，不要就地改寫）
□ 若不做：明確說不做，不要留著當已完成
```

**這件事的用途不變**：CLAUDE.md 要求「有多種構造時用最嚴格的那個」。
若能說明為什麼，就能預測哪種構造對新策略才是對的門檻。

### 任務 Y：兩個負相關 sleeve 的 Sharpe（內容不變，零進度）

`scripts/` 底下沒有 sleeve／blend／combine 腳本。從零開始。

實測的三族橫斷面關係：

```
配對                  Spearman 中位   前 10 重疊中位   完全不重疊的期數
動能突破 / 均值回歸        −0.505            0          34/36
籌碼跟隨 / 均值回歸        −0.320            0          28/36
動能突破 / 籌碼跟隨        +0.526            3           3/36
```

**排名平均已被否定**（兩個相反排序平均到分布中段，而中段最差：
50~80% 桶的 5 日漲停率 0.62%，低於最低 50% 的 0.83%）。

−0.505 的另一個用途是**兩個 sleeve 各半資金**。降低投組變異不需要
任何一邊變強。

⚠️ **代價是稀釋。** 均值回歸年化淨 5.2%、籌碼跟隨 12.3%，都遠弱於
動能突破。Sharpe 會不會改善是可測的，尚未測。

⚠️ 均值回歸已 `publishable=False`
（`taiwan_quant/strategies/families.py:308`）。做這個測試時**不要**為了
跑通而改旗標——診斷腳本直接取 `STRATEGY_FAMILIES`，不受旗標限制。

⚠️ **成本會吃掉一部分。** 兩個 sleeve 各半資金代表每個 sleeve 的
單筆金額減半，而成本分層對小額不利（禁令 4：0050 成分股 0.3%
零股、0051 0.4%）。40 萬本金下這不是小數。**Sharpe 要報淨值，
不要只報毛。**

### 任務 Z：守門接線，以及一段死碼的處置

原任務 U。比工作單 08 寫的更嚴重。

#### Z-1：`assert_holding_price_completeness` 是死碼

```
定義於   taiwan_quant/data/integrity.py:128
引用者   tests/test_data_integrity.py:13, 76
生產碼   零
```

你在任務 P 花力氣修的那個範圍缺陷，修的是一個**生產環境沒人呼叫的
函式**。`validate_oos_momentum.py` 的守門來自
`select_holding_positions`，那個函式在 `integrity.py:188-192` 獨立拋錯，
**從不呼叫 assert**。

依 KISS／YAGNI，兩個出口：

```
A. 刪掉，連同它的測試。select_holding_positions 已涵蓋實際需求。
B. 留著，但在 docstring 寫明它的用途是什麼、誰該呼叫它、為什麼
   select_holding_positions 不呼叫它。
```

**選 A 或 B，不要留現狀。** 現狀是一個有測試、看起來被守護、
實際上不執行的函式——那比沒有守門更危險，因為它讓人以為有守門。

#### Z-2：`record_forward.py` 沒有守門

```
scripts/record_forward.py   不 import 任何 integrity 函式
```

**這是唯一在產出真實前推證據的腳本。** 它記的帳本
（2026-11-09 揭曉第一期）是整個系統之後唯一的樣本外證據來源。
它現在沒有任何價格完整性檢查。

它已經是你的（`7e5890d` 之後）。接上 `select_holding_positions`
或至少 `complete_holding_decision_dates`。

⚠️ **接守門會改變它的行為。** 若前推期間有候選股停牌，現在會拋錯
而不是靜默跳過。**那是我們要的**，但你要在交回報告裡說明：
接上之後在開發集重放一次，有沒有觸發？觸發幾次？

#### Z-3：30 支腳本，3 支有守門——這要是決定不是遺漏

```
diagnose_data_integrity.py     ✓
diagnose_delistings.py         ✓
validate_oos_momentum.py       ✓
其餘 27 支                      ✗
```

不必全部接。但要分類：哪些是**不需要**（純特徵診斷、不結算報酬），
哪些是**遺漏**。寫成一張表放進說明文件。

---

## 報數字的規矩（累積到第六張工作單）

### ⚠️ 新增：不要在未驗證旗標的 grep 後面串 `|| echo "無殘留"`

我這次踩到。本機 `rg` 經 rtk hook 解析到 BSD `grep`，
`--glob` / `-g` / `--type` 全部報 usage error：

```bash
rg -n "shift(-1 -" --glob '*.py' || echo "  無殘留"
#                  ^^^^^^ usage error，不是零匹配
#                                        ↑ fallback 因旗標錯誤觸發，印出乾淨的假報告
```

**`||` 分不出「零匹配」和「命令壞了」。** 兩次假陰性。
改用 `grep -rn --include="*.py"`，或直接寫 Python 掃檔——後者沒有
routing 問題。

### 改動的腳本必須實際執行，並附上輸出

`901 passed` 證明測試通過，不證明腳本能跑。工作單 08 的缺陷就是這條
沒做到——`795 passed` 而 `diagnose_constraints.py` 從第一個決策日拋錯。

### 只有一個觀察非零時不要報 t

`mean == SE` 是代數恆等式不是巧合。你在 07 更正過，維持。

### 點估計不算結論，要配對標準誤，並說檢定力

範例：投組約束「全部嚴格 vs baseline」點估計 −1.75 pp，配對後
SE 1.58 pp、t = −1.11。2 SE = 3.16 pp 而優勢本身 3.28 pp——
**這個檢定只能偵測「約束把整個優勢消滅」。**

### 統計函式的每個輸入都要有實測依據，預設值不是依據

```
deflated_sharpe_ratio(..., sharpe_std=1.0)      預設值，門檻 1.799，什麼都不通過
                           sharpe_std=0.1958    實測，門檻 0.180，幾乎都通過
n_observations                                  有效期數 = 天數 ÷ 持有期，需 >= 30
```

### ⚠️ 更正：對照組規格的**理由**已改，做法不變

CLAUDE.md 的「必跑對照組」第四條在 09-17 加入時，理由寫「隨機對照組
太弱，一個無資訊模型也能顯著贏它（+1.88%，t = 2.32）」。

**那個理由是 off-by-one 造出來的。** 正確持有期下打亂 − 隨機 = −0.01%，
t = −0.02。

**但做法要保留，理由換成：**

> 置換檢定是管線的驗證。它現在「通不過」（打亂 − 隨機 = −0.01%，
> t = −0.02），**那正是我們要的結果**——證明優勢不是來自洩漏。
> 它先前「通過」才是警訊。

三條實作要求不變：權重／模型的時間行為一致、允許雙向、
**稀疏度等於被比較策略的特徵數**。

---

## 禁令 6 的現況

```
開發集    2015-01 ~ 2023-12-29    可以反覆跑
OOS      2024-01 ~ 2026-09-11    已經用掉，不可再跑
前推     2026-09-11 起            帳本已記兩版，2026-11-09 揭曉
```

⚠️ 凍結守門 `FROZEN_DATA_START = date(2026, 9, 14)` **只擋前推區間**。
2024-01 ~ 2026-09-11 沒有機械保護，每個腳本都要明確傳
`--end 2023-12-29`。

⚠️ **任務 V 會改到 `validate_oos_momentum.py` 的依賴鏈。**
改完**不要重跑它**——那會動用已用掉的區間。用開發集的診斷腳本驗證。

⚠️ **任務 Z-2 接守門後同理**：`record_forward.py` 在開發集重放可以，
不要用它去碰 OOS 區間。

---

## 不要做的

```
✗ 重跑 validate_oos_momentum.py（禁令 6）
✗ 用 record_forward.py 碰 2024-01 ~ 2026-09-11（禁令 6）
✗ 在統一 helper 時「順手」修掉浮出來的新 bug 再說已統一（要分開報）
✗ 只比對兩條程式路徑一致就當測試（我上次就是這樣讓錯的慣例通過）
✗ 就地改寫已發佈的報告數字（禁令 9，要加註）
✗ 為了讓任務 Y 跑通而改掉 families.publishable 旗標
✗ 在 integrity.py 裡寫第二份下市判斷（要呼叫 delisting.py）
✗ 給守門加繞過開關
✗ 開新分支或合併到 main
```

---

## 交回時附上

```
□ 六個 forward 呼叫點統一後，各自腳本的真實輸出
□ 統一前後的數字對照（預期為零差異，非零就是有第七套慣例）
□ 手算驗證的新測試（不是兩路徑互比）
□ 三份 stale 報告重跑後：哪些數字變了、變多少，逐項列出
□ 舊檔名的處置：選了 A/B/C 哪一個、為什麼、是否三組一致
□ assert_holding_price_completeness 的處置：刪或留，理由
□ record_forward.py 接守門後在開發集重放的觸發次數
□ 30 支腳本的守門分類表（需要 / 不需要 / 遺漏）
□ 任務 X：假說成立或不成立，或明確說不做
□ 任務 Y：Sharpe 要報淨值，不要只報毛
□ 一份 qlib-tw-trader/docs/原理說明/ 的說明文件，檔名含日期
□ 測試數與變化（目前 901 passed）
□ 起點 commit SHA（應為 f441bc9）
□ 自我審查：這次改動有沒有讓任何數字往「變好看」的方向動？
  如果有，那個方向是不是有獨立證據支持？
```

前 15 個抓到的 bug 裡大部分是「讓結果變好看」的方向被抓出來的。
**先自己問這個問題，比等人來問便宜。**

而這一次最大的一個是反過來的——持有期多算一天讓我的**對照組**變好看，
於是我得出「對照組太弱」的結論。**往任何方向偏的錯都值得同樣的懷疑。**

---

⚠️ 本文件為工程交接紀錄，不構成投資建議。所有績效數字為歷史回測，
不代表未來表現。本系統只產出投資建議，不涉及任何下單行為。
