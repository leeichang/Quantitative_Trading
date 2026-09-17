# Codex 工作單：守門的範圍錯了，兩支診斷腳本現在跑不起來

- 日期：2026-09-17
- 分支：`codex/oos-freeze`（**不要開新分支，不要動 main**）
- 前一張：`07_Codex工作單_資料完整性與機制收斂.md`
- 起點：`54bb460 fix: guard only rankable price candidates`（已確認同步）

---

## 第零部分：任務 M 你做對的部分

先說清楚，因為這張工作單的其餘篇幅都在講一個缺陷。

### 683 個缺口日全部外部驗證，而且補寫 0 列

```
72 檔候選股｜683 個無價日（164 單日 + 95 段區塊共 519 日）
FinMind 逐日回查：683 日全部沒有正值 OHLC
TWSE 官方端點抽查：零成交、`--` 價格
stock_daily 修復前後都是 2,914,166 列
```

**工作假說（「那是資料庫漏列」）被推翻，而你照實寫，沒有用前值或 0 偽造
成交。** 那是這張工作單裡最重要的一件事做對了——我在 07 裡特別警告過
「方向弄反會把真停牌的日子憑空補出價格，那比缺值更糟」，你沒有踩。

### dataset 改左連接是正確的機制修正

價格是交易日曆的主體，籌碼缺值只代表某項特徵不可用。舊的 inner join
讓籌碼缺值連價格日一起刪除，那是用無關資料改寫排名。

政策 A 毛報酬 4.9307% → 5.2015%（+0.27 pp）是這個修正的副產品，
方向雖然變好但可獨立驗證——**你主動指出這件事而不是把它當成績，
那是對的**。

### 單一價格框架採用了正確的設計

`load_prices(adjusted=True)` 單次查詢、同框架帶 `raw_open`/`raw_close`，
`load_price_views` 降為會發 `DeprecationWarning` 的相容層且**只查一次
資料庫**。這消除了我原本擔心的 adjusted/actual 索引漂移。

### t = −1.00 的更正

你把「48 期中只有一個差值非零、t 值是代數產物」寫進更正，正確。

### 憑證

我獨立掃過工作樹與 git 全歷史所有 blob（嚴格 JWT 三段樣式），**零命中**。
你說「程式碼、報告及 Git 歷史中未保存」成立。使用者已被告知要自行撤銷重簽。

### ⚠️ 測試數不是退步

你報 `795 passed, 11 deselected`，主線現在跑 **864 passed**。差額是你跑
測試之後才合併進來的三批（LightGBM baseline +34、無資訊對照組 +18、
`publishable` 旗標 +6）。**沒有掉測試。**

---

## 第一部分：阻塞缺陷——守門讓兩支開發集診斷無法執行

### 重現（我這邊逐字輸出）

```
$ .venv/bin/python scripts/diagnose_constraints.py
taiwan_quant.data.integrity.DataIntegrityError: 決策日 2018-03-30 的候選價格不完整：
  2325 T+H 2018-05-31, 2311 T+H 2018-05-31

$ .venv/bin/python scripts/diagnose_position_costs.py
taiwan_quant.data.integrity.DataIntegrityError: 決策日 2016-11-09 的候選價格不完整：
  3474 T+H 2017-01-06
```

**兩支都無法完成。** 而你報告裡的 A/B 數字來自
`scripts/diagnose_delistings.py`——**唯一沒接守門的那支**。

守門接線的現況：

```
有守門   validate_oos_momentum.py
有守門   diagnose_constraints.py          ← 崩
有守門   diagnose_position_costs.py       ← 崩
無守門   diagnose_delistings.py
無守門   diagnose_cost_floor.py
無守門   diagnose_limit_up_events.py
無守門   validate_hand_scores_vs_uninformed.py
無守門   validate_lgbm_baseline.py
無守門   validate_etf_rotation.py
無守門   record_forward.py
```

### 缺陷一：守門把「下市」當成資料缺陷，與你自己的政策 B 矛盾

被擋掉的三檔全是下市：

```
3474 華亞科   下市 2016-12-06   與台灣美光股份轉換
2325 矽品     下市 2018-04-30   與日月光共同股份轉換成立日月光投控
2311 日月光   下市 2018-04-30   同上
```

**下市不是資料缺陷，是有處理方式的市場事件**——而處理方式就是你在任務 K
建的政策 B：以最後一個真實收盤價結算。

任務 K 建了三分類正是為了區分這件事：

```
delisted                stock_master.delisted_date 有值，永久停止交易
suspended_or_missing    目標日缺值但之後仍有價
missing_entry           連次一交易日開盤價都不存在
```

**而 `assert_holding_price_completeness` 完全沒有查詢它。** 它只看
「T+H 有沒有正值收盤」，於是把三分類的第一類誤判成錯誤。

### 缺陷二：守門檢查整個候選池，不是可能被選到的名字

commit 訊息寫 `guard only rankable price candidates`，但呼叫端傳的是
`candidates=ranked.index`——**`dropna()` 之後的全部候選，約 148~150 檔**。

實測那三檔的策略排名（現行程式、`N_POSITIONS = 10`）：

```
決策日 2016-11-09｜ranked 150 檔
   3474 華亞科   排名 13/150   分數 0.5911   不會進前 10

決策日 2018-03-30｜ranked 148 檔
   2325 矽品    排名 61/148   分數 0.5447   不會進前 10
   2311 日月光   排名 39/148   分數 0.6015   不會進前 10
```

**三檔全部在前 10 之外。守門為了排名 13、39、61 的名字中止整個回測，
而它們本來就不會被買。**

（附註：你任務 K 的表格寫 3474 策略排名 10。現行程式算到 13——那是
left-join 改動造成的分數位移，與你報的 +0.27 pp 同一個來源。兩個數字
都對，只是來自不同版本的程式。）

### 守門的方向是對的，範圍錯了

我同意你寫的「停牌不是可成交價格，明確失敗比悄悄縮小候選池正確」——
先前正是主線指出 `dropna()` 的靜默剔除。

但現在的結果是：**加了守門之後，兩支診斷從「靜默剔除」變成「完全不能跑」。**
那不是改善，是把一個安靜的錯換成一個吵鬧的停擺。

---

## 任務 P：修守門（阻塞，優先於一切）

### 做什麼

1. **守門查三分類，`delisted` 不報錯。**

   `taiwan_quant/validation/delisting.py` 已經有分類器。守門應該：

   ```
   delisted               → 不是錯誤。交由結算政策處理（政策 B）
   suspended_or_missing   → 這才是真正的完整性問題，報錯
   missing_entry          → 明確排除，不可進場
   complete               → 通過
   ```

   ⚠️ **不要把判斷寫進 `integrity.py` 第二份。** 分類邏輯已經在
   `delisting.py`，守門呼叫它——兩份實作遲早漂移（先前 ATR%、漲停價、
   價格視圖都踩過這件事）。

2. **檢查範圍縮到會被買進的名字。**

   守門要知道 `N`。介面自己決定，但語意必須是「**排序後實際會成交的
   前 N 檔**」，不是整個候選池。

   ⚠️ 這裡有一個順序問題要想清楚：若前 N 檔裡有一檔因守門被排除，
   遞補的第 N+1 檔也要檢查。**不可只檢查一次就放行。**

3. **三支接了守門的腳本要能跑完。** 見驗收。

### 為什麼不提供繞過開關

主線在 `families.publishable` 那次的理由一樣：要研究就用沒有守門的
診斷腳本。守門的價值在於它會擋，加了開關它就等於不存在。

### 驗收（缺一不可）

```
□ .venv/bin/python scripts/diagnose_constraints.py         完整跑完，附輸出
□ .venv/bin/python scripts/diagnose_position_costs.py      完整跑完，附輸出
□ .venv/bin/python scripts/diagnose_delistings.py          完整跑完，附輸出
□ 守門確實仍會觸發：構造一個 suspended_or_missing 落在前 N 檔的案例，
  證明它拋錯。**從不觸發的守門不是守門**
□ 守門不再對 delisted 報錯：以 2016-11-09 / 3474 與 2018-03-30 / 2325+2311
  為迴歸案例
□ 全套測試通過（目前 864 passed）
```

⚠️ **第一到第三項是這次缺陷的根因**：改了腳本但沒有執行它。
請把輸出貼進交回報告。

---

## 任務 Q：守門接線的一致性（次要，但要有決定）

10 支腳本裡只有 3 支有守門。這不是錯，但**必須是一個明確的決定而不是
遺漏**。

請在說明文件裡寫清楚分類，並讓實際接線與它一致：

| 類別 | 是否需要守門 | 理由 |
|---|---|---|
| 會產生可推播結果的 | 需要 | 例：`record_forward.py` |
| 一次性 OOS 驗證 | 需要 | 例：`validate_oos_momentum.py` |
| 純診斷／掃描 | ？ | 由你判斷並寫理由 |

⚠️ `record_forward.py` 是**主線負責**的檔案（見檔案分工），你只要在
文件裡寫「它應該接守門」，不要代改。

---

## 檔案分工

### 你可以動

```
taiwan_quant/data/integrity.py
taiwan_quant/validation/delisting.py
taiwan_quant/data/loader.py
taiwan_quant/data/dataset.py
scripts/diagnose_constraints.py
scripts/diagnose_position_costs.py
scripts/diagnose_delistings.py
scripts/validate_oos_momentum.py
任何新檔案與對應測試
```

### 主線獨占，請不要動

```
scripts/record_forward.py                    ← 仍用 load_price_views，主線會遷移
scripts/diagnose_cost_floor.py
scripts/diagnose_limit_up_events.py
scripts/validate_hand_scores_vs_uninformed.py
scripts/validate_lgbm_baseline.py
scripts/validate_etf_rotation.py
taiwan_quant/labeling/limit_up.py
taiwan_quant/models/lgbm_baseline.py
taiwan_quant/validation/uninformed.py
taiwan_quant/ranking/etf_rotation.py
taiwan_quant/strategies/families.py          ← publishable 旗標剛改過
```

若任務 P 的介面改動會影響主線獨占的檔案，**列清單不代改**。

---

## 報數字的規矩（累積到第五張工作單）

### ⚠️ 新增：改動的腳本必須實際執行，並附上輸出

這次的缺陷就是這一條沒做到。`795 passed` 證明測試通過，但
`diagnose_constraints.py` 從第一個決策日就拋錯——**單元測試不會執行腳本**。

交回報告裡每一支被改動的腳本都要有一段真實輸出。

### 只有一個觀察非零時不要報 t

`mean == SE` 是警訊不是巧合。你已經在 07 更正過，維持。

### 點估計不算結論，要配對標準誤，並說檢定力

範例：投組約束「全部嚴格 vs baseline」點估計 −1.75 pp，配對後
SE 1.58 pp、t = −1.11。2 SE = 3.16 pp 而優勢本身 3.28 pp——
**這個檢定只能偵測「約束把整個優勢消滅」**。

### 統計函式的每個輸入都要有實測依據，預設值不是依據

主線踩過兩次：

```
deflated_sharpe_ratio(..., sharpe_std=1.0)   預設值 → 運氣門檻 1.799 → 什麼都不通過
                             sharpe_std=0.1958（實測）→ 門檻 0.180 → 幾乎都通過
n_observations                有效期數 = 天數 ÷ 持有期，需 >= 30
```

### 對照組不可太弱（2026-09-17 新增規格）

CLAUDE.md 的「必跑對照組」已加第四條「無資訊對照組」。實測依據：
一個標籤被打亂、證明沒有預測資訊的 LightGBM，仍顯著打敗隨機選股
（+1.88%／趟，t = 2.32）。

三條實作要求（見 CLAUDE.md）：權重／模型的時間行為一致、允許雙向、
**稀疏度等於被比較策略的特徵數**。

---

## 禁令 6 的現況

```
開發集    2015-01 ~ 2023-12-29    可以反覆跑
OOS      2024-01 ~ 2026-09-11    已經用掉，不可再跑
前推     2026-09-11 起            帳本已記兩版，2026-11-09 揭曉
```

⚠️ 凍結守門 `FROZEN_DATA_START = date(2026, 9, 14)` **只擋前推區間**。
2024-01 ~ 2026-09-11 沒有機械保護，每個腳本都要明確傳 `--end 2023-12-29`。

⚠️ `validate_oos_momentum.py` 接了守門。**修完不要為了驗證而重跑它**——
那會動用已用掉的區間。用開發集的三支診斷驗證即可。

---

## 不要做的

```
✗ 重跑 validate_oos_momentum.py（禁令 6）
✗ 在 integrity.py 裡寫第二份下市判斷（要呼叫 delisting.py）
✗ 給守門加繞過開關
✗ 就地改寫已發佈的報告數字（禁令 9，要加註）
✗ 動「主線獨占」清單裡的檔案
✗ 開新分支或合併到 main
```

---

## 交回時附上

```
□ 三支診斷腳本的真實輸出（這是本次的重點）
□ 守門仍會觸發的證明案例
□ 一份 qlib-tw-trader/docs/原理說明/ 的說明文件，檔名含日期
□ 測試數與變化（目前 864 passed）
□ 起點 commit SHA
□ 自我審查：這次改動有沒有讓任何數字往「變好看」的方向動？
```
