# Codex 工作單：門檻修正與 pseudo-OOS 評估

- 日期：2026-09-14
- 前一張：[04_Codex工作單_OOS凍結.md](04_Codex工作單_OOS凍結.md)（任務 A~C 已完成，D 未做）
- 對應：[03_待辦與改進方向.md](03_待辦與改進方向.md) 第 2、3 項
- 原理說明：`qlib-tw-trader/docs/原理說明/2026-09-14_OOS凍結與前推帳本原理.md`

---

## 第零部分：你回報的三個阻塞

### 阻塞 1：reviewer-only 限制 → **你擋得對，已解除**

`AGENTS.md:5` 確實把你限定成 reviewer，是我寫工作單時沒跟分工文件對齊。
**04 那張其實也違反同一條**（任務 A~C 是開發），只是那次沒擋。

已改：`taiwan-quant/AGENTS.md` 與 `docs/需求規劃/202609/01_決策紀錄.md` 的 D5。

```
修訂前   Claude Code 開發，Codex 只審查、不寫程式
修訂後   兩邊都可以開發；誰寫的由另一邊審
```

為什麼選擇改分工而不是改工作單：「不同模型互審」這層防線靠**交叉審查**
就能保住，不需要綁死誰不能寫程式。原本的限制讓可用人力少一半，卻沒有多
換到任何安全性。

⚠️ **但驗收清單沒有降級。** 它從「審別人的程式」變成「你自己交件前的驗收
標準」。交叉審查是第二道防線，不是第一道。任務完成後我會審你的 diff。

### 阻塞 2：「E、F 沒做所以不能做 G」→ **那正是要你做的事**

這些是**指派給你的開發任務**，不是先決條件。順序要求是說「做完前一個
再做下一個」，不是「等別人做完」。

（順序已於 2026-09-15 改為 **H → E → G**，F 降級為可選。見第二部分開頭。）

### 阻塞 3：`history.db` 不在你的工作區 → **成立，要實體傳輸**

你在 `/Volumes/cympotek/working/Quantitative_Trading/`，
資料在 `/Volumes/Mac/Quantitative Trading/`——**不同機器**。

`history.db` 967 MB 被 `.gitignore:57` (`*.db`) 排除，**不隨 clone 走**。
這不是路徑設錯，是檔案真的不在那台機器上。

傳輸檔已備妥（`VACUUM INTO` 的乾淨副本 + gzip）：

```
taiwan-quant/data/history_transfer.db.gz          218 MB（解開後 915 MB）
taiwan-quant/data/history_transfer.db.gz.sha256

09454d11fa0ec0caf9d5f1ffd7c4476f69e5eb254d3c20248ec9bf52f755537e
```

用 `VACUUM INTO` 而不是直接複製，是為了拿到不含 WAL 的乾淨單檔——
直接 `cp` 一個開著 WAL 的 SQLite 可能少掉最後幾筆交易。

還原步驟：

```bash
cd "<你的工作區>/taiwan-quant/data"
shasum -a 256 -c history_transfer.db.gz.sha256   # 先驗
gunzip -c history_transfer.db.gz > history.db
sqlite3 history.db "SELECT COUNT(*), COUNT(DISTINCT stock_id), MIN(date), MAX(date) FROM stock_daily;"
# 必須輸出：2914166|1218|2015-01-05|2026-09-11
```

我這邊已經實際解壓驗過一次，數字相符：

```
stock_daily          2,914,166 列｜1,218 檔｜2015-01-05 ~ 2026-09-11
stock_daily_adj      2,931,134 列
stock_master             2,597 列（含 159 檔已下市 → 禁令 2 的解法）
universe_history        13,950 列
```

⚠️ **不要重跑回補來取得資料。** 現有資料已與上游逐筆驗證（OHLCV
88,389/88,389 完全相同），重跑要 5 小時、會撞 FinMind 額度，而且引入
新的不確定性。

⚠️ **傳輸檔只是搬運用的，不要 commit。** `.gitignore` 已涵蓋 `*.db`
與 `*.db.gz`；確認一下你那邊也是。

### 沒有資料的話能做什麼

**任務 H 的第 1~3 步不需要 `history.db`**：複製兩個模組、寫手算值測試、
與既有 `walk_forward.py` 交叉比對切分，全部用合成資料就能做完。
只有第 4 步（接進 `validate_oos_trailing.py` 實跑）要等資料。

E 和 G 一定要等資料。

---

## 第一部分：上一張工作單的回饋

任務 A~C 都做完了，凍結守門做得紮實。四個缺陷已修掉，**請不要退回去**。

### 訓練窗口凍結——你自己修了，而且比我的版本好

我在 review 時發現 `--oos-start` 會連帶永久凍結訓練集，同時你也發現並
推了 `2265e94` + `0d954b5`。**採用你的版本，我的那版已丟棄。**

我把 `--dev-end` 當成不動的硬牆，凍結這個實驗就沒有入口了。
你多加一個 `--freeze-model`，兩個實驗都保住，而且哪一個在跑是寫在
指令上的，不是藏在參數副作用裡。腳本還會把模式印出來——這點做得好。

`dev_end` 的語意我們也不同，同樣採用你的：

| | 我的 | 你的 |
|---|---|---|
| `dev_end` 是什麼 | 釘死的牆 | 初始落後量，隨 fold 前移 |
| 緊貼 `oos_start` 時 | fold 2 起凍結 | 純擴張（等於沒設） |

任務 D 的實際用法 `dev_end = 2023-12-29`、`oos_start = 2024-01-02` 是
**相鄰**的——這時相鄰的 `dev_end` 只是在說「參數是用 2023 以前的資料
選的」，不該連帶改變訓練行為。你對。

**我只補了三個測試**，因為你的測試用 240 個決策日、5 個 fold，而且
`dev_end` 緊貼 `oos_start`（等於沒作用）：

| 測試 | 守什麼 |
|---|---|
| `test_oos_start_alone_expands_across_many_folds` | 320 日、11 個 fold 仍持續擴張 |
| `test_expanding_after_oos_start_still_embargoes_every_fold` | 每個 fold 都**正好**隔離 12 期 |
| `test_early_dev_end_keeps_a_constant_lag_not_a_fixed_wall` | 提早的 `dev_end` 走落後量語意 |

第三個最重要：`dev_end` 緊貼時走不到那條分支，那段語意原本沒人守。

### 我改的三個地方

| # | 檔案 | 改動 | commit |
|---|---|---|---|
| 1 | `forward_predictions.py` | `edge_z` 進主鍵與唯一鍵 | `673e75b` |
| 2 | `data/calendar.py`（新）、`scripts/record_forward.py` | 揭曉日改用實際交易日曆 | `82cbb10` |
| 3 | `forward_predictions.py` | 刪掉重複索引 `ux_forward_prediction_identity` | `673e75b` |

```
651 passed（原 633，+18）
```

### `edge_z` 那個改動的教訓值得記一下

`edge_z` 只加進 UNIQUE 不夠，主鍵也要有。實測：

```
edge_z=0.3 寫入 1 筆
edge_z=0.8 寫入 0 筆   ← 同一次執行 predicted_at 相同，撞主鍵被靜默丟掉
```

我第一版單元測試用不同 `predicted_at` 建兩筆，**測試通過但保護不存在**。
是端到端 smoke test 抓到的。

**測試要守性質，不要守目前的輸出。**

---

## 第二部分：接下來做什麼

> **2026-09-15 修訂**：評估 `HKUDS/Vibe-Trading` 之後新增任務 H（CPCV），
> 並把任務 F（block bootstrap）降級為可選。
>
> 執行順序改為 **H → E → G**，F 有餘力再做。
>
> 為什麼：任務 E 的第二條驗收是「擺盪幅度要降到數十 pp」，原本用
> 「標的池取 140/150/160 跑三次」當量尺——那是土法煉鋼。CPCV 用同一份
> 資料生 15 條路徑，直接給出分布，是正規的量尺。
>
> **沒有量尺就沒辦法驗收 E，所以量尺要先做。**
>
> 完整評估見 `qlib-tw-trader/docs/原理說明/2026-09-15_外部專案評估_AutoHedge與VibeTrading.md`。

| 順序 | 任務 | 狀態 |
|---|---|---|
| 1 | **H：接 CPCV + PBO / DSR** | 新增，先做（它是量尺） |
| 2 | **E：修門檻** | 不變，但驗收改用 CPCV |
| 3 | **G：pseudo-OOS 評估** | 不變 |
| — | F：block bootstrap | **降級為可選** |

---

### 任務 H：接 CPCV + PBO / DSR（先做這個）

#### 為什麼排第一

`03_待辦與改進方向.md` 開頭那張表是整份文件最重要的證據：

```
組合             v6         v7          差異
均值回歸 × 60    +954.87%   +123.93%   −831 pp
均值回歸 × 120   +294.62%   +802.42%   +508 pp
等權對照          +701.37%   +688.79%    −13 pp   ← 對照組只動 13 pp
```

v6 → v7 只補了 23 期缺漏的市值快照，**策略參數完全沒動**。

我當時的結論「不是 bug，是路徑相依」是對的——但**當時沒有辦法量化它**。
單一 walk-forward 只給一條路徑，你無從知道那條路徑有多大代表性。

CPCV（combinatorial purged cross-validation）用同一份資料生出多條路徑，
給出報酬的**分布**而不是點估計。

#### 來源：`HKUDS/Vibe-Trading` 的 `quantlib`

```
external/Vibe-Trading/agent/src/quantlib/crossvalidation.py     585 行
external/Vibe-Trading/agent/src/quantlib/multipletesting.py     568 行
授權   MIT
依賴   numpy / pandas / scipy   ← 與本專案完全相同，不引入新依賴
```

⚠️ `external/` 已加進 `.gitignore`，那是 clone 來研究用的，不是子模組。
**所以你 pull 不到那份程式碼**，要自己取。整包 117 MB，用 sparse checkout
只拿需要的兩個目錄：

```bash
mkdir -p external && cd external
git clone --depth 1 --filter=blob:none --sparse \
  https://github.com/HKUDS/Vibe-Trading.git
cd Vibe-Trading
git sparse-checkout set agent/src/quantlib agent/tests/quantlib
git rev-parse --short HEAD    # 記下這個 commit，要寫進來源註記
```

拿不到網路的話跟我說，我從這邊把兩個檔案傳過去。

我已經用**本專案的實際規模**（370 個決策期、標籤跨度 12 期）實測過：

```
CPCV：15 條路徑（單一 walk-forward 只有 1 條）
  訓練 231｜測試 123｜purged 12｜embargoed 4｜訓練測試重疊 False

purged_walk_forward_splits：4 folds，每 fold 剛好 purge 12 期
  ← 與本專案的 embargo_periods(60, 5) = 12 完全一致

PBO（20 組純雜訊）        0.329
PBO（其中 1 組有真優勢）  0.029     ← 有鑑別力

DSR(observed=0.90, n_trials=20, n_observations=30)
  expected_maximum_sharpe  0.95     ← 掃 20 組，純靠運氣的期望最佳值
  survives                 False    ← 我最好的 0.90 比運氣還低
```

#### 做什麼

**不要整包 vendor 進來。** 12,076 行裡只需要兩個模組。

```
1. 複製 crossvalidation.py + multipletesting.py 到
   taiwan_quant/validation/external/，保留 MIT 授權標頭與來源 URL + commit
2. 為本專案實際會呼叫的函式寫**自己的**測試，比對手算值（禁令 10）
   ⚠️ 不可拿他們的測試當作我們的測試
3. 交叉比對：用 purged_walk_forward_splits 與既有 walk_forward.py
   跑同一組參數，fold 切法應一致（實測 purge 都是 12 期）
4. 再接 CPCV 與 PBO / DSR 到 validate_oos_trailing.py
```

**第 3 步是最便宜的檢查**：兩套獨立實作的切分結果應該一致。
不一致就代表有一邊錯了，而且那是我自己看不出來的錯。

⚠️ **`purged_kfold_splits` 不要用。** K-fold 會拿測試段**之後**的資料
訓練，對估計泛化能力是對的，但違反禁令 5（滾動訓練，只用預測期之前的
資料）。他們模組自己的 docstring 也點明了這件事。

#### 驗收

```
□ 交叉比對通過：purged_walk_forward_splits 與 walk_forward.py 切法一致
□ CPCV 能對現有最佳組合（籌碼跟隨 × 60）產出 15 條路徑的報酬分布
□ 報告並列：中位數、5%/95% 分位、全距
□ PBO 與 DSR 有數字，且 PBO > 0.5 時報告明確寫「判定過擬合」
□ 每個外部函式都有本專案自己的手算值測試
```

⚠️ **CPCV 的 15 條路徑不是 15 個獨立樣本。** 它們共用同一份歷史，只是
切法不同。它回答的是「這條策略對切分方式有多敏感」，不是「多了 14 倍
證據」。**用它否定比用它肯定可靠得多。** 報告裡要寫清楚這一點。

---

### 任務 E：修門檻，讓訊號／成交比降到 10:1

#### 問題

```
籌碼跟隨 × 60 日｜訊號 44,319 筆，成交 102 筆   = 435 : 1
```

**99.8% 的訊號被槽位擋掉。** 60 日持有下期望報酬遠大於成本，幾乎所有
候選都通過 `E[R] − cost > 0`——門檻沒有在篩選任何東西。

實際成交哪 102 筆，取決於「槽位剛好空出來時誰在排隊」。

#### 為什麼這件事必須排在 pseudo-OOS 前面

見任務 H 的 v6/v7 對照表：估計量的抽樣變異蓋過訊號。
**先用不穩定的估計量去燒評估區間，等於白燒。**

#### 做什麼：三個方案並列跑，不要只挑一個

| 方案 | 做法 | 副作用 |
|---|---|---|
| E1 相對排名 | 只取當日期望報酬前 N%（N = 5 / 10 / 20） | 標的池小時樣本更少 |
| E2 拉高 `edge_z` | 1.0 → 1.5 / 2.0 / 2.5 | 可能完全沒訊號 |
| E3 定期全部換倉 | 每 60 日重選 Top 3，不等槽位 | 換手率上升，成本要重算 |

⚠️ **E3 的成本必須重算，不可沿用現有數字。** 換手率是一級輸出
（CLAUDE.md 規格 13）。

⚠️ **這三個都是策略參數。只能在開發集 2019-01 ~ 2023-12 上掃，
不准碰 2024 之後。** 掃參數時用 `--end 2023-12-29`。

⚠️ **不要用 `--dev-end 2023-12-29 --oos-start 2024-01-02` 來掃參數。**
那組參數會照樣跑完 2024-2026 並把結果印出來——你就看到了。
`--dev-end` 管的是訓練資料，`--end` 管的才是「載不載入」。掃參數階段
評估區間必須**根本不進記憶體**。

#### 驗收（兩個都要達到，缺一不可）

```
□ 訊號／成交比 ≤ 10:1
□ 路徑變異：CPCV 15 條路徑的總報酬全距，從數百個百分點降到數十個百分點
```

第二條用**任務 H 做好的 CPCV** 量——這就是為什麼 H 要先做。

```bash
# 修門檻前先量一次當基準線，修完再量一次
.venv/bin/python scripts/validate_oos_trailing.py \
  --horizons 60 --cpcv --end 2023-12-29
```

> **2026-09-15 修訂**：原本這條寫「標的池取 140/150/160 跑三次看全距」。
> 那是土法煉鋼——三個點估計不構成分布，而且改標的池同時動到了訊號來源，
> 分不清變異來自門檻還是來自換股。CPCV 固定資料、只換切分方式，量到的
> 才是門檻造成的路徑敏感度。

**只達成第一條不算做完。** 訊號／成交比是手段，路徑變異才是目的。

---

### 任務 F：block bootstrap 信賴區間（**可選，降級**）

> **2026-09-15 降級。** H → E → G 做完有餘力再做，不做也不影響結論。

#### 為什麼降級

原本的理由是「有效獨立樣本只有 15~30，需要信賴區間」。理由仍然成立，
但**它量錯了東西**：

```
block bootstrap   在「同一條權益曲線」上估變異
CPCV              在「切分方式」上估變異
```

v6/v7 那張表證明，害我擺盪 −831 pp 的**不是隨機性**，是資料完整性造成
的路徑改變。那是切分敏感度，不是抽樣雜訊。**任務 H 才是對症的那個。**

#### 真要做的話

對權益曲線做 block bootstrap，**區塊長度取一個持有期**（60 日持有就用
60 個交易日的區塊），產出總報酬與 Sharpe 的信賴區間。

放 `taiwan_quant/validation/bootstrap.py`，比照 `rank_ic.py` 的形狀：
純函式 + dataclass 報告 + `describe()`。

⚠️ **不要直接抄 Vibe-Trading 的 `timeseries.bootstrap_sharpe`。** 它是
逐點 IID 重抽：

```python
bootstrap_stats[i] = statistic_func(sample[rng.integers(0, n, size=n)])
#                                          ^^^ 每一筆獨立抽，打散了時序
```

報酬有自相關，IID 重抽會**低估**信賴區間寬度——區間看起來比實際窄，
結論比實際有把握。介面設計（回傳 `point_estimate` / `ci_lower` /
`ci_upper` / `bootstrap_std` 的 dict）可以參考，重抽邏輯必須換成區塊。

#### 驗收

```
□ 能回答「+1022% 的 95% 信賴區間是多少」
□ 區塊長度可設定，且有測試證明區塊長度=1 時退化成 iid bootstrap
□ 有測試證明：自相關序列下，區塊版的區間比 iid 版寬
□ 下界低於買進持有時，報告要寫「無法區分」，不可只印點估計
```

---

### 任務 G：pseudo-OOS 評估（最後才做）

這是上一張工作單的任務 D。**做完 H 與 E 才做。**

```
開發集    2019-01-02 ~ 2023-12-29    可反覆看
評估段    2024-01-02 ~ 2026-09-11    ← 已被看過 7 次
```

#### 驗收標準跟平常不一樣：只能用來否定

```
✗ 評估段報酬為負                → 淘汰
✗ 回撤遠大於開發集               → 淘汰（過擬合的典型症狀）
✗ Sharpe 比開發集掉一半以上       → 淘汰
✓ 都沒發生                      → 「未被否定」，不是「已被證明」
```

理由是統計力不足：

```
評估段 2.7 年 → 有效獨立樣本 10（60 日持有）
能偵測的最小 Sharpe = 2/sqrt(2.7) = 1.22
目前最佳組合 Sharpe 0.92 ~ 1.13   ← 正好在偵測不到的區間
```

**報告必須明確標示「此區間先前已被看過 7 次，不是全新 OOS」。**
你上次提出這一點是對的，不要因為做完就淡化它。

#### 指令

```bash
.venv/bin/python scripts/validate_oos_trailing.py \
  --horizons 60 120 \
  --dev-end 2023-12-29 \
  --oos-start 2024-01-02
```

輸出寫進 `docs/需求規劃/202609/07_pseudoOOS評估結果.md`。

---

## 明確不要做的事

| 不要做 | 為什麼 |
|---|---|
| 把 `--oos-start` 改回凍結訓練集 | 見第一部分。要凍結用 `--freeze-model` |
| 把 `edge_z` 從主鍵拿掉 | 同一次執行掃參數會靜默丟資料 |
| 用 `pd.offsets.BDay` 算任何台股日期 | 過一次農曆年差 14 天 |
| 在 2024 年之後的資料上掃參數 | 觸犯 CLAUDE.md 禁令 6。掃參數一律加 `--end 2023-12-29` |
| 用 `--dev-end` 當掃參數的防線 | 它只擋訓練資料，2024+ 的結果照樣印出來 |
| 只做任務 E 的第一條驗收就收工 | 訊號／成交比是手段，擺盪幅度才是目的 |
| 跳過 H、E 直接做 G | 用不穩定的估計量燒評估區間，等於白燒 |
| 用 `purged_kfold_splits` | K-fold 會拿測試段之後的資料訓練，違反禁令 5 |
| 直接抄他們的 `bootstrap_sharpe` | 是 IID 重抽，對自相關報酬會低估區間寬度 |
| 把 `external/` commit 進版控 | 那是 clone 來研究的，已在 `.gitignore` |
| 重跑資料回補 | 資料已逐筆驗證（OHLCV 88,389/88,389 相同） |
| 硬編碼任何 token | 走環境變數。程式已有佔位字串偵測 |

---

## 驗收清單

```
□ 651 個測試仍全過（不可為了通過而改測試的預期值）

任務 H（先做）
□ 交叉比對通過：purged_walk_forward_splits 與 walk_forward.py 切法一致
□ CPCV 能對籌碼跟隨 × 60 產出 15 條路徑的報酬分布
□ 報告並列中位數、5%/95% 分位、全距
□ PBO 與 DSR 有數字；PBO > 0.5 時明確寫「判定過擬合」
□ 報告寫明「15 條路徑不是 15 個獨立樣本」
□ 每個外部函式都有本專案自己的手算值測試（不可沿用他們的測試）

任務 E
□ 訊號／成交比 ≤ 10:1
□ CPCV 15 條路徑的總報酬全距 ≤ 數十 pp（修門檻前先量一次當基準線）
□ 三個方案（E1/E2/E3）都有數字，不是只挑一個
□ E3 的換手率與年化成本拖累有並列（規格 13）

任務 G
□ pseudo-OOS 報告寫明「已被看過 7 次」

全部
□ 每項改動都有對應的 qlib-tw-trader/docs/原理說明/YYYY-MM-DD_*.md
```

---

## 環境備忘

```bash
cd "/Volumes/Mac/Quantitative Trading/taiwan-quant"
.venv/bin/python -V             # Python 3.12.12
.venv/bin/python -m pytest -q   # 目前 651 passed
```

- 資料：`data/history.db`（967 MB，2015-01 ~ 2026-09，1,218 檔含 159 檔下市）
- 完整 OOS 回測約 25 分鐘／組，**先用 `scripts/diagnose_rank_ic.py` 快篩**
  （幾分鐘，能先擋掉沒有排序能力的組合）
- pandas 3.0 copy-on-write：就地改寫唯讀陣列會壞，新程式一律回傳新物件
- 不需要 FinMind token，E~G 全部本機完成

---

⚠️ 本文件為工程交接，不構成投資建議。
