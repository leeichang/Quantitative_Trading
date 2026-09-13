# Codex 工作單：OOS 凍結機制

- 日期：2026-09-13
- 對應：[03_待辦與改進方向.md](03_待辦與改進方向.md) 第 1 項
- 執行者：Codex（或任何要接手這項的人）

---

## 先解決你提出的三個阻塞

### 阻塞 1：「本機沒有 history.db、data.db 只有 4KB」→ **不成立**

檔案在，而且是完整的。你很可能看到了 **WAL 模式的附屬檔**：

```
history.db        967 MB   ← 主檔
history.db-shm     32 KB   ← 共享記憶體（你可能看到這個）
history.db-wal      0 B    ← 預寫日誌
```

先自己驗一次：

```bash
ls -l "/Volumes/Mac/Quantitative Trading/taiwan-quant/data/"
sqlite3 "/Volumes/Mac/Quantitative Trading/taiwan-quant/data/history.db" \
  "SELECT COUNT(*), COUNT(DISTINCT stock_id), MIN(date), MAX(date) FROM stock_daily;"
```

預期輸出：

```
2914166|1218|2015-01-05|2026-09-11
```

`qlib-tw-trader/data/data.db` 同理，應為 `88401|100|2023-01-03|2026-09-11`。

**如果你在另一台機器上**：那不是「缺資料」是「要搬過去」。967 MB 用
rsync 傳比重跑 5 小時回補快得多，而且避免 FinMind 額度問題。

⚠️ **不要重跑回補。** 現有資料已與上游逐筆驗證過（OHLCV 88,389/88,389
完全相同），重跑只會引入新的不確定性。

### 阻塞 2：「`--oos-start` 參數不存在」→ **成立，這是你要做的第一件事**

目前 OOS 起點是推導出來的，不是指定的：

```python
# scripts/validate_oos_trailing.py:357
first_train = FIRST_TRAIN_DAYS // DECISION_STRIDE   # 750 // 5 = 150
oos_start = dates[first_train]                       # ← 第 150 個決策日
```

`--start` / `--end` 是**資料載入範圍**，不是切分點。

### 阻塞 3：「2024-01~2026-09 已被看過 7 次」→ **成立，而且是我先講的**

見 [03_待辦與改進方向.md](03_待辦與改進方向.md) 的「我跑了 7 次」表格。

**所以不要把它當成全新 OOS。** 處理方式見下面的任務 C。

---

## 三個任務，按順序做

### 任務 A：加切分參數（必做，最小改動）

`scripts/validate_oos_trailing.py` 加兩個參數：

```python
parser.add_argument("--dev-end", default=None,
                    help="開發集結束日（ISO）。之後的資料只在 --oos-start 指定時使用")
parser.add_argument("--oos-start", default=None,
                    help="OOS 起始日（ISO）。指定時覆蓋由 FIRST_TRAIN_DAYS 推導的起點")
```

**修改點在 `run_walk_forward`**：目前 `oos_start = dates[first_train]`，
改成「若呼叫端指定了日期，就用最接近且不早於它的決策日」。

驗收：

```bash
.venv/bin/python scripts/validate_oos_trailing.py \
  --horizons 60 --oos-start 2024-01-02
# 輸出的「OOS 期間」必須顯示 2024-01-02 起，不是 2019-01-29
```

**必須加測試。** 至少三個：

| 測試 | 守什麼 |
|---|---|
| 指定 `--oos-start` 時，訓練集不含該日之後的決策 | 切分正確 |
| 未指定時行為與現在完全相同 | 不破壞既有結果的可重現性 |
| `--oos-start` 早於暖機結束時拋錯 | 不可用不足的歷史訓練 |

---

### 任務 B：凍結守門（最重要，也最便宜）

**問題**：我看了 7 次 OOS，每次都有「好理由」。光靠自律不管用。

在 `taiwan_quant/data/loader.py` 加：

```python
FROZEN_DATA_START = date(2026, 9, 14)
"""
凍結起點。資料實際只到 2026-09-11，所以這條線目前不會擋到任何東西——
它是為**未來新增的資料**而設的。

載入超過這個日期的資料需要明確傳 `unlock_frozen=True`，而且每次解鎖
都要寫進 log。目的是讓「又偷看了一次」變成拋錯，而不是靜默發生。
"""
```

`load_prices()` 加參數 `unlock_frozen: bool = False`：

- `end` 超過 `FROZEN_DATA_START` 且未解鎖 → 拋 `DataNotAvailableError`，
  訊息要說明「這是凍結區間，解鎖請傳 unlock_frozen=True 並記錄理由」
- 解鎖時 append 一筆到 `data/frozen_access.log`：時間戳、呼叫端、理由

**驗收**：

```
□ 不解鎖載入 2026-09-20 的資料 → 拋錯
□ 解鎖後可載入，且 log 多一行
□ 載入 2026-09-11 以前的資料完全不受影響（現有 619 個測試全過）
```

⚠️ **不要把凍結日設成 2024-01-01。** 那會擋掉現有所有測試與腳本。
凍結的是**未來**，不是過去。

---

### 任務 C：前推預測記錄（唯一會收斂的東西）

這是三項裡最有長期價值的。

**建一張表 + 一個指令**：

```sql
CREATE TABLE forward_predictions (
    predicted_at   TEXT NOT NULL,   -- 預測產生的實際時間
    data_asof      TEXT NOT NULL,   -- 用到的資料截止日
    strategy_version TEXT NOT NULL, -- 程式版本（禁令 7、8）
    family         TEXT NOT NULL,
    horizon        INTEGER NOT NULL,
    stock_id       TEXT NOT NULL,
    rank           INTEGER NOT NULL,
    score          REAL NOT NULL,
    expected_return REAL,
    trail_pct      REAL NOT NULL,
    entry_price    REAL NOT NULL,
    due_date       TEXT NOT NULL,   -- 標籤揭曉日
    realized_return REAL,           -- 到期後回填
    settled_at     TEXT,
    PRIMARY KEY (predicted_at, family, horizon, stock_id)
);
```

兩個指令：

```bash
scripts/record_forward.py           # 產生預測並存檔
scripts/record_forward.py --settle  # 對已到期的預測回填實際報酬
```

**為什麼這件事最重要**：時間往前走，樣本自己會長，而且**從來沒有被
看過**。60 日持有一年累積 4 個乾淨樣本。

```
1 年 →  4 個樣本   能偵測 Sharpe > 2.0
4 年 → 17 個樣本   能偵測 Sharpe > 1.0
6 年 → 25 個樣本   能偵測 Sharpe > 0.82
```

慢，但這是唯一會收斂的路。**越早開始越好，所以先做。**

---

## 任務 D（可選）：pseudo-OOS 評估

做完 A~C 之後才做這個，而且**驗收標準必須改**。

```
開發集    2019-01-02 ~ 2023-12-29    可反覆看
評估段    2024-01-02 ~ 2026-09-11    ← 已被看過 7 次
```

**只能用來否定，不能用來證明**：

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
你提出這一點是對的，不要因為做完就淡化它。

---

## 明確不要做的事

| 不要做 | 為什麼 |
|---|---|
| 重跑資料回補 | 資料在，且已逐筆驗證。重跑只引入新變數 |
| 把凍結日設成 2024-01-01 | 會擋掉現有 619 個測試 |
| 調 `PULLBACK_QUANTILE` / `edge_z` / 槽位數 | 觸犯 CLAUDE.md 紅線。要調就先用掉凍結 OOS |
| 把 pseudo-OOS 的結果稱為「樣本外驗證」 | 它被看過 7 次 |
| 在任何地方硬編碼 FinMind token | 走環境變數。程式已有佔位字串偵測 |

---

## 實作補充：切分點與模型凍結是兩件事

`--oos-start` 只指定 OOS 從哪個決策日開始，預設維持擴張式
walk-forward：前一個 fold 的結果揭曉後，可成為下一個 fold 的訓練資料。

```bash
# 實際持續運作方式：訓練集隨 fold 擴張（預設）
.venv/bin/python scripts/validate_oos_trailing.py \
  --horizons 60 --oos-start 2024-01-02

# 研究模型完全不更新時的衰退：必須明確指定
.venv/bin/python scripts/validate_oos_trailing.py \
  --horizons 60 --oos-start 2024-01-02 --freeze-model
```

`FROZEN_DATA_START = 2026-09-14` 凍結的是未來資料存取，不是模型訓練
窗口。現有資料只到 2026-09-11，所以目前不會自然觸發守門；等新增
2026-09-14 起的資料後，才需要明確解鎖並留下理由。

---

## 驗收清單

```
□ 確認 history.db 存在且為 2,914,166 列（不是重建）
□ --oos-start / --dev-end 參數可用，且未指定時行為不變
□ 切分邏輯有 3 個以上測試
□ FROZEN_DATA_START 守門生效，解鎖會寫 log
□ 現有 619 個測試全部仍然通過
□ forward_predictions 表建立，record_forward.py 可產生與結算
□ （可選）pseudo-OOS 報告明確標示「已被看過 7 次」
```

---

## 環境備忘

```bash
cd "/Volumes/Mac/Quantitative Trading/taiwan-quant"
.venv/bin/python -V           # Python 3.12.12
.venv/bin/python -m pytest -q # 目前 619 passed
```

pandas 3.0 的 copy-on-write 會讓就地改寫唯讀陣列直接壞掉——
**新程式一律回傳新物件**。

不需要 FinMind token：A~D 全部在本機完成，零網路請求。

---

⚠️ 本文件為工程交接，不構成投資建議。
