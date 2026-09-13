# 驗證結果：qlib-tw-trader 兩步實測

- 日期：2026-09-11 ~ 2026-09-12
- 依據：[評估_qlib-tw-trader.md](評估_qlib-tw-trader.md) 第六節「採用後的第一批待辦」第 1、2 步
- 原始紀錄：[驗證/](驗證/)

---

## 結論（先講）

| 步驟 | 結果 |
|---|---|
| ① Look-ahead 防護驗證 | ✅ **通過**。23 個測試全過，含正控制組確認測試有鑑別力 |
| ② 接上成本重跑回測 | ❌ **觸發紅線**。毛報酬 +26.27% → 扣成本 **−19.08%**，同期市場 +81.67% |

**[01_決策紀錄.md](01_決策紀錄.md) D1 設下的紅線已觸發**：

> 紅線提前到第 2 步：成本接上去之後如果 Sharpe 從 1.724 掉到跑不贏 0050，就不要再往下投資工時。

**建議：停止對 `src/services/walk_forward_backtester.py` 這條產品化路徑投入工時。** 但資料層與因子庫的沿用價值不變（見最後一節）。

---

## 步驟 ①：Look-ahead 防護 — 通過

### 做法

不採信 README 自稱，用三層獨立驗證：

| 層 | 方法 | 檔案 |
|---|---|---|
| 1 靜態掃描 | 掃全部 303 個因子公式，找負數 `Ref`（未來參照） | `tests/test_lookahead_audit.py` |
| 2 語意錨定 | 對 label 與合成作弊公式做正控制，證明掃描器抓得到 | 同上 |
| 3 時序契約 | 進場偏移、embargo、訓練/驗證切分的常數一致性 | 同上 |
| 4 動態截斷 | **物理刪掉未來資料**重新匯出，因子值必須不變 | `tests/test_lookahead_truncation.py` |

### 結果：23 passed

```
[Layer 1] 靜態掃描：303 個因子定義
  ✓ PASS — 無任何因子使用負數 Ref（未來參照）
  ✓ PASS — 無可疑運算子 ('Future', 'Shift(-', 'Lead(')
  掃描範圍：{'technical': 136, 'chips': 97, 'revenue': 10, 'interaction': 60}

[Layer 2] 語意錨定：掃描器自我驗證
  ✓ PASS — 掃描器在 label 中命中未來參照 [-3, -1]
  ✓ PASS — 合成作弊公式偵測
  ✓ PASS — 過去參照未被誤判

[Layer 3] 時序契約
  ✓ PASS — 進場非當日收盤：LABEL_ENTRY_OFFSET = 1（T+1 進場）
  ✓ PASS — label 與偏移一致：label 偏移 [1, 3] vs 常數 [1, 3]
  ✓ PASS — embargo 覆蓋 label 跨度：EMBARGO_DAYS = 7 >= label 跨度 3
  ✓ PASS — 訓練期大於驗證期：TRAIN_DAYS = 504, VALID_DAYS = 100
```

動態截斷測試（10 個抽樣因子，跨不同視窗長度與運算子）：

```
正控制組（必須 CHANGED）
  ✓ __label__              CHANGED   full=-0.02395  trunc=None
  ✓ __cheat_next_close__   CHANGED   full=0.03942   trunc=None

受測因子（必須 IDENTICAL）
  ✓ kbar_kmid / roc_5 / roc_60 / ma_20 / std_20
  ✓ rsv_20 / corr_close_vol_20 / beta_20 / vol_ma_20 / qtlu_60
    全部 IDENTICAL
```

正控制組 CHANGED 是關鍵——它證明截斷真的生效，所以那 10 個 IDENTICAL 不是假通過。

### 附帶發現：qlib 的 `end_time` 不能當截斷手段

```python
# end_time 設在 T，label 仍算得出來（讀到 T 之後的資料）
D.features(["2330"], ["Ref($close,-3)/Ref($close,-1)-1"], end_time="2026-06-30")
# → -0.023952...   與不截斷完全相同
```

`end_time` 只裁切輸出範圍，運算式仍會讀底層儲存中的未來資料。任何用 `end_time` 做的「截斷測試」都是無效的。必須物理上不要把未來資料寫進資料集。

**這條經驗直接適用於我們自建系統的 look-ahead 掃描器設計。**

---

## 步驟 ②：接上成本重跑回測 — 觸發紅線

### 前置：把系統跑到能訓練，踩了 4 個坑

| # | 問題 | 性質 | 處理 |
|---|---|---|---|
| 1 | 303 因子 dropna 後樣本歸零，LightGBM 收到空資料集 | 資料不完整 + repo 無防護 | 新增 `scripts/disable_sparse_factors.py`，停用 63 個、剩 240 個 |
| 2 | LightGBM 寫死 `device: "gpu"`，pip wheel 是 CPU-only | repo 可攜性 bug | 改 `_detect_lgb_device()` 開機實測 |
| 3 | `assignment destination is read-only` | pandas 3.0 copy-on-write 不相容 | `double_ensemble.py` 改在可寫副本上 permute |
| 4 | FinMind 匿名額度撞牆（每小時 600 次） | 外部限制 | PER/月營收/集保/借券資料不完整，相關因子已停用 |

坑 4 實證了 Kimi 在原始對話中遇到的同一問題（見 [來源原文/06_Kimi](來源原文/06_Kimi_台股量化交易方案.md)）。

### 訓練規模

- 資料：100 檔、893 交易日（2023-01-03 ~ 2026-09-11）
- 因子：240 個（原 303 個，停用 63 個）
- 模型：**20 個**（2026W16 ~ 2026W35），序列訓練，**總耗時 78.4 分**（平均 4.1 分/模型）
- 回測：20 週，Top-10，日度調倉（原專案 API 預設 `topk` 策略），初始資金 40 萬

### 最重要的發現：valid IC 對 live IC 沒有預測力

```
平均 valid IC（驗證期）  +0.0362
平均 live IC（樣本外）   -0.0263      ← 負的
IC 衰減                  172.6%
valid/live IC 相關係數   -0.1588      ← 負相關
```

**這是比虧錢更嚴重的問題。** valid IC 是 Walk-Forward 用來選模型、選超參數的唯一依據。它與 live IC 的相關係數是 **−0.159**，意思是：

> 驗證期表現好的模型，樣本外表現反而略差。

換句話說，**這套流程的模型選擇機制在本次區間內是失效的**，不是「效果打折」而是「方向相反」。

訓練過程中 valid IC 的時序也印證了同一件事：

```
W16 0.0529  W17 0.0659  W18 0.0703  W19 0.0712   ← 高點
W28 0.0204  W29 0.0065  W31 0.0009               ← 幾乎歸零
W33 0.0107  W34 0.0262  W35 0.0244
```

後段週別的驗證期落在 2026-04 ~ 08，IC 掉到 0.01 以下。**因子優勢在 2026 年 4 月之後崩掉。**

### 成本情境對照（20 週實測）

| 情境 | 累積 | 年化 | Sharpe | MaxDD | 勝率 | 年化成本 |
|---|---|---|---|---|---|---|
| 無成本（毛報酬） | **+26.27%** | +83.40% | **1.047** | 41.24% | 60.0% | 0.00% |
| 原專案模型（手續費+證交稅，無滑價） | +0.41% | +1.06% | 0.566 | 47.16% | 60.0% | 59.58% |
| **6折 + 0.3% 滑價（0050）** | **−19.08%** | −42.34% | **0.113** | 52.16% | 60.0% | 115.56% |
| 6折 + 0.4% 滑價（0051 中型股） | −25.97% | −54.25% | −0.073 | 54.09% | 55.0% | 138.59% |
| 無折扣 + 0.3% 滑價 | −23.09% | −49.46% | 0.007 | 53.27% | 55.0% | 128.69% |
| **市場基準（等權 100 檔）** | **+81.67%** | +372.23% | — | — | — | — |

三條關鍵讀數：

1. **成本吃掉 45.36 個百分點**（+26.27% → −19.08%），Sharpe 從 1.047 掉到 0.113
2. **光是「原專案模型 vs 加滑價」的差距就有 19.5 個百分點**（+0.41% → −19.08%）。原專案不含滑價這件事不是小數點問題
3. **對市場超額 −100.76 個百分點**

### 換手率實測：是 README 的 27.4 倍

```
平均週換手率（單邊）   271.5%
年化來回次數           141.2
年化成本拖累           115.56%
```

README 宣稱最佳策略 `HoldDrop(K=10,H=3,D=1)` 週換手率 **9.9%**。本次實測 **271.5%**，是其 **27.4 倍**。

原因：API 預設的 `topk` 策略是**日度調倉**，每個預測日都重選 Top-10。10 檔裡每天換掉 2~3 檔，累積起來就是週換手 271%。這在台股 1.071%/趟的成本結構下完全不可執行。

這也解釋了為什麼 README 的數字看起來好：它報的是 `HoldDrop` 策略（有持有期參數 `H` 壓低換手），而**產品化路徑（Dashboard/API）跑的是換手率 27 倍的 `topk`，且完全不扣成本**。

---

## 公平起見：本次量測的四個限制

必須把不利於結論的因素也講清楚：

1. **只有 20 週**，樣本極少，統計上不足以下結論。README 的 156 週需約 16 小時訓練，本次只跑了 78 分鐘。
2. **用的是 `topk` 日度調倉，不是 README 報最佳績效的 `HoldDrop`。**換手率與成本拖累因此高出很多。若改跑 HoldDrop，成本拖累會從 115% 降到約 5.5%（解析估算）。
3. **240/303 因子**。因 FinMind 額度限制停用 63 個（籌碼 27、營收 10、交互 24、技術 2）。
4. **這 20 週市場漲 81.67%**（等權 100 檔，20 週內），是極端多頭。集中持 10 檔的策略在這種全面普漲的環境下跑輸大盤，本身不能單獨當成「沒有 alpha」的證據。

**但第 4 點不能解釋 live IC 為負。** IC 是橫斷面排名相關係數，與市場整體漲跌無關。live IC = −0.0263 與 valid/live 相關係數 −0.159 這兩個數字，不受多頭環境影響。這是本次最站得住腳的負面證據。

---

## 對決策的影響

### 維持沿用的部分（價值不變）

| 項目 | 理由 |
|---|---|
| 資料層（TWSE + FinMind + yfinance 三源降級） | 實測可跑，涵蓋 D6 需求。修補 1 處 CSV 解析後全線打通 |
| 303 因子庫定義 | 通過 look-ahead 驗證，自己寫至少 2 週 |
| Walk-Forward 骨架（504/100/embargo 7、T+1 進場） | 時序契約正確，這是最容易寫錯的部分 |
| `scripts/evaluate_models.py` 的成本函式 | 幾乎逐字命中 D3 規格（含 `MIN_COMMISSION = 20`） |
| `scripts/` 的 9 種策略變體（含 HoldDrop） | 控換手率的實作範本 |

### 不再投入的部分

| 項目 | 理由 |
|---|---|
| `src/services/walk_forward_backtester.py` 這條產品化路徑 | 日度調倉 + 無成本，兩個設計都要重寫，不如自建 |
| 直接沿用其模型選擇機制 | valid/live IC 相關係數 −0.159，選模型的依據失效 |
| DoubleEnsemble 當第一版模型 | 先建立可信的 baseline 與 PBO 校正，再談複雜模型 |

### 新增到自建系統的規格要求

本次實測逼出四條原本沒有的規格：

1. **換手率必須是回測的一級輸出**，不是事後才算。任何策略報告都要並列週換手率與年化成本拖累。
2. **valid/live IC 相關係數必須監控**。若接近 0 或為負，代表模型選擇機制失效，這時調參數是在調噪音。
3. **成本敏感度必須至少三檔並列**：無成本 / 含手續費稅 / 含滑價。這次「+0.41% vs −19.08%」的差距全部來自滑價。
4. **look-ahead 掃描器不能用 `end_time` 做截斷**，必須物理重新匯出資料集。

### 更新 D7：策略族的持有期必須參數化

原本 D7 只寫「5 日 triple-barrier」。本次實測顯示**持有期／調倉頻率是成本的主導因素**，必須從一開始就是策略的一級參數，而不是事後調整：

```
週換手率 9.9%   → 年化成本   5.51%   （HoldDrop，可執行）
週換手率 271.5% → 年化成本 115.56%   （日度調倉，不可執行）
```

---

## 重跑這些驗證的指令

```bash
cd "/Volumes/Mac/Quantitative Trading/qlib-tw-trader"

# 步驟 ①
.venv/bin/python -m pytest tests/test_lookahead_audit.py tests/test_lookahead_truncation.py -v
PYTHONPATH=. .venv/bin/python tests/test_lookahead_audit.py        # 中文報告
PYTHONPATH=. .venv/bin/python tests/test_lookahead_truncation.py

# 成本模型單元測試（預期值都手算在註解裡）
.venv/bin/python -m pytest tests/test_costs.py -v
PYTHONPATH=. .venv/bin/python tests/test_costs.py                  # 成本對照表

# 步驟 ②（需已訓練模型）
PYTHONPATH=. .venv/bin/python scripts/compare_cost_impact.py 2026W16 2026W35 --capital 400000

# 重新訓練（19 個模型約 78 分）
PYTHONPATH=. .venv/bin/python scripts/train_weeks.py 2026W16 2026W35 --skip-trained
```

## 原始紀錄檔

| 檔案 | 內容 |
|---|---|
| `驗證/03_lookahead_pytest.log` | 步驟① pytest 輸出 |
| `驗證/04_lookahead_report.txt` | 步驟① 中文報告 |
| `驗證/06_truncation_report.txt` | 動態截斷測試 |
| `驗證/07_lookahead_pytest_full.log` | 兩支合跑 23 passed |
| `驗證/08_cost_model_table.txt` | 成本模型對照表（解析估算） |
| `驗證/11_disable_sparse_factors.log` | 停用了哪 63 個因子與原因 |
| `驗證/12_backtest_raw.json` | 回測 API 原始回應 |
| `驗證/13_cost_impact_report.txt` | **步驟② 成本情境對照（本文件主要依據）** |
| `qlib-tw-trader/scripts/output/train_weeks_*.jsonl` | 20 個模型逐一訓練紀錄 |
| `qlib-tw-trader/scripts/output/cost_impact_*.json` | 成本對照原始數據 |

入門文件：[qlib-tw-trader/docs/QUICKSTART.zh-TW.md](../../../qlib-tw-trader/docs/QUICKSTART.zh-TW.md)

---

⚠️ 本文件為軟體工程驗證紀錄，不構成投資建議。所有回測數字為研究用途，樣本量不足以支持任何投資決策；回測績效不代表未來表現。
