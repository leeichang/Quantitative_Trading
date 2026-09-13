# D1 評估報告：qlib-tw-trader

- 評估日期：2026-09-11
- 依據：[01_決策紀錄.md](01_決策紀錄.md) D1
- 專案：https://github.com/Docat0209/qlib-tw-trader（MIT，commit `774b3fb`）
- 本機路徑：`/Volumes/Mac/Quantitative Trading/qlib-tw-trader`
- 評估方式：**實際安裝並跑起來**，不只讀文件

---

## 一、結論

### **部分沿用**

沿用它的**資料層 + 303 因子庫 + Walk-Forward 骨架 + `scripts/` 的成本函式**；
自建**5 日 triple-barrier 標記 + 150 檔標的池 + 交易計畫輸出 + PBO 校正 + Telegram 推播**。

**理由一句話**：它把最花時間、最容易寫錯的部分（台股多源資料同步、303 個因子定義、embargo 正確的 walk-forward）都做完了，但**產品化路徑（API/Dashboard）的回測不扣任何交易成本**，而那正是我們最在意的紅線。

| 項目 | 判定 |
|---|---|
| 資料層（TWSE + FinMind + yfinance 三源降級） | ✅ 沿用（需 1 處修補，已修） |
| 303 因子庫 + 因子管理 | ✅ 沿用 |
| Walk-Forward 骨架（504/100/embargo 7） | ✅ 沿用 |
| `scripts/evaluate_models.py` 成本函式 | ✅ 沿用並補滑價 |
| 9 種策略變體（含 HoldDrop 控換手） | ✅ 沿用 |
| DoubleEnsemble + Optuna | ✅ 沿用 |
| React Dashboard（9 頁） | ⚠️ 白拿，但非必需 |
| `src/services/walk_forward_backtester.py` | ❌ **必須改**：完全不扣成本 |
| Label（2 日報酬） | ❌ 改為 5 日 |
| 標的池（市值前 100） | ❌ 改為 0050+0051 = 150 |
| PBO / Deflated Sharpe | ❌ 沒有，自建 |
| 交易計畫輸出（entry/target/stop） | ❌ 沒有，自建 |
| Telegram 推播 | ❌ 沒有，自建 |

---

## 二、安裝實錄（含所有踩到的坑）

環境：macOS Darwin 25.3.0 / Apple Silicon

| # | 問題 | 解法 | 嚴重度 |
|---|---|---|---|
| 1 | 系統 `python3` 是 3.9.6，專案需 ≥3.10 | 用 `/opt/homebrew/bin/python3.12` 建 venv | 低 |
| 2 | 系統 `node` 是 v16，Vite 6 需 ≥18 | `nvm use 22.14.0` | 低 |
| 3 | Docker daemon 沒開 | 改走手動安裝路徑 | 低 |
| 4 | `lightgbm` 匯入失敗：`Library not loaded: @rpath/libomp.dylib` | `brew install libomp` | 中 |
| 5 | **`requirements.txt` 漏了 `yfinance`**，`app.py` 直接無法匯入 | `pip install yfinance` | 中（repo bug） |
| 6 | **`STOCK_DAY_ALL` 端點已改回傳 CSV**，程式當 JSON 解 → 整條管線第一步就 500 | 自行修補（見第四節） | **高（repo bug）** |

`pip install -r requirements.txt` 會連帶拉進 `jupyter` + `jupyterlab` + `mlflow`（pyqlib 的相依），下載量大、約 10 分鐘。實際裝出的版本是 **pandas 3.0.5 / numpy 2.5.3 / pyqlib 0.9.7**，目前未見相容性問題。

### 驗證結果

```
pytest                     26 passed, 2 failed
                           （2 個失敗均為 STOCK_DAY_ALL 的 CSV 問題，修補後成因已解除）
後端  uvicorn :8000        ✅ /api/v1/system/health → {"status":"ok"}
前端  vite :3000           ✅ Dashboard 正常渲染，無 console error
因子庫 POST /factors/seed   ✅ inserted 303
標的池 POST /universe/sync  ✅ total 100（台積電市值 624,970 億、rank 1）
交易日曆 POST /sync/calendar ✅ new_dates 23
日K同步 POST /sync/bulk      ✅ total 1,379、inserted 100
三大法人 POST /sync/institutional/bulk ✅ total 16,407、inserted 100（走 TWSE，不需 FinMind）
月營收  POST /sync/monthly-revenue/stock/2330 ✅ fetched 81、inserted 80（FinMind 匿名即可）
```

**FinMind token 不是必要的**（`token_loaded: false` 仍可抓月營收），匿名額度足夠做初期評估。這修正了 D6 的假設：初期連免費註冊都可以省。

---

## 三、D1 檢查清單逐項回答

### 1. 資料層是否可換成 FinMind？還原股價與公布日怎麼處理？

**可，而且它本來就三源並用**，優先序與我們 D6 的選擇完全一致：

| 優先序 | 來源 | 涵蓋 |
|---|---|---|
| 1 | TWSE OpenAPI | OHLCV、PER/PBR、三大法人、融資券、外資持股 |
| 2 | FinMind | 月營收、財報 |
| 3 | yfinance | **還原收盤價** |

- **還原股價**：有獨立的 `/api/v1/sync/adj/*` 端點，走 yfinance 抓還原收盤價。這比我們附件裡兩支 `.py` 用未還原價好。
- **公布日對齊**：`EMBARGO_DAYS = 7` 用來防 label lookahead。月營收有 `missing_months` 追蹤。但**沒有看到 `announce_date` 欄位級別的對齊**，需進一步查證（列為待確認項）。

### 2. 成本模型有沒有含台股證交稅 0.3%？費率可不可配置？

**分兩套，答案相反 —— 這是本次評估最重要的發現。**

`scripts/evaluate_models.py:45-50`：

```python
# 交易成本（玉山證券，電子下單 6 折）
COMMISSION_RATE = 0.001425
COMMISSION_DISCOUNT = 0.6
EFFECTIVE_COMMISSION = COMMISSION_RATE * COMMISSION_DISCOUNT
MIN_COMMISSION = 20
TRANSACTION_TAX = 0.003
```

```python
def calc_trade_cost(amount: float, is_sell: bool) -> float:
    commission = max(abs(amount) * EFFECTIVE_COMMISSION, MIN_COMMISSION)
    tax = abs(amount) * TRANSACTION_TAX if is_sell else 0.0
    return commission + tax
```

這幾乎逐字命中我們 D3 的規格 —— **包含 `MIN_COMMISSION = 20` 這個我特別強調、附件程式沒處理的細節**。費率是模組層常數，可配置。

但是：

```
$ grep -rniE "commission|slippage|tax|cost" src/ --include="*.py"
（無任何命中）
```

**`src/` 整個目錄零成本代碼。** `src/services/walk_forward_backtester.py`（36 KB，就是 `POST /api/v1/backtest/walk-forward` 背後那支）的 `_calculate_week_return()` 只做：

```python
# 計算 close[T+3]/close[T+1]-1（對齊 2-day label 定義）
```

純價格報酬，不扣手續費、不扣證交稅、不扣滑價。**Dashboard 上看到的回測數字是毛報酬。**

**兩套都缺滑價。** `calc_trade_cost` 只有手續費 + 證交稅。對我們 40 萬資金必須做零股的情境（D4），缺滑價是硬傷。

### 3. Walk-Forward 實作是否可信（有沒有偷用未來資料）？

**骨架可信。** `src/shared/constants.py`：

```python
TRAIN_DAYS = 504   # 訓練期：2 年
VALID_DAYS = 100   # 驗證期：約 4 個月
EMBARGO_DAYS = 7   # Embargo：7 天（防止 label lookahead）
RETRAIN_THRESHOLD_DAYS = 7  # 每週重訓

LABEL_EXPR = "Ref($close, -3) / Ref($close, -1) - 1"
LABEL_ENTRY_OFFSET = 1   # 買入日偏移：T+1
LABEL_EXIT_OFFSET = 3    # 賣出日偏移：T+3
```

- **T+1 進場**，不是 T 日收盤進場 → 這正是我在 D7 為籌碼族定死的規格，它已經做對了
- 有 7 天 embargo 隔開訓練/驗證
- README 明言「以股票代碼確定性排序」（避免隨機性洩漏）
- `simulate()` 的市場 SMA 有註明 `no lookahead: uses close up to each date`

**但我沒有獨立驗證過。**要真正證實需重跑 162 個模型的完整訓練（數小時 + 三年歷史資料），本次 2 天評估未執行。列為採用後的第一項待辦：跑 look-ahead 掃描器（把 T 日之後資料設 NaN，確認特徵值不變）。

### 4. 支不支援週頻決策？

**不完全支援，需改。**

- 訓練是**週頻**（每週重訓）✅ 符合我們需求
- 但預測與回測是**日頻調倉**：`_calculate_week_return()` 對每個預測日都重選 Top-K
- Label 是 **2 日報酬**（T+1 買、T+3 賣），我們要 **5 日**

改法：`LABEL_EXPR` 改 `Ref($close, -6) / Ref($close, -1) - 1`、`LABEL_EXIT_OFFSET = 6`、`LABEL_DELAY_DAYS` 連帶調整。**改完必須全部重訓**（162 個模型）。

另外 `scripts/` 的 `HoldDropStrategy(k, h, d)` 有持有期參數 `h`，可用來壓低換手率 —— README 最佳策略 `HoldDrop(K=10,H=3,D=1)` 週換手率僅 9.9%。這條路比改 label 更省事，值得先試。

### 5. 標的池能不能限制成 0050+0051 = 150 檔？退市股怎麼處理？

**能，而且是資料工作不是程式手術。**

`src/repositories/models.py:200`：

```python
class StockUniverse(Base):
    __tablename__ = "stock_universe"
    stock_id: Mapped[str]
    name: Mapped[str]
    market_cap: Mapped[int]   # 市值（億）
    rank: Mapped[int]          # 市值排名
    updated_at: Mapped[datetime]
```

標的池是一張 DB 表，`POST /universe/sync` 從 TWSE 現價 × 發行股數算市值取前 100。改成 150 檔只要改篩選條件。

**但 survivorship 沒處理。** `sync_universe` 用**今天**的市值排名建池，`updated_at` 只有一個時間戳，**沒有歷史成分股快照**。用今日名單回溯訓練 2015-2025 就是 survivorship bias。

它的排除規則倒是寫得不錯（`universe.py:151-158`）：排除非 4 位數代號、開頭 0（ETF）、`-KY`、`*`（全額交割）、`-創`。

→ **這是必修項**：加一張 `stock_universe_history(as_of_date, stock_id, rank)`，每季快照。

### 6. 宣稱的 Sharpe 1.724 是哪段期間、in-sample 還是 OOS？

**答得出來，而且答得很清楚 —— 這題它拿高分。**

README 明載：

> 所有結果皆為**樣本外**，來自 156 週（3 年）Walk-Forward 回測，每週重新訓練模型。

最佳策略 `HoldDrop(K=10, H=3, D=1)`：年化 55.1%、年化超額 +23.9%、Sharpe 1.724、MaxDD 38.7%、週換手 9.9%、t-stat 1.89。

**但它自己也把難看的數字攤出來了**，這點值得加分：

| 年度 | 超額 | Sharpe | 勝率 | MaxDD |
|---|---|---|---|---|
| 2023 | +80.0% | 2.96 | 54.5% | 18.0% |
| 2024 | **−8.4%** | 0.45 | 47.9% | 21.2% |
| 2025 | +19.6% | 1.62 | 51.3% | 29.4% |

**我對這組數字的保留意見（三點，都很重要）：**

1. **t-stat 1.89 未達 2**，統計上不顯著。三年 156 週樣本不足以區分技巧與運氣。
2. **績效高度集中在 2023**（+80%）。2024 輸大盤 8.4%。扣掉 2023，這個策略平庸。
3. **沒有 PBO / Deflated Sharpe。**`scripts/` 跑了 **9 種基礎策略 × 7 組 hedge config ≈ 63 種組合**，然後報告「最佳」那個。63 組合裡挑最高 Sharpe，1.724 必然被選擇偏誤污染。README 自己引了 Harvey/Liu/Zhu 的多重測試論文，卻沒做校正。
4. **成本不含滑價**，且 `FIXED_AMOUNT_PER_STOCK = 50_000` 假設每筆固定 5 萬（10 檔 = 50 萬），沒有整股/零股約束。

**結論：1.724 我不採信為可執行績效，但採信它是「同一設定下 DoubleEnsemble 比單一 LightGBM 好 71%」的相對證據** —— 這也正是 README 自己主張的用法（它明說「跨條件的 IC 直接比較沒有意義，有意義的是相對改善幅度」）。這個自我克制比數字本身更值得信任。

### 7. 授權條款、相依套件版本是否還能安裝？

- **MIT** ✅
- 能安裝，但要踩第二節那 6 個坑。`requirements.txt` 只鎖下限（`>=`），沒有 lock file → 環境不可重現。**必修**：產生 `requirements.lock` 或改用 `uv`/`poetry`。

---

## 四、我已做的修補

### 修補內容

TWSE 於 2026 年把 `STOCK_DAY_ALL` 從 JSON 改為 CSV，且多了開頭「日期」欄位。原程式 `resp.json()` 直接炸，導致 `POST /universe/sync`（管線第一步）回 500，**整個系統無法取得任何資料**。

`src/adapters/twse.py` — 新增 CSV 解析，並讓 `_fetch_rwd` 自動偵測回應格式：

```python
def _csv_to_rwd(text: str) -> dict | None:
    """
    把 RWD 端點回傳的 CSV 轉成舊版 JSON 結構

    2026 年起部分 RWD 端點（如 STOCK_DAY_ALL）改為回傳 CSV，
    且多出一個開頭的「日期」欄位。此函式還原為原本的
    {"stat": "OK", "date": "YYYYMMDD", "fields": [...], "data": [[...]]}，
    讓下游解析邏輯不需改動。
    """
```

`src/interfaces/routers/universe.py` — 原本 router 內嵌自己的 `httpx` 呼叫（DRY 違反），改為呼叫 adapter 的 `_fetch_rwd`，一併吃到 CSV 支援。

### 修補後驗證

```
POST /api/v1/universe/sync   → {"success":true,"total":100}
POST /api/v1/sync/bulk       → {"total":1379,"inserted":100}
```

> 這兩處修補是 git working tree 上的未提交變更，可用 `git diff` 檢視、`git checkout` 還原。

### 順手發現的程式品質問題（未修）

- `universe.py:130` 用裸 `except:`（吞掉所有例外，含 `KeyboardInterrupt`）
- router 層直接做資料抓取與市值計算，應下放到 service/adapter
- `MI_MARGN` 端點回傳 `tables` 陣列而非 `data`，需確認 adapter 是否處理

---

## 五、與我們七項決策的落差表

| 決策 | qlib-tw-trader 現況 | 落差 | 工時 |
|---|---|---|---|
| D2 池 = 0050+0051 = 150 | 市值前 100，無歷史成分股 | 改篩選 + 加歷史快照表 + 重訓 | 2 天 + 重訓 |
| D3 成本含證交稅 + MIN_FEE | `scripts/` 有 ✅ / `src/` 完全沒有 ❌ | 把成本抽成 `config/costs.py`，接進 `walk_forward_backtester` | 2 天 |
| D3 滑價分層 0.3%/0.4% | **兩套都沒有滑價** | 新增 | 0.5 天 |
| D4 零股、部位百分比、33% 上限 | `FIXED_AMOUNT_PER_STOCK=50_000` 固定值 | 改為按資金比例 + 零股取整 + 雙上限 | 1 天 |
| D5 Codex reviewer | 無 | 加 `AGENTS.md` | 0.5 天 |
| D6 FinMind 分批 + 429 退避 | 有限流間隔，未見退避重試 | 補退避 | 0.5 天 |
| D7 5 日 triple-barrier | 2 日報酬點估計 | 改 `LABEL_EXPR` + 實作 triple-barrier | 2 天 + 重訓 |
| D7 三族比較 | 有 9 種策略但都是同一模型的排名變體，不是三個策略族 | 自建動能/籌碼/均值回歸三族 | 4 天 |
| D7 PBO / DSR | 無 | 自建 | 1.5 天 |
| 交易計畫輸出（entry/target/stop/R:R） | 只輸出排名選股 | 自建 | 1.5 天 |
| Telegram 推播 | 無 | 自建 | 1 天 |
| 對照組（買進持有 / 0050 / 隨機） | 有市場報酬對照，無 0050、無隨機 | 補 | 0.5 天 |

**改造合計約 17 個工作日**，與 D1 原估自建 21 天相比省 4 天 —— 但省下來的不只工時：

- **303 個因子定義**（Alpha158 量價 109 + 台股籌碼 107 + 交互 50 + 增強 37）自己寫至少 2 週，而且容易寫錯
- **台股多源資料同步**（TWSE 6 個端點 + FinMind + yfinance，含降級與覆蓋率追蹤，`sync_service.py` 60 KB）自己寫至少 2 週
- **DoubleEnsemble 實作**（ICDM 2020）自己寫 3～5 天
- **正確的 embargo walk-forward 骨架**

實質節省估計 **4～6 週**。

---

## 六、採用後的第一批待辦（依序）

1. **跑 look-ahead 掃描器**驗證第 3 題的「可信」不是文件自稱 —— 這是採用它的前提，不通過就放棄
2. 把成本抽成 `config/costs.py`，補滑價，接進 `walk_forward_backtester`，**重跑一次回測看 Sharpe 掉多少**
3. 加 `stock_universe_history` 季度快照，解決 survivorship
4. 產生 lock file 固定環境
5. 把 `STOCK_DAY_ALL` 的 CSV 修補送 PR 回上游（社群回饋，也避免下次 clone 又踩）
6. 標的池擴到 150 檔，重訓
7. 改 5 日 label，重訓
8. 自建 PBO/DSR、交易計畫輸出、Telegram

**紅線提前到第 2 步**：成本接上去之後如果 Sharpe 從 1.724 掉到跑不贏 0050，就不要再往下投資工時，回頭重新選策略族。

---

## 七、風險提醒

- 本報告評估的是**軟體工程可用性**，不是投資價值。
- README 的績效數字我**未獨立重現**（需數小時完整訓練），僅做文件一致性與程式邏輯檢查。
- Sharpe 1.724 存在多重測試偏誤（63 組合選最佳）、樣本不足（t-stat 1.89）、績效集中單一年度（2023 佔 +80%）、成本不含滑價四項已知問題。**不應據此預期實際績效。**
- MaxDD 38.7% 對 40 萬資金意味最大回撤約 15.5 萬。
- 本文件不構成投資建議。
