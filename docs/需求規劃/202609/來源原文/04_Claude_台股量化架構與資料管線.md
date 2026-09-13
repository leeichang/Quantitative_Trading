# Claude — 台股量化架構與資料管線

- 來源：https://claude.ai/share/a3d4991e-eb4e-4e21-ab4f-bea18d1d733b
- 分享者：cymmetrik
- 擷取日期：2026-09-11
- 關聯附件：`fetch data.py`、`add indicators.py`（即本對話產出的兩支程式，已收錄於同資料夾）

---

## USER（第 1 則）

> 有什麼股票量化交易的開源軟體可以用在台灣股市？給我一個整合你建議的開源軟體和 Claude code 或 codex 的完整量化交易軟體架構！只要先到回測比較出結果！給三個最佳投資建議就好！傳送訊息到 telegram 給使用者就可以！不用直接下單？可以產出未來 5 個交易日
>
> 台積電
> 買進區間：2,330～2,370
> 目標價：2,550
> 失效價：2,280

---

## ASSISTANT（第 2 則）

這是一個五層架構：**資料 → 特徵 → 回測 → Claude Code 協調 → Telegram 推播**。以下是各層的開源選擇與實作重點。

### 1. 資料層

- **FinMind（推薦首選）**：免費方案 600 req/hr，資料源自證交所，涵蓋日 K、籌碼面、三大法人、財報，且已支援 `llms.txt`，Claude Code 能直接讀文件自己寫串接程式
- **yfinance**：備援資料源，適合校驗
- 需注意：除權息要還原；財報／月營收要用「**公布日**」而非「所屬期間」，否則回測會有 **look-ahead bias**

### 2. 特徵工程

TA-Lib 或 pandas-ta：MA、KD、MACD、RSI、布林通道等指標

### 3. 回測引擎

- **backtrader**：成熟、社群大，適合事件驅動策略
- **vectorbt**：向量化運算，跑大量參數組合速度快很多
- **FinLab（台股原生開源套件）**：內建台股回測框架，處理除權息與公布日邏輯，對台股比通用框架省事很多

### 4. Claude Code 協調層

這層不是套件，是流程：在專案裡放一個 `CLAUDE.md` 說明資料來源、回測規則、輸出格式，讓 Claude Code 讀取 FinMind/backtrader 文件後自動寫 `fetch_data.py` → `run_backtest.py` → `generate_report.py` 的 pipeline，並依照回測結果排名，只挑出通過的標的組成報告。Codex 用法相同，差別只在 CLI 介面。

### 5. Telegram 推播

`python-telegram-bot`，用 Bot API 的 `sendMessage` 搭配 Markdown 格式。範例模板：

```
📊 台股回測選股報告（未來5交易日）

🔹 台積電 (2330)
買進區間：2,330～2,370
目標價：2,550
失效價：2,280
勝率／年化報酬：{來自回測結果}

🔹 標的2 ...
🔹 標的3 ...

⚠️ 僅回測參考，非即時下單
```

### ⚠️ 重要提醒（本對話中最關鍵的一段意見）

> 你貼的台積電數字是你給的範例格式，**不是我實際跑出來的回測結果**——我沒有連接你的資料源和回測引擎，無法憑空產生真實的買進區間/目標價/勝率。這套架構建好後，「前三名標的」要由 Claude Code 實際跑完 backtrader/vectorbt 的績效排名（夏普值、勝率、最大回撤）後才能產生，不是我現在用文字生成。另外我不是投資顧問，這類輸出僅供策略研究參考，實際投資風險需自行評估。

---

## USER（第 3 則）

> 幫我寫第一版 fetch_data.py

---

## ASSISTANT（第 4 則）

（產出檔案 `fetch_data.py`，即附件 `fetch data.py`）

用法：

```bash
pip install pandas requests --break-system-packages
python3 fetch_data.py 2330 --start 2020-01-01
python3 fetch_data.py 2330 2317 2454 --start 2020-01-01   # 一次抓多檔
python3 fetch_data.py update                              # 之後每天增量更新
```

重點設計：

- **增量更新**：讀本地 `data/{股票代號}.csv` 的最後日期，只抓缺少的區間，省 API 額度
- FinMind 免費方案有 request 上限，多檔之間有 `sleep`
- 想用付費 token 就設 `FINMIND_TOKEN` 環境變數

> 下一步通常是 `add_indicators.py`（接 TA-Lib/pandas-ta 算技術指標）或直接寫 `run_backtest.py`（接 backtrader/vectorbt）。

---

## USER（第 5 則）

> add_indicators.py

---

## ASSISTANT（第 6 則）

（產出檔案 `add_indicators.py`，即附件 `add indicators.py`）

用法：

```bash
pip install pandas-ta --break-system-packages
python3 add_indicators.py 2330
python3 add_indicators.py all      # 處理 data/ 底下所有已快取股票
```

輸出到 `data/processed/{股票代號}.csv`，欄位含 MA5/10/20/60、RSI14、KD(9,3,3)、MACD、布林通道、均量，回測引擎可直接讀這份檔案。

> 選 **pandas-ta 而不是 TA-Lib**，是因為它是純 Python、pip 直接裝，不用另外編譯 C library，在之後要交給 Claude Code 自動化的環境下比較不會卡安裝。

> 接下來要寫 `run_backtest.py`（接 backtrader 或 vectorbt 跑策略回測）嗎？
