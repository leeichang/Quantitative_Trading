# ChatGPT — 台股量化軟體比較

- 來源：https://chatgpt.com/share/6aa34959-81f4-83ee-8ec5-d6c8aa40f921
- 擷取日期：2026-09-11
- 對話輪次：5（分享起點為助理回覆，第一則使用者提問未含在分享範圍內）

---

## ASSISTANT（第 1 則）

可以，而且你這個需求我會重新調整架構：不要把「長期量化分析」和「當日即時交易訊號」混在一起。

你真正要做的比較像：**AI Quant Research + Real-time Signal Engine**

每天先用大量歷史資料找出「今天值得注意的股票」，盤中再用即時行情重新評分，最後只把 Top 3 投資建議送到 Telegram。完全不下單。

這個方向非常適合用 Claude Code / Codex 當「AI 開發團隊」，而不是讓 LLM 在盤中直接決定買賣。

### 一、建議的最終架構

```
                         ┌────────────────────────────┐
                         │      Claude Code / Codex   │
                         │  Architect                 │
                         │  Coder                     │
                         │  Test / QA                 │
                         │  Quant Research Assistant  │
                         └─────────────┬──────────────┘
                                       │
                         開發 / 產生策略 / 測試
                                       ▼
┌──────────────────────────────────────────────────────────────┐
│                    AI QUANT PLATFORM                         │
│  ┌──────────────────────┐       ┌─────────────────────────┐  │
│  │ Historical Research  │       │ Real-time Signal Engine │  │
│  │ 10年歷史資料         │       │ 今日 Tick / 1m / 5m     │  │
│  │ 基本面 / 籌碼        │       │ 即時成交價 / Volume     │  │
│  │ 技術指標 / 因子      │       │ Bid / Ask               │  │
│  │ ML / Backtest        │       │ Intraday Momentum       │  │
│  └──────────┬───────────┘       └──────────┬──────────────┘  │
│             ▼                              ▼                 │
│       Long-term Score                Intraday Score          │
│             └──────────────┬───────────────┘                 │
│                            ▼                                 │
│                    Signal Fusion Engine                      │
│                            ▼                                 │
│                       Ranking Engine                         │
│                            ▼                                 │
│                     TOP 3 Candidates                         │
│                            ▼                                 │
│                    Telegram Notifier                         │
└──────────────────────────────────────────────────────────────┘
```

最重要的設計是：**Historical Engine ≠ Intraday Engine**

### 二、應該使用哪些開源軟體？

- 核心：Python、PostgreSQL、Pandas / Polars、NumPy、Backtrader
- 台股資料：TWSE OpenAPI、TPEX、Shioaji
- 即時行情：使用 **Shioaji**，不要自己直接碰 TWSE 即時行情。Shioaji 提供台股即時行情 streaming，可取得 Tick / Quote。
- 回測：**Backtrader**。核心為 `Data Feed → Strategy → Analyzer → Backtest`，支援多時間框架，可做 `Daily trend + 5-minute entry`。

> TWSE 有官方 OpenAPI，也有歷史交易資料；即時交易資訊則有另外的資料取得與授權機制。

### 三、甚至不建議第一版使用 FinLab 當核心（修正前一個回答）

如果目標只是「找股票 → 回測 → 看結果」，FinLab 非常方便。
但若要「自己建立一個長期可以演化的 AI Quant System」，改成：

```
              PostgreSQL
        ┌──────────┴─────────┐
        ▼                    ▼
Historical Engine      Realtime Engine
        ▼                    ▼
   Factor Engine        Signal Engine
        ▼                    ▼
   Backtrader             Scoring
        └─────────┬──────────┘
                  ▼
            Ranking Engine
                  ▼
               TOP 3
```

FinLab 可以當 **Research Accelerator**，而不是系統核心。

### 四、最重要：分成兩個時間尺度

**A. 長期分析 Engine**（每天收盤後 18:00 跑）

```
18:00 → 更新歷史資料 → 重新計算所有股票 → Factor Calculation
     → ML Model → Backtest → Portfolio Simulation → Long-term Score
```

分析項目：

- Fundamental：EPS、ROE、ROA、營益率、毛利率、營收成長、自由現金流、負債比、殖利率、P/E、P/B
- Momentum：20 日、60 日、120 日、250 日
- Technical：MA、MACD、RSI、ATR、ADX、Bollinger、Volume
- 籌碼：外資、投信、自營商、融資、融券、大戶

輸出 LongTermScore，例如：台積電 91.2、聯發科 87.6、廣達 85.3

### 五、第二個才是「即時 Engine」

```
09:00 → Shioaji streaming → Tick → 1-minute aggregation → 5-minute aggregation
      → Realtime Feature → Intraday Score
```

特徵：即時漲幅、成交量異常、Volume Ratio、VWAP、突破、回撤、Momentum、Bid/Ask imbalance、5 分鐘 RSI、5 分鐘 MACD、市場強弱、族群強弱

### 六、兩個 Score 最後才合併

`Final Score = LongTerm × 60% + Intraday × 40%`

但建議**不要固定權重**，依 Market Regime 調整：

```
Market Regime
┌──────┴───────┐
多頭          空頭
LT 70%        LT 40%
IT 30%        IT 60%
```

這會比單純 60% + 40% 聰明很多。

### 七、最後只輸出 TOP 3

不要把 50 檔股票全部丟給使用者。產品應只告訴使用者：

```
╔══════════════════════════╗
║     今日 Top 3           ║
╠══════════════════════════╣
║ 1. 台積電 2330  Score 92 ║
║    LT 95 / Intraday 88   ║
║ 2. 廣達 2382    Score 89 ║
║ 3. 聯發科 2454  Score 87 ║
╚══════════════════════════╝
```

然後才附原因：✓ 長期趨勢向上 ✓ EPS 成長 ✓ 法人持續買超 ✓ 股價站上 60MA ✓ 今日成交量 +180% ✓ 5 分鐘突破 ✓ 產業族群同步走強

### 八、Telegram 訊息設計

```
🚨 台股即時量化訊號
2026/09/11 10:35

🥇 2330 台積電   Score: 94.2
長期: ★★★★★ 95   即時: ★★★★★ 93
訊號：20日突破 / 成交量 2.3x / VWAP 上方 / 外資趨勢正向 / 半導體族群強勢
風險：中
---
🥈 2454 聯發科   Score: 91.7   長期 93 / 即時 90
訊號：動能增強 / Volume +145% / 5m breakout
---
🥉 2382 廣達     Score: 89.8   長期 88 / 即時 92
訊號：AI server sector 強 / Intraday momentum / VWAP breakout

⚠️ 僅供研究，不執行交易
```

### 九、「即時」不能每一 Tick 都跑 AI

不要 `Tick → Claude → 分析 → Telegram`，這非常昂貴且毫無必要。應該：

```
Tick → Realtime Data Store → Feature Calculator → Quant Signal Engine
     → Score changed? ──NO──→ ignore
                      └─YES─→ threshold? → Telegram
```

例如：只有 Score 改變超過 5 分才通知。或者：`Score > 90 AND Intraday Score > 85 AND Volume Ratio > 1.5 AND 沒有最近 30 分鐘重複訊號` 才發 Telegram。

### 十、AI 不應該在核心交易決策迴圈裡

不要 `行情 → Claude → 買/不買`。應該：

```
行情 → Quant Engine → 數學模型 → Score → Rank → TOP 3 → AI →「解釋為什麼」→ Telegram
```

也就是：**Quant 負責決策，AI 負責解釋。**這樣比較可測試。

### 十一、Claude Code / Codex 放在「開發層」

```
                 Quant Project
          ┌───────────┼────────────┐
          ▼           ▼            ▼
      Claude Code   Codex        Human
          ▼           ▼             │
       Research     Coding          │
          └─────┬─────┘             │
                ▼                   │
             GitLab/GitHub          │
                ▼                   │
              CI/CD ◄───────────────┘
                ▼
              Tests → Backtest
```

### 十二、甚至可以讓 Claude Code 自動「發現策略」

給 Claude Code 的研究條件範例：

```
市場：台股   期間：2015-2025
請尋找：1. Momentum factor 2. Value factor 3. Quality factor
        4. Volume factor 5. Institutional buying
要求：- 不允許 look-ahead bias - 加入交易成本 - 加入滑價
      - Walk-forward validation - Out-of-sample test
產生：strategy_001 ... strategy_050
```

然後自動：`50 strategies → Backtest → (Training / OOS) → Ranking → Top 10`

再讓 Claude Code 分析：為什麼 Strategy #17 在 2018-2025 有效？是否 overfitting？是否 survivorship bias？是否 look-ahead bias？是否只在某一段市場有效？

**這才是 AI 真正適合量化的地方。**

### 十三、資料庫設計

```
quant_db
├── security
├── market_daily
├── market_minute
├── market_tick
├── fundamental
├── institutional
├── margin
├── corporate_action
├── factor_value
├── technical_indicator
├── strategy
├── backtest
├── backtest_trade
├── signal
├── recommendation
└── notification
```

其中最重要的是 `recommendation`：

```
recommendation
--------------------
id, trade_date, timestamp, stock_id
long_term_score, intraday_score, final_score
rank, signal, confidence
entry_zone, stop_loss, target_price
reason_json
model_version, strategy_version
created_at
```

這樣未來可反查：「為什麼 2026/09/11 10:35 系統推薦這支股票？」對 AI Quant 系統非常重要。

### 十四、回測不能只看報酬率

必須建立：CAGR、Annual Return、Sharpe、Sortino、Max Drawdown、Calmar、Win Rate、Profit Factor、Turnover、Average Holding Period、Exposure

而且至少切分：

```
Train 2015-2021 / Validation 2022-2023 / Out-of-Sample 2024-2025 / Paper 2026
```

不要 2015~2026 全部拿來最佳化，否則非常容易得到漂亮但沒有用的結果。

### 十五、再加一層「Market Regime」（第二版）

使用 TAIEX、市場 breadth、漲跌家數、成交量、VIX、波動率、產業輪動判斷 Bull / Neutral / Bear：

- Bull → Momentum strategy
- Neutral → Quality + Value
- Bear → Defensive / Cash

這樣系統不是每天都用同一套策略。

### 十六、整個系統的時間流程

```
前一天 18:00 → Historical Pipeline (Fundamental / Factor / ML)
            → Backtest → Long-term Rank → Top 50 → 夜間完成
08:30 Pre-market → Top 20
09:00 開盤 → Shioaji → Real-time Tick → Intraday Engine
          → Signal Fusion → Top 3 → Telegram
```

### 十七、第一版 MVP 壓得很小

不要一開始做 100 個因子。

- Historical：5~10 年 Daily OHLCV → 20/60/120/250 MA → Momentum → Volume → EPS → ROE → 殖利率 → Backtrader
- Intraday：Shioaji → Tick → 1m → 5m → VWAP → Volume Ratio → Momentum
- Fusion：LongTerm 60% + Intraday 40%
- Output：TOP 3
- Notification：Telegram

### 十八、最終技術 Stack

| 層 | 技術 |
|---|---|
| OS | Ubuntu |
| Language | Python |
| API / Control | FastAPI |
| Historical DB | PostgreSQL |
| Analytics | Polars + NumPy |
| Backtest | Backtrader |
| Taiwan Data | TWSE / TPEX |
| Real-time | Shioaji |
| Cache | Redis |
| Scheduler | APScheduler / Celery |
| ML | LightGBM / XGBoost |
| AI | Claude / Codex |
| Notification | Telegram Bot |
| Version | Git |
| Deployment | Docker |
| Monitoring | Grafana + Prometheus |
| Logs | Loki |
| Frontend | 第一版甚至不要 |

### 十九、Repository 結構

```
taiwan-quant/
├── apps/
│   ├── api/
│   ├── historical_worker/
│   ├── realtime_worker/
│   ├── backtest_worker/
│   └── notification_worker/
├── quant/
│   ├── factors/ indicators/ strategies/ portfolio/ risk/ regime/ ranking/
├── data/
│   ├── twse/ tpex/ shioaji/
├── backtest/
│   ├── datasets/ strategies/ reports/
├── ai/
│   ├── research/ strategy_generator/ backtest_reviewer/ explanation/
├── telegram/
├── tests/
├── migrations/
├── docker/
├── CLAUDE.md
├── AGENTS.md
├── pyproject.toml
└── docker-compose.yml
```

`CLAUDE.md` 規定 Claude，所有策略必須：

1. 禁止 Look-ahead bias
2. 禁止 Survivorship bias
3. 必須加入交易成本
4. 必須加入滑價
5. 必須 Walk-forward
6. 必須 Out-of-sample
7. 必須保存 strategy version
8. 必須保存 backtest parameters
9. 不得直接修改 production
10. 所有策略必須有 unit test

Codex 作為第二個 reviewer：`Claude 開發 Strategy #17 → Codex Code Review → Quant Review → 找 bias → Backtest`

### 結論

這個構想真正的核心不是「量化交易」，而是：**建立一個每天自動研究台股、盤中即時重新評分、最後只告訴我 Top 3 的 AI 投資研究員。**

第一版完全不需要前端、不需要自動下單、不需要複雜 UI。

另外，即時資料要特別注意：TWSE 的直接即時資料有正式的資訊授權/費用機制；若只是個人透過券商 API 做研究，Shioaji 是比較實際的入口。TWSE 官方也明確區分即時、延遲及歷史資料的取得方式。

> 如果是我替你做，下一步會直接進入「MVP 技術規格」：把 PostgreSQL schema、Docker Compose、Python package、Backtrader 回測介面、Shioaji 即時資料介面、Top-3 scoring algorithm、Telegram message schema，以及 CLAUDE.md + AGENTS.md 一次定義出來。這樣就可以直接丟給 Claude Code / Codex 開工。

---

## USER（第 2 則）

> 依據歷史資料去預測，提供買進賣出的高低點這種比較好？還是還需要考慮即時狀況？正確的量化交易是那種？

---

## ASSISTANT（第 3 則）

如果目標是「真正提高量化策略的可交易性」，只靠歷史資料預測買賣高低點，**不建議**。

比較正確的量化交易思路是：**歷史資料負責建立「股票在什麼條件下具有正期望值」；即時資料負責判斷「現在是否進入這個條件」。**

而不是：`歷史資料 → 預測明天最高點/最低點 → 精準買最低、賣最高`（後者很容易變成過度擬合）。

### 1. 「預測高低點」其實不是量化交易的核心

你即使知道「明天最高可能 1,120」，也不知道先到 1,060 還是先到 1,120。所以真正可交易的問題應該改成：

**現在買進，未來 N 天的「風險調整後報酬」是否值得承擔？**

### 2. 正統量化比較接近這個概念

```
今天 → 取得所有歷史與目前資訊 → 模型計算
→ P(未來20日報酬 > 5%) / P(未來20日報酬 < -5%) / Expected Return / Expected Risk
```

| 股票 | 預期20日報酬 | 下跌風險 | 勝率 | Score |
|---|---|---|---|---|
| A | +9.2% | -4.1% | 68% | 92 |
| B | +7.8% | -8.3% | 61% | 81 |
| C | +5.1% | -2.2% | 73% | 79 |

買 A 不是因為模型認為 A 明天一定會漲，而是歷史上「目前這種狀態」出現時，A 的未來報酬分布具有優勢。

### 3. 「長期 + 即時」想法是對的，但改成三層

```
                    Quant System
          ┌──────────────┼──────────────┐
      Long-term       Medium-term    Real-time
       Regime           Signal         Trigger
       6~10年          20~120日         今日
          └──────────────┼──────────────┘
                    Decision Engine
                         ▼
                       TOP 3
```

### 4. 第一層：歷史資料決定「值不值得買」

分析基本面（營收成長、EPS 成長、ROE、毛利率、營益率、FCF、負債、估值）、技術面（20D/60D/120D/250D momentum、Volatility、ATR）、籌碼（外資、投信、自營商、融資、融券）。

不是直接預測「明天漲多少」，而是建立 **Market State → Future Return Distribution**：

```
目前狀態：EPS growth ↑ / ROE ↑ / 20D momentum ↑ / 60D momentum ↑ / Foreign buying ↑ / Volume ↑
↓ Historical Similarity ↓
未來20日：平均 +7.8% / Median +5.9% / P(positive) = 71% / Max DD = -4.8%
```

### 5. 第二層：中期訊號

長期模型說台積電很好，但今天突然爆大量、跌破 20MA、法人大賣、半導體族群轉弱——這時不應該因為「長期 Score = 95」就硬買。

Medium-term Engine 分析 5D / 20D / 60D 的 Trend、Momentum、Volatility、Volume、Sector Rotation、Institutional Flow。

### 6. 第三層：即時狀況

角色不是「重新預測公司的價值」，而是回答：**現在是不是一個好的進場時機？**

```
Long-term 台積電 = 94「值得關注」
09:00 Volume +20% → 09:30 Volume +80% → 10:00 突破 VWAP
→ 10:05 5m high breakout → 10:10 半導體族群同步上漲 → 10:15 外資期貨方向配合
Intraday Score = 91
Final = Long-term 94 / Medium-term 87 / Intraday 91 → 91.3
```

### 7. 這是一個「分層決策系統」

```
全台股 → Long-term (Fundamental/Quality/Value/Momentum) → Top 100
      → Medium-term (Trend/Momentum/Volume/Institutional) → Top 20
      → Intraday (1m/5m/VWAP/Volume/Breakout/Market Regime) → Top 3 → Telegram
```

### 8. 「買點、賣點」還要不要做？要。

但不要做成「預測精準最低點 / 最高點」，而應該做成 **Entry Zone / Exit Zone**：

```
目前價格：1,080
合理進場區：1,065～1,085
第一目標：1,130
第二目標：1,160
風險失效：1,035
```

這比 `Buy = 1,067 / Sell = 1,153` 更合理。

### 9. 更進階：預測「報酬分布」

```
P(+10%) = 32% / P(+5%) = 58% / P(0%) = 73% / P(-5%) = 21% / P(-10%) = 8%
Expected Return = +7.2% / Expected Volatility = 18%
Expected Max Drawdown = -6.3% / Risk\Reward = 2.4

Score = Expected Return × Probability ÷ Risk
```

### 10. 不要只找「最會漲的股票」

Top 3 若全部是 AI / 半導體 = **Factor concentration**，風險很高。加入 Portfolio-level constraint：

- 最多 2 支同產業
- 最多 1 支高波動
- 至少 1 支 defensive / quality
- Expected Return + Risk + Correlation 一起計算

### 11. Top 3 產出漏斗

```
Candidate 100 → Long-term Model → Top 30 → Medium-term Model → Top 10
→ Intraday Model → Top 5 → Portfolio Optimizer → Top 3
```

Telegram：

```
🥇 Stock A  Expected Return: +8.4%  Probability: 72%  Risk: Medium
   Entry Zone: 1,050~1,070  Target: 1,130  Invalidation: 1,025
🥈 Stock B  Expected Return: +6.8%  Probability: 69%  Risk: Low
🥉 Stock C  Expected Return: +6.1%  Probability: 74%  Risk: Low
```

### 12. AI 到底放哪裡？

**不要讓 Claude / Codex 在盤中直接決定買賣。**

```
Market Data → Quant Models → Statistical Score → Risk Engine → Top 3
→ Claude → 自然語言解釋 → Telegram
```

Claude 可以說：「台積電進入 Top 3 的主要原因，是長期品質因子與 60 日動能均位於歷史高分位；今日盤中成交量同步放大，且價格維持 VWAP 上方，因此即時訊號沒有否定長期訊號。」

但不要讓 Claude 說：「我覺得台積電現在應該買。」前者可以測試，後者不能可靠回測。

### 13-14. 最推薦的「正確量化」方向

不是「預測股價」，而是：**預測條件式未來報酬 + 風險 + 進場時機**，即 `P(Return > X | Current State)`。

```
Long-term + Medium-term + Intraday + Risk + Correlation → Portfolio Ranking → TOP 3
```

各層角色：

- 歷史模型：「這支股票值得買嗎？」
- 中期模型：「最近趨勢還成立嗎？」
- 即時模型：「今天現在這個時間點，進場條件成立嗎？」
- Risk Engine：「即使看對，這筆交易值得承擔這個風險嗎？」
- Portfolio Engine：「在所有候選股票中，哪 3 支現在的風險/報酬比最好？」

**延伸：不要只做一個模型。**最好讓 5～20 個不同邏輯的策略各自產生 signal，再做 ensemble / model selection。這樣 Backtrader 回測平台才真正有價值。

---

## USER（第 4 則）

> 我要的是下周的投資建議！台機電 買入2350，賣出2550

---

## ASSISTANT（第 5 則）

你要的其實不是「盤中自動交易系統」，而是：**每週固定產生下一週的投資計畫：哪一檔、什麼價格區間買、什麼價格賣、風險失效點。**

這會讓架構做一個重要調整：**即時資料仍然要，但它是用來修正「下週計畫」，不是讓系統每天重新選股。**

### 改後架構

```
台股全部股票
    ▼
Historical Engine（5~10年歷史資料）
    ▼
找出歷史優勢型態
    ▼
Forecast Engine（預測未來 5~10 日）
    ▼
Expected Return / Risk
    ▼
候選股票 Top 20
    ▼
Market Condition（台股/產業/法人）
    ▼
Top 10 Candidates
    ▼
Intraday Engine（本週五盤中狀況）
    ▼
Entry / Target Model
    ▼
Portfolio Rank
    ▼
TOP 3 → Telegram
```

### 要預測的不是「最高價」

```
台積電目前 2,450
系統不要直接預測「下週最高 2,550」，而是計算：
未來 5 個交易日：
  P(達到 2,550) = 63%
  P(跌到 2,350) = 21%
  Expected Return = +4.8%
  Expected Drawdown = -3.7%
然後才產生：
  買進區間：2,330～2,370
  目標價：2,550
  失效價：2,280
```

### 「買 2350、賣 2550」是一個完整交易計畫

```
2330 台積電   目前價格：2450
下週策略：
  買入：2350 ± 20
  第一目標：2500
  第二目標：2550
  停損 / 模型失效：2280
  預期持有：3～10 個交易日
  風險：中
  量化信心：78%
```

### 即時資料的真正用途

需要，但不是用來重新預測。若週五下午發生「外資突然大賣 + 半導體指數大跌 + 台積電跌破關鍵均線 + 成交量異常」，系統應該：

```
原始模型 → Buy 2350 / Target 2550 → Realtime Risk Filter → ❌ 暫停這個建議
```

而不是「不管怎樣，下週 2350 買。」

### 「週模型 + 即時修正」

- **週模型**（每週五）：`Historical Data → Factor → ML → Backtest → Forecast → Risk → 下週候選`
- **即時模型**（週五收盤前 + 下週每天）：`即時行情 → 市場狀況 → 產業狀況 → 法人 → 波動 → Trend → 是否仍符合原本模型`，只做 **Confirm / Reduce Confidence / Cancel**，而不是每天重新亂選股票。

### 每週日晚上輸出

```
╔══════════════════════════════╗
║       下週 TOP 3             ║
╠══════════════════════════════╣
║ 🥇 台積電 2330               ║
║ 買入：2,330～2,370           ║
║ 目標：2,550                  ║
║ 失效：2,280                  ║
║ 預期報酬：+7.5%              ║
║ 信心：82%                    ║
║ 🥈 聯發科 2454 ...           ║
║ 🥉 廣達 2382 ...             ║
╚══════════════════════════════╝
```

每天盤中只做：🟢 建議仍有效 / 🟡 信心下降 / 🔴 原策略失效

### 這比「每天推薦三支」更好

因為真正要測試的是：**如果系統每週給我 3 個交易計畫，我長期執行的績效如何？**這就可以非常乾淨地回測。

統計：總交易數、勝率、平均獲利、平均虧損、Profit Factor、CAGR、Maximum Drawdown、Sharpe、平均持有天數。

### 把「買入價」本身模型化

不要人工設定 `Buy = Current Price × 0.96`，而是讓模型研究「歷史上這種型態，什麼價格位置進場最有效？」

```
Current = 2450
歷史相似型態最佳 Entry = -3.5% ～ -5.2%
2450 × 0.965 = 2364 ; 2450 × 0.948 = 2323
→ Entry Zone = 2320～2365
然後模型再計算 Target / Stop / Expected Return / Probability / Risk-Reward
```

### 核心問題定義（給 Claude Code / Codex 開發的模型）

```
給定：T 日以前所有可以知道的資料
預測：T+1 ~ T+10
輸出：
  P(reach_target), P(hit_stop)
  Expected_Return, Expected_Drawdown
  Entry_Low, Entry_High
  Target_1, Target_2
  Stop
  Holding_Period, Confidence
嚴格禁止使用 T 日以後的資料。
```

這樣回測才不會作弊。

### 台積電當下實際資料（引用）

> 以目前能查到的 9/10 資料，台積電收在 2,450 元，當日下跌 15 元；9/10 三大法人合計賣超約 2,489 張，其中外資賣超約 1,944 張。（來源：Goodinfo! 台灣股市資訊網）

所以如果模型在今天真的產生「下週台積電：Buy 2,350 / Target 2,550」，它就必須把目前的法人動向、近期價格行為、波動率與市場狀態一起納入，而不能只因為長期趨勢看多就給出這個價格。

### 產品定位一句話

> 「每週告訴我下週哪 3 支股票值得買、多少錢買、多少錢賣；盤中只負責告訴我這個計畫還有效嗎。」

MVP 非常乾淨：
① 每週產生 Top 3 → ② 每支股票產生 Buy / Target / Stop → ③ 回測整個歷史週期 → ④ 每日即時資料驗證 → ⑤ Telegram 通知。
