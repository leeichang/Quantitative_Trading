[English](README.md) | [繁體中文](README.zh-TW.md)

<h1 align="center">qlib-tw-trader</h1>

<p align="center">
  <strong>台股端到端量化交易系統</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black" alt="React">
  <img src="https://img.shields.io/badge/Qlib-Microsoft-0078D4" alt="Qlib">
  <img src="https://img.shields.io/badge/LightGBM-DoubleEnsemble-9ACD32" alt="LightGBM">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
</p>

<p align="center">
  <img src="docs/images/screenshots/dashboard.png" alt="Dashboard" width="800">
</p>

## 什麼是 qlib-tw-trader？

qlib-tw-trader 是一個為台股市場打造的全端量化交易研究平台。它自動化了量化研究員手動進行的完整工作流程：

1. **資料擷取** — 從台灣證券交易所（TWSE）、FinMind、Yahoo Finance 收集資料
2. **計算約 300 個 alpha 因子** — 涵蓋量價型態、三大法人籌碼流向、跨資產交互作用
3. **訓練 ML 模型** — 使用 DoubleEnsemble（ICDM 2020）：自動選擇有用特徵並重新加權困難樣本的迭代集成
4. **策略回測** — 嚴謹的 156 週 Walk-Forward 框架，每週模型僅使用過去資料訓練
5. **產生每日信號** — 對台股市值前 100 大股票進行截面排名，預測 2 天報酬
6. **全面監控** — 透過 React 儀表板搭配 WebSocket 即時更新

系統設計了**嚴格的 lookahead bias 防護**：T 日交易僅使用 T-1 收盤後可得的特徵，訓練與驗證集之間設有 7 天 embargo。這不是玩具回測 — 而是一個能產出可信賴結果的研究工具。

## Walk-Forward 回測結果

所有結果皆為**樣本外**，來自 156 週（3 年）Walk-Forward 回測，每週重新訓練模型。

### LightGBM vs DoubleEnsemble

| 指標 | LightGBM | DoubleEnsemble | 變化 |
|------|:--------:|:--------------:|:----:|
| Backtest IC | 0.0107 | **0.0166** | +55% |
| IC 衰減（驗證期 → 回測期） | 78.5% | **56.0%** | -23pp |
| 最佳策略 Sharpe | 1.006 | **1.724** | +71% |
| 最佳超額報酬 | +0.2% | **+23.9%** | |
| 分位數單調性（rho） | 0.90 | **1.00** | 完美 |
| Spread t-stat | 1.25 | **2.30** | 顯著 |

### 最佳策略：HoldDrop(K=10, H=3, D=1)

| 指標 | 數值 | 指標 | 數值 |
|------|:----:|------|:----:|
| 年化報酬 | 55.1% | 年化超額 | +23.9% |
| Sharpe | 1.724 | 最大回撤 | 38.7% |
| 週換手率 | 9.9% | t-stat | 1.89 |

<details>
<summary><strong>年度績效分解</strong></summary>

| 年度 | 超額 | Sharpe | 勝率 | MaxDD |
|:----:|:----:|:------:|:----:|:-----:|
| 2023 | +80.0% | 2.96 | 54.5% | 18.0% |
| 2024 | -8.4% | 0.45 | 47.9% | 21.2% |
| 2025 | +19.6% | 1.62 | 51.3% | 29.4% |

</details>

<details>
<summary><strong>市場 Regime 分析</strong></summary>

| Regime | Mean IC | 超額（bps/日） | 勝率 |
|:------:|:-------:|:--------------:|:----:|
| 熊市 | **0.0354** | +10.0 | **55.2%** |
| 盤整 | 0.0146 | **+13.4** | 50.0% |
| 牛市 | -0.0002 | +1.8 | 50.2% |

模型在股票分化大的熊市中排名能力最強。

</details>

<details>
<summary><strong>與 Qlib 官方 Benchmark 對比</strong></summary>

我們的 IC/ICIR 低於 [Qlib CSI300 benchmark](https://github.com/microsoft/qlib/tree/main/examples/benchmarks)（IC 0.052, ICIR 0.42），但 **Sharpe 更高**（1.72 vs 1.34）。這反映了結構性差異：

| 因素 | Qlib Benchmark | 本專案 |
|------|:--------------:|:------:|
| 股票池 | CSI300（300 支） | TW100（100 支） |
| 市場 | 中國 A 股（散戶主導） | 台股（法人主導） |
| 持股數 | top-50 | top-10（集中） |
| 標籤 | 1 天報酬 | 2 天報酬 |

跨條件的 IC 直接比較沒有意義。有意義的指標是同一設定下 LightGBM → DoubleEnsemble 的**相對改善幅度**。

</details>

## 截圖展示

<details>
<summary><strong>模型訓練 — 162 個已訓練模型的週曆視圖</strong></summary>
<img src="docs/images/screenshots/training.png" alt="Training" width="800">
</details>

<details>
<summary><strong>模型評估 — Rolling IC、累積報酬、策略績效</strong></summary>
<img src="docs/images/screenshots/evaluation.png" alt="Evaluation" width="800">
</details>

<details>
<summary><strong>因子管理 — 266 個因子的公式與選取率</strong></summary>
<img src="docs/images/screenshots/factors.png" alt="Factors" width="800">
</details>

<details>
<summary><strong>Walk-Forward 回測 — 週選擇、IC 分析、權益曲線</strong></summary>
<img src="docs/images/screenshots/backtest.png" alt="Backtest" width="800">
</details>

<details>
<summary><strong>訓練品質 — Jaccard 相似度、IC 穩定性、ICIR 追蹤</strong></summary>
<img src="docs/images/screenshots/quality.png" alt="Quality" width="800">
</details>

<details>
<summary><strong>資料集 — 多來源資料覆蓋率與同步狀態</strong></summary>
<img src="docs/images/screenshots/datasets.png" alt="Datasets" width="800">
</details>

## 運作原理

<p align="center">
  <img src="docs/images/pipeline.svg" alt="交易流程" width="600">
</p>

## 功能特色

- **DoubleEnsemble（ICDM 2020）** — 內建樣本加權和特徵選擇的迭代集成。IC 比單一 LightGBM 高 55%。[[論文]](https://arxiv.org/abs/2010.01265)
- **約 300 個因子庫** — Alpha158 量價因子（109）、台股三大法人籌碼因子（107）、交互因子（50）、增強因子（37），涵蓋波動率 regime、動量、流動性、微結構
- **Walk-Forward 回測** — 156 週樣本外測試，包含 IC 衰減分析、分位數展開、9 種策略變體比較
- **嚴格防止 Lookahead Bias** — T 日交易僅用 T-1 特徵。訓練/驗證集間 7 天 embargo。以股票代碼確定性排序
- **多來源資料同步** — TWSE OpenAPI、FinMind、yfinance 自動同步，優先順序降級與覆蓋率追蹤
- **全端儀表板** — 9 個頁面：Dashboard、Factors、Training、Evaluation、Backtest、Quality、Predictions、Positions、Datasets
- **Optuna 超參數搜尋** — 貝葉斯最佳化 DoubleEnsemble 參數

## 技術棧

| 層級 | 技術 |
|------|------|
| **後端** | FastAPI, SQLAlchemy 2.0, SQLite (WAL mode) |
| **前端** | React 18, Vite, TailwindCSS, Zustand, Recharts |
| **模型** | Qlib (Microsoft), LightGBM, DoubleEnsemble, Optuna |
| **資料** | TWSE OpenAPI, FinMind, yfinance |
| **即時通訊** | WebSocket |

## 快速開始

### Docker（推薦）

```bash
git clone https://github.com/Docat0209/qlib-tw-trader.git
cd qlib-tw-trader

cp .env.example .env
# 編輯 .env 填入 FinMind API token（免費：https://finmindtrade.com/）

docker compose up --build
```

- 前端：http://localhost:3000
- 後端 API：http://localhost:8000
- Swagger 文件：http://localhost:8000/docs

### 手動安裝

```bash
# 後端
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows
pip install -r requirements.txt
cp .env.example .env
uvicorn src.interfaces.app:app --reload --port 8000

# 前端（另開終端）
cd frontend && npm install && npm run dev

# 初始化因子庫（約 300 個因子）
curl -X POST http://localhost:8000/api/v1/factors/seed
```

## 系統架構

<p align="center">
  <img src="docs/images/architecture.svg" alt="系統架構" width="800">
</p>

<details>
<summary><strong>專案結構</strong></summary>

```
qlib-tw-trader/
├── src/
│   ├── adapters/           # TWSE, FinMind, yfinance 資料客戶端
│   ├── interfaces/         # FastAPI 路由、Schema、WebSocket
│   ├── repositories/       # 資料庫存取 + 因子定義（約 300 個）
│   ├── services/           # 訓練、預測、回測、Qlib 導出
│   └── shared/             # 常數、型別、週曆工具
├── frontend/               # React 18 SPA（9 個頁面）
├── tests/                  # pytest 測試套件（28 個測試）
├── scripts/                # 分析腳本（模型評估、時段分析、IC）
└── data/                   # 資料庫 + 模型 + Qlib 導出（gitignored）
```

</details>

## API

啟動後可在 http://localhost:8000/docs 查看互動式文件。

<details>
<summary><strong>主要端點</strong></summary>

| 端點 | 說明 |
|------|------|
| `POST /api/v1/sync/all` | 同步所有資料來源 |
| `POST /api/v1/factors/seed` | 初始化約 300 個因子 |
| `POST /api/v1/models/train` | 訓練指定週的模型 |
| `POST /api/v1/backtest/walk-forward` | 執行 Walk-Forward 回測 |
| `GET /api/v1/backtest/walk-forward/summary` | 聚合回測指標 |
| `POST /api/v1/predictions/today/generate` | 產生今日預測 |
| `GET /api/v1/predictions/today` | 取得當前選股 |
| `GET /api/v1/predictions/history` | 預測歷史記錄 |

</details>

## 資料來源

| 優先序 | 來源 | 涵蓋範圍 |
|:------:|------|----------|
| 1 | TWSE OpenAPI | OHLCV、PER/PBR（每日 17:30 後可用） |
| 2 | FinMind | 三大法人、融資融券、月營收（免費 600 次/時） |
| 3 | yfinance | 還原收盤價（無限制） |

## 參考文獻

**核心模型：**
- Zhang et al. **"DoubleEnsemble: A New Ensemble Method Based on Sample Reweighting and Feature Selection for Financial Data Analysis."** ICDM 2020. [[論文]](https://arxiv.org/abs/2010.01265)
- Ke et al. **"LightGBM: A Highly Efficient Gradient Boosting Decision Tree."** NeurIPS 2017. [[論文]](https://papers.nips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html)
- Akiba et al. **"Optuna: A Next-generation Hyperparameter Optimization Framework."** KDD 2019. [[論文]](https://arxiv.org/abs/1907.10902)

**量化金融：**
- Gu, Kelly & Xiu. **"Empirical Asset Pricing via Machine Learning."** Review of Financial Studies, 2020. [[論文]](https://doi.org/10.1093/rfs/hhaa009)
- Grinold & Kahn. **"Active Portfolio Management."** McGraw-Hill, 1999.
- Harvey, Liu & Zhu. **"...and the Cross-Section of Expected Returns."** Review of Financial Studies, 2016. [[論文]](https://doi.org/10.1093/rfs/hhv059)
- Novy-Marx & Velikov. **"A Taxonomy of Anomalies and Their Trading Costs."** Review of Financial Studies, 2016. [[論文]](https://doi.org/10.1093/rfs/hhv063)
- Faber. **"A Quantitative Approach to Tactical Asset Allocation."** Journal of Wealth Management, 2007. [[論文]](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=962461)

**平台：**
- Yang et al. **"Qlib: An AI-oriented Quantitative Investment Platform."** 2020. [[repo]](https://github.com/microsoft/qlib)
- Lopez de Prado. **"Advances in Financial Machine Learning."** Wiley, 2018. [[書籍]](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086)

## 貢獻

歡迎貢獻。詳見 [CONTRIBUTING.md](CONTRIBUTING.md) 了解環境設定與開發流程。

## 授權

MIT。詳見 [LICENSE](LICENSE)。
