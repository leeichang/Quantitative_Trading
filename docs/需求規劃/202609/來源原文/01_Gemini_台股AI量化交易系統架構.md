# Gemini (Pro) — 台股 AI 量化交易系統架構

- 來源：https://share.gemini.google/JGj2qwB2z7Hb → 實際頁面 https://gemini.google.com/share/ed6792d17207
- 建立時間：2026-09-11 07:23（發布 08:19）
- 擷取日期：2026-09-11

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

針對台灣股市的量化交易，結合 AI 輔助開發（如 Claude Code 或 Codex）與強大的開源工具，您可以打造一套全自動的「數據擷取 ➔ AI 策略編寫與回測 ➔ 訊號生成 ➔ Telegram 推播」系統，完全不需要直接串接券商 API 承擔自動下單的實單風險。

### 台股量化開源軟體堆疊

**市場數據源：FinMind**
專為台股設計的開源 API，能輕鬆取得歷史股價、三大法人買賣超、融資融券等籌碼面及基本面數據，是台股量化的首選。

**回測引擎：Backtrader**
Python 生態系中最成熟、資源最豐富的回測框架。支援多商品回測與圖表輸出，架構清晰，非常適合讓 LLM 產出交易邏輯並直接載入執行。

**AI 大腦：Claude Code（或 OpenAI Codex）**
作為程式碼生成的代理引擎。可將台股的市場特徵與近期總經摘要交給它，讓 AI 自動編寫 Backtrader 交易策略，並幫助解析回測報表（如夏普值、最大回撤）。

**推播通知：python-telegram-bot**
輕量級的 Telegram API 串接套件，負責將回測比較出的「最佳投資訊號」推播至手機。

### 量化交易系統運作流程

1. **資料擷取與前處理（FinMind）**
   系統每日盤後定時觸發，透過 FinMind API 抓取台股（例如台灣 50 成分股）的價量資料與法人籌碼數據，並清洗轉換為 Backtrader 支援的 pandas DataFrame 格式。

2. **AI 策略生成與動態調整（Claude Code）**
   系統將市場波動度等特徵以 Prompt 形式傳遞給 Claude Code，要求其產出對應行情的 Python 交易邏輯腳本（如動能突破、均線回歸），並自動設定技術指標的動態參數。

3. **歷史數據回測與報表分析（Backtrader）**
   將生成的策略腳本批量投入 Backtrader 引擎進行歷史回測。系統會自動淘汰最大回撤（Max Drawdown）過高或夏普值（Sharpe Ratio）過低的策略，進而篩選出勝率最高的前三大標的。

4. **訊號生成與推播（python-telegram-bot）**
   系統根據勝出策略在最後一個交易日的狀態，推算未來的具體「買進區間」、「目標價」與「失效價（停損價）」，並透過 Telegram Bot 將格式化的分析報告發送。

### Telegram 系統推播成果範例

> ⚠️ 假設性情境說明：以下推播內容為展示此量化軟體架構的模擬範例（已整合指定的台積電數據條件）。這代表系統回測完成後，手機上會收到的實際訊息格式，並非真實的即時金融市場預測。

```
🤖 【AI 量化系統推播】未來 5 個交易日最佳投資建議
系統已完成 Backtrader 歷史回測，以下為綜合評分排名前三之交易訊號：

1️⃣ 台積電 (2330)
   買進區間：2,330～2,370
   目標價：2,550
   失效價 (停損)：2,280
   AI 策略簡評：籌碼面外資延續買超，股價回測月線具強勁支撐，回測模型預估勝率達 76%。

2️⃣ 聯發科 (2454)
   買進區間：4,600～4,680
   目標價：4,950
   失效價 (停損)：4,500
   AI 策略簡評：動能指標 MACD 即將黃金交叉，AI 邊緣運算晶片營收成長預期帶動資金進駐，適合波段操作。

3️⃣ 廣達 (2382)
   買進區間：320～335
   目標價：370
   失效價 (停損)：310
   AI 策略簡評：伺服器出貨量穩健，回測短均線乖離率已收斂，下檔風險低，具備良好風報比。
```

後續方向選項：索取 FinMind 串接 Backtrader 程式碼 / 實作 Telegram 推播機器人 / 讓 Claude 寫量化策略

---

## USER（第 3 則）

> 請提供將 FinMind 台股數據轉換並匯入 Backtrader 進行回測的 Python 程式碼範例。

---

## ASSISTANT（第 4 則）

### 1. 安裝必要套件

```bash
pip install FinMind backtrader pandas
```

### 2. Python 完整實作程式碼

```python
import pandas as pd
import backtrader as bt
from FinMind.data import DataLoader
from datetime import datetime

# ==========================================
# 步驟 1：使用 FinMind 抓取台股歷史數據
# ==========================================
def get_finmind_data(stock_id: str, start_date: str, end_date: str) -> pd.DataFrame:
    print(f"正在抓取 {stock_id} 從 {start_date} 到 {end_date} 的數據...")
    api = DataLoader()
    df = api.taiwan_stock_daily(
        stock_id=stock_id,
        start_date=start_date,
        end_date=end_date
    )

    # 檢查是否有抓到資料
    if df.empty:
        raise ValueError(f"無法獲取 {stock_id} 的數據，請檢查連線或日期設定。")

    # ==========================================
    # 步驟 2：資料前處理 (對齊 Backtrader 格式)
    # ==========================================
    # FinMind 預設欄位：date, stock_id, Trading_Volume, Trading_money, open, max, min, close, spread, Trading_turnover
    # Backtrader 需要：datetime, open, high, low, close, volume
    df = df.rename(columns={
        'date': 'datetime',
        'max': 'high',
        'min': 'low',
        'Trading_Volume': 'volume'
    })

    # 將 datetime 轉換為 pandas 的 datetime 格式，並設為 Index
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)

    # 僅保留 Backtrader 需使用的欄位
    df = df[['open', 'high', 'low', 'close', 'volume']]

    return df

# ==========================================
# 步驟 3：撰寫 Backtrader 交易策略
# ==========================================
class SmaCrossStrategy(bt.Strategy):
    """
    簡單的雙均線交叉策略 (Golden Cross / Death Cross)
    搭配停損停利概念
    """
    params = (
        ('fast_period', 10), # 短均線參數
        ('slow_period', 20), # 長均線參數
    )

    def __init__(self):
        # 宣告技術指標
        sma_fast = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.fast_period)
        sma_slow = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.slow_period)

        # 宣告交叉訊號：1 代表短線向上穿越長線 (黃金交叉)，-1 代表向下穿越 (死亡交叉)
        self.crossover = bt.indicators.CrossOver(sma_fast, sma_slow)

    def log(self, txt, dt=None):
        """用於記錄交易日誌"""
        dt = dt or self.datas[0].datetime.date(0)
        print(f'[{dt.isoformat()}] {txt}')

    def next(self):
        # 如果帳戶內沒有建立部位
        if not self.position:
            # 發生黃金交叉 -> 買進
            if self.crossover > 0:
                self.log(f'📈 買進訊號，收盤價: {self.data.close[0]}')
                # 在台股，通常以 1000 股 (一張) 為單位
                self.buy(size=1000)

        # 如果帳戶內已有部位
        else:
            # 發生死亡交叉 -> 賣出
            if self.crossover < 0:
                self.log(f'📉 賣出訊號，收盤價: {self.data.close[0]}')
                self.close()

# ==========================================
# 步驟 4：主程式 - 啟動回測引擎 Cerebro
# ==========================================
if __name__ == '__main__':
    # 1. 取得清洗好的資料
    # 回測區間：2023年到2024年底
    df_stock = get_finmind_data('2330', '2023-01-01', '2024-12-31')

    # 2. 轉換為 Backtrader 的 DataFeed
    data_feed = bt.feeds.PandasData(dataname=df_stock)

    # 3. 初始化大腦 (Cerebro)
    cerebro = bt.Cerebro()

    # 4. 餵入資料與策略
    cerebro.adddata(data_feed)
    cerebro.addstrategy(SmaCrossStrategy)

    # 5. 設定初始資金與交易條件
    cerebro.broker.setcash(1000000.0) # 初始資金 100 萬台幣
    cerebro.broker.setcommission(commission=0.001425) # 設定台股券商手續費率 (約 0.1425%)

    # 6. 執行回測
    print(f'💰 初始總資產: {cerebro.broker.getvalue():.2f}')
    cerebro.run()
    print(f'💰 最終總資產: {cerebro.broker.getvalue():.2f}')

    # 7. 繪製回測圖表 (可選)
    # cerebro.plot(style='candlestick')
```

### 關鍵轉換細節說明

- **欄位重命名（Renaming）**：FinMind 抓出來的最高價叫 `max`，最低價叫 `min`。Backtrader 需要標準 OHLCV 格式，必須用 `.rename()` 替換為 `high` 與 `low`。
- **日期索引（Datetime Index）**：Backtrader 的 `PandasData` 強制要求傳入的 DataFrame 將日期設為 Index，且格式為 datetime 物件，`df.set_index('datetime')` 是最關鍵的一步。
- **交易單位（Volume / Size）**：FinMind 的 `Trading_Volume` 是「股數」而非「張數」。因此 `self.buy(size=1000)` 指定買入 1000 股（即 1 張台積電），與美股直接買 1 股邏輯不同，`setcash()` 也需設大一點（例如 100 萬）以避免保證金不足導致下單失敗。

> 接下來可將 `SmaCrossStrategy` 區塊的邏輯交給 Claude Code，提示它：「請幫我把上方的策略，改為計算 RSI 低於 30 買進，並在進場時自動掛上 5% 停損與 10% 停利的 Backtrader 策略」。

---

## USER（第 5 則）

> 如何下 Prompt 讓 Claude Code 幫我寫出帶有停損停利功能的 Backtrader 策略腳本？

---

## ASSISTANT（第 6 則）

在 Backtrader 中，實作停損與停利最好的方式是使用 **Bracket Order（括號訂單）** 或 **OCO（One Cancels the Other，二擇一刪單）** 機制。如果只是模糊地告訴 AI「幫我加上停損」，它可能會寫出在 `next()` 迴圈裡每天檢查收盤價的土法煉鋼寫法，這會導致回測產生嚴重的「滑價誤差」或「未來函數」問題。

### 給 Claude 的黃金 Prompt 範本

```
【角色設定】
你現在是一位精通 Python 與 Backtrader 框架的資深量化交易工程師。

【任務目標】
請幫我撰寫一個 Backtrader 的 Strategy Class。這是一個針對台灣股市的回測策略，
請嚴格遵循 Backtrader 的架構（包含 __init__ 與 next 方法）。

【策略邏輯】
- 技術指標：請使用 RSI (相對強弱指標)，週期設定為 14 天。
- 進場條件：當日收盤時，若 RSI 跌破 30，產生買進訊號。
- 交易單位：台股以張為單位，每次買進訊號發生時，固定買入 1000 股。
- 出場條件（停損停利核心）：
  請使用 Backtrader 的 buy_bracket 方法（或 OCO 訂單機制）來同時掛出
  「主進場單」、「停利單」與「停損單」。
  停利點（Take Profit）：進場價的 +10%。
  停損點（Stop Loss）：進場價的 -5%。
- 部位控管：若帳戶內已經有持倉部位（self.position 不為空），則不重複進場。

【輸出要求】
- 請提供完整的 class RsiBracketStrategy(bt.Strategy): 程式碼。
- 必須包含 log 函數，在控制台印出「進場價」、「停損觸發」、「停利觸發」的日期與價格資訊。
- 程式碼請加上詳細的中文註解。
```

### 驗收 Claude 程式碼的關鍵點

標準且安全的 Backtrader 停損停利寫法應該長這樣：

```python
    def next(self):
        # 檢查是否有未完成的訂單，避免重複下單
        if self.order:
            return

        # 如果沒有持倉，且 RSI 小於 30
        if not self.position and self.rsi[0] < 30:

            # 取得當前收盤價作為參考基準
            entry_price = self.data.close[0]

            # 計算停利與停損價格
            take_profit_price = entry_price * 1.10  # +10% 停利
            stop_loss_price = entry_price * 0.95    # -5% 停損

            self.log(f'發送買進 Bracket 訂單，市價約: {entry_price:.2f}，停利: {take_profit_price:.2f}，停損: {stop_loss_price:.2f}')

            # 【關鍵】使用 buy_bracket 同時送出三張單
            self.buy_bracket(
                size=1000,
                exectype=bt.Order.Market,        # 主單：市價買進
                stopprice=stop_loss_price,       # 停損單：觸價停損
                stopexec=bt.Order.Stop,
                limitprice=take_profit_price,    # 停利單：限價停利
                limitexec=bt.Order.Limit
            )
```

**為什麼必須用 `buy_bracket`？**
如果把停利停損寫在 `next()` 裡面（例如 `if current_price < stop_loss_price: self.sell()`），系統必須等到「隔天」收盤或開盤才能動作。如果隔天遇到台股跳空跌停，回測數據就會失真。使用 `buy_bracket` 是將委託單直接掛在交易所（回測引擎）中，一旦盤中價格觸及就立刻執行，歷史回測結果才會貼近真實交易。

> 接下來，要實作 Telegram 推播嗎？
