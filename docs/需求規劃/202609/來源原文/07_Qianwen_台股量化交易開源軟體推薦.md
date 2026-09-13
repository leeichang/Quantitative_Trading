# 千問 Qwen — 台股量化交易開源軟體推薦

- 來源：https://qianwen.my.cn/share/chat/39c4fead682049b1a44e5a3bae0fbe63
- 日期：2026-09-11
- 引用來源數：20 篇
- 關聯附件：`1789108251314.docx`（即本對話後續產出的「核心回測與推送代碼框架」，內容收錄於本檔末段）

---

## USER

> 有什麼股票量化交易的開源軟體可以用在台灣股市？給我一個整合你建議的開源軟體和 Claude code 或 codex 的完整量化交易軟體架構！只要先到回測比較出結果！給三個最佳投資建議就好！傳送訊息到 telegram 給使用者就可以！不用直接下單？可以產出未來 5 個交易日
>
> 台積電
> 買進區間：2,330～2,370
> 目標價：2,550
> 失效價：2,280

---

## ASSISTANT

針對您的需求，我為您規劃了一套結合開源軟體與 AI 代理（Claude Code / Codex）的台灣股市量化回測架構，並整理了最佳投資建議與台積電的短線交易區間。

### 一、台灣股市量化開源軟體推薦

- **Backtrader**：經典的本地開源全能回測框架，支援精細的滑點與佣金模擬，非常適合用來打磨與精細複盤台股策略。
- **VnPy (VeighNa)**：國內知名的全棧開源量化平台，模組化設計且支援多市場交易接口，適合用於策略開發與後續的實盤對接。

> 註：本來源是七份中唯一推薦 **VnPy** 的。

### 二、整合 AI 的量化回測軟體架構

結合 Claude Code 或 Codex，採用「**多智能體協同模式**」搭建從研究到回測的自動化工作流：

1. **數據獲取與預處理**：利用 AI 自動對接台股數據源（如 **AKShare** 或 **Tushare**），拉取歷史 K 線與財務數據，並進行復權與清洗
2. **策略開發與迭代**：透過結構化 Prompt 讓 Codex 或 Claude Code 編寫雙均線、RSI 等策略代碼。AI 可自動進行參數掃描與交易成本（手續費、滑點）測算，並透過 **Traceback 日誌進行自我糾錯**
3. **回測與績效分析**：將策略接入 Backtrader 或 VnPy 進行歷史回測。AI 負責產出標準化績效分析面板（如夏普比率、最大回撤、盈虧分布），並**評估結果是真實優勢還是統計巧合**
4. **Telegram 訊息推送模組**：回測完成或每日盤後，用 Python 呼叫 Telegram Bot API（`sendMessage`），將回測比較結果或未來交易區間自動推送到指定的 Telegram 頻道或使用者

### 三、三個最佳投資建議

> 註：本來源將「三個最佳投資建議」詮釋為**三個資產配置方向**，而非三檔個股。

1. **採用「啞鈴策略」應對震盪**：一手配置高景氣度的 AI 算力與科技龍頭，另一手配置低估值、高股息的紅利板塊（如銀行、公用事業）。平衡市場波動，兼顧進攻與防守。
2. **降低交易頻率，回歸基本面**：當前市場量化交易佔比極高，短線頻繁交易容易陷入追漲殺跌的循環。建議減少主觀情緒干擾，將精力放在中長線價值投資與基本面研究上。
3. **做好資產配置與動態平衡**：將資金分為短期備用金、中期穩健底倉與長期增值資金。設定固定的資產比例（如股債比例），並定期再平衡，透過紀律性操作實現「賣漲買跌」。

### 四、台積電未來 5 個交易日交易區間

- 買進區間：2,330 ～ 2,370
- 目標價：2,550
- 失效價：2,280

> 免責聲明：以上資訊僅供參考與技術探討，不構成任何實質投資建議。市場有風險，投資需謹慎。

> 需要我直接幫你生成可運行的回測代碼框架嗎？可以包含數據拉取、策略邏輯、績效分析和 Telegram 推送模組。

---

## 後續產出（附件 `1789108251314.docx`）

### 一、核心回測與推送代碼框架

```python
import backtrader as bt
import requests
import logging
from datetime import datetime

# 1. Telegram 消息推送模块
class TelegramNotifier:
    """封装 Telegram Bot API 的推送类"""
    def __init__(self, token, chat_id):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}/sendMessage"
        self.logger = logging.getLogger(__name__)

    def send_message(self, text):
        """发送文本消息到 Telegram"""
        payload = {"chat_id": self.chat_id, "text": text, "parse_mode": "HTML"}
        try:
            response = requests.post(self.base_url, json=payload, timeout=10)
            response.raise_for_status()
            self.logger.info("Telegram 消息发送成功")
        except Exception as e:
            self.logger.error(f"Telegram 消息发送失败: {e}")

# 2. 策略逻辑模块 (以双均线策略为例)
class SmaCross(bt.Strategy):
    params = (('fast', 5), ('slow', 20),)

    def __init__(self):
        sma_fast = bt.ind.SMA(period=self.p.fast)
        sma_slow = bt.ind.SMA(period=self.p.slow)
        self.crossover = bt.ind.CrossOver(sma_fast, sma_slow)

    def next(self):
        if not self.position:
            if self.crossover > 0:
                self.buy()
        elif self.crossover < 0:
            self.close()

# 3. 回测执行与 Telegram 结果推送
def run_backtest_and_notify(ticker, start_date, end_date, tg_notifier):
    cerebro = bt.Cerebro()

    # 这里可以替换为 AKShare 或 Tushare 获取的台股历史数据
    # data = bt.feeds.GenericCSVData(dataname='TSMC.csv', ...)
    # cerebro.adddata(data)

    cerebro.addstrategy(SmaCross)
    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.001)  # 模拟交易成本

    # 运行回测
    print(f'正在回測 {ticker}...')
    results = cerebro.run()
    final_value = cerebro.broker.getvalue()

    # 生成回测摘要
    summary = (
        f" <b>台股量化回測報告</b>\n"
        f"標的：{ticker}\n"
        f"區間：{start_date} 至 {end_date}\n"
        f"期末資金：${final_value:,.2f}\n"
        f"策略：雙均線 (5日/20日)"
    )

    # 推送到 Telegram
    tg_notifier.send_message(summary)
    return results

# 4. 主程序入口
if __name__ == '__main__':
    # 替换为你的 Telegram Bot Token 和 Chat ID
    TOKEN = 'YOUR_TELEGRAM_BOT_TOKEN'
    CHAT_ID = 'YOUR_CHAT_ID'

    notifier = TelegramNotifier(token=TOKEN, chat_id=CHAT_ID)

    # 执行台積電回測並推送結果
    run_backtest_and_notify(
        ticker="2330.TW",
        start_date="2025-01-01",
        end_date="2026-09-11",
        tg_notifier=notifier
    )
```

### 二、框架使用與擴展建議

1. **環境準備**：`pip install backtrader requests`
2. **數據接入**：框架中預留了數據接入位置。針對台股，建議使用 **AKShare** 或 **Tushare** 拉取台積電（2330.TW）的歷史日線數據，並轉換為 Backtrader 支援的 `GenericCSVData` 格式。
3. **安全提醒**：務必將 `YOUR_TELEGRAM_BOT_TOKEN` 和 `YOUR_CHAT_ID` 替換為真實憑證。實際部署時，建議將這些敏感資訊存放在**環境變數或 `.env` 檔案**中，避免硬編碼導致洩露。
4. **進階分析**：若需更專業的績效指標（如夏普比率、最大回撤），可在 `run_backtest_and_notify` 函數中整合 **QuantStats** 庫，生成 HTML 分析報告後，將報告連結或核心指標一併推送到 Telegram。
