#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
台股量化交易管線（回測 + 訊號推播版）
=================================================
用途：抓台股資料 → 多策略回測比較 → 產出投資建議 → 傳送到 Telegram
資料源：FinMind（開源，台股技術面/籌碼面/基本面）
回測引擎：vectorbt（向量化、極速、適合大量參數掃描）
Telegram：python-telegram-bot（只推播，不下單）

安裝：
    pip install FinMind vectorbt python-telegram-bot pandas numpy

環境變數：
    export FINMIND_TOKEN=你的token        # https://finmindtrade.com 免費註冊，可省則省
    export TELEGRAM_BOT_TOKEN=你的bot token   # 找 @BotFather 建立
    export TELEGRAM_CHAT_ID=你的chat id       # 找 @userinfobot 取得

執行：
    python quant_tw_pipeline.py --stock 2330
"""

import os
import argparse
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# 1. 資料層：FinMind 抓台股日線（含還原股價，避免除權息失真）
# ----------------------------------------------------------------------
def load_data(stock_id: str, start: str = "2020-01-01") -> pd.DataFrame:
    from FinMind.data import DataLoader
    dl = DataLoader()
    token = os.getenv("FINMIND_TOKEN")
    if token:
        dl.login_by_token(api_token=token)
    df = dl.taiwan_stock_daily(stock_id=stock_id, start_date=start)
    df = df.rename(columns={"date": "Date", "open": "Open", "max": "High",
                            "min": "Low", "close": "Close", "Trading_Volume": "Volume"})
    df["Date"] = pd.to_datetime(df["Date"])
    return df.set_index("Date")[["Open", "High", "Low", "Close", "Volume"]].sort_index()

# ----------------------------------------------------------------------
# 2. 回測層：vectorbt 同場比較 3 種策略（含台股真實成本模型）
#    手續費 0.1425%*0.6（網路下單折扣）+ 賣出證交稅 0.3% + 滑價 0.1%
# ----------------------------------------------------------------------
def run_backtest(df: pd.DataFrame):
    import vectorbt as vbt

    close = df["Close"]
    open_ = df["Open"]
    fees = 0.1425 * 0.6 / 100
    slippage = 0.001

    results = {}

    # 策略A：雙均線交叉（5/20 日）— 趨勢跟隨
    fast, slow = vbt.MA.run(close, 5), vbt.MA.run(close, 20)
    entries = fast.ma_crossed_above(slow)
    exits = fast.ma_crossed_below(slow)
    pf = vbt.Portfolio.from_signals(close, entries, exits,
                                    fees=fees, slippage=slippage,
                                    freq="1D", init_cash=1_000_000,
                                    direction="longonly")
    results["雙均線交叉(5/20)"] = pf

    # 策略B：RSI 均值回歸（RSI<30 買、RSI>70 賣）
    rsi = vbt.RSI.run(close, window=14)
    entries = rsi.rsi_crossed_below(30)
    exits = rsi.rsi_crossed_above(70)
    pf = vbt.Portfolio.from_signals(close, entries, exits,
                                    fees=fees, slippage=slippage,
                                    freq="1D", init_cash=1_000_000,
                                    direction="longonly")
    results["RSI均值回歸(14)"] = pf

    # 策略C：布林通道突破（收盤突破上軌買、跌破中軌賣）
    bb = vbt.BBANDS.run(close, window=20, alpha=2)
    entries = close.vbt.crossed_above(bb.upper)
    exits = close.vbt.crossed_below(bb.middle)
    pf = vbt.Portfolio.from_signals(close, entries, exits,
                                    fees=fees, slippage=slippage,
                                    freq="1D", init_cash=1_000_000,
                                    direction="longonly")
    results["布林突破(20,2)"] = pf

    # 基準：買進持有
    bh = vbt.Portfolio.from_holding(close, fees=fees, freq="1D", init_cash=1_000_000)
    results["買進持有(基準)"] = bh

    rows = []
    for name, pf in results.items():
        stats = pf.stats()
        rows.append({
            "策略": name,
            "總報酬%": round(pf.total_return() * 100, 2),
            "年化報酬%": round(stats.get("Annualized Return [%]", np.nan), 2),
            "最大回撤%": round(stats.get("Max Drawdown [%]", np.nan), 2),
            "夏普率": round(stats.get("Sharpe Ratio", np.nan), 2),
            "交易次數": int(stats.get("Total Trades", 0)),
        })
    return pd.DataFrame(rows).set_index("策略")

# ----------------------------------------------------------------------
# 3. 訊號層：用回測勝出的策略邏輯，產出未來 5 個交易日情境表
#    （本版採「條件式計畫」：依使用者給的區間產出，不預測點位）
# ----------------------------------------------------------------------
def build_plan(stock_id: str, df: pd.DataFrame,
               buy_low: float, buy_high: float,
               target: float, stop: float) -> str:
    last = float(df["Close"].iloc[-1])
    last_date = df.index[-1].strftime("%Y-%m-%d")
    entry = (buy_low + buy_high) / 2
    upside = (target / entry - 1) * 100
    downside = (stop / entry - 1) * 100
    rr = (target - entry) / (entry - stop)

    # 取得接下來 5 個營業日（跳過週末）
    days = pd.bdate_range(start=df.index[-1] + pd.Timedelta(days=1), periods=5)
    day_str = "、".join(d.strftime("%m/%d(%a)") for d in days)

    if last >= buy_high:
        stance = f"現價 {last:.0f} 已高於買進區間上緣，建議不追高，等待回測區間下緣 {buy_low:.0f} 附近再分批承接。"
    elif last >= buy_low:
        stance = f"現價 {last:.0f} 落在買進區間內，可依計畫分批建立部位。"
    elif last >= stop:
        stance = f"現價 {last:.0f} 低於買進區間，可於接近失效價 {stop:.0f} 前觀察承接，或等站回 {buy_low:.0f} 再進場。"
    else:
        stance = f"現價 {last:.0f} 已跌破失效價 {stop:.0f}，依紀律退出觀望，不攤平。"

    msg = f"""📊 【{stock_id} 台積電】量化回測 + 交易計畫
資料截至：{last_date}｜收盤：{last:.0f}

🔬 回測比較（近5年、含手續費稅費滑價）：
{BT_TABLE}

📋 未來 5 個交易日條件式計畫（{day_str}）：
・買進區間：{buy_low:.0f} ~ {buy_high:.0f}（以中間價 {entry:.0f} 試算）
・目標價：{target:.0f}（+{upside:.1f}%）
・失效價：{stop:.0f}（-{abs(downside):.1f}%）
・風險報酬比：1 : {rr:.1f}

現況判斷：{stance}

建議資金控管：單筆風險 ≤ 總資金 1%，
部位張數 = 總資金×1% ÷ ({entry:.0f}−{stop:.0f}) ÷ 1000

⚠️ 僅供研究參考，不構成投資建議。回測績效不代表未來。"""
    return msg

# ----------------------------------------------------------------------
# 4. 推播層：Telegram（只送訊息，完全不接券商、不下單）
# ----------------------------------------------------------------------
def send_telegram(msg: str):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("\n" + msg + "\n\n（未設定 TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID，僅印出）")
        return
    from telegram import Bot
    import asyncio
    async def _send():
        bot = Bot(token=token)
        await bot.send_message(chat_id=chat_id, text=msg)
    asyncio.run(_send())
    print("✅ 已推播到 Telegram")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stock", default="2330")
    ap.add_argument("--buy-low", type=float, default=2330)
    ap.add_argument("--buy-high", type=float, default=2370)
    ap.add_argument("--target", type=float, default=2550)
    ap.add_argument("--stop", type=float, default=2280)
    args = ap.parse_args()

    df = load_data(args.stock)
    bt = run_backtest(df)
    print(bt)
    BT_TABLE = bt.to_string()
    send_telegram(build_plan(args.stock, df, args.buy_low, args.buy_high,
                             args.target, args.stop))
