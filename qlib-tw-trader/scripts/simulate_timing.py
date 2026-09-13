"""
比較不同交易時段的執行成本差異

場景：每天需要賣出一支股票、買入一支股票（換股）
因為是隨機股票，實際執行價格 ≈ 該時段的全市場平均偏差。

三種策略：
  A. 早賣午買：09:00 賣出，12:00 買入
  B. 全在開盤：09:00 買賣
  C. 全在收盤：close 買賣（基準）
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

INSTRUMENTS_FILE = Path("data/qlib/instruments/all.txt")
BATCH_SIZE = 25


def load_stock_list() -> list[str]:
    with open(INSTRUMENTS_FILE) as f:
        return [line.strip().split()[0] for line in f if line.strip()]


def download_intraday_data(stock_ids: list[str]) -> pd.DataFrame:
    tickers = [f"{sid}.TW" for sid in stock_ids]
    all_records = []

    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i : i + BATCH_SIZE]
        print(f"  下載 {i + 1}-{min(i + BATCH_SIZE, len(tickers))} / {len(tickers)}...")
        data = yf.download(batch, interval="1h", period="2y", progress=False)
        if data.empty:
            continue
        if data.index.tz is not None:
            data.index = data.index.tz_convert("Asia/Taipei")
        else:
            data.index = data.index.tz_localize("UTC").tz_convert("Asia/Taipei")

        if len(batch) == 1:
            ticker = batch[0]
            stock_id = ticker.replace(".TW", "")
            sub = data.dropna(subset=["Close"]).reset_index()
            dt_col = "Datetime" if "Datetime" in sub.columns else sub.columns[0]
            all_records.append(pd.DataFrame({
                "datetime": sub[dt_col], "symbol": stock_id,
                "open": sub["Open"].values, "close": sub["Close"].values,
            }))
        else:
            for ticker in batch:
                stock_id = ticker.replace(".TW", "")
                try:
                    sub = data.xs(ticker, level="Ticker", axis=1)
                except KeyError:
                    continue
                sub = sub.dropna(subset=["Close"])
                if sub.empty:
                    continue
                sub = sub.reset_index()
                dt_col = "Datetime" if "Datetime" in sub.columns else sub.columns[0]
                all_records.append(pd.DataFrame({
                    "datetime": sub[dt_col], "symbol": stock_id,
                    "open": sub["Open"].values, "close": sub["Close"].values,
                }))
        if i + BATCH_SIZE < len(tickers):
            time.sleep(1)

    if not all_records:
        return pd.DataFrame()
    return pd.concat(all_records, ignore_index=True)


def compute_daily_edges(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    每天每支股票的各時段 open 相對日收盤的偏差，
    然後計算每天的「全市場平均偏差」。
    """
    df = raw_df.copy()
    df["trading_date"] = df["datetime"].dt.date
    df["hour"] = df["datetime"].dt.hour

    df = df[df["hour"].between(9, 13)].copy()

    # daily close = 13:00 bar close
    daily_close = df.groupby(["symbol", "trading_date"])["close"].transform("last")
    df["dev_open"] = (df["open"] - daily_close) / daily_close

    # 每天每時段的全市場平均偏差
    daily_avg = (
        df.groupby(["trading_date", "hour"])["dev_open"]
        .mean()
        .unstack(level="hour")
    )

    # 重命名 columns
    daily_avg.columns = [f"dev_{h:02d}" for h in daily_avg.columns]

    # 計算各策略的每日邊際
    # 邊際 = 賣出偏差 - 買入偏差（相對收盤基準）
    # 正值 = 比收盤策略多賺的
    result = pd.DataFrame(index=daily_avg.index)

    # A: 早賣午買 (sell@09, buy@12)
    result["edge_A"] = daily_avg["dev_09"] - daily_avg["dev_12"]

    # B: 全在開盤 (sell@09, buy@09) → 互相抵消 = 0
    result["edge_B"] = 0.0

    # C: 全在收盤 → 基準 = 0
    result["edge_C"] = 0.0

    # 額外：sell@09 only（賣出在開盤的優勢）
    result["sell_edge_09"] = daily_avg["dev_09"]

    # 額外：buy@12 only（買入在午盤的優勢，負值 = 買便宜）
    result["buy_edge_12"] = -daily_avg["dev_12"]

    # 記錄各時段偏差
    for col in daily_avg.columns:
        result[col] = daily_avg[col]

    result["weekday"] = pd.to_datetime(result.index).weekday

    return result.dropna()


def main():
    print("=" * 70)
    print("交易時段執行成本比較")
    print("=" * 70)

    stock_ids = load_stock_list()
    print(f"股票清單：{len(stock_ids)} 支")

    print("\n下載 1h 資料中...")
    raw_df = download_intraday_data(stock_ids)
    if raw_df.empty:
        print("無法取得任何資料！")
        return
    print(f"下載完成：{len(raw_df):,} 筆")

    print("\n計算每日邊際...")
    edges = compute_daily_edges(raw_df)
    n_days = len(edges)
    print(f"有效交易日：{n_days}")

    # ==============================
    # 1. 每日平均邊際
    # ==============================
    print("\n" + "=" * 70)
    print("1. 每日平均邊際（相對收盤買賣基準）")
    print("=" * 70)

    strategies = [
        ("A. 早賣午買 (sell@09, buy@12)", "edge_A"),
        ("B. 全在開盤 (sell@09, buy@09)", "edge_B"),
        ("C. 全在收盤 (基準)", "edge_C"),
    ]

    for name, col in strategies:
        avg = edges[col].mean() * 100
        print(f"  {name}: {avg:+.4f}%/天")

    print(f"\n  A vs C 日均優勢: {edges['edge_A'].mean() * 100:+.4f}%")
    print(f"  其中 賣出邊際 (sell@09 vs close): {edges['sell_edge_09'].mean() * 100:+.4f}%")
    print(f"  其中 買入邊際 (buy@12 vs close): {edges['buy_edge_12'].mean() * 100:+.4f}%")

    # ==============================
    # 2. 累積邊際
    # ==============================
    print("\n" + "=" * 70)
    print("2. 累積邊際")
    print("=" * 70)

    cum_a = edges["edge_A"].cumsum()

    print(f"\n  期間：{edges.index[0]} ~ {edges.index[-1]}（{n_days} 天）")
    print(f"  A 累積邊際：{cum_a.iloc[-1] * 100:+.3f}%")
    print(f"  年化（250天）：{edges['edge_A'].mean() * 250 * 100:+.3f}%")

    # ==============================
    # 3. 星期分解
    # ==============================
    print("\n" + "=" * 70)
    print("3. 星期分解（A 策略日均邊際）")
    print("=" * 70)

    weekday_names = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    wd_stats = edges.groupby("weekday")["edge_A"].agg(["mean", "std", "count"])

    print(f"\n  {'星期':>4} | {'日均邊際':>10} | {'標準差':>10} | {'天數':>6}")
    print("  " + "-" * 40)
    for wd in range(5):
        if wd in wd_stats.index:
            r = wd_stats.loc[wd]
            print(f"  {weekday_names[wd]:>4} | {r['mean'] * 100:>+8.4f}% | {r['std'] * 100:>8.4f}% | {int(r['count']):>6}")

    # ==============================
    # 4. 各時段平均偏差
    # ==============================
    print("\n" + "=" * 70)
    print("4. 各時段 Open 偏差（全期間每日平均）")
    print("=" * 70)

    dev_cols = [c for c in edges.columns if c.startswith("dev_")]
    print(f"\n  {'時段':>6} | {'Avg偏差':>10} | {'作為賣出邊際':>12} | {'作為買入成本':>12}")
    print("  " + "-" * 50)
    for col in sorted(dev_cols):
        hour = col.replace("dev_", "")
        avg = edges[col].mean()
        print(f"  {hour}  | {avg * 100:>+8.4f}% | {avg * 100:>+10.4f}% | {-avg * 100:>+10.4f}%")

    # ==============================
    # 5. 實際金額影響
    # ==============================
    print("\n" + "=" * 70)
    print("5. 實際金額影響（假設每日交易金額 100 萬）")
    print("=" * 70)

    daily_amount = 1_000_000  # TWD
    daily_edge_twd = edges["edge_A"].mean() * daily_amount
    annual_edge_twd = daily_edge_twd * 250

    print(f"\n  每日交易金額：{daily_amount:,} TWD")
    print(f"  A 策略日均多賺：{daily_edge_twd:+,.0f} TWD")
    print(f"  A 策略年化多賺：{annual_edge_twd:+,.0f} TWD")

    for amount in [5_000_000, 10_000_000]:
        annual = edges["edge_A"].mean() * amount * 250
        print(f"  若每日 {amount / 1_000_000:.0f}M：年化 {annual:+,.0f} TWD")


if __name__ == "__main__":
    main()
