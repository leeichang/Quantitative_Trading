#!/usr/bin/env python3
"""
add_indicators.py — 讀取 fetch_data.py 產生的日K快取，計算技術指標並輸出

用法：
    # 處理單一股票
    python3 add_indicators.py 2330

    # 處理多檔
    python3 add_indicators.py 2330 2317 2454

    # 處理 data/ 底下所有已快取的股票
    python3 add_indicators.py all

輸入：data/{股票代號}.csv        （fetch_data.py 產生）
輸出：data/processed/{股票代號}.csv

計算的指標：
    MA5 / MA10 / MA20 / MA60       均線
    RSI14                           相對強弱指標
    KD (K9 / D9)                    隨機指標
    MACD / MACD_signal / MACD_hist  指數平滑異同移動平均
    BB_upper / BB_mid / BB_lower    布林通道 (20, 2)
    VOL_MA20                        20日均量
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import pandas_ta as ta

DATA_DIR = Path("data")
OUTPUT_DIR = DATA_DIR / "processed"

# FinMind TaiwanStockPrice 欄位對應到通用 OHLCV 名稱
COLUMN_MAP = {
    "open": "open",
    "max": "high",
    "min": "low",
    "close": "close",
    "Trading_Volume": "volume",
}


def load_raw(stock_id: str) -> pd.DataFrame:
    csv_path = DATA_DIR / f"{stock_id}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到 {csv_path}，請先執行 fetch_data.py {stock_id}")

    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.rename(columns=COLUMN_MAP)

    required = {"date", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"[{stock_id}] 資料缺少必要欄位：{missing}")

    df = df.sort_values("date").reset_index(drop=True)
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 均線
    for window in (5, 10, 20, 60):
        df[f"MA{window}"] = ta.sma(df["close"], length=window)

    # RSI
    df["RSI14"] = ta.rsi(df["close"], length=14)

    # KD (Stochastic)，pandas-ta 預設欄位為 STOCHk_9_3_3 / STOCHd_9_3_3
    stoch = ta.stoch(df["high"], df["low"], df["close"], k=9, d=3, smooth_k=3)
    if stoch is not None:
        df["K9"] = stoch.iloc[:, 0]
        df["D9"] = stoch.iloc[:, 1]

    # MACD
    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    if macd is not None:
        df["MACD"] = macd.iloc[:, 0]
        df["MACD_hist"] = macd.iloc[:, 1]
        df["MACD_signal"] = macd.iloc[:, 2]

    # 布林通道
    bb = ta.bbands(df["close"], length=20, std=2)
    if bb is not None:
        df["BB_lower"] = bb.iloc[:, 0]
        df["BB_mid"] = bb.iloc[:, 1]
        df["BB_upper"] = bb.iloc[:, 2]

    # 均量
    df["VOL_MA20"] = ta.sma(df["volume"], length=20)

    return df


def process_stock(stock_id: str) -> None:
    print(f"[{stock_id}] 讀取原始資料 ...")
    raw = load_raw(stock_id)

    print(f"[{stock_id}] 計算技術指標 ...")
    enriched = add_indicators(raw)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{stock_id}.csv"
    enriched.to_csv(out_path, index=False)
    print(f"[{stock_id}] 已輸出 {len(enriched)} 筆、{len(enriched.columns)} 個欄位 → {out_path}")


def existing_cached_stock_ids() -> list[str]:
    if not DATA_DIR.exists():
        return []
    return [p.stem for p in DATA_DIR.glob("*.csv") if p.parent == DATA_DIR]


def main() -> None:
    parser = argparse.ArgumentParser(description="計算技術指標並輸出到 data/processed/")
    parser.add_argument(
        "stock_ids", nargs="+",
        help="股票代號（如 2330），或輸入 all 處理所有已快取股票",
    )
    args = parser.parse_args()

    if args.stock_ids == ["all"]:
        stock_ids = existing_cached_stock_ids()
        if not stock_ids:
            print("data/ 底下尚無任何快取，請先執行 fetch_data.py")
            sys.exit(1)
    else:
        stock_ids = args.stock_ids

    for stock_id in stock_ids:
        try:
            process_stock(stock_id)
        except (FileNotFoundError, ValueError) as e:
            print(f"[{stock_id}] 略過：{e}")


if __name__ == "__main__":
    main()
