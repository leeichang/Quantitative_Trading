#!/usr/bin/env python3
"""
fetch_data.py — FinMind 台股日K資料抓取與本地 CSV 快取

用法：
    # 抓單一股票，從指定日期到今天（若已有快取，會自動增量更新）
    python3 fetch_data.py 2330 2020-01-01

    # 更新 data/ 底下所有已存在的快取股票到今天
    python3 fetch_data.py update

    # 一次抓多檔
    python3 fetch_data.py 2330 2317 2454 --start 2020-01-01

環境變數：
    FINMIND_TOKEN   FinMind API token（免費方案可留空，但有 request 上限）

FinMind 免費方案限制約 600 req/hr，本腳本每次呼叫間會 sleep 避免被限流。
"""

import argparse
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"
DATASET = "TaiwanStockPrice"  # 日K：開高低收、成交量、漲跌
DATA_DIR = Path("data")
REQUEST_INTERVAL_SEC = 1.0  # 每次 request 間隔，避免撞到 rate limit


def fetch_range(stock_id: str, start_date: str, end_date: str, token: str = "") -> pd.DataFrame:
    """向 FinMind 拉取指定股票、指定期間的日K資料"""
    params = {
        "dataset": DATASET,
        "data_id": stock_id,
        "start_date": start_date,
        "end_date": end_date,
        "token": token,
    }
    resp = requests.get(FINMIND_URL, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()

    if payload.get("status") != 200:
        raise RuntimeError(f"[{stock_id}] FinMind 回傳錯誤：{payload.get('msg')}")

    df = pd.DataFrame(payload["data"])
    return df


def load_cache(stock_id: str) -> pd.DataFrame | None:
    """讀取本地已存在的快取，沒有則回傳 None"""
    csv_path = DATA_DIR / f"{stock_id}.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path, parse_dates=["date"])
    return None


def save_cache(stock_id: str, df: pd.DataFrame) -> None:
    """存檔，依日期排序、去重"""
    DATA_DIR.mkdir(exist_ok=True)
    df = df.sort_values("date").drop_duplicates(subset="date", keep="last")
    csv_path = DATA_DIR / f"{stock_id}.csv"
    df.to_csv(csv_path, index=False)
    print(f"[{stock_id}] 已儲存 {len(df)} 筆資料 → {csv_path}")


def update_stock(stock_id: str, start_date: str, token: str, end_date: str | None = None) -> None:
    """增量更新單一股票：若已有快取，只抓缺少的日期"""
    end_date = end_date or date.today().isoformat()
    cached = load_cache(stock_id)

    if cached is not None and not cached.empty:
        last_date = cached["date"].max().date()
        fetch_start = (last_date + timedelta(days=1)).isoformat()
        if fetch_start > end_date:
            print(f"[{stock_id}] 已是最新，無需更新")
            return
    else:
        cached = pd.DataFrame()
        fetch_start = start_date

    print(f"[{stock_id}] 抓取 {fetch_start} ~ {end_date} ...")
    new_df = fetch_range(stock_id, fetch_start, end_date, token)

    if new_df.empty:
        print(f"[{stock_id}] 此區間無新資料（可能是非交易日）")
        if cached.empty:
            return
        save_cache(stock_id, cached)
        return

    new_df["date"] = pd.to_datetime(new_df["date"])
    merged = pd.concat([cached, new_df], ignore_index=True)
    save_cache(stock_id, merged)


def existing_cached_stock_ids() -> list[str]:
    """列出 data/ 底下目前已有快取的股票代號"""
    if not DATA_DIR.exists():
        return []
    return [p.stem for p in DATA_DIR.glob("*.csv")]


def main() -> None:
    parser = argparse.ArgumentParser(description="FinMind 台股日K資料抓取與快取")
    parser.add_argument(
        "stock_ids", nargs="+",
        help="股票代號（如 2330），或輸入 update 來更新所有已快取股票",
    )
    parser.add_argument("--start", default="2015-01-01", help="起始日期，預設 2015-01-01")
    parser.add_argument("--end", default=None, help="結束日期，預設為今天")
    args = parser.parse_args()

    token = os.environ.get("FINMIND_TOKEN", "")

    if args.stock_ids == ["update"]:
        stock_ids = existing_cached_stock_ids()
        if not stock_ids:
            print("data/ 底下尚無任何快取，請先指定股票代號抓取初始資料")
            sys.exit(1)
    else:
        stock_ids = args.stock_ids

    for i, stock_id in enumerate(stock_ids):
        update_stock(stock_id, args.start, token, args.end)
        if i < len(stock_ids) - 1:
            time.sleep(REQUEST_INTERVAL_SEC)


if __name__ == "__main__":
    main()
