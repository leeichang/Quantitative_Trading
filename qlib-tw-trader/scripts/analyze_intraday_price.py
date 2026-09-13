"""
台股日內價格實證分析

使用 yfinance 1h bars 分析各時段價格與日收盤價的偏差，
找出最佳買入（做多）與賣出時段。
"""

import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

INSTRUMENTS_FILE = Path("data/qlib/instruments/all.txt")
REPORT_PATH = Path("reports/intraday-price-analysis.md")
BATCH_SIZE = 25
WEEKDAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri"]
# 台股交易時段 30m bars
VALID_SLOTS = ["09:00", "10:00", "11:00", "12:00", "13:00"]


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
            all_records.append(
                pd.DataFrame(
                    {
                        "datetime": sub[dt_col],
                        "symbol": stock_id,
                        "open": sub["Open"].values,
                        "close": sub["Close"].values,
                    }
                )
            )
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
                all_records.append(
                    pd.DataFrame(
                        {
                            "datetime": sub[dt_col],
                            "symbol": stock_id,
                            "open": sub["Open"].values,
                            "close": sub["Close"].values,
                        }
                    )
                )

        if i + BATCH_SIZE < len(tickers):
            time.sleep(1)

    if not all_records:
        return pd.DataFrame()

    return pd.concat(all_records, ignore_index=True)


def compute_deviations(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["trading_date"] = df["datetime"].dt.date
    df["time_slot"] = df["datetime"].dt.strftime("%H:%M")
    df["weekday"] = df["datetime"].dt.weekday

    # 只保留台股交易時段
    df = df[df["time_slot"].isin(VALID_SLOTS)].copy()

    # 每天每支股票的收盤價 = 當天最後一根 bar 的 close
    daily_close = df.groupby(["symbol", "trading_date"])["close"].transform("last")

    df["dev_close"] = (df["close"] - daily_close) / daily_close
    df["dev_open"] = (df["open"] - daily_close) / daily_close

    return df


def aggregate_stats(df: pd.DataFrame) -> dict:
    results = {}

    for group_name, group_cols in [
        ("by_weekday_slot", ["weekday", "time_slot"]),
        ("by_slot", ["time_slot"]),
    ]:
        g = df.groupby(group_cols)

        stats = g.agg(
            mean_dev_close=("dev_close", "mean"),
            median_dev_close=("dev_close", "median"),
            std_dev_close=("dev_close", "std"),
            mean_dev_open=("dev_open", "mean"),
            median_dev_open=("dev_open", "median"),
            std_dev_open=("dev_open", "std"),
            count=("dev_close", "count"),
        )

        stats["pct_below_close"] = g["dev_close"].apply(lambda x: (x <= 0).mean())
        stats["pct_below_open"] = g["dev_open"].apply(lambda x: (x <= 0).mean())
        stats["pct_within_15bps_close"] = g["dev_close"].apply(
            lambda x: (x.abs() <= 0.0015).mean()
        )
        stats["pct_within_10bps_close"] = g["dev_close"].apply(
            lambda x: (x.abs() <= 0.0010).mean()
        )

        results[group_name] = stats

    return results


def format_pivot_table(
    stats: pd.DataFrame,
    value_col: str,
    fmt: str = "{:+.3f}",
    pct: bool = False,
) -> str:
    """將 (weekday, time_slot) 統計轉為 markdown 熱力圖表格"""
    if stats.index.nlevels < 2:
        return ""

    pivot = stats[value_col].unstack(level="time_slot")
    slots = [s for s in VALID_SLOTS if s in pivot.columns]

    header = "| 星期＼時段 | " + " | ".join(slots) + " |"
    sep = "|-----------|" + "|".join("------:" for _ in slots) + "|"

    rows = [header, sep]
    for wd in range(5):
        if wd not in pivot.index:
            continue
        cells = []
        for s in slots:
            val = pivot.loc[wd, s] if s in pivot.columns else float("nan")
            if pd.isna(val):
                cells.append("N/A")
            elif pct:
                cells.append(f"{val * 100:.1f}%")
            else:
                cells.append(fmt.format(val * 100))
        rows.append(f"| **{WEEKDAY_NAMES[wd]}** | " + " | ".join(cells) + " |")

    return "\n".join(rows)


def format_slot_table(stats: pd.DataFrame, columns: list[tuple[str, str, str]]) -> str:
    """格式化 by_slot 統計表。fmt: 'pct', 'dev', 'int'"""
    header = "| 時段 | " + " | ".join(label for _, label, _ in columns) + " |"
    sep = "|-------|" + "|".join("------:" for _ in columns) + "|"

    rows = [header, sep]
    for slot in VALID_SLOTS:
        if slot not in stats.index:
            continue
        cells = [slot]
        for col, _, fmt in columns:
            val = stats.loc[slot, col]
            if fmt == "pct":
                cells.append(f"{val * 100:.1f}%")
            elif fmt == "int":
                cells.append(f"{int(val):,}")
            else:
                cells.append(f"{val * 100:+.3f}%")
        rows.append("| " + " | ".join(cells) + " |")

    return "\n".join(rows)


def generate_report(stats: dict, coverage: dict) -> str:
    ws = stats["by_weekday_slot"]
    bs = stats["by_slot"]

    date_min = coverage["date_min"].strftime("%Y-%m-%d")
    date_max = coverage["date_max"].strftime("%Y-%m-%d")
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # 排除 13:00 bar（close = daily close by construction）
    bs_ex13 = bs.drop("13:00", errors="ignore")

    s = []

    # Header
    s.append("# 台股日內價格實證分析報告（1 小時線）")
    s.append("")
    s.append(f"**生成時間**：{now}")
    s.append(f"**資料期間**：{date_min} ~ {date_max}")
    s.append(f"**股票數**：{coverage['stocks_with_data']} 支（市值前 100 大）")
    s.append("**資料來源**：yfinance 1h bars")
    s.append("")
    s.append("---")

    # Coverage
    s.append("")
    s.append("## 資料覆蓋統計")
    s.append("")
    s.append("| 指標 | 數值 |")
    s.append("|------|------|")
    s.append(f"| 有資料的股票數 | {coverage['stocks_with_data']} |")
    s.append(f"| 無資料的股票數 | {coverage['stocks_without_data']} |")
    s.append(f"| 總交易日數 | {coverage['trading_days']} |")
    s.append(f"| 總觀測數 | {coverage['total_obs']:,} |")
    s.append("")
    s.append("> **注意**：13:00 bar 的 close = 日收盤價（by construction）。")
    s.append("")
    s.append("---")

    # Section 1: Overall stats
    s.append("")
    s.append("## 1. 整體統計（不分星期）")
    s.append("")
    s.append("Bar Open 偏差 = 該時段開始時的市場價格 vs 日收盤價。負值 = 低於收盤（買入有利），正值 = 高於收盤（賣出有利）。")
    s.append("")
    s.append(
        format_slot_table(
            bs,
            [
                ("mean_dev_open", "Avg Open偏差", "dev"),
                ("mean_dev_close", "Avg Close偏差", "dev"),
                ("median_dev_open", "Median Open偏差", "dev"),
                ("pct_below_open", "Open≤收盤%", "pct"),
                ("pct_below_close", "Close≤收盤%", "pct"),
                ("pct_within_15bps_close", "Close±15bps%", "pct"),
                ("count", "觀測數", "int"),
            ],
        )
    )
    s.append("")

    # Section 2: Best BUY time
    s.append("---")
    s.append("")
    s.append("## 2. 最佳買入時段（做多）")
    s.append("")
    s.append("買入時段的衡量：Bar Open 偏差越負越好（買入價低於收盤價）。")
    s.append("")

    # Rank by mean_dev_open ascending (most negative first)
    s.append("### 2.1 時段排名（Bar Open 偏差，由低到高）")
    s.append("")
    buy_rank = bs_ex13["mean_dev_open"].sort_values()
    s.append("| 排名 | 時段 | Avg Open偏差 | Open≤收盤% | Median Open偏差 |")
    s.append("|------|-------|-------------|-----------|----------------|")
    for rank, (slot, val) in enumerate(buy_rank.items(), 1):
        below = bs_ex13.loc[slot, "pct_below_open"]
        median = bs_ex13.loc[slot, "median_dev_open"]
        s.append(f"| {rank} | {slot} | {val * 100:+.3f}% | {below * 100:.1f}% | {median * 100:+.3f}% |")
    s.append("")

    # Heatmap
    s.append("### 2.2 星期 × 時段 熱力圖（Avg Open 偏差 %）")
    s.append("")
    s.append(format_pivot_table(ws, "mean_dev_open"))
    s.append("")

    # Best buy recommendation
    best_buy = buy_rank.index[0]
    s.append("### 2.3 買入建議")
    s.append("")
    s.append(f"**推薦買入時段：{best_buy}**（平均偏差 {buy_rank.iloc[0] * 100:+.3f}%）")
    s.append("")

    # Find best weekday × slot for buying
    ws_ex13 = ws.drop("13:00", level="time_slot", errors="ignore")
    best_buy_ws = ws_ex13["mean_dev_open"].sort_values().head(5)
    s.append("**最佳買入 Top-5（星期 × 時段）**：")
    s.append("")
    s.append("| 排名 | 星期 | 時段 | Avg Open偏差 |")
    s.append("|------|------|------|-------------|")
    for rank, ((wd, slot), val) in enumerate(best_buy_ws.items(), 1):
        s.append(f"| {rank} | {WEEKDAY_NAMES[wd]} | {slot} | {val * 100:+.3f}% |")
    s.append("")

    # Section 3: Best SELL time
    s.append("---")
    s.append("")
    s.append("## 3. 最佳賣出時段")
    s.append("")
    s.append("賣出時段的衡量：Bar Open 偏差越正越好（賣出價高於收盤價）。")
    s.append("")

    # Rank by mean_dev_open descending (most positive first)
    s.append("### 3.1 時段排名（Bar Open 偏差，由高到低）")
    s.append("")
    sell_rank = bs_ex13["mean_dev_open"].sort_values(ascending=False)
    s.append("| 排名 | 時段 | Avg Open偏差 | Open>收盤% | Median Open偏差 |")
    s.append("|------|-------|-------------|-----------|----------------|")
    for rank, (slot, val) in enumerate(sell_rank.items(), 1):
        above = 1 - bs_ex13.loc[slot, "pct_below_open"]
        median = bs_ex13.loc[slot, "median_dev_open"]
        s.append(f"| {rank} | {slot} | {val * 100:+.3f}% | {above * 100:.1f}% | {median * 100:+.3f}% |")
    s.append("")

    # Heatmap (same data, different perspective)
    s.append("### 3.2 星期 × 時段 熱力圖（Open > 收盤% ）")
    s.append("")
    # Create "pct_above_open" for display
    ws_copy = ws.copy()
    ws_copy["pct_above_open"] = 1 - ws_copy["pct_below_open"]
    s.append(format_pivot_table(ws_copy, "pct_above_open", pct=True))
    s.append("")

    # Best sell recommendation
    best_sell = sell_rank.index[0]
    s.append("### 3.3 賣出建議")
    s.append("")
    s.append(f"**推薦賣出時段：{best_sell}**（平均偏差 {sell_rank.iloc[0] * 100:+.3f}%）")
    s.append("")

    # Find best weekday × slot for selling
    best_sell_ws = ws_ex13["mean_dev_open"].sort_values(ascending=False).head(5)
    s.append("**最佳賣出 Top-5（星期 × 時段）**：")
    s.append("")
    s.append("| 排名 | 星期 | 時段 | Avg Open偏差 |")
    s.append("|------|------|------|-------------|")
    for rank, ((wd, slot), val) in enumerate(best_sell_ws.items(), 1):
        s.append(f"| {rank} | {WEEKDAY_NAMES[wd]} | {slot} | {val * 100:+.3f}% |")
    s.append("")

    # Section 4: Favorable ratios heatmap
    s.append("---")
    s.append("")
    s.append("## 4. 做多有利比例（Bar Open ≤ 收盤價 %）")
    s.append("")
    s.append(format_pivot_table(ws, "pct_below_open", pct=True))
    s.append("")

    # Section 5: Near close heatmap
    s.append("## 5. 接近收盤價比例（Bar Close ±0.15%）")
    s.append("")
    s.append(format_pivot_table(ws, "pct_within_15bps_close", pct=True))
    s.append("")

    # Section 6: Key findings
    s.append("---")
    s.append("")
    s.append("## 6. 關鍵發現")
    s.append("")
    s.append(f"1. **最佳買入時段**：{best_buy}（Open 平均低於收盤 {abs(buy_rank.iloc[0]) * 100:.3f}%）" if buy_rank.iloc[0] < 0 else f"1. **最佳買入時段**：{best_buy}（Open 平均偏差 {buy_rank.iloc[0] * 100:+.3f}%，所有時段均高於收盤）")
    s.append(f"2. **最佳賣出時段**：{best_sell}（Open 平均高於收盤 {sell_rank.iloc[0] * 100:.3f}%）")

    # 13:00 bar open info
    if "13:00" in bs.index:
        s.append(f"3. **13:00 價格**（最後 30 分鐘開始）：Open 偏差 {bs.loc['13:00', 'mean_dev_open'] * 100:+.3f}%，{bs.loc['13:00', 'pct_below_open'] * 100:.1f}% 低於收盤")
    s.append("")

    return "\n".join(s)


def print_results(stats: dict, coverage: dict):
    bs = stats["by_slot"]

    print("\n" + "=" * 80)
    print("整體統計（不分星期）")
    print("=" * 80)
    print(f"{'時段':>6} | {'Avg Open偏差':>12} | {'Avg Close偏差':>13} | {'Open≤收盤%':>10} | {'Close≤收盤%':>11} | {'±15bps%':>7} | {'觀測數':>8}")
    print("-" * 80)
    for slot in VALID_SLOTS:
        if slot not in bs.index:
            continue
        r = bs.loc[slot]
        print(
            f"{slot}  | {r['mean_dev_open'] * 100:>+9.3f}%  | {r['mean_dev_close'] * 100:>+10.3f}%  | {r['pct_below_open'] * 100:>7.1f}%  | {r['pct_below_close'] * 100:>8.1f}%  | {r['pct_within_15bps_close'] * 100:>5.1f}% | {int(r['count']):>8,}"
        )

    # Best buy/sell summary
    bs_ex13 = bs.drop("13:00", errors="ignore")
    best_buy = bs_ex13["mean_dev_open"].idxmin()
    best_sell = bs_ex13["mean_dev_open"].idxmax()
    print(f"\n最佳買入時段：{best_buy}（Avg Open偏差 {bs.loc[best_buy, 'mean_dev_open'] * 100:+.3f}%）")
    print(f"最佳賣出時段：{best_sell}（Avg Open偏差 {bs.loc[best_sell, 'mean_dev_open'] * 100:+.3f}%）")


def main():
    print("=" * 80)
    print("台股日內價格實證分析（yfinance 1h bars）")
    print("=" * 80)

    stock_ids = load_stock_list()
    print(f"股票清單：{len(stock_ids)} 支")

    print("\n下載 1h 資料中（最近 2 年）...")
    raw_df = download_intraday_data(stock_ids)

    if raw_df.empty:
        print("無法取得任何資料！")
        return

    print(f"下載完成：{len(raw_df):,} 筆原始資料")

    print("\n計算偏差中...")
    dev_df = compute_deviations(raw_df)
    print(f"有效觀測：{len(dev_df):,} 筆")

    print("\n聚合統計中...")
    stats = aggregate_stats(dev_df)

    coverage = {
        "stocks_with_data": dev_df["symbol"].nunique(),
        "stocks_without_data": len(stock_ids) - dev_df["symbol"].nunique(),
        "trading_days": dev_df["trading_date"].nunique(),
        "total_obs": len(dev_df),
        "date_min": dev_df["datetime"].min(),
        "date_max": dev_df["datetime"].max(),
    }

    print_results(stats, coverage)

    print("\n產生報告中...")
    report = generate_report(stats, coverage)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"報告已儲存至：{REPORT_PATH}")


if __name__ == "__main__":
    main()
