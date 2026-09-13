"""
回填單一股票多年歷史資料

從各資料源（FinMind/TWSE/yfinance）下載指定股票的完整歷史資料，
包含 OHLCV、還原股價、PER、三大法人、融資融券、外資持股等。

用法: python scripts/backfill_stock.py <stock_id> <start_date> <end_date>
範例: python scripts/backfill_stock.py 2330 2020-01-01 2026-01-28
"""

import asyncio
import sys
from datetime import date, timedelta

from src.repositories import init_db
from src.services import DataService, Dataset


async def backfill_stock(stock_id: str, start_date: date, end_date: date):
    """回填單一股票所有資料類型"""
    init_db()
    ds = DataService()

    datasets = [
        ("OHLCV", Dataset.OHLCV, ds.get_ohlcv),
        ("AdjClose", Dataset.ADJ_CLOSE, ds.get_adj_close),
        ("PER", Dataset.PER, ds.get_per),
        ("Institutional", Dataset.INSTITUTIONAL, ds.get_institutional),
        ("Margin", Dataset.MARGIN, ds.get_margin),
        ("Shareholding", Dataset.SHAREHOLDING, ds.get_shareholding),
    ]

    print("=" * 70)
    print(f"回填股票: {stock_id}")
    print(f"期間: {start_date} ~ {end_date}")
    print("=" * 70)

    results = {}

    for name, dataset, getter in datasets:
        print(f"\n{'─' * 50}")
        print(f"取得 {name}...")

        try:
            data = await getter(stock_id, start_date, end_date)
            count = len(data)
            results[name] = count

            if data:
                first_date = data[0].date
                last_date = data[-1].date
                print(f"  ✅ {count} 筆 ({first_date} ~ {last_date})")
            else:
                print(f"  ⚠️ 無資料")

        except Exception as e:
            print(f"  ❌ 錯誤: {e}")
            results[name] = 0

        # 限流
        await asyncio.sleep(1)

    # 總結
    print(f"\n{'=' * 70}")
    print("回填結果")
    print("=" * 70)
    for name, count in results.items():
        status = "✅" if count > 0 else "❌"
        print(f"  {name}: {status} {count} 筆")

    total = sum(results.values())
    print(f"\n總計: {total} 筆資料")

    return results


async def main():
    if len(sys.argv) >= 4:
        stock_id = sys.argv[1]
        start_date = date.fromisoformat(sys.argv[2])
        end_date = date.fromisoformat(sys.argv[3])
    else:
        # 預設值
        stock_id = "2330"
        start_date = date(2020, 1, 1)
        end_date = date(2026, 1, 28)

    await backfill_stock(stock_id, start_date, end_date)


if __name__ == "__main__":
    asyncio.run(main())
