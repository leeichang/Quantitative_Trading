"""
開盤前資料完整度檢查

檢查 TWSE/FinMind/yfinance 各資料源是否已更新到指定日期，
用於確認每日交易前的資料完整度。

用法: python scripts/check_previous_day.py [日期]
範例: python scripts/check_previous_day.py 2026-01-29

若不指定日期，自動檢查前一個交易日。
退出碼：0 = 關鍵資料完整，1 = 資料不完整。
"""

import asyncio
import sys
from datetime import date, datetime, timedelta
import httpx

# 預設檢查日期（會被參數覆蓋）
TARGET_DATE = date(2026, 1, 29)
TARGET_DATE_STR = "2026-01-29"
TARGET_DATE_ROC7 = "1150129"  # TWSE OpenAPI 格式（民國年7位）


def parse_date_arg():
    """解析命令列日期參數"""
    global TARGET_DATE, TARGET_DATE_STR, TARGET_DATE_ROC7

    if len(sys.argv) > 1:
        TARGET_DATE_STR = sys.argv[1]
        TARGET_DATE = date.fromisoformat(TARGET_DATE_STR)
    else:
        # 自動計算前一個交易日（簡單版：週一查週五，其他查前一天）
        today = date.today()
        if today.weekday() == 0:  # 週一
            TARGET_DATE = today - timedelta(days=3)
        elif today.weekday() == 6:  # 週日
            TARGET_DATE = today - timedelta(days=2)
        else:
            TARGET_DATE = today - timedelta(days=1)
        TARGET_DATE_STR = TARGET_DATE.isoformat()

    # 轉換為民國年格式
    roc_year = TARGET_DATE.year - 1911
    TARGET_DATE_ROC7 = f"{roc_year}{TARGET_DATE.strftime('%m%d')}"


def parse_roc7_date(raw: str) -> str | None:
    """解析民國年7位日期（如 1150129），返回 YYYY-MM-DD"""
    if not raw or len(raw) != 7:
        return None
    try:
        year = int(raw[:3]) + 1911
        month = raw[3:5]
        day = raw[5:7]
        return f"{year}-{month}-{day}"
    except:
        return None


# ============================================
# 資料檢查清單
# ============================================

DATASETS_TO_CHECK = {
    # === TWSE OpenAPI（使用 OpenAPI 檢查）===
    "TaiwanStockPrice": {
        "name": "日K線",
        "source": "twse_openapi",
        "url": "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL",
        "critical": True,
    },
    "TaiwanStockPER": {
        "name": "PER/PBR",
        "source": "twse_openapi",
        "url": "https://openapi.twse.com.tw/v1/exchangeReport/BWIBBU_ALL",
        "critical": True,
    },
    "TaiwanStockInstitutionalInvestorsBuySell": {
        "name": "三大法人(個股)",
        "source": "twse_openapi",
        "url": "https://openapi.twse.com.tw/v1/exchangeReport/TWT43U_ALL",
        "critical": True,
    },
    "TaiwanStockMarginPurchaseShortSale": {
        "name": "融資融券",
        "source": "twse_openapi",
        "url": "https://openapi.twse.com.tw/v1/exchangeReport/MI_MARGN_ALL",
        "critical": True,
    },
    "TaiwanStockShareholding": {
        "name": "外資持股",
        "source": "twse_openapi",
        "url": "https://openapi.twse.com.tw/v1/exchangeReport/MI_QFIIS_ALL",
        "critical": False,  # T+1
    },

    # === FinMind ===
    "TaiwanFuturesDaily": {
        "name": "期貨日成交",
        "source": "finmind",
        "dataset": "TaiwanFuturesDaily",
        "data_id": "TX",
        "critical": False,
    },
    "TaiwanOptionDaily": {
        "name": "選擇權日成交",
        "source": "finmind",
        "dataset": "TaiwanOptionDaily",
        "data_id": "TXO",
        "critical": False,
    },
    "TaiwanFuturesInstitutionalInvestors": {
        "name": "期貨三大法人",
        "source": "finmind",
        "dataset": "TaiwanFuturesInstitutionalInvestors",
        "data_id": "TX",
        "critical": False,
    },
    "TaiwanExchangeRate": {
        "name": "匯率",
        "source": "finmind",
        "dataset": "TaiwanExchangeRate",
        "data_id": "USD",
        "critical": False,
    },

    # === yfinance ===
    "TaiwanStockPriceAdj": {
        "name": "還原股價",
        "source": "yfinance",
        "stock_id": "2330",
        "critical": True,
    },
}


async def check_twse_openapi(client: httpx.AsyncClient, key: str, info: dict) -> dict:
    """檢查 TWSE OpenAPI"""
    try:
        resp = await client.get(info["url"], timeout=30, follow_redirects=True)
        if resp.status_code != 200:
            return {
                "key": key,
                "name": info["name"],
                "ok": False,
                "date": None,
                "count": 0,
                "error": f"HTTP {resp.status_code}",
            }

        # OpenAPI 可能返回 HTML（無資料時）
        content_type = resp.headers.get("content-type", "")
        if "html" in content_type:
            return {
                "key": key,
                "name": info["name"],
                "ok": False,
                "date": None,
                "count": 0,
                "error": "無資料 (HTML)",
            }

        data = resp.json()
        if not data or not isinstance(data, list):
            return {
                "key": key,
                "name": info["name"],
                "ok": False,
                "date": None,
                "count": 0,
                "error": "空資料",
            }

        # 找日期欄位（OpenAPI 用 Date）
        first_row = data[0]
        raw_date = first_row.get("Date", first_row.get("日期", ""))
        latest_date = parse_roc7_date(raw_date) if raw_date else None
        ok = latest_date == TARGET_DATE_STR

        return {
            "key": key,
            "name": info["name"],
            "ok": ok,
            "date": latest_date or raw_date,
            "count": len(data),
            "error": None,
        }

    except Exception as e:
        return {
            "key": key,
            "name": info["name"],
            "ok": False,
            "date": None,
            "count": 0,
            "error": str(e)[:50],
        }


async def check_finmind(client: httpx.AsyncClient, key: str, info: dict) -> dict:
    """檢查 FinMind API"""
    try:
        params = {
            "dataset": info["dataset"],
            "start_date": TARGET_DATE_STR,
            "end_date": TARGET_DATE_STR,
        }
        if info.get("data_id"):
            params["data_id"] = info["data_id"]

        resp = await client.get(
            "https://api.finmindtrade.com/api/v4/data",
            params=params,
            timeout=30,
        )
        if resp.status_code != 200:
            return {
                "key": key,
                "name": info["name"],
                "ok": False,
                "date": None,
                "count": 0,
                "error": f"HTTP {resp.status_code}",
            }

        result = resp.json()
        if result.get("status") != 200:
            return {
                "key": key,
                "name": info["name"],
                "ok": False,
                "date": None,
                "count": 0,
                "error": result.get("msg", "Unknown")[:30],
            }

        data = result.get("data", [])
        if not data:
            return {
                "key": key,
                "name": info["name"],
                "ok": False,
                "date": None,
                "count": 0,
                "error": "無資料",
            }

        # 檢查是否有目標日期的資料
        dates = [str(row.get("date", ""))[:10] for row in data if row.get("date")]
        has_target = TARGET_DATE_STR in dates
        latest = max(dates) if dates else None

        return {
            "key": key,
            "name": info["name"],
            "ok": has_target,
            "date": latest,
            "count": len(data),
            "error": None,
        }

    except Exception as e:
        return {
            "key": key,
            "name": info["name"],
            "ok": False,
            "date": None,
            "count": 0,
            "error": str(e)[:50],
        }


def check_yfinance(key: str, info: dict) -> dict:
    """檢查 yfinance"""
    try:
        import yfinance as yf

        ticker = yf.Ticker(f"{info['stock_id']}.TW")
        hist = ticker.history(
            start=TARGET_DATE_STR,
            end=(TARGET_DATE + timedelta(days=1)).isoformat(),
        )

        if hist.empty:
            return {
                "key": key,
                "name": info["name"],
                "ok": False,
                "date": None,
                "count": 0,
                "error": "無資料",
            }

        latest = hist.index[-1].strftime("%Y-%m-%d")
        ok = latest == TARGET_DATE_STR

        return {
            "key": key,
            "name": info["name"],
            "ok": ok,
            "date": latest,
            "count": len(hist),
            "error": None,
        }

    except Exception as e:
        return {
            "key": key,
            "name": info["name"],
            "ok": False,
            "date": None,
            "count": 0,
            "error": str(e)[:50],
        }


async def main():
    parse_date_arg()

    print("=" * 70)
    print("開盤前資料完整度檢查")
    print(f"檢查日期: {TARGET_DATE_STR} ({TARGET_DATE.strftime('%A')})")
    print(f"檢查時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    results = []

    async with httpx.AsyncClient() as client:
        for key, info in DATASETS_TO_CHECK.items():
            print(f"檢查 {info['name']}...", end=" ", flush=True)

            if info["source"] == "twse_openapi":
                result = await check_twse_openapi(client, key, info)
                await asyncio.sleep(2)  # TWSE 限流
            elif info["source"] == "finmind":
                result = await check_finmind(client, key, info)
                await asyncio.sleep(1)
            elif info["source"] == "yfinance":
                result = check_yfinance(key, info)

            result["critical"] = info.get("critical", False)
            result["source"] = info["source"]
            results.append(result)

            status = "✅" if result["ok"] else "❌"
            print(status)

    # 結果摘要
    print(f"\n{'=' * 70}")
    print(f"{'資料':<25} {'來源':<12} {'狀態':<5} {'日期':<12} {'筆數':<8} {'備註'}")
    print(f"{'=' * 70}")

    for r in results:
        status = "✅" if r["ok"] else "❌"
        date_str = r["date"] or "N/A"
        count = r["count"]
        note = r["error"] or ""
        critical = "⚠️" if r["critical"] and not r["ok"] else ""
        source = r["source"].replace("twse_", "").upper()
        print(f"{r['name']:<25} {source:<12} {status:<5} {date_str:<12} {count:<8} {critical}{note}")

    # 統計
    print(f"\n{'=' * 70}")
    ok_count = sum(1 for r in results if r["ok"])
    critical_ok = sum(1 for r in results if r["critical"] and r["ok"])
    critical_total = sum(1 for r in results if r["critical"])

    print(f"總計: {ok_count}/{len(results)} 資料完整")
    print(f"關鍵資料: {critical_ok}/{critical_total}")

    # 判斷是否可以開始交易
    all_critical_ok = all(r["ok"] for r in results if r["critical"])

    print(f"\n{'=' * 70}")
    if all_critical_ok:
        print("✅ 關鍵資料完整，可以進行交易")
    else:
        print("❌ 關鍵資料不完整，請稍後再試")
        missing = [r["name"] for r in results if r["critical"] and not r["ok"]]
        print(f"   缺少: {', '.join(missing)}")

    return 0 if all_critical_ok else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
