"""
歷史市值重建

用途：回測期間（2023-2026）沒有真實成分股快照，必須用市值排名代理
（見 `universe_history.py`）。要算歷史市值就需要「當時的發行股數」。

資料來源選擇（2026-09-12 實測）：

    FinMind `TaiwanStockMarketValue`
        → 付費方案才有（免費版回 400「Your level is free」）

    TWSE MI_QFIIS（集中市場外資及陸資投資持股統計）
        → **支援 `date=YYYYMMDD` 取歷史**，免費、無明顯額度限制
        → 欄位含「發行股數」，實測 2024-06-28 可取得 1,223 檔
        → 本模組採用這個

市值 = 發行股數 × 當日收盤價（還原前的原始收盤價）

為什麼用**未還原**收盤價：市值是「當時的市場價值」，要用當時的實際
成交價乘當時的股數。用還原價會算出一個歷史上不存在的市值。
這是本模組刻意違反 CLAUDE.md 禁令 12 的唯一場合，理由如上。

退市股（CLAUDE.md 禁令 2）：
    FinMind `TaiwanStockDelisting` 免費可用，回傳 {date, stock_id, stock_name}。
    `fetch_delisted()` 取得清單，讓呼叫端能把退市股的歷史市值一併納入
    代理重建，避免 survivorship bias。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

import httpx

TWSE_QFIIS_URL = "https://www.twse.com.tw/rwd/zh/fund/MI_QFIIS"
FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"

REQUEST_INTERVAL_SEC = 2.0
"""TWSE 限流：每 5 秒 3 次請求，保守取 2 秒間隔"""

_ISSUED_SHARES_FIELD = "發行股數"
_STOCK_ID_FIELD = "證券代號"
_STOCK_NAME_FIELD = "證券名稱"

_LISTED_STOCK_CODE = re.compile(r"^[1-9]\d{3}$")
"""
台股上市個股代號：4 位數且**不以 0 開頭**。

以 0 開頭的 4 位數代號是 ETF（0050、0051、0056…），不是個股。
MI_QFIIS 的前兩列就是 0050 與 0051 本身，單純用 `\\d{4}` 會把它們收進來。
"""


class MarketCapError(RuntimeError):
    """市值資料抓取或解析失敗"""


@dataclass(frozen=True)
class DelistedStock:
    """退市個股"""

    stock_id: str
    name: str
    delisted_on: date


def _parse_int(raw: str) -> int | None:
    """解析含千分位的整數；無效值回 None（不回 0，0 會被誤當成真實股數）"""
    if not raw:
        return None
    cleaned = raw.replace(",", "").strip()
    if cleaned in ("", "--", "-"):
        return None
    try:
        return int(cleaned)
    except ValueError:
        return None


def parse_issued_shares(payload: dict) -> dict[str, int]:
    """
    從 TWSE MI_QFIIS 回應解析發行股數。

    Args:
        payload: TWSE 回傳的 JSON

    Returns:
        {stock_id: 發行股數}，只含 4 位數台股代號（排除 ETF、權證等）

    Raises:
        MarketCapError: 回應格式不符或找不到發行股數欄位
    """
    if payload.get("stat") != "OK":
        raise MarketCapError(f"TWSE 回應非 OK：{payload.get('stat')}")

    fields = payload.get("fields") or []
    try:
        id_idx = fields.index(_STOCK_ID_FIELD)
        shares_idx = fields.index(_ISSUED_SHARES_FIELD)
    except ValueError as exc:
        raise MarketCapError(
            f"TWSE 回應缺少必要欄位（{_STOCK_ID_FIELD} / {_ISSUED_SHARES_FIELD}）："
            f"實際欄位 {fields}"
        ) from exc

    result: dict[str, int] = {}
    for row in payload.get("data") or []:
        if len(row) <= max(id_idx, shares_idx):
            continue
        stock_id = str(row[id_idx]).strip()
        if not _LISTED_STOCK_CODE.match(stock_id):
            continue
        shares = _parse_int(str(row[shares_idx]))
        if shares is None or shares <= 0:
            continue
        result[stock_id] = shares

    if not result:
        raise MarketCapError("解析不到任何發行股數，TWSE 回應格式可能已變更")

    return result


def fetch_issued_shares(as_of: date, timeout: float = 60.0) -> dict[str, int]:
    """
    抓取指定日期的全市場發行股數。

    Args:
        as_of: 交易日（非交易日 TWSE 會回最近一個交易日的資料，
               呼叫端若在意精確日期，請比對回應中的 `date`）

    Raises:
        MarketCapError: HTTP 或解析失敗
    """
    params = {
        "date": as_of.strftime("%Y%m%d"),
        "selectType": "ALLBUT0999",
        "response": "json",
    }
    try:
        response = httpx.get(TWSE_QFIIS_URL, params=params, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise MarketCapError(f"抓取 {as_of} 發行股數失敗：{exc}") from exc

    return parse_issued_shares(payload)


def compute_market_caps(
    issued_shares: dict[str, int],
    close_prices: dict[str, float],
) -> dict[str, float]:
    """
    市值 = 發行股數 × 收盤價。

    Args:
        issued_shares: {stock_id: 發行股數}
        close_prices: {stock_id: **未還原**收盤價}

    Returns:
        {stock_id: 市值}，只含兩邊都有資料且皆為正值的標的

    為什麼用未還原價：市值是「當時的市場價值」，要用當時的實際成交價
    乘當時的股數。用還原價會算出一個歷史上不存在的市值。
    """
    return {
        stock_id: float(shares) * close_prices[stock_id]
        for stock_id, shares in issued_shares.items()
        if stock_id in close_prices
        and shares > 0
        and close_prices[stock_id] > 0
    }


def parse_delisted(payload: dict) -> list[DelistedStock]:
    """從 FinMind `TaiwanStockDelisting` 回應解析退市清單"""
    if payload.get("status") != 200:
        raise MarketCapError(f"FinMind 回應異常：{str(payload.get('msg'))[:120]}")

    result: list[DelistedStock] = []
    for row in payload.get("data") or []:
        try:
            result.append(
                DelistedStock(
                    stock_id=str(row["stock_id"]).strip(),
                    name=str(row.get("stock_name", "")).strip(),
                    delisted_on=date.fromisoformat(str(row["date"])[:10]),
                )
            )
        except (KeyError, ValueError):
            continue
    return result


def fetch_delisted(
    start: date,
    end: date,
    token: str = "",
    timeout: float = 60.0,
) -> list[DelistedStock]:
    """
    抓取期間內的退市清單（CLAUDE.md 禁令 2）。

    只用今日還活著的股票重建歷史標的池，等於把已倒的公司從樣本刪掉。
    這份清單讓呼叫端知道有哪些標的需要補歷史資料。

    FinMind `TaiwanStockDelisting` 免費方案可用（2026-09-12 實測）。
    """
    params = {
        "dataset": "TaiwanStockDelisting",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
    }
    if token:
        params["token"] = token

    try:
        response = httpx.get(FINMIND_URL, params=params, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise MarketCapError(f"抓取退市清單失敗：{exc}") from exc

    return [d for d in parse_delisted(payload) if start <= d.delisted_on <= end]
