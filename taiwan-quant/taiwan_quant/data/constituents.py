"""
0050 / 0051 成分股抓取

依據 D2：標的池 = 元大台灣 50（0050，50 檔）+ 元大中型 100（0051，100 檔）
= 150 檔。

資料來源：元大投信官網的持股比重頁（權威來源，基金公司自己公布）

    https://www.yuantaetfs.com/product/detail/0050/ratio
    https://www.yuantaetfs.com/product/detail/0051/ratio

為什麼不用其他來源（實測結論）：

    FinMind  `TaiwanStockMarketValueWeight` / `TaiwanStockMarketValue`
             → 付費方案才有（免費版回 400「Your level is free」）
    TWSE OpenAPI
             → 143 個端點中只有指數行情（TAI50I），**沒有成分股清單**
    TWSE ETFReport
             → 只有定期定額交易戶數排行，非成分股

技術細節：頁面是 Nuxt SSR，成分股資料內嵌在 `window.__NUXT__` 的
minified payload 裡，欄位值以變數引用表示：

    StockWeights:[{code:hl, ym:a, name:hm, ename:hn, weights:57.01, qty:560288622}, ...]

其中 `hl` / `hm` 是 IIFE 的參數名，真值在呼叫引數列。本模組重建這張
變數表再代入，因此**不需要瀏覽器**，curl 即可。

已知脆弱性：這個解析依賴 Nuxt 的 payload 格式。官網改版就會壞。
因此 `fetch_constituents()` 會驗證結果筆數（0050 應為 50、0051 應為 100），
解析出來不對就明確報錯，不會靜默回傳半套資料。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date

import httpx

YUANTA_URL = "https://www.yuantaetfs.com/product/detail/{etf_id}/ratio"

EXPECTED_COUNT: dict[str, int] = {
    "0050": 50,
    "0051": 100,
}
"""各 ETF 的預期成分股數。實際值偏離就報錯，不接受半套資料"""

COUNT_TOLERANCE = 5
"""容許的檔數偏差。基金會有少量現金部位或調整期間的差異"""

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)

_LISTED_STOCK_CODE = re.compile(r"^[1-9]\d{3}$")
"""
台股上市個股代號：4 位數且不以 0 開頭。

以 0 開頭的 4 位數代號是 ETF（0050、0056…）。持股清單理論上不含 ETF，
但與 `market_cap.py` 用同一條規則，避免兩邊行為不一致。
"""


class ConstituentFetchError(RuntimeError):
    """成分股抓取或解析失敗"""


@dataclass(frozen=True)
class Constituent:
    """單一成分股"""

    stock_id: str
    name: str
    weight: float
    """基金權重（%）"""

    shares: int
    """基金持有股數"""


@dataclass(frozen=True)
class ConstituentSnapshot:
    """某 ETF 在某交易日的成分股快照"""

    etf_id: str
    as_of: date
    constituents: tuple[Constituent, ...]

    @property
    def stock_ids(self) -> list[str]:
        return [c.stock_id for c in self.constituents]

    @property
    def total_weight(self) -> float:
        return sum(c.weight for c in self.constituents)


# ══════════════════════════════════════════════════════════════
# Nuxt payload 解析
# ══════════════════════════════════════════════════════════════

_IIFE_HEAD = re.compile(r"window\.__NUXT__\s*=\s*\(function\(([^)]{1,40000})\)\{")
"""
不對參數列長度設下限。真實頁面有 929 個參數，但測試用的最小 payload
只有幾個——防線應該是後面的筆數驗證，不是這裡的長度啟發式。
"""

_STOCK_WEIGHT_ENTRY = re.compile(
    r"\{code:(?P<code>[A-Za-z_$][\w$]*|\"[^\"]*\"),"
    r"ym:[^,]*,"
    r"name:(?P<name>[A-Za-z_$][\w$]*|\"[^\"]*\"),"
    r"ename:[^,]*,"
    r"weights:(?P<weights>-?[\d.]+|[A-Za-z_$][\w$]*),"
    r"qty:(?P<qty>-?\d+|[A-Za-z_$][\w$]*)\}"
)

_TRADE_DATE = re.compile(r"(?:tradeDate|TradeDate|datadate|DataDate):\"?(\d{4})[/-](\d{2})[/-](\d{2})")


def _split_args(raw: str) -> list[str]:
    """
    切分 IIFE 的呼叫引數列，尊重字串引號與轉義。

    不能直接 `split(",")`，因為字串值本身含逗號（例如公司英文名）。
    """
    args: list[str] = []
    buf: list[str] = []
    in_string = False
    escaped = False

    for ch in raw:
        if escaped:
            buf.append(ch)
            escaped = False
            continue
        if ch == "\\":
            buf.append(ch)
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            buf.append(ch)
            continue
        if ch == "," and not in_string:
            args.append("".join(buf).strip())
            buf = []
            continue
        buf.append(ch)

    args.append("".join(buf).strip())
    return args


def _build_variable_map(html: str) -> dict[str, object]:
    """重建 Nuxt IIFE 的「參數名 → 實際值」對照表"""
    head = _IIFE_HEAD.search(html)
    if not head:
        raise ConstituentFetchError("找不到 window.__NUXT__ IIFE，官網 payload 格式可能已變更")

    names = [n.strip() for n in head.group(1).split(",")]

    call_start = html.rfind("}(", head.end())
    if call_start < 0:
        raise ConstituentFetchError("找不到 IIFE 呼叫引數列")

    close = html.find("));", call_start)
    if close < 0:
        close = html.find(")）", call_start)
    if close < 0:
        raise ConstituentFetchError("找不到 IIFE 引數列結尾")

    raw_args = html[call_start + 2 : close]
    values: dict[str, object] = {}

    for name, token in zip(names, _split_args(raw_args)):
        values[name] = _parse_token(token)

    return values


_JS_NUMBER = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$")
"""
JS 數字字面值，**含前導點小數**（`.69`）與尾點（`3.`）。

JSON 規範不允許這兩種寫法，`json.loads(".69")` 會拋 JSONDecodeError。
元大官網的權重就是這種寫法（`weights:.69`），漏掉會讓權重被當成 0.0。
"""


def _parse_token(token: str) -> object:
    """把單一引數 token 轉成 Python 值"""
    if token.startswith('"') and token.endswith('"'):
        try:
            return json.loads(token)
        except json.JSONDecodeError:
            return token.strip('"')
    if token in ("null", "void 0", "undefined"):
        return None
    if token == "true":
        return True
    if token == "false":
        return False
    if _JS_NUMBER.match(token):
        # 先走 float 才能吃下 JSON 不合法的 `.69` / `3.`
        return float(token)
    try:
        return json.loads(token)
    except json.JSONDecodeError:
        return token


def _resolve(token: str, variables: dict[str, object]) -> object:
    """把可能是變數名的 token 解析成實際值"""
    if token.startswith('"'):
        return _parse_token(token)
    if token in variables:
        return variables[token]
    return _parse_token(token)


def parse_constituents(html: str, etf_id: str) -> ConstituentSnapshot:
    """
    從元大投信頁面 HTML 解析成分股。

    Args:
        html: 頁面原始 HTML
        etf_id: ETF 代號（0050 / 0051），只用於標記與筆數驗證

    Raises:
        ConstituentFetchError: 解析不到資料，或筆數與預期偏離過大
    """
    variables = _build_variable_map(html)

    entries = list(_STOCK_WEIGHT_ENTRY.finditer(html))
    if not entries:
        raise ConstituentFetchError(
            f"{etf_id}：解析不到 StockWeights 項目，官網 payload 格式可能已變更"
        )

    seen: set[str] = set()
    constituents: list[Constituent] = []

    for match in entries:
        code = _resolve(match.group("code"), variables)
        name = _resolve(match.group("name"), variables)
        weight = _resolve(match.group("weights"), variables)
        qty = _resolve(match.group("qty"), variables)

        if not isinstance(code, str) or not code.strip():
            continue
        code = code.strip()
        # 只收上市個股代號（4 位數、不以 0 開頭），
        # 排除期貨（TX、NYF）、ETF（00xx）與其他商品
        if not _LISTED_STOCK_CODE.match(code):
            continue
        if code in seen:
            continue
        seen.add(code)

        constituents.append(
            Constituent(
                stock_id=code,
                name=str(name).strip() if name else "",
                weight=float(weight) if isinstance(weight, (int, float)) else 0.0,
                shares=int(qty) if isinstance(qty, (int, float)) else 0,
            )
        )

    expected = EXPECTED_COUNT.get(etf_id)
    if expected is not None and abs(len(constituents) - expected) > COUNT_TOLERANCE:
        raise ConstituentFetchError(
            f"{etf_id}：解析出 {len(constituents)} 檔，與預期 {expected} 檔偏離超過 "
            f"{COUNT_TOLERANCE}。拒絕回傳半套資料——請檢查官網是否改版。"
        )

    as_of = _parse_trade_date(html) or date.today()
    constituents.sort(key=lambda c: c.weight, reverse=True)

    return ConstituentSnapshot(
        etf_id=etf_id,
        as_of=as_of,
        constituents=tuple(constituents),
    )


def _parse_trade_date(html: str) -> date | None:
    """從頁面抓資料日期"""
    match = _TRADE_DATE.search(html)
    if not match:
        return None
    try:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    except ValueError:
        return None


# ══════════════════════════════════════════════════════════════
# 抓取
# ══════════════════════════════════════════════════════════════


def fetch_constituents(etf_id: str, timeout: float = 60.0) -> ConstituentSnapshot:
    """
    抓取單一 ETF 的當期成分股。

    Args:
        etf_id: "0050" 或 "0051"
        timeout: HTTP 超時（秒）

    Raises:
        ConstituentFetchError: HTTP 失敗或解析失敗
    """
    url = YUANTA_URL.format(etf_id=etf_id)
    try:
        response = httpx.get(
            url,
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise ConstituentFetchError(f"{etf_id}：抓取 {url} 失敗：{exc}") from exc

    return parse_constituents(response.text, etf_id)


def fetch_universe_constituents(
    etf_ids: tuple[str, ...] = ("0050", "0051"),
) -> dict[str, ConstituentSnapshot]:
    """
    抓取整個標的池的成分股（D2：0050 + 0051 = 150 檔）。

    Returns:
        {etf_id: 快照}

    0050 與 0051 依指數設計不重疊（0051 是市值排名 51~150），
    但本函式不強制這個假設——重疊檢查由呼叫端的 universe 組裝邏輯負責。
    """
    return {etf_id: fetch_constituents(etf_id) for etf_id in etf_ids}
