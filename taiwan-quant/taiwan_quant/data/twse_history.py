"""
TWSE 歷史日報解析

## 為什麼用 TWSE 而不是 yfinance

兩者都能取得 2015 年起的台股日線，但差別是決定性的：

| | TWSE 日報 | yfinance |
|---|---|---|
| 價格基準 | **原始成交價** | **分割還原價** |
| 涵蓋範圍 | 當天**所有掛牌證券** | 需要先知道代號 |
| 下市股票 | 有（當年就在名單上） | 通常查不到 |
| ETF | 有（0050 等） | 有 |

### 價格基準不同，不可混接

實測 2023~2026 重疊期比對（8 檔）：

    7 檔   OHLC 相對誤差中位數 0.000000   完全吻合
    2881   OHLC 四欄都差 2.4390%          系統性偏差

2.4390% = 1 − 1/1.025。查證：富邦金 2023-09-04、2024-09-09、2025-09-25
各有一次股票股利，Yahoo 記為 split。**yfinance 的 OHLC 是分割還原價，
TWSE 是原始成交價。** 在 2023-01 接起來，會讓有配股的個股在接點產生
假跳空。

### 下市股票是 survivorship 的關鍵

MI_INDEX 列的是「當天實際有掛牌交易的證券」。2015-01-05 有 911 檔，
其中不少今天已經下市。用今天的名單回溯歷史會把它們全部漏掉——
那正是 survivorship bias。

## 三個端點

```
MI_INDEX    每日收盤行情（全部）   OHLCV + 成交金額，含 ETF
T86         三大法人買賣超         個股別
MI_MARGN    融資融券彙總           個股別
```

都以 `date=YYYYMMDD` 取單日，非交易日回 `stat` 錯誤訊息。

## 解析失敗一律拋錯，不回空清單

回空清單會讓 backfill 把那天標記成「已完成、沒有資料」，之後不再重試，
資料就永久缺一天且無人察覺。這與 CLAUDE.md 的「無法計算與算出來很差
必須區分」是同一條原則。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ── 代號規則 ──────────────────────────────────────────────

ORDINARY_CODE = re.compile(r"^(?:\d{4}|00\d{3,4})$")
"""
一般證券：**4 位純數字的個股／舊 ETF，或 00 開頭的 5~6 位純數字 ETF**。

⚠️ **2026-09-19 修正：原本只收 4 位，漏掉了所有現代 ETF。**

`^\\d{4}$` 讓 0050／0051／0056（恰好 4 碼）有資料，而 00712、00878、
006208 完全不在庫裡。2026-09-17 得出「ETF 輪動沒幫助」時手上只有那三檔，
那個結論的標的範圍比它看起來的窄得多。

### 為什麼 4 碼規則原本存在，以及為什麼它是多餘的

原意是排除權證（6 位如 031234）。但價格端點用
`type=ALLBUT0999`，表格標題明寫「不含權證、牛熊證、可展延牛熊證」——
**權證在來源就被排除了**，4 碼規則是第二道牆，而它實際擋掉的是 ETF。

T86（籌碼）端點沒有那個排除，一天 6,880 個代號絕大多數是權證，
所以那裡仍然需要代號過濾——而權證是 6 位**含英文或非 00 開頭**，
本規則照樣擋得住。

### 繼續排除的類別

```
00631L / 00633L   槓桿型，每日重設，複利路徑與持有期假設不相容
00632R / 00634R   反向型，同上
00635U            期貨信託
00625K / 00636K   特殊類別
2887A / 2891B     特別股（4 碼 + 字母）
```

**字母後綴一律排除**，所以規則只放行純數字。

⚠️ **存進資料庫 ≠ 可以交易。** 交易候選的白名單是
`data/etf_universe.is_etf`，仍然只有 0050／0051／0056；本規則放寬不會
自動把新 ETF 放進候選池（`test_twse_is_etf_follows_the_code_rule_not_the_trading_allowlist`
釘住這件事）。
"""

ETF_PREFIX = "00"
"""台股 ETF 代號以 00 開頭（0050、0056、0051 ...）"""

NO_TRADE_MARKERS = frozenset({"--", "---", "", "n/a", "na"})
"""當天沒有成交的標記。**不是 0**——當成 0 會讓報酬率算出 −100%"""

ROC_DATE = re.compile(r"(\d{2,3})年(\d{1,2})月(\d{1,2})日")

QUOTE_TABLE_KEYWORD = "每日收盤行情"
MARGIN_TABLE_KEYWORD = "融資融券"

T86_ALIASES: dict[str, tuple[str, ...]] = {
    # 2018 起「外資」被拆成「外陸資（不含外資自營商）」與「外資自營商」。
    # 取窄定義以與既有 2023~2026 的資料一致（實測：既有 foreign_buy
    # 等於 外陸資買進，不含外資自營商）。
    "foreign_buy": ("外陸資買進股數(不含外資自營商)", "外資買進股數"),
    "foreign_sell": ("外陸資賣出股數(不含外資自營商)", "外資賣出股數"),
    "trust_buy": ("投信買進股數",),
    "trust_sell": ("投信賣出股數",),
    "dealer_self_buy": ("自營商買進股數(自行買賣)",),
    "dealer_self_sell": ("自營商賣出股數(自行買賣)",),
    "dealer_hedge_buy": ("自營商買進股數(避險)",),
    "dealer_hedge_sell": ("自營商賣出股數(避險)",),
}
"""
T86 欄位名稱對照。

**必須依名稱取值，不可用位置索引。** 實測：T86 在 2015 年是 16 欄、
2018 年起變成 19 欄（外資被拆成兩組）。用 2015 的索引去解析 2024 的
回應，外資對、投信與自營商全錯——97 檔裡只有 1 檔吻合，而且不會拋錯。
"""

MI_MARGN_EXPECTED_FIELDS = (
    "代號", "名稱",
    "買進", "賣出", "現金償還", "前日餘額", "今日餘額", "次一營業日限額",
    "買進", "賣出", "現券償還", "前日餘額", "今日餘額", "次一營業日限額",
    "資券互抵", "註記",
)
"""
MI_MARGN 的欄位名稱**有重複**（`買進` 同時出現在融資與融券兩區塊），
所以只能用位置索引。改用名稱查會取到錯的區塊。

代價是結構一變就會錯，所以解析前先比對這張表；不符就拋錯。
實測 2015 與 2024 的欄位完全相同。
"""


class TwseParseError(RuntimeError):
    """TWSE 回應無法解析"""


# ══════════════════════════════════════════════════════════════
# 基礎解析
# ══════════════════════════════════════════════════════════════


def parse_number(text: object) -> float | None:
    """
    把 TWSE 的字串數值轉成 float。

    Returns:
        數值；無成交標記或無法解析時回 `None`（**不是 0**）

    處理千分位逗號、前導正負號、全形空白。HTML 片段
    （漲跌欄的 `<p style=...>-</p>`）視為無法解析。
    """
    if text is None:
        return None
    cleaned = str(text).strip().replace(",", "").replace("　", "")
    if cleaned.lower() in NO_TRADE_MARKERS:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_int(text: object) -> int | None:
    """整數欄位；小數會被截斷（TWSE 的股數欄位本來就是整數）"""
    value = parse_number(text)
    return None if value is None else int(value)


def is_ordinary_security(code: str) -> bool:
    """是否為一般證券（4 位純數字），排除權證與槓桿型"""
    return bool(ORDINARY_CODE.match(code.strip()))


def is_etf(code: str) -> bool:
    """是否為 ETF"""
    stripped = code.strip()
    return is_ordinary_security(stripped) and stripped.startswith(ETF_PREFIX)


def roc_date_to_iso(text: str) -> str | None:
    """
    民國日期轉 ISO。

    `104年01月05日` → `2015-01-05`

    找不到民國格式回 `None`——呼叫端要自己決定用請求日期補。
    """
    match = ROC_DATE.search(text or "")
    if match is None:
        return None
    year, month, day = (int(g) for g in match.groups())
    return f"{year + 1911:04d}-{month:02d}-{day:02d}"


def _require_ok(payload: dict, endpoint: str) -> None:
    stat = payload.get("stat")
    if stat != "OK":
        raise TwseParseError(f"{endpoint} 的 stat 不是 OK：{stat!r}")


def _find_table(payload: dict, keyword: str, endpoint: str) -> dict:
    for table in payload.get("tables") or []:
        if keyword in (table.get("title") or ""):
            return table
    raise TwseParseError(f"{endpoint} 找不到標題含「{keyword}」的表格")


# ══════════════════════════════════════════════════════════════
# MI_INDEX（每日收盤行情）
# ══════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class DailyQuote:
    """單日單檔的行情"""

    stock_id: str
    name: str
    date: str
    """ISO 日期"""

    open: float
    high: float
    low: float
    close: float
    volume: int
    """成交股數"""

    turnover: float
    """成交金額（元）。市值資料缺席時，用它當流動性排名依據"""

    @property
    def is_etf(self) -> bool:
        return is_etf(self.stock_id)


def parse_mi_index(payload: dict, trade_date: str | None = None) -> list[DailyQuote]:
    """
    解析 MI_INDEX 的每日收盤行情表。

    Args:
        payload: TWSE 回應
        trade_date: ISO 日期；None 時從表格標題的民國日期取

    Returns:
        當天所有**有成交**的一般證券（含 ETF）

    Raises:
        TwseParseError: stat 非 OK、找不到行情表、或日期無法判定

    OHLC 任一為 `--` 的列會被剔除——那代表當天沒有成交，補 0 會讓
    報酬率算出 −100%。
    """
    _require_ok(payload, "MI_INDEX")
    table = _find_table(payload, QUOTE_TABLE_KEYWORD, "MI_INDEX")

    date_iso = trade_date or roc_date_to_iso(table.get("title") or "")
    if date_iso is None:
        raise TwseParseError("MI_INDEX 無法判定交易日期（標題沒有民國日期）")

    quotes: list[DailyQuote] = []
    for row in table.get("data") or []:
        if len(row) < 9:
            continue
        code = str(row[0]).strip()
        if not is_ordinary_security(code):
            continue

        prices = [parse_number(row[i]) for i in (5, 6, 7, 8)]
        if any(p is None or p <= 0 for p in prices):
            # 當天無成交（`--`）或價格異常 → 剔除，不補值
            continue

        volume = parse_int(row[2])
        turnover = parse_number(row[4])
        if volume is None or turnover is None:
            continue

        open_, high, low, close = prices  # type: ignore[misc]
        quotes.append(DailyQuote(
            stock_id=code,
            name=str(row[1]).strip(),
            date=date_iso,
            open=open_,
            high=high,
            low=low,
            close=close,
            volume=volume,
            turnover=turnover,
        ))

    return quotes


# ══════════════════════════════════════════════════════════════
# T86（三大法人買賣超）
# ══════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class InstitutionalRow:
    """單日單檔的三大法人買賣（股數）"""

    stock_id: str
    date: str
    foreign_buy: int
    foreign_sell: int
    trust_buy: int
    trust_sell: int
    dealer_buy: int
    dealer_sell: int
    """自營商買賣已合併自行買賣與避險兩部位，與既有 schema 一致"""


def _t86_indices(fields: list[str]) -> dict[str, int]:
    """
    建立「語意名稱 → 欄位索引」對照。

    Raises:
        TwseParseError: 任一必要欄位在兩種命名下都找不到
    """
    position = {name.strip(): i for i, name in enumerate(fields)}
    mapping: dict[str, int] = {}
    for key, aliases in T86_ALIASES.items():
        index = next((position[a] for a in aliases if a in position), None)
        if index is None:
            raise TwseParseError(
                f"T86 找不到欄位 {key}（試過 {aliases}）。"
                f"實際欄位：{fields}"
            )
        mapping[key] = index
    return mapping


def parse_t86(payload: dict, trade_date: str) -> list[InstitutionalRow]:
    """
    解析 T86 三大法人買賣超日報。

    Args:
        payload: TWSE 回應
        trade_date: ISO 日期（T86 回應本身不帶日期）

    Returns:
        當天所有一般證券的買賣股數

    Raises:
        TwseParseError: stat 非 OK、缺 fields、或必要欄位找不到

    兩個必須注意的地方：

    1. **依欄位名稱取值。** T86 在 2015 年是 16 欄、2018 年起 19 欄。
       用固定索引會靜默取到錯的欄位（實測 97 檔只有 1 檔吻合）。

    2. **自營商要合併自行買賣與避險。** 既有 schema 的
       `dealer_buy` / `dealer_sell` 是合併值，只取一邊會少算一半以上。
    """
    _require_ok(payload, "T86")

    fields = payload.get("fields")
    if not fields:
        raise TwseParseError("T86 回應缺少 fields，無法依名稱解析")
    index = _t86_indices(list(fields))
    needed = max(index.values())

    rows: list[InstitutionalRow] = []
    for row in payload.get("data") or []:
        if len(row) <= needed:
            continue
        code = str(row[0]).strip()
        if not is_ordinary_security(code):
            continue

        values = {key: parse_int(row[i]) for key, i in index.items()}
        if any(v is None for v in values.values()):
            continue

        rows.append(InstitutionalRow(
            stock_id=code,
            date=trade_date,
            foreign_buy=values["foreign_buy"],
            foreign_sell=values["foreign_sell"],
            trust_buy=values["trust_buy"],
            trust_sell=values["trust_sell"],
            dealer_buy=values["dealer_self_buy"] + values["dealer_hedge_buy"],
            dealer_sell=values["dealer_self_sell"] + values["dealer_hedge_sell"],
        ))

    return rows


# ══════════════════════════════════════════════════════════════
# MI_MARGN（融資融券）
# ══════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class MarginRow:
    """單日單檔的融資融券（交易單位＝張）"""

    stock_id: str
    date: str
    margin_buy: int
    margin_sell: int
    margin_balance: int
    short_buy: int
    short_sell: int
    short_balance: int


def parse_mi_margn(payload: dict, trade_date: str) -> list[MarginRow]:
    """
    解析 MI_MARGN 融資融券彙總表。

    Args:
        payload: TWSE 回應
        trade_date: ISO 日期

    Returns:
        當天所有一般證券的融資融券

    Raises:
        TwseParseError: stat 非 OK、找不到彙總表

    餘額 0 的列**要保留**——那是真實資訊（沒人融資），
    與 MI_INDEX 的 `--`（沒有成交）不同。
    """
    _require_ok(payload, "MI_MARGN")
    table = _find_table(payload, MARGIN_TABLE_KEYWORD, "MI_MARGN")

    fields = tuple(str(f).strip() for f in (table.get("fields") or []))
    if fields and fields != MI_MARGN_EXPECTED_FIELDS:
        raise TwseParseError(
            "MI_MARGN 欄位結構與預期不符，位置索引會取到錯的值。\n"
            f"  預期：{MI_MARGN_EXPECTED_FIELDS}\n"
            f"  實際：{fields}"
        )

    rows: list[MarginRow] = []
    for row in table.get("data") or []:
        if len(row) < 14:
            continue
        code = str(row[0]).strip()
        if not is_ordinary_security(code):
            continue

        values = [parse_int(row[i]) for i in (2, 3, 6, 8, 9, 12)]
        if any(v is None for v in values):
            continue
        m_buy, m_sell, m_balance, s_buy, s_sell, s_balance = values  # type: ignore[misc]

        rows.append(MarginRow(
            stock_id=code,
            date=trade_date,
            margin_buy=m_buy,
            margin_sell=m_sell,
            margin_balance=m_balance,
            short_buy=s_buy,
            short_sell=s_sell,
            short_balance=s_balance,
        ))

    return rows
