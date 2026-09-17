"""
資料載入層

沿用 ../qlib-tw-trader 已同步好的 SQLite。**不重寫資料同步層**——
那部分已經驗證可用（TWSE + FinMind + yfinance 三源降級，實測 100 檔 ×
893 交易日 × 99.9% 覆蓋率），重寫沒有價值。

依據 ../docs/需求規劃/202609/評估_qlib-tw-trader.md 的「部分沿用」結論。

已知限制（必須在報告中揭露）：

1. **標的池是市值排名代理，不是真正的 0050+0051 成分股。**
   D2 要求 0050（50 家）+ 0051（100 家）= 150 檔，但上游只同步了
   「市值前 100」。目前以市值排名當代理，並在 `UniverseWarning` 中標明。

2. **沒有歷史成分股快照 → survivorship bias。**
   上游 `stock_universe` 只有一個 `updated_at`，用今日名單回溯歷史會
   高估績效（被剔除的弱股不在樣本內）。這是已記錄的缺陷，不是可忽略的。

3. **PER / 月營收 / 集保 / 借券資料不完整**（FinMind 免費額度限制）。
   相關特徵不可使用，或必須先檢查覆蓋率。
"""

from __future__ import annotations

import inspect
import json
import sqlite3
import warnings
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DB_PATH = Path(__file__).resolve().parents[3] / "qlib-tw-trader" / "data" / "data.db"

PRICE_COLUMNS = ("open", "high", "low", "close", "volume")

RAW_CLOSE_COLUMN = "raw_close"
RAW_OPEN_COLUMN = "raw_open"
"""
未還原的實際成交價，**一律存在**（`adjusted=False` 時等於 `close` / `open`）。

只保留 open 與 close 兩個，因為只有它們被用來做交易決策：

    進場  T+1 開盤   →  raw_open
    出場  T+H 收盤   →  raw_close

high / low 只進 ATR 與柵欄寬度，那些都是**比例**運算，用還原價才對。

## 為什麼要獨立一欄

兩種價格回答不同的問題：

    close      這一趟賺了多少      禁令 12 要求用還原價
    raw_close  買不買得起整張      成本分層要用實際成交價

`resolve_tier(price, amount)` 用 `amount >= price * LOT_SIZE` 判斷整股。
N=10 時 amount = 40,000，門檻是股價 40 元。傳還原價進去會算錯：

    年度   還原/實際平均   40 元門檻分層錯邊的比例
    2016      0.8666          13.41%
    2020      0.8715          10.89%
    2024      0.9259           3.46%

還原是回溯調整、錨在最新日，所以越早的日期還原價越低——**開發集
正是落差最大的那一段**。而錯的方向是「看起來買得起整張」，也就是
低估成本。

⚠️ 一律存在是刻意的：呼叫端不需要 `if adjusted` 分支，少一個分支就少
一個忘記處理的可能。"""

FROZEN_DATA_START = date(2026, 9, 14)
"""
凍結資料起點（含）。

載入此日之後的價格必須明確解鎖並留下理由。現有資料只到 2026-09-11，
所以不影響既有驗證；守門只約束未來新增、真正未被看過的資料。
"""


class DataNotAvailableError(RuntimeError):
    """上游資料庫缺少必要資料"""


@dataclass(frozen=True)
class PriceViews:
    """同一查詢範圍的還原價與實際價；分開保存，禁止混用同一欄位。"""

    adjusted: pd.DataFrame
    """報酬、特徵與標記使用的還原 OHLC。"""

    actual: pd.DataFrame
    """成交可負擔性與張／零股判定使用的未還原 OHLC。"""


@dataclass(frozen=True)
class UniverseWarning:
    """標的池的已知偏誤，必須隨報告一起輸出"""

    is_constituent_proxy: bool
    """True 表示用市值排名代理成分股，不是真正的 0050/0051 名單"""

    has_historical_snapshot: bool
    """False 表示沒有歷史成分股快照，存在 survivorship bias"""

    stock_count: int
    target_count: int
    """D2 要求的檔數（150）"""

    ranking_basis: str = "market_cap"
    """
    排名依據。

    `market_cap` 走上游 `stock_universe`；`turnover` 走回補的歷史快照
    （FinMind 免費層沒有股數，算不出市值，用成交金額當流動性代理）。
    """

    def describe(self) -> list[str]:
        """產出人可讀的警告清單，供報告直接引用"""
        notes: list[str] = []
        if self.is_constituent_proxy:
            notes.append(
                f"標的池為「市值前 {self.stock_count}」代理，"
                f"非真正 0050+0051 成分股（D2 要求 {self.target_count} 檔）"
            )
        if not self.has_historical_snapshot:
            notes.append(
                "無歷史成分股快照，以今日名單回溯歷史 → 存在 survivorship bias，"
                "回測績效會被高估"
            )
        if self.ranking_basis == "turnover":
            notes.append(
                "標的池依**成交金額**排名，不是市值。"
                "實測對今日真實 0050+0051 的命中率僅 72.7%——"
                "系統性漏掉大市值低週轉的傳產（中鋼、亞泥、和泰車）"
            )
        elif self.ranking_basis == "market_cap" and self.has_historical_snapshot:
            # 94.7% 是**季度快照**對今日真實名單的實測命中率，
            # 不適用於上游 `stock_universe` 的單一當期快照——
            # 那份沒有被驗證過，不可套用同一句宣稱。
            notes.append(
                "標的池依**市值**排名，命中今日真實 0050+0051 的 94.7%（142/150）。"
                "仍是代理——真實指數另有流動性門檻、產業分散與自由流通量調整"
            )
        if self.stock_count < self.target_count:
            notes.append(
                f"實際可用 {self.stock_count} 檔，少於 D2 要求的 {self.target_count} 檔"
            )
        return notes


@dataclass(frozen=True)
class Universe:
    """標的池快照"""

    stocks: pd.DataFrame
    """欄位：stock_id / name / market_cap / rank，依 rank 升冪"""

    warning: UniverseWarning

    @property
    def stock_ids(self) -> list[str]:
        return self.stocks["stock_id"].tolist()

    def tier_of(self, stock_id: str) -> str:
        """
        判斷流動性分層，供成本模型選滑價（config/costs.py 的 Tier）。

        代理規則：市值排名 <= 50 視為 0050 級，其餘視為 0051 級。
        真正的成分股名單到手後應改為查表。
        """
        row = self.stocks.loc[self.stocks["stock_id"] == stock_id]
        if row.empty:
            raise KeyError(f"{stock_id} 不在標的池內")
        return "0050" if int(row["rank"].iloc[0]) <= 50 else "0051"


def _connect(db_path: Path) -> sqlite3.Connection:
    if not db_path.exists():
        raise DataNotAvailableError(
            f"找不到上游資料庫：{db_path}\n"
            "請先在 qlib-tw-trader 執行資料同步（見 qlib-tw-trader/docs/QUICKSTART.zh-TW.md）"
        )
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)


def load_universe(
    db_path: Path = DEFAULT_DB_PATH,
    limit: int = 150,
    target_count: int = 150,
) -> Universe:
    """
    載入標的池。

    Args:
        db_path: 上游 SQLite 路徑
        limit: 取市值前幾名
        target_count: D2 要求的檔數，用於產生落差警告

    Returns:
        Universe（含已知偏誤警告）
    """
    con = _connect(db_path)
    try:
        stocks = pd.read_sql_query(
            "SELECT stock_id, name, market_cap, rank FROM stock_universe "
            "ORDER BY rank ASC LIMIT ?",
            con,
            params=(limit,),
        )
    finally:
        con.close()

    if stocks.empty:
        raise DataNotAvailableError(
            "stock_universe 表為空。請先執行 POST /api/v1/universe/sync"
        )

    return Universe(
        stocks=stocks,
        warning=UniverseWarning(
            is_constituent_proxy=True,
            has_historical_snapshot=False,
            stock_count=len(stocks),
            target_count=target_count,
        ),
    )


HISTORY_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "history.db"
"""
回補的長歷史資料庫（2015 起）。

schema 與上游 `data.db` 完全一致，但多了 `stock_universe_history`
與 `stock_master`（含已下市證券）。見
`scripts/backfill_finmind_history.py`。
"""


UNIVERSE_BASES = ("market_cap", "turnover")
"""
標的池的排名依據。

實測對今日真實 0050 + 0051 名單的命中率：

    market_cap   142 / 150 = 94.7%
    turnover     109 / 150 = 72.7%

成交金額系統性漏掉大市值低週轉的傳產（中鋼、亞泥、和泰車），
而 0050 / 0051 是市值加權指數。**預設用 market_cap。**

turnover 版保留，因為 `04` 與 `05` 的驗證結果都是用它跑的，
刪掉就無法重現。
"""

DEFAULT_UNIVERSE_BASIS = "market_cap"


def _has_table(con: sqlite3.Connection, name: str) -> bool:
    row = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


def load_universe_at(
    as_of: date,
    db_path: Path = HISTORY_DB_PATH,
    limit: int = 150,
    target_count: int = 150,
    basis: str = DEFAULT_UNIVERSE_BASIS,
) -> Universe:
    """
    載入**指定日期當時**的標的池。

    Args:
        as_of: 決策日
        db_path: 長歷史資料庫路徑
        limit: 取前幾名
        target_count: D2 要求的檔數，用於落差警告
        basis: 排名依據，見 `UNIVERSE_BASES`

    Returns:
        Universe（`has_historical_snapshot=True`）

    Raises:
        DataNotAvailableError: 資料庫不存在、basis 不認得、
            或 `as_of` 之前沒有任何快照

    **只取不晚於 `as_of` 的最新快照。** 取更晚的等於在決策日就知道
    未來的成分股名單（禁令 1）。早於第一個快照時拋錯，不可退回最早的
    那一份——那同樣是 look-ahead。

    這個函式存在的理由是禁令 2：2015 年的標的池裡有 34 檔後來下市的
    股票（含日月光、矽品），用今天的名單回溯會把它們全部漏掉。
    """
    if basis not in UNIVERSE_BASES:
        raise DataNotAvailableError(
            f"不認得的 basis {basis!r}，可用的是 {list(UNIVERSE_BASES)}"
        )

    con = _connect(db_path)
    try:
        # 新表帶 basis；舊環境只有 stock_universe_history（等同 turnover）
        if _has_table(con, "universe_history"):
            table, metric_col = "universe_history", "metric"
            where, params = "basis = ? AND as_of_date <= ?", (basis, as_of.isoformat())
            snapshot_sql = (
                f"SELECT MAX(as_of_date) FROM {table} WHERE {where}"
            )
        else:
            # 舊環境只有 stock_universe_history（等同 turnover）。
            # 退回它而不是拋錯——還沒跑過 build_universe_marketcap.py 的
            # 環境仍要能用。`ranking_basis` 會如實標示成 turnover，
            # 所以報告不會誤稱它是市值排名。
            table, metric_col = "stock_universe_history", "turnover"
            where, params = "as_of_date <= ?", (as_of.isoformat(),)
            snapshot_sql = f"SELECT MAX(as_of_date) FROM {table} WHERE {where}"

        snapshot = con.execute(snapshot_sql, params).fetchone()[0]
        if snapshot is None:
            earliest = con.execute(f"SELECT MIN(as_of_date) FROM {table}").fetchone()[0]
            raise DataNotAvailableError(
                f"{as_of} 之前沒有任何標的池快照（最早的快照是 {earliest}）。"
                "不可退回最早的那一份——那等於用未來的名單回測。"
            )

        basis_clause = "AND u.basis = ?" if table == "universe_history" else ""
        query_params: tuple = (
            (snapshot, basis, limit) if basis_clause else (snapshot, limit)
        )
        stocks = pd.read_sql_query(
            f"""
            SELECT u.stock_id,
                   COALESCE(m.name, '') AS name,
                   u.{metric_col} AS market_cap,
                   u.rank
            FROM {table} AS u
            LEFT JOIN stock_master AS m ON m.stock_id = u.stock_id
            WHERE u.as_of_date = ? {basis_clause}
            ORDER BY u.rank ASC
            LIMIT ?
            """,
            con,
            params=query_params,
        )
    finally:
        con.close()

    if stocks.empty:
        raise DataNotAvailableError(f"快照 {snapshot} 為空")

    return Universe(
        stocks=stocks,
        warning=UniverseWarning(
            is_constituent_proxy=True,
            has_historical_snapshot=True,
            stock_count=len(stocks),
            target_count=target_count,
            ranking_basis=basis if table == "universe_history" else "turnover",
        ),
    )


def load_prices(
    stock_ids: list[str] | None = None,
    start: date | None = None,
    end: date | None = None,
    db_path: Path = DEFAULT_DB_PATH,
    adjusted: bool = True,
    drop_incomplete: bool = True,
    unlock_frozen: bool = False,
    frozen_reason: str | None = None,
) -> pd.DataFrame:
    """
    載入日 K。

    Args:
        stock_ids: 要載入的股票代號；None 表示全部
        start / end: 日期範圍（含端點）
        db_path: 上游 SQLite 路徑
        adjusted: True 則用還原收盤價覆寫 close（CLAUDE.md 禁令 12）
        drop_incomplete: True 則剔除 OHLC 有缺漏或非正值的列
        unlock_frozen: 明確允許讀取凍結日起的資料；每次都會寫稽核 log
        frozen_reason: 解鎖理由，會與時間戳及呼叫端一起寫入 log

    關於 drop_incomplete：
        上游資料有洞（停牌、缺漏）。實測案例：2317 在 2025-07-30 有一列
        OHLC 全為 NULL。單一 NaN 會汙染 `np.quantile`，而 NaN 通過門檻
        比較時不會拋錯、而是讓 `NaN < 門檻` 為 False —— 靜默放行。
        預設剔除；要檢視被剔除了什麼請用 `price_quality_report()`。

    Returns:
        MultiIndex (stock_id, date) 的 DataFrame，
        欄位 open / high / low / close / volume。

    關於還原股價（禁令 12）：
        上游 `stock_daily` 是**未還原**價，`stock_daily_adj` 存還原收盤價。
        `adjusted=True` 時用 adj_close 覆寫 close，並以同一比例調整 OHL，
        維持 K 線形狀一致。除權息日不調整會讓報酬出現假跳空。
    """
    con = _connect(db_path)
    try:
        _guard_frozen_access(
            con,
            stock_ids,
            end,
            db_path,
            unlock_frozen,
            frozen_reason,
            access_kind="prices",
        )

        where: list[str] = []
        params: list[object] = []

        if stock_ids:
            placeholders = ",".join("?" * len(stock_ids))
            where.append(f"d.stock_id IN ({placeholders})")
            params.extend(stock_ids)
        if start:
            where.append("d.date >= ?")
            params.append(start.isoformat())
        if end:
            where.append("d.date <= ?")
            params.append(end.isoformat())

        clause = f"WHERE {' AND '.join(where)}" if where else ""
        sql = f"""
            SELECT d.stock_id, d.date, d.open, d.high, d.low, d.close, d.volume,
                   a.adj_close
            FROM stock_daily AS d
            LEFT JOIN stock_daily_adj AS a
                   ON a.stock_id = d.stock_id AND a.date = d.date
            {clause}
            ORDER BY d.stock_id ASC, d.date ASC
        """
        df = pd.read_sql_query(sql, con, params=params)
    finally:
        con.close()

    if df.empty:
        raise DataNotAvailableError("查無日 K 資料，請確認標的與日期範圍")

    df["date"] = pd.to_datetime(df["date"])

    # 實際成交價要在 _apply_adjustment 覆寫 close 之前留下來。
    # 之後再從還原價乘回因子還原不了——因子本身就是缺值補出來的。
    df[RAW_CLOSE_COLUMN] = df["close"].astype("float64")
    df[RAW_OPEN_COLUMN] = df["open"].astype("float64")

    if adjusted:
        df = _apply_adjustment(df)

    df = df.drop(columns=["adj_close"])

    if drop_incomplete:
        df = df[_is_complete(df)]
        if df.empty:
            raise DataNotAvailableError("剔除缺漏列後無資料可用")

    return df.set_index(["stock_id", "date"]).sort_index()


def load_price_views(
    stock_ids: list[str] | None = None,
    start: date | None = None,
    end: date | None = None,
    db_path: Path = DEFAULT_DB_PATH,
    drop_incomplete: bool = True,
    unlock_frozen: bool = False,
    frozen_reason: str | None = None,
) -> PriceViews:
    """已棄用的相容介面；由單一價格框架衍生實際開收盤視圖。"""
    warnings.warn(
        "load_price_views 已棄用；請使用 load_prices 的 raw_open / raw_close",
        DeprecationWarning,
        stacklevel=2,
    )
    common = {
        "stock_ids": stock_ids,
        "start": start,
        "end": end,
        "db_path": db_path,
        "drop_incomplete": drop_incomplete,
        "unlock_frozen": unlock_frozen,
        "frozen_reason": frozen_reason,
    }
    adjusted = load_prices(adjusted=True, **common)
    # 保留舊 PriceViews.actual 的完整欄位契約，但不做第二次 SQL 查詢。
    # 還原因子對同一根 OHLC 完全相同，所以可由 raw_close / close 反推
    # high/low；open/close 則直接使用 loader 在還原前留下的精確原值。
    actual = adjusted.copy()
    inverse_adjustment = actual[RAW_CLOSE_COLUMN] / actual["close"]
    actual["open"] = actual[RAW_OPEN_COLUMN]
    actual["high"] = actual["high"] * inverse_adjustment
    actual["low"] = actual["low"] * inverse_adjustment
    actual["close"] = actual[RAW_CLOSE_COLUMN]
    return PriceViews(adjusted=adjusted, actual=actual)


def _latest_price_date(
    con: sqlite3.Connection, stock_ids: list[str] | None
) -> date | None:
    """取得本次標的範圍的資料截止日，防止 end=None 繞過凍結。"""
    if stock_ids:
        placeholders = ",".join("?" * len(stock_ids))
        row = con.execute(
            f"SELECT MAX(date) FROM stock_daily WHERE stock_id IN ({placeholders})",
            stock_ids,
        ).fetchone()
    else:
        row = con.execute("SELECT MAX(date) FROM stock_daily").fetchone()
    return date.fromisoformat(row[0]) if row and row[0] else None


def _guard_frozen_access(
    con: sqlite3.Connection,
    stock_ids: list[str] | None,
    end: date | None,
    db_path: Path,
    unlock_frozen: bool,
    frozen_reason: str | None,
    access_kind: str,
) -> None:
    """價格與籌碼共用的凍結守門，避免從任一入口偷看未來。"""
    requested_end = end or _latest_price_date(con, stock_ids)
    if requested_end is None or requested_end < FROZEN_DATA_START:
        return
    if not unlock_frozen:
        raise DataNotAvailableError(
            f"{FROZEN_DATA_START} 起是凍結區間；解鎖請傳 "
            "unlock_frozen=True 並記錄理由"
        )
    if frozen_reason is None or not frozen_reason.strip():
        raise DataNotAvailableError("解鎖凍結區間必須記錄理由，不可留白")
    _log_frozen_access(db_path, requested_end, frozen_reason, access_kind)


def _log_frozen_access(
    db_path: Path,
    requested_end: date,
    reason: str,
    access_kind: str,
) -> None:
    """以 JSON Lines 記錄每次凍結資料解鎖，不讓重看 OOS 靜默發生。"""
    caller = "unknown"
    this_file = Path(__file__).resolve()
    for frame in inspect.stack()[1:]:
        if Path(frame.filename).resolve() != this_file:
            caller = f"{frame.filename}:{frame.function}:{frame.lineno}"
            break

    record = {
        "accessed_at": datetime.now(UTC).isoformat(),
        "requested_end": requested_end.isoformat(),
        "access_kind": access_kind,
        "caller": caller,
        "reason": reason,
    }
    log_path = db_path.parent / "frozen_access.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def _is_complete(df: pd.DataFrame) -> pd.Series:
    """OHLC 皆為有限正值的列遮罩"""
    mask = pd.Series(True, index=df.index)
    for col in OHLC_COLUMNS:
        values = pd.to_numeric(df[col], errors="coerce")
        mask &= values.notna() & (values > 0)
    return mask


def price_quality_report(
    stock_ids: list[str] | None = None,
    start: date | None = None,
    end: date | None = None,
    db_path: Path = DEFAULT_DB_PATH,
) -> pd.DataFrame:
    """
    列出每檔標的的資料洞，供報告揭露。

    Returns:
        欄位 stock_id / total_rows / incomplete_rows / incomplete_dates，
        只保留有缺漏的標的。

    必須揭露而不是靜默丟棄：資料洞會影響 ATR 視窗與報酬分位數的樣本數，
    使用者有權知道哪幾天被剔除了。
    """
    raw = load_prices(
        stock_ids, start, end, db_path=db_path, adjusted=False, drop_incomplete=False
    ).reset_index()

    complete = _is_complete(raw)
    bad = raw.loc[~complete, ["stock_id", "date"]]

    records = [
        {
            "stock_id": stock_id,
            "total_rows": int((raw["stock_id"] == stock_id).sum()),
            "incomplete_rows": len(group),
            "incomplete_dates": [d.date().isoformat() for d in group["date"]],
        }
        for stock_id, group in bad.groupby("stock_id")
    ]
    return pd.DataFrame(
        records, columns=["stock_id", "total_rows", "incomplete_rows", "incomplete_dates"]
    )


OHLC_COLUMNS = ("open", "high", "low", "close")


def _apply_adjustment(df: pd.DataFrame) -> pd.DataFrame:
    """
    用還原收盤價調整 OHLC。

    調整因子 = adj_close / close，同比例套用到 open/high/low，
    讓 K 線形狀不變、水準對齊還原價。

    ## 缺值時補**因子**，不是保留原價

    實測 bug：0050 在 2025-06-18 做了 1:4 分割（188.65 → 47.57）。
    還原價正確處理了它，但有 11 天缺還原價。原本「缺值時保留原價」
    的做法，讓那些天留在**分割前的尺度**：

        2021-04-06  原始 137.65（被保留）
        鄰近日       還原後約 33.5
        → 單日 4 倍尖峰 → 0050 的假回撤 84.20%

    還原因子在兩次公司行為之間是**常數**，所以正確做法是沿用前一個
    已知因子（序列開頭則用之後的第一個）。

    整檔完全沒有還原價時才保留原價——那時沒有因子可補。

    `is_adjusted` 仍然只標記**原生**有還原價的列，讓報告分得出哪些
    是查來的、哪些是推的。

    不就地改寫傳入的 DataFrame（CLAUDE.md 程式風格：回傳新物件）。

    註：SQLite 讀出的價格欄位可能是 int64，pandas 3.0 不允許把 float
    寫進 int64 欄位，因此先統一轉 float64 再運算。
    """
    # `adj_close` 也要轉。整檔都沒有還原價時它是全 NULL，SQLite 讀成
    # object dtype，`np.isfinite` 會拋 TypeError——而這個函式的說明明寫
    # 「整檔完全沒有還原價時保留原價」，所以那條路徑必須真的走得通。
    # 目前資料庫 1,218 檔全部有還原價，所以這是潛在而非現行故障；
    # 但還原價是 `adj_backfill_log` 那個獨立步驟補的，新上市股在補完前
    # 就會踩到。
    numeric = df.astype(
        {col: "float64" for col in (*OHLC_COLUMNS, "adj_close")}
    )

    raw_factor = numeric["adj_close"] / numeric["close"]
    usable = raw_factor.notna() & np.isfinite(raw_factor) & (raw_factor > 0)
    factor = raw_factor.where(usable)

    # 因子分標的補。跨標的外溢會讓另一檔的價格整段偏掉。
    if "stock_id" in numeric.columns:
        grouped = factor.groupby(numeric["stock_id"], sort=False)
        factor = grouped.ffill()
        factor = factor.groupby(numeric["stock_id"], sort=False).bfill()
    else:
        factor = factor.ffill().bfill()

    # 整檔都沒有還原價 → 補完仍是 NaN → 保留原價（因子 1.0）
    safe_factor = factor.fillna(1.0)

    return numeric.assign(
        **{col: numeric[col] * safe_factor for col in OHLC_COLUMNS},
        is_adjusted=usable,
    )


def load_chips(
    stock_ids: list[str] | None = None,
    start: date | None = None,
    end: date | None = None,
    db_path: Path = DEFAULT_DB_PATH,
    unlock_frozen: bool = False,
    frozen_reason: str | None = None,
) -> pd.DataFrame:
    """
    載入籌碼資料，格式對齊 `features.chips.CHIPS_REQUIRED_COLUMNS`。

    Args:
        stock_ids: 股票代號；None 表示全部
        start / end: 日期範圍（含端點）
        db_path: 上游 SQLite 路徑
        unlock_frozen: 明確允許讀取凍結日起的資料；每次都會寫稽核 log
        frozen_reason: 解鎖理由，不可省略或留白

    Returns:
        MultiIndex (stock_id, date)，欄位：
            foreign_net / trust_net / dealer_net   三大法人買賣超淨額
            margin_balance / short_balance          融資 / 融券餘額
            volume / close                          正規化與比率計算用

    關於「淨額」：上游把買進與賣出分開存（`foreign_buy` / `foreign_sell`），
    但特徵層要的是淨額。轉換在載入層做，讓 `features/chips.py` 只處理
    一種欄位語意。

    關於缺漏：籌碼資料缺漏的列一律**剔除，不補 0**。補 0 會被誤讀成
    「法人當天沒買賣」，而實際是「不知道」——兩者對訊號的意義完全不同。
    """
    con = _connect(db_path)
    try:
        _guard_frozen_access(
            con,
            stock_ids,
            end,
            db_path,
            unlock_frozen,
            frozen_reason,
            access_kind="chips",
        )
        where: list[str] = []
        params: list[object] = []

        if stock_ids:
            placeholders = ",".join("?" * len(stock_ids))
            where.append(f"d.stock_id IN ({placeholders})")
            params.extend(stock_ids)
        if start:
            where.append("d.date >= ?")
            params.append(start.isoformat())
        if end:
            where.append("d.date <= ?")
            params.append(end.isoformat())

        clause = f"WHERE {' AND '.join(where)}" if where else ""
        sql = f"""
            SELECT d.stock_id, d.date, d.close, d.volume,
                   i.foreign_buy, i.foreign_sell,
                   i.trust_buy, i.trust_sell,
                   i.dealer_buy, i.dealer_sell,
                   m.margin_balance, m.short_balance
            FROM stock_daily AS d
            JOIN stock_daily_institutional AS i
              ON i.stock_id = d.stock_id AND i.date = d.date
            JOIN stock_daily_margin AS m
              ON m.stock_id = d.stock_id AND m.date = d.date
            {clause}
            ORDER BY d.stock_id ASC, d.date ASC
        """
        df = pd.read_sql_query(sql, con, params=params)
    finally:
        con.close()

    if df.empty:
        raise DataNotAvailableError("查無籌碼資料，請確認標的與日期範圍")

    df["date"] = pd.to_datetime(df["date"])

    numeric = df.astype({
        col: "float64"
        for col in (
            "close", "volume",
            "foreign_buy", "foreign_sell",
            "trust_buy", "trust_sell",
            "dealer_buy", "dealer_sell",
            "margin_balance", "short_balance",
        )
    })

    result = numeric.assign(
        foreign_net=numeric["foreign_buy"] - numeric["foreign_sell"],
        trust_net=numeric["trust_buy"] - numeric["trust_sell"],
        dealer_net=numeric["dealer_buy"] - numeric["dealer_sell"],
    )[
        [
            "stock_id", "date",
            "foreign_net", "trust_net", "dealer_net",
            "margin_balance", "short_balance",
            "volume", "close",
        ]
    ]

    result = result[result.notna().all(axis=1)]
    if result.empty:
        raise DataNotAvailableError("剔除缺漏列後無籌碼資料可用")

    return result.set_index(["stock_id", "date"]).sort_index()


def coverage_report(db_path: Path = DEFAULT_DB_PATH) -> pd.DataFrame:
    """
    各資料表的覆蓋率，用來決定哪些特徵可用。

    以 stock_daily 列數為基準。低覆蓋率的表（例如 FinMind 額度撞牆導致
    抓不完的 PER / 月營收）對應的特徵不可使用，否則 dropna 會清空樣本
    —— 這正是 qlib-tw-trader 訓練失敗的原因。
    """
    tables = (
        "stock_daily",
        "stock_daily_adj",
        "stock_daily_institutional",
        "stock_daily_margin",
        "stock_daily_per",
        "stock_daily_securities_lending",
        "stock_daily_shareholding",
        "stock_monthly_revenue",
    )

    con = _connect(db_path)
    try:
        baseline = con.execute("SELECT COUNT(*) FROM stock_daily").fetchone()[0]
        if baseline == 0:
            raise DataNotAvailableError("stock_daily 無資料")

        records = []
        for table in tables:
            try:
                rows = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            except sqlite3.OperationalError:
                rows = 0
            records.append(
                {
                    "table": table,
                    "rows": rows,
                    "coverage": rows / baseline,
                    "usable": rows / baseline >= 0.80,
                }
            )
    finally:
        con.close()

    return pd.DataFrame(records).sort_values("coverage", ascending=False).reset_index(drop=True)
