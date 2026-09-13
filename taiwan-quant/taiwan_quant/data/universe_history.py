"""
標的池歷史快照

解決 D2 與 CLAUDE.md 禁令 2 的 survivorship bias：回測需要知道
「在 2024-06-28 那天，0050 的成分股是哪 50 檔」。用今日名單回溯歷史會
把被剔除的弱股從樣本刪掉，績效被系統性高估。

三條不可協商的規則：

1. **絕不使用晚於決策日的快照。**
   `min(available_dates)` 這種寫法看起來很合理，但那等於用未來的成分股
   名單回測過去，是最嚴重的作弊之一。沒有可用快照時一律走代理重建。

2. **每個標的池都要宣告來源（`Provenance`）。**
   `REAL`（真實快照）或 `PROXY`（市值排名代理）必須明示，報告據此標註
   可信度。不可讓下游以為代理資料是真的。

3. **代理重建必須納入退市股。**
   `build_proxy_universe()` 只看傳進來的市值字典，不做任何「這檔還活著嗎」
   的過濾——呼叫端有責任把退市股的歷史市值也放進來
   （FinMind `TaiwanStockDelisting` 免費可用，可取得退市清單）。

資料來源現況（2026-09-12 實測）：

    真實成分股   元大投信官網（`constituents.py`）→ 只有**當期**
    歷史發行股數 TWSE MI_QFIIS 支援 `date=YYYYMMDD` → 可重建歷史市值
    歷史市值     FinMind `TaiwanStockMarketValue` → **付費方案才有**
    退市清單     FinMind `TaiwanStockDelisting` → 免費可用

因此：**今天起往後**每季存一份真實快照；**回測期間（2023-2026）**只能用
市值排名代理，並在報告明確標註。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from pathlib import Path

DEFAULT_STORE_DIR = Path(__file__).resolve().parents[2] / "data" / "universe_history"

TIER_ETF_IDS: tuple[str, ...] = ("0050", "0051")
"""標的池組成的 ETF，順序決定 `stock_ids` 的排列與分層"""

PROXY_LARGE_TIER_RANK = 50
"""代理模式下，市值排名前 N 名視為 0050 級（滑價較低）"""

DEFAULT_TOP_N = 150
"""D2 要求的標的池規模：0050（50）+ 0051（100）"""


class Provenance(str, Enum):
    """標的池來源"""

    REAL = "real"
    """來自真實成分股快照"""

    PROXY = "proxy"
    """由市值排名重建的代理名單"""


@dataclass(frozen=True)
class UniverseResolution:
    """
    某決策日的標的池，含來源與已知偏誤。

    `warnings` 必須隨回測報告一起輸出——代理名單產生的績效與真實
    成分股不可混為一談。
    """

    as_of: date
    provenance: Provenance
    stock_ids: list[str]
    snapshot_date: date | None
    """REAL 時為實際採用的快照日期；PROXY 時為 None"""

    tiers: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def tier_of(self, stock_id: str) -> str:
        """
        取得流動性分層（`config.costs.Tier` 的值）。

        REAL 模式用實際 ETF 歸屬判定；PROXY 模式用市值排名代理。
        """
        if stock_id not in self.tiers:
            raise KeyError(f"{stock_id} 不在 {self.as_of} 的標的池內")
        return self.tiers[stock_id]


class SnapshotStore:
    """
    成分股快照的檔案儲存。

    一個 JSON 檔一個日期：`{store_dir}/{YYYY-MM-DD}.json`
    格式簡單、可用 git diff 檢視、不需要資料庫（MVP 原則）。
    """

    def __init__(self, store_dir: Path = DEFAULT_STORE_DIR) -> None:
        self.store_dir = Path(store_dir)

    def _path(self, as_of: date) -> Path:
        return self.store_dir / f"{as_of.isoformat()}.json"

    def save(self, as_of: date, members: dict[str, list[str]]) -> Path:
        """
        存一份快照。

        Args:
            as_of: 快照日期（成分股資料的交易日）
            members: {etf_id: [stock_id, ...]}

        Raises:
            ValueError: 所有 ETF 的成員都是空的
        """
        if not any(members.values()):
            raise ValueError("快照不可為空——空名單會讓下游誤以為那天沒有成分股")

        self.store_dir.mkdir(parents=True, exist_ok=True)
        path = self._path(as_of)
        payload = {
            "as_of": as_of.isoformat(),
            "members": {etf: list(ids) for etf, ids in members.items()},
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return path

    def load(self, as_of: date) -> dict[str, list[str]] | None:
        """讀取指定日期的快照；不存在回傳 None"""
        path = self._path(as_of)
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload["members"]

    def available_dates(self) -> list[date]:
        """所有已存快照的日期，升冪"""
        if not self.store_dir.exists():
            return []
        dates: list[date] = []
        for path in self.store_dir.glob("*.json"):
            try:
                dates.append(date.fromisoformat(path.stem))
            except ValueError:
                continue
        return sorted(dates)

    def latest_on_or_before(self, as_of: date) -> date | None:
        """
        找出不晚於 `as_of` 的最近快照日期。

        **這個「不晚於」是關鍵。**回傳 None 時呼叫端必須走代理重建，
        不可退而使用最早的未來快照。
        """
        candidates = [d for d in self.available_dates() if d <= as_of]
        return max(candidates) if candidates else None


def build_proxy_universe(
    market_caps: dict[str, float],
    top_n: int = DEFAULT_TOP_N,
) -> list[str]:
    """
    用市值排名重建代理標的池。

    Args:
        market_caps: {stock_id: 市值}。**呼叫端必須把退市股也放進來**
                     （禁令 2）——本函式不做「這檔還活著嗎」的過濾。
        top_n: 取前幾名

    Returns:
        依市值降冪的股票代號；市值相同時以代號升冪排序，確保可重現。

    Raises:
        ValueError: 市值資料為空
    """
    if not market_caps:
        raise ValueError("市值資料為空，無法重建代理標的池")

    usable = {sid: cap for sid, cap in market_caps.items() if cap > 0}
    # 市值降冪；同市值時用代號升冪，讓結果可重現（禁令 7、8）
    ordered = sorted(usable.items(), key=lambda kv: (-kv[1], kv[0]))
    return [sid for sid, _ in ordered[:top_n]]


def resolve_universe(
    as_of: date,
    store: SnapshotStore | None = None,
    market_caps: dict[str, float] | None = None,
    top_n: int = DEFAULT_TOP_N,
) -> UniverseResolution:
    """
    取得某決策日的標的池。

    優先用不晚於 `as_of` 的真實快照；沒有就用市值排名代理，並附上
    survivorship 警告。

    Args:
        as_of: 決策日
        store: 快照儲存；None 表示用預設路徑
        market_caps: 代理重建用的市值資料（需含退市股）
        top_n: 標的池規模

    Raises:
        ValueError: 無可用快照且未提供 `market_caps`
    """
    store = store if store is not None else SnapshotStore()

    snapshot_date = store.latest_on_or_before(as_of)
    if snapshot_date is not None:
        members = store.load(snapshot_date)
        if members:
            return _from_snapshot(as_of, snapshot_date, members)

    if not market_caps:
        raise ValueError(
            f"{as_of} 沒有可用的真實快照（最早快照晚於此日），"
            "且未提供 market_caps 供代理重建。"
            "拒絕回傳空池，也拒絕使用未來快照。"
        )

    return _from_proxy(as_of, market_caps, top_n)


def _from_snapshot(
    as_of: date, snapshot_date: date, members: dict[str, list[str]]
) -> UniverseResolution:
    """由真實快照組裝，分層取自實際 ETF 歸屬"""
    stock_ids: list[str] = []
    tiers: dict[str, str] = {}

    for etf_id in TIER_ETF_IDS:
        for stock_id in members.get(etf_id, []):
            if stock_id in tiers:
                continue
            tiers[stock_id] = etf_id
            stock_ids.append(stock_id)

    # 名單中若有非 TIER_ETF_IDS 的 ETF，一併納入但分層歸為中型（保守）
    for etf_id, ids in members.items():
        if etf_id in TIER_ETF_IDS:
            continue
        for stock_id in ids:
            if stock_id in tiers:
                continue
            tiers[stock_id] = "0051"
            stock_ids.append(stock_id)

    return UniverseResolution(
        as_of=as_of,
        provenance=Provenance.REAL,
        stock_ids=stock_ids,
        snapshot_date=snapshot_date,
        tiers=tiers,
        warnings=[],
    )


def _from_proxy(
    as_of: date, market_caps: dict[str, float], top_n: int
) -> UniverseResolution:
    """由市值排名組裝代理名單，並附上必須揭露的偏誤警告"""
    stock_ids = build_proxy_universe(market_caps, top_n)
    tiers = {
        stock_id: ("0050" if rank < PROXY_LARGE_TIER_RANK else "0051")
        for rank, stock_id in enumerate(stock_ids)
    }

    return UniverseResolution(
        as_of=as_of,
        provenance=Provenance.PROXY,
        stock_ids=stock_ids,
        snapshot_date=None,
        tiers=tiers,
        warnings=[
            f"{as_of} 無真實成分股快照，改用市值前 {top_n} 名**代理** "
            "0050+0051，非實際成分股",
            "代理名單以事後可得的市值排名重建，可能含 survivorship bias："
            "若市值資料未涵蓋當期已退市個股，績效會被高估",
            f"流動性分層亦為代理（前 {PROXY_LARGE_TIER_RANK} 名視為 0050 級）",
        ],
    )
