"""
回測資料集組裝

把 `load_prices` 與 `load_chips` 的 MultiIndex 表攤成
`{股票代號: 日 K}`，並擋掉序列過短的標的。

## 為什麼要擋短序列

籌碼資料的覆蓋期間比價格短。inner join 之後價格序列會被砍到與籌碼
一樣長——**而且不會拋錯**。

實測案例：7769 join 後只剩 71 根，卻被當成完整標的送進回測。暖機期
250 根都不夠，它產出的每一筆決策都建立在不足的歷史上。

## 為什麼沒有籌碼的標的要保留

某檔缺籌碼不代表它的價格資料沒用。動能族與均值回歸族只需要 OHLCV，
丟掉整檔等於平白縮小標的池。籌碼族會在自己的 `required_columns`
檢查時跳過它。
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

CHIP_COLUMNS = (
    "foreign_net",
    "trust_net",
    "dealer_net",
    "margin_balance",
    "short_balance",
)

MIN_SERIES_LENGTH = 400
"""
標的最短序列長度。

短於此值的標的（近期上市、或籌碼資料只有末段）無法提供足夠的
暖機 + 訓練樣本，納入只會汙染全域時間軸。
"""


@dataclass(frozen=True)
class Dataset:
    """組裝結果。剔除紀錄要保留，否則無法解釋「為什麼只跑了 38 檔」"""

    by_stock: dict[str, pd.DataFrame]

    skipped: tuple[tuple[str, int], ...]
    """(代號, 實際長度) — 序列過短被剔除"""

    missing: tuple[str, ...]
    """完全查無價格資料"""

    @property
    def lengths(self) -> tuple[int, ...]:
        return tuple(len(b) for b in self.by_stock.values())

    def describe(self) -> str:
        if not self.by_stock:
            return "資料集為空——所有標的都被剔除"
        lines = [
            f"可用 {len(self.by_stock)} 檔｜"
            f"序列長度 {min(self.lengths)} ~ {max(self.lengths)} 根"
        ]
        if self.skipped:
            lines.append(
                f"剔除過短標的 {len(self.skipped)} 檔："
                f"{[f'{s}({n})' for s, n in self.skipped]}"
            )
        if self.missing:
            lines.append(f"查無資料 {len(self.missing)} 檔：{list(self.missing)}")
        return "\n".join(lines)


def build_dataset(
    stock_ids: list[str],
    prices: pd.DataFrame,
    chips: pd.DataFrame | None = None,
    min_length: int = MIN_SERIES_LENGTH,
) -> Dataset:
    """
    把 MultiIndex 的價格與籌碼表組成 `{代號: 日 K}`。

    Args:
        stock_ids: 要納入的標的
        prices: MultiIndex(stock_id, date) 的價格表（**不會被修改**）
        chips: MultiIndex(stock_id, date) 的籌碼表；None 表示不併入
        min_length: 最短序列長度，短於此值剔除

    Returns:
        Dataset（含剔除紀錄）

    籌碼以 left join 併入。缺值保留為 NaN，讓需要籌碼的策略自行跳過，
    但不能連帶刪除動能等只需要價格的策略日期。
    """
    price_ids = set(prices.index.get_level_values("stock_id"))
    chip_ids = (
        set(chips.index.get_level_values("stock_id")) if chips is not None else set()
    )

    by_stock: dict[str, pd.DataFrame] = {}
    skipped: list[tuple[str, int]] = []
    missing: list[str] = []

    for sid in stock_ids:
        if sid not in price_ids:
            missing.append(sid)
            continue

        bars = prices.xs(sid, level="stock_id")
        if chips is not None and sid in chip_ids:
            bars = bars.join(
                chips.xs(sid, level="stock_id")[list(CHIP_COLUMNS)], how="left"
            )

        if len(bars) < min_length:
            skipped.append((sid, len(bars)))
            continue

        by_stock[sid] = bars

    return Dataset(
        by_stock=by_stock,
        skipped=tuple(skipped),
        missing=tuple(missing),
    )
