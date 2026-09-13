#!/usr/bin/env python3
"""
停用依賴稀疏欄位的因子

背景：qlib-tw-trader 的訓練流程會對所有啟用因子做 dropna。只要有一個
因子整欄是 NaN，所有樣本都會被丟掉，LightGBM 收到空資料集並拋出：

    Check failed: (num_data) > (0) at .../dataset.cpp, line 44

這在資料未完整同步時必然發生（例如 FinMind 免費額度撞牆，
PER / 月營收 / 集保 / 借券資料抓不完）。原專案沒有這道防護。

本腳本量測各底層欄位的實際覆蓋率，把覆蓋率低於門檻的欄位所對應的
因子停用，並印出完整清單，讓「訓練用了哪些因子」是可稽核的。

用法：
    PYTHONPATH=. .venv/bin/python scripts/disable_sparse_factors.py            # 預覽
    PYTHONPATH=. .venv/bin/python scripts/disable_sparse_factors.py --apply    # 實際寫入
    PYTHONPATH=. .venv/bin/python scripts/disable_sparse_factors.py --reset    # 全部重新啟用
"""

from __future__ import annotations

import argparse
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

DB_PATH = Path("data/data.db")
COVERAGE_THRESHOLD = 0.80
"""欄位覆蓋率門檻：低於此比例即視為稀疏，停用相關因子"""

# 底層欄位 → (來源資料表, 該表中的欄位名)
# 覆蓋率 = 該表列數 / stock_daily 列數
FIELD_SOURCE: dict[str, str] = {
    "open": "stock_daily",
    "high": "stock_daily",
    "low": "stock_daily",
    "close": "stock_daily",
    "volume": "stock_daily",
    "adj_close": "stock_daily_adj",
    "foreign_buy": "stock_daily_institutional",
    "foreign_sell": "stock_daily_institutional",
    "trust_buy": "stock_daily_institutional",
    "trust_sell": "stock_daily_institutional",
    "dealer_buy": "stock_daily_institutional",
    "dealer_sell": "stock_daily_institutional",
    "margin_buy": "stock_daily_margin",
    "margin_sell": "stock_daily_margin",
    "margin_balance": "stock_daily_margin",
    "short_buy": "stock_daily_margin",
    "short_sell": "stock_daily_margin",
    "short_balance": "stock_daily_margin",
    "pe_ratio": "stock_daily_per",
    "pb_ratio": "stock_daily_per",
    "dividend_yield": "stock_daily_per",
    "revenue": "stock_monthly_revenue",
    "lending_volume": "stock_daily_securities_lending",
    "foreign_shares": "stock_daily_shareholding",
    "foreign_ratio": "stock_daily_shareholding",
    "foreign_remaining_shares": "stock_daily_shareholding",
    "foreign_remaining_ratio": "stock_daily_shareholding",
    "total_shares": "stock_daily_shareholding",
    "chinese_upper_limit_ratio": "stock_daily_shareholding",
    "foreign_upper_limit_ratio": "stock_daily_shareholding",
}

FIELD_PATTERN = re.compile(r"\$([a-zA-Z_][a-zA-Z0-9_]*)")


@dataclass(frozen=True)
class FieldCoverage:
    field: str
    table: str
    rows: int
    coverage: float

    @property
    def is_sparse(self) -> bool:
        return self.coverage < COVERAGE_THRESHOLD


def measure_coverage(con: sqlite3.Connection) -> dict[str, FieldCoverage]:
    """量測每個底層欄位的資料覆蓋率"""
    cur = con.cursor()
    baseline = cur.execute("SELECT COUNT(*) FROM stock_daily").fetchone()[0]
    if baseline == 0:
        raise RuntimeError("stock_daily 無資料，請先執行資料同步")

    row_counts: dict[str, int] = {}
    for table in set(FIELD_SOURCE.values()):
        row_counts[table] = cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    return {
        field: FieldCoverage(
            field=field,
            table=table,
            rows=row_counts[table],
            coverage=row_counts[table] / baseline,
        )
        for field, table in FIELD_SOURCE.items()
    }


def fields_used(expression: str) -> set[str]:
    """取出因子公式參照的底層欄位"""
    return set(FIELD_PATTERN.findall(expression))


def main() -> None:
    parser = argparse.ArgumentParser(description="停用依賴稀疏欄位的因子")
    parser.add_argument("--apply", action="store_true", help="實際寫入資料庫")
    parser.add_argument("--reset", action="store_true", help="重新啟用全部因子")
    args = parser.parse_args()

    if not DB_PATH.exists():
        raise SystemExit(f"找不到資料庫：{DB_PATH}（請在專案根目錄執行）")

    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    if args.reset:
        cur.execute("UPDATE factors SET enabled = 1")
        con.commit()
        total = cur.execute("SELECT COUNT(*) FROM factors").fetchone()[0]
        print(f"已重新啟用全部 {total} 個因子")
        con.close()
        return

    coverage = measure_coverage(con)
    sparse = {f for f, c in coverage.items() if c.is_sparse}

    print("=" * 78)
    print("底層欄位覆蓋率")
    print("=" * 78)
    print(f"{'欄位':<30}{'來源表':<34}{'覆蓋率':>10}")
    for field in sorted(coverage, key=lambda f: coverage[f].coverage):
        c = coverage[field]
        flag = "  ← 稀疏" if c.is_sparse else ""
        print(f"{c.field:<30}{c.table:<34}{c.coverage * 100:>9.1f}%{flag}")
    print()
    print(f"門檻 {COVERAGE_THRESHOLD * 100:.0f}%，判定稀疏欄位 {len(sparse)} 個：{sorted(sparse)}")
    print()

    factors = cur.execute("SELECT id, name, category, expression FROM factors").fetchall()
    to_disable: list[tuple[int, str, str, set[str]]] = []
    unknown_fields: set[str] = set()

    for fid, name, category, expr in factors:
        used = fields_used(expr)
        unknown_fields |= used - set(FIELD_SOURCE)
        hit = used & sparse
        if hit:
            to_disable.append((fid, name, category, hit))

    print("=" * 78)
    print(f"將停用 {len(to_disable)} / {len(factors)} 個因子")
    print("=" * 78)
    by_category: dict[str, int] = {}
    for _, _, category, _ in to_disable:
        by_category[category] = by_category.get(category, 0) + 1
    print(f"分類統計：{by_category}")
    print()
    for fid, name, category, hit in to_disable[:20]:
        print(f"  {name:<32}[{category:<12}] 缺 {sorted(hit)}")
    if len(to_disable) > 20:
        print(f"  ...（其餘 {len(to_disable) - 20} 個省略）")
    print()

    if unknown_fields:
        print(f"⚠️  公式中出現未登記的欄位（未納入判斷）：{sorted(unknown_fields)}")
        print()

    remaining = len(factors) - len(to_disable)
    print(f"訓練可用因子：{remaining} 個")

    if not args.apply:
        print()
        print("（預覽模式，未寫入。加 --apply 實際執行）")
        con.close()
        return

    cur.execute("UPDATE factors SET enabled = 1")
    if to_disable:
        cur.executemany(
            "UPDATE factors SET enabled = 0 WHERE id = ?",
            [(fid,) for fid, _, _, _ in to_disable],
        )
    con.commit()

    enabled_now = cur.execute("SELECT COUNT(*) FROM factors WHERE enabled = 1").fetchone()[0]
    print()
    print(f"已寫入：啟用 {enabled_now} 個、停用 {len(factors) - enabled_now} 個")
    con.close()


if __name__ == "__main__":
    main()
