"""
低周轉集中持有 vs 0050 買進持有（開發集）

## 為什麼這是新問題

專案至今測的全是高周轉策略（H=5 ~ 120，每年 2 ~ 50 趟）。那些測試的
結論一律被成本吃掉，而 2026-09-19 算出原因：

```
同一個 1.071% 的來回成本
事件策略 H=40   每年 6.30 趟  →  年化拖累 6.75%
買進持有 5 年   每年 0.20 趟  →  年化拖累 0.21%
```

**「成本吃掉一切」的前提是高周轉。** 低周轉從來沒有被正式測過。

## 為什麼測規則而不是名單

使用者提的組合是「00712 ＋ 台積電 ＋ 台達電各 1/3」。**那個名單不能
誠實回測**——它是 2026 年挑的，拿去跑 2015 起的歷史就是禁令 1 的
look-ahead：我已經知道台積電這十年漲了多少。

所以這裡測**事前規則**：決策日當天依市值排名選前 N 檔，買進後完全
不動到區間結束。名單由當時的資料決定，不由我決定。

⚠️ 這**不是**在回答「00712 ＋ 台積電 ＋ 台達電好不好」。那個問題
無法誠實回答，而且那是個別投資建議。

## 多個進場日

單一進場日的結果是一次抽樣。2015-01 進場與 2018-01 進場可能差很多，
而那個差異與策略無關。所以掃多個進場年份並報全距。

## 成本

進場一次、出場一次，逐檔用 `config.costs.resolve_tier`。
同時並列 `config.slippage` 的跳動單位模型（2026-09-19 新增），因為
禁令 4 的零股滑價 0.3% 沒有量測支撐，而它在低周轉下影響很小——
**這一點本身就是低周轉的優勢，值得看見。**

## 禁令

```
禁令 1   名單由決策日當時的市值排名決定，不用未來資訊
禁令 2   標的池用時點快照（universe_history.as_of_date）
禁令 6   --end 預設且硬限 2023-12-29
```
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from taiwan_quant.config import slippage as SLIP  # noqa: E402
from taiwan_quant.config.costs import DEFAULT as COST  # noqa: E402
from taiwan_quant.config.costs import resolve_tier  # noqa: E402
from taiwan_quant.data.etf_universe import is_etf  # noqa: E402
from taiwan_quant.data.loader import (  # noqa: E402
    DEFAULT_UNIVERSE_BASIS,
    HISTORY_DB_PATH,
    load_prices,
)

DEV_END = date(2023, 12, 29)
CAPITAL = 400_000.0
LARGE_TIER_SIZE = 50
BENCHMARKS = ("0050", "0056", "0051")
TOP_N_GRID = (1, 3, 5, 10)
ENTRY_YEARS = (2015, 2016, 2017, 2018, 2019)
"""多個進場年份。單一進場日的結果是一次抽樣，不是策略的性質"""


class DiagnosticError(RuntimeError):
    """資料不足。"""


@dataclass(frozen=True)
class HoldResult:
    """一個 (規則, 進場日) 組合的買進持有結果。"""

    label: str
    entry: str
    exit: str
    names: tuple[str, ...]
    total_return: float
    annualised: float
    max_drawdown: float
    entry_cost: float
    """進出各一次的總成本率（佔期初資金）"""

    @property
    def net_total(self) -> float:
        return (1.0 + self.total_return) * (1.0 - self.entry_cost) - 1.0


def top_n_at(con: sqlite3.Connection, as_of: str, n: int) -> list[str]:
    """
    決策日當時市值排名前 n 檔（禁令 1、2）。

    `universe_history` 的快照是逐季的，所以取 <= as_of 的最近一筆。
    """
    snapshot = con.execute(
        "SELECT MAX(as_of_date) FROM universe_history "
        "WHERE basis = ? AND as_of_date <= ?",
        (DEFAULT_UNIVERSE_BASIS, as_of),
    ).fetchone()[0]
    if snapshot is None:
        raise DiagnosticError(f"{as_of} 之前沒有 {DEFAULT_UNIVERSE_BASIS} 快照")
    return [
        row[0]
        for row in con.execute(
            "SELECT stock_id FROM universe_history "
            "WHERE basis = ? AND as_of_date = ? AND rank <= ? ORDER BY rank",
            (DEFAULT_UNIVERSE_BASIS, snapshot, n),
        )
    ]


def equity_curve(
    closes: pd.DataFrame, names: list[str], entry: pd.Timestamp
) -> pd.Series:
    """等權買進持有，不再平衡。缺任一檔的歷史就拋錯，不靜默剔除。"""
    window = closes.loc[entry:, names]
    first = window.iloc[0]
    missing = [n for n in names if not np.isfinite(first[n]) or first[n] <= 0]
    if missing:
        raise DiagnosticError(f"{entry.date()} 缺進場價：{missing}")
    # 等權：每檔投入相同金額，之後不動
    normalised = window.divide(first, axis=1)
    return normalised.mean(axis=1).dropna()


def round_trip_cost(
    names: list[str],
    closes: pd.DataFrame,
    volumes: pd.DataFrame,
    entry: pd.Timestamp,
    large_members: set[str],
    *,
    use_tick_model: bool,
    odd_lot_multiplier: float = 2.0,
) -> float:
    """
    進出各一次的平均來回成本率。

    `use_tick_model=True` 時用 `config.slippage` 的跳動單位模型，
    否則用 `config.costs` 的兩段式（禁令 4）。**兩個都報**——
    零股 0.3% 沒有量測支撐，而低周轉下這個差異幾乎不影響結論，
    那一點本身就值得看見。
    """
    per_position = CAPITAL / len(names)
    rates: list[float] = []
    for name in names:
        price = float(closes.at[entry, name])
        if use_tick_model:
            turnover = price * float(volumes.at[entry, name])
            estimate = SLIP.estimate(
                price=price,
                amount=per_position,
                daily_turnover=turnover,
                whole_lots=SLIP.affordable_whole_lots(price, per_position),
                odd_lot_multiplier=odd_lot_multiplier,
            )
            if not np.isfinite(estimate.total):
                raise DiagnosticError(f"{name} 在 {entry.date()} 當日無成交額")
            fee = COST.commission(per_position) / per_position
            rates.append(2 * fee + COST.tax(per_position) / per_position
                         + 2 * estimate.total)
        else:
            tier = resolve_tier(
                actual_price=price, adjusted_price=price,
                amount=per_position, large=name in large_members,
                is_etf=is_etf(name),
            )
            rates.append(COST.round_trip_rate(tier))
    return float(np.mean(rates))


def summarise(curve: pd.Series) -> tuple[float, float, float]:
    """總報酬、年化、最大回撤。"""
    total = float(curve.iloc[-1] / curve.iloc[0] - 1.0)
    years = (curve.index[-1] - curve.index[0]).days / 365.25
    annualised = (1.0 + total) ** (1.0 / years) - 1.0 if years > 0 else float("nan")
    drawdown = float((curve / curve.cummax() - 1.0).min())
    return total, annualised, drawdown


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(HISTORY_DB_PATH))
    parser.add_argument("--end", default=DEV_END.isoformat())
    parser.add_argument(
        "--output", type=Path, default=Path("reports/low_turnover_dev.json")
    )
    args = parser.parse_args()

    end = date.fromisoformat(args.end)
    if end > DEV_END:
        parser.error("禁令 6：不得載入 2024+；--end 最晚為 2023-12-29")

    db_path = Path(args.db)
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        members = [
            row[0]
            for row in con.execute(
                "SELECT DISTINCT stock_id FROM universe_history WHERE basis = ?",
                (DEFAULT_UNIVERSE_BASIS,),
            )
        ]
        load_ids = sorted(set(members) | set(BENCHMARKS))
        prices = load_prices(
            load_ids, start=date(2015, 1, 1), end=end,
            adjusted=True, db_path=db_path,
        )
        closes = prices["close"].unstack("stock_id").sort_index()
        volumes = prices["volume"].unstack("stock_id").sort_index()
        calendar = list(closes.index)

        # market_cap 快照從 2015-04-01 起（2015-01 那期缺，見
        # 03_待辦與改進方向.md 第 9 項），所以進場日必須晚於第一期快照，
        # 否則選名單時沒有時點資料可用。
        first_snapshot = con.execute(
            "SELECT MIN(as_of_date) FROM universe_history WHERE basis = ?",
            (DEFAULT_UNIVERSE_BASIS,),
        ).fetchone()[0]
        if first_snapshot is None:
            raise DiagnosticError("universe_history 沒有任何 market_cap 快照")
        earliest = pd.Timestamp(first_snapshot)
        print(f"最早市值快照 {first_snapshot}——進場日不得早於此\n")

        results: list[HoldResult] = []
        for year in ENTRY_YEARS:
            entry = next(
                (d for d in calendar if d.year == year and d >= earliest), None
            )
            if entry is None:
                continue
            large_members = set(top_n_at(con, str(entry.date()), LARGE_TIER_SIZE))

            rules: list[tuple[str, list[str]]] = []
            for n in TOP_N_GRID:
                try:
                    names = top_n_at(con, str(entry.date()), n)
                except DiagnosticError:
                    continue
                if len(names) == n:
                    rules.append((f"市值前 {n} 檔", names))
            for etf in BENCHMARKS:
                if etf in closes.columns:
                    rules.append((f"{etf} 買進持有", [etf]))

            for label, names in rules:
                try:
                    curve = equity_curve(closes, names, entry)
                    total, annualised, drawdown = summarise(curve)
                    cost = round_trip_cost(
                        names, closes, volumes, entry, large_members,
                        use_tick_model=False,
                    )
                except DiagnosticError:
                    continue
                results.append(HoldResult(
                    label=label, entry=str(entry.date()),
                    exit=str(curve.index[-1].date()), names=tuple(names),
                    total_return=total, annualised=annualised,
                    max_drawdown=drawdown, entry_cost=cost,
                ))
    finally:
        con.close()

    if not results:
        raise DiagnosticError("沒有任何組合跑得出來")

    payload = {
        "end": args.end,
        "capital": CAPITAL,
        "entry_years": list(ENTRY_YEARS),
        "top_n_grid": list(TOP_N_GRID),
        "results": [
            {
                "label": r.label, "entry": r.entry, "exit": r.exit,
                "names": list(r.names), "total_return": r.total_return,
                "annualised": r.annualised, "max_drawdown": r.max_drawdown,
                "round_trip_cost": r.entry_cost, "net_total": r.net_total,
            }
            for r in results
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    print(f"=== 低周轉買進持有（進場後完全不動，至 {args.end}）===\n")
    labels = sorted({r.label for r in results},
                    key=lambda s: (not s[0].isdigit(), s))
    print(f"{'規則':>14s} " + " ".join(f"{y:>9d}" for y in ENTRY_YEARS)
          + f" {'年化全距':>10s}")
    for label in labels:
        row = f"{label:>14s} "
        anns = []
        for year in ENTRY_YEARS:
            hit = next((r for r in results
                        if r.label == label and r.entry.startswith(str(year))), None)
            if hit is None:
                row += f"{'—':>9s} "
            else:
                row += f"{hit.annualised:>9.2%} "
                anns.append(hit.annualised)
        row += f" {max(anns) - min(anns):>9.2%}" if len(anns) > 1 else ""
        print(row)

    print("\n=== 成本在低周轉下幾乎不重要 ===")
    for label in labels[:6]:
        hit = next((r for r in results if r.label == label), None)
        if hit is None:
            continue
        years = 9.0
        print(f"  {label:>14s} 來回成本 {hit.entry_cost:.3%}"
              f"｜攤到 {years:.0f} 年 = 每年 {hit.entry_cost / years:.4%}")
    print(f"\n已寫入 {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
