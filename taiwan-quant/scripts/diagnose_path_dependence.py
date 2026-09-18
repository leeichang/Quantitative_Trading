"""
同一個擾動、兩種選股方案：路徑相依的對照實驗

## 為什麼要這支腳本

`2026-09-18_門檻形同虛設與路徑相依的根因.md` 用兩份既有報告估了擺盪
幅度（槽位排隊 v6→v7 數百 pp vs 定期換倉 task M 前後 −6.6 ~ −24.2 pp），
但明白標了一個限制：

> ⚠️ **兩個擾動不是同一個，這不是對照實驗。** v6→v7 補了 23 期快照；
> task M 改了 join 方式。方向與數量級的對比成立，精確倍數不成立。

**這支腳本補那個缺口**：對同一組擾動，同時跑槽位排隊與定期換倉。

## 擾動怎麼造

每個決策日從當時的標的池隨機移掉 `DROP_PER_DATE` 檔。那模擬 v6→v7
的實際變化——市值快照補齊後，部分日期的前 150 名換了幾檔。

```
基準      完整標的池
擾動 i    每個決策日移掉 3 檔（150 檔的 2%），種子 i
```

**兩個方案吃完全相同的擾動後標的池**，所以報酬差異只能來自選股機制。

## 成本

一次性預算約 22 分鐘，其中 `precompute_outcomes` 佔 83%（1093s）。

**但那兩個預算都是逐檔的，與標的池成員無關**，所以只做一次；
每個擾動只需重跑 `resolve_members`（4.4s）與下游的 walk-forward。
擾動數量因此幾乎免費。

## 禁令 6

`--end` 預設 2023-12-29 且硬性拒絕 2024+。開發集可以反覆跑。

## 用法

    python scripts/diagnose_path_dependence.py
    python scripts/diagnose_path_dependence.py --perturbations 8 --drop 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.validate_oos_trailing import (  # noqa: E402
    RANDOM_SEED,
    WARMUP_DAYS,
    enumerate_decisions,
    infer_tiers,
    make_price_lookup,
    precompute_outcomes,
    precompute_scores,
    resolve_members,
    run_walk_forward,
    to_signals,
    trading_calendar,
)
from taiwan_quant.backtest.portfolio_sim import simulate_portfolio  # noqa: E402
from taiwan_quant.data.dataset import build_dataset  # noqa: E402
from taiwan_quant.data.loader import (  # noqa: E402
    DEFAULT_UNIVERSE_BASIS,
    HISTORY_DB_PATH,
    load_chips,
    load_prices,
)
from taiwan_quant.strategies.families import STRATEGY_FAMILIES  # noqa: E402
from taiwan_quant.validation.benchmarks import ETF_BENCHMARKS  # noqa: E402
from taiwan_quant.validation.path_dependence import divergence  # noqa: E402
from taiwan_quant.validation.thresholds import (  # noqa: E402
    annotate_tie_counts,
    select_periodic_rebalances,
)

DEV_END = date(2023, 12, 29)
UNIVERSE_SIZE = 150
HORIZON = 60
"""60 日持有，與 v6/v7 那張表一致才能對比"""

N_SLOTS = 3
DECISION_STRIDE = 5

DEFAULT_PERTURBATIONS = 6
DEFAULT_DROP = 3
"""每個決策日移掉的檔數。150 檔的 2%，對應 v6→v7 的名單微調"""

SLOT = "槽位排隊"
REBALANCE = "定期換倉"


class DiagnosticError(RuntimeError):
    """輸入或資料不足。"""


@dataclass(frozen=True)
class Run:
    """一個 (方案, 擾動) 組合的結果。"""

    scheme: str
    perturbation: int
    total_return: float
    n_trades: int
    slot_blocked: int
    trades: tuple[object, ...]

    @property
    def signal_trade_ratio(self) -> float:
        if self.n_trades == 0:
            return float("inf")
        return (self.n_trades + self.slot_blocked) / self.n_trades


def perturb(
    members_at: dict[pd.Timestamp, set[str]], *, drop: int, seed: int
) -> dict[pd.Timestamp, set[str]]:
    """
    每個決策日隨機移掉 `drop` 檔，**不修改傳入的字典**。

    `seed=0` 回傳未擾動的副本，當作基準——讓基準與擾動走完全相同的
    程式路徑，避免「基準比較快所以比較準」這類混淆。
    """
    if drop < 0:
        raise DiagnosticError(f"drop 不可為負，得到 {drop}")
    if seed == 0 or drop == 0:
        return {day: set(names) for day, names in members_at.items()}

    rng = np.random.default_rng(seed)
    out: dict[pd.Timestamp, set[str]] = {}
    for day, names in members_at.items():
        ordered = sorted(names)
        if len(ordered) <= drop:
            out[day] = set(ordered)
            continue
        victims = rng.choice(len(ordered), size=drop, replace=False)
        out[day] = {n for i, n in enumerate(ordered) if i not in set(victims.tolist())}
    return out


def _run_scheme(
    scheme: str,
    raw_signals: list,
    oos_calendar: list[pd.Timestamp],
    close_lookup,
    execution_lookup,
) -> tuple[float, int, int, tuple]:
    """
    跑一個方案，回 (總報酬, 成交筆數, 被槽位擋掉的訊號數, 成交紀錄)。

    ⚠️ `select_periodic_rebalances` 的換倉節點是
    `range(0, len(calendar), rebalance_every)`——**傳進去的是哪一份日曆
    決定了節點落在哪裡。** 必須傳 `oos_calendar`，與
    `evaluate_threshold_schemes.py` 一致。

    傳完整日曆（2197 天）會讓節點散佈在 2015-2023 全段，而訊號只存在於
    暖機後的決策日，於是大多數節點是空的。本腳本第一版就是這樣，
    動能突破跑出 −5.38%（正確值量級是 +300%）。
    """
    if scheme == REBALANCE:
        selected = select_periodic_rebalances(
            raw_signals, oos_calendar, execution_lookup, HORIZON, N_SLOTS
        )
        signals = list(selected.signals)
    elif scheme == SLOT:
        signals = annotate_tie_counts(raw_signals)
    else:
        raise DiagnosticError(f"未知方案 {scheme}")

    result = simulate_portfolio(
        signals, close_lookup, oos_calendar, n_slots=N_SLOTS
    )
    return (
        result.total_return,
        result.n_trades,
        result.slot_blocked_signals,
        result.trades,
    )


def _spread(values: list[float]) -> dict[str, float]:
    """擺盪幅度。全距是驗收條件直接問的量。"""
    array = np.array(values, dtype=float)
    return {
        "min": float(array.min()),
        "max": float(array.max()),
        "range_pp": float((array.max() - array.min()) * 100),
        "std_pp": float(array.std(ddof=1) * 100) if array.size > 1 else 0.0,
        "median": float(np.median(array)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(HISTORY_DB_PATH))
    parser.add_argument("--end", default=DEV_END.isoformat())
    parser.add_argument("--perturbations", type=int, default=DEFAULT_PERTURBATIONS)
    parser.add_argument("--drop", type=int, default=DEFAULT_DROP)
    parser.add_argument(
        "--output", type=Path, default=Path("reports/path_dependence_dev.json")
    )
    args = parser.parse_args()

    end = date.fromisoformat(args.end)
    if end > DEV_END:
        parser.error("禁令 6：不得載入 2024+；--end 最晚為 2023-12-29")
    if args.perturbations < 1:
        parser.error("--perturbations 至少為 1")

    db_path = Path(args.db)
    started = time.time()

    import sqlite3

    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        members = [
            row[0]
            for row in con.execute(
                "SELECT DISTINCT stock_id FROM universe_history "
                "WHERE basis = ? ORDER BY stock_id",
                (DEFAULT_UNIVERSE_BASIS,),
            )
        ]
    finally:
        con.close()
    load_ids = sorted(set(members) | set(ETF_BENCHMARKS))
    print(f"標的 {len(members)} 檔｜載入 {len(load_ids)} 檔", flush=True)

    prices = load_prices(
        load_ids, start=date(2015, 1, 1), end=end, adjusted=True, db_path=db_path
    )
    chips = load_chips(members, start=date(2015, 1, 1), end=end, db_path=db_path)
    by_stock = build_dataset(load_ids, prices, chips).by_stock
    calendar = trading_calendar(by_stock)
    decision_dates = calendar[WARMUP_DAYS::DECISION_STRIDE]
    close_lookup = make_price_lookup(by_stock)
    execution_lookup = make_price_lookup(by_stock, column="open", exact=True)
    tiers = infer_tiers(db_path, UNIVERSE_SIZE, DEFAULT_UNIVERSE_BASIS)
    print(f"日曆 {len(calendar)}｜決策日 {len(decision_dates)}"
          f"｜載入完成 {time.time() - started:.0f}s", flush=True)

    # 逐檔預算，與標的池成員無關 → 只做一次
    scores = precompute_scores(by_stock)
    print(f"分數預算完成 {time.time() - started:.0f}s", flush=True)
    outcomes = precompute_outcomes(by_stock, HORIZON, decision_dates)
    print(f"標記預算完成 {time.time() - started:.0f}s", flush=True)

    baseline_members = resolve_members(
        decision_dates, db_path, UNIVERSE_SIZE, DEFAULT_UNIVERSE_BASIS
    )
    if not baseline_members:
        raise DiagnosticError("解析不到任何決策日的標的池")

    families: dict[str, object] = {}
    for family in STRATEGY_FAMILIES:
        runs: list[Run] = []
        for index in range(args.perturbations + 1):
            members_at = perturb(baseline_members, drop=args.drop, seed=index)
            by_date = enumerate_decisions(
                outcomes, scores[family.name], decision_dates, members_at
            )
            raw_picks, _, oos_start, _, _ = run_walk_forward(
                by_date, tiers, np.random.default_rng(RANDOM_SEED), None, HORIZON
            )
            if oos_start is None:
                continue
            oos_calendar = [day for day in calendar if day >= oos_start]
            raw_signals = to_signals(raw_picks, calendar, tiers)
            for scheme in (SLOT, REBALANCE):
                total, n_trades, blocked, trades = _run_scheme(
                    scheme, raw_signals, oos_calendar,
                    close_lookup, execution_lookup,
                )
                runs.append(Run(scheme, index, total, n_trades, blocked, trades))
            print(f"  {family.name}｜擾動 {index}"
                  f"｜{time.time() - started:.0f}s", flush=True)

        by_scheme: dict[str, object] = {}
        for scheme in (SLOT, REBALANCE):
            picked = [r for r in runs if r.scheme == scheme]
            if not picked:
                continue
            base = next((r for r in picked if r.perturbation == 0), None)
            shifted = [r for r in picked if r.perturbation != 0]
            overlaps = (
                [divergence(base.trades, r.trades).jaccard for r in shifted]
                if base is not None else []
            )
            by_scheme[scheme] = {
                "baseline_total_return": base.total_return if base else None,
                "signal_trade_ratio": base.signal_trade_ratio if base else None,
                "n_trades": base.n_trades if base else None,
                "perturbed_totals": [r.total_return for r in shifted],
                "spread": _spread([r.total_return for r in picked]),
                "fill_overlap_vs_baseline": overlaps,
                "median_fill_overlap": (
                    float(np.median(overlaps)) if overlaps else None
                ),
            }
        families[family.name] = by_scheme

    payload: dict[str, object] = {
        "end": args.end,
        "horizon": HORIZON,
        "n_slots": N_SLOTS,
        "universe_size": UNIVERSE_SIZE,
        "perturbations": args.perturbations,
        "drop_per_date": args.drop,
        "decision_dates": len(decision_dates),
        "elapsed_seconds": round(time.time() - started, 1),
        "families": families,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    print(f"\n=== 同一擾動下的擺盪幅度"
          f"（每決策日移掉 {args.drop} 檔 × {args.perturbations} 次）===")
    print(f"{'策略族':12s} {'方案':10s} {'基準報酬':>10s} {'訊號/成交':>10s}"
          f" {'全距 pp':>9s} {'標準差 pp':>10s} {'成交重疊':>9s}")
    for name, schemes in families.items():
        for scheme, entry in schemes.items():  # type: ignore[union-attr]
            overlap = entry["median_fill_overlap"]
            overlap_text = f"{overlap:8.1%}" if overlap is not None else "       —"
            print(f"{name[:10]:12s} {scheme:10s}"
                  f" {entry['baseline_total_return']:+9.2%}"
                  f" {entry['signal_trade_ratio']:9.1f}"
                  f" {entry['spread']['range_pp']:+8.1f}"
                  f" {entry['spread']['std_pp']:9.1f}"
                  f" {overlap_text}")
    print(f"\n已寫入 {args.output}｜耗時 {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
