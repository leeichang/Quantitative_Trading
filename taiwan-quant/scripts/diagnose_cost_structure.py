#!/usr/bin/env python3
"""工作單 11：融資事件的成本結構、門檻形狀與跌深均值回歸。"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.diagnose_event_reactions import (  # noqa: E402
    MIN_OBSERVATIONS,
    REFRESH_EVERY,
    WARMUP,
    build_universe_mask,
)
from scripts.validate_oos_trailing import resolve_members  # noqa: E402
from taiwan_quant.config.costs import (  # noqa: E402
    DEFAULT,
    FEE_TAX_ONLY,
    GROSS,
    CostModel,
    resolve_tier,
)
from taiwan_quant.data.etf_universe import is_etf  # noqa: E402
from taiwan_quant.data.integrity import DataIntegrityError, forward_returns  # noqa: E402
from taiwan_quant.data.loader import (  # noqa: E402
    DEFAULT_UNIVERSE_BASIS,
    HISTORY_DB_PATH,
    RAW_OPEN_COLUMN,
    load_chips,
    load_prices,
)
from taiwan_quant.validation.bootstrap import block_bootstrap  # noqa: E402
from taiwan_quant.validation.event_study import (  # noqa: E402
    capacity_constrained_fills,
    capacity_economics,
    expanding_quantile_mask,
)

DEV_END = date(2023, 12, 29)
START = date(2015, 1, 1)
CAPITAL = 400_000.0
HOLDING_DAYS = 40
UNIVERSE_SIZE = 150
LARGE_SIZE = 50
N_VALUES = (3, 5, 10, 20)
QUANTILES = (0.990, 0.995, 0.998)
AE_HORIZONS = (60, 120)
SEED = 20260919
N_DRAWS = 2000
PRIOR_CELLS = 62


def _pivot(prices: pd.DataFrame, column: str) -> pd.DataFrame:
    return prices[column].unstack("stock_id").sort_index()


def _cost_rates(
    *,
    fills: pd.DataFrame,
    adjusted_opens: pd.DataFrame,
    raw_opens: pd.DataFrame,
    large_mask: pd.DataFrame,
    amount: float,
    model: CostModel,
) -> pd.DataFrame:
    """只為實際成交部位計算 T+1 開盤時的逐檔來回成本率。"""
    rates = pd.DataFrame(np.nan, index=fills.index, columns=fills.columns)
    positions = {day: index for index, day in enumerate(fills.index)}
    for day, stock_id in zip(*np.where(fills.to_numpy(dtype=bool)), strict=True):
        decision = fills.index[day]
        entry_position = positions[decision] + 1
        if entry_position >= len(fills.index):
            raise DataIntegrityError(f"{decision.date()} 沒有 T+1 進場日")
        entry = fills.index[entry_position]
        adjusted = float(adjusted_opens.at[entry, fills.columns[stock_id]])
        actual = float(raw_opens.at[entry, fills.columns[stock_id]])
        if not np.isfinite(adjusted) or not np.isfinite(actual) or actual <= 0:
            raise DataIntegrityError(
                f"{fills.columns[stock_id]} 在 {entry.date()} 缺少可成交開盤價"
            )
        sid = str(fills.columns[stock_id])
        tier = resolve_tier(
            actual_price=actual,
            adjusted_price=adjusted,
            amount=amount,
            large=bool(large_mask.at[decision, sid]),
            is_etf=is_etf(sid),
        )
        rates.at[decision, sid] = model.round_trip_cost(amount, tier) / amount
    return rates


def _evaluate(
    *,
    label: str,
    event_mask: pd.DataFrame,
    scores: pd.DataFrame,
    universe: pd.DataFrame,
    large_mask: pd.DataFrame,
    adjusted_opens: pd.DataFrame,
    raw_opens: pd.DataFrame,
    returns: pd.DataFrame,
    n_slots: int,
    holding_days: int,
) -> dict[str, object]:
    eligible = event_mask.astype(bool) & universe.astype(bool) & returns.notna()
    capacity = capacity_constrained_fills(
        eligible,
        scores,
        n_slots=n_slots,
        holding_days=holding_days,
    )
    amount = CAPITAL / n_slots
    economics: dict[str, object] = {}
    for name, model in (
        ("gross", GROSS),
        ("fee_tax", FEE_TAX_ONLY),
        ("default", DEFAULT),
    ):
        costs = _cost_rates(
            fills=capacity.fills,
            adjusted_opens=adjusted_opens,
            raw_opens=raw_opens,
            large_mask=large_mask,
            amount=amount,
            model=model,
        )
        result = capacity_economics(
            returns=returns,
            fills=capacity.fills,
            universe_mask=universe,
            cost_rates=costs,
        )
        economics[name] = {
            "cost_per_trip": result.cost_per_trip,
            "net_per_trip": result.net_per_trip,
            "net_standard_deviation": result.net_standard_deviation,
            "annualized_net": (1.0 + result.net_per_trip) ** (252 / holding_days) - 1.0,
            "per_period": list(result.net_paired),
        }
    default_costs = _cost_rates(
        fills=capacity.fills,
        adjusted_opens=adjusted_opens,
        raw_opens=raw_opens,
        large_mask=large_mask,
        amount=amount,
        model=DEFAULT,
    )
    base = capacity_economics(
        returns=returns,
        fills=capacity.fills,
        universe_mask=universe,
        cost_rates=default_costs,
    )
    boot = block_bootstrap(
        np.asarray(base.gross_paired),
        np.mean,
        block_length=1,
        n_draws=N_DRAWS,
        seed=SEED,
    )
    return {
        "label": label,
        "n_slots": n_slots,
        "holding_days": holding_days,
        "amount_per_position": amount,
        "n_offered": capacity.n_offered,
        "n_filled": capacity.n_filled,
        "capture_rate": capacity.capture_rate,
        "theoretical_capture_rate": min(
            1.0,
            (n_slots / holding_days)
            / (capacity.n_offered / max(1, int(eligible.any(axis=1).sum()))),
        ),
        "gross_excess": base.gross_excess,
        "gross_bootstrap_lower": boot.lower,
        "gross_bootstrap_upper": boot.upper,
        "gross_per_period": list(base.gross_paired),
        "cost_sensitivity": economics,
    }


def run(db_path: Path, end: date) -> dict[str, object]:
    started = time.time()
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as con:
        members = [
            row[0]
            for row in con.execute(
                "SELECT DISTINCT stock_id FROM universe_history "
                "WHERE basis = ? ORDER BY stock_id",
                (DEFAULT_UNIVERSE_BASIS,),
            )
        ]
    prices = load_prices(members, start=START, end=end, adjusted=True, db_path=db_path)
    chips = load_chips(members, start=START, end=end, db_path=db_path)
    opens = _pivot(prices, "open")
    raw_opens = _pivot(prices, RAW_OPEN_COLUMN).reindex_like(opens)
    closes = _pivot(prices, "close").reindex_like(opens)
    volumes = _pivot(prices, "volume").reindex_like(opens)
    calendar = list(opens.index)
    snapshot_dates = calendar[WARMUP::20]
    universe_at = resolve_members(
        snapshot_dates, db_path, UNIVERSE_SIZE, DEFAULT_UNIVERSE_BASIS
    )
    large_at = resolve_members(
        snapshot_dates, db_path, LARGE_SIZE, DEFAULT_UNIVERSE_BASIS
    )
    universe = build_universe_mask(opens.index, opens.columns, universe_at)
    large_mask = build_universe_mask(opens.index, opens.columns, large_at)
    universe.iloc[:WARMUP] = False
    large_mask.iloc[:WARMUP] = False

    margin = chips["margin_balance"].unstack("stock_id").reindex_like(opens)
    margin_strength = (
        margin.diff() / margin.rolling(20).mean().abs().replace(0, np.nan)
    )
    masks = {
        q: expanding_quantile_mask(
            margin_strength,
            quantile=q,
            refresh_every=REFRESH_EVERY,
            min_observations=MIN_OBSERVATIONS,
        )
        for q in QUANTILES
    }
    forward40 = forward_returns(opens, closes, holding_days=HOLDING_DAYS)

    aa = [
        _evaluate(
            label=f"margin_q99_n{n}",
            event_mask=masks[0.99],
            scores=margin_strength,
            universe=universe,
            large_mask=large_mask,
            adjusted_opens=opens,
            raw_opens=raw_opens,
            returns=forward40,
            n_slots=n,
            holding_days=HOLDING_DAYS,
        )
        for n in N_VALUES
    ]
    ab = [
        _evaluate(
            label=f"margin_q{q:.3f}_n10",
            event_mask=masks[q],
            scores=margin_strength,
            universe=universe,
            large_mask=large_mask,
            adjusted_opens=opens,
            raw_opens=raw_opens,
            returns=forward40,
            n_slots=10,
            holding_days=HOLDING_DAYS,
        )
        for q in QUANTILES
    ]

    daily_return = closes.pct_change()
    volume_ratio = volumes / volumes.rolling(20).mean().replace(0, np.nan)
    down_event = (daily_return < -0.05) & (volume_ratio > 3.0)
    down_strength = (-daily_return) * volume_ratio
    ae2 = [
        _evaluate(
            label=f"down5_volume3_h{horizon}_n10",
            event_mask=down_event,
            scores=down_strength,
            universe=universe,
            large_mask=large_mask,
            adjusted_opens=opens,
            raw_opens=raw_opens,
            returns=forward_returns(opens, closes, holding_days=horizon),
            n_slots=10,
            holding_days=horizon,
        )
        for horizon in AE_HORIZONS
    ]
    added_cells = len(N_VALUES) + len(QUANTILES) + len(AE_HORIZONS)
    cumulative_cells = PRIOR_CELLS + added_cells
    return {
        "end": end.isoformat(),
        "capital": CAPITAL,
        "seed": SEED,
        "n_draws": N_DRAWS,
        "added_cells": added_cells,
        "cumulative_cells": cumulative_cells,
        "gumbel_threshold": float(np.sqrt(2.0 * np.log(cumulative_cells))),
        "AA_cost_vs_concentration": aa,
        "AB_quantile_shape": ab,
        "AE2_down_volume_mean_reversion": ae2,
        "AE1_foreign_short": {
            "status": "not_run",
            "reason": "專案沒有實際借券費率；工作單禁止用猜測成本建模",
        },
        "elapsed_seconds": round(time.time() - started, 1),
    }


def _print_rows(title: str, rows: list[dict[str, object]]) -> None:
    print(f"\n{title}")
    print("設定                         成交/事件   抓取率   毛超額     95% 區間"
          "        成本/趟    淨/趟    淨sd")
    for row in rows:
        default = row["cost_sensitivity"]["default"]
        print(
            f"{row['label']:<28} {row['n_filled']:>4}/{row['n_offered']:<5}"
            f" {row['capture_rate']:>7.1%} {row['gross_excess']:>+8.3%}"
            f" [{row['gross_bootstrap_lower']:>+7.3%},{row['gross_bootstrap_upper']:>+7.3%}]"
            f" {default['cost_per_trip']:>8.3%} {default['net_per_trip']:>+8.3%}"
            f" {default['net_standard_deviation']:>7.3%}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=HISTORY_DB_PATH)
    parser.add_argument("--end", type=date.fromisoformat, default=DEV_END)
    parser.add_argument(
        "--out", type=Path, default=Path("reports/cost_structure_dev.json")
    )
    args = parser.parse_args()
    if args.end > DEV_END:
        parser.error(f"禁令 6：--end 最晚為 {DEV_END}")
    payload = run(args.db, args.end)
    _print_rows("AA 成本結構 vs 集中度", payload["AA_cost_vs_concentration"])
    _print_rows("AB 事件門檻形狀", payload["AB_quantile_shape"])
    _print_rows("AE-2 跌深爆量容量受限", payload["AE2_down_volume_mean_reversion"])
    print(
        f"\n新增 {payload['added_cells']} 格｜累計 {payload['cumulative_cells']} 格"
        f"｜Gumbel |t| 門檻 {payload['gumbel_threshold']:.2f}"
    )
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"已寫入 {args.out}")


if __name__ == "__main__":
    main()
