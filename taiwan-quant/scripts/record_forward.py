#!/usr/bin/env python3
"""產生不可回寫的前推預測，或結算已揭曉的移動停損報酬。"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from weekly_plan_trailing import (  # noqa: E402
    CALIBRATION_BINS,
    CALIBRATION_MIN_SAMPLES,
    CORRELATION_WINDOW,
    INDUSTRY_MAP,
    STRATEGY_VERSION,
    build_calibration_set,
    build_signal,
    compute_beta,
    compute_correlations,
)

from taiwan_quant.config.costs import Tier  # noqa: E402
from taiwan_quant.data.dataset import build_dataset  # noqa: E402
from taiwan_quant.data.loader import (  # noqa: E402
    FROZEN_DATA_START,
    HISTORY_DB_PATH,
    load_chips,
    load_prices,
    load_universe_at,
)
from taiwan_quant.features.technical import build_technical  # noqa: E402
from taiwan_quant.forward_predictions import (  # noqa: E402
    ForwardPrediction,
    list_unsettled_predictions,
    record_forward_predictions,
    settle_forward_prediction,
)
from taiwan_quant.labeling.trailing_stop import label_trailing  # noqa: E402
from taiwan_quant.ranking.trailing_portfolio import (  # noqa: E402
    DEFAULT_EDGE_Z,
    TrailingCandidate,
    select_trailing_portfolio,
)
from taiwan_quant.strategies.families import STRATEGY_FAMILIES  # noqa: E402
from taiwan_quant.validation.calibration import (  # noqa: E402
    CalibrationError,
    fit_return_calibrator,
)


def latest_price_date(db_path: Path) -> date:
    """查詢資料截止日；只讀 metadata，不繞過價格凍結守門。"""
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as con:
        value = con.execute("SELECT MAX(date) FROM stock_daily").fetchone()[0]
    if value is None:
        raise RuntimeError("stock_daily 無資料")
    return date.fromisoformat(value)


def generate_predictions(
    db_path: Path,
    as_of: date,
    start: date,
    horizon: int,
    capital: float,
    family_names: list[str],
    limit: int,
    edge_z: float,
) -> list[ForwardPrediction]:
    """使用既有校準、門檻、成本與投組約束產生前推預測。"""
    universe = load_universe_at(as_of, db_path=db_path, limit=150)
    stock_ids = universe.stock_ids[:limit]
    unlock = as_of >= FROZEN_DATA_START
    prices = load_prices(
        stock_ids,
        start=start,
        end=as_of,
        db_path=db_path,
        adjusted=True,
        unlock_frozen=unlock,
        frozen_reason="record_forward 產生前推預測" if unlock else None,
    )
    chips = load_chips(
        stock_ids,
        start=start,
        end=as_of,
        db_path=db_path,
        unlock_frozen=unlock,
        frozen_reason="record_forward 產生前推預測" if unlock else None,
    )
    by_stock = build_dataset(stock_ids, prices, chips).by_stock
    tiers = {sid: Tier(universe.tier_of(sid)) for sid in by_stock}
    returns_by_stock = {sid: bars["close"].pct_change() for sid, bars in by_stock.items()}
    market_returns = pd.DataFrame(returns_by_stock).mean(axis=1)

    atr_ratios = {
        sid: float(value)
        for sid, bars in by_stock.items()
        if np.isfinite(value := build_technical(bars)["true_range_ratio_14"].iloc[-1])
    }
    atr_percentiles = (
        pd.Series(atr_ratios).rank(pct=True).to_dict() if atr_ratios else {}
    )
    train_end = pd.Timestamp(as_of) - pd.Timedelta(days=horizon * 2)
    predicted_at = datetime.now().astimezone().isoformat()
    due_date = (pd.Timestamp(as_of) + pd.offsets.BDay(horizon)).date().isoformat()
    output: list[ForwardPrediction] = []

    families = [family for family in STRATEGY_FAMILIES if family.name in family_names]
    unknown = sorted(set(family_names) - {family.name for family in families})
    if unknown:
        raise ValueError(f"未知策略族：{unknown}")

    for family in families:
        scores, returns = build_calibration_set(by_stock, family, horizon, train_end)
        try:
            calibrator = fit_return_calibrator(
                scores,
                returns,
                n_bins=CALIBRATION_BINS,
                min_samples_per_bin=CALIBRATION_MIN_SAMPLES,
            )
        except CalibrationError as exc:
            print(f"跳過 {family.name}：校準失敗：{exc}")
            continue

        signals = []
        for sid, bars in by_stock.items():
            signal = build_signal(
                sid,
                bars,
                family,
                horizon,
                calibrator,
                volatility_pct=atr_percentiles.get(sid, 0.5),
                beta=compute_beta(
                    returns_by_stock[sid], market_returns, CORRELATION_WINDOW
                ),
            )
            if signal is not None:
                signals.append(signal)
        signal_by_id = {signal.stock_id: signal for signal in signals}
        candidates = [
            TrailingCandidate(
                stock_id=signal.stock_id,
                expected_gross_return=signal.expected_gross_return,
                return_std=signal.return_std,
                n_samples=signal.n_samples,
                trail_pct=signal.trail_pct,
                entry_price=signal.entry_price,
                tier=tiers.get(signal.stock_id, Tier.MID),
                industry=INDUSTRY_MAP.get(signal.stock_id, "其他"),
                volatility_pct=signal.volatility_pct,
                beta=signal.beta,
                max_horizon=horizon,
            )
            for signal in signals
        ]
        correlations = compute_correlations(
            {sid: returns_by_stock[sid] for sid in signal_by_id},
            CORRELATION_WINDOW,
        )
        result = select_trailing_portfolio(
            candidates, capital, correlations, edge_z=edge_z
        )
        for rank, position in enumerate(result.positions, start=1):
            candidate = position.candidate
            signal = signal_by_id[candidate.stock_id]
            output.append(ForwardPrediction(
                predicted_at=predicted_at,
                data_asof=as_of.isoformat(),
                strategy_version=STRATEGY_VERSION,
                family=family.name,
                horizon=horizon,
                stock_id=candidate.stock_id,
                rank=rank,
                score=signal.score,
                expected_return=candidate.expected_gross_return,
                trail_pct=candidate.trail_pct,
                entry_price=candidate.entry_price,
                due_date=due_date,
            ))
    return output


def settle_predictions(db_path: Path) -> tuple[int, int]:
    """只結算已有足夠未來交易日的預測。"""
    pending = list_unsettled_predictions(db_path)
    if not pending:
        return 0, 0
    latest = latest_price_date(db_path)
    stock_ids = sorted({item.stock_id for item in pending})
    earliest = min(date.fromisoformat(item.data_asof) for item in pending)
    prices = load_prices(
        stock_ids,
        start=earliest,
        end=latest,
        db_path=db_path,
        adjusted=True,
        unlock_frozen=latest >= FROZEN_DATA_START,
        frozen_reason="record_forward --settle 回填實現報酬",
    )
    settled = 0
    for item in pending:
        try:
            bars = prices.xs(item.stock_id, level="stock_id")
        except KeyError:
            continue
        positions = np.flatnonzero(bars.index <= pd.Timestamp(item.data_asof))
        if not len(positions):
            continue
        decision_idx = int(positions[-1])
        if len(bars) - decision_idx - 1 < item.horizon:
            continue
        outcome = label_trailing(
            bars,
            decision_idx=decision_idx,
            trail_pct=item.trail_pct,
            max_horizon=item.horizon,
        )
        if outcome is None:
            continue
        if settle_forward_prediction(
            db_path,
            item,
            realized_return=outcome.gross_return,
            settled_at=datetime.now(UTC),
        ):
            settled += 1
    return settled, len(pending)


def main() -> None:
    parser = argparse.ArgumentParser(description="記錄或結算不可回寫的前推預測")
    parser.add_argument("--settle", action="store_true", help="結算已到期預測")
    parser.add_argument("--db", type=Path, default=HISTORY_DB_PATH)
    parser.add_argument("--as-of", default=None, help="資料截止日；預設資料庫最新日")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--capital", type=float, default=400_000.0)
    parser.add_argument("--limit", type=int, default=150)
    parser.add_argument("--edge-z", type=float, default=DEFAULT_EDGE_Z)
    parser.add_argument(
        "--family",
        action="append",
        dest="families",
        help="可重複指定；預設記錄全部三個策略族",
    )
    args = parser.parse_args()

    if args.settle:
        settled, pending = settle_predictions(args.db)
        print(f"已結算 {settled} / {pending} 筆待結算預測")
        return

    as_of = date.fromisoformat(args.as_of) if args.as_of else latest_price_date(args.db)
    names = args.families or [family.name for family in STRATEGY_FAMILIES]
    predictions = generate_predictions(
        args.db,
        as_of,
        date.fromisoformat(args.start),
        args.horizon,
        args.capital,
        names,
        args.limit,
        args.edge_z,
    )
    inserted = record_forward_predictions(args.db, predictions)
    print(f"產生 {len(predictions)} 筆，新增 {inserted} 筆前推預測")
    for item in predictions:
        print(
            f"  {item.family} #{item.rank} {item.stock_id}｜"
            f"分數 {item.score:.4f}｜預計揭曉 {item.due_date}"
        )


if __name__ == "__main__":
    main()
