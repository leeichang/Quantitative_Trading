#!/usr/bin/env python3
"""
Walk-Forward 樣本外驗證（移動停損版，路線 A）

## 要回答的問題

前一版（`validate_oos.py`）的結論是：

    9 組 triple-barrier 策略全部輸給等權買進持有，差距 85 個百分點。
    機制原因：目標價把上檔封死。

路線 A 拿掉目標價，改成移動停損。**這個腳本就是驗證那個假設。**

    如果移動停損仍然輸給買進持有 → 問題不在目標價，在選股本身。
    如果移動停損跟上了 → 假設成立，機制解釋正確。

## 與前一版的差別

只有三處：

    標記    label_trailing         取代 label_one
    寬度    derive_trail_width     取代 derive_width
    門檻    E[毛報酬] − cost > 0   取代 P(+1) ≥ (stop+cost)/(target+stop)

Walk-Forward 切分、對照組、成本模型、持倉不重疊規則完全沿用——
換了條件就不可比。

## 對照組（CLAUDE.md 必跑）

    隨機進場      每個決策日隨機挑 3 檔，用同一組移動停損
    等權買進持有   期初買進全標的池等權持有到期末

用法：
    .venv/bin/python scripts/validate_oos_trailing.py
    .venv/bin/python scripts/validate_oos_trailing.py --horizons 40 60 --limit 40
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from taiwan_quant.backtest.portfolio_sim import (  # noqa: E402
    PriceLookup,
    Signal,
    simulate_portfolio,
)
from taiwan_quant.config.costs import DEFAULT, Tier  # noqa: E402
from taiwan_quant.data.dataset import build_dataset  # noqa: E402
from taiwan_quant.data.loader import (  # noqa: E402
    DEFAULT_UNIVERSE_BASIS,
    HISTORY_DB_PATH,
    UNIVERSE_BASES,
    DataNotAvailableError,
    load_chips,
    load_prices,
    load_universe_at,
)
from taiwan_quant.labeling.trail_width import derive_trail_width  # noqa: E402
from taiwan_quant.labeling.trailing_stop import label_trailing  # noqa: E402
from taiwan_quant.ranking.trailing_portfolio import DEFAULT_EDGE_Z  # noqa: E402
from taiwan_quant.strategies.families import STRATEGY_FAMILIES  # noqa: E402
from taiwan_quant.validation.benchmarks import (  # noqa: E402
    ETF_BENCHMARKS,
    effective_samples,
    equal_weight_equity,
    etf_benchmark_curves,
)
from taiwan_quant.validation.binning import find_bin  # noqa: E402
from taiwan_quant.validation.calibration import (  # noqa: E402
    CalibrationError,
    fit_return_calibrator,
)
from taiwan_quant.validation.stats import deflated_sharpe_ratio  # noqa: E402
from taiwan_quant.validation.walk_forward import walk_forward_folds  # noqa: E402

PULLBACK_QUANTILE = 0.80
"""移動停損取歷史最大回落的第幾分位。見 labeling/trail_width.py"""

CALIBRATION_BINS = 6
CALIBRATION_MIN_SAMPLES = 50

WARMUP_DAYS = 250
DECISION_STRIDE = 5
"""週頻決策：每 5 個交易日一次"""

TOP_N = 3
TEST_WINDOW_DAYS = 60

FIRST_TRAIN_DAYS = 750
"""
第一個 fold 的訓練期長度（交易日，約 3 年）。

資料回補到 2015 之後這個妥協解除了：

    2,850 個交易日 − 暖機 250 − 訓練 750 = OOS 約 1,850 天 ≈ 7.3 年

涵蓋 2018 貿易戰、2020 疫情崩跌、2022 空頭——正是先前每一條限制
都指向的缺口。
"""

UNIVERSE_SIZE = 150
"""每個決策日取當時標的池的前幾名（D2 要求 150 檔）"""

RANDOM_SEED = 20260912
TRADING_DAYS_PER_YEAR = 252.0


@dataclass(frozen=True)
class Decision:
    """一筆決策及其實際結果"""

    decision_date: pd.Timestamp
    stock_id: str
    score: float
    trail_pct: float
    gross_return: float
    holding_days: int
    exit_reason: str
    highest_price: float
    """期間最高價，用來檢查移動停損到底跟到多高"""


def trading_calendar(by_stock: dict[str, pd.DataFrame]) -> list[pd.Timestamp]:
    """
    全標的共用的交易日曆（所有標的日期的聯集，升冪）。

    **決策日必須來自共用日曆，不可用各檔的位置索引。**

    實測 bug：原本各檔跑 `range(WARMUP, len(bars), 5)`，但各檔起始日
    不同，同一個位置索引落在不同日曆日。結果同一決策日的候選數是
    [1, 2, 36, 35, 1, 3, 35, 1, 6]——候選只有 1 檔時「挑 Top 3」等於
    沒有挑，與隨機對照 100% 重疊卻看起來贏了買進持有。
    """
    dates: set[pd.Timestamp] = set()
    for bars in by_stock.values():
        dates.update(bars.index)
    return sorted(dates)


@dataclass(frozen=True)
class Outcome:
    """單一 (標的, 決策日, horizon) 的移動停損結果。與策略族無關"""

    trail_pct: float
    gross_return: float
    holding_days: int
    exit_reason: str
    highest_price: float


def precompute_scores(
    by_stock: dict[str, pd.DataFrame],
) -> dict[str, dict[str, pd.Series]]:
    """
    一次算完所有 (策略族 × 標的) 的分數序列。

    分數**不依賴 horizon**，所以三個 horizon 共用同一份。
    向量化版比逐點呼叫快 54~85 倍（實測），是 11 年 × 586 檔可行的前提。
    """
    out: dict[str, dict[str, pd.Series]] = {}
    for family in STRATEGY_FAMILIES:
        series_by_stock: dict[str, pd.Series] = {}
        for stock_id, bars in by_stock.items():
            if not set(family.required_columns).issubset(bars.columns):
                continue
            series_by_stock[stock_id] = family.score_series(bars)
        out[family.name] = series_by_stock
    return out


def precompute_outcomes(
    by_stock: dict[str, pd.DataFrame],
    horizon: int,
    decision_dates: list[pd.Timestamp],
) -> dict[str, dict[pd.Timestamp, Outcome]]:
    """
    一次算完所有 (標的, 決策日) 的停損寬度與實際結果。

    **與策略族無關**，所以三族共用——否則同一份 width + label 會被
    重算三次（實測佔總成本約一半）。
    """
    out: dict[str, dict[pd.Timestamp, Outcome]] = {}
    for stock_id, bars in by_stock.items():
        positions = {d: i for i, d in enumerate(bars.index)}
        per_date: dict[pd.Timestamp, Outcome] = {}
        for decision_date in decision_dates:
            idx = positions.get(decision_date)
            if idx is None or idx < WARMUP_DAYS:
                continue

            width = derive_trail_width(
                bars, decision_idx=idx, horizon=horizon,
                pullback_quantile=PULLBACK_QUANTILE,
            )
            if width is None:
                continue

            result = label_trailing(
                bars, decision_idx=idx,
                trail_pct=width.trail_pct, max_horizon=horizon,
            )
            if result is None:
                continue

            per_date[decision_date] = Outcome(
                trail_pct=width.trail_pct,
                gross_return=result.gross_return,
                holding_days=result.holding_days,
                exit_reason=result.exit_reason,
                highest_price=result.highest_price,
            )
        if per_date:
            out[stock_id] = per_date
    return out


def enumerate_decisions(
    outcomes: dict[str, dict[pd.Timestamp, Outcome]],
    scores: dict[str, pd.Series],
    decision_dates: list[pd.Timestamp],
    members_at: dict[pd.Timestamp, set[str]],
) -> dict[pd.Timestamp, list[Decision]]:
    """
    組出「決策日 → 候選清單」。

    `members_at` 是**當時的標的池**（禁令 2）。只用今天的名單回溯歷史，
    2015 年的 150 檔裡會有 80 檔看不見——包含日月光、矽品這種當年的
    前 15 大。
    """
    by_date: dict[pd.Timestamp, list[Decision]] = {}

    for stock_id, per_date in outcomes.items():
        score_series = scores.get(stock_id)
        if score_series is None:
            continue

        for decision_date, outcome in per_date.items():
            if stock_id not in members_at.get(decision_date, frozenset()):
                continue
            try:
                score = float(score_series.loc[decision_date])
            except KeyError:
                continue
            if not np.isfinite(score):
                continue

            by_date.setdefault(decision_date, []).append(Decision(
                decision_date=decision_date,
                stock_id=stock_id,
                score=score,
                trail_pct=outcome.trail_pct,
                gross_return=outcome.gross_return,
                holding_days=outcome.holding_days,
                exit_reason=outcome.exit_reason,
                highest_price=outcome.highest_price,
            ))

    return by_date


def resolve_members(
    decision_dates: list[pd.Timestamp],
    db_path: Path,
    universe_size: int,
    basis: str = DEFAULT_UNIVERSE_BASIS,
) -> dict[pd.Timestamp, set[str]]:
    """
    查出每個決策日**當時**的標的池成員。

    快照是逐季的，所以同一季的決策日共用一份——先按快照日期快取，
    避免對每個決策日各查一次資料庫。
    """
    members: dict[pd.Timestamp, set[str]] = {}
    cache: dict[str, set[str]] = {}

    for decision_date in decision_dates:
        try:
            universe = load_universe_at(
                decision_date.date(), db_path=db_path,
                limit=universe_size, basis=basis,
            )
        except DataNotAvailableError:
            continue
        key = str(universe.stocks["rank"].sum()) + str(len(universe.stocks))
        ids = cache.setdefault(key, set(universe.stock_ids))
        members[decision_date] = ids

    return members


def run_walk_forward(
    by_date: dict[pd.Timestamp, list[Decision]],
    tiers: dict[str, Tier],
    rng: np.random.Generator,
    edge_z: float,
    horizon: int,
    oos_start_override: pd.Timestamp | None = None,
    dev_end: pd.Timestamp | None = None,
) -> tuple[
    list[tuple[pd.Timestamp, Decision, float]],
    list[tuple[pd.Timestamp, Decision, float]],
    pd.Timestamp | None,
    int,
    int,
]:
    """
    執行滾動 Walk-Forward，產出**所有通過門檻的訊號**。

    不在這裡挑 Top N、也不做持倉不重疊——那是組合層的事。
    `backtest/portfolio_sim.py` 會用 3 個槽位去消化這些訊號，
    重疊的部分由槽位限制自然擋掉。

    前一版在這裡硬性規定「前一筆出場後才開新倉」，因為 `engine.py`
    只容得下一個持倉。後果是 OOS 曝險只有 23~39%，而對照組買進持有
    是 100%——在多頭裡光這個差距就足以輸掉，與選股能力無關。

    Returns:
        (策略訊號, 隨機訊號, OOS 起始日, fold 數, 校準失敗的 fold 數)

        訊號為 (決策日, Decision, 排名分數)。
    """
    dates = sorted(by_date)
    if not dates:
        return [], [], None, 0, 0

    # 決策日本身已經排除暖機期（見 enumerate_decisions），
    # 所以這裡只需換算訓練期，**不可再加 WARMUP_DAYS**。
    first_train = FIRST_TRAIN_DAYS // DECISION_STRIDE
    test_span = max(1, TEST_WINDOW_DAYS // DECISION_STRIDE)

    if first_train + test_span > len(dates):
        return [], [], None, 0, 0

    strategy: list[tuple[pd.Timestamp, Decision, float]] = []
    random_signals: list[tuple[pd.Timestamp, Decision, float]] = []
    folds = 0
    calibration_failures = 0
    if oos_start_override is None:
        oos_start = dates[first_train]
    else:
        oos_start = next(
            (day for day in dates if day >= oos_start_override),
            None,
        )
        if oos_start is None:
            raise ValueError(f"OOS 起點 {oos_start_override.date()} 晚於所有決策日")

    # **標籤隔離（embargo）。** 訓練集尾端的決策，其標籤要等 horizon
    # 天後才揭曉——那些天落在測試段內，等於讓校準器看過測試期的走勢。
    # 實測污染比例 8.0%（第一個 fold）~ 2.4%（最後一個）。
    # 見 taiwan_quant/validation/walk_forward.py
    for train_dates, test_dates in walk_forward_folds(
        dates,
        first_train=first_train,
        test_span=test_span,
        horizon=horizon,
        stride=DECISION_STRIDE,
        oos_start=oos_start_override,
        dev_end=dev_end,
    ):
        folds += 1

        train_decisions = [d for dt in train_dates for d in by_date[dt]]
        try:
            calibrator = fit_return_calibrator(
                np.array([d.score for d in train_decisions]),
                np.array([d.gross_return for d in train_decisions]),
                n_bins=CALIBRATION_BINS,
                min_samples_per_bin=CALIBRATION_MIN_SAMPLES,
            )
        except CalibrationError:
            calibration_failures += 1
            continue

        # 每箱的標準誤，用來判斷優勢是否顯著
        se_by_bin = {
            i: (b.return_std / np.sqrt(b.n_samples) if b.return_std else None)
            for i, b in enumerate(calibrator.bins)
        }

        for decision_date in test_dates:
            candidates = by_date[decision_date]
            if not candidates:
                continue

            for d in candidates:
                expected = calibrator.predict(d.score)
                if expected is None:
                    continue

                cost = DEFAULT.round_trip_rate(tiers.get(d.stock_id, Tier.MID))
                net = expected - cost
                if net <= 0:
                    continue

                # 優勢須大於一個標準誤，否則與 0 在統計上分不開
                bin_index = find_bin(d.score, calibrator.bins)
                se = se_by_bin.get(bin_index) if bin_index is not None else None
                if se is None or net < edge_z * se:
                    continue

                strategy.append((decision_date, d, net))

            # 隨機對照在**每個決策日**都嘗試進場，與策略面對同樣的
            # 槽位競爭。給隨機分數，讓組合層的排序也是隨機的。
            for d in candidates:
                random_signals.append((decision_date, d, float(rng.random())))

    return strategy, random_signals, oos_start, folds, calibration_failures


def to_signals(
    picks: list[tuple[pd.Timestamp, Decision, float]],
    calendar: list[pd.Timestamp],
    tiers: dict[str, Tier],
) -> list[Signal]:
    """把決策轉成組合模擬器的訊號（出場日由持有天數推算）"""
    position = {d: i for i, d in enumerate(calendar)}
    signals: list[Signal] = []
    for decision_date, d, rank in picks:
        start = position.get(decision_date)
        if start is None:
            continue
        exit_idx = min(start + d.holding_days, len(calendar) - 1)
        signals.append(Signal(
            decision_date=decision_date,
            exit_date=calendar[exit_idx],
            stock_id=d.stock_id,
            gross_return=d.gross_return,
            rank_score=rank,
            tier=tiers.get(d.stock_id, Tier.MID),
        ))
    return signals


def make_price_lookup(by_stock: dict[str, pd.DataFrame]) -> PriceLookup:
    """以收盤價逐日標記市值；缺當日報價時沿用前一日"""
    tables = {
        sid: bars["close"].astype(float) for sid, bars in by_stock.items()
    }

    def lookup(stock_id: str, day: pd.Timestamp) -> float | None:
        series = tables.get(stock_id)
        if series is None:
            return None
        window = series.loc[series.index <= day]
        if window.empty:
            return None
        value = float(window.iloc[-1])
        return value if np.isfinite(value) and value > 0 else None

    return lookup


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Walk-Forward 樣本外驗證（移動停損，長歷史）")
    parser.add_argument("--horizons", type=int, nargs="*", default=[20, 40, 60])
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2026-09-11")
    parser.add_argument(
        "--dev-end",
        default=None,
        help="開發集結束日（ISO）。之後的資料只在 --oos-start 指定時使用",
    )
    parser.add_argument(
        "--oos-start",
        default=None,
        help="OOS 起始日（ISO）。指定時覆蓋由 FIRST_TRAIN_DAYS 推導的起點",
    )
    parser.add_argument("--edge-z", type=float, default=DEFAULT_EDGE_Z)
    parser.add_argument("--slots", type=int, default=TOP_N)
    parser.add_argument("--universe-size", type=int, default=UNIVERSE_SIZE)
    parser.add_argument("--basis", default=DEFAULT_UNIVERSE_BASIS,
                        choices=list(UNIVERSE_BASES),
                        help="標的池排名依據。市值版命中真實成分股 94.7%%、"
                             "成交金額版 72.7%%")
    parser.add_argument("--db", default=str(HISTORY_DB_PATH))
    parser.add_argument(
        "--unlock-frozen",
        action="store_true",
        help="明確允許評估 2026-09-14 起的凍結資料（必須同時提供理由）",
    )
    parser.add_argument(
        "--frozen-reason",
        default=None,
        help="解鎖凍結資料的稽核理由",
    )
    args = parser.parse_args()

    dev_end = pd.Timestamp(date.fromisoformat(args.dev_end)) if args.dev_end else None
    oos_start_override = (
        pd.Timestamp(date.fromisoformat(args.oos_start)) if args.oos_start else None
    )
    if dev_end is not None and oos_start_override is None:
        parser.error("--dev-end 必須與 --oos-start 一起使用")
    if dev_end is not None and dev_end >= oos_start_override:
        parser.error("--dev-end 必須早於 --oos-start")
    if args.unlock_frozen and not (args.frozen_reason or "").strip():
        parser.error("--unlock-frozen 必須同時提供非空白的 --frozen-reason")

    db_path = Path(args.db)

    print("=" * 112)
    print("Walk-Forward 樣本外驗證 — 移動停損（路線 A）｜長歷史 + 逐季標的池")
    print("=" * 112)
    print(f"期間 {args.start} ~ {args.end}｜資料庫 {db_path.name}")
    print(f"暖機 {WARMUP_DAYS} 日｜首次訓練 {FIRST_TRAIN_DAYS} 日｜"
          f"每個 fold 測試 {TEST_WINDOW_DAYS} 日")
    print(f"移動停損分位 {PULLBACK_QUANTILE:.0%}｜優勢門檻 {args.edge_z:.1f} 個標準誤"
          f"｜組合槽位 {args.slots}｜標的池 {args.universe_size} 檔/季")
    if oos_start_override is not None and oos_start_override < pd.Timestamp("2026-09-14"):
        print("⚠️  此評估區間先前已被看過 7 次，不是全新 OOS；只能用來否定，不能證明。")
    print()

    # ── 標的池歷來成員（含已下市）──
    #
    # **必須查與 resolve_members 相同的 basis**，否則市值版獨有的成員
    # 不會被載入價格，會在候選清單裡靜默消失（實測 31 檔）。
    members_con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    has_new_table = members_con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='universe_history'"
    ).fetchone() is not None

    if has_new_table:
        all_members = [r[0] for r in members_con.execute(
            "SELECT DISTINCT stock_id FROM universe_history WHERE basis = ? "
            "ORDER BY stock_id", (args.basis,))]
        delisted = members_con.execute(
            "SELECT COUNT(DISTINCT u.stock_id) FROM universe_history u "
            "JOIN stock_master m USING (stock_id) "
            "WHERE u.basis = ? AND m.delisted_date IS NOT NULL", (args.basis,)
        ).fetchone()[0]
    else:
        all_members = [r[0] for r in members_con.execute(
            "SELECT DISTINCT stock_id FROM stock_universe_history ORDER BY stock_id")]
        delisted = members_con.execute(
            "SELECT COUNT(DISTINCT u.stock_id) FROM stock_universe_history u "
            "JOIN stock_master m USING (stock_id) WHERE m.delisted_date IS NOT NULL"
        ).fetchone()[0]
    members_con.close()
    print(f"標的池排名依據 {args.basis}")
    print(f"標的池歷來成員 {len(all_members)} 檔"
          f"（其中 {delisted} 檔已下市——納入才沒有 survivorship bias）")

    # ETF 對照組的價格要一起載入。它們不在標的池裡（不會被選中），
    # 只作為必跑對照（CLAUDE.md）。
    load_ids = sorted(set(all_members) | set(ETF_BENCHMARKS))

    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    prices = load_prices(load_ids, start=start, end=end,
                         adjusted=True, db_path=db_path,
                         unlock_frozen=args.unlock_frozen,
                         frozen_reason=args.frozen_reason)
    chips = load_chips(
        all_members,
        start=start,
        end=end,
        db_path=db_path,
        unlock_frozen=args.unlock_frozen,
        frozen_reason=args.frozen_reason,
    )
    dataset = build_dataset(load_ids, prices, chips)
    print(dataset.describe())
    print()

    by_stock = dataset.by_stock
    calendar = trading_calendar(by_stock)
    decision_dates = calendar[WARMUP_DAYS::DECISION_STRIDE]
    price_lookup = make_price_lookup(by_stock)

    print(f"交易日曆 {len(calendar)} 天｜決策日 {len(decision_dates)} 個", flush=True)

    print("解析逐季標的池成員 ...", flush=True)
    members_at = resolve_members(decision_dates, db_path,
                                 args.universe_size, args.basis)
    sizes = [len(v) for v in members_at.values()]
    print(f"  涵蓋 {len(members_at)} 個決策日｜每日候選 "
          f"{min(sizes)} ~ {max(sizes)} 檔", flush=True)

    print("預算分數序列（三族 × 全標的，一次算完）...", flush=True)
    t0 = time.time()
    scores = precompute_scores(by_stock)
    print(f"  完成，耗時 {time.time() - t0:.1f}s", flush=True)

    tiers = infer_tiers(db_path, args.universe_size, args.basis)

    rows: list[dict[str, object]] = []
    n_trials = len(STRATEGY_FAMILIES) * len(args.horizons)

    for horizon in args.horizons:
        print(f"\n預算 horizon={horizon} 的停損寬度與標記 ...", flush=True)
        t0 = time.time()
        outcomes = precompute_outcomes(by_stock, horizon, decision_dates)
        print(f"  {len(outcomes)} 檔、耗時 {time.time() - t0:.1f}s", flush=True)

        for family in STRATEGY_FAMILIES:
            print(f"  {family.name} × {horizon} 日 ...", flush=True)
            by_date = enumerate_decisions(
                outcomes, scores[family.name], decision_dates, members_at)
            rng = np.random.default_rng(RANDOM_SEED)

            strat_sig, rand_sig, oos_start, folds, failures = run_walk_forward(
                by_date,
                tiers,
                rng,
                args.edge_z,
                horizon,
                oos_start_override=oos_start_override,
                dev_end=dev_end,
            )

            if not strat_sig or oos_start is None:
                rows.append({"family": family.name, "horizon": horizon,
                             "folds": folds, "note": "OOS 無任何標的通過門檻"})
                continue

            oos_calendar = [d for d in calendar if d >= oos_start]
            strategy = simulate_portfolio(
                to_signals(strat_sig, calendar, tiers),
                price_lookup, oos_calendar, n_slots=args.slots, cost=DEFAULT)
            random_sim = simulate_portfolio(
                to_signals(rand_sig, calendar, tiers),
                price_lookup, oos_calendar, n_slots=args.slots, cost=DEFAULT)
            # 等權對照只用個股——ETF 本身就是一籃子，混進去等於重複計入
            stock_only = {k: v for k, v in by_stock.items()
                          if k not in ETF_BENCHMARKS}
            bh_curve = equal_weight_equity(stock_only, oos_calendar)
            bh_return = float(bh_curve.iloc[-1]) - 1.0
            bh_peak = bh_curve.cummax()
            bh_maxdd = float(((bh_peak - bh_curve) / bh_peak).max())

            etf_curves = etf_benchmark_curves(by_stock, oos_calendar)
            etf_stats = {
                etf: {
                    "total": float(curve.iloc[-1]) - 1.0,
                    "maxdd": float(((curve.cummax() - curve) / curve.cummax()).max()),
                }
                for etf, curve in etf_curves.items()
            }

            trade_dates = sorted({d for d, _, _ in strat_sig})
            exits = [d.exit_reason for _, d, _ in strat_sig]

            rows.append({
                "family": family.name, "horizon": horizon, "folds": folds,
                "calibration_failures": failures, "signals": len(strat_sig),
                "trades": strategy.n_trades,
                "effective": effective_samples(trade_dates, horizon, DECISION_STRIDE),
                "total": strategy.total_return,
                "annualized": strategy.annualized_return,
                "sharpe": strategy.sharpe, "maxdd": strategy.max_drawdown,
                "exposure": strategy.exposure,
                "random_total": random_sim.total_return,
                "random_maxdd": random_sim.max_drawdown,
                "bh_return": bh_return, "bh_maxdd": bh_maxdd,
                "etf": etf_stats,
                "oos_start": oos_start, "oos_end": oos_calendar[-1],
                "oos_days": len(oos_calendar),
                "stopped_pct": exits.count("trailing_stop") / len(exits),
                "avg_trail": float(np.mean([d.trail_pct for _, d, _ in strat_sig])),
                "note": "",
            })

    report(rows, n_trials, args.slots)


def infer_tiers(db_path: Path, universe_size: int,
                basis: str = DEFAULT_UNIVERSE_BASIS) -> dict[str, Tier]:
    """
    以**最近一次快照**的排名決定流動性分層。

    分層只影響滑價（0.3% vs 0.4%），用最新排名是可接受的簡化；
    逐季變動的分層會讓成本在同一筆交易的進出場之間跳動，反而更難解讀。
    """
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        has_new = con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='universe_history'"
        ).fetchone() is not None
        table = "universe_history" if has_new else "stock_universe_history"
        clause = "AND basis = ?" if has_new else ""
        extra: tuple = (basis,) if has_new else ()

        latest = con.execute(
            f"SELECT MAX(as_of_date) FROM {table} "
            f"WHERE 1=1 {clause}", extra).fetchone()[0]
        rows = con.execute(
            f"SELECT stock_id, rank FROM {table} WHERE as_of_date = ? {clause}",
            (latest, *extra)).fetchall()
        tiers = {sid: (Tier.LARGE if rank <= 50 else Tier.MID) for sid, rank in rows}
        # 快照裡沒有的（早年成員、已下市）保守視為中型股，滑價較高
        everything = con.execute(
            f"SELECT DISTINCT stock_id FROM {table} WHERE 1=1 {clause}",
            extra).fetchall()
        for (sid,) in everything:
            tiers.setdefault(sid, Tier.MID)
    finally:
        con.close()
    return tiers


def report(rows: list[dict], n_trials: int, slots: int) -> None:
    """輸出結果表與結論"""
    print()
    print("─" * 112)
    print(f"樣本外結果（{slots} 槽位組合、含 6 折 + 滑價成本、逐日標記市值）")
    print("─" * 112)
    print(f"{'策略族':<10}{'持有':>5}{'訊號':>7}{'成交':>6}{'總報酬':>11}{'年化':>9}"
          f"{'Sharpe':>8}{'MaxDD':>8}{'曝險':>7}"
          f"{'隨機':>11}{'買進持有':>11}{'對照MaxDD':>10}")
    print("─" * 112)

    for r in rows:
        if r.get("note"):
            print(f"{r['family']:<10}{r['horizon']:>5}  {r['note']}")
            continue
        sharpe = f"{r['sharpe']:.2f}" if r["sharpe"] is not None else "n/a"
        print(
            f"{r['family']:<10}{r['horizon']:>5}{r['signals']:>7}{r['trades']:>6}"
            f"{r['total'] * 100:>10.2f}%{r['annualized'] * 100:>8.2f}%"
            f"{sharpe:>8}{r['maxdd'] * 100:>7.2f}%{r['exposure'] * 100:>6.0f}%"
            f"{r['random_total'] * 100:>10.2f}%{r['bh_return'] * 100:>10.2f}%"
            f"{r['bh_maxdd'] * 100:>9.2f}%")

    print("─" * 112)
    scored = [r for r in rows if not r.get("note")]
    if scored:
        f = scored[0]
        print(f"OOS 期間 {f['oos_start'].date()} ~ {f['oos_end'].date()}"
              f"（{f['oos_days']} 個交易日，{f['oos_days'] / 252:.2f} 年）")
    print()

    if scored and scored[0].get("etf"):
        print("必跑對照組（CLAUDE.md）：")
        etf_names = {"0050": "台灣50", "0051": "中型100", "0056": "高股息"}
        first = scored[0]
        print(f"  {'等權買進持有（個股）':<22}"
              f"{first['bh_return'] * 100:>10.2f}%   MaxDD {first['bh_maxdd'] * 100:>6.2f}%")
        for etf, stat in first["etf"].items():
            label = f"{etf} {etf_names.get(etf, '')} 買進持有"
            print(f"  {label:<22}{stat['total'] * 100:>10.2f}%   "
                  f"MaxDD {stat['maxdd'] * 100:>6.2f}%")
        print()

    print("移動停損行為：")
    for r in scored:
        print(f"  {r['family']} × {r['horizon']} 日｜"
              f"平均停損幅度 {r['avg_trail'] * 100:.2f}%｜"
              f"觸停損出場 {r['stopped_pct'] * 100:.1f}%｜"
              f"訊號 {r['signals']} 中成交 {r['trades']}"
              f"（槽位擋掉 {r['signals'] - r['trades']}）")
    print()

    print("=" * 112)
    print("結論")
    print("=" * 112)
    beats_random = [r for r in scored if r["total"] > r["random_total"]]
    beats_bh = [r for r in scored if r["total"] > r["bh_return"]]
    lower_dd = [r for r in scored if r["maxdd"] < r["bh_maxdd"]]
    beats_0050 = [
        r for r in scored
        if "0050" in r.get("etf", {}) and r["total"] > r["etf"]["0050"]["total"]
    ]
    has_0050 = any("0050" in r.get("etf", {}) for r in scored)

    print(f"  跑贏隨機進場       {len(beats_random)} / {len(scored)}")
    print(f"  跑贏等權買進持有   {len(beats_bh)} / {len(scored)}")
    if has_0050:
        print(f"  跑贏 0050 買進持有 {len(beats_0050)} / {len(scored)}")
    else:
        print("  ⚠️  0050 對照缺席——CLAUDE.md 要求必跑，報告不完整")
    print(f"  回撤低於買進持有   {len(lower_dd)} / {len(scored)}")
    print()

    if not scored:
        print("  沒有任何組合產生交易。")
    elif beats_bh:
        print(f"  {len(beats_bh)} 組跑贏買進持有：")
        for r in sorted(beats_bh, key=lambda x: -x["total"]):
            print(f"    {r['family']} × {r['horizon']} 日｜"
                  f"{r['total'] * 100:+.2f}% vs 買進持有 {r['bh_return'] * 100:+.2f}%｜"
                  f"MaxDD {r['maxdd'] * 100:.2f}% vs {r['bh_maxdd'] * 100:.2f}%｜"
                  f"曝險 {r['exposure'] * 100:.0f}%｜成交 {r['trades']} 筆"
                  f"（有效樣本 {r['effective']}）")
            if "0050" in r.get("etf", {}):
                e = r["etf"]["0050"]
                verdict = "✓ 贏" if r["total"] > e["total"] else "✗ 輸"
                print(f"      vs 0050 買進持有 {e['total'] * 100:+.2f}%"
                      f"（MaxDD {e['maxdd'] * 100:.2f}%）→ {verdict}")
            if r["sharpe"] is not None:
                dsr = deflated_sharpe_ratio(
                    observed_sharpe=r["sharpe"], n_trials=n_trials,
                    n_observations=max(r["effective"], 2), sharpe_std=1.0)
                verdict = "顯著" if dsr.is_significant else "**無法排除運氣**"
                print(f"      Sharpe {r['sharpe']:.3f}｜"
                      f"DSR {dsr.deflated_sharpe:.4f}（N={n_trials}）→ {verdict}")
    else:
        best = max(scored, key=lambda r: r["total"])
        print("  **沒有任何組合跑贏等權買進持有。**")
        print(f"  最佳：{best['family']} × {best['horizon']} 日｜"
              f"{best['total'] * 100:+.2f}% vs {best['bh_return'] * 100:+.2f}%")

    print()
    print("  樣本量：")
    for r in scored:
        flag = "  ⚠️ 過少" if r["effective"] < 30 else ""
        print(f"    {r['family']} × {r['horizon']} 日｜"
              f"有效獨立樣本 {r['effective']}{flag}")

    print()
    print("=" * 112)
    print("⚠️  OOS 涵蓋 2018 貿易戰、2020 疫情崩跌、2022 空頭。不構成投資建議。")
    print("=" * 112)


if __name__ == "__main__":
    main()
