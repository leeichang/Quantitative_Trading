"""
模型評估腳本 — 整合策略回測 + 模型診斷

產出：
  - report.md            統一報告
  - strategy_metrics.csv  策略指標表
  - ic_horizon.png        Multi-Horizon IC
  - rolling_ic.png        Rolling IC + CUSUM
  - equity_curve.png      Top 策略權益曲線
  - monthly_heatmap.png   月度超額收益熱力圖

所有輸出存放在 scripts/output/{run_hash}/ 下。
"""

import hashlib
import json
import pickle
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.shared.constants import LABEL_EXPR, LOOKBACK_DAYS

# ══════════════════════════════════════════════════════════════
# 常數
# ══════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "data" / "models"
QLIB_DATA_DIR = PROJECT_ROOT / "data" / "qlib"
OUTPUT_BASE = PROJECT_ROOT / "scripts" / "output"

# 交易成本（玉山證券，電子下單 6 折）
COMMISSION_RATE = 0.001425
COMMISSION_DISCOUNT = 0.6
EFFECTIVE_COMMISSION = COMMISSION_RATE * COMMISSION_DISCOUNT
MIN_COMMISSION = 20
TRANSACTION_TAX = 0.003

# 投資參數
FIXED_AMOUNT_PER_STOCK = 50_000

# 回測範圍
START_MODEL_WEEK = "2023W01"
END_MODEL_WEEK = "2025W51"

# 診斷參數
HORIZONS = [1, 2, 3, 5, 10, 20]
N_QUANTILES = 5
TOP_K = 10


# ══════════════════════════════════════════════════════════════
# Run Hash
# ══════════════════════════════════════════════════════════════

def compute_run_hash() -> str:
    from src.repositories.factors import ALL_FACTORS

    model_type = "unknown"
    models_dir = Path(MODELS_DIR)
    if models_dir.exists():
        for d in sorted(models_dir.iterdir()):
            if d.is_dir() and (d / "model.pkl").exists():
                with open(d / "model.pkl", "rb") as f:
                    model = pickle.load(f)
                model_type = type(model).__name__
                break

    config = f"{model_type}|{LABEL_EXPR}|{len(ALL_FACTORS)}|{START_MODEL_WEEK}~{END_MODEL_WEEK}"
    return hashlib.md5(config.encode()).hexdigest()[:8]


# ══════════════════════════════════════════════════════════════
# 資料準備
# ══════════════════════════════════════════════════════════════

def export_qlib_data(start_date: date, end_date: date) -> None:
    from src.repositories.database import get_session
    from src.services.qlib_exporter import ExportConfig, QlibExporter

    session = get_session()
    try:
        exporter = QlibExporter(session)
        result = exporter.export(ExportConfig(
            start_date=start_date, end_date=end_date, output_dir=QLIB_DATA_DIR,
        ))
        print(f"  Exported: {result.stocks_exported} stocks, {result.fields_per_stock} fields, "
              f"{result.calendar_days} days")
    finally:
        session.close()


def init_qlib() -> None:
    import qlib
    from qlib.config import REG_CN
    qlib.init(provider_uri=str(QLIB_DATA_DIR), region=REG_CN)


def get_instruments() -> list[str]:
    instruments_file = QLIB_DATA_DIR / "instruments" / "all.txt"
    with open(instruments_file) as f:
        return [line.strip().split()[0] for line in f if line.strip()]


def load_model(model_name: str) -> tuple:
    model_dir = MODELS_DIR / model_name
    with open(model_dir / "model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(model_dir / "factors.json") as f:
        factors = json.load(f)
    config = {}
    config_path = model_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    return model, factors, config


def process_inf(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        mask = np.isinf(df[col])
        if mask.any():
            col_mean = df.loc[~mask, col].mean()
            df.loc[mask, col] = col_mean if not np.isnan(col_mean) else 0
    return df


def zscore_by_date(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(level="datetime", group_keys=False).apply(
        lambda x: (x - x.mean()) / (x.std() + 1e-8)
    )


def preload_data(
    factors: list[dict], instruments: list[str],
    start_date: date, end_date: date,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from qlib.data import D

    print("  Loading features...")
    fields = [f["expression"] for f in factors]
    names = [f["name"] for f in factors]

    features = D.features(
        instruments=instruments, fields=fields,
        start_time=start_date.strftime("%Y-%m-%d"),
        end_time=end_date.strftime("%Y-%m-%d"),
    )
    if not features.empty:
        features.columns = names
    print(f"  Features: {len(features):,} rows")

    extended_end = end_date + timedelta(days=10)
    print("  Loading prices...")
    prices = D.features(
        instruments=instruments, fields=["$close"],
        start_time=start_date.strftime("%Y-%m-%d"),
        end_time=extended_end.strftime("%Y-%m-%d"),
    )
    if not prices.empty:
        prices.columns = ["close"]
    print(f"  Prices: {len(prices):,} rows")

    return features, prices


def load_extended_prices(
    instruments: list[str], start_date: date, end_date: date, extra_days: int = 40,
) -> tuple[pd.DataFrame, list[date]]:
    from qlib.data import D

    extended_end = end_date + timedelta(days=extra_days)
    prices = D.features(
        instruments=instruments, fields=["$close"],
        start_time=start_date.strftime("%Y-%m-%d"),
        end_time=extended_end.strftime("%Y-%m-%d"),
    )
    prices.columns = ["close"]
    close_wide = prices["close"].unstack(level="instrument")
    trading_days = sorted([
        d.date() if hasattr(d, 'date') else d for d in close_wide.index
    ])
    close_wide.index = trading_days
    return close_wide, trading_days


# ══════════════════════════════════════════════════════════════
# 模型收集 & 預測
# ══════════════════════════════════════════════════════════════

@dataclass
class WeekModelInfo:
    predict_week: str
    model_week: str
    model_name: str
    is_fallback: bool


def parse_week_id(week_id: str) -> tuple[int, int]:
    return int(week_id[:4]), int(week_id[5:])


def get_next_week_id(week_id: str) -> str:
    year, week = parse_week_id(week_id)
    friday = date.fromisocalendar(year, week, 5)
    next_monday = friday + timedelta(days=3)
    iso_year, iso_week, _ = next_monday.isocalendar()
    return f"{iso_year}W{iso_week:02d}"


def get_previous_week_id(week_id: str) -> str:
    year, week = parse_week_id(week_id)
    monday = date.fromisocalendar(year, week, 1)
    prev_friday = monday - timedelta(days=3)
    iso_year, iso_week, _ = prev_friday.isocalendar()
    return f"{iso_year}W{iso_week:02d}"


def compare_week_ids(a: str, b: str) -> int:
    ya, wa = parse_week_id(a)
    yb, wb = parse_week_id(b)
    if ya != yb:
        return -1 if ya < yb else 1
    if wa != wb:
        return -1 if wa < wb else 1
    return 0


def get_weeks_in_range(start: str, end: str) -> list[str]:
    weeks = []
    current = start
    while compare_week_ids(current, end) <= 0:
        weeks.append(current)
        current = get_next_week_id(current)
    return weeks


def get_week_date_range(week_id: str) -> tuple[date, date]:
    year, week = parse_week_id(week_id)
    monday = date.fromisocalendar(year, week, 1)
    friday = date.fromisocalendar(year, week, 5)
    return monday, friday


def collect_models(start_week: str, end_week: str) -> list[WeekModelInfo]:
    all_weeks = get_weeks_in_range(start_week, end_week)

    existing_models: dict[str, str] = {}
    for d in MODELS_DIR.iterdir():
        if d.is_dir() and (d / "model.pkl").exists():
            name = d.name
            parts = name.split("-")
            if len(parts) >= 1:
                existing_models[parts[0]] = name

    result = []
    for week_id in all_weeks:
        predict_week = get_next_week_id(week_id)

        if week_id in existing_models:
            result.append(WeekModelInfo(
                predict_week=predict_week, model_week=week_id,
                model_name=existing_models[week_id], is_fallback=False,
            ))
        else:
            current = get_previous_week_id(week_id)
            for _ in range(10):
                if current in existing_models:
                    result.append(WeekModelInfo(
                        predict_week=predict_week, model_week=current,
                        model_name=existing_models[current], is_fallback=True,
                    ))
                    break
                current = get_previous_week_id(current)

    return result


def predict_week(
    model, factors: list[dict], features_cache: pd.DataFrame,
    predict_start: date, predict_end: date,
) -> pd.DataFrame:
    start_str = predict_start.strftime("%Y-%m-%d")
    end_str = predict_end.strftime("%Y-%m-%d")

    df = features_cache.loc[
        (features_cache.index.get_level_values("datetime") >= start_str) &
        (features_cache.index.get_level_values("datetime") <= end_str)
    ].copy()

    names = [f["name"] for f in factors]
    available = [n for n in names if n in df.columns]
    if available:
        df = df[available]

    if df.empty:
        return pd.DataFrame()

    df = process_inf(df)
    df = zscore_by_date(df)
    df = df.fillna(0)

    predictions = model.predict(df.values)
    pred_series = pd.Series(predictions, index=df.index, name="score")
    pred_df = pred_series.unstack(level="instrument")
    pred_df.index = pd.to_datetime(pred_df.index.date)

    return pred_df


def generate_daily_scores(
    model_infos: list[WeekModelInfo], features_cache: pd.DataFrame,
) -> dict[date, pd.Series]:
    daily_scores: dict[date, pd.Series] = {}
    model_cache: dict[str, tuple] = {}

    for i, info in enumerate(model_infos):
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(model_infos)}] Predicting {info.predict_week}...")

        if info.model_name not in model_cache:
            model_cache[info.model_name] = load_model(info.model_name)
        model, factors, _ = model_cache[info.model_name]

        predict_start, predict_end = get_week_date_range(info.predict_week)
        pred_df = predict_week(model, factors, features_cache, predict_start, predict_end)

        if pred_df.empty:
            continue

        for dt in pred_df.index:
            scores = pred_df.loc[dt].dropna()
            if not scores.empty:
                daily_scores[dt.date() if hasattr(dt, 'date') else dt] = scores

    return daily_scores


# ══════════════════════════════════════════════════════════════
# 策略
# ══════════════════════════════════════════════════════════════

@dataclass
class DailyRecord:
    dt: date
    gross_return: float
    net_return: float
    market_return: float
    cost: float
    turnover: float
    n_holdings: int
    n_buy: int
    n_sell: int


@dataclass
class StrategyResult:
    name: str
    records: list[DailyRecord]
    total_cost: float


@dataclass
class HedgeConfig:
    sma_period: int
    cash_ratio: float  # 0.0 = fully cash, 1.0 = fully invested


class Strategy(ABC):
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def decide(self, scores: pd.Series, prev_holdings: dict[str, int]) -> tuple[list[str], list[str]]: ...


class TopKStrategy(Strategy):
    def __init__(self, k: int):
        self.k = k

    def name(self) -> str:
        return f"TopK(K={self.k})"

    def decide(self, scores, prev_holdings):
        ranked_df = scores.reset_index()
        ranked_df.columns = ["symbol", "score"]
        ranked_df = ranked_df.sort_values(by=["score", "symbol"], ascending=[False, True]).head(self.k)
        new_set = set(ranked_df["symbol"].tolist())
        old_set = set(prev_holdings.keys())
        return [s for s in ranked_df["symbol"] if s not in old_set], [s for s in old_set if s not in new_set]


class TopKDropStrategy(Strategy):
    def __init__(self, k: int, d: int):
        self.k = k
        self.d = d

    def name(self) -> str:
        return f"TopKDrop(K={self.k},D={self.d})"

    def decide(self, scores, prev_holdings):
        ranked_df = scores.reset_index()
        ranked_df.columns = ["symbol", "score"]
        ranked_df = ranked_df.sort_values(by=["score", "symbol"], ascending=[False, True])
        topk_set = set(ranked_df["symbol"].head(self.k).tolist())
        old_set = set(prev_holdings.keys())

        sell_candidates = sorted(
            [s for s in old_set if s not in topk_set],
            key=lambda s: scores.get(s, -999),
        )
        to_sell = sell_candidates[:self.d]
        buy_candidates = [s for s in ranked_df["symbol"] if s not in old_set]
        to_buy = buy_candidates[:len(to_sell)]

        current_count = len(old_set) - len(to_sell) + len(to_buy)
        if current_count < self.k:
            remaining = [s for s in buy_candidates if s not in to_buy]
            to_buy = to_buy + remaining[:self.k - current_count]

        return to_buy, to_sell


class HoldPeriodStrategy(Strategy):
    def __init__(self, k: int, h: int):
        self.k = k
        self.h = h

    def name(self) -> str:
        return f"HoldPeriod(K={self.k},H={self.h})"

    def decide(self, scores, prev_holdings):
        ranked_df = scores.reset_index()
        ranked_df.columns = ["symbol", "score"]
        ranked_df = ranked_df.sort_values(by=["score", "symbol"], ascending=[False, True])
        topk = ranked_df["symbol"].head(self.k).tolist()
        topk_set = set(topk)
        old_set = set(prev_holdings.keys())

        to_sell = [s for s in old_set if s not in topk_set and prev_holdings[s] >= self.h]
        buy_candidates = [s for s in topk if s not in old_set]
        to_buy = buy_candidates[:len(to_sell)]

        current_count = len(old_set) - len(to_sell) + len(to_buy)
        if current_count < self.k:
            remaining = [s for s in buy_candidates if s not in to_buy]
            to_buy = to_buy + remaining[:self.k - current_count]

        return to_buy, to_sell


class HoldDropStrategy(Strategy):
    def __init__(self, k: int, h: int, d: int):
        self.k = k
        self.h = h
        self.d = d

    def name(self) -> str:
        return f"HoldDrop(K={self.k},H={self.h},D={self.d})"

    def decide(self, scores, prev_holdings):
        ranked_df = scores.reset_index()
        ranked_df.columns = ["symbol", "score"]
        ranked_df = ranked_df.sort_values(by=["score", "symbol"], ascending=[False, True])
        topk_set = set(ranked_df["symbol"].head(self.k).tolist())
        old_set = set(prev_holdings.keys())

        sell_candidates = sorted(
            [s for s in old_set if s not in topk_set and prev_holdings[s] >= self.h],
            key=lambda s: scores.get(s, -999),
        )
        to_sell = sell_candidates[:self.d]
        buy_candidates = [s for s in ranked_df["symbol"] if s not in old_set]
        to_buy = buy_candidates[:len(to_sell)]

        current_count = len(old_set) - len(to_sell) + len(to_buy)
        if current_count < self.k:
            remaining = [s for s in buy_candidates if s not in to_buy]
            to_buy = to_buy + remaining[:self.k - current_count]

        return to_buy, to_sell


class HoldDropBottomStrategy(Strategy):
    def __init__(self, k: int, h: int, d: int, b: int):
        self.k = k
        self.h = h
        self.d = d
        self.b = b

    def name(self) -> str:
        return f"HoldDropBot(K={self.k},H={self.h},D={self.d},B={self.b})"

    def decide(self, scores, prev_holdings):
        ranked_df = scores.reset_index()
        ranked_df.columns = ["symbol", "score"]
        ranked_df = ranked_df.sort_values(by=["score", "symbol"], ascending=[False, True])
        all_symbols = ranked_df["symbol"].tolist()
        topk_set = set(all_symbols[:self.k])
        bottom_b_set = set(all_symbols[-self.b:]) if len(all_symbols) >= self.b else set()
        old_set = set(prev_holdings.keys())

        force_sell = [s for s in old_set if s in bottom_b_set]
        regular_candidates = sorted(
            [s for s in old_set if s not in topk_set and s not in force_sell and prev_holdings[s] >= self.h],
            key=lambda s: scores.get(s, -999),
        )
        to_sell = force_sell + regular_candidates[:self.d]

        new_holding_set = old_set - set(to_sell)
        buy_candidates = [s for s in all_symbols if s not in new_holding_set]
        to_buy = buy_candidates[:max(0, self.k - len(new_holding_set))]

        return to_buy, to_sell


class ThresholdExitStrategy(Strategy):
    def __init__(self, k: int, m: int):
        self.k = k
        self.m = m

    def name(self) -> str:
        return f"ThresholdExit(K={self.k},M={self.m})"

    def decide(self, scores, prev_holdings):
        ranked_df = scores.reset_index()
        ranked_df.columns = ["symbol", "score"]
        ranked_df = ranked_df.sort_values(by=["score", "symbol"], ascending=[False, True])
        topk = ranked_df["symbol"].head(self.k).tolist()
        topm_set = set(ranked_df["symbol"].head(self.m).tolist())
        old_set = set(prev_holdings.keys())

        to_sell = [s for s in old_set if s not in topm_set]
        buy_candidates = [s for s in topk if s not in old_set]
        to_buy = buy_candidates[:len(to_sell)]

        current_count = len(old_set) - len(to_sell) + len(to_buy)
        if current_count < self.k:
            remaining = [s for s in buy_candidates if s not in to_buy]
            to_buy = to_buy + remaining[:self.k - current_count]

        return to_buy, to_sell


class ScoreWeightedStrategy(Strategy):
    def __init__(self, k: int):
        self.k = k

    def name(self) -> str:
        return f"ScoreWeighted(K={self.k})"

    def decide(self, scores, prev_holdings):
        ranked_df = scores.reset_index()
        ranked_df.columns = ["symbol", "score"]
        ranked_df = ranked_df.sort_values(by=["score", "symbol"], ascending=[False, True]).head(self.k)
        new_set = set(ranked_df["symbol"].tolist())
        old_set = set(prev_holdings.keys())
        return [s for s in ranked_df["symbol"] if s not in old_set], [s for s in old_set if s not in new_set]


def build_strategies() -> list[Strategy]:
    strategies = []
    for k in [5, 10, 15, 20, 30]:
        strategies.append(TopKStrategy(k))
    for k, d in [(10, 1), (10, 2), (10, 3), (20, 2), (20, 5)]:
        strategies.append(TopKDropStrategy(k, d))
    for k, h in [(10, 3), (10, 5), (20, 3), (20, 5)]:
        strategies.append(HoldPeriodStrategy(k, h))
    for k, h, d in [
        (10, 2, 1), (10, 3, 1), (10, 4, 1), (10, 5, 1),
        (10, 7, 1), (10, 10, 1), (10, 5, 2), (10, 3, 2),
    ]:
        strategies.append(HoldDropStrategy(k, h, d))
    for k, h, d, b in [
        (10, 3, 1, 5), (10, 3, 1, 10), (10, 3, 1, 20), (10, 3, 1, 30),
        (10, 3, 1, 40), (10, 3, 1, 50),
        (10, 4, 1, 10), (10, 4, 1, 20), (10, 4, 1, 30),
    ]:
        strategies.append(HoldDropBottomStrategy(k, h, d, b))
    for k, m in [(10, 20), (10, 30), (10, 50)]:
        strategies.append(ThresholdExitStrategy(k, m))
    for k in [10, 20]:
        strategies.append(ScoreWeightedStrategy(k))
    return strategies


def build_hedged_strategies() -> list[tuple[Strategy, HedgeConfig]]:
    base_strategies = [
        HoldDropStrategy(10, 3, 1),
        HoldDropStrategy(10, 4, 1),
        TopKDropStrategy(10, 1),
    ]
    sma_configs = [
        HedgeConfig(50, 0.0), HedgeConfig(50, 0.5),
        HedgeConfig(100, 0.0), HedgeConfig(100, 0.5),
        HedgeConfig(200, 0.0), HedgeConfig(200, 0.5),
    ]
    result = []
    for s in base_strategies:
        for hc in sma_configs:
            result.append((s, hc))
    return result


# ══════════════════════════════════════════════════════════════
# 回測模擬引擎
# ══════════════════════════════════════════════════════════════

def calc_trade_cost(amount: float, is_sell: bool) -> float:
    commission = max(abs(amount) * EFFECTIVE_COMMISSION, MIN_COMMISSION)
    tax = abs(amount) * TRANSACTION_TAX if is_sell else 0.0
    return commission + tax


def simulate(
    strategy: Strategy,
    daily_scores: dict[date, pd.Series],
    close_wide: pd.DataFrame,
    trading_days: list,
    hedge_config: HedgeConfig | None = None,
) -> StrategyResult:
    day_to_idx = {d: i for i, d in enumerate(trading_days)}
    holdings: dict[str, int] = {}
    total_cost = 0.0
    records: list[DailyRecord] = []

    score_dates = sorted(daily_scores.keys())
    is_score_weighted = isinstance(strategy, ScoreWeightedStrategy)
    k = getattr(strategy, 'k', 10)

    # Precompute market SMA for hedging (no lookahead: uses close up to each date)
    market_cum = None
    market_sma = None
    if hedge_config:
        market_daily = close_wide.pct_change(fill_method=None).mean(axis=1).fillna(0)
        market_cum = (1 + market_daily).cumprod()
        market_sma = market_cum.rolling(
            hedge_config.sma_period, min_periods=hedge_config.sma_period,
        ).mean()

    for pred_date in score_dates:
        if pred_date not in day_to_idx:
            candidates = [d for d in trading_days if d <= pred_date]
            if not candidates:
                continue
            idx = day_to_idx[candidates[-1]]
        else:
            idx = day_to_idx[pred_date]

        if idx + 2 >= len(trading_days):
            continue
        t1 = trading_days[idx + 1]
        t2 = trading_days[idx + 2]

        if t1 not in close_wide.index or t2 not in close_wide.index:
            continue
        prices_t1 = close_wide.loc[t1].dropna()
        prices_t2 = close_wide.loc[t2].dropna()

        scores = daily_scores[pred_date]

        common_all = prices_t1.index.intersection(prices_t2.index)
        if len(common_all) < 2:
            continue
        all_returns = (prices_t2[common_all] - prices_t1[common_all]) / prices_t1[common_all]
        market_return = float(all_returns.mean())

        to_buy, to_sell = strategy.decide(scores, holdings)
        to_sell = [s for s in to_sell if s in prices_t1.index]
        to_buy = [s for s in to_buy if s in prices_t1.index and s in prices_t2.index]

        for s in to_sell:
            holdings.pop(s, None)
        for s in to_buy:
            holdings[s] = 0
        for s in list(holdings.keys()):
            holdings[s] += 1

        held_stocks = [s for s in holdings if s in common_all]
        if held_stocks:
            if is_score_weighted and len(held_stocks) > 1:
                weights = {s: max(scores.get(s, 0), 0) for s in held_stocks}
                total_score = sum(weights.values())
                if total_score > 0:
                    gross_return = sum(
                        (weights[s] / total_score) * float(all_returns.get(s, 0))
                        for s in held_stocks
                    )
                else:
                    gross_return = float(all_returns[held_stocks].mean())
            else:
                gross_return = float(all_returns[held_stocks].mean())
        else:
            gross_return = 0.0

        n_sell = len(to_sell)
        n_buy = len(to_buy)
        n_held = len(holdings)

        day_cost = 0.0
        for _ in range(n_sell):
            day_cost += calc_trade_cost(FIXED_AMOUNT_PER_STOCK, is_sell=True)
        for _ in range(n_buy):
            day_cost += calc_trade_cost(FIXED_AMOUNT_PER_STOCK, is_sell=False)
        total_position = n_held * FIXED_AMOUNT_PER_STOCK if n_held > 0 else k * FIXED_AMOUNT_PER_STOCK
        cost_rate = day_cost / total_position if total_position > 0 else 0

        # Apply SMA hedging: scale returns by invest_ratio
        invest_ratio = 1.0
        if hedge_config and market_cum is not None and market_sma is not None:
            hedge_date = trading_days[idx]
            if hedge_date in market_cum.index and hedge_date in market_sma.index:
                cum_val = market_cum[hedge_date]
                sma_val = market_sma[hedge_date]
                if pd.notna(sma_val) and cum_val < sma_val:
                    invest_ratio = hedge_config.cash_ratio

        gross_return = gross_return * invest_ratio
        cost_rate = cost_rate * invest_ratio
        net_return = gross_return - cost_rate

        total_cost += day_cost * invest_ratio
        turnover = (n_sell + n_buy) / (2 * n_held) if n_held > 0 else 0

        records.append(DailyRecord(
            dt=pred_date, gross_return=gross_return, net_return=net_return,
            market_return=market_return, cost=day_cost * invest_ratio, turnover=turnover,
            n_holdings=n_held, n_buy=n_buy, n_sell=n_sell,
        ))

    strat_name = strategy.name()
    if hedge_config:
        strat_name += f"+SMA{hedge_config.sma_period}({int(hedge_config.cash_ratio * 100)}%)"
    return StrategyResult(name=strat_name, records=records, total_cost=total_cost)


# ══════════════════════════════════════════════════════════════
# 統計指標
# ══════════════════════════════════════════════════════════════

@dataclass
class Metrics:
    strategy: str
    ann_return: float
    ann_excess: float
    sharpe: float | None
    max_drawdown: float
    calmar: float | None
    win_rate: float
    avg_turnover: float
    total_cost: float
    info_ratio: float | None
    t_stat: float | None
    n_days: int


def compute_metrics(result: StrategyResult) -> Metrics:
    records = result.records
    if not records:
        return Metrics(strategy=result.name, ann_return=0, ann_excess=0,
                       sharpe=None, max_drawdown=0, calmar=None, win_rate=0,
                       avg_turnover=0, total_cost=0, info_ratio=None, t_stat=None, n_days=0)

    n_days = len(records)
    ann_factor = 250 / n_days if n_days > 0 else 1

    daily_returns = [r.net_return for r in records]
    market_returns = [r.market_return for r in records]

    cum_return = 1.0
    for r in daily_returns:
        cum_return *= (1 + r)
    cum_market = 1.0
    for r in market_returns:
        cum_market *= (1 + r)

    ann_return = ((cum_return ** ann_factor) - 1) * 100
    ann_excess = ann_return - ((cum_market ** ann_factor) - 1) * 100

    arr = np.array(daily_returns)
    sharpe = float(arr.mean() / arr.std() * np.sqrt(250)) if len(arr) >= 2 and arr.std() > 0 else None

    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for r in daily_returns:
        equity *= (1 + r)
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak)
    max_drawdown = max_dd * 100

    calmar = ann_return / max_drawdown if max_drawdown > 0 else None

    excess_daily = [r.net_return - r.market_return for r in records]
    win_rate = sum(1 for e in excess_daily if e > 0) / len(excess_daily) * 100

    avg_turnover = float(np.mean([r.turnover for r in records]) * 100)

    excess_arr = np.array(excess_daily)
    info_ratio = float(excess_arr.mean() / excess_arr.std() * np.sqrt(250)) if excess_arr.std() > 0 else None
    t_val, _ = stats.ttest_1samp(excess_daily, 0)
    t_stat = float(t_val) if not np.isnan(t_val) else None

    return Metrics(
        strategy=result.name, ann_return=round(ann_return, 2),
        ann_excess=round(ann_excess, 2),
        sharpe=round(sharpe, 3) if sharpe is not None else None,
        max_drawdown=round(max_drawdown, 2),
        calmar=round(calmar, 3) if calmar is not None else None,
        win_rate=round(win_rate, 1), avg_turnover=round(avg_turnover, 2),
        total_cost=round(result.total_cost, 0),
        info_ratio=round(info_ratio, 3) if info_ratio is not None else None,
        t_stat=round(t_stat, 3) if t_stat is not None else None,
        n_days=n_days,
    )


# ══════════════════════════════════════════════════════════════
# IC 計算
# ══════════════════════════════════════════════════════════════

def compute_daily_ic(
    daily_scores: dict[date, pd.Series],
    close_wide: pd.DataFrame, trading_days: list[date],
    horizon: int = 1,
) -> pd.Series:
    day_to_idx = {d: i for i, d in enumerate(trading_days)}
    ic_data = {}

    for pred_date in sorted(daily_scores.keys()):
        if pred_date not in day_to_idx:
            continue
        idx = day_to_idx[pred_date]
        if idx + 1 + horizon >= len(trading_days):
            continue

        t_buy = trading_days[idx + 1]
        t_sell = trading_days[idx + 1 + horizon]

        if t_buy not in close_wide.index or t_sell not in close_wide.index:
            continue

        returns = (close_wide.loc[t_sell] - close_wide.loc[t_buy]) / close_wide.loc[t_buy]
        scores = daily_scores[pred_date]
        common = scores.dropna().index.intersection(returns.dropna().index)
        if len(common) < 10:
            continue

        corr, _ = stats.spearmanr(scores[common], returns[common])
        if not np.isnan(corr):
            ic_data[pred_date] = corr

    return pd.Series(ic_data).sort_index()


# ══════════════════════════════════════════════════════════════
# 診斷分析
# ══════════════════════════════════════════════════════════════

def analyze_multi_horizon_ic(
    daily_scores: dict[date, pd.Series],
    close_wide: pd.DataFrame, trading_days: list[date],
) -> dict:
    result = {"horizons": {}, "optimal": 1, "series": {}}

    for h in HORIZONS:
        ic_series = compute_daily_ic(daily_scores, close_wide, trading_days, horizon=h)
        if len(ic_series) < 10:
            continue
        mean_ic = float(ic_series.mean())
        std_ic = float(ic_series.std())
        result["horizons"][h] = {
            "ic": mean_ic,
            "icir": mean_ic / std_ic if std_ic > 0 else 0,
            "pos_rate": float((ic_series > 0).mean()),
        }
        result["series"][h] = ic_series

    if result["horizons"]:
        result["optimal"] = max(result["horizons"], key=lambda h: result["horizons"][h]["ic"])
    return result


def analyze_rolling_ic(
    daily_scores: dict[date, pd.Series],
    close_wide: pd.DataFrame, trading_days: list[date],
) -> dict:
    daily_ic = compute_daily_ic(daily_scores, close_wide, trading_days, horizon=1)
    rolling_20 = daily_ic.rolling(20, min_periods=10).mean()
    mean_ic = float(daily_ic.mean())
    std_ic = float(daily_ic.std())

    ic_df = daily_ic.to_frame("ic")
    ic_df["year"] = [d.year for d in ic_df.index]
    ic_df["quarter"] = [f"{d.year}Q{(d.month - 1) // 3 + 1}" for d in ic_df.index]

    # CUSUM
    ic_vals = daily_ic.values
    cusum = np.cumsum(ic_vals - mean_ic) / (std_ic if std_ic > 0 else 1)
    threshold = np.sqrt(len(cusum)) * 0.5
    breaks = [i for i in range(1, len(cusum))
              if abs(cusum[i]) > threshold and abs(cusum[i - 1]) <= threshold]

    return {
        "daily_ic": daily_ic,
        "rolling_20": rolling_20,
        "mean_ic": mean_ic,
        "icir": mean_ic / std_ic if std_ic > 0 else 0,
        "ic_pos_rate": float((daily_ic > 0).mean()),
        "by_year": ic_df.groupby("year")["ic"].mean().to_dict(),
        "by_quarter": ic_df.groupby("quarter")["ic"].mean().to_dict(),
        "cusum": cusum,
        "breaks": breaks,
    }


def analyze_market_regimes(
    daily_scores: dict[date, pd.Series],
    close_wide: pd.DataFrame, trading_days: list[date],
    daily_ic: pd.Series, daily_records: list[DailyRecord],
) -> dict:
    market_daily = close_wide.pct_change(fill_method=None).mean(axis=1).dropna()
    cum_market = (1 + market_daily).cumprod()
    sma50 = cum_market.rolling(50, min_periods=30).mean()
    vol20 = market_daily.rolling(20, min_periods=10).std()
    vol_median = float(vol20.median())

    regime_labels = pd.Series(index=market_daily.index, dtype=str)
    for d in regime_labels.index:
        if pd.isna(sma50.get(d)) or pd.isna(vol20.get(d)):
            regime_labels[d] = "unknown"
        elif cum_market[d] > sma50[d] and vol20[d] <= vol_median:
            regime_labels[d] = "bull"
        elif cum_market[d] <= sma50[d] and vol20[d] > vol_median:
            regime_labels[d] = "bear"
        else:
            regime_labels[d] = "sideways"

    excess_by_date = {r.dt: r.net_return - r.market_return for r in daily_records}

    regime_stats = {}
    for regime in ["bull", "sideways", "bear"]:
        regime_dates = set(regime_labels[regime_labels == regime].index)
        ic_in = daily_ic[[d for d in daily_ic.index if d in regime_dates]]
        excess_in = [excess_by_date[d] for d in excess_by_date if d in regime_dates]

        regime_stats[regime] = {
            "n_days": len(regime_dates),
            "mean_ic": float(ic_in.mean()) if len(ic_in) > 0 else 0,
            "mean_excess": float(np.mean(excess_in)) * 10000 if excess_in else 0,
            "win_rate": sum(1 for e in excess_in if e > 0) / len(excess_in) * 100 if excess_in else 0,
        }

    valid = regime_labels[regime_labels != "unknown"]
    transitions = sum(1 for i in range(1, len(valid)) if valid.iloc[i] != valid.iloc[i - 1])

    return {"stats": regime_stats, "transitions": transitions}


def analyze_quantile_spread(
    daily_scores: dict[date, pd.Series],
    close_wide: pd.DataFrame, trading_days: list[date],
) -> dict:
    day_to_idx = {d: i for i, d in enumerate(trading_days)}
    quantile_returns: dict[date, dict[int, float]] = {}

    for pred_date in sorted(daily_scores.keys()):
        if pred_date not in day_to_idx:
            continue
        idx = day_to_idx[pred_date]
        if idx + 2 >= len(trading_days):
            continue

        t1, t2 = trading_days[idx + 1], trading_days[idx + 2]
        if t1 not in close_wide.index or t2 not in close_wide.index:
            continue

        returns = (close_wide.loc[t2] - close_wide.loc[t1]) / close_wide.loc[t1]
        scores = daily_scores[pred_date]
        common = scores.dropna().index.intersection(returns.dropna().index)
        if len(common) < N_QUANTILES * 2:
            continue

        s, r = scores[common], returns[common]
        try:
            q_labels = pd.qcut(s.rank(method='first'), N_QUANTILES, labels=False)
        except ValueError:
            q_labels = pd.cut(s, N_QUANTILES, labels=False)

        day_q = {}
        for q in range(N_QUANTILES):
            mask = q_labels == q
            if mask.any():
                day_q[q] = float(r[mask].mean())
        if len(day_q) == N_QUANTILES:
            quantile_returns[pred_date] = day_q

    qr_df = pd.DataFrame(quantile_returns).T
    avg_returns = qr_df.mean()
    mono_corr, mono_p = stats.spearmanr(range(N_QUANTILES), avg_returns.values)
    spread = qr_df[N_QUANTILES - 1] - qr_df[0]
    t_val, p_val = stats.ttest_1samp(spread.dropna(), 0)

    return {
        "avg_returns": avg_returns,
        "mono_rho": float(mono_corr),
        "spread_bps": float(spread.mean()) * 10000,
        "spread_t": float(t_val) if not np.isnan(t_val) else 0,
    }


def analyze_score_quality(daily_scores: dict[date, pd.Series]) -> dict:
    prev_topk = None
    prev_scores = None
    overlaps, taus = [], []

    for d in sorted(daily_scores.keys()):
        scores = daily_scores[d]
        topk_today = set(scores.nlargest(TOP_K).index)

        if prev_topk is not None:
            overlaps.append(len(topk_today & prev_topk) / TOP_K)

        if prev_scores is not None:
            common = scores.index.intersection(prev_scores.index)
            if len(common) >= 10:
                t, _ = stats.kendalltau(scores[common], prev_scores[common])
                if not np.isnan(t):
                    taus.append(float(t))

        prev_topk = topk_today
        prev_scores = scores

    all_unique = [float(daily_scores[d].nunique()) for d in daily_scores]
    all_std = [float(daily_scores[d].std()) for d in daily_scores]

    return {
        "avg_unique": float(np.mean(all_unique)),
        "avg_std": float(np.mean(all_std)),
        "topk_overlap": float(np.mean(overlaps)) if overlaps else 0,
        "kendall_tau": float(np.mean(taus)) if taus else 0,
    }


# ══════════════════════════════════════════════════════════════
# 新增分析
# ══════════════════════════════════════════════════════════════

def analyze_monthly_excess(records: list[DailyRecord]) -> pd.DataFrame:
    """Year × Month 月度超額收益 (bps)"""
    rows = [{"date": r.dt, "excess": r.net_return - r.market_return} for r in records]
    df = pd.DataFrame(rows)
    df["year"] = df["date"].apply(lambda d: d.year)
    df["month"] = df["date"].apply(lambda d: d.month)
    pivot = df.groupby(["year", "month"])["excess"].mean() * 10000  # bps
    return pivot.unstack(level="month").rename(
        columns={i: f"{i:02d}" for i in range(1, 13)}
    )


def analyze_yearly_breakdown(records: list[DailyRecord]) -> pd.DataFrame:
    """年度績效分解"""
    rows = []
    df = pd.DataFrame([{
        "date": r.dt, "net": r.net_return, "market": r.market_return,
        "excess": r.net_return - r.market_return,
    } for r in records])
    df["year"] = df["date"].apply(lambda d: d.year)

    for year, grp in df.groupby("year"):
        n = len(grp)
        ann_factor = 250 / n if n > 0 else 1
        cum_net = (1 + grp["net"]).prod()
        cum_mkt = (1 + grp["market"]).prod()
        ann_ret = (cum_net ** ann_factor - 1) * 100
        ann_mkt = (cum_mkt ** ann_factor - 1) * 100
        excess_arr = grp["excess"].values
        sharpe = float(grp["net"].mean() / grp["net"].std() * np.sqrt(250)) if grp["net"].std() > 0 else 0

        # Max drawdown
        equity = (1 + grp["net"]).cumprod()
        peak = equity.cummax()
        dd = ((peak - equity) / peak).max() * 100

        rows.append({
            "year": year, "ann_return": round(ann_ret, 1),
            "ann_excess": round(ann_ret - ann_mkt, 1),
            "sharpe": round(sharpe, 2),
            "win_rate": round(float((excess_arr > 0).mean()) * 100, 1),
            "max_dd": round(dd, 1), "n_days": n,
        })

    return pd.DataFrame(rows).set_index("year")


def analyze_top_drawdowns(records: list[DailyRecord], top_n: int = 5) -> list[dict]:
    """Top-N 最大 drawdown 明細"""
    if not records:
        return []

    dates = [r.dt for r in records]
    equity = [1.0]
    for r in records:
        equity.append(equity[-1] * (1 + r.net_return))
    equity = equity[1:]

    peak = equity[0]
    peak_idx = 0
    drawdowns = []  # (start_idx, trough_idx, depth)

    current_dd_start = 0
    current_dd_depth = 0.0
    in_drawdown = False

    for i, eq in enumerate(equity):
        if eq >= peak:
            if in_drawdown:
                drawdowns.append({
                    "start": dates[current_dd_start],
                    "trough": dates[current_dd_start + int(np.argmin(equity[current_dd_start:i + 1]))],
                    "end": dates[i],
                    "depth": round(current_dd_depth * 100, 1),
                    "duration": (dates[i] - dates[current_dd_start]).days,
                })
                in_drawdown = False
                current_dd_depth = 0.0
            peak = eq
            peak_idx = i
        else:
            dd = (peak - eq) / peak
            if not in_drawdown:
                in_drawdown = True
                current_dd_start = peak_idx
            current_dd_depth = max(current_dd_depth, dd)

    # 末尾未結束的 drawdown
    if in_drawdown:
        drawdowns.append({
            "start": dates[current_dd_start],
            "trough": dates[current_dd_start + int(np.argmin(equity[current_dd_start:]))],
            "end": dates[-1],
            "depth": round(current_dd_depth * 100, 1),
            "duration": (dates[-1] - dates[current_dd_start]).days,
        })

    return sorted(drawdowns, key=lambda d: d["depth"], reverse=True)[:top_n]


def analyze_factor_usage(model_infos: list[WeekModelInfo]) -> pd.DataFrame:
    """統計所有模型的因子使用頻率"""
    factor_counts: dict[str, int] = {}
    total_models = 0

    seen_models = set()
    for info in model_infos:
        if info.model_name in seen_models:
            continue
        seen_models.add(info.model_name)
        total_models += 1

        factors_path = MODELS_DIR / info.model_name / "factors.json"
        if factors_path.exists():
            with open(factors_path) as f:
                factors = json.load(f)
            for fac in factors:
                name = fac.get("name", fac.get("expression", "unknown"))
                factor_counts[name] = factor_counts.get(name, 0) + 1

    rows = [{"factor": k, "count": v, "usage_rate": round(v / total_models * 100, 1)}
            for k, v in factor_counts.items()]
    df = pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)
    df["total_models"] = total_models
    return df


def analyze_overfit(
    model_infos: list[WeekModelInfo],
    daily_scores: dict[date, pd.Series],
    close_wide: pd.DataFrame, trading_days: list[date],
) -> dict:
    """Compare validation IC (from DB) vs backtest IC (actual)"""
    import sqlite3
    conn = sqlite3.connect(str(PROJECT_ROOT / "data" / "data.db"))
    db_models = pd.read_sql(
        "SELECT name, model_ic, icir FROM training_runs WHERE status = 'completed'", conn,
    )
    conn.close()

    valid_ic_map = dict(zip(db_models["name"], db_models["model_ic"]))
    day_to_idx = {d: i for i, d in enumerate(trading_days)}

    results = []
    seen = set()
    for info in model_infos:
        if info.model_name in seen:
            continue
        seen.add(info.model_name)

        predict_start, predict_end = get_week_date_range(info.predict_week)
        week_ics = []
        for pred_date in sorted(daily_scores.keys()):
            if pred_date < predict_start or pred_date > predict_end:
                continue
            if pred_date not in day_to_idx:
                continue
            idx = day_to_idx[pred_date]
            if idx + 3 >= len(trading_days):
                continue
            t_buy = trading_days[idx + 1]
            t_sell = trading_days[idx + 3]
            if t_buy not in close_wide.index or t_sell not in close_wide.index:
                continue
            returns = (close_wide.loc[t_sell] - close_wide.loc[t_buy]) / close_wide.loc[t_buy]
            scores = daily_scores[pred_date]
            common = scores.dropna().index.intersection(returns.dropna().index)
            if len(common) < 10:
                continue
            corr, _ = stats.spearmanr(scores[common], returns[common])
            if not np.isnan(corr):
                week_ics.append(corr)

        if week_ics:
            results.append({
                "model": info.model_name,
                "week": info.model_week,
                "valid_ic": valid_ic_map.get(info.model_name),
                "backtest_ic": float(np.mean(week_ics)),
                "n_days": len(week_ics),
            })

    df = pd.DataFrame(results)
    if df.empty:
        return {"df": df, "summary": {}}

    valid = df["valid_ic"].dropna()
    backtest = df["backtest_ic"]
    corr_val = float(valid.corr(backtest)) if len(valid) > 2 else 0

    return {
        "df": df,
        "summary": {
            "avg_valid_ic": float(valid.mean()),
            "avg_backtest_ic": float(backtest.mean()),
            "ic_decay_pct": float((valid.mean() - backtest.mean()) / valid.mean() * 100)
            if valid.mean() != 0 else 0,
            "correlation": corr_val,
            "n_models": len(df),
            "backtest_ic_positive_pct": float((backtest > 0).mean()) * 100,
        },
    }


def analyze_factor_ic(
    features_cache: pd.DataFrame,
    close_wide: pd.DataFrame, trading_days: list[date],
    daily_scores: dict[date, pd.Series],
) -> pd.DataFrame:
    """Compute per-factor cross-sectional Spearman IC"""
    day_to_idx = {d: i for i, d in enumerate(trading_days)}

    fc_dates = features_cache.index.get_level_values("datetime").unique()
    fc_date_map = {}
    for ts in fc_dates:
        d = ts.date() if hasattr(ts, "date") else ts
        fc_date_map[d] = ts

    all_ics = []
    for pred_date in sorted(daily_scores.keys()):
        if pred_date not in day_to_idx or pred_date not in fc_date_map:
            continue
        idx = day_to_idx[pred_date]
        if idx + 2 >= len(trading_days):
            continue

        t_buy = trading_days[idx + 1]
        t_sell = trading_days[idx + 2]
        if t_buy not in close_wide.index or t_sell not in close_wide.index:
            continue

        returns = (close_wide.loc[t_sell] - close_wide.loc[t_buy]) / close_wide.loc[t_buy]

        ts = fc_date_map[pred_date]
        day_features = features_cache.xs(ts, level="datetime")
        day_features = day_features.replace([np.inf, -np.inf], np.nan)

        common = day_features.dropna(how="all").index.intersection(returns.dropna().index)
        if len(common) < 10:
            continue

        ic_series = day_features.loc[common].corrwith(returns[common], method="spearman")
        all_ics.append(ic_series)

    if not all_ics:
        return pd.DataFrame()

    ic_df = pd.DataFrame(all_ics)
    rows = []
    for col in ic_df.columns:
        vals = ic_df[col].dropna()
        if len(vals) < 10:
            continue
        rows.append({
            "factor": col,
            "mean_ic": float(vals.mean()),
            "std_ic": float(vals.std()),
            "icir": float(vals.mean() / vals.std()) if vals.std() > 0 else 0,
            "ic_pos_rate": float((vals > 0).mean()),
            "abs_ic": abs(float(vals.mean())),
            "n_days": len(vals),
        })

    return pd.DataFrame(rows).sort_values("icir", ascending=False).reset_index(drop=True)


def analyze_feature_importance(
    model_infos: list[WeekModelInfo], factor_ic_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compare LightGBM feature importance vs factor IC"""
    importance_sums: dict[str, float] = {}
    n_models = 0

    seen = set()
    for info in model_infos:
        if info.model_name in seen:
            continue
        seen.add(info.model_name)

        try:
            model_dir = MODELS_DIR / info.model_name
            with open(model_dir / "model.pkl", "rb") as f:
                model = pickle.load(f)
            with open(model_dir / "factors.json") as f:
                factors = json.load(f)
        except Exception:
            continue

        names = [fac["name"] for fac in factors]
        importances = model.feature_importance()
        if len(names) != len(importances):
            continue

        n_models += 1
        for name, imp in zip(names, importances):
            importance_sums[name] = importance_sums.get(name, 0) + imp

    if not importance_sums or n_models == 0:
        return pd.DataFrame()

    rows = [{"factor": k, "avg_importance": v / n_models} for k, v in importance_sums.items()]
    imp_df = pd.DataFrame(rows).sort_values("avg_importance", ascending=False).reset_index(drop=True)

    if not factor_ic_df.empty:
        imp_df = imp_df.merge(factor_ic_df[["factor", "mean_ic", "icir"]], on="factor", how="left")

    return imp_df


def analyze_prediction_dispersion(daily_scores: dict[date, pd.Series]) -> dict:
    """Analyze prediction score distribution and separation power"""
    metrics = []
    for d in sorted(daily_scores.keys()):
        scores = daily_scores[d]
        if len(scores) < 20:
            continue

        sorted_scores = scores.sort_values(ascending=False)
        top10_mean = float(sorted_scores.head(10).mean())
        bot10_mean = float(sorted_scores.tail(10).mean())

        metrics.append({
            "date": d,
            "score_range": float(scores.max() - scores.min()),
            "score_std": float(scores.std()),
            "top10_mean": top10_mean,
            "bot10_mean": bot10_mean,
            "top_bot_gap": top10_mean - bot10_mean,
        })

    df = pd.DataFrame(metrics)
    if df.empty:
        return {"df": df, "summary": {}}

    avg_gap = float(df["top_bot_gap"].mean())
    avg_std = float(df["score_std"].mean())

    return {
        "df": df,
        "summary": {
            "avg_range": float(df["score_range"].mean()),
            "avg_std": avg_std,
            "avg_top_bot_gap": avg_gap,
            "gap_to_std_ratio": avg_gap / avg_std if avg_std > 0 else 0,
        },
    }


# ══════════════════════════════════════════════════════════════
# 圖表
# ══════════════════════════════════════════════════════════════

def plot_ic_horizon(h_result: dict, output_dir: Path) -> None:
    horizons = sorted(h_result["horizons"].keys())
    ics = [h_result["horizons"][h]["ic"] for h in horizons]
    icirs = [h_result["horizons"][h]["icir"] for h in horizons]
    labels = [f"{h}d" for h in horizons]
    optimal = h_result["optimal"]
    colors = ["#e74c3c" if h == optimal else "#3498db" for h in horizons]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle("Multi-Horizon IC Analysis", fontsize=14)

    ax1.bar(labels, ics, color=colors)
    ax1.set_ylabel("Mean IC")
    ax1.set_title(f"Mean IC by Horizon (optimal: {optimal}-day)")
    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax1.grid(True, alpha=0.3)

    ax2.bar(labels, icirs, color=colors)
    ax2.set_ylabel("ICIR")
    ax2.set_title("ICIR by Horizon")
    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "ic_horizon.png", dpi=150)
    plt.close()


def plot_rolling_ic(r_result: dict, output_dir: Path) -> None:
    daily_ic = r_result["daily_ic"]
    rolling_20 = r_result["rolling_20"]

    fig, ax = plt.subplots(figsize=(16, 6))
    fig.suptitle("Rolling IC & Structural Breaks", fontsize=14)

    ax.scatter(daily_ic.index, daily_ic.values, alpha=0.15, s=8, color="gray", label="Daily IC")
    ax.plot(rolling_20.index, rolling_20.values, color="steelblue", linewidth=1.5, label="20d Rolling Mean")
    ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)
    ax.axhline(y=r_result["mean_ic"], color="red", linestyle="--", alpha=0.5,
               label=f"Mean={r_result['mean_ic']:.4f}")

    for bi in r_result["breaks"]:
        if bi < len(daily_ic):
            ax.axvline(x=daily_ic.index[bi], color="red", linestyle=":", alpha=0.7)

    ax.set_ylabel("Spearman IC")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "rolling_ic.png", dpi=150)
    plt.close()


def plot_equity_curve(
    results: list[StrategyResult], all_metrics: list[Metrics], output_dir: Path,
) -> None:
    # 取 Sharpe 最高的 5 個策略
    sorted_m = sorted(
        [m for m in all_metrics if m.sharpe is not None],
        key=lambda m: m.sharpe, reverse=True,
    )[:5]
    top_names = {m.strategy for m in sorted_m}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    fig.suptitle("Top-5 Strategy Equity Curves (Fixed Mode)", fontsize=14)

    for result in results:
        if result.name not in top_names or not result.records:
            continue
        dates = [r.dt for r in result.records]
        cum = [1.0]
        for r in result.records:
            cum.append(cum[-1] * (1 + r.net_return))
        ax1.plot(dates, cum[1:], label=result.name, alpha=0.8)

    # 大盤基準
    bench = next((r for r in results if r.records), None)
    if bench:
        dates = [r.dt for r in bench.records]
        bench_cum = [1.0]
        for r in bench.records:
            bench_cum.append(bench_cum[-1] * (1 + r.market_return))
        ax1.plot(dates, bench_cum[1:], label="Market", color="black", linestyle="--", linewidth=2, alpha=0.5)

    ax1.set_ylabel("Cumulative Return")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Drawdown
    for result in results:
        if result.name not in top_names or not result.records:
            continue
        dates = [r.dt for r in result.records]
        equity = 1.0
        peak_val = 1.0
        dd_list = []
        for r in result.records:
            equity *= (1 + r.net_return)
            peak_val = max(peak_val, equity)
            dd_list.append(-(peak_val - equity) / peak_val * 100)
        ax2.plot(dates, dd_list, label=result.name, alpha=0.8)

    ax2.set_ylabel("Drawdown (%)")
    ax2.legend(fontsize=8, loc="lower left")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "equity_curve.png", dpi=150)
    plt.close()


def plot_monthly_heatmap(monthly_df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))

    data = monthly_df.values
    vmax = max(abs(np.nanmin(data)), abs(np.nanmax(data)), 1)

    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(data.shape[1]))
    ax.set_xticklabels(monthly_df.columns)
    ax.set_yticks(range(data.shape[0]))
    ax.set_yticklabels([str(y) for y in monthly_df.index])

    # 在每格中標數值
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if not np.isnan(val):
                color = "white" if abs(val) > vmax * 0.6 else "black"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=9, color=color)

    ax.set_xlabel("Month")
    ax.set_ylabel("Year")
    ax.set_title("Monthly Avg Daily Excess Return (bps) — Best Strategy")
    fig.colorbar(im, ax=ax, label="bps")

    plt.tight_layout()
    plt.savefig(output_dir / "monthly_heatmap.png", dpi=150)
    plt.close()


# ══════════════════════════════════════════════════════════════
# 報告
# ══════════════════════════════════════════════════════════════

def write_report(
    output_dir: Path, run_hash: str, elapsed: float,
    h_result: dict, r_result: dict, reg_result: dict,
    q_result: dict, s_result: dict,
    monthly_df: pd.DataFrame, yearly_df: pd.DataFrame,
    drawdowns: list[dict], factor_df: pd.DataFrame,
    all_metrics: list[Metrics], best_strategy: str,
    overfit_result: dict = None, factor_ic_df: pd.DataFrame = None,
    feat_imp_df: pd.DataFrame = None, dispersion_result: dict = None,
) -> None:
    lines = []

    # Header
    from src.repositories.factors import ALL_FACTORS
    lines.append("# Model Evaluation Report\n")
    lines.append(f"**Run**: `{run_hash}` | **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Config**: label=`{LABEL_EXPR}`, factors={len(ALL_FACTORS)}, "
                 f"range={START_MODEL_WEEK}~{END_MODEL_WEEK}")
    lines.append(f"**Runtime**: {elapsed:.1f}s\n")

    # Executive Summary
    optimal_h = h_result["optimal"]
    best_m = next((m for m in all_metrics if m.strategy == best_strategy), None)
    lines.append("## Executive Summary\n")
    lines.append(f"- **信號品質**: mono rho={q_result['mono_rho']:.2f}, "
                 f"spread={q_result['spread_bps']:.1f} bps (t={q_result['spread_t']:.2f})")
    lines.append(f"- **最佳 horizon**: {optimal_h}-day "
                 f"(IC={h_result['horizons'].get(optimal_h, {}).get('ic', 0):.4f})")
    lines.append(f"- **IC 穩定性**: ICIR={r_result['icir']:.2f}, IC>0={r_result['ic_pos_rate']:.1%}")
    lines.append(f"- **分數區分度**: Top-K overlap={s_result['topk_overlap']:.1%}, "
                 f"tau={s_result['kendall_tau']:.3f}")
    if best_m:
        lines.append(f"- **最佳策略**: {best_strategy} "
                     f"(excess={best_m.ann_excess:+.1f}%, sharpe={best_m.sharpe}, "
                     f"win={best_m.win_rate:.1f}%)")
    if overfit_result and overfit_result.get("summary"):
        o = overfit_result["summary"]
        lines.append(f"- **過擬合**: Valid IC={o['avg_valid_ic']:.4f} → "
                     f"Backtest IC={o['avg_backtest_ic']:.4f} (衰減 {o['ic_decay_pct']:.0f}%)")
    if dispersion_result and dispersion_result.get("summary"):
        p = dispersion_result["summary"]
        lines.append(f"- **預測分散度**: Top-Bot gap/std={p['gap_to_std_ratio']:.2f}")
    lines.append("")

    # 1. Multi-Horizon IC
    lines.append("---\n\n## 1. Multi-Horizon IC\n")
    lines.append("| Horizon | Mean IC | ICIR | IC>0% |")
    lines.append("|---------|---------|------|-------|")
    for h in sorted(h_result["horizons"].keys()):
        d = h_result["horizons"][h]
        marker = " **" if h == optimal_h else ""
        lines.append(f"| {h}-day{marker} | {d['ic']:.4f} | {d['icir']:.2f} | {d['pos_rate']:.1%} |")
    lines.append("")

    # 2. Rolling IC
    lines.append("---\n\n## 2. Rolling IC\n")
    lines.append(f"- Overall: mean IC={r_result['mean_ic']:.4f}, ICIR={r_result['icir']:.2f}, "
                 f"IC>0={r_result['ic_pos_rate']:.1%}")
    lines.append(f"- CUSUM breaks: {len(r_result['breaks'])}\n")
    lines.append("| Year | Mean IC |")
    lines.append("|------|---------|")
    for y in sorted(r_result["by_year"].keys()):
        lines.append(f"| {y} | {r_result['by_year'][y]:.4f} |")
    lines.append("")
    lines.append("| Quarter | Mean IC |")
    lines.append("|---------|---------|")
    for q in sorted(r_result["by_quarter"].keys()):
        lines.append(f"| {q} | {r_result['by_quarter'][q]:.4f} |")
    lines.append("")

    # 3. Market Regime
    lines.append("---\n\n## 3. Market Regime\n")
    lines.append("| Regime | Days | Mean IC | Excess (bps) | Win% |")
    lines.append("|--------|------|---------|-------------|------|")
    for regime in ["bull", "sideways", "bear"]:
        s = reg_result["stats"][regime]
        lines.append(f"| {regime} | {s['n_days']} | {s['mean_ic']:.4f} | "
                     f"{s['mean_excess']:.1f} | {s['win_rate']:.1f}% |")
    lines.append(f"\nRegime transitions: {reg_result['transitions']}")
    lines.append("")

    # 4. Monthly Excess Heatmap
    lines.append("---\n\n## 4. Monthly Excess Return (bps)\n")
    lines.append(f"Strategy: {best_strategy}\n")
    months = monthly_df.columns.tolist()
    lines.append("| Year | " + " | ".join(months) + " |")
    lines.append("|------|" + "|".join(["------"] * len(months)) + "|")
    for year in monthly_df.index:
        vals = [f"{monthly_df.loc[year, m]:.1f}" if not np.isnan(monthly_df.loc[year, m]) else "-"
                for m in months]
        lines.append(f"| {year} | " + " | ".join(vals) + " |")
    lines.append("")

    # 5. Yearly Performance
    lines.append("---\n\n## 5. Yearly Performance\n")
    lines.append("| Year | Excess% | Sharpe | Win% | MaxDD% | Days |")
    lines.append("|------|---------|--------|------|--------|------|")
    for year in yearly_df.index:
        r = yearly_df.loc[year]
        lines.append(f"| {year} | {r['ann_excess']:+.1f} | {r['sharpe']:.2f} | "
                     f"{r['win_rate']:.1f} | {r['max_dd']:.1f} | {r['n_days']} |")
    lines.append("")

    # 6. Top Drawdowns
    lines.append("---\n\n## 6. Top-5 Drawdowns\n")
    lines.append("| # | Start | Trough | End | Depth% | Duration |")
    lines.append("|---|-------|--------|-----|--------|----------|")
    for i, dd in enumerate(drawdowns):
        lines.append(f"| {i+1} | {dd['start']} | {dd['trough']} | {dd['end']} | "
                     f"{dd['depth']:.1f} | {dd['duration']}d |")
    lines.append("")

    # 7. Factor Usage
    lines.append("---\n\n## 7. Factor Usage (Top-20)\n")
    total_models = int(factor_df["total_models"].iloc[0]) if len(factor_df) > 0 else 0
    lines.append(f"Total unique models: {total_models}\n")
    lines.append("| # | Factor | Count | Usage% |")
    lines.append("|---|--------|-------|--------|")
    for i, row in factor_df.head(20).iterrows():
        lines.append(f"| {i+1} | {row['factor']} | {row['count']} | {row['usage_rate']}% |")
    lines.append("")

    # 8. Strategy Comparison (Top-10)
    lines.append("---\n\n## 8. Strategy Comparison (Top-10 by Sharpe)\n")
    sorted_m = sorted(
        [m for m in all_metrics if m.sharpe is not None],
        key=lambda m: m.sharpe, reverse=True,
    )[:10]
    lines.append("| Strategy | Excess% | Sharpe | MaxDD% | Win% | Turn% | IR | t-stat |")
    lines.append("|----------|---------|--------|--------|------|-------|-----|--------|")
    for m in sorted_m:
        ir_str = f"{m.info_ratio:.2f}" if m.info_ratio else "N/A"
        t_str = f"{m.t_stat:.2f}" if m.t_stat else "N/A"
        lines.append(f"| {m.strategy} | {m.ann_excess:+.1f} | {m.sharpe:.3f} | "
                     f"{m.max_drawdown:.1f} | {m.win_rate:.1f} | {m.avg_turnover:.1f} | "
                     f"{ir_str} | {t_str} |")
    lines.append("")

    # 9. Score Quality
    lines.append("---\n\n## 9. Score Quality\n")
    lines.append(f"- Unique scores/day: {s_result['avg_unique']:.0f}")
    lines.append(f"- Score std: {s_result['avg_std']:.4f}")
    lines.append(f"- Top-{TOP_K} overlap: {s_result['topk_overlap']:.1%}")
    lines.append(f"- Kendall tau: {s_result['kendall_tau']:.3f}")
    lines.append("")

    # 10. Overfit Diagnostics
    if overfit_result and overfit_result.get("summary"):
        o = overfit_result["summary"]
        lines.append("---\n\n## 10. Overfit Diagnostics\n")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Avg Validation IC | {o['avg_valid_ic']:.4f} |")
        lines.append(f"| Avg Backtest IC | {o['avg_backtest_ic']:.4f} |")
        lines.append(f"| IC Decay | {o['ic_decay_pct']:.1f}% |")
        lines.append(f"| Valid-Backtest Correlation | {o['correlation']:.3f} |")
        lines.append(f"| Models with Backtest IC > 0 | {o['backtest_ic_positive_pct']:.1f}% |")
        lines.append(f"| Total Models | {o['n_models']} |")
        lines.append("")

    # 11. Factor IC
    if factor_ic_df is not None and not factor_ic_df.empty:
        n_pos = int((factor_ic_df["mean_ic"] > 0).sum())
        n_total = len(factor_ic_df)
        lines.append("---\n\n## 11. Factor IC\n")
        lines.append(f"{n_total} factors analyzed, {n_pos} with positive mean IC\n")
        lines.append("### Top-10 by ICIR\n")
        lines.append("| # | Factor | Mean IC | ICIR | IC>0% |")
        lines.append("|---|--------|---------|------|-------|")
        for i, (_, row) in enumerate(factor_ic_df.head(10).iterrows()):
            lines.append(f"| {i+1} | {row['factor']} | {row['mean_ic']:.4f} | "
                         f"{row['icir']:.3f} | {row['ic_pos_rate']:.1%} |")
        lines.append("")
        lines.append("### Bottom-10 by ICIR\n")
        lines.append("| # | Factor | Mean IC | ICIR | IC>0% |")
        lines.append("|---|--------|---------|------|-------|")
        for i, (_, row) in enumerate(factor_ic_df.tail(10).iterrows()):
            lines.append(f"| {i+1} | {row['factor']} | {row['mean_ic']:.4f} | "
                         f"{row['icir']:.3f} | {row['ic_pos_rate']:.1%} |")
        lines.append("")

    # 12. Feature Importance vs IC
    if feat_imp_df is not None and not feat_imp_df.empty:
        lines.append("---\n\n## 12. Feature Importance vs IC (Top-20)\n")
        lines.append("| # | Factor | Importance | Mean IC | ICIR |")
        lines.append("|---|--------|------------|---------|------|")
        for i, (_, row) in enumerate(feat_imp_df.head(20).iterrows()):
            ic_str = f"{row['mean_ic']:.4f}" if pd.notna(row.get("mean_ic")) else "N/A"
            icir_str = f"{row['icir']:.3f}" if pd.notna(row.get("icir")) else "N/A"
            lines.append(f"| {i+1} | {row['factor']} | {row['avg_importance']:.1f} | "
                         f"{ic_str} | {icir_str} |")
        lines.append("")

    # 13. Prediction Dispersion
    if dispersion_result and dispersion_result.get("summary"):
        p = dispersion_result["summary"]
        lines.append("---\n\n## 13. Prediction Dispersion\n")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Avg Score Range | {p['avg_range']:.4f} |")
        lines.append(f"| Avg Score Std | {p['avg_std']:.4f} |")
        lines.append(f"| Avg Top10-Bot10 Gap | {p['avg_top_bot_gap']:.4f} |")
        lines.append(f"| Gap/Std Ratio | {p['gap_to_std_ratio']:.2f} |")
        lines.append("")

    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Report: {report_path}")


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    run_hash = compute_run_hash()
    output_dir = OUTPUT_BASE / run_hash
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Model Evaluation — run {run_hash}")
    print(f"Range: {START_MODEL_WEEK} ~ {END_MODEL_WEEK}")
    print(f"Output: {output_dir}")
    print("=" * 70)

    # 1. 收集模型
    print("\n[1/7] Collecting models...")
    model_infos = collect_models(START_MODEL_WEEK, END_MODEL_WEEK)
    print(f"  {len(model_infos)} weeks")
    fallback_count = sum(1 for m in model_infos if m.is_fallback)
    if fallback_count:
        print(f"  {fallback_count} fallback models")

    if not model_infos:
        print("ERROR: No models found!")
        return

    # 2. 導出 + 預載資料
    print("\n[2/7] Loading data...")
    first_predict_start = get_week_date_range(model_infos[0].predict_week)[0]
    last_predict_end = get_week_date_range(model_infos[-1].predict_week)[1]

    export_start = first_predict_start - timedelta(days=LOOKBACK_DAYS)
    export_end = last_predict_end

    print(f"  Exporting qlib: {export_start} ~ {export_end}")
    export_qlib_data(export_start, export_end)

    print("  Initializing qlib...")
    init_qlib()
    instruments = get_instruments()
    print(f"  {len(instruments)} instruments")

    _, first_factors, _ = load_model(model_infos[0].model_name)
    features_cache, prices_cache = preload_data(
        first_factors, instruments, first_predict_start, last_predict_end,
    )

    close_wide = prices_cache["close"].unstack(level="instrument")
    trading_days = sorted([d.date() if hasattr(d, 'date') else d for d in close_wide.index])
    close_wide.index = trading_days

    # Extended prices for multi-horizon IC
    print("  Loading extended prices...")
    ext_close, ext_days = load_extended_prices(instruments, first_predict_start, last_predict_end)

    print(f"  {len(trading_days)} trading days")

    # 3. 產生分數
    print("\n[3/7] Generating daily scores...")
    daily_scores = generate_daily_scores(model_infos, features_cache)
    print(f"  {len(daily_scores)} days with scores")

    # 4. 策略回測
    print("\n[4/7] Running strategy backtests...")
    strategies = build_strategies()
    hedged_strategies = build_hedged_strategies()
    total_count = len(strategies) + len(hedged_strategies)
    all_results: list[StrategyResult] = []
    all_metrics: list[Metrics] = []

    for i, strategy in enumerate(strategies):
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{total_count}] {strategy.name()}...")
        result = simulate(strategy, daily_scores, close_wide, trading_days)
        all_results.append(result)
        all_metrics.append(compute_metrics(result))

    for i, (strategy, hc) in enumerate(hedged_strategies):
        idx = len(strategies) + i + 1
        if idx % 10 == 0:
            print(f"  [{idx}/{total_count}] {strategy.name()}+SMA{hc.sma_period}...")
        result = simulate(strategy, daily_scores, close_wide, trading_days, hedge_config=hc)
        all_results.append(result)
        all_metrics.append(compute_metrics(result))

    print(f"  Total: {total_count} strategies ({len(strategies)} base + {len(hedged_strategies)} hedged)")

    # 找最佳策略
    best_m = max(
        [m for m in all_metrics if m.sharpe is not None],
        key=lambda m: m.sharpe,
    )
    best_strategy = best_m.strategy
    best_result = next(r for r in all_results if r.name == best_strategy)
    print(f"  Best: {best_strategy} (Sharpe={best_m.sharpe}, Excess={best_m.ann_excess}%)")

    # 5. 診斷分析
    print("\n[5/7] Running diagnostics...")
    print("  Multi-horizon IC...")
    h_result = analyze_multi_horizon_ic(daily_scores, ext_close, ext_days)
    print("  Rolling IC...")
    r_result = analyze_rolling_ic(daily_scores, ext_close, ext_days)
    print("  Market regimes...")
    reg_result = analyze_market_regimes(
        daily_scores, close_wide, trading_days, r_result["daily_ic"], best_result.records,
    )
    print("  Quantile spread...")
    q_result = analyze_quantile_spread(daily_scores, close_wide, trading_days)
    print("  Score quality...")
    s_result = analyze_score_quality(daily_scores)

    # 6. 新增分析
    print("\n[6/7] Additional analyses...")
    print("  Monthly excess...")
    monthly_df = analyze_monthly_excess(best_result.records)
    print("  Yearly breakdown...")
    yearly_df = analyze_yearly_breakdown(best_result.records)
    print("  Top drawdowns...")
    drawdowns = analyze_top_drawdowns(best_result.records)
    print("  Factor usage...")
    factor_df = analyze_factor_usage(model_infos)
    print("  Overfit diagnostics...")
    overfit_result = analyze_overfit(model_infos, daily_scores, close_wide, trading_days)
    print("  Factor IC...")
    factor_ic_df = analyze_factor_ic(features_cache, close_wide, trading_days, daily_scores)
    del features_cache
    print("  Feature importance vs IC...")
    feat_imp_df = analyze_feature_importance(model_infos, factor_ic_df)
    print("  Prediction dispersion...")
    dispersion_result = analyze_prediction_dispersion(daily_scores)

    # 7. 輸出
    print("\n[7/7] Generating outputs...")

    # CSV
    rows = [{
        "strategy": m.strategy, "ann_return": m.ann_return, "ann_excess": m.ann_excess,
        "sharpe": m.sharpe, "max_drawdown": m.max_drawdown, "calmar": m.calmar,
        "win_rate": m.win_rate, "avg_turnover": m.avg_turnover, "total_cost": m.total_cost,
        "info_ratio": m.info_ratio, "t_stat": m.t_stat, "n_days": m.n_days,
    } for m in all_metrics]
    csv_path = output_dir / "strategy_metrics.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  CSV: {csv_path}")

    # 圖表
    plot_ic_horizon(h_result, output_dir)
    print("  ic_horizon.png")
    plot_rolling_ic(r_result, output_dir)
    print("  rolling_ic.png")
    plot_equity_curve(all_results, all_metrics, output_dir)
    print("  equity_curve.png")
    plot_monthly_heatmap(monthly_df, output_dir)
    print("  monthly_heatmap.png")

    # 報告
    elapsed = time.time() - t_start
    write_report(
        output_dir, run_hash, elapsed,
        h_result, r_result, reg_result, q_result, s_result,
        monthly_df, yearly_df, drawdowns, factor_df,
        all_metrics, best_strategy,
        overfit_result=overfit_result, factor_ic_df=factor_ic_df,
        feat_imp_df=feat_imp_df, dispersion_result=dispersion_result,
    )

    # Console summary
    print(f"\nDone in {elapsed:.1f}s")
    print(f"Output: {output_dir}")
    print(f"\nTop-5 strategies:")
    sorted_m = sorted(
        [m for m in all_metrics if m.sharpe is not None],
        key=lambda m: m.sharpe, reverse=True,
    )[:5]
    for m in sorted_m:
        print(f"  {m.strategy:<35} Excess={m.ann_excess:+6.1f}%  Sharpe={m.sharpe:.3f}  "
              f"Win={m.win_rate:.1f}%  Turn={m.avg_turnover:.1f}%")


if __name__ == "__main__":
    main()
