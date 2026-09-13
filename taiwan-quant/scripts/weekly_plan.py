#!/usr/bin/env python3
"""
端到端週頻計畫產生器

把所有模組串起來，在真實台股資料上跑出一份週頻交易計畫：

    資料載入 → 特徵 → 柵欄推導 → triple-barrier 標記
        → 訓練/預測 → Top 3 選股 → 部位規模 → Telegram 推播

**這一版刻意不用 ML 模型。** 先用一個透明的規則式基準（動能分位）當
`P(+1)` 的代理，確認整條管線接得起來、數字合理。有了可信的 baseline
再談 LightGBM——CLAUDE.md：先有 baseline 再談複雜度。

用法：
    .venv/bin/python scripts/weekly_plan.py                 # dry-run，只印不發
    .venv/bin/python scripts/weekly_plan.py --send          # 實際推播
    .venv/bin/python scripts/weekly_plan.py --stocks 2330 2317 2454
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from taiwan_quant.config.costs import DEFAULT, Tier  # noqa: E402
from taiwan_quant.data.loader import load_prices, load_universe  # noqa: E402
from taiwan_quant.data.universe_history import (  # noqa: E402
    Provenance,
    SnapshotStore,
    resolve_universe,
)
from taiwan_quant.features.technical import build_technical  # noqa: E402
from taiwan_quant.labeling.barrier_width import derive_width  # noqa: E402
from taiwan_quant.labeling.triple_barrier import label_one  # noqa: E402
from taiwan_quant.notify.telegram import (  # noqa: E402
    TelegramConfig,
    TelegramNotifier,
    format_weekly_plan,
)
from taiwan_quant.ranking.portfolio import Candidate, select_portfolio  # noqa: E402
from taiwan_quant.validation.calibration import (  # noqa: E402
    CalibrationError,
    Calibrator,
    fit_calibrator,
    reliability_report,
)

STRATEGY_VERSION = "momentum_baseline_tb5d@v0.2.0"

CALIBRATION_BINS = 8
CALIBRATION_MIN_SAMPLES = 30
CALIBRATION_TRAIN_END = date(2025, 12, 31)
"""
校準只能用這天之前的資料擬合。

用全樣本擬合等於讓系統知道未來的命中率分布（CLAUDE.md 禁令 1）。
"""

HORIZON = 5
ATR_MULTIPLE = 0.8
TARGET_QUANTILE = 0.85
MIN_RISK_REWARD = 2.0
CORRELATION_WINDOW = 60

CAPITAL_DEFAULT = 400_000.0

# 產業對照（MVP：只涵蓋常見標的，缺的歸為「其他」）
INDUSTRY_MAP: dict[str, str] = {
    "2330": "半導體", "2454": "半導體", "2303": "半導體", "3711": "半導體",
    "2379": "半導體", "3034": "半導體", "2408": "半導體", "3443": "半導體",
    "2317": "電子組裝", "2382": "電子組裝", "3231": "電子組裝", "2356": "電子組裝",
    "2308": "電子零件", "2327": "電子零件", "2383": "電子零件", "2345": "網通",
    "2881": "金融", "2882": "金融", "2884": "金融", "2885": "金融",
    "2886": "金融", "2887": "金融", "2891": "金融", "2892": "金融",
    "1301": "塑化", "1303": "塑化", "6505": "塑化", "1326": "塑化",
    "2002": "鋼鐵", "2603": "航運", "2609": "航運", "2615": "航運",
    "3045": "電信", "4904": "電信", "2412": "電信",
    "1216": "食品", "2912": "零售",
}


@dataclass(frozen=True)
class StockSignal:
    """單一標的的訊號與柵欄"""

    stock_id: str
    prob_up: float
    target_pct: float
    stop_pct: float
    entry_price: float
    volatility_pct: float
    beta: float
    decision_date: pd.Timestamp


def momentum_score(features: pd.DataFrame) -> float:
    """
    規則式基準的 `P(+1)` 代理。

    **這不是模型，是基準。** 用三段動能與量能的加權分數，映射到 [0, 1]。

    為什麼先做這個：
      1. 完全透明，任何異常一眼看得出來
      2. 沒有訓練，不可能過擬合
      3. 提供 ML 模型必須超越的地板——跑不贏這個就不必談複雜模型
    """
    row = features.iloc[-1]
    parts = [
        np.tanh(row["momentum_20"] * 8),
        np.tanh(row["momentum_60"] * 4),
        np.tanh(row["ma_ratio_20"] * 20),
        np.tanh((row["volume_ratio_20"] - 1.0) * 2),
    ]
    if any(pd.isna(p) for p in parts):
        return float("nan")
    # tanh 輸出在 [-1, 1]，平均後線性映射到 [0.2, 0.8]
    return float(0.5 + 0.3 * np.mean(parts))


def compute_beta(stock_returns: pd.Series, market_returns: pd.Series) -> float:
    """對等權市場組合的 beta"""
    aligned = pd.concat([stock_returns, market_returns], axis=1, sort=True).dropna()
    if len(aligned) < 30:
        return float("nan")
    cov = aligned.cov().iloc[0, 1]
    var = aligned.iloc[:, 1].var()
    return float(cov / var) if var > 0 else float("nan")


def build_signal(
    stock_id: str,
    bars: pd.DataFrame,
    market_returns: pd.Series,
    atr_percentiles: dict[str, float],
    calibrator: Calibrator,
) -> StockSignal | None:
    """
    對單一標的產生訊號；任一環節資料不足就回 None。

    分數必須經過校準才能當 `P(+1)` 使用。校準器回 None（樣本不足或
    分數超出訓練範圍）時整筆訊號作廢——「無法評估」不等於「機率很低」。
    """
    if len(bars) < 200:
        return None

    features = build_technical(bars)
    raw_score = momentum_score(features)
    if not np.isfinite(raw_score):
        return None

    prob = calibrator.predict(raw_score)
    if prob is None:
        return None

    decision_idx = len(bars) - 1
    width = derive_width(
        bars,
        decision_idx=decision_idx,
        horizon=HORIZON,
        atr_multiple=ATR_MULTIPLE,
        target_quantile=TARGET_QUANTILE,
        min_risk_reward=MIN_RISK_REWARD,
    )
    if width is None:
        return None

    beta = compute_beta(bars["close"].pct_change(), market_returns)
    if not np.isfinite(beta):
        return None

    return StockSignal(
        stock_id=stock_id,
        prob_up=prob,
        target_pct=width.target_pct,
        stop_pct=width.stop_pct,
        entry_price=float(bars["close"].iloc[-1]),
        volatility_pct=atr_percentiles.get(stock_id, 0.5),
        beta=beta,
        decision_date=bars.index[-1],
    )


def build_calibration_set(
    bars_by_stock: dict[str, pd.DataFrame],
    train_end: date,
) -> tuple[np.ndarray, np.ndarray]:
    """
    蒐集訓練期的（分數, triple-barrier 標籤）配對，供校準使用。

    只走訓練期資料（`train_end` 之前的決策日），避免 look-ahead。

    對每個決策日：
        1. 用**該日之前**的資料算特徵與分數
        2. 用同一日的柵欄推導產生 target/stop
        3. 標記實際結果
    """
    scores: list[float] = []
    labels: list[int] = []

    for bars in bars_by_stock.values():
        feature_frame = build_technical(bars)
        train_mask = bars.index.date <= train_end

        # 從第 200 根開始（特徵視窗需要），到訓練期結束
        for idx in range(200, len(bars)):
            if not train_mask[idx]:
                break

            score = momentum_score(feature_frame.iloc[: idx + 1])
            if not np.isfinite(score):
                continue

            width = derive_width(
                bars,
                decision_idx=idx,
                horizon=HORIZON,
                atr_multiple=ATR_MULTIPLE,
                target_quantile=TARGET_QUANTILE,
                min_risk_reward=MIN_RISK_REWARD,
            )
            if width is None:
                continue

            outcome = label_one(
                bars,
                decision_idx=idx,
                target_pct=width.target_pct,
                stop_pct=width.stop_pct,
                horizon=HORIZON,
            )
            if outcome is None:
                continue

            scores.append(score)
            labels.append(outcome.label)

    return np.asarray(scores), np.asarray(labels)


def compute_correlations(
    returns_by_stock: dict[str, pd.Series], window: int
) -> dict[tuple[str, str], float]:
    """兩兩之間的 N 日報酬相關係數"""
    frame = pd.DataFrame(returns_by_stock).tail(window)
    matrix = frame.corr()
    ids = list(matrix.columns)
    return {
        (a, b): float(matrix.loc[a, b])
        for i, a in enumerate(ids)
        for b in ids[i + 1:]
        if np.isfinite(matrix.loc[a, b])
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="週頻交易計畫產生器")
    parser.add_argument("--stocks", nargs="*", default=None, help="預設用標的池")
    parser.add_argument("--capital", type=float, default=CAPITAL_DEFAULT)
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2026-09-11")
    parser.add_argument("--send", action="store_true", help="實際推播（預設 dry-run）")
    parser.add_argument("--limit", type=int, default=60, help="最多處理幾檔")
    args = parser.parse_args()

    print("=" * 72)
    print("週頻交易計畫產生器")
    print("=" * 72)
    print(f"策略版本  {STRATEGY_VERSION}")
    print(f"資金      {args.capital:,.0f} TWD")
    print()

    # ── 標的池 ──
    store = SnapshotStore()
    as_of = date.fromisoformat(args.end)
    universe_note = ""

    if args.stocks:
        stock_ids = args.stocks
        tiers = {sid: Tier.LARGE for sid in stock_ids}
        universe_note = "標的池：使用者指定"
    else:
        base = load_universe(limit=150)
        resolution = resolve_universe(
            as_of,
            store=store,
            market_caps={
                row.stock_id: float(row.market_cap)
                for row in base.stocks.itertuples()
            },
        )
        available = set(base.stock_ids)
        stock_ids = [s for s in resolution.stock_ids if s in available][: args.limit]
        tiers = {
            sid: Tier(resolution.tier_of(sid)) for sid in stock_ids
        }
        provenance = "真實成分股快照" if resolution.provenance is Provenance.REAL else "市值排名代理"
        universe_note = f"標的池：{provenance}（{len(stock_ids)} 檔）"

        print(f"標的池來源  {resolution.provenance.value}")
        if resolution.snapshot_date:
            print(f"快照日期    {resolution.snapshot_date}")
        for warning in resolution.warnings:
            print(f"  ⚠️  {warning}")
        print()

    # ── 資料 ──
    print(f"載入 {len(stock_ids)} 檔日 K ...")
    prices = load_prices(
        stock_ids,
        start=date.fromisoformat(args.start),
        end=as_of,
        adjusted=True,
    )
    available_ids = [
        sid for sid in stock_ids
        if sid in prices.index.get_level_values("stock_id")
    ]
    bars_by_stock = {
        sid: prices.xs(sid, level="stock_id") for sid in available_ids
    }
    print(f"實際取得 {len(bars_by_stock)} 檔")
    print()

    # ── 市場基準與波動度分位 ──
    returns_by_stock = {
        sid: bars["close"].pct_change() for sid, bars in bars_by_stock.items()
    }
    market_returns = pd.DataFrame(returns_by_stock).mean(axis=1)

    atr_ratios: dict[str, float] = {}
    for sid, bars in bars_by_stock.items():
        feature_frame = build_technical(bars)
        value = feature_frame["true_range_ratio_14"].iloc[-1]
        if np.isfinite(value):
            atr_ratios[sid] = float(value)

    if atr_ratios:
        series = pd.Series(atr_ratios)
        atr_percentiles = series.rank(pct=True).to_dict()
    else:
        atr_percentiles = {}

    # ── 機率校準（只用訓練期資料）──
    print(f"擬合校準器（訓練期至 {CALIBRATION_TRAIN_END}）...")
    cal_scores, cal_labels = build_calibration_set(bars_by_stock, CALIBRATION_TRAIN_END)
    print(f"  校準樣本 {len(cal_scores):,} 筆")

    try:
        calibrator = fit_calibrator(
            cal_scores,
            cal_labels,
            n_bins=CALIBRATION_BINS,
            min_samples_per_bin=CALIBRATION_MIN_SAMPLES,
        )
    except CalibrationError as exc:
        print(f"  ✗ 校準失敗：{exc}")
        print("  未校準的分數不是機率，拒絕產出計畫。")
        return

    print()
    print(reliability_report(calibrator))
    print()

    # ── 訊號 ──
    print("產生訊號 ...")
    signals: list[StockSignal] = []
    skipped: dict[str, int] = {"資料不足": 0, "柵欄或校準失敗": 0}

    for sid, bars in bars_by_stock.items():
        signal = build_signal(sid, bars, market_returns, atr_percentiles, calibrator)
        if signal is None:
            key = "資料不足" if len(bars) < 200 else "柵欄或校準失敗"
            skipped[key] += 1
            continue
        signals.append(signal)

    print(f"  產生 {len(signals)} 個訊號｜略過 {skipped}")
    print()

    if not signals:
        print("無任何訊號，結束。")
        return

    # ── 候選 ──
    candidates = [
        Candidate(
            stock_id=s.stock_id,
            prob_up=s.prob_up,
            target_pct=s.target_pct,
            stop_pct=s.stop_pct,
            entry_price=s.entry_price,
            tier=tiers.get(s.stock_id, Tier.MID),
            industry=INDUSTRY_MAP.get(s.stock_id, "其他"),
            volatility_pct=s.volatility_pct,
            beta=s.beta,
        )
        for s in signals
    ]

    correlations = compute_correlations(
        {s.stock_id: returns_by_stock[s.stock_id] for s in signals},
        CORRELATION_WINDOW,
    )

    # ── 選股 ──
    result = select_portfolio(candidates, args.capital, correlations, cost=DEFAULT)

    print("─" * 72)
    print("選股結果")
    print("─" * 72)
    print(result.describe())
    print()

    reasons: dict[str, int] = {}
    for rejection in result.rejected:
        reasons[rejection.reason.value] = reasons.get(rejection.reason.value, 0) + 1
    print(f"剔除 {len(result.rejected)} 檔：{reasons}")
    print()

    # ── 訊息 ──
    decision_date = signals[0].decision_date
    week_id = f"{decision_date.year}W{decision_date.isocalendar().week:02d}"

    message = format_weekly_plan(
        result,
        week_id=week_id,
        strategy_version=STRATEGY_VERSION,
        data_as_of=str(decision_date.date()),
        universe_note=universe_note,
        benchmark_lines=[
            "⚠️ 對照組尚未接入：本版為規則式基準，未跑 Walk-Forward",
            "　 正式推播前必須並列 買進持有 / 0050 / 隨機進場",
        ],
    )

    notifier = TelegramNotifier(TelegramConfig.from_env(), dry_run=not args.send)
    notifier.send(message)

    print()
    print("=" * 72)
    print("這份計畫的效力（誠實聲明）")
    print("=" * 72)
    print("  ✗ 使用規則式動能基準，**不是訓練過的模型**")
    print("  ✓ 分數已用訓練期 triple-barrier 標籤校準為真實機率")
    print("  ✗ 未跑 Walk-Forward，沒有樣本外驗證")
    print("  ✗ 未並列對照組（買進持有 / 0050 / 隨機進場）")
    print("  ✗ 未做 PBO / Deflated Sharpe 校正")
    print("  ✓ 證明整條管線接得起來，各模組在真實資料上可運作")
    print()
    print("  → 這是**管線驗證**，不是投資建議。")
    print("=" * 72)


if __name__ == "__main__":
    main()
