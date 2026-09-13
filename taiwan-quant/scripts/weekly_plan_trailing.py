#!/usr/bin/env python3
"""
端到端週頻計畫產生器（移動停損版，路線 A）

    資料載入 → 移動停損寬度推導 → 標記 → 期望報酬校準
        → Top 3 選股 → 部位規模 → 分批進場階梯 → Telegram 推播

與 `weekly_plan.py` 的差別：

    寬度   derive_trail_width      取代 derive_width
    標記   label_trailing          取代 label_one
    校準   fit_return_calibrator   取代 fit_calibrator
    門檻   E[毛報酬] − cost > 0    取代 P(+1) ≥ 動態門檻
    輸出   移動停損 + 分批階梯      取代 目標價 + 失效價

`weekly_plan.py` **不動**（禁令 9：已發佈的策略檔案不得就地改寫），
它仍是 triple-barrier 的對照組。

## ⚠️ 樣本外結論尚未支持推播

見 `../docs/需求規劃/202609/04_路線A驗證結果.md`：

    跑贏隨機進場       8 / 9   ✓
    跑贏等權買進持有   3 / 9   ← 但在槽位數上會翻盤

所以本腳本預設 **dry-run**，且輸出一定帶上這個限制聲明。
要推播必須明確加 `--send`。

用法：
    .venv/bin/python scripts/weekly_plan_trailing.py                # 只印不發
    .venv/bin/python scripts/weekly_plan_trailing.py --send
    .venv/bin/python scripts/weekly_plan_trailing.py --horizon 60   # 持有約 3 個月
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
from taiwan_quant.data.dataset import build_dataset  # noqa: E402
from taiwan_quant.data.loader import load_chips, load_prices, load_universe  # noqa: E402
from taiwan_quant.features.technical import build_technical  # noqa: E402
from taiwan_quant.labeling.trail_width import derive_trail_width  # noqa: E402
from taiwan_quant.labeling.trailing_stop import label_trailing  # noqa: E402
from taiwan_quant.notify.telegram import (  # noqa: E402
    TelegramConfig,
    TelegramNotifier,
    format_weekly_plan,
)
from taiwan_quant.ranking.trailing_portfolio import (  # noqa: E402
    DEFAULT_EDGE_Z,
    TrailingCandidate,
    select_trailing_portfolio,
)
from taiwan_quant.strategies.families import STRATEGY_FAMILIES  # noqa: E402
from taiwan_quant.validation.binning import find_bin  # noqa: E402
from taiwan_quant.validation.calibration import (  # noqa: E402
    CalibrationError,
    ReturnCalibrator,
    fit_return_calibrator,
    return_reliability_report,
)

STRATEGY_VERSION = "trailing_stop_portfolio@v0.1.0"

PULLBACK_QUANTILE = 0.80
CALIBRATION_BINS = 6
CALIBRATION_MIN_SAMPLES = 50

WARMUP_DAYS = 250
DECISION_STRIDE = 5
CORRELATION_WINDOW = 60

INDUSTRY_MAP: dict[str, str] = {
    "2330": "半導體", "2454": "半導體", "2303": "半導體", "3034": "半導體",
    "2317": "電子代工", "2382": "電子代工", "4938": "電子代工",
    "2881": "金融", "2882": "金融", "2891": "金融", "2884": "金融",
    "1301": "塑化", "1303": "塑化", "6505": "塑化",
    "2412": "電信", "3045": "電信", "4904": "電信",
    "1216": "食品", "2912": "通路",
}

DISCLAIMER = (
    "樣本外驗證：跑贏隨機 8/9、跑贏等權買進持有 3/9，"
    "但該結果在槽位數上會翻盤，有效獨立樣本僅 5~16 個。"
)


@dataclass(frozen=True)
class TrailSignal:
    """單檔的移動停損訊號"""

    stock_id: str
    score: float
    expected_gross_return: float
    return_std: float | None
    n_samples: int
    trail_pct: float
    entry_price: float
    volatility_pct: float
    beta: float


def build_calibration_set(
    by_stock: dict[str, pd.DataFrame],
    family,
    horizon: int,
    train_end: pd.Timestamp,
) -> tuple[np.ndarray, np.ndarray]:
    """
    蒐集訓練期的 (分數, 實際毛報酬)。

    **只取 `train_end` 之前、且標籤已經確定的樣本**（禁令 1）。
    `label_trailing` 在未來 K 棒不足時回 None，所以「已確定」是它保證的。
    """
    scores: list[float] = []
    returns: list[float] = []

    for bars in by_stock.values():
        if not set(family.required_columns).issubset(bars.columns):
            continue
        for idx in range(WARMUP_DAYS, len(bars), DECISION_STRIDE):
            if bars.index[idx] > train_end:
                break
            width = derive_trail_width(
                bars, decision_idx=idx, horizon=horizon,
                pullback_quantile=PULLBACK_QUANTILE,
            )
            if width is None:
                continue
            outcome = label_trailing(
                bars, decision_idx=idx,
                trail_pct=width.trail_pct, max_horizon=horizon,
            )
            if outcome is None:
                continue
            score = family.score_fn(bars.iloc[: idx + 1])
            if not np.isfinite(score):
                continue
            scores.append(float(score))
            returns.append(outcome.gross_return)

    return np.asarray(scores), np.asarray(returns)


def build_signal(
    stock_id: str,
    bars: pd.DataFrame,
    family,
    horizon: int,
    calibrator: ReturnCalibrator,
    volatility_pct: float,
    beta: float,
) -> TrailSignal | None:
    """產生單檔的當期訊號；任一環節無法確定就回 None"""
    last_idx = len(bars) - 1
    if last_idx < WARMUP_DAYS:
        return None

    width = derive_trail_width(
        bars, decision_idx=last_idx, horizon=horizon,
        pullback_quantile=PULLBACK_QUANTILE,
    )
    if width is None:
        return None

    score = family.score_fn(bars)
    if not np.isfinite(score):
        return None

    expected = calibrator.predict(float(score))
    if expected is None:
        return None

    bin_index = find_bin(float(score), calibrator.bins)
    bucket = calibrator.bins[bin_index] if bin_index is not None else None
    if bucket is None:
        return None

    return TrailSignal(
        stock_id=stock_id,
        score=float(score),
        expected_gross_return=expected,
        return_std=bucket.return_std,
        n_samples=bucket.n_samples,
        trail_pct=width.trail_pct,
        entry_price=float(bars["close"].iloc[-1]),
        volatility_pct=volatility_pct,
        beta=beta,
    )


def compute_beta(
    stock_returns: pd.Series, market_returns: pd.Series, window: int
) -> float:
    """對等權市場的 beta；無法計算時回 1.0（中性，不當成 defensive）"""
    joined = pd.concat([stock_returns, market_returns], axis=1).dropna().tail(window)
    if len(joined) < window // 2:
        return 1.0
    variance = joined.iloc[:, 1].var()
    if not np.isfinite(variance) or variance == 0:
        return 1.0
    return float(joined.iloc[:, 0].cov(joined.iloc[:, 1]) / variance)


def compute_correlations(
    returns_by_stock: dict[str, pd.Series], window: int
) -> dict[tuple[str, str], float]:
    """兩兩相關係數；算不出的配對不放進字典（選股層會保守拒絕）"""
    frame = pd.DataFrame(returns_by_stock).tail(window)
    matrix = frame.corr()
    out: dict[tuple[str, str], float] = {}
    for a in matrix.index:
        for b in matrix.columns:
            value = matrix.at[a, b]
            if a != b and np.isfinite(value):
                out[(str(a), str(b))] = float(value)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="週頻計畫（移動停損）")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--horizon", type=int, default=60,
                        help="最長持有交易日數（60 ≈ 3 個月）")
    parser.add_argument("--capital", type=float, default=400_000.0)
    parser.add_argument("--family", default="動能突破")
    parser.add_argument("--edge-z", type=float, default=DEFAULT_EDGE_Z)
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--as-of", default="2026-09-11")
    parser.add_argument("--send", action="store_true", help="實際推播")
    args = parser.parse_args()

    family = next((f for f in STRATEGY_FAMILIES if f.name == args.family), None)
    if family is None:
        names = [f.name for f in STRATEGY_FAMILIES]
        print(f"未知策略族 {args.family}，可選：{names}")
        return

    as_of = date.fromisoformat(args.as_of)
    train_end = pd.Timestamp(as_of) - pd.Timedelta(days=args.horizon * 2)

    print("=" * 80)
    print("週頻交易計畫 — 移動停損（路線 A）")
    print("=" * 80)
    print(f"策略族 {family.name}｜最長持有 {args.horizon} 交易日"
          f"（約 {args.horizon / 21:.0f} 個月）")
    print(f"資金 {args.capital:,.0f} 元｜資料截至 {as_of}")
    print(f"版本 {STRATEGY_VERSION}")
    print()

    universe = load_universe(limit=150)
    stock_ids = universe.stock_ids[: args.limit]

    prices = load_prices(
        stock_ids, start=date.fromisoformat(args.start), end=as_of, adjusted=True
    )
    chips = load_chips(
        stock_ids, start=date.fromisoformat(args.start), end=as_of
    )
    dataset = build_dataset(stock_ids, prices, chips)
    print(dataset.describe())
    print()

    by_stock = dataset.by_stock
    tiers = {sid: Tier(universe.tier_of(sid)) for sid in by_stock}

    # ── 市場基準、波動度分位、beta ──
    returns_by_stock = {
        sid: bars["close"].pct_change() for sid, bars in by_stock.items()
    }
    market_returns = pd.DataFrame(returns_by_stock).mean(axis=1)

    atr_ratios: dict[str, float] = {}
    for sid, bars in by_stock.items():
        value = build_technical(bars)["true_range_ratio_14"].iloc[-1]
        if np.isfinite(value):
            atr_ratios[sid] = float(value)
    atr_percentiles = (
        pd.Series(atr_ratios).rank(pct=True).to_dict() if atr_ratios else {}
    )

    # ── 校準（只用訓練期） ──
    print(f"擬合期望報酬校準器（訓練期至 {train_end.date()}）...")
    scores, returns = build_calibration_set(by_stock, family, args.horizon, train_end)
    print(f"  校準樣本 {len(scores):,} 筆")

    try:
        calibrator = fit_return_calibrator(
            scores, returns,
            n_bins=CALIBRATION_BINS,
            min_samples_per_bin=CALIBRATION_MIN_SAMPLES,
        )
    except CalibrationError as exc:
        print(f"  ✗ 校準失敗：{exc}")
        print("  未校準的分數不是期望報酬，拒絕產出計畫。")
        return

    print()
    print(return_reliability_report(calibrator))
    print()

    if not calibrator.has_discrimination:
        print("⚠️  校準器顯示分數對報酬沒有鑑別力（最高箱與最低箱的差距"
              "小於一趟來回成本）。")
        print("    仍然產出計畫供檢視，但這個訊號不該用來下決定。")
        print()

    # ── 訊號 ──
    signals: list[TrailSignal] = []
    for sid, bars in by_stock.items():
        signal = build_signal(
            sid, bars, family, args.horizon, calibrator,
            volatility_pct=atr_percentiles.get(sid, 0.5),
            beta=compute_beta(
                returns_by_stock[sid], market_returns, CORRELATION_WINDOW
            ),
        )
        if signal is not None:
            signals.append(signal)

    print(f"產生 {len(signals)} 個訊號（共 {len(by_stock)} 檔）")
    print()

    if not signals:
        print("無任何訊號，結束。")
        return

    candidates = [
        TrailingCandidate(
            stock_id=s.stock_id,
            expected_gross_return=s.expected_gross_return,
            return_std=s.return_std,
            n_samples=s.n_samples,
            trail_pct=s.trail_pct,
            entry_price=s.entry_price,
            tier=tiers.get(s.stock_id, Tier.MID),
            industry=INDUSTRY_MAP.get(s.stock_id, "其他"),
            volatility_pct=s.volatility_pct,
            beta=s.beta,
            max_horizon=args.horizon,
        )
        for s in signals
    ]

    correlations = compute_correlations(
        {s.stock_id: returns_by_stock[s.stock_id] for s in signals},
        CORRELATION_WINDOW,
    )

    result = select_trailing_portfolio(
        candidates, args.capital, correlations, cost=DEFAULT, edge_z=args.edge_z
    )

    print("─" * 80)
    print("選股結果")
    print("─" * 80)
    print(result.describe())
    print()

    if result.rejected:
        print("剔除紀錄（前 10 筆）：")
        for rejection in result.rejected[:10]:
            print(f"  {rejection.stock_id}  {rejection.reason.value}"
                  f"  {rejection.detail}")
        print()

    # ── 推播 ──
    week_id = pd.Timestamp(as_of).strftime("%GW%V")
    message = format_weekly_plan(
        result,
        week_id=week_id,
        strategy_version=STRATEGY_VERSION,
        data_as_of=str(as_of),
        universe_note=f"標的池：市值排名代理（{len(by_stock)} 檔）",
        benchmark_lines=[DISCLAIMER],
    )

    print("─" * 80)
    print("推播內容")
    print("─" * 80)
    print(message)
    print()

    if not args.send:
        print("（dry-run；要實際推播請加 --send）")
        return

    try:
        config = TelegramConfig.from_env()
    except (KeyError, ValueError) as exc:
        print(f"✗ Telegram 設定不完整：{exc}")
        print("  需要環境變數 TELEGRAM_BOT_TOKEN 與 TELEGRAM_CHAT_ID。")
        return

    TelegramNotifier(config).send(message)
    print("✓ 已推播")


if __name__ == "__main__":
    main()
