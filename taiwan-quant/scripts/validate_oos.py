#!/usr/bin/env python3
"""
Walk-Forward 樣本外驗證

回答診斷網格留下的三個疑問：

    1. 訓練期 2023-2025 是台股大多頭，基礎勝率 19.47% 是不是那段行情的產物？
    2. 三族的鑑別力只有 1~5 個百分點，樣本外還剩多少？
    3. 跑不跑得贏「隨機進場」與「等權買進持有」？

## 方法

滾動擴張窗口（expanding window）：

    Fold 1   train 2023-01 ~ 2024-03   test 2024-04 ~ 2024-06
    Fold 2   train 2023-01 ~ 2024-06   test 2024-07 ~ 2024-09
    ...

每個 fold：
    1. 只用 train 段擬合校準器（反 look-ahead）
    2. 在 test 段逐週決策 → 選 Top 3 → 用 triple-barrier 記錄實際結果
    3. 同期跑「隨機進場」與「等權買進持有」對照

## 對照組（CLAUDE.md 必跑）

    隨機進場      每週隨機挑 3 檔，用同一組柵欄
    等權買進持有   期初買進全標的池等權持有到期末

若策略跑不贏這兩者，就是沒有優勢——如實寫進報告（Kimi 的意見）。

## 樣本重疊的誠實處理

持有 60 日但每週決策 → 連續 12 筆樣本的持有期重疊。
**有效獨立樣本數遠小於總筆數**，報告會同時列出兩者。

用法：
    .venv/bin/python scripts/validate_oos.py
    .venv/bin/python scripts/validate_oos.py --horizons 20 60 --limit 40
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

from taiwan_quant.backtest.engine import TradePlan, describe_sensitivity, run_cost_sensitivity  # noqa: E402
from taiwan_quant.config.costs import DEFAULT, Tier  # noqa: E402
from taiwan_quant.data.loader import load_chips, load_prices, load_universe  # noqa: E402
from taiwan_quant.labeling.barrier_width import derive_width  # noqa: E402
from taiwan_quant.labeling.triple_barrier import label_one  # noqa: E402
from taiwan_quant.strategies.families import STRATEGY_FAMILIES, StrategyFamily  # noqa: E402
from taiwan_quant.validation.calibration import CalibrationError, fit_calibrator  # noqa: E402
from taiwan_quant.validation.stats import deflated_sharpe_ratio  # noqa: E402

ATR_MULTIPLE = 0.8
TARGET_QUANTILE = 0.85
MIN_RISK_REWARD = 2.0

CALIBRATION_BINS = 6
CALIBRATION_MIN_SAMPLES = 50

WARMUP_DAYS = 250
DECISION_STRIDE = 5
"""週頻決策：每 5 個交易日一次"""

TOP_N = 3
TEST_WINDOW_DAYS = 60
"""每個 fold 的測試期長度（交易日），約 3 個月"""

FIRST_TRAIN_DAYS = 500
"""第一個 fold 的訓練期長度（交易日），約 2 年"""

RANDOM_SEED = 20260912

TRADING_DAYS_PER_YEAR = 252.0
"""台股一年約 252 個交易日，用來把持有期換算成「一年幾期」"""


MIN_SERIES_LENGTH = 400
"""
標的最短序列長度。

短於此值的標的（近期上市、或籌碼資料只有末段）無法提供足夠的
暖機 + 訓練樣本，納入只會汙染全域時間軸。
"""


@dataclass(frozen=True)
class Decision:
    """一筆決策及其實際結果"""

    decision_date: pd.Timestamp
    """
    決策日。

    **必須用日期而非位置索引當全域時間軸**——各檔序列起始日不同，
    同一個位置索引在不同標的可能是不同日期，用它切 fold 會讓
    訓練期與測試期在標的之間錯位。
    """

    stock_id: str
    score: float
    prob: float
    target_pct: float
    stop_pct: float
    label: int
    gross_return: float
    holding_days: int


def enumerate_decisions(
    data_by_stock: dict[str, pd.DataFrame],
    family: StrategyFamily,
    horizon: int,
) -> dict[pd.Timestamp, list[Decision]]:
    """
    枚舉所有決策日的所有標的結果，**依決策日期分組**。

    一次算完供所有 fold 重用——各 fold 的差別只在「用哪些決策擬合校準器」
    與「評估哪些決策」，底層的分數與標籤是同一份。
    """
    by_date: dict[pd.Timestamp, list[Decision]] = {}

    for stock_id, bars in data_by_stock.items():
        if not set(family.required_columns).issubset(bars.columns):
            continue

        for idx in range(WARMUP_DAYS, len(bars), DECISION_STRIDE):
            width = derive_width(
                bars, decision_idx=idx, horizon=horizon,
                atr_multiple=ATR_MULTIPLE, target_quantile=TARGET_QUANTILE,
                min_risk_reward=MIN_RISK_REWARD,
            )
            if width is None:
                continue

            outcome = label_one(
                bars, decision_idx=idx,
                target_pct=width.target_pct, stop_pct=width.stop_pct,
                horizon=horizon,
            )
            if outcome is None:
                continue

            score = family.score_fn(bars.iloc[: idx + 1])
            if not np.isfinite(score):
                continue

            decision_date = bars.index[idx]
            by_date.setdefault(decision_date, []).append(
                Decision(
                    decision_date=decision_date,
                    stock_id=stock_id,
                    score=score,
                    prob=float("nan"),
                    target_pct=width.target_pct,
                    stop_pct=width.stop_pct,
                    label=outcome.label,
                    gross_return=outcome.gross_return,
                    holding_days=outcome.holding_days,
                )
            )

    return by_date


def to_plans(
    picks: list[tuple[pd.Timestamp, Decision]],
    tiers: dict[str, Tier],
    horizon: int,
) -> list[TradePlan]:
    """把選中的決策轉成回測用的交易計畫"""
    return [
        TradePlan(
            week_id=decision_date.strftime("%Y-%m-%d"),
            stock_id=d.stock_id,
            gross_return=d.gross_return,
            holding_days=d.holding_days,
            tier=tiers.get(d.stock_id, Tier.MID),
        )
        for decision_date, d in picks
    ]


def run_walk_forward(
    by_date: dict[pd.Timestamp, list[Decision]],
    tiers: dict[str, Tier],
    horizon: int,
    rng: np.random.Generator,
) -> tuple[
    list[tuple[pd.Timestamp, Decision]],
    list[tuple[pd.Timestamp, Decision]],
    int,
    int,
]:
    """
    執行滾動 Walk-Forward。

    時間軸以**決策日期**切分，不是位置索引——各檔序列起始日不同，
    用位置索引會讓訓練/測試期在標的之間錯位。

    Returns:
        (策略選中的決策, 隨機選中的決策, fold 數, 校準失敗的 fold 數)
    """
    dates = sorted(by_date)
    if not dates:
        return [], [], 0, 0

    # 決策日每 DECISION_STRIDE 個交易日一次，換算成「幾個決策日」
    first_train = (WARMUP_DAYS + FIRST_TRAIN_DAYS) // DECISION_STRIDE
    test_span = max(1, TEST_WINDOW_DAYS // DECISION_STRIDE)

    strategy_picks: list[tuple[pd.Timestamp, Decision]] = []
    random_picks: list[tuple[pd.Timestamp, Decision]] = []
    folds = 0
    calibration_failures = 0

    cursor = first_train
    last_exit: pd.Timestamp | None = None
    all_dates = dates

    while cursor + test_span <= len(dates):
        train_dates = dates[:cursor]
        test_dates = dates[cursor : cursor + test_span]
        folds += 1

        train_decisions = [d for dt in train_dates for d in by_date[dt]]
        try:
            calibrator = fit_calibrator(
                np.array([d.score for d in train_decisions]),
                np.array([d.label for d in train_decisions]),
                n_bins=CALIBRATION_BINS,
                min_samples_per_bin=CALIBRATION_MIN_SAMPLES,
            )
        except CalibrationError:
            calibration_failures += 1
            cursor += test_span
            continue

        for decision_date in test_dates:
            # 只在前一筆持倉出場後才開新倉。
            #
            # 回測引擎把各期報酬**依序複合**，餵重疊持倉等於假設每筆
            # 都是接續的——實測會把 105 個重疊的 60 日持倉複合成
            # +10,618% 的天文數字。持倉重疊必須在這裡擋掉。
            if last_exit is not None and decision_date < last_exit:
                continue

            candidates = by_date[decision_date]

            # ── 策略：依校準機率超過門檻的程度排序 ──
            scored: list[tuple[float, Decision]] = []
            for d in candidates:
                prob = calibrator.predict(d.score)
                if prob is None:
                    continue
                cost = DEFAULT.round_trip_rate(tiers.get(d.stock_id, Tier.MID))
                threshold = (d.stop_pct + cost) / (d.target_pct + d.stop_pct)
                if prob <= threshold:
                    continue
                scored.append((prob - threshold, d))

            scored.sort(key=lambda pair: (-pair[0], pair[1].stock_id))
            picked = scored[:TOP_N]

            # 隨機對照必須與策略在**同樣的決策日**進場，否則兩者的
            # 市場環境不同，比較沒有意義
            if not picked or not candidates:
                continue

            strategy_picks.extend((decision_date, d) for _, d in picked)

            chosen = rng.choice(
                len(candidates), size=min(TOP_N, len(candidates)), replace=False
            )
            random_picks.extend((decision_date, candidates[int(c)]) for c in chosen)

            # 下一筆最早的進場日 = 本批最長持倉的出場日
            max_hold = max(d.holding_days for _, d in picked)
            exit_pos = all_dates.index(decision_date) + max_hold
            last_exit = all_dates[min(exit_pos, len(all_dates) - 1)]

        cursor += test_span

    return strategy_picks, random_picks, folds, calibration_failures


def buy_and_hold_return(
    data_by_stock: dict[str, pd.DataFrame],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> float:
    """
    等權買進持有：start_date 買進、end_date 賣出。

    **期間必須與策略完全一致**，否則不可比。第一版的 bug 就是
    買進持有只涵蓋 7 個月、策略涵蓋 2 年多，比較毫無意義。

    以日期對齊而非位置索引——各檔起始日不同。
    """
    returns = []
    for bars in data_by_stock.values():
        window = bars.loc[(bars.index >= start_date) & (bars.index <= end_date)]
        if len(window) < 2:
            continue
        entry = float(window["close"].iloc[0])
        exit_price = float(window["close"].iloc[-1])
        if entry > 0:
            returns.append(exit_price / entry - 1)
    return float(np.mean(returns)) if returns else float("nan")


def effective_samples(picks: list[tuple[pd.Timestamp, Decision]], horizon: int) -> int:
    """
    有效獨立樣本數。

    持有 horizon 天但每 DECISION_STRIDE 天決策一次 → 持有期重疊。
    以「決策日間隔 >= horizon 個交易日」估算不重疊的樣本數。

    報告必須同時列出總筆數與有效樣本數——用總筆數算統計顯著性
    會嚴重高估。
    """
    if not picks:
        return 0
    unique_dates = sorted({d for d, _ in picks})
    stride_between = max(1, horizon // DECISION_STRIDE)
    return len(unique_dates[::stride_between])


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-Forward 樣本外驗證")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--horizons", type=int, nargs="*", default=[20, 40, 60])
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2026-09-11")
    args = parser.parse_args()

    universe = load_universe(limit=150)
    stock_ids = universe.stock_ids[: args.limit]

    print("=" * 104)
    print("Walk-Forward 樣本外驗證")
    print("=" * 104)
    print(f"標的 {len(stock_ids)} 檔｜期間 {args.start} ~ {args.end}")
    print(f"暖機 {WARMUP_DAYS} 日｜首次訓練 {FIRST_TRAIN_DAYS} 日｜"
          f"每個 fold 測試 {TEST_WINDOW_DAYS} 日")
    print()

    prices = load_prices(
        stock_ids, start=date.fromisoformat(args.start),
        end=date.fromisoformat(args.end), adjusted=True,
    )
    chips = load_chips(
        stock_ids, start=date.fromisoformat(args.start),
        end=date.fromisoformat(args.end),
    )

    data_by_stock: dict[str, pd.DataFrame] = {}
    tiers: dict[str, Tier] = {}
    skipped_short: list[tuple[str, int]] = []
    for sid in stock_ids:
        if sid not in prices.index.get_level_values("stock_id"):
            continue
        bars = prices.xs(sid, level="stock_id")
        if sid in chips.index.get_level_values("stock_id"):
            bars = bars.join(
                chips.xs(sid, level="stock_id")[
                    ["foreign_net", "trust_net", "dealer_net",
                     "margin_balance", "short_balance"]
                ],
                how="inner",
            )
        if len(bars) < MIN_SERIES_LENGTH:
            skipped_short.append((sid, len(bars)))
            continue
        data_by_stock[sid] = bars
        tiers[sid] = Tier(universe.tier_of(sid))

    lengths = [len(b) for b in data_by_stock.values()]
    print(f"可用 {len(data_by_stock)} 檔｜序列長度 {min(lengths)} ~ {max(lengths)} 根")
    if skipped_short:
        print(f"剔除過短標的 {len(skipped_short)} 檔（< {MIN_SERIES_LENGTH} 根）："
              f"{[f'{s}({n})' for s, n in skipped_short]}")
    print()

    reference = max(data_by_stock.values(), key=len)

    rows: list[dict[str, object]] = []
    n_trials = len(STRATEGY_FAMILIES) * len(args.horizons)

    for horizon in args.horizons:
        for family in STRATEGY_FAMILIES:
            print(f"  {family.name} × {horizon} 日 ...", flush=True)
            by_date = enumerate_decisions(data_by_stock, family, horizon)
            rng = np.random.default_rng(RANDOM_SEED)

            strategy_picks, random_picks, folds, failures = run_walk_forward(
                by_date, tiers, horizon, rng
            )

            if not strategy_picks:
                rows.append({
                    "family": family.name, "horizon": horizon,
                    "folds": folds, "trades": 0, "note": "OOS 無任何標的通過門檻",
                })
                continue

            # 買進持有必須與本組策略的實際交易期間對齊
            trade_dates = sorted({d for d, _ in strategy_picks})
            bh_start_date = trade_dates[0]
            bh_end_pos = min(
                list(reference.index).index(trade_dates[-1])
                + max(d.holding_days for _, d in strategy_picks),
                len(reference) - 1,
            )
            bh_end_date = reference.index[bh_end_pos]
            bh_return = buy_and_hold_return(data_by_stock, bh_start_date, bh_end_date)

            # 每期 = horizon 個交易日，不是一週。
            # 不傳這個值，年化與 Sharpe 會被當成週頻算——實測會把
            # 8 期的 +198% 年化成 +122,560%。
            periods_per_year = TRADING_DAYS_PER_YEAR / horizon

            strategy = run_cost_sensitivity(
                to_plans(strategy_picks, tiers, horizon),
                periods_per_year=periods_per_year,
            )
            random_result = run_cost_sensitivity(
                to_plans(random_picks, tiers, horizon),
                periods_per_year=periods_per_year,
            )

            net = strategy["6折+滑價（預設）"]
            rnd = random_result["6折+滑價（預設）"]

            rows.append({
                "family": family.name,
                "horizon": horizon,
                "folds": folds,
                "calibration_failures": failures,
                "trades": net.trades,
                "periods": net.weeks,
                "effective": len(trade_dates),
                "win_rate": net.win_rate,
                "gross": net.gross_cumulative_return,
                "net": net.net_cumulative_return,
                "sharpe": net.sharpe,
                "maxdd": net.max_drawdown,
                "annualized": net.annualized_return,
                "random_net": rnd.net_cumulative_return,
                "random_win": rnd.win_rate,
                "bh_return": bh_return,
                "bh_start": bh_start_date,
                "bh_end": bh_end_date,
                "sensitivity": strategy,
                "note": "",
            })

    print()
    print("─" * 104)
    print("樣本外結果（含 6 折 + 滑價成本）")
    print("─" * 104)
    print(f"{'策略族':<10}{'持有':>5}{'期數':>5}{'勝率':>8}"
          f"{'毛報酬':>10}{'淨報酬':>10}{'年化':>10}{'Sharpe':>8}"
          f"{'隨機淨':>10}{'買進持有':>10}{'MaxDD':>8}")
    print("─" * 104)

    for r in rows:
        if r.get("note"):
            print(f"{r['family']:<10}{r['horizon']:>5}{'':>5}{'':>8}"
                  f"{'':>10}{'':>10}{'':>10}{'':>8}{'':>10}{'':>10}{'':>8}"
                  f"  {r['note']}")
            continue
        sharpe = f"{r['sharpe']:.2f}" if r["sharpe"] is not None else "n/a"
        print(
            f"{r['family']:<10}{r['horizon']:>5}{r['periods']:>5}"
            f"{r['win_rate'] * 100:>7.1f}%"
            f"{r['gross'] * 100:>9.2f}%{r['net'] * 100:>9.2f}%"
            f"{r['annualized'] * 100:>9.2f}%{sharpe:>8}"
            f"{r['random_net'] * 100:>9.2f}%{r['bh_return'] * 100:>9.2f}%"
            f"{r['maxdd'] * 100:>7.2f}%"
        )

    print("─" * 104)
    print()
    for r in rows:
        if not r.get("note"):
            print(f"  買進持有期間（{r['family']} × {r['horizon']} 日）："
                  f"{r['bh_start'].date()} ~ {r['bh_end'].date()}")
            break
    print()

    # ── 結論 ──
    print("=" * 104)
    print("結論")
    print("=" * 104)

    scored = [r for r in rows if not r.get("note")]
    beats_random = [r for r in scored if r["net"] > r["random_net"]]
    beats_bh = [r for r in scored if r["net"] > r["bh_return"]]

    print(f"  跑贏隨機進場       {len(beats_random)} / {len(scored)}")
    print(f"  跑贏等權買進持有   {len(beats_bh)} / {len(scored)}")
    print()

    if not beats_random:
        print("  **沒有任何組合跑贏隨機進場。**")
        print("  策略沒有選股能力——診斷網格顯示的『優勢』全部來自柵欄結構，")
        print("  樣本外沒有留下任何鑑別力。")
    elif not beats_bh:
        print("  有組合跑贏隨機，但**沒有任何組合跑贏等權買進持有**。")
        print("  依 Kimi 的判準：短打策略沒有優勢，不如買進持有。")
    else:
        print(f"  {len(beats_bh)} 組同時跑贏隨機與買進持有：")
        for r in sorted(beats_bh, key=lambda x: -x["net"]):
            print(f"    {r['family']} × {r['horizon']} 日｜"
                  f"淨 {r['net'] * 100:+.2f}% vs 買進持有 {r['bh_return'] * 100:+.2f}%"
                  f"｜獨立交易期 {r['periods']}")
            if r["sharpe"] is not None:
                dsr = deflated_sharpe_ratio(
                    observed_sharpe=r["sharpe"],
                    n_trials=n_trials,
                    n_observations=max(r["effective"], 2),
                    sharpe_std=1.0,
                )
                verdict = "顯著" if dsr.is_significant else "**無法排除運氣**"
                print(f"      Sharpe {r['sharpe']:.3f}｜"
                      f"DSR {dsr.deflated_sharpe:.4f}（N={n_trials}）→ {verdict}")

    print()
    print("  樣本量警告：")
    for r in scored:
        if r["periods"] < 20:
            print(f"    {r['family']} × {r['horizon']} 日｜"
                  f"僅 {r['periods']} 個獨立交易期——樣本過少，"
                  "統計結論極不穩定")

    print()
    print("  成本敏感度（最佳組合）：")
    if scored:
        best = max(scored, key=lambda r: r["net"])
        print(f"    {best['family']} × {best['horizon']} 日")
        print()
        print(describe_sensitivity(best["sensitivity"]))

    print()
    print("=" * 104)
    print("⚠️  本驗證為樣本外滾動測試，但 OOS 期間仍落在 2024-2026 的台股多頭。")
    print("    不構成投資建議。")
    print("=" * 104)


if __name__ == "__main__":
    main()
