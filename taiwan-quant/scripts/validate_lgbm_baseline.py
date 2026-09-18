#!/usr/bin/env python3
"""
LightGBM baseline vs 手工分數（開發集，walk-forward）

## 這在回答什麼

到目前為止所有工作都在改**機制**與**成本**，訊號一直是手工分數。
累積的否定清單顯示那兩條已經挖完：

```
週頻抓漲停   否定    三族合成   否定    新聞   否定
ETF 輪動     否定    本金       不是瓶頸
```

**剩下唯一沒動過的是訊號本身。** CLAUDE.md 也明文要求
「先有 LightGBM baseline」才談其他模型。

## 比較必須是同一條件

```
決策日     完全相同（開發集、暖機 750、每 40 日一次）
標的池     完全相同（時點市值前 150）
標籤       完全相同（T+1 開盤買、T+41 收盤賣）
持倉數     完全相同（N=10）
成本       完全相同（逐檔 resolve_tier，禁令 4 的 large 分層）
唯一差別   排序用 LightGBM 預測，還是手工分數
```

**只換排序來源，其他一律不動。** 否則差異來自哪裡分不出來。

## 未調參

`DEFAULT_PARAMS` 固定一組，`N_TRIALS = 1`。掃參數會讓多重測試懲罰
爆掉——先前掃 126 組時 DSR 0.0042、掃 16 組時 PBO 0.625~0.750。
**baseline 的價值就在 n_trials = 1。**

## 每一期都重新訓練，而且淨化標籤

```
在位置 p 預測   訓練列必須滿足 d + 1 + horizon <= p
```

40 日標籤在 41 個交易日後才揭曉，少減 1 就洩漏一期，而且不會拋錯。
見 `models/lgbm_baseline.purged_training_positions`。

## 禁令 6

一律跑開發集（預設 `--end 2023-12-29`）。
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.validate_oos_trailing as V  # noqa: E402, N812
from taiwan_quant.config.costs import (  # noqa: E402
    DEFAULT as DEFAULT_COST,
)
from taiwan_quant.config.costs import (
    resolve_tier,
)
from taiwan_quant.data.dataset import build_dataset  # noqa: E402
from taiwan_quant.data.etf_universe import is_etf, merge_etf_candidates  # noqa: E402
from taiwan_quant.data.integrity import (  # noqa: E402
    complete_holding_decision_dates,
    forward_returns,
    select_holding_positions,
)
from taiwan_quant.data.loader import (  # noqa: E402
    HISTORY_DB_PATH,
    RAW_OPEN_COLUMN,
    load_chips,
    load_prices,
)
from taiwan_quant.models.lgbm_baseline import (  # noqa: E402
    DEFAULT_PARAMS,
    N_TRIALS,
    NUM_BOOST_ROUND,
    build_dataset_for_model,
    purged_training_positions,
    train_and_predict,
)
from taiwan_quant.ranking.tie_break import (  # noqa: E402
    DEFAULT_TIE_SEED,
    deterministic_jitter,
)
from taiwan_quant.validation.delisting import load_delisted_dates  # noqa: E402
from taiwan_quant.validation.stats import deflated_sharpe_ratio  # noqa: E402

FAMILY = "動能突破"
HOLDING_DAYS = 40
DECISION_STRIDE = 40
N_POSITIONS = 10
CAPITAL = 400_000.0
UNIVERSE_SIZE = 150
LARGE_TIER_SIZE = 50
UNIVERSE_BASIS = "market_cap"
MIN_CANDIDATES = 30
WARMUP = 750
DEV_END = date(2023, 12, 29)

RANDOM_DRAWS = 200
"""隨機選 N 檔的模擬次數（CLAUDE.md 必跑對照組）"""
RANDOM_BENCHMARK_SEED = 20260917


def random_gross_median(
    realized: pd.Series,
    tradeable: list[str],
    *,
    n_positions: int,
    n_draws: int,
    seed: int,
) -> float:
    """固定排序母體與 seed，回傳可跨 process 重現的隨機選股中位數。"""
    population = np.asarray(sorted(tradeable), dtype=object)
    rng = np.random.default_rng(seed)
    draws = [
        float(realized[list(rng.choice(population, n_positions, replace=False))].mean())
        for _ in range(n_draws)
    ]
    return float(np.median(draws))


def period_cost(
    picks: list[str],
    entry_day: pd.Timestamp,
    raw_opens: pd.DataFrame,
    opens: pd.DataFrame,
    large_members: set[str],
) -> float:
    """逐檔成本（禁令 3、4）。可負擔性用實際價，分層看市值前 50"""
    rates = []
    for stock_id in picks:
        actual = float(raw_opens.loc[entry_day, stock_id])
        adjusted = float(opens.loc[entry_day, stock_id])
        tier = resolve_tier(
            actual_price=actual, adjusted_price=adjusted,
            amount=CAPITAL / N_POSITIONS,
            large=stock_id in large_members, is_etf=is_etf(stock_id),
        )
        rates.append(DEFAULT_COST.round_trip_rate(tier))
    return float(np.mean(rates))


def summarise(periods: list[dict], label: str) -> dict:
    """每期毛／成本／淨 → 彙總"""
    if not periods:
        return {"label": label, "periods": 0}
    frame = pd.DataFrame(periods)
    frame["year"] = pd.to_datetime(frame["decision_date"]).dt.year
    nets = frame["net"].to_numpy()
    trips = 252 / HOLDING_DAYS
    yearly = frame.groupby("year")["net"].mean()
    curve = np.cumprod(1 + nets)
    return {
        "label": label,
        "periods": len(frame),
        "gross_per_trip": float(frame["gross"].mean()),
        "cost_per_trip": float(frame["cost"].mean()),
        "net_per_trip": float(nets.mean()),
        "total_return": float(curve[-1] - 1),
        "annualised": float(curve[-1] ** (trips / len(nets)) - 1),
        "sharpe": float(nets.mean() / nets.std(ddof=1) * np.sqrt(trips))
        if len(nets) > 1 and nets.std(ddof=1) > 0 else 0.0,
        "positive_years": f"{int((yearly > 0).sum())}/{len(yearly)}",
        "series": dict(zip(frame["decision_date"], frame["net"], strict=True)),
    }


def paired(a: dict, b: dict) -> dict:
    """b − a 的配對差異。只有一個觀察非零時不報 t（那是代數恆等式）"""
    shared = sorted(set(a.get("series", {})) & set(b.get("series", {})))
    if len(shared) < 3:
        return {"n": len(shared), "note": "共同期數不足"}
    diff = np.array([b["series"][d] - a["series"][d] for d in shared])
    nonzero = int((np.abs(diff) > 1e-12).sum())
    result = {"n": len(diff), "mean_difference": float(diff.mean()),
              "nonzero_periods": nonzero}
    if nonzero <= 1 or diff.std(ddof=1) == 0:
        result["note"] = (
            f"只有 {nonzero} 期非零——mean == SE 是代數恆等式，t 值無意義"
        )
        return result
    stderr = float(diff.std(ddof=1) / np.sqrt(len(diff)))
    result["standard_error"] = stderr
    result["t"] = float(diff.mean() / stderr)
    result["verdict"] = (
        "量不出差別" if abs(result["t"]) < 2
        else ("模型顯著較好" if result["t"] > 0 else "模型顯著較差")
    )
    return result


def run(db_path: Path, end: date) -> dict:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    members = [
        row[0] for row in con.execute(
            "SELECT DISTINCT stock_id FROM universe_history WHERE basis = ?",
            (UNIVERSE_BASIS,),
        )
    ]
    con.close()
    members = list(merge_etf_candidates(tuple(members), include=True))

    prices = load_prices(members, start=date(2015, 1, 1), end=end,
                         adjusted=True, db_path=db_path)
    chips = load_chips(members, start=date(2015, 1, 1), end=end, db_path=db_path)
    by_stock = build_dataset(members, prices, chips).by_stock

    calendar = V.trading_calendar(by_stock)
    scores = V.precompute_scores(by_stock)[FAMILY]
    frame = lambda col: pd.DataFrame(  # noqa: E731
        {sid: bars[col].astype(float) for sid, bars in by_stock.items()}
    ).reindex(calendar)
    opens, closes, raw_opens = frame("open"), frame("close"), frame(RAW_OPEN_COLUMN)
    delisted_dates = load_delisted_dates(db_path, as_of=end)
    # T+1 開盤進、T+HOLDING_DAYS 收盤出 = HOLDING_DAYS 天持有。
    # 第一版寫 shift(-1 - HOLDING_DAYS)，那是 H+1 天，與
    # diagnose_constraints / cost_floor / position_costs 以及
    # data/integrity.holding_dates 都不一致。
    forward = forward_returns(opens, closes, holding_days=HOLDING_DAYS)

    model_data = build_dataset_for_model(by_stock, calendar, HOLDING_DAYS)

    decision_dates = complete_holding_decision_dates(
        calendar,
        calendar[WARMUP::DECISION_STRIDE],
        holding_days=HOLDING_DAYS,
    )
    members_at = V.resolve_members(
        decision_dates, db_path, UNIVERSE_SIZE, UNIVERSE_BASIS)
    large_at = V.resolve_members(
        decision_dates, db_path, LARGE_TIER_SIZE, UNIVERSE_BASIS)

    model_periods, hand_periods, random_periods, equal_periods = [], [], [], []
    shuffled_periods: list[dict] = []
    overlaps: list[int] = []
    gain_totals: dict[str, float] = {}

    for day in decision_dates:
        position = calendar.index(day)
        allowed = tuple(members_at.get(day) or ())
        if not allowed:
            continue
        hand = pd.Series(
            {sid: scores[sid].get(day, np.nan) for sid in allowed if sid in scores}
        ).dropna()
        realized = forward.loc[day]
        common = hand.index
        if len(common) < MIN_CANDIDATES:
            continue

        entry_day = calendar[position + 1]
        large_members = set(large_at.get(day, set()))
        tradeable = [
            sid for sid in common
            if np.isfinite(raw_opens.loc[entry_day, sid])
            and raw_opens.loc[entry_day, sid] > 0
        ]
        if len(tradeable) < MIN_CANDIDATES:
            continue

        def record(
            bucket: list,
            picks: list[str],
            realized_period: pd.Series = realized,
            trade_day: pd.Timestamp = entry_day,
            historical_large: set[str] = large_members,
            decision_day: pd.Timestamp = day,
        ) -> None:
            settlements = select_holding_positions(
                decision_date=decision_day,
                calendar=calendar,
                ordered_candidates=picks,
                opens=opens,
                closes=closes,
                holding_days=HOLDING_DAYS,
                n_positions=len(picks),
                delisted_dates=delisted_dates,
            )
            picks = [position.stock_id for position in settlements]
            gross = float(
                np.mean([position.gross_return for position in settlements])
            )
            cost = period_cost(
                picks, trade_day, raw_opens, opens, historical_large
            )
            bucket.append({"decision_date": str(decision_day.date()), "gross": gross,
                           "cost": cost, "net": gross - cost})

        ordered_hand = sorted(
            tradeable,
            key=lambda sid: (-float(hand[sid]),
                             deterministic_jitter(sid, DEFAULT_TIE_SEED)),
        )
        hand_picks = ordered_hand[:N_POSITIONS]
        record(hand_periods, hand_picks)

        train_range = purged_training_positions(
            position, HOLDING_DAYS, warmup=250)
        predicted, gains = train_and_predict(
            model_data, calendar, train_range, position, tuple(tradeable))
        for name, value in gains.items():
            gain_totals[name] = gain_totals.get(name, 0.0) + value
        if not predicted.empty:
            ordered_model = sorted(
                predicted.index,
                key=lambda sid: (-float(predicted[sid]),
                                 deterministic_jitter(sid, DEFAULT_TIE_SEED)),
            )
            model_picks = ordered_model[:N_POSITIONS]
            record(model_periods, model_picks)
            overlaps.append(len(set(model_picks) & set(hand_picks)))

        # 洩漏偵測：在每個日期之內打亂標籤後重訓。資料形狀不變，
        # 特徵與標籤的關係被破壞——若還能贏過隨機，優勢不是來自預測。
        shuffled_pred, _ = train_and_predict(
            model_data, calendar, train_range, position, tuple(tradeable),
            shuffle_seed=DEFAULT_TIE_SEED + position)
        if not shuffled_pred.empty:
            ordered_shuffled = sorted(
                shuffled_pred.index,
                key=lambda sid: (-float(shuffled_pred[sid]),
                                 deterministic_jitter(sid, DEFAULT_TIE_SEED)),
            )
            record(shuffled_periods, ordered_shuffled[:N_POSITIONS])

        complete_tradeable = [
            sid for sid in tradeable if np.isfinite(realized.loc[sid])
        ]
        random_gross = random_gross_median(
            realized,
            complete_tradeable,
            n_positions=N_POSITIONS,
            n_draws=RANDOM_DRAWS,
            seed=RANDOM_BENCHMARK_SEED + position,
        )
        median_cost = period_cost(hand_picks, entry_day, raw_opens, opens,
                                  large_members)
        random_periods.append({"decision_date": str(day.date()),
                               "gross": random_gross,
                               "cost": median_cost,
                               "net": random_gross - median_cost})
        equal_periods.append({"decision_date": str(day.date()),
                              "gross": float(realized[complete_tradeable].mean()),
                              "cost": 0.0,
                              "net": float(realized[complete_tradeable].mean())})
        print(f"  {day.date()} 完成", flush=True)

    model = summarise(model_periods, "LightGBM baseline")
    hand = summarise(hand_periods, f"手工分數（{FAMILY}）")
    shuffled_summary = summarise(shuffled_periods, "標籤打亂（洩漏偵測）")
    random_summary = summarise(random_periods, "隨機 10 檔（200 次中位）")
    effective = len(calendar) // HOLDING_DAYS
    dsr = {}
    if model.get("sharpe", 0) > 0 and effective >= 30:
        outcome = deflated_sharpe_ratio(
            observed_sharpe=model["sharpe"], n_trials=N_TRIALS,
            n_observations=effective, sharpe_std=1.0,
        )
        dsr = {"deflated_sharpe": float(outcome.deflated_sharpe),
               "expected_max_sharpe": float(outcome.expected_max_sharpe),
               "n_trials": N_TRIALS, "n_observations": effective,
               "note": "n_trials=1 時期望最佳 Sharpe 退化為 0（無選擇偏誤）"}

    return {
        "end": str(end), "decision_dates": len(decision_dates),
        "params": {
            **DEFAULT_PARAMS,
            "num_boost_round": NUM_BOOST_ROUND,
            "random_benchmark_seed": RANDOM_BENCHMARK_SEED,
            "random_benchmark_draws": RANDOM_DRAWS,
            "random_benchmark_population_order": "stock_id ascending",
        },
        "n_trials": N_TRIALS,
        "model": model, "hand": hand,
        "shuffled_labels": shuffled_summary,
        "random_median": random_summary,
        "equal_weight": summarise(equal_periods, "等權全池（無成本）"),
        "paired": {
            # 「真模型 − 打亂」才是正確的虛無假設。見報告的說明：
            # 用任意函數取前 10 檔 ≠ 均勻隨機抽 10 檔，因為任意函數
            # 會繼承一個因子傾斜。
            "真模型 − 隨機": paired(random_summary, model),
            "打亂 − 隨機": paired(random_summary, shuffled_summary),
            "真模型 − 打亂": paired(shuffled_summary, model),
            "真模型 − 手工": paired(hand, model),
            "手工 − 打亂": paired(shuffled_summary, hand),
        },
        "top10_overlap": {
            "median": float(np.median(overlaps)) if overlaps else None,
            "mean": float(np.mean(overlaps)) if overlaps else None,
            "zero_overlap_periods": int(sum(1 for x in overlaps if x == 0)),
            "periods": len(overlaps),
        },
        "dsr": dsr,
        "feature_importance": dict(sorted(
            gain_totals.items(), key=lambda kv: -kv[1])),
    }


def report(payload: dict) -> None:
    print(f"\n開發集 {payload['end']} 為止｜決策日 {payload['decision_dates']} 期"
          f"｜試驗數 {payload['n_trials']}（未調參）\n")
    print(f"{'組合':<26}{'期數':>6}{'毛/趟':>9}{'成本/趟':>9}{'淨/趟':>9}"
          f"{'年化':>9}{'Sharpe':>8}{'逐年正':>8}")
    for key in ("model", "hand", "shuffled_labels", "random_median",
                "equal_weight"):
        block = payload[key]
        if block.get("periods", 0) == 0:
            print(f"{block['label']:<26}  無有效期數")
            continue
        print(f"{block['label']:<26}{block['periods']:>6}"
              f"{block['gross_per_trip']:>8.2%}{block['cost_per_trip']:>9.3%}"
              f"{block['net_per_trip']:>9.2%}{block['annualised']:>9.1%}"
              f"{block['sharpe']:>8.2f}{block['positive_years']:>8}")
    print()

    print("── 配對比較 " + "─" * 54)
    print(f"{'':<16}{'差異':>9}{'標準誤':>9}{'t':>7}{'判定':>14}")
    for label, pair in payload["paired"].items():
        if "t" not in pair:
            print(f"{label:<16}{pair.get('mean_difference', 0):>+9.2%}"
                  f"  {pair.get('note', '')}")
            continue
        print(f"{label:<16}{pair['mean_difference']:>+9.2%}"
              f"{pair['standard_error']:>9.2%}{pair['t']:>7.2f}"
              f"{pair['verdict']:>14}")
    print()
    print("   ⚠️ **「真模型 − 打亂」才是正確的虛無假設。**")
    print("      用任意函數取前 10 檔 ≠ 均勻隨機抽 10 檔——任意函數會")
    print("      繼承一個因子傾斜，所以「贏過隨機」這個門檻太低。")
    print()

    overlap = payload["top10_overlap"]
    if overlap["periods"]:
        print("── 模型與手工分數選出來的前 10 檔重疊 " + "─" * 27)
        print(f"   中位 {overlap['median']:.0f} 檔｜平均 {overlap['mean']:.1f} 檔"
              f"｜完全不重疊 {overlap['zero_overlap_periods']}/{overlap['periods']} 期")
        print("   重疊高 → 模型只是重新發現同一個訊號，不是新資訊")
        print()

    shuffled = payload["shuffled_labels"]
    if shuffled.get("periods", 0):
        print("── 洩漏偵測：標籤打亂後 " + "─" * 41)
        print(f"   淨/趟 {shuffled['net_per_trip']:+.2%}"
              f"｜隨機 {payload['random_median']['net_per_trip']:+.2%}"
              f"｜真模型 {payload['model']['net_per_trip']:+.2%}")
        gap = shuffled["net_per_trip"] - payload["random_median"]["net_per_trip"]
        print(f"   打亂 − 隨機 = {gap:+.2%}／趟")
        print("   打亂後應該掉到隨機水準。若仍明顯較高 → **「隨機」這個")
        print("   對照組對模型策略太弱**，不是管線洩漏（見配對比較）。")
        print()
        print("   毛報酬的解剖：")
        for key, label in (("equal_weight", "持有全池"),
                           ("shuffled_labels", "任意函數取前 10"),
                           ("hand", "手工分數取前 10"),
                           ("model", "模型取前 10")):
            print(f"     {label:<20}{payload[key]['gross_per_trip']:>8.2%}")
        print()

    importance = payload.get("feature_importance") or {}
    if importance:
        total = sum(importance.values()) or 1.0
        print("── 特徵重要度（gain 累計，前 10）" + "─" * 32)
        for name, value in list(importance.items())[:10]:
            print(f"   {name:<26}{value / total:>7.1%}")
        print()

    if payload["dsr"]:
        dsr = payload["dsr"]
        print("── DSR " + "─" * 60)
        print(f"   DSR {dsr['deflated_sharpe']:.4f}"
              f"｜運氣門檻 {dsr['expected_max_sharpe']:.4f}"
              f"｜有效觀測 {dsr['n_observations']}｜試驗 {dsr['n_trials']}")
        print(f"   {dsr['note']}")
        print("   ⚠️ n_trials=1 讓 DSR 幾乎必然通過。**那不是證據**——")
        print("      它只說明「沒有掃參數」，不說明訊號有用。")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=HISTORY_DB_PATH)
    parser.add_argument("--end", type=date.fromisoformat, default=DEV_END)
    parser.add_argument("--out", type=Path,
                        default=Path("reports/lgbm_baseline_dev.json"))
    args = parser.parse_args()
    if args.end > DEV_END:
        parser.error(f"禁令 6：--end 最晚為 {DEV_END}")

    payload = run(args.db, args.end)
    report(payload)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\n原始輸出：{args.out}")


if __name__ == "__main__":
    main()
