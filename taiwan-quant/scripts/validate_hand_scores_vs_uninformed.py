#!/usr/bin/env python3
"""
三個手工分數 vs 無資訊對照組（開發集）

## 這在回答什麼

2026-09-17 實測證明 CLAUDE.md 原本的「隨機進場」對照組太弱——一個標籤
被打亂、證明沒有預測資訊的 LightGBM，仍然顯著打敗它：

```
打亂 − 隨機 = +1.88%／趟   SE 0.81%   t = 2.32   顯著較好
```

所以規格加了第四條對照組。**而先前所有「超過隨機」的結論都要用新門檻
重算**，包括：

> 樣本外 +138.83%，超過隨機 95% 分位 +120.97%

這份診斷做那件重算，對象是三個手工分數族。

## 對照組怎麼造

手工分數的形式是固定權重的特徵混合：

```python
_blend(parts=[_squash(row["momentum_20"], ...), ...], weights=[2.0, 1.5, ...])
```

所以對照組保留**同一個形式**，只把手挑權重換成隨機權重：

```
每個決策日   把 32 個特徵各自取當日橫斷面分位（尺度無關）
每次抽樣     抽一組標準常態權重，**套用到所有決策日**
分數         分位 × 權重的加權和
選股         取前 N 檔
```

⚠️ 權重在時間上固定。每期重抽會失去持續性，退化回隨機選股。

## 為什麼用分位而不是原始特徵值

`momentum_120` 與 `rsi_14` 的量級差兩個數量級，直接加權會讓量級大的
主導——那是量級的效果，不是權重的。

## 禁令 6

一律跑開發集（預設 `--end 2023-12-29`）。
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from taiwan_quant.config.costs import (  # noqa: E402
    DEFAULT as DEFAULT_COST,
    resolve_tier,
)
from taiwan_quant.data.dataset import build_dataset  # noqa: E402
from taiwan_quant.data.etf_universe import is_etf, merge_etf_candidates  # noqa: E402
from taiwan_quant.data.loader import (  # noqa: E402
    HISTORY_DB_PATH,
    RAW_OPEN_COLUMN,
    load_chips,
    load_prices,
)
from taiwan_quant.models.lgbm_baseline import build_features  # noqa: E402
from taiwan_quant.ranking.tie_break import (  # noqa: E402
    DEFAULT_TIE_SEED,
    deterministic_jitter,
)
from taiwan_quant.validation.uninformed import (  # noqa: E402
    DEFAULT_DRAWS,
    NullDistribution,
    cross_sectional_ranks,
    draw_weights,
    random_weight_scores,
)

import scripts.validate_oos_trailing as V  # noqa: E402

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

FAMILY_SPARSITY: dict[str, int] = {
    "動能突破": 4,
    "籌碼跟隨": 6,
    "均值回歸": 4,
}
"""
每個族實際使用的特徵數，用來匹配對照組的稀疏度。

由 `strategies/families.py` 的 `_*_from_row` 逐一數出來：

```
_momentum_from_row        4   ma_ratio_20, momentum_20, momentum_60, volume_ratio_20
_chips_from_row           6   foreign_net_ratio_5/20, institution_agreement_20,
                              institution_strength, margin_to_volume_20, trust_net_ratio_5
_mean_reversion_from_row  4   bollinger_position_20, high_low_position_20,
                              ma_ratio_60, rsi_14
```

⚠️ **密集對照組（32 個特徵全部非零）太弱。** 實測中位只有 +1.45%，
僅比純隨機選股（+1.05%）高 0.4 pp——32 個隨機正負權重會互相抵銷。
匹配稀疏度才是公平的門檻。

改動 `families.py` 的特徵組合時，這張表要跟著改。
"""


def pick_cost(
    picks: list[str],
    entry_day: pd.Timestamp,
    raw_opens: pd.DataFrame,
    opens: pd.DataFrame,
    large_members: set[str],
) -> float:
    """逐檔成本（禁令 3、4）"""
    rates = []
    for stock_id in picks:
        tier = resolve_tier(
            actual_price=float(raw_opens.loc[entry_day, stock_id]),
            adjusted_price=float(opens.loc[entry_day, stock_id]),
            amount=CAPITAL / N_POSITIONS,
            large=stock_id in large_members,
            is_etf=is_etf(stock_id),
        )
        rates.append(DEFAULT_COST.round_trip_rate(tier))
    return float(np.mean(rates))


def ordered_by(scores: pd.Series) -> list[str]:
    """分數降冪 + 確定性抖動破平手"""
    usable = scores.dropna()
    return sorted(
        usable.index,
        key=lambda sid: (-float(usable[sid]),
                         deterministic_jitter(sid, DEFAULT_TIE_SEED)),
    )


def run(db_path: Path, end: date, n_draws: int) -> dict:
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
    all_scores = V.precompute_scores(by_stock)
    families = list(all_scores)
    features = build_features(by_stock)

    def frame(column: str) -> pd.DataFrame:
        return pd.DataFrame(
            {sid: bars[column].astype(float) for sid, bars in by_stock.items()}
        ).reindex(calendar)

    opens, closes, raw_opens = frame("open"), frame("close"), frame(RAW_OPEN_COLUMN)
    # T+1 開盤進、T+HOLDING_DAYS 收盤出 = HOLDING_DAYS 天持有。
    # 第一版寫 shift(-1 - HOLDING_DAYS)，那是 H+1 天，與
    # diagnose_constraints / cost_floor / position_costs 以及
    # data/integrity.holding_dates 都不一致。
    forward = closes.shift(-HOLDING_DAYS) / opens.shift(-1) - 1

    decision_dates = [
        day for day in calendar[WARMUP::DECISION_STRIDE]
        if calendar.index(day) + HOLDING_DAYS < len(calendar)
    ]
    members_at = V.resolve_members(
        decision_dates, db_path, UNIVERSE_SIZE, UNIVERSE_BASIS)
    large_at = V.resolve_members(
        decision_dates, db_path, LARGE_TIER_SIZE, UNIVERSE_BASIS)

    n_features = features.shape[1]
    # 每個族用自己的稀疏度抽一套對照組，不共用
    weights_by_family = {
        name: draw_weights(n_features, n_draws=n_draws,
                           sparsity=FAMILY_SPARSITY.get(name))
        for name in families
    }

    hand_nets: dict[str, list[float]] = {name: [] for name in families}
    draw_nets: dict[str, list[list[float]]] = {
        name: [[] for _ in range(n_draws)] for name in families
    }
    pool_nets: list[float] = []
    used_dates: list[str] = []

    for day in decision_dates:
        position = calendar.index(day)
        allowed = tuple(members_at.get(day) or ())
        if not allowed:
            continue
        realized = forward.loc[day].dropna()
        entry_day = calendar[position + 1]
        tradeable = [
            sid for sid in realized.index
            if sid in allowed
            and np.isfinite(raw_opens.loc[entry_day, sid])
            and raw_opens.loc[entry_day, sid] > 0
        ]
        if len(tradeable) < MIN_CANDIDATES:
            continue
        try:
            today = features.xs(day, level="date")
        except KeyError:
            continue
        today = today.loc[today.index.intersection(tradeable)]
        if len(today) < MIN_CANDIDATES:
            continue

        large_members = set(large_at.get(day, set()))
        used_dates.append(str(day.date()))
        pool_nets.append(float(realized[list(today.index)].mean()))

        for name in families:
            series = all_scores[name]
            hand = pd.Series(
                {sid: series[sid].get(day, np.nan)
                 for sid in today.index if sid in series}
            )
            picks = ordered_by(hand)[:N_POSITIONS]
            if len(picks) < N_POSITIONS:
                hand_nets[name].append(float("nan"))
                continue
            gross = float(realized[picks].mean())
            hand_nets[name].append(
                gross - pick_cost(picks, entry_day, raw_opens, opens,
                                  large_members))

        ranks = cross_sectional_ranks(today)
        for name in families:
            family_weights = weights_by_family[name]
            for index in range(n_draws):
                picks = ordered_by(random_weight_scores(
                    ranks, family_weights[index]))[:N_POSITIONS]
                if len(picks) < N_POSITIONS:
                    draw_nets[name][index].append(float("nan"))
                    continue
                gross = float(realized[picks].mean())
                draw_nets[name][index].append(
                    gross - pick_cost(picks, entry_day, raw_opens, opens,
                                      large_members))
        print(f"  {day.date()} 完成", flush=True)

    results = {}
    null_by_family = {}
    for name in families:
        values = np.asarray(hand_nets[name], dtype=float)
        draw_means = [
            float(np.nanmean(series)) for series in draw_nets[name]
            if np.isfinite(series).any()
        ]
        null_by_family[name] = draw_means
        if not np.isfinite(values).any() or not draw_means:
            results[name] = {"periods": 0}
            continue
        observed = float(np.nanmean(values))
        null = NullDistribution(draws=tuple(draw_means), observed=observed)
        # 配對：同一批決策日，手工分數 vs 抽樣中位那一組
        median_index = int(np.argsort(draw_means)[len(draw_means) // 2])
        paired = np.asarray(hand_nets[name], dtype=float) - np.asarray(
            draw_nets[name][median_index], dtype=float)
        paired = paired[np.isfinite(paired)]
        stderr = (float(paired.std(ddof=1) / np.sqrt(len(paired)))
                  if len(paired) > 1 else 0.0)
        results[name] = {
            "periods": int(np.isfinite(values).sum()),
            "net_per_trip": observed,
            "null_percentile": null.percentile_of_observed(),
            "null_median": float(np.median(draw_means)),
            "null_p95": float(np.quantile(draw_means, 0.95)),
            "null_best": float(np.max(draw_means)),
            "sparsity": FAMILY_SPARSITY.get(name),
            "paired_vs_median_draw": {
                "mean_difference": float(paired.mean()),
                "standard_error": stderr,
                "t": float(paired.mean() / stderr) if stderr > 0 else 0.0,
                "n": len(paired),
            },
            "describe": null.describe(),
        }

    return {
        "end": str(end),
        "periods": len(used_dates),
        "n_draws": n_draws,
        "n_features": n_features,
        "pool_net_per_trip": float(np.mean(pool_nets)) if pool_nets else 0.0,
        "null_draws_by_family": null_by_family,
        "families": results,
    }


def report(payload: dict) -> None:
    print(f"\n開發集 {payload['end']} 為止｜{payload['periods']} 期"
          f"｜每族 {payload['n_draws']} 組隨機權重"
          f"｜特徵池 {payload['n_features']} 欄\n")
    print(f"   持有全池（無成本）{payload['pool_net_per_trip']:+.2%}\n")

    print("── 手工分數 vs 匹配稀疏度的無資訊對照組 " + "─" * 25)
    print(f"{'策略族':<12}{'稀疏':>5}{'淨/趟':>9}{'百分位':>8}{'對照中位':>10}"
          f"{'對照最好':>10}{'配對差異':>10}{'標準誤':>9}{'t':>7}{'判定':>14}")
    for name, block in payload["families"].items():
        if block.get("periods", 0) == 0:
            print(f"{name:<12}  無有效期數")
            continue
        pair = block["paired_vs_median_draw"]
        verdict = ("量不出差別" if abs(pair["t"]) < 2
                   else ("顯著較好" if pair["t"] > 0 else "顯著較差"))
        print(f"{name:<12}{block['sparsity']:>5}{block['net_per_trip']:>9.2%}"
              f"{block['null_percentile']:>8.0%}{block['null_median']:>10.2%}"
              f"{block['null_best']:>10.2%}"
              f"{pair['mean_difference']:>+10.2%}{pair['standard_error']:>9.2%}"
              f"{pair['t']:>7.2f}{verdict:>14}")
    print()
    print("   百分位 = 手工分數贏過多少比例的隨機權重組合")
    print("   ⚠️ 抽樣共用同一份歷史，不是獨立策略。用它算分位可以，")
    print("      用它宣稱統計顯著不行——配對欄才是檢定。")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=HISTORY_DB_PATH)
    parser.add_argument("--end", type=date.fromisoformat, default=DEV_END)
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--out", type=Path,
                        default=Path("reports/hand_vs_uninformed_dev.json"))
    args = parser.parse_args()
    if args.end > DEV_END:
        parser.error(f"禁令 6：--end 最晚為 {DEV_END}")

    payload = run(args.db, args.end, args.draws)
    report(payload)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\n原始輸出：{args.out}")


if __name__ == "__main__":
    main()
