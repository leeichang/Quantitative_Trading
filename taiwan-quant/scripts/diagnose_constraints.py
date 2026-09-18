#!/usr/bin/env python3
"""
投組約束的代價：逐條量測（開發集）

## 為什麼要先量

CLAUDE.md 的四條約束寫在「投組約束（**Top 3**）」標題下：

```
最多 2 支同產業          3 檔裡是 67% 集中度上限
最多 1 支高波動          3 檔裡是 33%
至少 1 支 defensive      3 檔裡是 33%
兩兩相關係數 < 0.7       與檔數無關
```

**套到 N=10 時前三條的嚴格程度完全改變**：同產業 2/10 = 20%、
高波動 1/10 = 10%。而動能突破選出來的本來就是齊漲的同族群、高波動
名字——第一筆帳本 10 檔裡有 6 檔金融保險。

所以「維持 2 還是按比例放寬」是一個**有代價的決定**。兩邊都猜是最差的
做法，先量再定。

## 湊不滿 N 檔時的兩種算法

約束擋掉太多時會湊不滿 10 檔。這時怎麼配資金有兩種算法，**結論不同**：

```
等權攤到實際檔數   k 檔各 1/k   資金全投入，但集中度上升（違反約束本意）
固定 1/N + 現金    k 檔各 1/N   集中度不變，剩下 (N−k)/N 空手
```

本腳本**兩個都報**，主要看固定 1/N：如果約束說「你找不到 10 檔分散的
標的」，那麼把錢擠進 6 檔等於承認約束無效。

## 禁令 6

一律跑開發集（預設 `--end 2023-12-29`）。2024-01 ~ 2026-08 已經在
`momentum_top10_h40@oos-2026-09-16` 用掉，**不得在上面挑約束參數**。
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
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
)
from taiwan_quant.data.loader import (  # noqa: E402
    HISTORY_DB_PATH,
    RAW_OPEN_COLUMN,
    load_chips,
    load_prices,
)
from taiwan_quant.ranking.constraints import (  # noqa: E402
    ConstraintLimits,
    greedy_pick,
    promote_defensive,
)
from taiwan_quant.ranking.portfolio_features import (  # noqa: E402
    atr_ratio_frame,
    betas,
    build_candidates,
    load_industries,
    percentiles_at,
    return_correlations,
)
from taiwan_quant.ranking.tie_break import (  # noqa: E402
    DEFAULT_TIE_SEED,
    deterministic_jitter,
)

FAMILY = "動能突破"
HOLDING_DAYS = 40
DECISION_STRIDE = 40
N_POSITIONS = 10
CAPITAL = 400_000.0
UNIVERSE_SIZE = 150
UNIVERSE_BASIS = "market_cap"
MIN_CANDIDATES = 30

DEV_END = date(2023, 12, 29)
TRIPS_PER_YEAR = 252 / HOLDING_DAYS

OFF_INDUSTRY = N_POSITIONS
"""同產業上限設成 N 等於關掉這條約束（永遠擋不到）"""

OFF_CORRELATION = 1.0
"""相關上限 1.0 只會擋掉恰好 ±1 的配對，實質等於關掉"""


def limits(
    *,
    industry: int = OFF_INDUSTRY,
    high_vol: int = N_POSITIONS,
    corr: float = OFF_CORRELATION,
    defensive: bool = False,
) -> ConstraintLimits:
    """預設全關，只打開要量的那一條"""
    return ConstraintLimits(
        top_n=N_POSITIONS,
        max_same_industry=industry,
        max_high_volatility=high_vol,
        max_correlation=corr,
        require_defensive=defensive,
    )


CONFIGS: dict[str, ConstraintLimits] = {
    "baseline（無約束）": limits(),
    "同產業 ≤ 2": limits(industry=2),
    "同產業 ≤ 3": limits(industry=3),
    "同產業 ≤ 4": limits(industry=4),
    "高波動 ≤ 1": limits(high_vol=1),
    "高波動 ≤ 3": limits(high_vol=3),
    "相關 < 0.7": limits(corr=0.7),
    "至少 1 defensive": limits(defensive=True),
    "全部嚴格（2/1/0.7/def）": limits(
        industry=2, high_vol=1, corr=0.7, defensive=True
    ),
    "按比例放寬（3/3/0.7/def）": limits(
        industry=3, high_vol=3, corr=0.7, defensive=True
    ),
}


def evaluate(
    configuration: ConstraintLimits,
    decision_dates: list[pd.Timestamp],
    calendar: list[pd.Timestamp],
    scores: dict[str, dict],
    members_at: dict,
    large_at: dict,
    opens: pd.DataFrame,
    actual_opens: pd.DataFrame,
    closes: pd.DataFrame,
    forward: pd.DataFrame,
    atr_ratios: pd.DataFrame,
    industries: dict[str, str],
) -> dict:
    """跑一組約束，回傳可比較的彙總"""
    periods: list[dict] = []
    reasons: Counter[str] = Counter()
    negative_rho_rejections = 0

    for day in decision_dates:
        allowed = tuple(members_at.get(day) or ())
        if not allowed:
            continue
        ranked = pd.Series(
            {sid: scores[sid].get(day, np.nan) for sid in allowed if sid in scores}
        ).dropna()
        realized = forward.loc[day].dropna()
        common = ranked.index.intersection(realized.index)
        if len(common) < MIN_CANDIDATES:
            continue

        ordered_ids = sorted(
            common,
            key=lambda sid: (
                -float(ranked[sid]),
                deterministic_jitter(sid, DEFAULT_TIE_SEED),
            ),
        )
        candidates = build_candidates(
            ordered=ordered_ids,
            scores={sid: float(ranked[sid]) for sid in ordered_ids},
            industries=industries,
            atr_pct=percentiles_at(atr_ratios, day, tuple(ordered_ids)),
            beta_map=betas(closes, day, tuple(ordered_ids)),
        )
        correlations = return_correlations(closes, day, tuple(ordered_ids))

        picked, rejected = greedy_pick(candidates, correlations, configuration)
        if configuration.require_defensive:
            promoted = promote_defensive(
                picked, candidates, correlations, configuration
            )
            if promoted is not None:
                picked = promoted
        for rejection in rejected:
            reasons[rejection.reason.value] += 1
            if "相關係數 -" in rejection.detail:
                negative_rho_rejections += 1

        if not picked:
            continue

        ids = [candidate.stock_id for candidate in picked]
        gross = float(realized[ids].mean())
        entry_day = calendar[calendar.index(day) + 1]
        large_members = large_at.get(day, set())

        # 兩種資金配置的成本不同：固定 1/N 每檔只有 4 萬（多走零股），
        # 等權攤到 k 檔時每檔 400,000/k 更多（更容易買得起整張）。
        def mean_cost(
            amount: float,
            *,
            picked_ids: tuple[str, ...] = tuple(ids),
            trade_day: pd.Timestamp = entry_day,
            historical_large: frozenset[str] = frozenset(large_members),
        ) -> float:
            return float(
                np.mean(
                    [
                        DEFAULT_COST.round_trip_rate(
                            resolve_tier(
                                actual_price=float(actual_opens.loc[trade_day, sid]),
                                adjusted_price=float(opens.loc[trade_day, sid]),
                                amount=amount,
                                large=sid in historical_large,
                                is_etf=is_etf(sid),
                            )
                        )
                        for sid in picked_ids
                    ]
                )
            )

        filled = len(ids) / N_POSITIONS
        periods.append(
            {
                "decision_date": str(day.date()),
                "n_picked": len(ids),
                "gross": gross,
                # 固定 1/N：沒填滿的部分空手，報酬與成本一起按比例縮小
                "net_fixed": (gross - mean_cost(CAPITAL / N_POSITIONS)) * filled,
                # 等權攤到實際檔數：資金全投入
                "net_equal": gross - mean_cost(CAPITAL / len(ids)),
            }
        )

    if not periods:
        return {"periods": 0}

    frame = pd.DataFrame(periods)
    frame["year"] = pd.to_datetime(frame["decision_date"]).dt.year
    summary = {
        "periods": len(frame),
        # 配對檢定要用的原始序列。只有彙總值無法算配對差異的標準誤，
        # 而未配對的標準誤在這裡會大到讓每一組看起來都「沒有差別」。
        "series": {
            row["decision_date"]: row["net_fixed"]
            for row in periods
        },
        "avg_picked": float(frame["n_picked"].mean()),
        "short_periods": int((frame["n_picked"] < N_POSITIONS).sum()),
        "gross_per_trip": float(frame["gross"].mean()),
        "rejections": dict(reasons),
        "negative_rho_rejections": negative_rho_rejections,
    }
    for label, column in (("fixed", "net_fixed"), ("equal", "net_equal")):
        nets = frame[column].to_numpy()
        yearly = frame.groupby("year")[column].mean()
        summary[label] = {
            "net_per_trip": float(nets.mean()),
            "annualised": float((1 + nets.mean()) ** TRIPS_PER_YEAR - 1),
            "cumulative": float(np.prod(1 + nets) - 1),
            "sharpe": float(
                nets.mean() / nets.std(ddof=1) * np.sqrt(TRIPS_PER_YEAR)
            )
            if nets.std(ddof=1) > 0
            else 0.0,
            "positive_years": f"{int((yearly > 0).sum())}/{len(yearly)}",
            "worst_year": float(yearly.min()),
        }
    return summary


def run(db_path: Path, end: date, unlock: bool, reason: str | None) -> dict:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    members = [
        row[0]
        for row in con.execute(
            "SELECT DISTINCT stock_id FROM universe_history WHERE basis = ?",
            (UNIVERSE_BASIS,),
        )
    ]
    con.close()

    # ETF 一律載入：0050 是 beta 的基準，缺它 betas() 會拋錯
    members = list(merge_etf_candidates(tuple(members), include=True))
    prices = load_prices(
        members,
        start=date(2015, 1, 1),
        end=end,
        db_path=db_path,
        unlock_frozen=unlock,
        frozen_reason=reason,
    )
    chips = load_chips(
        members,
        start=date(2015, 1, 1),
        end=end,
        db_path=db_path,
        unlock_frozen=unlock,
        frozen_reason=reason,
    )
    by_stock = build_dataset(members, prices, chips).by_stock

    calendar = V.trading_calendar(by_stock)
    scores = V.precompute_scores(by_stock)[FAMILY]
    opens = pd.DataFrame(
        {sid: bars["open"].astype(float) for sid, bars in by_stock.items()}
    ).reindex(calendar)
    actual_opens = (
        prices[RAW_OPEN_COLUMN].unstack("stock_id").reindex(calendar)
    )
    closes = pd.DataFrame(
        {sid: bars["close"].astype(float) for sid, bars in by_stock.items()}
    ).reindex(calendar)
    forward = forward_returns(opens, closes, holding_days=HOLDING_DAYS)
    atr_ratios = atr_ratio_frame(by_stock)
    industries = load_industries(db_path)

    warmup = 750
    decision_dates = complete_holding_decision_dates(
        calendar,
        calendar[warmup::DECISION_STRIDE],
        holding_days=HOLDING_DAYS,
    )
    members_at = V.resolve_members(
        decision_dates, db_path, UNIVERSE_SIZE, UNIVERSE_BASIS
    )
    large_at = V.resolve_members(
        decision_dates, db_path, 50, UNIVERSE_BASIS
    )

    results = {}
    for name, configuration in CONFIGS.items():
        results[name] = evaluate(
            configuration,
            decision_dates,
            calendar,
            scores,
            members_at,
            large_at,
            opens,
            actual_opens,
            closes,
            forward,
            atr_ratios,
            industries,
        )
        print(f"  {name} 完成", flush=True)
    return {
        "dev_end": str(end),
        "decision_dates": len(decision_dates),
        "configs": results,
    }


BASELINE = "baseline（無約束）"


def paired_report(payload: dict) -> None:
    """
    每一組 vs baseline 的**配對**差異。

    為什麼一定要配對：兩組跑的是同一批決策日、同一份候選池，只有約束
    不同。未配對的標準誤裡包含「這一期市場好不好」的共同變異，那個變異
    在配對相減時會抵銷掉。

    ⚠️ 36 期、優勢集中在少數名字，所以 |t| < 2 一律當成「量不出差別」，
    不可解讀為「沒有差別」。
    """
    configs = payload["configs"]
    base = configs.get(BASELINE, {}).get("series")
    if not base:
        return

    print("── 對 baseline 的配對差異（淨/趟，固定 1/N） " + "─" * 22)
    print(f"{'約束':<26}{'差異':>9}{'標準誤':>9}{'t':>7}{'判讀':>14}")
    for name, summary in configs.items():
        series = summary.get("series")
        if name == BASELINE or not series:
            continue
        shared = sorted(set(base) & set(series))
        diff = np.array([series[day] - base[day] for day in shared])
        if len(diff) < 2 or diff.std(ddof=1) == 0:
            print(f"{name:<26}{diff.mean():>8.2%}{'—':>9}{'—':>7}{'完全相同':>14}")
            continue
        stderr = diff.std(ddof=1) / np.sqrt(len(diff))
        t_stat = diff.mean() / stderr
        verdict = "量不出差別" if abs(t_stat) < 2 else (
            "顯著較差" if t_stat < 0 else "顯著較好"
        )
        print(
            f"{name:<26}{diff.mean():>8.2%}{stderr:>9.2%}"
            f"{t_stat:>7.2f}{verdict:>14}"
        )
    print()


def report(payload: dict) -> None:
    print(f"\n開發集 {payload['dev_end']} 為止｜決策日 {payload['decision_dates']} 期\n")
    header = (
        f"{'約束':<26}{'填滿':>6}{'不足':>6}{'毛/趟':>9}"
        f"{'淨/趟':>9}{'年化':>9}{'Sharpe':>8}{'逐年正':>8}{'最差年':>9}"
    )
    for label in ("fixed", "equal"):
        title = "固定 1/N + 現金" if label == "fixed" else "等權攤到實際檔數"
        print(f"── {title} " + "─" * 46)
        print(header)
        for name, summary in payload["configs"].items():
            if summary.get("periods", 0) == 0:
                print(f"{name:<26}  無有效期數")
                continue
            block = summary[label]
            print(
                f"{name:<26}{summary['avg_picked']:>6.1f}"
                f"{summary['short_periods']:>6}"
                f"{summary['gross_per_trip']:>8.2%}"
                f"{block['net_per_trip']:>9.2%}"
                f"{block['annualised']:>9.1%}"
                f"{block['sharpe']:>8.2f}"
                f"{block['positive_years']:>8}"
                f"{block['worst_year']:>9.2%}"
            )
        print()

    paired_report(payload)

    print("── 剔除原因 " + "─" * 50)
    for name, summary in payload["configs"].items():
        if not summary.get("rejections"):
            continue
        detail = "  ".join(
            f"{reason}={count}"
            for reason, count in sorted(summary["rejections"].items())
            if reason != "top_n_reached"
        )
        if detail:
            negative = summary.get("negative_rho_rejections", 0)
            suffix = f"（其中負相關剔除 {negative}）" if negative else ""
            print(f"{name:<26}{detail}{suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=HISTORY_DB_PATH)
    parser.add_argument(
        "--end",
        type=date.fromisoformat,
        default=DEV_END,
        help="資料載入上限。預設開發集結束日；改動等於動用 OOS（禁令 6）",
    )
    parser.add_argument("--unlock-frozen", action="store_true")
    parser.add_argument("--reason", default=None)
    parser.add_argument(
        "--out", type=Path, default=Path("reports/constraint_cost_dev.json")
    )
    args = parser.parse_args()

    payload = run(args.db, args.end, args.unlock_frozen, args.reason)
    report(payload)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n原始輸出：{args.out}")


if __name__ == "__main__":
    main()
