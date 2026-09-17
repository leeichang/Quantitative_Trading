#!/usr/bin/env python3
"""
只在極端分數出現時才進場：漲停事件驅動（開發集）

## 這在回答什麼

實測顯示動能突破的分數**只在極端值有資訊**：

```
分數分位     筆數      平均毛報酬   5 日漲停率
最低 50%    47,927      0.15%      1.8%
50~80%      28,756      0.26%      1.6%    ← 比最低 50% 還差
80~90%       9,585                 3.0%
90~95%       4,793                 4.6%
95~99%       3,834                 9.6%
最高 1%        959      0.85%     20.8%    ← 只有這一段
```

**不是單調的。** 80~95% 區間比 50~80% 還差。

現行策略每 40 日固定選前 10 名，所以大部分持倉來自沒有資訊的區間。
這份診斷問的是：**只在最高 q 分位出現時才進場、其餘時間空手，會怎樣？**

## 時點紀律（禁令 1）

「最高 1%」必須是**只用 ≤ t 的分數**算出來的分位，不是全期分布。
用全期分布定義門檻是 look-ahead——那等於先知道整段歷史的分數範圍。

分位每 `THRESHOLD_REFRESH` 個交易日重算一次（門檻移動很慢，逐日重算
在 330k 筆上要 360M 次運算）。重算時只用當日及之前的資料。

## 這仍然是參數掃描

q × H 是 12 格。**不要從裡面挑最好的。** 這份診斷的輸出是
「事件頻率」與「漲停命中率」，報酬欄只是讓代價看得見。
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
from taiwan_quant.labeling.limit_up import (  # noqa: E402
    hit_within,
    limit_up_events,
    locked_all_day,
)
from taiwan_quant.ranking.tie_break import (  # noqa: E402
    DEFAULT_TIE_SEED,
    deterministic_jitter,
)

import scripts.validate_oos_trailing as V  # noqa: E402

FAMILY = "動能突破"
CAPITAL = 400_000.0
UNIVERSE_SIZE = 150
UNIVERSE_BASIS = "market_cap"
WARMUP = 750

LARGE_TIER_SIZE = 50
"""
市值前 50 名視為 0050 級（滑價 0.3%），51~150 走中型 100（0.4%）。

禁令 4 的明文分層。第一版全部走 `resolve_tier` 的預設 `large=True`，
把排名 51~150 的名字也套 0.3%，低估成本。
"""

N_POSITIONS = 3
"""
固定 3 檔。

事件驅動的重點是「少而準」，而 3 檔在 40 萬下每檔 133,333 元——
實測整股比例 69.4%，成本 0.793%/趟，是可達的成本地板附近。
（見 `2026-09-16_成本地板與可負擔性用錯價格.md`）
"""

THRESHOLD_REFRESH = 20
"""分位門檻的重算間隔（交易日）。門檻移動很慢，逐日重算不划算"""

QUANTILE_GRID = (0.99, 0.98, 0.95)
HORIZON_GRID = (5, 10, 20, 40)

DEV_END = date(2023, 12, 29)


def expanding_thresholds(
    scores: pd.DataFrame, quantile: float, refresh: int = THRESHOLD_REFRESH
) -> pd.Series:
    """
    只用 ≤ t 的分數算出的 `quantile` 分位門檻。

    Args:
        scores: 分數矩陣（index 為交易日，column 為 stock_id）
        quantile: 分位（0.99 = 最高 1%）
        refresh: 重算間隔（交易日）

    Returns:
        與 `scores.index` 對齊的門檻序列。暖機不足處為 NaN

    **池化分位**：把所有標的、所有歷史日期的分數放在一起取分位，
    與原始那份分析的定義一致（959 筆 / 95,854 筆 ≈ 1%）。

    不是橫斷面分位——橫斷面每天都會有「最高 1%」，那就不是事件了。
    池化分位才能回答「今天有沒有出現歷史上罕見的高分」。
    """
    if not 0 < quantile < 1:
        raise ValueError(f"quantile 必須落在 (0, 1)，得到 {quantile}")

    values = scores.to_numpy(dtype="float64")
    result = pd.Series(np.nan, index=scores.index)
    for position in range(WARMUP, len(scores), refresh):
        window = values[: position + 1]
        finite = window[np.isfinite(window)]
        if finite.size < 1000:
            continue
        threshold = float(np.quantile(finite, quantile))
        result.iloc[position: position + refresh] = threshold
    return result


def simulate(
    quantile: float,
    horizon: int,
    calendar: list[pd.Timestamp],
    score_frame: pd.DataFrame,
    members_at: dict,
    large_at: dict,
    opens: pd.DataFrame,
    raw_opens: pd.DataFrame,
    closes: pd.DataFrame,
    limit_hits: pd.DataFrame,
    locked: pd.DataFrame,
) -> dict:
    """
    逐日模擬：門檻觸發才進場，最多 `N_POSITIONS` 檔同時在倉。

    資金未用完的部分是現金（零報酬），所以年化報酬含**現金拖累**——
    這是事件驅動策略的真實代價，用「每趟平均」會完全看不到。
    """
    thresholds = expanding_thresholds(score_frame, quantile)

    cash = CAPITAL
    holdings: list[dict] = []
    equity: list[tuple[pd.Timestamp, float]] = []
    trades: list[dict] = []
    days_with_signal = 0
    blocked_locked = 0
    slot_days = 0
    """持倉「檔·日」數，用來算資金真正投入的比例（現金拖累的分母）"""

    for index, day in enumerate(calendar):
        # ---- 先出場（T+horizon 收盤） ----
        still: list[dict] = []
        for position in holdings:
            if index >= position["exit_index"]:
                exit_price = closes.at[calendar[position["exit_index"]],
                                       position["stock_id"]]
                if not np.isfinite(exit_price):
                    still.append(position)
                    continue
                gross = exit_price / position["entry_price"] - 1
                net = gross - position["cost"]
                cash += position["amount"] * (1 + net)
                trades.append({**{k: position[k] for k in
                                  ("stock_id", "decision_date", "cost")},
                               "gross": float(gross), "net": float(net),
                               "limit_up_hit": bool(position["limit_up_hit"])})
            else:
                still.append(position)
        holdings = still

        # ---- 再進場（門檻觸發，T+1 開盤） ----
        threshold = thresholds.iloc[index]
        if np.isfinite(threshold) and index + 1 + horizon < len(calendar):
            allowed = tuple(members_at.get(day) or ())
            if allowed:
                today = score_frame.loc[day, list(
                    set(allowed) & set(score_frame.columns)
                )].dropna()
                qualifying = today[today >= threshold]
                if not qualifying.empty:
                    days_with_signal += 1
                held = {p["stock_id"] for p in holdings}
                ordered = sorted(
                    qualifying.index,
                    key=lambda sid: (-float(qualifying[sid]),
                                     deterministic_jitter(sid, DEFAULT_TIE_SEED)),
                )
                entry_day = calendar[index + 1]
                large_members = large_at.get(day, set())
                for stock_id in ordered:
                    if len(holdings) >= N_POSITIONS or stock_id in held:
                        continue
                    # 進場日漲停鎖死就買不到——不可假設買得到買不到的東西
                    if bool(locked.at[entry_day, stock_id]):
                        blocked_locked += 1
                        continue
                    entry_price = opens.at[entry_day, stock_id]
                    tradeable = raw_opens.at[entry_day, stock_id]
                    if not (np.isfinite(entry_price) and entry_price > 0
                            and np.isfinite(tradeable) and tradeable > 0):
                        continue
                    # 部位大小 = 可用現金 ÷ 剩餘空槓位。
                    #
                    # 第一版用固定的 `CAPITAL / N_POSITIONS`，並在
                    # `cash < amount` 時跳過。400,000 / 3 = 133,333.33，
                    # 所以 3 × amount **恰好等於**初始資金——任何一點虧損
                    # 都讓第三個槓位永久填不滿。
                    #
                    # 實測那個刀鋒效應會主導結果：成本從 0.865% 改成
                    # 0.940%（只差 0.075 pp）就讓 q=0.99/H=40 的交易數
                    # 63 → 75、毛報酬 4.34% → 6.31%。那是路徑分岔，
                    # 不是策略差異。
                    free_slots = N_POSITIONS - len(holdings)
                    amount = cash / free_slots
                    if amount <= 0:
                        continue
                    tier = resolve_tier(
                        actual_price=float(tradeable),
                        adjusted_price=float(entry_price),
                        amount=amount,
                        large=stock_id in large_members,
                        is_etf=is_etf(stock_id),
                    )
                    cash -= amount
                    holdings.append({
                        "stock_id": stock_id,
                        "decision_date": str(day.date()),
                        "entry_price": float(entry_price),
                        "exit_index": index + 1 + horizon,
                        "amount": amount,
                        "cost": DEFAULT_COST.round_trip_cost(amount, tier) / amount,
                        "limit_up_hit": bool(limit_hits.at[day, stock_id]),
                    })

        # ---- 評價（持倉用當日收盤） ----
        value = cash
        for position in holdings:
            price = closes.at[day, position["stock_id"]]
            if np.isfinite(price):
                value += position["amount"] * (price / position["entry_price"])
            else:
                value += position["amount"]
        equity.append((day, value))
        if index >= WARMUP:
            slot_days += len(holdings)

    frame = pd.DataFrame(equity, columns=["date", "equity"]).set_index("date")
    frame = frame.iloc[WARMUP:]
    if frame.empty or not trades:
        return {"quantile": quantile, "horizon": horizon, "trades": 0}

    curve = frame["equity"] / frame["equity"].iloc[0]
    years = len(curve) / 252
    daily = curve.pct_change().dropna()
    trade_frame = pd.DataFrame(trades)

    return {
        "quantile": quantile,
        "horizon": horizon,
        "trades": len(trade_frame),
        "days_with_signal": days_with_signal,
        "blocked_by_locked_limit": blocked_locked,
        "limit_up_hit_rate": float(trade_frame["limit_up_hit"].mean()),
        "gross_per_trade": float(trade_frame["gross"].mean()),
        "cost_per_trade": float(trade_frame["cost"].mean()),
        "net_per_trade": float(trade_frame["net"].mean()),
        "total_return": float(curve.iloc[-1] - 1),
        "annualised": float(curve.iloc[-1] ** (1 / years) - 1),
        "sharpe": float(daily.mean() / daily.std(ddof=1) * np.sqrt(252))
        if daily.std(ddof=1) > 0 else 0.0,
        "max_drawdown": float((curve / curve.cummax() - 1).min()),
        "years": float(years),
        # 資金真正投入的比例。事件驅動大部分時間空手，而「每筆平均報酬」
        # 完全看不到那段現金拖累——年化欄才看得到。
        "capital_deployed": slot_days / (len(frame) * N_POSITIONS),
        # 命中與沒命中的報酬分開報——加權平均會藏掉「沒命中拖多少」
        "gross_when_hit": float(
            trade_frame.loc[trade_frame["limit_up_hit"], "gross"].mean()
        ) if trade_frame["limit_up_hit"].any() else float("nan"),
        "gross_when_miss": float(
            trade_frame.loc[~trade_frame["limit_up_hit"], "gross"].mean()
        ) if (~trade_frame["limit_up_hit"]).any() else float("nan"),
    }


def base_rates(
    calendar: list[pd.Timestamp],
    score_frame: pd.DataFrame,
    members_at: dict,
    limit_hits: pd.DataFrame,
    horizon: int,
) -> list[dict]:
    """
    分位桶的漲停命中率——用**擴張視窗**門檻，不是全期分布。

    原始那份分析用全期分位分桶，那是 look-ahead。這裡重算一次，
    看時點安全的版本是否還是「只有最高 1% 有資訊」。
    """
    edges = (0.0, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 1.0)
    cuts = {q: expanding_thresholds(score_frame, q) for q in edges[1:-1]}

    buckets: dict[str, list[bool]] = {}
    for index, day in enumerate(calendar):
        if index < WARMUP:
            continue
        allowed = tuple(members_at.get(day) or ())
        if not allowed:
            continue
        columns = list(set(allowed) & set(score_frame.columns))
        today = score_frame.loc[day, columns].dropna()
        if today.empty:
            continue
        for lower, upper in zip(edges[:-1], edges[1:], strict=True):
            low = cuts[lower].iloc[index] if lower in cuts else -np.inf
            high = cuts[upper].iloc[index] if upper in cuts else np.inf
            if not (np.isfinite(low) or lower == 0.0):
                continue
            selected = today[(today >= low) & (today < high)]
            if selected.empty:
                continue
            label = f"{lower:.0%}~{upper:.0%}"
            buckets.setdefault(label, []).extend(
                bool(limit_hits.at[day, sid]) for sid in selected.index
            )

    return [
        {"bucket": label, "observations": len(values),
         "limit_up_rate": float(np.mean(values))}
        for label, values in buckets.items()
    ]


def report(payload: dict) -> None:
    print(f"\n開發集 {payload['dev_end']} 為止｜資金 {CAPITAL:,.0f} 元"
          f"｜{N_POSITIONS} 檔上限\n")

    print(f"── 分位桶的 {payload['base_rate_horizon']} 日漲停率"
          "（擴張視窗門檻，時點安全）" + "─" * 8)
    print(f"{'分位桶':<12}{'觀察數':>10}{'漲停率':>9}")
    for row in payload["base_rates"]:
        print(f"{row['bucket']:<12}{row['observations']:>10,}"
              f"{row['limit_up_rate']:>9.2%}")
    print()

    print("── 事件驅動模擬 " + "─" * 58)
    print(f"{'分位':>6}{'H':>5}{'交易數':>7}{'觸發日':>7}{'鎖死擋':>8}"
          f"{'漲停命中':>9}{'毛/筆':>8}{'成本':>8}{'淨/筆':>8}"
          f"{'資金投入':>9}{'年化':>8}{'Sharpe':>8}{'MaxDD':>9}")
    for item in payload["grid"]:
        if item.get("trades", 0) == 0:
            print(f"{item['quantile']:>6.2f}{item['horizon']:>5}   無交易")
            continue
        print(
            f"{item['quantile']:>6.2f}{item['horizon']:>5}{item['trades']:>7}"
            f"{item['days_with_signal']:>7}{item['blocked_by_locked_limit']:>8}"
            f"{item['limit_up_hit_rate']:>9.1%}{item['gross_per_trade']:>8.2%}"
            f"{item['cost_per_trade']:>8.3%}{item['net_per_trade']:>8.2%}"
            f"{item['capital_deployed']:>9.1%}"
            f"{item['annualised']:>8.1%}{item['sharpe']:>8.2f}"
            f"{item['max_drawdown']:>9.1%}"
        )
    print()

    print("── 命中與沒命中分開看 " + "─" * 50)
    print(f"{'分位':>6}{'H':>5}{'命中率':>8}{'命中時毛':>10}{'沒命中時毛':>11}")
    for item in payload["grid"]:
        if item.get("trades", 0) == 0:
            continue
        print(f"{item['quantile']:>6.2f}{item['horizon']:>5}"
              f"{item['limit_up_hit_rate']:>8.1%}"
              f"{item['gross_when_hit']:>10.2%}"
              f"{item['gross_when_miss']:>11.2%}")
    print()
    print("⚠️  q × H 是 12 格參數空間。**不要從裡面挑最好的那一格。**")
    print("    這份診斷的輸出是觸發頻率與漲停命中率，報酬欄只是讓代價看得見。")


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
    members = list(merge_etf_candidates(tuple(members), include=True))

    prices = load_prices(members, start=date(2015, 1, 1), end=end, adjusted=True,
                         db_path=db_path, unlock_frozen=unlock, frozen_reason=reason)
    chips = load_chips(members, start=date(2015, 1, 1), end=end, db_path=db_path,
                       unlock_frozen=unlock, frozen_reason=reason)
    by_stock = build_dataset(members, prices, chips).by_stock

    calendar = V.trading_calendar(by_stock)
    raw_scores = V.precompute_scores(by_stock)[FAMILY]

    def frame(column: str) -> pd.DataFrame:
        return pd.DataFrame(
            {sid: bars[column].astype(float) for sid, bars in by_stock.items()}
        ).reindex(calendar)

    opens, closes = frame("open"), frame("close")
    highs, lows = frame("high"), frame("low")
    raw_opens = frame(RAW_OPEN_COLUMN)
    score_frame = pd.DataFrame(
        {sid: pd.Series(values) for sid, values in raw_scores.items()}
    ).reindex(calendar)

    events = limit_up_events(closes)
    locked = locked_all_day(highs, lows, events)
    members_at = V.resolve_members(
        calendar[WARMUP:], db_path, UNIVERSE_SIZE, UNIVERSE_BASIS
    )
    large_at = V.resolve_members(
        calendar[WARMUP:], db_path, LARGE_TIER_SIZE, UNIVERSE_BASIS
    )

    hits_cache = {h: hit_within(events, h) for h in HORIZON_GRID}

    grid = []
    for quantile in QUANTILE_GRID:
        for horizon in HORIZON_GRID:
            grid.append(simulate(
                quantile, horizon, calendar, score_frame, members_at, large_at,
                opens, raw_opens, closes, hits_cache[horizon], locked,
            ))
        print(f"  q={quantile} 完成", flush=True)

    return {
        "dev_end": str(end),
        "capital": CAPITAL,
        "n_positions": N_POSITIONS,
        "base_rate_horizon": 5,
        "base_rates": base_rates(
            calendar, score_frame, members_at, hits_cache[5], horizon=5
        ),
        "grid": grid,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=HISTORY_DB_PATH)
    parser.add_argument("--end", type=date.fromisoformat, default=DEV_END)
    parser.add_argument("--unlock-frozen", action="store_true")
    parser.add_argument("--reason", default=None)
    parser.add_argument("--out", type=Path,
                        default=Path("reports/limit_up_events_dev.json"))
    args = parser.parse_args()

    payload = run(args.db, args.end, args.unlock_frozen, args.reason)
    report(payload)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\n原始輸出：{args.out}")


if __name__ == "__main__":
    main()
