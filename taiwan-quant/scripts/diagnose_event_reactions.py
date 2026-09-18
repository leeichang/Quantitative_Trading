"""
事件發生之後的報酬：漲停與籌碼極端值（開發集）

## 這在回答什麼

`scripts/diagnose_limit_up_events.py` 問的是「**分數**在最高 q 分位時
進場會怎樣」——條件在模型輸出上。

這支問的是相反的方向：「**事件已經是既成事實**之後,從 T+1 開盤進場
會怎樣？」條件在可觀測的事實上,不需要任何模型。

## 為什麼值得重做

排序法的檢定力已經用盡：

```
排序法   N=10 H=40 36 期   每趟 sd 15.4 pp   最小可測效應 5.0%／趟
                          而實測效應 3.1%／趟 —— 低於門檻
事件法   ~1,500 個交易日    SE 縮小 √(1500/36) ≈ 6.5 倍
                          最小可測效應 ≈ 0.8%／趟
```

**同一份資料,換觀測單位就多出 6 倍檢定力。**

## 進場價與可買性

漲停當天收盤買不到。用 `opens.shift(-1)` 進場,**跳空幅度算進成本**。

`locked_all_day`（高 = 低,一價到底）的日子連隔天開盤都排不到單,
實測佔漲停的 4.7%。本腳本同時報告：

```
all        全部事件
buyable    排除 T+1 仍鎖漲停的事件
```

**兩個都報,讓執行成本看得見而不是藏起來。**

## 多重測試

事件 × 持有期是一張網格。**不要從裡面挑最好的一格。**
報告附 `expected_max_by_luck`：把全部格子的超額當成獨立抽樣時,
運氣能給出的期望最大值。任何一格若沒超過它,就是運氣。

## 禁令

```
禁令 1   分位門檻只用 ≤ t 的資料（expanding_quantile_mask）
禁令 2   事件必須發生在當時的標的池內
禁令 6   --end 預設且硬限 2023-12-29
```
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.validate_oos_trailing import resolve_members  # noqa: E402
from taiwan_quant.data.loader import (  # noqa: E402
    DEFAULT_UNIVERSE_BASIS,
    HISTORY_DB_PATH,
    load_chips,
    load_prices,
)
from taiwan_quant.labeling.limit_up import (  # noqa: E402
    limit_up_events,
    locked_all_day,
)
from taiwan_quant.validation.bootstrap import block_bootstrap  # noqa: E402
from taiwan_quant.validation.event_study import (  # noqa: E402
    event_study,
    expanding_quantile_mask,
    forward_returns,
)

DEV_END = date(2023, 12, 29)
UNIVERSE_SIZE = 150
WARMUP = 750
HORIZONS = (5, 10, 20, 40)
QUANTILE = 0.99
REFRESH_EVERY = 20
MIN_OBSERVATIONS = 5000

SEED = 20260918
N_DRAWS = 2000


class DiagnosticError(RuntimeError):
    """資料不足或輸入不合法。"""


def _pivot(prices: pd.DataFrame, column: str) -> pd.DataFrame:
    return prices[column].unstack("stock_id").sort_index()


def build_universe_mask(
    index: pd.DatetimeIndex,
    columns: pd.Index,
    members_at: dict[pd.Timestamp, set[str]],
) -> pd.DataFrame:
    """
    禁令 2：每個 (股票, 日) 是否在**當時**的標的池內。

    `members_at` 只在決策日有值,所以用 forward-fill 補到每個交易日——
    標的池是逐季換的,季中不變。
    """
    frame = pd.DataFrame(False, index=index, columns=columns)
    known = sorted(members_at)
    if not known:
        raise DiagnosticError("解析不到任何標的池快照")
    for position, day in enumerate(known):
        end = known[position + 1] if position + 1 < len(known) else index[-1]
        window = (index >= day) & (index <= end)
        names = [c for c in columns if c in members_at[day]]
        frame.loc[window, names] = True
    return frame


def build_events(
    opens: pd.DataFrame,
    highs: pd.DataFrame,
    lows: pd.DataFrame,
    closes: pd.DataFrame,
    chips: pd.DataFrame | None,
) -> dict[str, pd.DataFrame]:
    """
    事件定義。全部只用 ≤ T 的資料,沒有一個看未來。

    漲停類用還原價（見 `labeling/limit_up.py` 的說明：交易所在除權息日
    也調整參考價,所以還原後的報酬才是限制適用的對象）。
    """
    hit = limit_up_events(closes)
    locked = locked_all_day(highs, lows, hit)

    events: dict[str, pd.DataFrame] = {
        "漲停": hit,
        "漲停未鎖死": hit & ~locked,
        "連兩根漲停": hit & hit.shift(1).fillna(False).astype(bool),
    }

    if chips is not None and not chips.empty:
        for name, column, quantile in (
            ("外資買超極端", "foreign_net", QUANTILE),
            ("融資暴增", "margin_balance", QUANTILE),
        ):
            if column not in chips.columns:
                continue
            frame = chips[column].unstack("stock_id").reindex(
                index=closes.index, columns=closes.columns
            )
            # 轉成相對自身成交量的比率,否則大型股永遠佔據極端值
            scaled = frame.diff() / frame.rolling(20).mean().abs().replace(0, np.nan)
            events[name] = expanding_quantile_mask(
                scaled, quantile=quantile,
                refresh_every=REFRESH_EVERY, min_observations=MIN_OBSERVATIONS,
            )
    return events


def multiple_testing_thresholds(n_cells: int) -> dict[str, float]:
    """
    網格的多重測試門檻,**以 t 值為單位**。

    ⚠️ 不可以拿「超額」去比一個用中位數標準誤算出的門檻。本腳本第一版
    就是那樣寫的,而各格的 SE 差 61 倍（0.103% ~ 6.305%）——結果它挑出
    「連兩根漲停／H=40」（超額 +3.399%）說值得追,而那一格 n=34 日、
    SE 6.305%、**t = 0.54**,是整張網格最沒有訊息的一格。

    t 值已經除掉各自的 SE,所以只能比 t。

    Returns:
        `gumbel`：n 個獨立標準常態的期望最大值 ≈ sqrt(2 ln n)
        `bonferroni`：家族錯誤率 5% 的雙尾臨界值
    """
    if n_cells < 2:
        return {"gumbel": float("nan"), "bonferroni": float("nan")}

    from math import erf, sqrt

    def inverse_normal(p: float) -> float:
        low, high = 0.0, 10.0
        for _ in range(200):
            mid = (low + high) / 2.0
            if 0.5 * (1.0 + erf(mid / sqrt(2.0))) < p:
                low = mid
            else:
                high = mid
        return (low + high) / 2.0

    return {
        "gumbel": float(np.sqrt(2.0 * np.log(n_cells))),
        "bonferroni": inverse_normal(1.0 - 0.025 / n_cells),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(HISTORY_DB_PATH))
    parser.add_argument("--end", default=DEV_END.isoformat())
    parser.add_argument(
        "--output", type=Path, default=Path("reports/event_reactions_dev.json")
    )
    args = parser.parse_args()

    end = date.fromisoformat(args.end)
    if end > DEV_END:
        parser.error("禁令 6：不得載入 2024+；--end 最晚為 2023-12-29")

    db_path = Path(args.db)
    started = time.time()

    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        members = [
            row[0]
            for row in con.execute(
                "SELECT DISTINCT stock_id FROM universe_history "
                "WHERE basis = ? ORDER BY stock_id",
                (DEFAULT_UNIVERSE_BASIS,),
            )
        ]
    finally:
        con.close()
    print(f"標的 {len(members)} 檔", flush=True)

    prices = load_prices(
        members, start=date(2015, 1, 1), end=end, adjusted=True, db_path=db_path
    )
    opens = _pivot(prices, "open")
    highs = _pivot(prices, "high")
    lows = _pivot(prices, "low")
    closes = _pivot(prices, "close")
    print(f"價格 {closes.shape[0]} 日 × {closes.shape[1]} 檔"
          f"｜{time.time() - started:.0f}s", flush=True)

    try:
        chips = load_chips(members, start=date(2015, 1, 1), end=end, db_path=db_path)
    except Exception as exc:  # noqa: BLE001
        print(f"⚠️ 籌碼載入失敗，只做漲停類事件：{exc}", flush=True)
        chips = None
    print(f"籌碼載入完成｜{time.time() - started:.0f}s", flush=True)

    calendar = list(closes.index)
    decision_dates = calendar[WARMUP::20]
    members_at = resolve_members(
        decision_dates, db_path, UNIVERSE_SIZE, DEFAULT_UNIVERSE_BASIS
    )
    universe = build_universe_mask(closes.index, closes.columns, members_at)
    # 暖機期不計入：前 WARMUP 天的標的池與特徵都還不穩
    universe.iloc[:WARMUP] = False
    print(f"標的池遮罩完成，涵蓋 {int(universe.to_numpy().sum())} 個 (股票,日)"
          f"｜{time.time() - started:.0f}s", flush=True)

    events = build_events(opens, highs, lows, closes, chips)
    for name, mask in events.items():
        n = int((mask.astype(bool) & universe).to_numpy().sum())
        print(f"  事件「{name}」：{n} 筆", flush=True)

    rows: list[dict[str, object]] = []
    for horizon in HORIZONS:
        forward = forward_returns(opens, closes, horizon=horizon)
        for name, mask in events.items():
            result = event_study(
                event=name, horizon=horizon, returns=forward,
                event_mask=mask, universe_mask=universe,
            )
            if result.n_dates < 30:
                print(f"  跳過 {name}/H={horizon}：只有 {result.n_dates} 日",
                      flush=True)
                continue
            boot = block_bootstrap(
                np.array(result.paired), np.mean,
                block_length=1, n_draws=N_DRAWS, seed=SEED,
            )
            rows.append({
                "event": name,
                "horizon": horizon,
                "n_events": result.n_events,
                "n_dates": result.n_dates,
                "events_per_date": result.events_per_date,
                "event_mean": result.event_mean,
                "baseline_mean": result.baseline_mean,
                "excess": result.excess,
                "standard_error": result.standard_error,
                "t": result.t_stat,
                "bootstrap_lower": boot.lower,
                "bootstrap_upper": boot.upper,
                "excludes_zero": boot.excludes_zero,
                "paired_series": list(result.paired),
            })
            print(f"  {result.describe()}", flush=True)

    if not rows:
        raise DiagnosticError("沒有任何格子有足夠的日數")

    luck = multiple_testing_thresholds(len(rows))
    payload: dict[str, object] = {
        "end": args.end,
        "universe_size": UNIVERSE_SIZE,
        "warmup": WARMUP,
        "horizons": list(HORIZONS),
        "quantile": QUANTILE,
        "seed": SEED,
        "n_draws": N_DRAWS,
        "multiple_testing_thresholds": luck,
        "elapsed_seconds": round(time.time() - started, 1),
        "grid": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    print(f"\n=== 事件發生之後的超額報酬（相對同日全池）===")
    print(f"{'事件':14s} {'H':>3s} {'事件數':>7s} {'日數':>5s} {'每日':>5s} "
          f"{'超額':>9s} {'SE':>7s} {'t':>6s} {'95% 區間':>21s} {'不含零':>6s}")
    for r in sorted(rows, key=lambda x: (x["event"], x["horizon"])):
        print(f"{str(r['event'])[:12]:14s} {r['horizon']:3d} {r['n_events']:7d} "
              f"{r['n_dates']:5d} {r['events_per_date']:5.1f} "
              f"{r['excess']:+8.3%} {r['standard_error']:7.3%} {r['t']:+6.2f} "
              f"[{r['bootstrap_lower']:+8.3%},{r['bootstrap_upper']:+8.3%}] "
              f"{'✓' if r['excludes_zero'] else '—':>6s}")

    print(f"\n=== 多重測試（{len(rows)} 格）===")
    print(f"期望最大 |t|（Gumbel）{luck['gumbel']:.2f}"
          f"｜Bonferroni 5% 家族 {luck['bonferroni']:.2f}")
    survivors = [r for r in rows if abs(float(r["t"])) > luck["gumbel"]]
    if not survivors:
        print("**沒有任何一格通過多重測試。整張網格都是運氣。**")
    else:
        for r in sorted(survivors, key=lambda x: -abs(float(x["t"]))):
            tag = ("過 Bonferroni" if abs(float(r["t"])) > luck["bonferroni"]
                   else "過 Gumbel")
            print(f"  {r['event']}／H={r['horizon']}"
                  f"｜超額 {float(r['excess']):+.3%}｜t = {float(r['t']):+.2f}｜{tag}")
    print("\n⚠️ 超額最大的那一格不一定是最可信的——比 t，不要比超額。")
    print(f"\n已寫入 {args.output}｜耗時 {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
