#!/usr/bin/env python3
"""
ETF 輪動的 walk-forward + PBO / DSR（開發集）

## 為什麼標的池只有兩檔

第一版用 8 檔台股 ETF 跑輪動，得到年化 12.5%、MaxDD −16.1%，看起來
比任何單一買進持有的回撤／報酬比好 1.5~2 倍。

**那個結果無效，因為 6 檔吃不下 40 萬的部位。** 開發集
（2015-01 ~ 2023-12）的日成交額中位與 133,333 元部位的佔比：

```
代號  名稱        日成交額中位    部位佔比中位   佔比>1% 的天數
0050 台灣50      6.71 億        0.020%        0/2197
0056 高股息      2.04 億        0.065%       99/2197
0061 寶滬深      1,805 萬       0.739%      976/2197
0052 富邦科技      667 萬       2.000%     1313/2145
0055 MSCI金融     242 萬       5.506%     1978/2197
0057 富邦摩台      113 萬      11.761%     2079/2118
0051 中型100       102 萬      13.075%     2170/2194
0053 元大電子       49 萬      27.378%     2183/2189
```

對照：股票策略實測「4 萬元部位對當日成交額中位僅 98.9 ppm」= 0.0099%。

**0053 的一個部位佔掉當日成交額的 27.4%。** `Tier.ETF_WHOLE` 的 0.05%
滑價是從跳動單位推的，而佔 27% 日成交量時跳動單位毫無意義。

### ⚠️ 差點犯的錯

0052 在 2024 年後的日成交額是 1.92 億，看起來完全夠。但開發集期間
只有 667 萬——**它是後來才長大的**。用 2024 年後的流動性去合理化
2015-2023 的回測，那本身就是 look-ahead。

所以篩選一律用**回測期間本身**的流動性。

### 通過的只有兩檔

```
篩選條件   部位佔比中位 < 0.1%  且  佔比 > 1% 的天數 < 5%
通過       0050、0056
```

兩檔的「輪動」是一個二元傾斜（大盤 vs 高股息），不是策略空間。
但它是**唯一流動性站得住的版本**，而且參數空間小 → 多重測試懲罰也小。

## 統計紀律

```
walk-forward   參數在訓練窗選，套用到下一個測試窗。不是我指定回看期
PBO            同一個 H 之內的候選矩陣，非重疊期數
DSR            n_observations 用有效期數（天數 ÷ H），需 >= 30
n_trials       誠實填 16（4 個回看 × 4 個 H），不是只報最好那一組
```

⚠️ H=120 的有效期數只有 17，**低於 DSR 的 30 下限**——與先前否定
120 日持有時同一面牆。該格會標記為不可檢定。

## 禁令 6

一律跑開發集（預設 `--end 2023-12-29`）。
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from taiwan_quant.config.costs import (  # noqa: E402
    FEE_DISCOUNT_DEFAULT,
    FEE_RATE,
    SLIPPAGE,
    Tier,
)
from taiwan_quant.data.loader import (  # noqa: E402
    HISTORY_DB_PATH,
    RAW_OPEN_COLUMN,
    load_prices,
)
from taiwan_quant.ranking.etf_rotation import (  # noqa: E402
    aligned_views,
    holding_return,
    top_k,
    trailing_momentum,
)
from taiwan_quant.validation.stats import (  # noqa: E402
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
)

LIQUID_ETFS = ("0050", "0056")
"""
通過流動性篩選的 ETF。見模組說明——其餘 6 檔吃不下 40 萬的部位。

⚠️ 這是**開發集期間**的流動性。標的池要擴充必須用擴充當時往前看的
成交額重新篩，不可用今天的流動性合理化歷史回測。
"""

CAPITAL = 400_000.0
LOOKBACK_GRID = (20, 60, 120, 250)
HORIZON_GRID = (20, 40, 60, 120)
N_TRIALS = len(LOOKBACK_GRID) * len(HORIZON_GRID)
"""DSR 的試驗數。k 固定為 1（兩檔選一），所以只有回看 × 持有期"""

MIN_DSR_OBSERVATIONS = 30
"""
DSR 的有效觀測下限。

先前餵錯兩次：一次用 CPCV 的 15 條路徑總報酬（那不是時間序列），
一次用 2,196 筆重疊日報酬（DSR 讀成 1.0000 而非 0.9741）。
有效期數 = 天數 ÷ 持有期。
"""

TRAIN_PERIODS = 8
"""walk-forward 的訓練窗長度（期）。8 期在 H=120 時是 3.8 年"""

DEV_END = date(2023, 12, 29)

# ETF 整股的來回成本率。兩種稅率並列——ETF（受益憑證）的證交稅
# 可能是 0.1% 而非股票的 0.3%，但那是稅法事實，未查證前不改
# config/costs.py，只在報告裡並列。
_FEE_BOTH_SIDES = FEE_RATE * FEE_DISCOUNT_DEFAULT * 2
_SLIP_BOTH_SIDES = SLIPPAGE[Tier.ETF_WHOLE] * 2
COST_TAX_STOCK = _FEE_BOTH_SIDES + 0.003 + _SLIP_BOTH_SIDES
COST_TAX_ETF = _FEE_BOTH_SIDES + 0.001 + _SLIP_BOTH_SIDES


def candidate_returns(
    closes: pd.DataFrame,
    opens: pd.DataFrame,
    lookback: int,
    horizon: int,
    start: int | None = None,
) -> tuple[list[int], list[float]]:
    """
    一組 (回看, 持有期) 在**非重疊**期數上的毛報酬序列。

    Args:
        closes / opens: 已對齊的還原價矩陣
        lookback: 回看交易日數
        horizon: 持有交易日數
        start: 決策網格的起點位置。預設 `max(LOOKBACK_GRID)`

    Returns:
        (決策日的整數位置, 每期毛報酬)

    決策間隔等於持有期：期數不重疊，所以 PBO 與 DSR 的觀測可以當獨立
    樣本處理。重疊會讓兩者都失效（先前踩過）。

    ## 為什麼起點要固定成最長的回看期

    第一版用 `start = max(lookback, 1)`，於是回看 20 的決策日是
    20, 20+H, ... 而回看 250 的是 250, 250+H, ...——**兩個網格只在
    (250−20) 是 H 的倍數時才重合**，H=20 時交集是空的，walk-forward
    直接回報「測試期不足」。

    所有候選必須跑在**同一個決策網格**上，否則比較的不是同一段歷史。
    代價是最短的回看期也要等最長的暖機完成，那是應該付的。
    """
    grid_start = max(LOOKBACK_GRID) if start is None else start
    if grid_start < lookback:
        raise ValueError(f"起點 {grid_start} 不足回看 {lookback}")
    positions, returns = [], []
    for position in range(grid_start, len(closes) - horizon - 1, horizon):
        picks = top_k(trailing_momentum(closes, position, lookback), 1)
        if not picks:
            continue
        positions.append(position)
        returns.append(holding_return(closes, opens, position, horizon, picks))
    return positions, returns


def walk_forward(
    closes: pd.DataFrame,
    opens: pd.DataFrame,
    horizon: int,
    train_periods: int = TRAIN_PERIODS,
) -> dict:
    """
    回看期在訓練窗選，套用到下一個測試期。

    Args:
        closes / opens: 已對齊的還原價矩陣
        horizon: 持有交易日數
        train_periods: 訓練窗的期數

    Returns:
        測試期的毛報酬序列與所選回看期

    **這才是 walk-forward 的重點**：回看期不是我指定的，是資料選的。
    第一版直接寫 `lookback=120`，那等於已經知道答案。
    """
    grids = {lb: candidate_returns(closes, opens, lb, horizon)
             for lb in LOOKBACK_GRID}
    # 只保留所有回看期都有值的共同決策位置，否則訓練窗比較的不是同一段
    common = sorted(set.intersection(*(set(p) for p, _ in grids.values())))
    table = {
        lb: {pos: ret for pos, ret in zip(positions, returns, strict=True)}
        for lb, (positions, returns) in grids.items()
    }

    chosen, test_returns, test_positions = [], [], []
    for index in range(train_periods, len(common)):
        train = common[index - train_periods:index]
        scores = {
            lb: float(np.mean([table[lb][pos] for pos in train]))
            for lb in LOOKBACK_GRID
        }
        best = max(LOOKBACK_GRID, key=lambda lb: (scores[lb], -lb))
        position = common[index]
        chosen.append(best)
        test_positions.append(position)
        test_returns.append(table[best][position])

    return {
        "horizon": horizon,
        "test_positions": test_positions,
        "gross": test_returns,
        "chosen_lookback": chosen,
        "candidate_table": table,
        "common_positions": common,
    }


def summarise(gross: list[float], horizon: int, cost: float) -> dict:
    """毛報酬序列 → 淨報酬、年化、Sharpe、回撤"""
    if not gross:
        return {"periods": 0}
    net = np.array(gross) - cost
    curve = np.cumprod(1.0 + net)
    trips = 252 / horizon
    years = len(net) / trips
    series = pd.Series(curve)
    return {
        "periods": int(len(net)),
        "gross_per_trip": float(np.mean(gross)),
        "cost_per_trip": float(cost),
        "net_per_trip": float(np.mean(net)),
        "total_return": float(curve[-1] - 1.0),
        "annualised": float(curve[-1] ** (1 / years) - 1.0) if years > 0 else 0.0,
        "sharpe": float(np.mean(net) / np.std(net, ddof=1) * np.sqrt(trips))
        if len(net) > 1 and np.std(net, ddof=1) > 0 else 0.0,
        "max_drawdown": float((series / series.cummax() - 1).min()),
        "years": float(years),
    }


def pbo_for_horizon(result: dict, n_splits: int = 16) -> dict | None:
    """
    同一個 H 之內、4 個回看期的候選矩陣做 CSCV。

    形狀是 (期數, 4)。期數不足 `n_splits` 時回 None——**不降低切分數
    硬跑**，那只會得到一個看起來有值的數字。
    """
    table, common = result["candidate_table"], result["common_positions"]
    matrix = np.array([[table[lb][pos] for lb in LOOKBACK_GRID] for pos in common])
    if matrix.shape[0] < n_splits * 2:
        return {"skipped": f"期數 {matrix.shape[0]} < {n_splits * 2}，CSCV 無法切分"}
    half = matrix.shape[0] // 2
    try:
        # 期數為奇數時 matrix[half:] 會多一列，形狀不符會拋錯
        outcome = probability_of_backtest_overfitting(
            matrix[:half], matrix[half:half * 2], n_splits=n_splits
        )
    except ValueError as exc:
        return {"skipped": str(exc)}
    return {"pbo": float(outcome.pbo), "n_candidates": matrix.shape[1],
            "n_periods": int(matrix.shape[0])}


def dsr_for(
    summary: dict, horizon: int, trading_days: int, trial_sharpe_std: float
) -> dict:
    """
    DSR，`n_observations` 用**有效**期數，`sharpe_std` 用實測的橫斷面標準差。

    ## 兩個輸入都很容易餵錯

    `n_observations`：先前餵錯兩次（CPCV 的 15 條路徑總報酬、2,196 筆
    重疊日報酬）。有效期數 = 天數 ÷ 持有期。

    `sharpe_std`：函式預設 1.0，意思是「各試驗的 Sharpe 橫斷面標準差
    是 1.0」。本次 16 個試驗的 Sharpe 落在 0.47~0.68，實測標準差約 0.1。

    ```
    sharpe_std = 1.0   運氣的期望最佳 Sharpe = 1.799   什麼都過不了
    sharpe_std = 0.1   運氣的期望最佳 Sharpe = 0.180   幾乎什麼都過
    ```

    **這個參數決定結論，所以必須用實測值，不可用預設值。**
    """
    effective = trading_days // horizon
    if effective < MIN_DSR_OBSERVATIONS:
        return {"skipped": f"有效觀測 {effective} < {MIN_DSR_OBSERVATIONS}，不可檢定"}
    if summary.get("sharpe", 0.0) <= 0:
        return {"skipped": f"Sharpe {summary.get('sharpe')} 非正，DSR 無意義"}
    if trial_sharpe_std <= 0:
        return {"skipped": f"試驗 Sharpe 標準差 {trial_sharpe_std} 非正"}
    outcome = deflated_sharpe_ratio(
        observed_sharpe=summary["sharpe"],
        n_trials=N_TRIALS,
        n_observations=effective,
        sharpe_std=trial_sharpe_std,
    )
    return {"deflated_sharpe": float(outcome.deflated_sharpe),
            "observed_sharpe": float(outcome.observed_sharpe),
            "expected_max_sharpe": float(outcome.expected_max_sharpe),
            "trial_sharpe_std": float(trial_sharpe_std),
            "n_trials": N_TRIALS, "n_observations": effective}


def benchmarks(closes: pd.DataFrame, warmup: int) -> list[dict]:
    """買進持有各檔 + 固定 50/50（不再平衡，成本只付一次）"""
    rows = []
    for stock_id in closes.columns:
        series = closes[stock_id].iloc[warmup:]
        total = float(series.iloc[-1] / series.iloc[0] - 1)
        years = (series.index[-1] - series.index[0]).days / 365.25
        rows.append({
            "name": f"{stock_id} 買進持有",
            "total_return": total,
            "annualised": float((1 + total) ** (1 / years) - 1),
            "max_drawdown": float((series / series.cummax() - 1).min()),
        })
    blend = closes.iloc[warmup:].div(closes.iloc[warmup]).mean(axis=1)
    total = float(blend.iloc[-1] - 1)
    years = (blend.index[-1] - blend.index[0]).days / 365.25
    rows.append({
        "name": "50/50 買進持有（不再平衡）",
        "total_return": total,
        "annualised": float((1 + total) ** (1 / years) - 1),
        "max_drawdown": float((blend / blend.cummax() - 1).min()),
    })
    return rows


def run(db_path: Path, end: date) -> dict:
    prices = load_prices(list(LIQUID_ETFS), start=date(2015, 1, 1), end=end,
                         adjusted=True, db_path=db_path)
    closes, opens, raw_opens = aligned_views(
        prices["close"].unstack("stock_id"),
        prices["open"].unstack("stock_id"),
        prices[RAW_OPEN_COLUMN].unstack("stock_id"),
    )
    # 兩檔選一，400,000 全押 → 都買得起整張（0050 一張約 13.5 萬）
    lots = {
        stock_id: float((CAPITAL >= raw_opens[stock_id] * 1000).mean())
        for stock_id in raw_opens.columns
    }

    # 16 個試驗（4 回看 × 4 持有期）的 Sharpe，供 DSR 的 sharpe_std 使用。
    # 用預設 1.0 會把運氣門檻拉到 1.799，讓任何東西都不通過。
    trial_sharpes = []
    for horizon in HORIZON_GRID:
        for lookback in LOOKBACK_GRID:
            _, gross = candidate_returns(closes, opens, lookback, horizon)
            block = summarise(gross, horizon, COST_TAX_STOCK)
            if block.get("periods", 0) > 1:
                trial_sharpes.append(block["sharpe"])
    trial_std = float(np.std(trial_sharpes, ddof=1)) if len(trial_sharpes) > 1 else 0.0

    horizons = {}
    for horizon in HORIZON_GRID:
        result = walk_forward(closes, opens, horizon)
        if not result["gross"]:
            horizons[str(horizon)] = {"skipped": "測試期不足"}
            continue
        summary_stock_tax = summarise(result["gross"], horizon, COST_TAX_STOCK)
        horizons[str(horizon)] = {
            "walk_forward_tax_0_3": summary_stock_tax,
            "walk_forward_tax_0_1": summarise(result["gross"], horizon, COST_TAX_ETF),
            "chosen_lookback_counts": {
                str(lb): int(result["chosen_lookback"].count(lb))
                for lb in LOOKBACK_GRID
            },
            "pbo": pbo_for_horizon(result),
            "dsr": dsr_for(summary_stock_tax, horizon, len(closes), trial_std),
            # 固定回看期的對照：walk-forward 若沒有比固定值好，
            # 說明「選參數」這件事本身沒有資訊
            "fixed_lookback": {
                str(lb): summarise(
                    candidate_returns(closes, opens, lb, horizon)[1],
                    horizon, COST_TAX_STOCK,
                )
                for lb in LOOKBACK_GRID
            },
        }

    return {
        "end": str(end),
        "universe": list(LIQUID_ETFS),
        "trading_days": len(closes),
        "first_day": str(closes.index[0].date()),
        "last_day": str(closes.index[-1].date()),
        "whole_lot_fraction": lots,
        "cost_tax_0_3": COST_TAX_STOCK,
        "cost_tax_0_1": COST_TAX_ETF,
        "n_trials": N_TRIALS,
        "trial_sharpes": trial_sharpes,
        "trial_sharpe_std": trial_std,
        "horizons": horizons,
        "benchmarks": benchmarks(closes, warmup=max(LOOKBACK_GRID)),
    }


def report(payload: dict) -> None:
    print(f"\n開發集 {payload['first_day']} ~ {payload['last_day']}"
          f"｜{payload['trading_days']} 交易日｜標的 {payload['universe']}")
    print(f"整股比例 {payload['whole_lot_fraction']}")
    print(f"來回成本  稅 0.3% → {payload['cost_tax_0_3']:.3%}"
          f"｜稅 0.1% → {payload['cost_tax_0_1']:.3%}")
    print(f"試驗數 {payload['n_trials']}（4 個回看 × 4 個持有期）"
          f"｜試驗 Sharpe 標準差 {payload['trial_sharpe_std']:.4f}"
          f"（實測，非預設 1.0）\n")

    print("── walk-forward（回看期由訓練窗選） " + "─" * 34)
    print(f"{'H':>5}{'期數':>6}{'毛/趟':>9}{'淨/趟':>9}{'年化.3%':>9}"
          f"{'年化.1%':>9}{'Sharpe':>8}{'MaxDD':>9}")
    for horizon in HORIZON_GRID:
        block = payload["horizons"].get(str(horizon), {})
        if "skipped" in block:
            print(f"{horizon:>5}  {block['skipped']}")
            continue
        a, b = block["walk_forward_tax_0_3"], block["walk_forward_tax_0_1"]
        print(f"{horizon:>5}{a['periods']:>6}{a['gross_per_trip']:>8.2%}"
              f"{a['net_per_trip']:>9.2%}{a['annualised']:>9.1%}"
              f"{b['annualised']:>9.1%}{a['sharpe']:>8.2f}"
              f"{a['max_drawdown']:>9.1%}")
    print()

    print("── 多重測試校正 " + "─" * 52)
    print(f"{'H':>5}{'PBO':>9}{'判定':>10}{'DSR':>10}{'有效觀測':>9}  備註")
    for horizon in HORIZON_GRID:
        block = payload["horizons"].get(str(horizon), {})
        if "skipped" in block:
            continue
        pbo, dsr = block["pbo"], block["dsr"]
        pbo_text = (f"{pbo['pbo']:.3f}" if "pbo" in pbo else "—")
        verdict = ("過擬合" if pbo.get("pbo", 0) > 0.5 else "通過") if "pbo" in pbo else "—"
        dsr_text = (f"{dsr['deflated_sharpe']:.4f}" if "deflated_sharpe" in dsr else "—")
        obs = dsr.get("n_observations", "—")
        note = pbo.get("skipped", "") or dsr.get("skipped", "")
        print(f"{horizon:>5}{pbo_text:>9}{verdict:>10}{dsr_text:>10}{str(obs):>9}  {note}")
    print()

    print("── walk-forward vs 固定回看期（年化，稅 0.3%）" + "─" * 22)
    print(f"{'H':>5}{'walk-fwd':>10}" + "".join(f"{'固定'+str(lb):>10}" for lb in LOOKBACK_GRID))
    for horizon in HORIZON_GRID:
        block = payload["horizons"].get(str(horizon), {})
        if "skipped" in block:
            continue
        row = f"{horizon:>5}{block['walk_forward_tax_0_3']['annualised']:>10.1%}"
        for lb in LOOKBACK_GRID:
            row += f"{block['fixed_lookback'][str(lb)]['annualised']:>10.1%}"
        print(row)
    print("   walk-forward 沒有贏過固定值 → 「選參數」這件事本身沒有資訊\n")

    print("── 必跑對照組 " + "─" * 54)
    print(f"{'組合':<28}{'總報酬':>10}{'年化':>9}{'MaxDD':>9}")
    for row in payload["benchmarks"]:
        print(f"{row['name']:<28}{row['total_return']:>10.2%}"
              f"{row['annualised']:>9.1%}{row['max_drawdown']:>9.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=HISTORY_DB_PATH)
    parser.add_argument("--end", type=date.fromisoformat, default=DEV_END)
    parser.add_argument("--out", type=Path,
                        default=Path("reports/etf_rotation_dev.json"))
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
