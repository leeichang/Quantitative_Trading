#!/usr/bin/env python3
"""用既有開發集逐期證據量測兩個各半資金 sleeve 的淨 Sharpe。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HOLDING_DAYS = 40
TRIPS_PER_YEAR = 252 / HOLDING_DAYS
PAIRS = (("動能突破", "均值回歸"), ("籌碼跟隨", "均值回歸"))


def annualized_sharpe(values: np.ndarray) -> float | None:
    """以互不重疊的 40 日期報酬估計年化 Sharpe。"""
    if len(values) < 2 or values.std(ddof=1) == 0:
        return None
    return float(values.mean() / values.std(ddof=1) * np.sqrt(TRIPS_PER_YEAR))


def summarize_sleeve_pair(left: list[float], right: list[float]) -> dict:
    """兩個 sleeve 各占總資金一半；輸入已扣除半資金下的逐檔成本。"""
    left_values = np.asarray(left, dtype=float)
    right_values = np.asarray(right, dtype=float)
    valid = np.isfinite(left_values) & np.isfinite(right_values)
    left_values, right_values = left_values[valid], right_values[valid]
    blend = 0.5 * left_values + 0.5 * right_values
    return {
        "periods": int(len(blend)),
        "left_net_per_trip": float(left_values.mean()),
        "right_net_per_trip": float(right_values.mean()),
        "blend_net_per_trip": float(blend.mean()),
        "left_net_sharpe": annualized_sharpe(left_values),
        "right_net_sharpe": annualized_sharpe(right_values),
        "blend_net_sharpe": annualized_sharpe(blend),
        "period_return_correlation": float(np.corrcoef(left_values, right_values)[0, 1]),
    }


def run(source: Path) -> dict:
    payload = json.loads(source.read_text(encoding="utf-8"))
    series = payload["half_capital_net_series_by_family"]
    return {
        "source": str(source),
        "end": payload["end"],
        "holding_days": HOLDING_DAYS,
        "capital_allocation": "每個 sleeve 50%；每檔 20,000 元；逐檔成本已扣除",
        "pairs": {
            f"{left} + {right}": summarize_sleeve_pair(series[left], series[right])
            for left, right in PAIRS
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=Path("reports/hand_vs_uninformed_dev.json"))
    parser.add_argument("--out", type=Path,
                        default=Path("reports/sleeve_blends_dev.json"))
    args = parser.parse_args()
    payload = run(args.source)
    for name, block in payload["pairs"].items():
        print(f"{name}: {block['periods']} 期｜淨/趟 {block['blend_net_per_trip']:+.2%}"
              f"｜淨 Sharpe {block['blend_net_sharpe']:.2f}"
              f"｜逐期相關 {block['period_return_correlation']:+.2f}")
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"原始輸出：{args.out}")


if __name__ == "__main__":
    main()
