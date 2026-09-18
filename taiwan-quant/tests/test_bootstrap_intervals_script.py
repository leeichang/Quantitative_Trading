"""
`scripts/diagnose_bootstrap_intervals.py` 的來源解析測試

兩種報告格式（開發集的多組 `series`、OOS 的單組 `per_period`）走不同
分支，而 OOS 的 `benchmarks` 是**半嵌套**的——ETF 在一層字典裡，其餘
是純量：

```json
"benchmarks": {
  "etf": {"0050": {"total_return": 2.42, ...}, ...},
  "equal_weight_universe": 0.6145,
  "random_p95": 1.2097
}
```

攤平寫錯會靜默少掉基準，而少掉的那個剛好可能是最強的 0050——
**那會讓「無法與任一基準區分」變成一句看起來比較好的結論。**
所以這裡逐個斷言基準數量與數值。
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.diagnose_bootstrap_intervals import (
    DiagnosticError,
    acceptance_check,
    load_source,
    total_return_intervals,
)


def _write(tmp_path, name, payload):
    path = tmp_path / name
    path.write_text(json.dumps(payload, ensure_ascii=False))
    return path


def _dev_payload(n=6):
    series = {f"2020-0{i % 9 + 1}-01": 0.01 * (i + 1) for i in range(n)}
    return {
        key: {"label": f"標籤{key}", "series": dict(series)}
        for key in ("model", "hand", "shuffled_labels",
                    "random_median", "equal_weight")
    }


# ══════════════════════════════════════════════════════════════
# 開發集格式
# ══════════════════════════════════════════════════════════════


def test_dev_format_loads_five_arms_and_the_pair_list(tmp_path):
    parsed = load_source(_write(tmp_path, "dev.json", _dev_payload()))

    assert [arm.key for arm in parsed.arms] == [
        "model", "hand", "shuffled_labels", "random_median", "equal_weight"
    ]
    assert len(parsed.pairs) == 5
    assert parsed.benchmarks == {"標籤equal_weight": pytest.approx(0.21 * 0 + (
        float(np.prod(1 + np.array([0.01 * (i + 1) for i in range(6)])) - 1)))}


def test_dev_format_rejects_a_missing_arm(tmp_path):
    payload = _dev_payload()
    del payload["hand"]

    with pytest.raises(DiagnosticError, match="hand.series"):
        load_source(_write(tmp_path, "dev.json", payload))


def test_dev_format_rejects_mismatched_period_counts(tmp_path):
    """期數不一致會讓配對比較對錯期，必須立即失敗而不是截斷"""
    payload = _dev_payload()
    payload["hand"]["series"] = {"2020-01-01": 0.05}

    with pytest.raises(DiagnosticError, match="期數不一致"):
        load_source(_write(tmp_path, "dev.json", payload))


# ══════════════════════════════════════════════════════════════
# OOS 格式
# ══════════════════════════════════════════════════════════════


def _oos_payload():
    return {
        "strategy_version": "momentum_top10_h40@test",
        "per_period": [
            {"decision_date": "2024-01-02", "gross": 0.08, "net": 0.07},
            {"decision_date": "2024-03-01", "gross": -0.03, "net": -0.04},
            {"decision_date": "2024-05-02", "gross": 0.12, "net": 0.11},
        ],
        "benchmarks": {
            "etf": {
                "0050": {"total_return": 2.4215, "sharpe": 1.90},
                "0051": {"total_return": 1.1022, "sharpe": 1.32},
            },
            "equal_weight_universe": 0.6145,
            "random_p95": 1.2097,
        },
    }


def test_oos_format_reads_the_net_series(tmp_path):
    parsed = load_source(_write(tmp_path, "oos.json", _oos_payload()))

    assert len(parsed.arms) == 1
    assert parsed.arms[0].label == "momentum_top10_h40@test"
    assert parsed.arms[0].series.tolist() == pytest.approx([0.07, -0.04, 0.11])
    assert parsed.pairs == ()


def test_oos_format_flattens_nested_etf_benchmarks(tmp_path):
    """
    **少攤平一個基準，就會漏掉那個比較。**

    0050 是最強的基準，漏掉它會讓結論看起來比較好。
    """
    parsed = load_source(_write(tmp_path, "oos.json", _oos_payload()))

    assert set(parsed.benchmarks) == {
        "0050", "0051", "equal_weight_universe", "random_p95"
    }
    assert parsed.benchmarks["0050"] == pytest.approx(2.4215)
    assert parsed.benchmarks["random_p95"] == pytest.approx(1.2097)


def test_oos_format_rejects_an_empty_per_period(tmp_path):
    payload = _oos_payload()
    payload["per_period"] = []

    with pytest.raises(DiagnosticError, match="per_period"):
        load_source(_write(tmp_path, "oos.json", payload))


def test_oos_format_rejects_per_period_without_net(tmp_path):
    payload = _oos_payload()
    payload["per_period"] = [{"decision_date": "2024-01-02", "gross": 0.08}]

    with pytest.raises(DiagnosticError, match="net"):
        load_source(_write(tmp_path, "oos.json", payload))


def test_missing_file_fails_fast(tmp_path):
    with pytest.raises(DiagnosticError, match="不存在"):
        load_source(tmp_path / "absent.json")


# ══════════════════════════════════════════════════════════════
# 驗收判定
# ══════════════════════════════════════════════════════════════


def test_a_benchmark_inside_the_interval_is_reported_as_indistinguishable(tmp_path):
    parsed = load_source(_write(tmp_path, "oos.json", _oos_payload()))
    totals = total_return_intervals(parsed.arms)

    check = acceptance_check(totals, parsed.benchmarks)
    entry = check["arms"]["strategy"]

    # 三期序列的區間極寬，四個基準應該全部落在裡面
    assert entry["verdict"] == "無法與任一基準區分"
    assert set(entry["benchmarks_inside_interval"]) <= set(parsed.benchmarks)


def test_a_benchmark_below_the_lower_bound_is_not_reported_as_inside():
    """基準遠低於區間下界時，那個比較是分辨得出來的"""
    totals = total_return_intervals(
        (_Arm("strategy", "測試", np.full(40, 0.05)),)
    )

    check = acceptance_check(totals, {"極低基準": -0.90})
    entry = check["arms"]["strategy"]

    assert entry["benchmarks_inside_interval"] == []
    assert entry["verdict"] == "至少一個基準在區間下界之下"


from scripts.diagnose_bootstrap_intervals import Arm as _Arm  # noqa: E402
