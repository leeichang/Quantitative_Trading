#!/usr/bin/env python3
"""
Look-ahead bias 動態截斷測試（tests/test_lookahead_audit.py 的補完）

靜態掃描只能證明「因子公式」沒寫負數 Ref。它證明不了實作層面的洩漏
（rolling 視窗對齊錯誤、merge 時序錯位、資料匯出時多塞了未來值）。

本測試用物理截斷驗證：

    1. 匯出「完整」Qlib 資料集（至 T_END）
    2. 匯出「截斷」Qlib 資料集（僅至 T_CUT，T_CUT < T_END）
    3. 對同一個因子、同一個日期 T_CUT，比對兩個資料集算出的值
    4. 若因子只用過去資料 → 兩者必須完全相同
       若因子偷看未來     → 截斷版會變成 NaN 或不同值

為什麼不能用 qlib 的 end_time 參數做截斷（已實測）：

    D.features(["2330"], ["Ref($close,-3)/Ref($close,-1)-1"],
               end_time="2026-06-30")
    → 仍算得出 -0.023952...（與不截斷完全相同）

    qlib 的 end_time 只裁切「輸出範圍」，運算式仍會讀取底層儲存中
    end_time 之後的資料。所以必須物理上不要把未來資料寫進資料集。

前置條件：
    DB 內需有資料，且已執行過 POST /api/v1/qlib/export/sync

執行：
    .venv/bin/python -m pytest tests/test_lookahead_truncation.py -v -s

單獨執行並產出報告：
    PYTHONPATH=. .venv/bin/python tests/test_lookahead_truncation.py
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from src.repositories.database import get_session
from src.services.qlib_exporter import ExportConfig, QlibExporter
from src.shared.constants import LABEL_EXPR

# 截斷點與資料集起點。T_CUT 需落在實際交易日上。
EXPORT_START = date(2023, 1, 1)
T_CUT = date(2026, 6, 30)
T_END = date(2026, 9, 11)

TEST_INSTRUMENT = "2330"

# 受測因子：跨類別取樣，涵蓋不同視窗長度與運算子
SAMPLE_FACTORS: dict[str, str] = {
    "kbar_kmid": "($close - $open) / $open",
    "roc_5": "Ref($close, 5) / $close - 1",
    "roc_60": "Ref($close, 60) / $close - 1",
    "ma_20": "Mean($close, 20) / $close",
    "std_20": "Std($close, 20) / $close",
    "rsv_20": "($close - Min($low, 20)) / (Max($high, 20) - Min($low, 20) + 1e-12)",
    "corr_close_vol_20": "Corr($close, Log($volume + 1), 20)",
    "beta_20": "Slope($close, 20) / $close",
    "vol_ma_20": "Mean($volume, 20) / ($volume + 1e-12)",
    "qtlu_60": "Quantile($close, 60, 0.8) / $close",
}

# 正控制組：依設計偷看未來，截斷後必須改變
CONTROL_FUTURE_FACTORS: dict[str, str] = {
    "__label__": LABEL_EXPR,
    "__cheat_next_close__": "Ref($close, -1) / $close - 1",
}


@dataclass(frozen=True)
class ComparisonResult:
    """單一因子在 T_CUT 的截斷前後比對結果"""

    factor_name: str
    expression: str
    full_value: float | None
    truncated_value: float | None

    @property
    def both_nan(self) -> bool:
        return pd.isna(self.full_value) and pd.isna(self.truncated_value)

    @property
    def is_identical(self) -> bool:
        if self.both_nan:
            return True
        if pd.isna(self.full_value) or pd.isna(self.truncated_value):
            return False
        return abs(float(self.full_value) - float(self.truncated_value)) < 1e-12

    @property
    def verdict(self) -> str:
        if self.is_identical:
            return "IDENTICAL"
        return "CHANGED"


def _export(output_dir: Path, end_date: date) -> None:
    """把 DB 資料匯出成 Qlib 格式到指定目錄"""
    session = get_session()
    try:
        exporter = QlibExporter(session)
        exporter.export(
            ExportConfig(
                start_date=EXPORT_START,
                end_date=end_date,
                output_dir=output_dir,
            )
        )
    finally:
        session.close()


def _read_factor_at(
    qlib_dir: Path, expressions: dict[str, str], target: date
) -> dict[str, float | None]:
    """
    在獨立子行程中初始化 qlib 並讀取指定日期的因子值。

    qlib.init() 是全域狀態且不可重複指向不同 provider_uri，
    因此必須用子行程隔離兩次讀取。
    """
    import json
    import subprocess
    import sys

    script = f"""
import json, warnings
warnings.filterwarnings("ignore")
import qlib
from qlib.data import D

qlib.init(provider_uri={str(qlib_dir)!r}, region="tw",
          expression_cache=None, dataset_cache=None)

exprs = {json.dumps(expressions)}
target = {target.isoformat()!r}
names = list(exprs.keys())
fields = [exprs[n] for n in names]

df = D.features([{TEST_INSTRUMENT!r}], fields,
                start_time={EXPORT_START.isoformat()!r}, end_time=target)

out = {{}}
key = ({TEST_INSTRUMENT!r}, target)
for i, n in enumerate(names):
    if key in df.index:
        v = df.loc[key].iloc[i]
        out[n] = None if v != v else float(v)
    else:
        out[n] = None
print("___RESULT___" + json.dumps(out))
"""
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parent.parent),
    )
    for line in proc.stdout.splitlines():
        if line.startswith("___RESULT___"):
            return json.loads(line.removeprefix("___RESULT___"))
    raise RuntimeError(
        f"子行程未回傳結果\nstdout:\n{proc.stdout[-2000:]}\nstderr:\n{proc.stderr[-2000:]}"
    )


def run_truncation_comparison() -> list[ComparisonResult]:
    """執行完整的截斷比對流程"""
    all_exprs = {**SAMPLE_FACTORS, **CONTROL_FUTURE_FACTORS}

    workdir = Path(tempfile.mkdtemp(prefix="lookahead_trunc_"))
    try:
        full_dir = workdir / "qlib_full"
        trunc_dir = workdir / "qlib_trunc"

        _export(full_dir, T_END)
        _export(trunc_dir, T_CUT)

        full_vals = _read_factor_at(full_dir, all_exprs, T_CUT)
        trunc_vals = _read_factor_at(trunc_dir, all_exprs, T_CUT)

        return [
            ComparisonResult(
                factor_name=name,
                expression=expr,
                full_value=full_vals.get(name),
                truncated_value=trunc_vals.get(name),
            )
            for name, expr in all_exprs.items()
        ]
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


@pytest.fixture(scope="module")
def comparison() -> dict[str, ComparisonResult]:
    """整個模組共用一次截斷比對（匯出成本高）"""
    return {r.factor_name: r for r in run_truncation_comparison()}


# ══════════════════════════════════════════════════════════════
# 正控制組先驗：證明這個測試方法有鑑別力
# ══════════════════════════════════════════════════════════════


@pytest.mark.parametrize("name", list(CONTROL_FUTURE_FACTORS))
def test_control_future_factor_changes_after_truncation(
    comparison: dict[str, ComparisonResult], name: str
) -> None:
    """
    正控制組：偷看未來的因子在截斷後必須改變。

    若這兩個測試沒過，代表截斷沒有生效（例如匯出時仍寫入未來資料），
    那麼下面所有 IDENTICAL 的結論都是假通過，不可採信。
    """
    result = comparison[name]
    assert result.verdict == "CHANGED", (
        f"截斷未生效：{name} 在截斷前後值相同（{result.full_value}），"
        f"代表本測試無鑑別力，其餘結論不可採信。公式：{result.expression}"
    )


# ══════════════════════════════════════════════════════════════
# 受測因子：截斷後不得改變
# ══════════════════════════════════════════════════════════════


@pytest.mark.parametrize("name", list(SAMPLE_FACTORS))
def test_sample_factor_is_truncation_invariant(
    comparison: dict[str, ComparisonResult], name: str
) -> None:
    """受測因子在 T_CUT 的值，不得因為「有沒有未來資料」而改變"""
    result = comparison[name]
    assert not result.both_nan, (
        f"{name} 在 {T_CUT} 兩邊都是 NaN，測試無意義（視窗長度可能超出資料範圍）"
    )
    assert result.verdict == "IDENTICAL", (
        f"{name} 截斷前後不一致 → 疑似 look-ahead\n"
        f"  公式      {result.expression}\n"
        f"  完整資料  {result.full_value}\n"
        f"  截斷資料  {result.truncated_value}"
    )


# ══════════════════════════════════════════════════════════════
# 獨立執行：產出人可讀報告
# ══════════════════════════════════════════════════════════════


def build_report() -> str:
    results = run_truncation_comparison()
    by_name = {r.factor_name: r for r in results}

    lines: list[str] = []
    add = lines.append

    add("=" * 88)
    add("Look-ahead Bias 動態截斷測試 — qlib-tw-trader")
    add("=" * 88)
    add("")
    add(f"資料集起點  {EXPORT_START}")
    add(f"截斷點      {T_CUT}   （截斷版資料集只到這天）")
    add(f"完整版終點  {T_END}")
    add(f"受測標的    {TEST_INSTRUMENT}")
    add(f"比對日期    {T_CUT}   （兩個資料集都有這天）")
    add("")

    add("─" * 88)
    add("正控制組（依設計偷看未來 → 必須 CHANGED，否則測試無鑑別力）")
    add("─" * 88)
    control_ok = True
    for name in CONTROL_FUTURE_FACTORS:
        r = by_name[name]
        ok = r.verdict == "CHANGED"
        control_ok = control_ok and ok
        add(f"  {'✓' if ok else '✗'} {name:<24} {r.verdict:<10} "
            f"full={r.full_value}  trunc={r.truncated_value}")
        add(f"      {r.expression}")
    add("")

    if not control_ok:
        add("  ⚠️  正控制組未通過 → 截斷沒有生效，以下結論全部不可採信。")
        add("")

    add("─" * 88)
    add("受測因子（只用過去資料 → 必須 IDENTICAL）")
    add("─" * 88)
    failures = []
    for name in SAMPLE_FACTORS:
        r = by_name[name]
        ok = r.verdict == "IDENTICAL" and not r.both_nan
        if not ok:
            failures.append(r)
        note = "（兩邊皆 NaN，測試無意義）" if r.both_nan else ""
        add(f"  {'✓' if ok else '✗'} {name:<24} {r.verdict:<10} "
            f"full={r.full_value}  trunc={r.truncated_value} {note}")
        if not ok:
            add(f"      {r.expression}")
    add("")

    add("=" * 88)
    if control_ok and not failures:
        add(f"總結：通過。{len(SAMPLE_FACTORS)} 個受測因子皆為截斷不變，")
        add("      且正控制組確認測試有鑑別力。")
    elif not control_ok:
        add("總結：無效。截斷未生效，需先修正測試方法。")
    else:
        add(f"總結：失敗。{len(failures)} 個因子截斷後改變，疑似 look-ahead。")
    add("=" * 88)
    add("")
    add("本測試的邊界（誠實聲明）：")
    add(f"  · 只抽驗 {len(SAMPLE_FACTORS)} 個因子（全庫 303 個），")
    add("    取樣涵蓋不同視窗長度與運算子，但不是全覆蓋。")
    add("  · 只測單一標的、單一截斷點。要完整證明需對多標的、多截斷點掃描。")
    add("  · 測的是「因子計算」層。模型訓練與回測迴圈的時序正確性，")
    add("    由 tests/test_lookahead_audit.py 的 Layer 3 契約檢查覆蓋。")

    return "\n".join(lines)


if __name__ == "__main__":
    print(build_report())
