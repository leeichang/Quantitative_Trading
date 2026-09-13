"""
Look-ahead 物理截斷掃描器

CLAUDE.md 規格 16，來自 qlib-tw-trader 驗證的實測教訓：

    掃描器不能用資料框架的 `end_time` 之類的參數做截斷，必須**物理切掉**
    未來列。實測 qlib：

        D.features(["2330"], ["Ref($close,-3)/Ref($close,-1)-1"],
                   end_time="2026-06-30")
        → 仍算得出 −0.023952...，與不截斷完全相同

    `end_time` 只裁切輸出範圍，運算式仍讀底層儲存的未來資料。任何用
    日期參數做的「截斷測試」都是無效的。

做法：

    full  = builder(bars)                  完整資料
    trunc = builder(bars.iloc[:i + 1])     物理切掉 i 之後的列
    比對第 i 列的值 → 只用過去資料的特徵必須完全相同

為什麼需要這個（靜態掃描抓不到的東西）：

    def zscore(bars):
        close = bars["close"]
        return (close - close.mean()) / close.std()

    這段程式**沒有任何未來參照語法**，找負數 shift 的靜態掃描完全看不到，
    但 `mean()` / `std()` 吃了整段樣本，資料一變長歷史值就變。
    只有截斷測試抓得到。

正控制組（步驟①的教訓）：
    沒有正控制組時，「全部通過」可能只是掃描器壞了。`scan_builder()`
    預設會跑一組故意作弊的特徵，抓不到就拋 `LookaheadError`，
    **絕不回報假通過**。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("open", "high", "low", "close")

DEFAULT_TOLERANCE = 1e-12
"""
浮點比較容許誤差。

運算順序不同會有 1e-16 級誤差，不該當成洩漏。但真洩漏通常會讓值
隨資料長度系統性偏移，遠大於這個門檻。
"""

DEFAULT_CUT_COUNT = 5
"""預設切點數。只切一點可能剛好躲過洩漏"""

FeatureFn = Callable[[pd.DataFrame], pd.Series]
BuilderFn = Callable[[pd.DataFrame], pd.DataFrame]


class LookaheadError(RuntimeError):
    """掃描器本身無法給出可信結論"""


# ══════════════════════════════════════════════════════════════
# 內建作弊特徵：正控制組
# ══════════════════════════════════════════════════════════════


def _cheat_next_close(bars: pd.DataFrame) -> pd.Series:
    """明天的收盤價（最直白的作弊）"""
    return bars["close"].shift(-1)


def _cheat_whole_sample_zscore(bars: pd.DataFrame) -> pd.Series:
    """用整段樣本統計量標準化（最隱蔽，靜態掃描抓不到）"""
    close = bars["close"]
    return (close - close.mean()) / close.std()


def _cheat_backward_fill(bars: pd.DataFrame) -> pd.Series:
    """bfill 把未來值搬到過去"""
    holed = bars["close"].copy()
    holed.iloc[::5] = np.nan
    return holed.bfill()


def _cheat_reverse_cumsum(bars: pd.DataFrame) -> pd.Series:
    """由後往前累加——剩餘期間的總報酬"""
    return bars["close"].iloc[::-1].cumsum().iloc[::-1]


CHEATING_FEATURES: dict[str, FeatureFn] = {
    "control_next_close": _cheat_next_close,
    "control_whole_sample_zscore": _cheat_whole_sample_zscore,
    "control_backward_fill": _cheat_backward_fill,
    "control_reverse_cumsum": _cheat_reverse_cumsum,
}
"""
正控制組目錄：每一個都必須被掃描器抓到。

涵蓋四種洩漏形態：明確未來參照、全樣本統計量、反向填補、反向累加。
抓不到任何一個就代表掃描器有盲點。
"""


# ══════════════════════════════════════════════════════════════
# 結果結構
# ══════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class TruncationFinding:
    """單一切點上的不一致"""

    feature: str
    cut_date: pd.Timestamp
    full_value: float | None
    truncated_value: float | None

    def describe(self) -> str:
        return (
            f"{self.feature} @ {self.cut_date.date()}  "
            f"完整={_fmt(self.full_value)}  截斷={_fmt(self.truncated_value)}"
        )


@dataclass(frozen=True)
class TruncationReport:
    """單一特徵的截斷測試結果"""

    feature: str
    cut_dates: tuple[pd.Timestamp, ...]
    findings: tuple[TruncationFinding, ...]

    @property
    def cuts_tested(self) -> int:
        return len(self.cut_dates)

    @property
    def is_clean(self) -> bool:
        return not self.findings

    def describe(self) -> str:
        if self.is_clean:
            return f"✓ {self.feature}：{self.cuts_tested} 個切點全部一致"
        lines = [f"✗ {self.feature}：{len(self.findings)}/{self.cuts_tested} 個切點不一致"]
        lines.extend(f"    {f.describe()}" for f in self.findings)
        return "\n".join(lines)


@dataclass(frozen=True)
class ScanResult:
    """整個 builder 的掃描結果"""

    reports: dict[str, TruncationReport]
    positive_control_passed: bool
    positive_control_detail: str

    @property
    def is_clean(self) -> bool:
        return all(r.is_clean for r in self.reports.values())

    @property
    def leaking_features(self) -> list[str]:
        return sorted(n for n, r in self.reports.items() if not r.is_clean)

    @property
    def clean_features(self) -> list[str]:
        return sorted(n for n, r in self.reports.items() if r.is_clean)

    def describe(self) -> str:
        lines = [
            "=" * 72,
            "Look-ahead 物理截斷掃描",
            "=" * 72,
            "",
            f"正控制組：{'✓ 通過' if self.positive_control_passed else '✗ 失敗'}"
            f"  {self.positive_control_detail}",
            "",
        ]
        if self.leaking_features:
            lines.append(f"疑似洩漏 {len(self.leaking_features)} 個特徵：")
            lines.extend(self.reports[n].describe() for n in self.leaking_features)
            lines.append("")
        lines.append(f"通過 {len(self.clean_features)} 個特徵：")
        lines.append("  " + ", ".join(self.clean_features) if self.clean_features else "  （無）")
        lines.append("")
        lines.append("=" * 72)
        lines.append("結論：" + ("全部通過" if self.is_clean else "有特徵疑似洩漏"))
        lines.append("=" * 72)
        return "\n".join(lines)


def _fmt(value: float | None) -> str:
    return "NaN" if value is None or pd.isna(value) else f"{value:.10g}"


# ══════════════════════════════════════════════════════════════
# 核心比對
# ══════════════════════════════════════════════════════════════


def _validate(bars: pd.DataFrame) -> None:
    missing = set(REQUIRED_COLUMNS) - set(bars.columns)
    if missing:
        raise ValueError(f"缺少必要欄位：{sorted(missing)}")
    if not bars.index.is_monotonic_increasing:
        raise ValueError("日 K 索引必須依日期升冪排序")


def _default_cut_indices(n_rows: int, cut_count: int = DEFAULT_CUT_COUNT) -> list[int]:
    """
    選分散在資料後段的切點。

    避開最前段（特徵視窗還沒滿，比什麼都是 NaN）與最後一列
    （截斷後等於完整資料，永遠通過）。
    """
    if n_rows < 4:
        return []
    lo = max(1, n_rows // 3)
    hi = n_rows - 2
    if hi < lo:
        return [hi] if hi >= 1 else []
    count = min(cut_count, hi - lo + 1)
    step = max(1, (hi - lo) // max(1, count - 1)) if count > 1 else 1
    cuts = sorted({min(hi, lo + i * step) for i in range(count)})
    return cuts


def _values_differ(full: Any, trunc: Any, tolerance: float) -> bool:
    """
    判斷兩個值是否不同。

    NaN 處理是關鍵（實測教訓）：
      · 兩邊都 NaN → 相同（視窗不足，正常）
      · 一邊 NaN   → **不同**（典型洩漏徵狀：完整算得出、截斷算不出）

    絕不能只寫 `full != trunc`——`nan != nan` 為 True 會誤判，
    而 `nan == nan` 為 False 又會讓「一邊 NaN」漏判。
    """
    full_nan = full is None or pd.isna(full)
    trunc_nan = trunc is None or pd.isna(trunc)

    if full_nan and trunc_nan:
        return False
    if full_nan != trunc_nan:
        return True

    try:
        return abs(float(full) - float(trunc)) > tolerance
    except (TypeError, ValueError):
        return full != trunc


def _extract(frame: pd.DataFrame, cut_date: pd.Timestamp, column: str) -> Any:
    """取出指定日期與欄位的值；取不到回 None"""
    if column not in frame.columns:
        return None
    if cut_date not in frame.index:
        return None
    value = frame.loc[cut_date, column]
    if isinstance(value, pd.Series):
        value = value.iloc[-1]
    return value


def scan_features(
    features: dict[str, FeatureFn],
    bars: pd.DataFrame,
    cut_indices: Sequence[int] | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict[str, TruncationReport]:
    """
    對一組特徵函式做物理截斷測試。

    Args:
        features: {名稱: 函式}，函式吃日 K 回傳 Series
        bars: 日 K（不會被修改）
        cut_indices: 切點的位置索引；None 表示自動分散選取
        tolerance: 浮點比較容許誤差

    Returns:
        {名稱: 報告}

    Raises:
        ValueError: 欄位缺失、索引未排序、切點超出範圍或指向最後一列
    """
    _validate(bars)

    if cut_indices is None:
        cuts = _default_cut_indices(len(bars))
    else:
        cuts = list(cut_indices)
        for idx in cuts:
            if not 0 <= idx < len(bars) - 1:
                raise ValueError(
                    f"切點 {idx} 無效：必須落在 0..{len(bars) - 2}。"
                    "最後一列不可當切點——截斷後與完整資料相同，永遠通過，"
                    "會製造假的安全感。"
                )

    def builder(frame: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({name: fn(frame) for name, fn in features.items()})

    return _scan_columns(builder, bars, cuts, tolerance, list(features))


def _scan_columns(
    builder: BuilderFn,
    bars: pd.DataFrame,
    cuts: Sequence[int],
    tolerance: float,
    column_names: Sequence[str] | None = None,
) -> dict[str, TruncationReport]:
    """對 builder 產出的每個欄位做截斷比對"""
    full = builder(bars)
    columns = list(column_names) if column_names is not None else list(full.columns)

    cut_dates = tuple(bars.index[i] for i in cuts)
    findings: dict[str, list[TruncationFinding]] = {name: [] for name in columns}

    for idx, cut_date in zip(cuts, cut_dates):
        truncated = builder(bars.iloc[: idx + 1])
        for name in columns:
            full_value = _extract(full, cut_date, name)
            trunc_value = _extract(truncated, cut_date, name)
            if _values_differ(full_value, trunc_value, tolerance):
                findings[name].append(
                    TruncationFinding(
                        feature=name,
                        cut_date=cut_date,
                        full_value=None if full_value is None or pd.isna(full_value)
                        else float(full_value),
                        truncated_value=None if trunc_value is None or pd.isna(trunc_value)
                        else float(trunc_value),
                    )
                )

    return {
        name: TruncationReport(
            feature=name,
            cut_dates=cut_dates,
            findings=tuple(findings[name]),
        )
        for name in columns
    }


def scan_builder(
    builder: BuilderFn,
    bars: pd.DataFrame,
    cut_indices: Sequence[int] | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
    run_positive_control: bool = True,
) -> ScanResult:
    """
    對整個特徵 builder 做物理截斷測試，並自我驗證。

    Args:
        builder: 吃日 K 回傳特徵 DataFrame 的函式
        bars: 日 K（不會被修改）
        cut_indices: 切點；None 表示自動選取
        tolerance: 浮點比較容許誤差
        run_positive_control: 是否跑正控制組（**預設開啟，不建議關**）

    Returns:
        ScanResult

    Raises:
        LookaheadError: 正控制組沒抓到作弊特徵。此時掃描器沒有鑑別力，
                        寧可拋錯也不回報「全部通過」——那是最危險的假通過。
    """
    _validate(bars)

    cuts = _default_cut_indices(len(bars)) if cut_indices is None else list(cut_indices)

    control_passed = True
    control_detail = "（未執行）"

    if run_positive_control:
        undetected: list[str] = []
        if not cuts:
            undetected = sorted(CHEATING_FEATURES)
        else:
            control_reports = _scan_columns(
                lambda frame: pd.DataFrame(
                    {name: fn(frame) for name, fn in CHEATING_FEATURES.items()}
                ),
                bars,
                cuts,
                tolerance,
                list(CHEATING_FEATURES),
            )
            undetected = sorted(
                name for name, report in control_reports.items() if report.is_clean
            )

        if undetected:
            raise LookaheadError(
                "正控制組失敗：以下作弊特徵未被偵測 → 掃描器沒有鑑別力，"
                f"拒絕回報結果：{undetected}。"
                f"（切點數 {len(cuts)}，資料 {len(bars)} 列——"
                "資料太短或切點為空都會導致這個結果）"
            )

        control_detail = f"{len(CHEATING_FEATURES)} 個作弊特徵全部被偵測"

    if not cuts:
        raise LookaheadError(
            f"無有效切點（資料僅 {len(bars)} 列）：截斷測試無法給出結論"
        )

    return ScanResult(
        reports=_scan_columns(builder, bars, cuts, tolerance),
        positive_control_passed=control_passed,
        positive_control_detail=control_detail,
    )
