#!/usr/bin/env python3
"""
Look-ahead bias 稽核（D1 評估第 3 題的獨立驗證）

qlib-tw-trader 的 README 宣稱「T 日交易僅使用 T-1 收盤後可得的特徵」。
本檔不採信文件自稱，用三層檢查獨立驗證：

  Layer 1 靜態掃描   掃全部 303 個因子定義，找未來參照運算子
  Layer 2 語意錨定   確認 label 本身「刻意」用未來資料，證明掃描器抓得到
  Layer 3 時序契約   確認進場偏移、embargo、訓練/驗證切分的常數設定正確

Qlib 語意（關鍵）：
    Ref($close,  N)  N 為正 → N 期「之前」（過去，安全）
    Ref($close, -N)  N 為負 → N 期「之後」（未來，look-ahead）

    證據：src/shared/constants.py 的 LABEL_EXPR
          "Ref($close, -3) / Ref($close, -1) - 1"
          註解寫明「Ref($close, -3) = close at T+3」

執行：
    .venv/bin/python -m pytest tests/test_lookahead_audit.py -v

單獨產出報告（不經 pytest）：
    .venv/bin/python tests/test_lookahead_audit.py
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pytest

from src.repositories.factors import ALL_FACTORS
from src.shared.constants import (
    EMBARGO_DAYS,
    LABEL_ENTRY_OFFSET,
    LABEL_EXIT_OFFSET,
    LABEL_EXPR,
    TRAIN_DAYS,
    VALID_DAYS,
)

# 未來參照：Ref(<任意運算式>, <負數>)
FUTURE_REF_PATTERN = re.compile(r"Ref\s*\(\s*(?P<expr>.+?)\s*,\s*(?P<offset>-\d+)\s*\)")

# Qlib 中其他可能洩漏未來資訊的運算子（保守清單，命中即人工複查）
SUSPICIOUS_OPERATORS = ("Future", "Shift(-", "Lead(")


@dataclass(frozen=True)
class FutureRefHit:
    """一筆未來參照命中"""

    factor_name: str
    category: str
    formula: str
    offset: int


def scan_factor(name: str, category: str, formula: str) -> list[FutureRefHit]:
    """掃描單一因子公式中的未來參照"""
    return [
        FutureRefHit(
            factor_name=name,
            category=category,
            formula=formula,
            offset=int(match.group("offset")),
        )
        for match in FUTURE_REF_PATTERN.finditer(formula)
    ]


def scan_all_factors() -> list[FutureRefHit]:
    """掃描全部因子定義"""
    hits: list[FutureRefHit] = []
    for factor in ALL_FACTORS:
        hits.extend(
            scan_factor(
                name=factor["name"],
                category=factor.get("category", "unknown"),
                formula=factor["expression"],
            )
        )
    return hits


def scan_suspicious_operators() -> list[tuple[str, str, str]]:
    """掃描保守清單上的可疑運算子"""
    return [
        (factor["name"], op, factor["expression"])
        for factor in ALL_FACTORS
        for op in SUSPICIOUS_OPERATORS
        if op in factor["expression"]
    ]


# ══════════════════════════════════════════════════════════════
# Layer 1：靜態掃描 — 因子不得參照未來
# ══════════════════════════════════════════════════════════════


def test_factor_library_is_not_empty() -> None:
    """前置條件：因子庫有載入，否則後面的掃描是假通過"""
    assert len(ALL_FACTORS) >= 300, f"因子數異常：{len(ALL_FACTORS)}"


def test_no_future_ref_in_any_factor() -> None:
    """核心檢查：303 個因子公式中不得出現負數 Ref（未來參照）"""
    hits = scan_all_factors()
    detail = "\n".join(
        f"  {h.factor_name} [{h.category}] offset={h.offset}  {h.formula}" for h in hits
    )
    assert not hits, f"發現 {len(hits)} 個因子參照未來資料：\n{detail}"


def test_no_suspicious_future_operators() -> None:
    """次要檢查：不得出現 Future/Shift(-/Lead( 等可疑運算子"""
    hits = scan_suspicious_operators()
    detail = "\n".join(f"  {name}: {op} in {expr}" for name, op, expr in hits)
    assert not hits, f"發現 {len(hits)} 個可疑運算子：\n{detail}"


# ══════════════════════════════════════════════════════════════
# Layer 2：語意錨定 — 證明掃描器真的抓得到
#
# 若掃描器壞掉（regex 寫錯、ALL_FACTORS 空的），Layer 1 會假通過。
# 用 label 當正控制組：label 依設計必須參照未來，掃描器必須抓到它。
# ══════════════════════════════════════════════════════════════


def test_scanner_detects_label_as_future_reference() -> None:
    """正控制組：LABEL_EXPR 依設計參照未來，掃描器必須命中"""
    hits = scan_factor("__label__", "label", LABEL_EXPR)
    assert hits, f"掃描器失效：未能在 LABEL_EXPR 中偵測未來參照：{LABEL_EXPR}"
    offsets = sorted(h.offset for h in hits)
    assert offsets == [-3, -1], f"label 偏移與預期不符：{offsets}（公式 {LABEL_EXPR}）"


def test_scanner_detects_synthetic_future_factor() -> None:
    """正控制組：注入一個明確作弊的公式，掃描器必須命中"""
    cheating = "Ref($close, -1) / $close - 1"  # 用明天的收盤價預測今天
    assert scan_factor("__cheat__", "test", cheating), "掃描器未能偵測明確作弊公式"


def test_scanner_accepts_past_reference() -> None:
    """負控制組：正常的過去參照不得被誤判"""
    legit = "Ref($close, 5) / $close - 1"
    assert not scan_factor("__legit__", "test", legit), "掃描器誤判過去參照為未來參照"


# ══════════════════════════════════════════════════════════════
# Layer 3：時序契約 — 常數設定必須構成有效的 walk-forward
# ══════════════════════════════════════════════════════════════


def test_entry_offset_is_not_same_day() -> None:
    """
    進場不得在 T 日收盤。

    T 日收盤後才算得出 T 日特徵（以及 T 日盤後才公布的三大法人資料），
    因此進場最早只能是 T+1。LABEL_ENTRY_OFFSET < 1 即為 look-ahead。
    """
    assert LABEL_ENTRY_OFFSET >= 1, (
        f"LABEL_ENTRY_OFFSET={LABEL_ENTRY_OFFSET} 代表 T 日收盤即進場，屬 look-ahead"
    )


def test_exit_after_entry() -> None:
    """出場必須晚於進場"""
    assert LABEL_EXIT_OFFSET > LABEL_ENTRY_OFFSET, (
        f"出場偏移 {LABEL_EXIT_OFFSET} 未晚於進場偏移 {LABEL_ENTRY_OFFSET}"
    )


def test_label_expr_matches_offsets() -> None:
    """
    LABEL_EXPR 與 ENTRY/EXIT 偏移必須一致。

    兩者若不同步（例如改了 label 忘了改 offset），回測進出場日期
    會與訓練標籤定義錯開，績效失真且難以察覺。
    """
    offsets = sorted(-h.offset for h in scan_factor("__label__", "label", LABEL_EXPR))
    expected = sorted([LABEL_ENTRY_OFFSET, LABEL_EXIT_OFFSET])
    assert offsets == expected, (
        f"LABEL_EXPR 偏移 {offsets} 與常數 {expected} 不一致；"
        f"LABEL_EXPR={LABEL_EXPR}, ENTRY={LABEL_ENTRY_OFFSET}, EXIT={LABEL_EXIT_OFFSET}"
    )


def test_embargo_covers_label_horizon() -> None:
    """
    Embargo 必須 >= label 的預測跨度。

    label 用到 T+EXIT 的價格，若 embargo 小於這個跨度，
    驗證集最前面幾天的樣本其 label 會與訓練集末端重疊 → 洩漏。
    """
    assert EMBARGO_DAYS >= LABEL_EXIT_OFFSET, (
        f"EMBARGO_DAYS={EMBARGO_DAYS} 小於 label 跨度 {LABEL_EXIT_OFFSET}，"
        "驗證集會與訓練集 label 重疊"
    )


def test_train_and_valid_windows_are_positive() -> None:
    """訓練與驗證窗口必須為正"""
    assert TRAIN_DAYS > 0 and VALID_DAYS > 0
    assert TRAIN_DAYS > VALID_DAYS, (
        f"訓練期 {TRAIN_DAYS} 不大於驗證期 {VALID_DAYS}，樣本配置異常"
    )


# ══════════════════════════════════════════════════════════════
# 獨立執行：產出人可讀的稽核報告
# ══════════════════════════════════════════════════════════════


def build_report() -> str:
    """產出稽核報告文字"""
    lines: list[str] = []
    add = lines.append

    add("=" * 72)
    add("Look-ahead Bias 稽核報告 — qlib-tw-trader")
    add("=" * 72)
    add("")

    add(f"[Layer 1] 靜態掃描：{len(ALL_FACTORS)} 個因子定義")
    hits = scan_all_factors()
    if hits:
        add(f"  ✗ FAIL — {len(hits)} 個因子參照未來資料")
        for h in hits:
            add(f"      {h.factor_name} [{h.category}] offset={h.offset}  {h.formula}")
    else:
        add("  ✓ PASS — 無任何因子使用負數 Ref（未來參照）")

    suspicious = scan_suspicious_operators()
    if suspicious:
        add(f"  ✗ FAIL — {len(suspicious)} 個可疑運算子")
        for name, op, expr in suspicious:
            add(f"      {name}: {op} in {expr}")
    else:
        add(f"  ✓ PASS — 無可疑運算子 {SUSPICIOUS_OPERATORS}")

    by_category: dict[str, int] = {}
    for factor in ALL_FACTORS:
        key = factor.get("category", "unknown")
        by_category[key] = by_category.get(key, 0) + 1
    add(f"  掃描範圍：{by_category}")
    add("")

    add("[Layer 2] 語意錨定：掃描器自我驗證")
    label_hits = scan_factor("__label__", "label", LABEL_EXPR)
    if label_hits:
        offsets = sorted(h.offset for h in label_hits)
        add(f"  ✓ PASS — 掃描器在 label 中命中未來參照 {offsets}")
        add(f"      LABEL_EXPR = {LABEL_EXPR}")
        add("      （label 依設計必須看未來；能抓到它代表 Layer 1 的通過是真的）")
    else:
        add("  ✗ FAIL — 掃描器連 label 都抓不到，Layer 1 結果不可信")

    cheat = scan_factor("__cheat__", "test", "Ref($close, -1) / $close - 1")
    add(f"  {'✓ PASS' if cheat else '✗ FAIL'} — 合成作弊公式偵測")
    legit = scan_factor("__legit__", "test", "Ref($close, 5) / $close - 1")
    add(f"  {'✓ PASS' if not legit else '✗ FAIL'} — 過去參照未被誤判")
    add("")

    add("[Layer 3] 時序契約")
    checks = [
        (
            "進場非當日收盤",
            LABEL_ENTRY_OFFSET >= 1,
            f"LABEL_ENTRY_OFFSET = {LABEL_ENTRY_OFFSET}（T+{LABEL_ENTRY_OFFSET} 進場）",
        ),
        (
            "出場晚於進場",
            LABEL_EXIT_OFFSET > LABEL_ENTRY_OFFSET,
            f"LABEL_EXIT_OFFSET = {LABEL_EXIT_OFFSET}",
        ),
        (
            "label 與偏移一致",
            sorted(-h.offset for h in label_hits)
            == sorted([LABEL_ENTRY_OFFSET, LABEL_EXIT_OFFSET]),
            f"label 偏移 {sorted(-h.offset for h in label_hits)} vs "
            f"常數 {sorted([LABEL_ENTRY_OFFSET, LABEL_EXIT_OFFSET])}",
        ),
        (
            "embargo 覆蓋 label 跨度",
            EMBARGO_DAYS >= LABEL_EXIT_OFFSET,
            f"EMBARGO_DAYS = {EMBARGO_DAYS} >= label 跨度 {LABEL_EXIT_OFFSET}",
        ),
        (
            "訓練期大於驗證期",
            TRAIN_DAYS > VALID_DAYS > 0,
            f"TRAIN_DAYS = {TRAIN_DAYS}, VALID_DAYS = {VALID_DAYS}",
        ),
    ]
    for title, passed, detail in checks:
        add(f"  {'✓ PASS' if passed else '✗ FAIL'} — {title}：{detail}")
    add("")

    all_passed = (
        not hits
        and not suspicious
        and bool(label_hits)
        and bool(cheat)
        and not legit
        and all(passed for _, passed, _ in checks)
    )
    add("=" * 72)
    add(f"總結：{'全部通過' if all_passed else '有項目未通過'}")
    add("=" * 72)
    add("")
    add("本稽核的邊界（誠實聲明）：")
    add("  · 這是靜態＋契約檢查，證明『因子定義層』與『時序常數』沒有作弊。")
    add("  · 它「不」證明特徵計算的實作沒有洩漏（例如 rolling 視窗對齊錯誤、")
    add("    merge 時序錯位）。那需要動態截斷測試：把 T 日之後的資料清空後")
    add("    重算特徵，比對 T 日的值是否改變。")
    add("  · 動態測試需要完整 Qlib 匯出資料，見 tests/test_lookahead_truncation.py。")

    return "\n".join(lines)


if __name__ == "__main__":
    print(build_report())
