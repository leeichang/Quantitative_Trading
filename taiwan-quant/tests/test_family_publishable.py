"""
策略族的可推播旗標

## 為什麼需要它

2026-09-17 對無資訊對照組實測：

```
策略族     淨/趟   百分位（200 組任意 4~6 特徵權重）
動能突破   3.48%    98%
籌碼跟隨   1.67%    65%
均值回歸   0.49%    10%    ← 比 90% 的任意權重還差
```

均值回歸也低於持有全池的 +2.44%。**它不是「還沒驗證」，是驗過而且輸。**

而 `weekly_plan_trailing.py` 接 `--family`，任何人都能傳它進推播路徑。

## 為什麼不從 STRATEGY_FAMILIES 刪掉

刪掉會讓先前的報告無法重跑（禁令 7、8）。所以族留著，只標記不可推播。

## 為什麼不給繞過開關

診斷腳本直接取 `STRATEGY_FAMILIES`，不受這個旗標影響——要研究它隨時
可以。推播路徑不需要例外，而每一個繞過開關遲早會被用。
"""

from __future__ import annotations

import pytest

from taiwan_quant.strategies.families import STRATEGY_FAMILIES, StrategyFamily


def family(name: str) -> StrategyFamily:
    return next(f for f in STRATEGY_FAMILIES if f.name == name)


@pytest.mark.unit
def test_mean_reversion_is_not_publishable() -> None:
    """實測輸給任意權重組合，停用於推播"""
    assert family("均值回歸").publishable is False


@pytest.mark.unit
def test_mean_reversion_states_the_measured_reason() -> None:
    """
    停用一個族必須寫實測依據，不可只留一個布林值。

    半年後看到 `publishable=False` 而沒有理由的人，只能選擇盲信或盲改。
    """
    reason = family("均值回歸").unpublishable_reason

    assert "0.30" in reason and "18" in reason, "要帶上實測數字"
    assert "2026-09-18" in reason, "要帶上日期"
    assert "原理說明" in reason, "要指向依據文件"
    # 舊數字必須保留並標明已更正，不可靜默替換
    assert "0.49" in reason and "已更正" in reason, (
        "更正後要留下舊數字與更正說明，否則讀者無從知道它變過"
    )


@pytest.mark.unit
def test_momentum_and_chips_stay_publishable() -> None:
    """
    這兩族對無資訊對照組也量不出優勢（t = 0.99 / 0.26），但**量不出
    差別不等於較差**——均值回歸是第 10 百分位，那是方向明確的負面。

    停用一個「量不出」的族會是過度反應，而且會讓帳本無可記錄。
    """
    assert family("動能突破").publishable is True
    assert family("籌碼跟隨").publishable is True


@pytest.mark.unit
def test_all_families_remain_in_the_catalogue() -> None:
    """
    停用不等於刪除。三族都要留著，否則先前的報告無法重跑（禁令 7、8）。
    """
    names = {f.name for f in STRATEGY_FAMILIES}
    assert names == {"動能突破", "籌碼跟隨", "均值回歸"}


@pytest.mark.unit
def test_unpublishable_without_reason_is_rejected() -> None:
    """
    `publishable=False` 而沒寫理由要在建構時就拋錯——不可事後補。
    """
    with pytest.raises(ValueError, match="沒有寫實測依據"):
        StrategyFamily(
            name="測試",
            score_fn=lambda bars: 0.0,
            required_columns=("close",),
            description="測試用",
            publishable=False,
        )


@pytest.mark.unit
def test_publishable_family_needs_no_reason() -> None:
    """可推播的族不必填理由，否則每個族都要寫一段廢話"""
    ok = StrategyFamily(
        name="測試",
        score_fn=lambda bars: 0.0,
        required_columns=("close",),
        description="測試用",
    )

    assert ok.publishable is True
    assert ok.unpublishable_reason == ""
