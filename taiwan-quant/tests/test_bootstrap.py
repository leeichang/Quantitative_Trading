"""
Block bootstrap 信賴區間的測試

## 為什麼要有這個模組

主線至今每一個「量不出差別」都來自 n=36 的配對 t 檢定。實測那些序列：

```
配對差異序列        lag1 自相關    偏態      峰度
模型 − 打亂           −0.023     +1.20     5.81
模型 − 手工           +0.080     −0.40     4.85
手工 − 打亂           −0.074     +1.45     4.92
```

**自相關幾乎是零，但峰度 4.9~5.8（常態是 3）。** 所以 t 檢定在這裡的
問題是**尾部**，不是相依——`03_待辦與改進方向.md` 第 3 項把原因寫成
「有效樣本太少」，那只說對一半。

而水準序列（非配對）是另一回事：

```
等權全池        lag1 −0.281
隨機 10 檔      lag1 −0.277
標籤打亂        lag1 −0.278
```

**負自相關。** 對累積報酬來說，負自相關會**降低**長期變異，所以 IID
重抽會把區間估得太寬。兩個方向都需要區塊長度可調。

`validation/external/__init__.py` 早先就記了不引入外部
`timeseries.bootstrap_sharpe` 的理由：「逐點 IID 重抽，對自相關報酬會
低估區間寬度」。本模組是那個決定的正面實作。

## 測試的設計原則

第 8 個測試（`test_blocks_widen_the_interval_on_autocorrelated_data`）是
承重的那一個：**如果區塊長度不影響區間寬度，整個模組就沒有意義。**

其餘測試盡量用手算的期望值斷言，而不是拿兩條程式路徑互比——
2026-09-18 的持有期 off-by-one 就是因為程式與測試照同一個誤解寫成，
所以測試驗證了我的理解而不是行為。
"""

from __future__ import annotations

import numpy as np
import pytest

from taiwan_quant.validation.bootstrap import (
    BootstrapError,
    BootstrapResult,
    block_bootstrap,
    compound_total_return,
    moving_block_indices,
    paired_difference,
)

# ══════════════════════════════════════════════════════════════
# moving_block_indices
# ══════════════════════════════════════════════════════════════


def test_indices_cover_the_requested_length_and_stay_in_range():
    rng = np.random.default_rng(0)
    idx = moving_block_indices(20, block_length=4, rng=rng)

    assert len(idx) == 20
    assert idx.min() >= 0
    assert idx.max() < 20


def test_block_length_equal_to_n_gives_a_circular_rotation():
    """單一區塊覆蓋整段時，重抽只是環狀旋轉——每個位置恰好出現一次"""
    rng = np.random.default_rng(7)
    idx = moving_block_indices(12, block_length=12, rng=rng)

    assert sorted(idx.tolist()) == list(range(12))
    # 環狀：相鄰差為 +1，只有一處回繞
    steps = np.diff(idx)
    assert (steps == 1).sum() == 11 - (idx[0] != 0)


def test_block_length_one_draws_each_position_independently():
    """區塊長度 1 就是 IID 重抽，所以會出現重複"""
    rng = np.random.default_rng(3)
    idx = moving_block_indices(50, block_length=1, rng=rng)

    assert len(set(idx.tolist())) < 50


def test_blocks_are_contiguous_runs():
    """區塊內必須連續，否則就不是 block bootstrap"""
    rng = np.random.default_rng(11)
    block = 5
    idx = moving_block_indices(20, block_length=block, rng=rng)

    for start in range(0, 20, block):
        run = idx[start : start + block]
        expected = (run[0] + np.arange(len(run))) % 20
        assert run.tolist() == expected.tolist()


# ══════════════════════════════════════════════════════════════
# 決定性
# ══════════════════════════════════════════════════════════════


def test_same_seed_gives_identical_result():
    series = np.array([0.01, -0.02, 0.03, 0.00, 0.05, -0.01, 0.02, 0.04])
    kwargs = dict(block_length=2, n_draws=200, seed=42)

    first = block_bootstrap(series, np.mean, **kwargs)
    second = block_bootstrap(series, np.mean, **kwargs)

    assert first.draws == second.draws
    assert first.lower == second.lower
    assert first.upper == second.upper


def test_different_seed_gives_a_different_draw_set():
    series = np.array([0.01, -0.02, 0.03, 0.00, 0.05, -0.01, 0.02, 0.04])

    first = block_bootstrap(series, np.mean, block_length=2, n_draws=200, seed=1)
    second = block_bootstrap(series, np.mean, block_length=2, n_draws=200, seed=2)

    assert first.draws != second.draws


# ══════════════════════════════════════════════════════════════
# 區間本身
# ══════════════════════════════════════════════════════════════


def test_interval_brackets_the_point_estimate():
    rng = np.random.default_rng(5)
    series = rng.normal(0.03, 0.10, size=200)

    result = block_bootstrap(series, np.mean, block_length=1, n_draws=2000, seed=9)

    assert result.point == pytest.approx(float(np.mean(series)))
    assert result.lower < result.point < result.upper


def test_iid_bootstrap_of_the_mean_recovers_the_analytic_standard_error():
    """
    區塊長度 1、統計量為平均數時，重抽分布的標準差應該收斂到 σ/√n。

    這是手算的期望值，不是拿另一條程式路徑比對。
    """
    rng = np.random.default_rng(17)
    n = 400
    series = rng.normal(0.0, 1.0, size=n)
    analytic_se = float(np.std(series, ddof=1) / np.sqrt(n))

    result = block_bootstrap(series, np.mean, block_length=1, n_draws=4000, seed=23)
    bootstrap_se = float(np.std(np.asarray(result.draws), ddof=1))

    assert bootstrap_se == pytest.approx(analytic_se, rel=0.10)
    # 95% 區間寬度 ≈ 2 × 1.96 × SE
    assert result.width == pytest.approx(2 * 1.96 * analytic_se, rel=0.15)


def test_blocks_widen_the_interval_on_autocorrelated_data():
    """
    **本模組存在的理由。**

    構造一個每個值重複 5 次的序列——區塊內完全相關，區塊間獨立。
    有效樣本數是 n/5，所以平均數的 SE 應該是 IID 估計的 √5 ≈ 2.24 倍。

    若區塊長度不影響寬度，這個測試會失敗，而整個模組就沒有意義。
    """
    rng = np.random.default_rng(31)
    block = 5
    base = rng.normal(0.0, 1.0, size=80)
    series = np.repeat(base, block)

    iid = block_bootstrap(series, np.mean, block_length=1, n_draws=3000, seed=13)
    blocked = block_bootstrap(series, np.mean, block_length=block,
                              n_draws=3000, seed=13)

    ratio = blocked.width / iid.width
    assert ratio == pytest.approx(np.sqrt(block), rel=0.20), (
        f"區塊寬度比應接近 √{block} = {np.sqrt(block):.2f}，得到 {ratio:.2f}"
    )


def test_negative_autocorrelation_narrows_the_blocked_interval():
    """
    水準序列實測是負自相關（lag1 ≈ −0.28）。方向與上一個測試相反：
    負自相關下區塊重抽的區間應該**比 IID 窄**。

    交替序列 +1, −1, +1, ... 的區塊和幾乎為零，變異遠小於 IID。
    """
    series = np.array([1.0, -1.0] * 60)

    iid = block_bootstrap(series, np.mean, block_length=1, n_draws=3000, seed=29)
    blocked = block_bootstrap(series, np.mean, block_length=2, n_draws=3000, seed=29)

    assert blocked.width < iid.width


def test_level_controls_the_interval_width():
    rng = np.random.default_rng(41)
    series = rng.normal(0.02, 0.08, size=150)

    narrow = block_bootstrap(series, np.mean, block_length=1,
                             n_draws=3000, seed=5, level=0.80)
    wide = block_bootstrap(series, np.mean, block_length=1,
                           n_draws=3000, seed=5, level=0.99)

    assert narrow.width < wide.width


def test_excludes_zero_reports_whether_the_interval_clears_zero():
    clearly_positive = np.full(60, 0.05)
    straddling = np.array([-0.05, 0.05] * 30)

    positive = block_bootstrap(clearly_positive, np.mean, block_length=1,
                               n_draws=500, seed=2)
    mixed = block_bootstrap(straddling, np.mean, block_length=1,
                            n_draws=500, seed=2)

    assert positive.excludes_zero is True
    assert mixed.excludes_zero is False


# ══════════════════════════════════════════════════════════════
# compound_total_return
# ══════════════════════════════════════════════════════════════


def test_compound_total_return_is_hand_checkable():
    """+10% 兩趟 = 1.1 × 1.1 − 1 = +21%，不是 +20%"""
    assert compound_total_return(np.array([0.10, 0.10])) == pytest.approx(0.21)


def test_compound_total_return_handles_a_loss():
    """+50% 然後 −50% 回到 0.75，不是 1.0"""
    assert compound_total_return(np.array([0.50, -0.50])) == pytest.approx(-0.25)


def test_compound_total_return_of_an_empty_series_is_zero():
    assert compound_total_return(np.array([])) == pytest.approx(0.0)


# ══════════════════════════════════════════════════════════════
# paired_difference
# ══════════════════════════════════════════════════════════════


def test_paired_difference_uses_the_elementwise_gap():
    """配對必須逐期相減後才重抽，不是各自重抽再相減"""
    a = np.array([0.10, 0.20, 0.30, 0.40])
    b = np.array([0.05, 0.15, 0.25, 0.35])

    result = paired_difference(a, b, block_length=1, n_draws=500, seed=1)

    # 每一期差異都恰好是 0.05，所以重抽分布退化為單點
    assert result.point == pytest.approx(0.05)
    assert result.lower == pytest.approx(0.05)
    assert result.upper == pytest.approx(0.05)
    assert result.width == pytest.approx(0.0)


def test_paired_difference_rejects_mismatched_lengths():
    with pytest.raises(BootstrapError, match="長度"):
        paired_difference(np.array([0.1, 0.2]), np.array([0.1]),
                          block_length=1, n_draws=10, seed=1)


# ══════════════════════════════════════════════════════════════
# 邊界條件（輸入驗證在系統邊界）
# ══════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "block_length, message",
    [(0, "區塊長度"), (-1, "區塊長度"), (9, "區塊長度")],
)
def test_invalid_block_length_fails_fast(block_length, message):
    with pytest.raises(BootstrapError, match=message):
        block_bootstrap(np.arange(8, dtype=float), np.mean,
                        block_length=block_length, n_draws=10, seed=1)


def test_zero_draws_fails_fast():
    with pytest.raises(BootstrapError, match="重抽次數"):
        block_bootstrap(np.arange(8, dtype=float), np.mean,
                        block_length=2, n_draws=0, seed=1)


@pytest.mark.parametrize("level", [0.0, 1.0, -0.1, 1.5])
def test_level_outside_the_open_unit_interval_fails_fast(level):
    with pytest.raises(BootstrapError, match="信賴水準"):
        block_bootstrap(np.arange(8, dtype=float), np.mean,
                        block_length=2, n_draws=10, seed=1, level=level)


def test_empty_series_fails_fast():
    with pytest.raises(BootstrapError, match="空序列"):
        block_bootstrap(np.array([]), np.mean, block_length=1, n_draws=10, seed=1)


def test_result_is_immutable():
    result = block_bootstrap(np.arange(8, dtype=float), np.mean,
                             block_length=2, n_draws=10, seed=1)

    assert isinstance(result, BootstrapResult)
    with pytest.raises(Exception):
        result.point = 0.0  # type: ignore[misc]
