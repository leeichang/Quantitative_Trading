"""
Moving-block bootstrap 信賴區間

## 為什麼不用 t 檢定就好

主線至今每一個「量不出差別」都來自 n=36 的配對 t 檢定。實測那些序列：

```
配對差異序列        lag1 自相關    偏態      峰度
模型 − 打亂           −0.023     +1.20     5.81
模型 − 手工           +0.080     −0.40     4.85
手工 − 打亂           −0.074     +1.45     4.92
```

**自相關幾乎是零，但峰度 4.9~5.8（常態是 3）。** t 檢定假設的是常態，
而 n=36 的中央極限定理對這種尾部收斂得很慢——區間的**尾端**會失準，
而尾端正是判定顯著與否的地方。

`03_待辦與改進方向.md` 第 3 項把原因寫成「有效樣本太少」。那只說對
一半：配對差異的有效樣本就是 36，沒有被相依性折損。**問題在尾部形狀。**

水準序列（非配對）則相反：

```
等權全池        lag1 −0.281
隨機 10 檔      lag1 −0.277
標籤打亂        lag1 −0.278
```

**負自相關。** 對累積報酬來說負自相關會降低長期變異，所以逐點 IID
重抽會把區間估得**太寬**。兩個方向都需要區塊長度可調，所以
`block_length` 是必填而不是有預設值——它是一個要說明理由的選擇。

## 為什麼不引用外部套件

`validation/external/__init__.py` 記過不引入 `timeseries.bootstrap_sharpe`
的理由：「逐點 IID 重抽，對自相關報酬會低估區間寬度」。本模組是那個
決定的正面實作，保留同樣的判斷。

## 區塊長度怎麼選

沒有普遍正確的值。實務上：

```
配對差異（自相關 ≈ 0）    block_length = 1     等於 IID 重抽，這裡是對的
水準序列（有自相關）       block_length >= 2    並報告對長度的敏感度
```

**單一長度的結果不構成結論。** 掃幾個長度、把寬度一起報出來，
讓讀者看到結論對這個選擇有多敏感——這與 2026-09-18 撤回
「超過隨機 95% 分位」的理由是同一條：由任意選擇決定的結論不是結論。

## 這個模組不做什麼

不回答「顯著嗎」。它回答「區間有多寬、含不含零」。
`excludes_zero` 是描述而不是判定——多重測試校正仍然要另外做
（`stats.deflated_sharpe_ratio`、`probability_of_backtest_overfitting`）。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

Statistic = Callable[[np.ndarray], float]
"""吃一維重抽樣本、回一個純量。`np.mean`、`compound_total_return` 都符合"""

DEFAULT_LEVEL = 0.95
DEFAULT_DRAWS = 2000
"""2000 次對 95% 區間足夠：分位數的蒙地卡羅誤差已小於序列本身的雜訊"""


class BootstrapError(ValueError):
    """重抽的輸入不合法。"""


@dataclass(frozen=True)
class BootstrapResult:
    """一次 block bootstrap 的完整結果（不可變）。"""

    point: float
    """原始序列上的統計量，不是重抽分布的平均——後者有偏"""

    lower: float
    upper: float
    level: float
    n_draws: int
    block_length: int
    draws: tuple[float, ...]
    """完整重抽分布。保留它才能事後換信賴水準或畫分布"""

    @property
    def width(self) -> float:
        return self.upper - self.lower

    @property
    def excludes_zero(self) -> bool:
        """區間是否整段在零的同一側。**這是描述，不是顯著性判定**"""
        return self.lower > 0.0 or self.upper < 0.0

    def describe(self) -> str:
        return (
            f"{self.point:+.4%}"
            f"｜{self.level:.0%} 區間 [{self.lower:+.4%}, {self.upper:+.4%}]"
            f"｜寬 {self.width:.4%}"
            f"｜區塊 {self.block_length}"
            f"｜{'不含零' if self.excludes_zero else '含零'}"
        )


def moving_block_indices(
    n: int,
    *,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    抽出一組長度 n 的環狀移動區塊索引。

    環狀（circular）而非截斷：每個原始位置被抽中的機率相同。截斷版本
    會讓序列尾端的樣本被低估，在 n=36 這種短序列上不是小事。

    Args:
        n: 原始序列長度
        block_length: 每個區塊的長度，1 等於 IID 重抽
        rng: 由呼叫端提供，讓決定性由上層控制

    Returns:
        長度恰好為 n 的索引陣列，值域 [0, n)
    """
    n_blocks = -(-n // block_length)  # 向上取整
    starts = rng.integers(0, n, size=n_blocks)
    offsets = np.arange(block_length)
    # 每個起點展開成一段連續區塊，再環狀取模
    indices = (starts[:, None] + offsets[None, :]) % n
    return indices.reshape(-1)[:n]


def _validate(
    series: np.ndarray,
    *,
    block_length: int,
    n_draws: int,
    level: float,
) -> None:
    """在系統邊界一次驗完，錯誤訊息要指出是哪一個輸入不合法。"""
    if series.size == 0:
        raise BootstrapError("空序列無法重抽")
    if block_length < 1 or block_length > series.size:
        raise BootstrapError(
            f"區塊長度必須在 1 與序列長度 {series.size} 之間，得到 {block_length}"
        )
    if n_draws < 1:
        raise BootstrapError(f"重抽次數必須至少為 1，得到 {n_draws}")
    if not 0.0 < level < 1.0:
        raise BootstrapError(f"信賴水準必須在 (0, 1) 之間，得到 {level}")


def block_bootstrap(
    series: Sequence[float] | np.ndarray,
    statistic: Statistic,
    *,
    block_length: int,
    n_draws: int = DEFAULT_DRAWS,
    seed: int,
    level: float = DEFAULT_LEVEL,
) -> BootstrapResult:
    """
    對 `series` 的 `statistic` 做 moving-block bootstrap 百分位區間。

    Args:
        series: 一維序列（每趟報酬、每日報酬皆可）
        statistic: 吃重抽樣本回純量。**不預設 Sharpe**——由呼叫端決定，
            這樣本模組不需要知道無風險利率或年化慣例
        block_length: 見模組 docstring。**必填，因為它是要說明理由的選擇**
        n_draws: 重抽次數
        seed: 必填。沒有預設值，因為「用了哪個種子」要寫進報告
        level: 信賴水準

    Returns:
        `BootstrapResult`；`point` 取自原始序列而非重抽平均

    Raises:
        BootstrapError: 任一輸入不合法
    """
    values = np.asarray(series, dtype=float)
    _validate(values, block_length=block_length, n_draws=n_draws, level=level)

    rng = np.random.default_rng(seed)
    draws = np.empty(n_draws, dtype=float)
    for draw in range(n_draws):
        idx = moving_block_indices(values.size, block_length=block_length, rng=rng)
        draws[draw] = float(statistic(values[idx]))

    tail = (1.0 - level) / 2.0
    lower, upper = np.quantile(draws, [tail, 1.0 - tail])
    return BootstrapResult(
        point=float(statistic(values)),
        lower=float(lower),
        upper=float(upper),
        level=level,
        n_draws=n_draws,
        block_length=block_length,
        draws=tuple(draws.tolist()),
    )


def usable_block_lengths(
    n: int, candidates: Sequence[int]
) -> tuple[int, ...]:
    """
    篩掉超過序列長度的區塊長度。

    短序列（例如前推帳本頭幾期）會讓掃描清單裡的大長度非法。靜默跳過
    會讓報告少一欄而沒人注意，直接拋錯則讓短序列完全無法診斷——
    **所以篩選要由呼叫端顯式做，並把篩掉的記進報告。**

    Returns:
        升冪、去重、且全部 <= n 的長度。若 `n < 1` 回空元組。
    """
    if n < 1:
        return ()
    return tuple(sorted({c for c in candidates if 1 <= c <= n}))


def compound_total_return(per_trip: Sequence[float] | np.ndarray) -> float:
    """
    每趟報酬複利成總報酬。

    +10% 兩趟是 1.1 × 1.1 − 1 = +21%，不是 +20%——加總會低估，
    而先前的報告都是複利記的，所以這裡也必須複利。

    空序列回 0.0（沒有交易就沒有報酬），不是 NaN。
    """
    values = np.asarray(per_trip, dtype=float)
    if values.size == 0:
        return 0.0
    return float(np.prod(1.0 + values) - 1.0)


def paired_difference(
    treatment: Sequence[float] | np.ndarray,
    control: Sequence[float] | np.ndarray,
    *,
    block_length: int,
    n_draws: int = DEFAULT_DRAWS,
    seed: int,
    level: float = DEFAULT_LEVEL,
) -> BootstrapResult:
    """
    兩個策略的**逐期差異**的平均，及其 bootstrap 區間。

    先逐期相減再重抽，不是各自重抽再相減——兩者共用同一段市場，
    配對才能把市場共同變動消掉。這也是配對 t 檢定的原意，本函式
    只是把常態假設換成重抽。

    Raises:
        BootstrapError: 兩序列長度不同，或其他輸入不合法
    """
    a = np.asarray(treatment, dtype=float)
    b = np.asarray(control, dtype=float)
    if a.shape != b.shape:
        raise BootstrapError(f"配對序列長度必須相同，得到 {a.shape} 與 {b.shape}")

    return block_bootstrap(
        a - b,
        np.mean,
        block_length=block_length,
        n_draws=n_draws,
        seed=seed,
        level=level,
    )
