"""
Walk-Forward 切分與標籤隔離（embargo）

## 為什麼要有這個模組

原本的切分寫在腳本裡，而且**漏了隔離**：

```python
train_dates = dates[:cursor]          # ← bug
test_dates  = dates[cursor : cursor + test_span]
```

持有 60 日、每 5 日決策時，訓練集最後 12 期的實際報酬要用到**測試期
的價格**才算得出來。校準器等於看過測試期的走勢。

污染比例：第一個 fold 8.0%（12/150），最後一個 2.4%（12/498）。

## 這與禁令 1 是同一條原則

CLAUDE.md 禁令 1：T 日決策只能用 T 日收盤前可得的資料。

延伸到訓練：**訓練時可用的標籤，必須在訓練期結束前就已經揭曉。**
標籤要等未來 60 天才知道答案，那 60 天就不能是測試期。

## 為什麼抽成模組而不是在腳本裡修

1. 切分邏輯要有測試——它錯的時候不會拋錯，只會讓結果變好看
2. `validate_oos.py` 與 `validate_oos_trailing.py` 都需要，不該有兩份
3. 之後換模型（Ridge / LightGBM）時，同一套切分才能直接沿用

## 隔離期數要無條件進位

持有 22 日、每 5 日決策 → 22/5 = 4.4。第 5 期的標籤仍有一部分落在
測試段內，必須一起隔離。捨去會留下部分污染。
"""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from collections.abc import Iterator

import pandas as pd


class WalkForwardError(RuntimeError):
    """切分參數不合法"""


def embargo_periods(horizon: int, stride: int) -> int:
    """
    需要隔離幾個決策期。

    Args:
        horizon: 持有交易日數（標籤要等這麼久才揭曉）
        stride: 決策間隔交易日數

    Returns:
        訓練集尾端要剔除的期數

    Raises:
        WalkForwardError: 任一參數非正

    **無條件進位。** 持有 22 日、每 5 日決策時 22/5 = 4.4，
    第 5 期的標籤仍有一部分落在測試段內。
    """
    if horizon < 1 or stride < 1:
        raise WalkForwardError(
            f"horizon 與 stride 都必須為正，得到 {horizon}, {stride}"
        )
    return math.ceil(horizon / stride)


def walk_forward_folds(
    dates: list[pd.Timestamp],
    first_train: int,
    test_span: int,
    horizon: int,
    stride: int,
    oos_start: pd.Timestamp | None = None,
    dev_end: pd.Timestamp | None = None,
    freeze_model: bool = False,
) -> Iterator[tuple[list[pd.Timestamp], list[pd.Timestamp]]]:
    """
    產生滾動擴張窗口的 (訓練日, 測試日)，**訓練集已做標籤隔離**。

    Args:
        dates: 全部決策日（升冪）
        first_train: 第一個 fold 的訓練期數
        test_span: 每個 fold 的測試期數
        horizon: 持有交易日數
        stride: 決策間隔交易日數
        oos_start: 指定時，以最接近且不早於此日的決策日開始測試
        dev_end: 初始開發集最後日期；後續 fold 預設仍依 walk-forward 擴張
        freeze_model: True 時永久固定在第一個 OOS fold 的訓練集；用來量測
            模型不更新時的衰退，不是一般 walk-forward

    Yields:
        (訓練日清單, 測試日清單)

    Raises:
        WalkForwardError: 日期未升冪排序，或參數非正

    隔離之後訓練集為空的 fold 會被**跳過**——用空集合擬合校準器只會
    拋錯或產生無意義的結果。
    """
    if first_train < 1 or test_span < 1:
        raise WalkForwardError(
            f"first_train 與 test_span 都必須為正，得到 {first_train}, {test_span}"
        )
    if any(b < a for a, b in zip(dates, dates[1:], strict=False)):
        raise WalkForwardError("決策日必須依日期升冪排序")
    if dev_end is not None and oos_start is None:
        raise WalkForwardError("--dev-end 只能與 --oos-start 一起使用")
    if freeze_model and oos_start is None:
        raise WalkForwardError("--freeze-model 只能與 --oos-start 一起使用")

    gap = embargo_periods(horizon, stride)
    initial_cursor: int | None = None
    initial_train_end: int | None = None
    if oos_start is None:
        cursor = first_train
    else:
        requested = pd.Timestamp(oos_start)
        cursor = bisect_left(dates, requested)
        if cursor < first_train:
            raise WalkForwardError(
                f"OOS 起點 {requested.date()} 早於最低訓練需求 "
                f"{first_train} 個決策期"
            )
        if cursor >= len(dates):
            raise WalkForwardError(f"OOS 起點 {requested.date()} 晚於所有決策日")

        initial_cursor = cursor
        # 第一個 OOS fold 必須剔除尾端 gap 期，避免尚未揭曉的標籤跨入
        # 測試段。後續 fold 預設照 walk-forward 擴張；先前 OOS 在標籤
        # 揭曉後成為訓練資料，這不是洩漏。
        label_cutoff = cursor - gap
        if dev_end is None:
            initial_train_end = label_cutoff
        else:
            development_end = pd.Timestamp(dev_end)
            if development_end >= dates[cursor]:
                raise WalkForwardError(
                    f"開發集結束日 {development_end.date()} 必須早於 "
                    f"OOS 起點 {dates[cursor].date()}"
                )
            initial_train_end = min(
                bisect_right(dates, development_end), label_cutoff
            )

    while cursor + test_span <= len(dates):
        if initial_train_end is None or initial_cursor is None:
            train_end = cursor - gap
        elif freeze_model:
            train_end = initial_train_end
        else:
            # 保留初始 dev_end 與 OOS 間的距離，同時每個 fold 納入已揭曉
            # 的前一期測試資料，維持 expanding walk-forward 的實驗性質。
            train_end = min(
                cursor - gap,
                initial_train_end + (cursor - initial_cursor),
            )
        if train_end > 0:
            yield dates[:train_end], dates[cursor : cursor + test_span]
        cursor += test_span
