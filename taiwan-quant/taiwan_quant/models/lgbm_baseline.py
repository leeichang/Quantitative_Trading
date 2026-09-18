"""
LightGBM baseline：CLAUDE.md 明文要求的第一個模型

## 為什麼現在才做

CLAUDE.md「明確不做」裡寫著：

```
❌ 深度學習（先有 LightGBM baseline）
```

而到目前為止所有工作都在改**機制**（校準器、門檻、槽位、成本、投組約束）
與**成本**，訊號本身一直是手工分數（`動能突破` / `籌碼跟隨` / `均值回歸`）。

累積的否定清單顯示機制與成本已經挖完：

```
週頻抓漲停        否定  毛報酬在扣成本之前就是負的
三族合成訊號      否定  動能與均值回歸 −0.505 負相關
新聞               否定  落地當天定價完
ETF 輪動          否定  輸買進持有，PBO 0.625~0.750
本金               不是瓶頸  可省 2.6 pp vs 需要 22 pp
```

**剩下唯一沒動過的是訊號本身。**

## 這是 baseline，不是調參

`DEFAULT_PARAMS` 是**固定的一組**，不掃超參數。理由：

掃參數會讓 `n_trials` 爆掉，而先前已經看過那個後果——掃 126 組時
DSR 0.0042 不通過，掃 16 組時 PBO 0.625~0.750。**一個未調參的
baseline 的 n_trials 是 1，那是它最有價值的地方。**

調參要等 baseline 證明「模型比手工分數好」之後才有意義。

## 目標是橫斷面排名，不是絕對報酬

```
絕對報酬    同一天所有名字共享大盤漂移 → 模型會去學「大盤會漲」
橫斷面排名   移除共同成分 → 學的是「今天哪一檔相對強」
```

而我們實際的用法就是橫斷面選前 N 檔，所以排名才是對的目標。

## 標籤淨化（禁令 1、5）

40 日標籤在決策日之後 41 個交易日才揭曉。訓練時若包含那些尚未揭曉的列
就是 look-ahead，而且**不會拋錯**——它只會讓模型看起來很準。

```
要在位置 p 做預測，訓練列 d 必須滿足   d + 1 + horizon <= p
```

見 `purged_training_positions`，有測試釘住邊界。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from taiwan_quant.features.chips import build_chips
from taiwan_quant.features.technical import build_technical


class BaselineError(RuntimeError):
    """模型輸入不合法"""


DEFAULT_PARAMS: dict[str, object] = {
    "objective": "regression",
    "metric": "l2",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "min_data_in_leaf": 100,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "feature_fraction": 0.8,
    "lambda_l2": 1.0,
    "verbose": -1,
    "num_threads": -1,
    "seed": 20260917,
    "deterministic": True,
}
"""
固定的一組超參數，**不掃描**。

用 LightGBM 的**原生參數名**，因為 `lgb.LGBMRegressor` 那個 sklearn
包裝需要 scikit-learn，而本專案沒有裝它。為了一層包裝加一個大依賴
不划算，原生 `lgb.train` 才是 LightGBM 本體。

`min_data_in_leaf=100` 刻意偏大：實測優勢集中在每期 1.7~4.3 檔名字上，
葉子太小會直接記住那幾檔。`seed` + `deterministic` 讓結果可重現（禁令 7）。

⚠️ 改動任何一項就等於多一次試驗，`N_TRIALS` 必須跟著加。
"""

NUM_BOOST_ROUND = 200
"""提升輪數。與 `DEFAULT_PARAMS` 一樣屬於「固定不掃」的一部分"""

N_TRIALS = 1
"""未調參，所以 DSR 的試驗數是 1——這是 baseline 最有價值的性質"""


@dataclass(frozen=True)
class Dataset:
    """特徵矩陣與標籤，共用同一個 MultiIndex"""

    features: pd.DataFrame
    """MultiIndex (date, stock_id) × 特徵欄"""

    target: pd.Series
    """橫斷面排名（0~1），同一個 index。未揭曉處為 NaN"""

    forward_return: pd.Series
    """原始 40 日毛報酬，同一個 index。評估用，不進訓練"""


def build_features(by_stock: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    組出 MultiIndex (date, stock_id) 的特徵矩陣。

    Args:
        by_stock: 每檔的日 K + 籌碼（`build_dataset` 的輸出）

    Returns:
        特徵矩陣。缺籌碼的標的只有技術面特徵，籌碼欄為 NaN

    Raises:
        BaselineError: 沒有任何標的算得出特徵

    LightGBM 原生支援 NaN，所以**不填補缺值**——填 0 會讓「沒有籌碼資料」
    與「籌碼淨額為零」變成同一件事。
    """
    frames = []
    for stock_id, bars in by_stock.items():
        technical = build_technical(bars)
        try:
            chips = build_chips(bars)
        except (ValueError, KeyError):
            chips = pd.DataFrame(index=bars.index)
        merged = pd.concat([technical, chips], axis=1)
        merged["stock_id"] = stock_id
        frames.append(merged.set_index("stock_id", append=True))

    if not frames:
        raise BaselineError("沒有任何標的算得出特徵")
    combined = pd.concat(frames).sort_index()
    combined.index.names = ["date", "stock_id"]
    return combined


def cross_sectional_rank(values: pd.Series) -> pd.Series:
    """
    每個日期之內的分位排名（0~1）。

    Args:
        values: MultiIndex (date, stock_id) 的數值

    Returns:
        同 index 的分位。單一標的的日期回 0.5（無從排序）

    移除大盤共同成分——否則模型會去學「大盤會漲」，那對橫斷面選股
    沒有用。
    """
    return values.groupby(level="date").rank(pct=True)


def build_dataset_for_model(
    by_stock: dict[str, pd.DataFrame],
    calendar: list[pd.Timestamp],
    horizon: int,
) -> Dataset:
    """
    特徵 + 標籤。

    Args:
        by_stock: 每檔的日 K + 籌碼
        calendar: 交易日曆
        horizon: 持有交易日數

    Returns:
        Dataset

    標籤是 **T+1 開盤買、T+horizon 收盤賣**的報酬。

    ## ⚠️ 第一版多算一天

    原本寫 `closes.shift(-1 - horizon)`，也就是 T+1+horizon 收盤出場——
    **那是 horizon + 1 天的持有期**。

    正確的推導：T+1 開盤進、X 收盤出，持有天數 = X − T。要 horizon 天
    就是 X = T + horizon。

    專案自己的算術也證實：CLAUDE.md 的持有期表寫「20 日 → 每年 12.6 趟」，
    252 / 20 = 12.6；若是 21 天會是 12.0。

    而 `data/integrity.holding_dates` 是這件事的單一來源：
    進場 `decision_index + 1`、出場 `decision_index + holding_days`。
    """
    features = build_features(by_stock)
    opens = pd.DataFrame(
        {sid: bars["open"].astype(float) for sid, bars in by_stock.items()}
    ).reindex(calendar)
    closes = pd.DataFrame(
        {sid: bars["close"].astype(float) for sid, bars in by_stock.items()}
    ).reindex(calendar)
    forward = (closes.shift(-horizon) / opens.shift(-1) - 1).stack(
        future_stack=True
    )
    forward.index.names = ["date", "stock_id"]

    aligned = forward.reindex(features.index)
    return Dataset(
        features=features,
        target=cross_sectional_rank(aligned),
        forward_return=aligned,
    )


def purged_training_positions(
    predict_position: int, horizon: int, warmup: int = 0
) -> tuple[int, int]:
    """
    可安全用於訓練的決策位置範圍（左閉右開）。

    Args:
        predict_position: 要預測的決策日位置
        horizon: 持有交易日數
        warmup: 特徵暖機需要的交易日數

    Returns:
        `(start, end)`，訓練列取 `start <= d < end`

    Raises:
        BaselineError: `horizon` 非正

    ## 邊界怎麼算

    位置 d 的標籤用 `opens[d+1]` 與 `closes[d+1+horizon]`，所以它在
    位置 `d + 1 + horizon` 才揭曉。要在位置 p 預測，訓練列必須滿足：

    ```
    d + 1 + horizon <= p        →        d <= p − horizon − 1
    ```

    所以 `end = p - horizon`（右開）。

    ⚠️ **少減 1 就洩漏一期**，而它不會拋錯——只會讓模型看起來變準。
    `test_purge_boundary_is_exact` 釘住這個。
    """
    if horizon < 1:
        raise BaselineError(f"horizon 至少為 1，得到 {horizon}")
    end = predict_position - horizon
    return warmup, max(warmup, end)


def train_and_predict(
    dataset: Dataset,
    calendar: list[pd.Timestamp],
    train_range: tuple[int, int],
    predict_position: int,
    candidates: tuple[str, ...],
    params: dict[str, object] | None = None,
    shuffle_seed: int | None = None,
) -> tuple[pd.Series, dict[str, float]]:
    """
    用淨化後的訓練列訓練，對 `predict_position` 當日的候選預測排名。

    Args:
        dataset: `build_dataset_for_model` 的輸出
        calendar: 交易日曆
        train_range: `purged_training_positions` 的輸出
        predict_position: 要預測的位置
        candidates: 該日可選的標的（時點標的池）
        params: 超參數；預設 `DEFAULT_PARAMS`
        shuffle_seed: 給定時**在每個日期之內打亂標籤**，用於洩漏偵測

    Returns:
        `(候選的預測排名, 特徵重要度)`。訓練樣本不足時回 `(空 Series, {})`

    Raises:
        BaselineError: `train_range` 不合法

    ## `shuffle_seed` 是洩漏偵測器

    在每個日期之內打亂標籤，特徵與標籤的真實關係就被破壞，但
    **資料的形狀、缺值模式、橫斷面結構全部不變**。

    打亂之後模型若還能贏過隨機，那個優勢就不是來自特徵預測標籤，
    而是來自切分或索引的洩漏。這是標準的置換檢定，而且它能抓到
    單元測試抓不到的東西——單元測試驗的是邊界算術，這個驗的是
    整條管線。
    """
    import lightgbm as lgb

    start, end = train_range
    if end <= start:
        return pd.Series(dtype=float), {}

    train_dates = calendar[start:end]
    rows = dataset.features.index.get_level_values("date").isin(train_dates)
    features = dataset.features[rows]
    target = dataset.target[rows]
    usable = target.notna()
    if usable.sum() < 500:
        return pd.Series(dtype=float), {}

    if shuffle_seed is not None:
        rng = np.random.default_rng(shuffle_seed)
        shuffled = target[usable].copy()
        # 在每個日期之內打亂，保留橫斷面結構（每天的排名仍是 0~1 的置換）
        for _, positions in shuffled.groupby(level="date").groups.items():
            # `permutation` 回傳新陣列；`shuffle` 會就地改寫，而 pandas
            # 切片的 `to_numpy()` 可能是唯讀視圖
            shuffled.loc[positions] = rng.permutation(
                shuffled.loc[positions].to_numpy()
            )
        target = target.copy()
        target[usable] = shuffled

    columns = list(features.columns)
    train_set = lgb.Dataset(
        features.loc[usable, columns].to_numpy(dtype="float64"),
        label=target[usable].to_numpy(dtype="float64"),
        feature_name=columns,
        free_raw_data=False,
    )
    booster = lgb.train(
        dict(params or DEFAULT_PARAMS), train_set,
        num_boost_round=NUM_BOOST_ROUND,
    )

    importance = dict(zip(
        columns,
        [float(x) for x in booster.feature_importance(importance_type="gain")],
        strict=True,
    ))

    day = calendar[predict_position]
    try:
        today = dataset.features.xs(day, level="date")
    except KeyError:
        return pd.Series(dtype=float), importance
    today = today.loc[today.index.intersection(list(candidates))]
    if today.empty:
        return pd.Series(dtype=float), importance

    predicted = booster.predict(today[columns].to_numpy(dtype="float64"))
    return pd.Series(np.asarray(predicted), index=today.index), importance
