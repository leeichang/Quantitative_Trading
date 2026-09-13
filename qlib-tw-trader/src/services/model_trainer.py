"""
模型訓練服務 - DoubleEnsemble + RD-Agent IC 去重複
"""

import hashlib
import json
import logging
import os
import pickle
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

from src.repositories.factor import FactorRepository
from src.repositories.models import Factor, TrainingRun
from src.repositories.training import TrainingRepository
from src.services.factor_selection import RobustFactorSelector
from src.shared.constants import LABEL_EXPR, LABEL_EXTEND_DAYS, TRAIN_DAYS, VALID_DAYS

TZ_TAIPEI = ZoneInfo("Asia/Taipei")

# 模型檔案目錄
MODELS_DIR = Path("data/models")


@dataclass
class FactorEvalResult:
    """因子評估結果"""

    factor_id: int
    factor_name: str
    ic_value: float
    selected: bool


@dataclass
class TrainingResult:
    """訓練結果"""

    run_id: int
    model_name: str
    model_ic: float
    icir: float | None
    selected_factor_ids: list[int]
    all_results: list[FactorEvalResult]


def _detect_lgb_device() -> str:
    """
    偵測本機 LightGBM 是否支援 GPU。

    pip 安裝的 lightgbm wheel 預設是 CPU-only build，硬寫 device="gpu"
    會在訓練途中拋出：

        GPU Tree Learner was not enabled in this build.
        Please recompile with CMake option -DUSE_GPU=1

    用環境變數 LGB_DEVICE 可強制指定（gpu / cpu），未指定時實測一次。
    """
    forced = os.environ.get("LGB_DEVICE", "").strip().lower()
    if forced in ("gpu", "cpu"):
        return forced

    try:
        import lightgbm as lgb
        import numpy as np

        probe = lgb.Dataset(np.zeros((10, 2)), label=np.zeros(10))
        lgb.train(
            {"objective": "regression", "device": "gpu", "verbosity": -1},
            probe,
            num_boost_round=1,
        )
        return "gpu"
    except Exception:
        return "cpu"


LGB_DEVICE = _detect_lgb_device()
"""本機可用的 LightGBM device，由 _detect_lgb_device() 實測決定"""


DEFAULT_LGB_PARAMS = {
    "objective": "regression",
    "metric": "mse",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "max_depth": 5,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 5.0,
    "lambda_l2": 5.0,
    "min_data_in_leaf": 30,
    "verbosity": -1,
    "seed": 42,
    "feature_pre_filter": False,
    "device": LGB_DEVICE,
    "gpu_use_dp": False,
}


class ModelTrainer:
    """模型訓練器 - DoubleEnsemble + Optuna 超參數優化"""

    # Optuna 調參設定
    OPTUNA_N_TRIALS = 50  # 搜索次數
    OPTUNA_TIMEOUT = 300  # 超時秒數（5分鐘）

    def __init__(self, qlib_data_dir: Path | str):
        self.data_dir = Path(qlib_data_dir)
        self._qlib_initialized = False
        self._last_ic_std: float | None = None  # 用於 ICIR 計算
        self._data_cache: dict[str, pd.DataFrame] = {}  # 資料快取
        self._optimized_params: dict | None = None  # Optuna 優化後的參數
        self._auto_optuna: bool = False  # 是否需要自動運行 Optuna

    def _init_qlib(self, force: bool = False) -> None:
        """
        初始化 qlib

        Args:
            force: 強制重新初始化（用於導出新資料後）
        """
        if self._qlib_initialized and not force:
            return

        try:
            import qlib
            from qlib.config import REG_CN

            # 強制重新初始化：清除快取並重新載入
            qlib.init(
                provider_uri=str(self.data_dir),
                region=REG_CN,
            )
            self._qlib_initialized = True
            self._data_cache.clear()  # 清除資料快取
        except ImportError:
            raise RuntimeError("qlib is not installed. Please run: pip install pyqlib")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize qlib: {e}")

    def _get_instruments(self) -> list[str]:
        """從 qlib instruments 目錄讀取股票清單"""
        instruments_file = self.data_dir / "instruments" / "all.txt"

        if instruments_file.exists():
            with open(instruments_file) as f:
                return [line.strip().split()[0] for line in f if line.strip()]

        # 備選：從 features 目錄取得
        features_dir = self.data_dir / "features"
        if features_dir.exists():
            return [d.name for d in features_dir.iterdir() if d.is_dir()]

        return []

    def get_data_date_range(self) -> tuple[date | None, date | None]:
        """取得 qlib 資料的日期範圍"""
        instruments_file = self.data_dir / "instruments" / "all.txt"

        if not instruments_file.exists():
            return None, None

        min_start = None
        max_end = None

        with open(instruments_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    try:
                        start = date.fromisoformat(parts[1])
                        end = date.fromisoformat(parts[2])
                        if min_start is None or start < min_start:
                            min_start = start
                        if max_end is None or end > max_end:
                            max_end = end
                    except ValueError:
                        continue

        return min_start, max_end

    def _load_data(
        self,
        factors: list[Factor],
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """
        載入因子資料和標籤

        注意：Label 定義為 Ref($close, -3) / Ref($close, -1) - 1
        即 T 日的 label = close[T+3] / close[T+1] - 1（2-day return）
        因此需要多取 LABEL_EXTEND_DAYS 天的資料以確保 end_date 日的 label 完整

        Returns:
            DataFrame with columns: [factor1, factor2, ..., label]
            Index: MultiIndex (datetime, instrument)
        """
        # 強制重新初始化 qlib，確保使用最新導出的資料
        self._init_qlib(force=True)
        from qlib.data import D
        from datetime import timedelta

        instruments = self._get_instruments()
        if not instruments:
            raise ValueError("No instruments found in qlib data directory")

        # 構建因子表達式
        fields = [f.expression for f in factors]
        names = [f.name for f in factors]

        # 標籤：T+1→T+3 的 2-day return
        label_expr = LABEL_EXPR
        all_fields = fields + [label_expr]
        all_names = names + ["label"]

        # 延伸 end_date 以確保 label 完整（需要 T+3 的價格）
        extended_end = end_date + timedelta(days=LABEL_EXTEND_DAYS)

        # 讀取資料
        df = D.features(
            instruments=instruments,
            fields=all_fields,
            start_time=start_date.strftime("%Y-%m-%d"),
            end_time=extended_end.strftime("%Y-%m-%d"),
        )

        if df.empty:
            return df

        # 重命名欄位
        df.columns = all_names

        # 只返回 end_date 前的資料（但 label 已經正確計算）
        # 這確保特徵只使用 end_date 前的資訊
        dates = df.index.get_level_values("datetime")
        if hasattr(dates[0], "date"):
            mask = pd.Series([d.date() <= end_date for d in dates], index=df.index)
        else:
            mask = dates.date <= end_date
        df = df[mask]

        return df

    def _prepare_train_valid_data(
        self,
        df: pd.DataFrame,
        train_start: date,
        train_end: date,
        valid_start: date,
        valid_end: date,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        準備訓練和驗證資料

        Returns:
            (X_train, X_valid, y_train, y_valid)
        """
        # 分離特徵和標籤
        feature_cols = [c for c in df.columns if c != "label"]
        X = df[feature_cols]
        y = df["label"]

        # Label 截面排名標準化（CSRankNorm）
        # 原因：
        # 1. 模型預測「排名」而非「收益率」更穩定（Learning to Rank 論文）
        # 2. Live IC 使用 Spearman（排名相關），與排名預測一致
        # 3. CSRankNorm 比 CSZScoreNorm 更穩健（對離群值不敏感）
        # 參考：qlib GRU/LSTM/AdaRNN 配置都使用 CSRankNorm
        y = self._rank_by_date(y)

        # 按日期分割
        train_mask = (df.index.get_level_values("datetime").date >= train_start) & \
                     (df.index.get_level_values("datetime").date <= train_end)
        valid_mask = (df.index.get_level_values("datetime").date >= valid_start) & \
                     (df.index.get_level_values("datetime").date <= valid_end)

        X_train = X[train_mask].dropna()
        X_valid = X[valid_mask].dropna()
        y_train = y[train_mask].dropna()
        y_valid = y[valid_mask].dropna()

        # 對齊索引
        common_train = X_train.index.intersection(y_train.index)
        common_valid = X_valid.index.intersection(y_valid.index)

        return (
            X_train.loc[common_train],
            X_valid.loc[common_valid],
            y_train.loc[common_train],
            y_valid.loc[common_valid],
        )

    def _process_inf(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        處理無窮大值（仿 qlib ProcessInf）

        將 inf/-inf 替換為該欄位的均值
        """
        df = df.copy()
        for col in df.columns:
            mask = np.isinf(df[col])
            if mask.any():
                col_mean = df.loc[~mask, col].mean()
                df.loc[mask, col] = col_mean if not np.isnan(col_mean) else 0
        return df

    def _zscore_by_date(self, df: pd.DataFrame) -> pd.DataFrame:
        """每日截面標準化（仿 qlib CSZScoreNorm）"""
        return df.groupby(level="datetime", group_keys=False).apply(
            lambda x: (x - x.mean()) / (x.std() + 1e-8)
        )

    def _rank_by_date(self, series: pd.Series) -> pd.Series:
        """
        每日截面排名標準化（仿 qlib CSRankNorm）

        將每日截面的值轉換成排名百分位 [0, 1]
        這比 ZScore 更穩健，且與 Spearman IC 計算一致

        參考：
        - qlib CSRankNorm: https://qlib.readthedocs.io/en/stable/component/data.html
        - Learning to Rank 論文: https://arxiv.org/abs/2012.07149
        """
        def rank_pct(x: pd.Series) -> pd.Series:
            # 排名後轉換成百分位 [0, 1]
            # 使用 average 方法處理相同值
            return x.rank(pct=True, method="average")

        return series.groupby(level="datetime", group_keys=False).apply(rank_pct)

    # DoubleEnsemble 固定參數（論文 + Qlib benchmark 一致，不需搜索）
    DE_NUM_MODELS = 3  # CSI300 benchmark 用 3（小股票池）
    DE_DECAY = 0.5  # 所有 Qlib benchmark 一致
    DE_EPOCHS = 50  # 搭配 early stopping

    def _optimize_hyperparameters(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
        n_trials: int | None = None,
        timeout: int | None = None,
        on_progress: Callable[[float, str], None] | None = None,
    ) -> dict:
        """
        使用 Optuna 優化 LightGBM 子模型超參數

        DoubleEnsemble 特有參數（num_models, decay 等）根據論文固定，
        只搜索 LightGBM 底層參數。
        """
        import optuna

        from src.services.double_ensemble import DoubleEnsembleModel

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        n_trials = n_trials or self.OPTUNA_N_TRIALS
        timeout = timeout or self.OPTUNA_TIMEOUT

        X_train_processed = self._process_inf(X_train)
        X_valid_processed = self._process_inf(X_valid)
        X_train_norm = self._zscore_by_date(X_train_processed).fillna(0)
        X_valid_norm = self._zscore_by_date(X_valid_processed).fillna(0)

        n_samples = len(X_train)
        scale_factor = max(0.1, min(1.0, n_samples / 100000))
        lambda_max = max(1.0, 50.0 * scale_factor)

        best_ic = [0.0]
        trial_count = [0]

        def objective(trial: optuna.Trial) -> float:
            lgb_params = {
                "objective": "regression",
                "metric": "mse",
                "boosting_type": "gbdt",
                "verbosity": -1,
                "seed": 42,
                "feature_pre_filter": False,
                "device": LGB_DEVICE,
                "gpu_use_dp": False,
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 16, 64),
                "max_depth": trial.suggest_int("max_depth", 4, 8),
                "lambda_l1": trial.suggest_float("lambda_l1", 0.01, lambda_max, log=True),
                "lambda_l2": trial.suggest_float("lambda_l2", 0.01, lambda_max, log=True),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.7, 1.0),
                "bagging_freq": 5,
            }

            model = DoubleEnsembleModel(
                num_models=self.DE_NUM_MODELS,
                epochs=self.DE_EPOCHS,
                decay=self.DE_DECAY,
                early_stopping_rounds=10,
                **lgb_params,
            )
            model.fit(
                X_train_norm.values, y_train.values,
                X_valid_norm.values, y_valid.values,
            )

            predictions = model.predict(X_valid_norm.values)
            if np.unique(predictions).size <= 1:
                return 0.0

            pred_df = pd.DataFrame({
                "pred": predictions,
                "label": y_valid.values,
            }, index=y_valid.index)

            def calc_ic(g: pd.DataFrame) -> float:
                if len(g) < 10:
                    return np.nan
                if g["pred"].nunique() == 1 or g["label"].nunique() == 1:
                    return np.nan
                return g["pred"].corr(g["label"], method="spearman")

            daily_ic = pred_df.groupby(level="datetime").apply(calc_ic)
            mean_ic = daily_ic.mean()
            ic = float(mean_ic) if not np.isnan(mean_ic) else 0.0

            trial_count[0] += 1
            if ic > best_ic[0]:
                best_ic[0] = ic

            if on_progress:
                on_progress(
                    round(2.0 + (trial_count[0] / n_trials) * 8.0, 1),
                    f"Optuna trial {trial_count[0]}/{n_trials}: IC={ic:.4f} (best: {best_ic[0]:.4f})"
                )

            return ic

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

        best = study.best_params
        return {
            "lgb_params": {
                "objective": "regression",
                "metric": "mse",
                "boosting_type": "gbdt",
                "verbosity": -1,
                "seed": 42,
                "feature_pre_filter": False,
                "device": LGB_DEVICE,
                "gpu_use_dp": False,
                "bagging_freq": 5,
                **best,
            },
        }

    def _train_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
        params: dict | None = None,
    ) -> Any:
        """
        訓練 DoubleEnsemble 模型

        Args:
            X_train, y_train: 訓練資料
            X_valid, y_valid: 驗證資料
            params: 超參數字典，包含 num_models, decay, lgb_params

        Returns:
            訓練好的 DoubleEnsembleModel
        """
        from src.services.double_ensemble import DoubleEnsembleModel

        X_train = self._process_inf(X_train)
        X_valid = self._process_inf(X_valid)
        X_train_norm = self._zscore_by_date(X_train).fillna(0)
        X_valid_norm = self._zscore_by_date(X_valid).fillna(0)

        if params is None:
            params = self._optimized_params or {}

        lgb_params = params.get("lgb_params", DEFAULT_LGB_PARAMS)

        model = DoubleEnsembleModel(
            num_models=self.DE_NUM_MODELS,
            epochs=self.DE_EPOCHS,
            decay=self.DE_DECAY,
            early_stopping_rounds=10,
            **lgb_params,
        )
        model.fit(
            X_train_norm.values, y_train.values,
            X_valid_norm.values, y_valid.values,
        )

        return model

    def _calculate_prediction_ic(
        self,
        model: Any,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
    ) -> float:
        """
        計算模型預測的 IC

        Returns:
            平均 IC 值
        """
        # 處理無窮大 + 標準化 + 填補 NaN
        X_valid_processed = self._process_inf(X_valid)
        X_valid_norm = self._zscore_by_date(X_valid_processed)
        X_valid_norm = X_valid_norm.fillna(0)

        # 預測
        predictions = model.predict(X_valid_norm.values)

        # 構建 DataFrame
        pred_df = pd.DataFrame({
            "pred": predictions,
            "label": y_valid.values,
        }, index=y_valid.index)

        # 計算每日截面 IC（使用 Spearman，與 Live IC 保持一致）
        def calc_spearman_ic(group: pd.DataFrame) -> float:
            if len(group) < 10:
                return np.nan
            if group["pred"].nunique() == 1 or group["label"].nunique() == 1:
                return np.nan
            return group["pred"].corr(group["label"], method="spearman")

        daily_ic = pred_df.groupby(level="datetime").apply(calc_spearman_ic)

        # 保存 IC 標準差
        self._last_ic_std = float(daily_ic.std()) if len(daily_ic) > 1 else None

        mean_ic = daily_ic.mean()
        return float(mean_ic) if not np.isnan(mean_ic) else 0.0

    def _calculate_daily_ic(
        self,
        model: Any,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
    ) -> np.ndarray:
        """
        計算模型預測的每日 IC（用於統計檢驗）

        Returns:
            每日 IC 的 numpy 陣列
        """
        # 處理無窮大 + 標準化 + 填補 NaN
        X_valid_processed = self._process_inf(X_valid)
        X_valid_norm = self._zscore_by_date(X_valid_processed)
        X_valid_norm = X_valid_norm.fillna(0)

        # 預測
        predictions = model.predict(X_valid_norm.values)

        # 構建 DataFrame
        pred_df = pd.DataFrame({
            "pred": predictions,
            "label": y_valid.values,
        }, index=y_valid.index)

        # 計算每日截面 IC（使用 Spearman，與 Live IC 保持一致）
        def calc_spearman_ic(group: pd.DataFrame) -> float:
            if len(group) < 10:
                return np.nan
            if group["pred"].nunique() == 1 or group["label"].nunique() == 1:
                return np.nan
            return group["pred"].corr(group["label"], method="spearman")

        daily_ic = pred_df.groupby(level="datetime").apply(calc_spearman_ic)

        return daily_ic.dropna().values

    def train(
        self,
        session: Session,
        train_start: date,
        train_end: date,
        valid_start: date,
        valid_end: date,
        week_id: str | None = None,
        factor_pool_hash: str | None = None,
        on_progress: Callable[[float, str], None] | None = None,
    ) -> TrainingResult:
        """
        執行 LightGBM 訓練（週訓練架構）

        流程：
        1. 訓練模型 A（train_start ~ train_end）
        2. 計算驗證期 IC（報告指標）
        3. 增量更新：A + 驗證期資料 → A'
        4. 保存 A'（部署用）

        Args:
            session: 資料庫 Session
            train_start: 訓練開始日期
            train_end: 訓練結束日期
            valid_start: 驗證開始日期
            valid_end: 驗證結束日期
            week_id: 週 ID（如 "2026W05"）
            factor_pool_hash: 因子池 hash
            on_progress: 進度回調 (progress: 0-100, message: str)

        Returns:
            TrainingResult
        """
        from src.shared.constants import EMBARGO_DAYS

        factor_repo = FactorRepository(session)
        training_repo = TrainingRepository(session)

        # 取得啟用的因子
        enabled_factors = factor_repo.get_all(enabled=True)
        if not enabled_factors:
            raise ValueError("No enabled factors found")

        candidate_ids = [f.id for f in enabled_factors]

        # 生成臨時模型名稱
        temp_name = f"{week_id or valid_end.strftime('%Y%m')}-pending"

        # 創建訓練記錄（含週訓練相關欄位）
        run = training_repo.create_run(
            name=temp_name,
            train_start=train_start,
            train_end=train_end,
            valid_start=valid_start,
            valid_end=valid_end,
            week_id=week_id,
            factor_pool_hash=factor_pool_hash,
            embargo_days=EMBARGO_DAYS,
        )
        run.candidate_factor_ids = json.dumps(candidate_ids)
        run.status = "running"
        session.commit()

        if on_progress:
            on_progress(0.0, "Initializing training...")

        try:
            # 預載入所有因子資料
            if on_progress:
                on_progress(2.0, "Loading factor data...")

            all_data = self._load_data(
                factors=enabled_factors,
                start_date=train_start,
                end_date=valid_end,
            )

            if all_data.empty:
                raise ValueError("No data available for the specified date range")

            # DoubleEnsemble 預設參數，選擇後自動運行 Optuna 找最佳超參數
            self._optimized_params = {"lgb_params": dict(DEFAULT_LGB_PARAMS)}
            self._auto_optuna = True
            if on_progress:
                on_progress(10.0, "Will auto-tune with Optuna after factor selection")

            # 執行因子選擇（IC 去重複）
            selected_factors, all_results, best_model, selection_stats = self._robust_factor_selection(
                factors=enabled_factors,
                all_data=all_data,
                train_start=train_start,
                train_end=train_end,
                valid_start=valid_start,
                valid_end=valid_end,
                on_progress=on_progress,
            )

            # 計算最終模型 IC（使用最佳模型的 IC）
            # 注意：必須使用 selected_factors 的原始順序，因為 LightGBM 按位置識別特徵
            if best_model is not None and selected_factors:
                factor_names = [f.name for f in selected_factors]
                X_valid = all_data[factor_names]
                y_valid = all_data["label"]

                # 分割驗證資料
                valid_mask = (all_data.index.get_level_values("datetime").date >= valid_start) & \
                             (all_data.index.get_level_values("datetime").date <= valid_end)
                X_valid = X_valid[valid_mask].dropna()
                y_valid = y_valid[valid_mask].dropna()
                common_idx = X_valid.index.intersection(y_valid.index)
                X_valid = X_valid.loc[common_idx]
                y_valid = y_valid.loc[common_idx]

                model_ic = self._calculate_prediction_ic(best_model, X_valid, y_valid)
            else:
                model_ic = 0.0

            # 計算 ICIR
            icir = self._calculate_icir(model_ic, len(selected_factors))

            # 保存因子結果
            for result in all_results:
                training_repo.add_factor_result(
                    run_id=run.id,
                    factor_id=result.factor_id,
                    ic_value=result.ic_value,
                    selected=result.selected,
                )

            # 生成最終模型名稱：{week_id}-{factor_pool_hash}
            # 若無 week_id，使用舊格式 YYYYMM-hash
            if week_id and factor_pool_hash:
                model_name = f"{week_id}-{factor_pool_hash}"
            else:
                hash_input = f"{run.id}-{valid_end.isoformat()}-{len(selected_factors)}"
                short_hash = hashlib.md5(hash_input.encode()).hexdigest()[:6]
                model_name = f"{valid_end.strftime('%Y%m')}-{short_hash}"

            # === 增量更新（驗證後重訓）===
            # 訓練完成後，用驗證期資料做增量更新
            # 保存的是 A'（增量更新後的模型），但報告的是 A 的 IC
            incremented_model = best_model
            if best_model is not None and selected_factors:
                if on_progress:
                    on_progress(96.0, "Incremental update with validation data...")

                factor_names = [f.name for f in selected_factors]
                X_valid_incr = all_data[factor_names]
                y_valid_incr = all_data["label"]

                valid_mask = (all_data.index.get_level_values("datetime").date >= valid_start) & \
                             (all_data.index.get_level_values("datetime").date <= valid_end)
                X_valid_incr = X_valid_incr[valid_mask].dropna()
                y_valid_incr = y_valid_incr[valid_mask].dropna()
                common_idx = X_valid_incr.index.intersection(y_valid_incr.index)
                X_valid_incr = X_valid_incr.loc[common_idx]
                y_valid_incr = y_valid_incr.loc[common_idx]

                if not X_valid_incr.empty:
                    X_valid_processed = self._process_inf(X_valid_incr)
                    X_valid_norm = self._zscore_by_date(X_valid_processed).fillna(0)
                    y_valid_rank = self._rank_by_date(y_valid_incr)

                    try:
                        best_model.incremental_update(
                            X_valid_norm.values, y_valid_rank.values, num_boost_round=50
                        )
                        incremented_model = best_model
                        if on_progress:
                            on_progress(98.0, "Incremental update completed")
                    except Exception as e:
                        if on_progress:
                            on_progress(98.0, f"Incremental update failed: {e}, using original model")
                        incremented_model = best_model

            # 完成訓練
            run.name = model_name
            run.selected_factor_ids = json.dumps([f.id for f in selected_factors])

            # 記錄因子選擇策略
            selection_config = {
                "method": selection_stats["method"],
                "incremental_update": True,
            }
            run.selection_method = selection_stats["method"]
            run.selection_config = json.dumps(selection_config)
            run.selection_stats = json.dumps(selection_stats)

            training_repo.complete_run(
                run_id=run.id,
                model_ic=model_ic,  # 報告的是模型 A 的 IC（驗證期 IC）
                icir=icir,
                factor_count=len(selected_factors),
            )

            # 保存模型檔案（保存增量更新後的 A'）
            config = {
                "train_start": train_start.isoformat(),
                "train_end": train_end.isoformat(),
                "valid_start": valid_start.isoformat(),
                "valid_end": valid_end.isoformat(),
                "week_id": week_id,
                "factor_pool_hash": factor_pool_hash,
                "model_ic": model_ic,  # 報告的 IC（模型 A）
                "icir": icir,
                "incremental_updated": incremented_model is not best_model,
            }
            if self._optimized_params:
                config["hyperparameters"] = self._optimized_params

            self._save_model_files(
                model_name=model_name,
                selected_factors=selected_factors,
                config=config,
                model=incremented_model,  # 保存增量更新後的模型
            )

            if on_progress:
                on_progress(100.0, "Training completed")

            # 計算並儲存訓練品質指標
            try:
                from src.services.stability import QualityMonitor

                quality_monitor = QualityMonitor(session)
                quality_monitor.compute_and_save(run)
                logger.info(f"Computed quality metrics for training run {run.id}")
            except Exception as qe:
                logger.warning(f"Failed to compute quality metrics: {qe}")

            return TrainingResult(
                run_id=run.id,
                model_name=model_name,
                model_ic=model_ic,
                icir=icir,
                selected_factor_ids=[f.id for f in selected_factors],
                all_results=all_results,
            )

        except Exception as e:
            # 標記訓練失敗
            run.status = "failed"
            run.completed_at = datetime.now(TZ_TAIPEI)
            session.commit()
            raise e

    def _calculate_single_factor_ic(
        self,
        factor: Factor,
        all_data: pd.DataFrame,
        train_start: date,
        train_end: date,
    ) -> float:
        """
        計算單因子 IC（用於排序）

        注意：只使用訓練期資料，避免資料洩漏
        """
        try:
            # 只使用訓練期資料（不可偷看驗證期）
            train_mask = (all_data.index.get_level_values("datetime").date >= train_start) & \
                         (all_data.index.get_level_values("datetime").date <= train_end)

            factor_data = all_data[[factor.name, "label"]][train_mask].dropna()
            if len(factor_data) < 100:
                return 0.0

            # 計算每日截面 IC（使用 Spearman，與 Live IC 保持一致）
            def calc_spearman_ic(group: pd.DataFrame) -> float:
                if len(group) < 10:
                    return np.nan
                if group[factor.name].nunique() == 1 or group["label"].nunique() == 1:
                    return np.nan
                return group[factor.name].corr(group["label"], method="spearman")

            daily_ic = factor_data.groupby(level="datetime").apply(calc_spearman_ic)
            mean_ic = daily_ic.mean()
            return float(mean_ic) if not np.isnan(mean_ic) else 0.0
        except Exception:
            return 0.0

    def _robust_factor_selection(
        self,
        factors: list[Factor],
        all_data: pd.DataFrame,
        train_start: date,
        train_end: date,
        valid_start: date,
        valid_end: date,
        on_progress: Callable[[float, str], None] | None = None,
    ) -> tuple[list[Factor], list[FactorEvalResult], Any, dict]:
        """
        因子選擇（RD-Agent IC 去重複）

        使用 RD-Agent 論文的方法移除高相關因子：
        - 計算因子間相關係數
        - 移除 corr >= 0.99 的冗餘因子
        - 保留高 IC 因子

        Returns:
            (selected_factors, all_results, best_model, selection_stats)
        """
        if on_progress:
            on_progress(11.0, "Robust selection: Preparing data...")

        # 準備因子資料
        factor_names = [f.name for f in factors]
        X = all_data[factor_names]
        y = all_data["label"]

        # 使用訓練期資料進行因子選擇
        train_mask = (all_data.index.get_level_values("datetime").date >= train_start) & \
                     (all_data.index.get_level_values("datetime").date <= train_end)
        X_train = X[train_mask]
        y_train = y[train_mask]

        # 執行三階段選擇
        def robust_progress(p: float, msg: str) -> None:
            if on_progress:
                # 11% ~ 90% 給三階段選擇
                progress = 11.0 + p * 0.79
                on_progress(progress, msg)

        # 使用 method="none"，因為因子去重已在因子頁面一次性完成
        # 這樣可以大幅加速訓練（不用每次計算 NxN 相關矩陣）
        robust_selector = RobustFactorSelector(method="none")
        result = robust_selector.select(
            factors=factors,
            X=X_train,
            y=y_train,
            on_progress=robust_progress,
        )

        selected_factors = result.selected_factors

        if on_progress:
            on_progress(90.0, f"Factor selection ({result.method}): {len(selected_factors)} factors selected")

        # 構建 all_results（與舊方法兼容）
        all_results: list[FactorEvalResult] = []
        selected_names = {f.name for f in selected_factors}
        for factor in factors:
            # 計算單因子 IC
            ic = self._calculate_single_factor_ic(factor, all_data, train_start, train_end)
            all_results.append(
                FactorEvalResult(
                    factor_id=factor.id,
                    factor_name=factor.name,
                    ic_value=ic,
                    selected=factor.name in selected_names,
                )
            )

        # 訓練最終模型
        best_model = None
        if selected_factors:
            if on_progress:
                on_progress(92.0, "Training final model...")

            selected_factor_names = [f.name for f in selected_factors]
            X_train_selected = X_train[selected_factor_names]

            # 驗證資料
            valid_mask = (all_data.index.get_level_values("datetime").date >= valid_start) & \
                         (all_data.index.get_level_values("datetime").date <= valid_end)
            X_valid = X[valid_mask][selected_factor_names]
            y_valid = y[valid_mask]

            # 處理缺失值
            train_valid = ~(X_train_selected.isna().any(axis=1) | y_train.isna())
            valid_valid = ~(X_valid.isna().any(axis=1) | y_valid.isna())

            X_train_clean = X_train_selected[train_valid]
            y_train_clean = y_train[train_valid]
            X_valid_clean = X_valid[valid_valid]
            y_valid_clean = y_valid[valid_valid]

            # 標準化特徵（不對標籤二次標準化，標籤已經是排名值）
            X_train_processed = self._process_inf(X_train_clean)
            X_train_norm = self._zscore_by_date(X_train_processed).fillna(0)
            # 標籤已經在 _prepare_train_valid_data 中用 _rank_by_date 標準化過
            # 不要再做 _zscore_by_date，否則會導致信號丟失
            y_train_final = y_train_clean

            X_valid_processed = self._process_inf(X_valid_clean)
            X_valid_norm = self._zscore_by_date(X_valid_processed).fillna(0)
            y_valid_final = y_valid_clean

            # 如果需要自動調參，在訓練前運行 Optuna
            if self._auto_optuna:
                if on_progress:
                    on_progress(92.0, f"Auto-tuning DoubleEnsemble with Optuna ({len(selected_factors)} factors)...")

                def optuna_progress(p: float, msg: str) -> None:
                    if on_progress:
                        progress = 92.0 + p * 0.02
                        on_progress(progress, msg)

                optimized_params = self._optimize_hyperparameters(
                    X_train=X_train_clean,
                    y_train=y_train_final,
                    X_valid=X_valid_clean,
                    y_valid=y_valid_final,
                    n_trials=15,  # DoubleEnsemble 每 trial 較慢，減少次數
                    timeout=300,  # 5 分鐘超時
                    on_progress=optuna_progress,
                )
                self._optimized_params = optimized_params
                lgb_best = optimized_params.get("lgb_params", {})
                logger.info(f"Optuna found best LGB params: lr={lgb_best.get('learning_rate', 0):.4f}, leaves={lgb_best.get('num_leaves')}")

                if on_progress:
                    on_progress(94.0, f"Optuna done: lr={lgb_best.get('learning_rate', 0):.4f}, leaves={lgb_best.get('num_leaves')}")

            try:
                best_model = self._train_model(
                    X_train_clean, y_train_final,
                    X_valid_clean, y_valid_final,
                    params=self._optimized_params,
                )
            except Exception as e:
                if on_progress:
                    on_progress(95.0, f"Model training failed: {e}")

        # 構建選擇統計
        selection_stats = {
            "method": result.method,
            "initial_factors": len(factors),
            "final_factors": len(selected_factors),
            **result.selection_stats,
            "stage_results": result.stage_results,
        }

        if on_progress:
            on_progress(95.0, f"Factor selection ({result.method}) completed")

        return selected_factors, all_results, best_model, selection_stats

    def _calculate_icir(self, ic: float, factor_count: int) -> float | None:
        """計算 IC Information Ratio (ICIR = IC / std(IC))"""
        if factor_count == 0 or ic == 0:
            return None

        ic_std = self._last_ic_std
        if ic_std is None or ic_std == 0:
            return None

        return ic / ic_std

    def _save_model_files(
        self,
        model_name: str,
        selected_factors: list[Factor],
        config: dict,
        model: Any = None,
    ) -> None:
        """保存模型檔案（含 LightGBM .pkl）"""
        model_dir = MODELS_DIR / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        # 保存配置
        config_path = model_dir / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        # 保存因子
        factors_data = [
            {
                "id": f.id,
                "name": f.name,
                "expression": f.expression,
            }
            for f in selected_factors
        ]
        factors_path = model_dir / "factors.json"
        with open(factors_path, "w", encoding="utf-8") as f:
            json.dump(factors_data, f, indent=2, ensure_ascii=False)

        # 保存 LightGBM 模型
        if model is not None:
            model_path = model_dir / "model.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)



def run_training(
    session: Session,
    qlib_data_dir: Path | str,
    train_end: date | None = None,
    on_progress: Callable[[int, str], None] | None = None,
) -> TrainingResult:
    """
    執行訓練的便利函數

    Args:
        session: 資料庫 Session
        qlib_data_dir: qlib 資料目錄
        train_end: 訓練結束日期（預設：今日 - 驗證期天數）
        on_progress: 進度回調
    """
    from datetime import timedelta

    today = date.today()

    if train_end is None:
        train_end = today - timedelta(days=VALID_DAYS)

    train_start = train_end - timedelta(days=TRAIN_DAYS)
    valid_start = train_end + timedelta(days=1)
    valid_end = today

    trainer = ModelTrainer(qlib_data_dir)
    return trainer.train(
        session=session,
        train_start=train_start,
        train_end=train_end,
        valid_start=valid_start,
        valid_end=valid_end,
        on_progress=on_progress,
    )
