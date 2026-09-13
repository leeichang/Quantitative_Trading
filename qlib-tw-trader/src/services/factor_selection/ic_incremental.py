"""IC 增量選擇法（Greedy Forward Selection）

算法：
1. 計算所有因子的單因子 IC（每日截面 Spearman），按 IC 絕對值降序排列
2. 初始化因子池 = [最高 IC 因子]，訓練 baseline LightGBM，記錄模型 IC
3. 依序嘗試加入因子：
   - 加入 → 訓練 LightGBM → 計算驗證期 IC
   - IC 提高 → 保留 / IC 未提高 → 剔除
4. 返回最終因子池

參考：docs/ic-incremental-selection.md
"""

import logging
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.repositories.models import Factor
from src.services.factor_selection.base import FactorSelectionResult, FactorSelector

logger = logging.getLogger(__name__)


class ICIncrementalSelector(FactorSelector):
    """IC 增量選擇法"""

    def __init__(
        self,
        lgbm_params: dict,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
        quick_rounds: int = 200,
    ):
        """
        Args:
            lgbm_params: LightGBM 訓練參數（來自超參數培養）
            X_valid: 驗證期特徵（未標準化，原始值）
            y_valid: 驗證期標籤（已 CSRankNorm）
            quick_rounds: 選擇階段的 LightGBM 訓練輪數（快速評估）
        """
        self.lgbm_params = lgbm_params
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.quick_rounds = quick_rounds

    def select(
        self,
        factors: list[Factor],
        X: pd.DataFrame,
        y: pd.Series,
        on_progress: Callable[[float, str], None] | None = None,
    ) -> FactorSelectionResult:
        """
        執行 IC 增量選擇

        Args:
            factors: 候選因子列表
            X: 訓練期特徵（未標準化）
            y: 訓練期標籤（已 CSRankNorm）
            on_progress: 進度回調 (progress: 0-100, message: str)
        """
        import lightgbm as lgb

        if len(factors) == 0:
            return FactorSelectionResult(
                selected_factors=[],
                selection_stats={"method": "ic_incremental", "input_count": 0, "output_count": 0},
                method="ic_incremental",
            )

        factor_names = [f.name for f in factors]
        available_names = [n for n in factor_names if n in X.columns]
        factors_map = {f.name: f for f in factors if f.name in available_names}

        if on_progress:
            on_progress(0, f"IC Incremental: Computing single-factor IC for {len(available_names)} factors...")

        # Step 1: 計算所有因子的單因子 IC（每日截面 Spearman）
        single_ics = self._compute_single_factor_ics(X, y, available_names)

        # 按 IC 絕對值降序排列
        sorted_names = sorted(available_names, key=lambda n: abs(single_ics.get(n, 0.0)), reverse=True)

        if not sorted_names:
            return FactorSelectionResult(
                selected_factors=[],
                selection_stats={"method": "ic_incremental", "input_count": len(factors), "output_count": 0},
                method="ic_incremental",
            )

        if on_progress:
            on_progress(5, f"IC Incremental: Top factor IC={single_ics[sorted_names[0]]:.4f}, starting selection...")

        # Step 2: 初始化 — 用最高 IC 因子訓練 baseline
        selected_names = [sorted_names[0]]
        best_ic = self._evaluate_model(selected_names, X, y, lgb)

        logger.info(f"IC Incremental: Baseline with [{sorted_names[0]}] IC={best_ic:.4f}")

        # Step 3: 依序嘗試加入因子
        total = len(sorted_names)
        trial_log: list[dict] = []

        for i, name in enumerate(sorted_names[1:], start=2):
            if on_progress:
                progress = 5 + (i / total) * 90
                on_progress(progress, f"IC Incremental: Testing factor {i}/{total} ({name})...")

            # 暫時加入
            candidate = selected_names + [name]
            candidate_ic = self._evaluate_model(candidate, X, y, lgb)

            if candidate_ic > best_ic:
                selected_names.append(name)
                trial_log.append({"name": name, "ic_before": best_ic, "ic_after": candidate_ic, "kept": True})
                best_ic = candidate_ic
                logger.debug(f"  + {name}: IC {best_ic:.4f} → {candidate_ic:.4f} (kept)")
            else:
                trial_log.append({"name": name, "ic_before": best_ic, "ic_after": candidate_ic, "kept": False})
                logger.debug(f"  - {name}: IC {best_ic:.4f} → {candidate_ic:.4f} (removed)")

        # 結果
        selected_factors = [factors_map[n] for n in selected_names]

        if on_progress:
            on_progress(100, f"IC Incremental: {len(selected_factors)} factors selected, IC={best_ic:.4f}")

        logger.info(
            f"IC Incremental: {len(factors)} → {len(selected_factors)} factors, "
            f"final IC={best_ic:.4f}"
        )

        stats = {
            "method": "ic_incremental",
            "input_count": len(factors),
            "output_count": len(selected_factors),
            "final_model_ic": best_ic,
            "single_factor_ics": {n: single_ics.get(n, 0.0) for n in selected_names},
            "selected_factor_names": selected_names,
            "trial_log": trial_log[-20:],  # 只保留最後 20 筆
        }

        return FactorSelectionResult(
            selected_factors=selected_factors,
            selection_stats=stats,
            method="ic_incremental",
        )

    def _compute_single_factor_ics(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        factor_names: list[str],
    ) -> dict[str, float]:
        """計算每個因子的單因子 IC（每日截面 Spearman，取平均）"""
        result = {}
        combined = X[factor_names].copy()
        combined["label"] = y

        for name in factor_names:
            try:
                factor_data = combined[[name, "label"]].dropna()
                if len(factor_data) < 100:
                    result[name] = 0.0
                    continue

                def calc_spearman(group: pd.DataFrame) -> float:
                    if len(group) < 10:
                        return np.nan
                    return group[name].corr(group["label"], method="spearman")

                daily_ic = factor_data.groupby(level="datetime").apply(calc_spearman)
                mean_ic = daily_ic.mean()
                result[name] = float(mean_ic) if not np.isnan(mean_ic) else 0.0
            except Exception:
                result[name] = 0.0

        return result

    def _evaluate_model(
        self,
        factor_names: list[str],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        lgb: Any,
    ) -> float:
        """用指定因子訓練 LightGBM 並計算驗證期 IC"""
        # 準備訓練資料
        X_tr = X_train[factor_names].copy()
        train_valid = ~(X_tr.isna().any(axis=1) | y_train.isna())
        X_tr = X_tr[train_valid]
        y_tr = y_train[train_valid]

        # 標準化
        X_tr = self._process_inf(X_tr)
        X_tr = self._zscore_by_date(X_tr).fillna(0)

        # 準備驗證資料
        X_vl = self.X_valid[factor_names].copy()
        valid_valid = ~(X_vl.isna().any(axis=1) | self.y_valid.isna())
        X_vl = X_vl[valid_valid]
        y_vl = self.y_valid[valid_valid]

        X_vl = self._process_inf(X_vl)
        X_vl = self._zscore_by_date(X_vl).fillna(0)

        if X_tr.empty or X_vl.empty:
            return 0.0

        # 訓練
        train_data = lgb.Dataset(X_tr.values, label=y_tr.values)
        valid_data = lgb.Dataset(X_vl.values, label=y_vl.values, reference=train_data)

        model = lgb.train(
            self.lgbm_params,
            train_data,
            num_boost_round=self.quick_rounds,
            valid_sets=[valid_data],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        )

        # 計算驗證期 IC（每日截面 Spearman）
        predictions = model.predict(X_vl.values)
        pred_df = pd.DataFrame({
            "pred": predictions,
            "label": y_vl.values,
        }, index=y_vl.index)

        def calc_spearman_ic(group: pd.DataFrame) -> float:
            if len(group) < 10:
                return np.nan
            if group["pred"].nunique() == 1 or group["label"].nunique() == 1:
                return np.nan
            return group["pred"].corr(group["label"], method="spearman")

        daily_ic = pred_df.groupby(level="datetime").apply(calc_spearman_ic)
        mean_ic = daily_ic.mean()
        return float(mean_ic) if not np.isnan(mean_ic) else 0.0

    def _process_inf(self, df: pd.DataFrame) -> pd.DataFrame:
        """處理無窮大值"""
        df = df.copy()
        for col in df.columns:
            mask = np.isinf(df[col])
            if mask.any():
                col_mean = df.loc[~mask, col].mean()
                df.loc[mask, col] = col_mean if not np.isnan(col_mean) else 0
        return df

    def _zscore_by_date(self, df: pd.DataFrame) -> pd.DataFrame:
        """每日截面標準化"""
        return df.groupby(level="datetime", group_keys=False).apply(
            lambda x: (x - x.mean()) / (x.std() + 1e-8)
        )
