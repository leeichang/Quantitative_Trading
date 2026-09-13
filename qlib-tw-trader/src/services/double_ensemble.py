"""
DoubleEnsemble Model (ICDM 2020)

Iterative ensemble of LightGBM sub-models with:
- Sample Reweighting (SR): focus on hard-to-learn samples
- Feature Selection (FS): permutation importance-based feature filtering

Compatible with lgb.Booster API: .predict(ndarray), .feature_importance()

Reference: Zhu et al. "DoubleEnsemble: A New Ensemble Method Based on Sample Reweighting
and Feature Selection for Financial Data Analysis" (ICDM 2020)
"""

import logging
from math import ceil
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DoubleEnsembleModel:
    """DoubleEnsemble wrapper compatible with lgb.Booster.predict() interface."""

    def __init__(
        self,
        num_models: int = 6,
        epochs: int = 100,
        decay: float = 0.5,
        enable_sr: bool = True,
        enable_fs: bool = True,
        alpha1: float = 1.0,
        alpha2: float = 1.0,
        bins_sr: int = 10,
        bins_fs: int = 5,
        sample_ratios: list[float] | None = None,
        early_stopping_rounds: int = 20,
        **lgb_params: Any,
    ):
        self.num_models = num_models
        self.epochs = epochs
        self.decay = decay
        self.enable_sr = enable_sr
        self.enable_fs = enable_fs
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.bins_sr = bins_sr
        self.bins_fs = bins_fs
        self.sample_ratios = sample_ratios or [0.8, 0.7, 0.6, 0.5, 0.4]
        self.early_stopping_rounds = early_stopping_rounds
        self.lgb_params = lgb_params

        self.sub_models: list[lgb.Booster] = []
        self.sub_weights: list[float] = []
        self.sub_features: list[np.ndarray] = []
        self.num_features: int = 0

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_valid: np.ndarray,
        y_valid: np.ndarray,
    ) -> None:
        n_samples, n_features = X_train.shape
        self.num_features = n_features
        self.sub_models = []
        self.sub_weights = []
        self.sub_features = []

        weights = np.ones(n_samples)
        features = np.arange(n_features)

        for k in range(self.num_models):
            logger.info(
                f"DoubleEnsemble: training sub-model {k + 1}/{self.num_models} "
                f"({len(features)} features)"
            )

            model = self._train_submodel(
                X_train[:, features],
                y_train,
                X_valid[:, features],
                y_valid,
                weights,
            )
            self.sub_models.append(model)
            self.sub_features.append(features.copy())
            self.sub_weights.append(1.0)

            if k == self.num_models - 1:
                break

            # Loss curve from this sub-model
            loss_curve = self._retrieve_loss_curve(
                model, X_train[:, features], y_train
            )

            # Ensemble prediction (weighted average of all sub-models so far)
            pred_ensemble = self._ensemble_predict(X_train)
            loss_values = (y_train - pred_ensemble) ** 2

            if self.enable_sr:
                weights = self._sample_reweight(loss_curve, loss_values, k + 1)

            if self.enable_fs:
                features = self._feature_selection(X_train, y_train)

    def _train_submodel(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_valid: np.ndarray,
        y_valid: np.ndarray,
        weights: np.ndarray,
    ) -> lgb.Booster:
        dtrain = lgb.Dataset(X_train, label=y_train, weight=weights)
        dvalid = lgb.Dataset(X_valid, label=y_valid)

        return lgb.train(
            self.lgb_params,
            dtrain,
            num_boost_round=self.epochs,
            valid_sets=[dvalid],
            callbacks=[
                lgb.early_stopping(self.early_stopping_rounds, verbose=False),
            ],
        )

    def _retrieve_loss_curve(
        self,
        model: lgb.Booster,
        X: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """Per-sample MSE at each tree iteration."""
        num_trees = model.num_trees()
        loss_curve = np.zeros((len(y), num_trees))
        for t in range(num_trees):
            pred = model.predict(X, num_iteration=t + 1)
            loss_curve[:, t] = (y - pred) ** 2
        return loss_curve

    def _sample_reweight(
        self,
        loss_curve: np.ndarray,
        loss_values: np.ndarray,
        k_th: int,
    ) -> np.ndarray:
        """Reweight samples: hard-to-learn samples get higher weight."""
        # Rank-normalize loss curve per column
        loss_curve_norm = pd.DataFrame(loss_curve).rank(axis=0, pct=True).values

        # h1: current ensemble loss rank
        h1 = pd.Series(loss_values).rank(pct=True).values

        # h2: learning trajectory (flat trajectory = hard sample)
        T = loss_curve.shape[1]
        part = max(int(T * 0.1), 1)
        l_start = loss_curve_norm[:, :part].mean(axis=1)
        l_end = loss_curve_norm[:, -part:].mean(axis=1)
        l_start = np.clip(l_start, 1e-7, None)
        h2 = pd.Series(l_end / l_start).rank(pct=True).values

        h = self.alpha1 * h1 + self.alpha2 * h2

        # Bin and compute per-bin average
        h_series = pd.Series(h)
        try:
            bins = pd.cut(h_series, self.bins_sr)
            h_avg = h_series.groupby(bins).transform("mean").values
        except ValueError:
            return np.ones(len(loss_values))

        nan_mask = np.isnan(h_avg)
        if nan_mask.any():
            h_avg[nan_mask] = h.mean()

        return 1.0 / (self.decay**k_th * h_avg + 0.1)

    def _feature_selection(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> np.ndarray:
        """Select features via permutation importance across current ensemble."""
        n_features = X_train.shape[1]

        pred_baseline = self._ensemble_predict(X_train)
        loss_baseline = (y_train - pred_baseline) ** 2

        # 在可寫副本上做 permutation，不要動到呼叫端傳進來的陣列。
        # pandas 3.0 的 copy-on-write 會讓 DataFrame 轉出的 ndarray 唯讀，
        # 原本的就地寫入會拋 ValueError: assignment destination is read-only。
        X_perm = np.array(X_train, copy=True)

        g_values = np.zeros(n_features)
        for f in range(n_features):
            orig_col = X_perm[:, f].copy()
            X_perm[:, f] = np.random.permutation(orig_col)

            pred_perm = self._ensemble_predict(X_perm)
            loss_perm = (y_train - pred_perm) ** 2

            X_perm[:, f] = orig_col

            loss_diff = loss_perm - loss_baseline
            g_values[f] = loss_diff.mean() / (loss_diff.std() + 1e-7)

        g_values = np.nan_to_num(g_values, nan=0.0)

        # Bin g-values by importance and sample from each bin
        g_df = pd.DataFrame({"feature": np.arange(n_features), "g_value": g_values})
        try:
            g_df["bins"] = pd.cut(g_df["g_value"], self.bins_fs)
        except ValueError:
            return np.arange(n_features)

        sorted_bins = sorted(g_df["bins"].dropna().unique(), reverse=True)

        selected: list[int] = []
        for i_b, bin_b in enumerate(sorted_bins):
            b_feats = g_df[g_df["bins"] == bin_b]["feature"].values
            if len(b_feats) == 0:
                continue
            ratio = self.sample_ratios[min(i_b, len(self.sample_ratios) - 1)]
            num_select = max(1, int(ceil(ratio * len(b_feats))))
            chosen = np.random.choice(
                b_feats, size=min(num_select, len(b_feats)), replace=False
            )
            selected.extend(chosen)

        result = np.array(sorted(set(selected)))
        return result if len(result) > 0 else np.arange(n_features)

    def _ensemble_predict(self, X: np.ndarray) -> np.ndarray:
        """Weighted average prediction from all current sub-models."""
        pred = np.zeros(len(X))
        w_sum = sum(self.sub_weights)
        for model, weight, feat_idx in zip(
            self.sub_models, self.sub_weights, self.sub_features
        ):
            pred += weight * model.predict(X[:, feat_idx])
        return pred / w_sum

    # ── Public API (compatible with lgb.Booster) ──

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Weighted average prediction."""
        return self._ensemble_predict(X)

    def feature_importance(self) -> np.ndarray:
        """Weighted feature importance aggregated across sub-models."""
        total = np.zeros(self.num_features)
        for model, weight, feat_idx in zip(
            self.sub_models, self.sub_weights, self.sub_features
        ):
            imp = model.feature_importance()
            for i, idx in enumerate(feat_idx):
                total[idx] += weight * imp[i]
        return total

    def incremental_update(
        self,
        X: np.ndarray,
        y: np.ndarray,
        num_boost_round: int = 50,
    ) -> None:
        """Incrementally update each sub-model (feature subsets stay fixed)."""
        params = dict(self.lgb_params)
        params["learning_rate"] = params.get("learning_rate", 0.05) * 0.2

        for i, (model, feat_idx) in enumerate(
            zip(self.sub_models, self.sub_features)
        ):
            data = lgb.Dataset(X[:, feat_idx], label=y)
            self.sub_models[i] = lgb.train(
                params,
                data,
                num_boost_round=num_boost_round,
                init_model=model,
                keep_training_booster=True,
            )
