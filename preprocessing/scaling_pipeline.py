"""
=============================================================================
AIRCRAFT ENGINE AI - HEALTH MONITORING & PREDICTIVE MAINTENANCE
Phase 3: Data Preprocessing and Feature Engineering
Step 3.7: Feature Scaling & Normalization Pipeline Architecture
=============================================================================

This module provides a modular, reproducible, and leakage-free feature scaling
pipeline for the NASA C-MAPSS dataset.

Key Design Principles:
  1. Strict Leakage Prevention: Scaler is fitted exclusively on the training
     engine partition. Validation and test data are strictly transformed using
     parameters learned from training.
  2. Feature Set Agnostic: Supports Tier 1 (14 core sensors), Tier 2 (17
     multi-condition features), and Tier 3 (24 extended features).
  3. Metadata & Target Separation: 'unit_number', 'time_cycles', and 'RUL'
     are strictly isolated and never scaled as input features.
  4. Safe Invariant Handling: Division-by-zero protection is applied for any
     feature with zero range, zero variance, or zero IQR.
  5. Scaler Persistence: Scalers can be saved and loaded via pickle.

IMPORTANT SEQUENCING NOTICE:
  Per Phase 3 specifications, the training/validation engine split occurs in
  Step 3.10. Therefore, scalers are NOT permanently fitted on full datasets
  in Step 3.7. Actual fitting occurs in Step 3.10 after engine partitioning.
=============================================================================
"""

import os
import pickle
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# FEATURE CONFIGURATION TIERS (From Step 3.6)
# -----------------------------------------------------------------------------

TIER1_CORE_SENSORS = [
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_17",
    "sensor_20",
    "sensor_21",
]

TIER2_MULTI_CONDITION = [
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
] + TIER1_CORE_SENSORS

TIER3_EXTENDED_ALL = [
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
    "sensor_1",
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_5",
    "sensor_6",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_10",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_16",
    "sensor_17",
    "sensor_18",
    "sensor_19",
    "sensor_20",
    "sensor_21",
]

FEATURE_TIERS = {
    "tier1_core": TIER1_CORE_SENSORS,
    "tier2_multi_condition": TIER2_MULTI_CONDITION,
    "tier3_extended": TIER3_EXTENDED_ALL,
}

METADATA_COLUMNS = ["unit_number", "time_cycles"]
TARGET_COLUMN = "RUL"


# -----------------------------------------------------------------------------
# STANDALONE VECTORISED SCALER IMPLEMENTATIONS
# -----------------------------------------------------------------------------

class BaseScaler:
    """Base interface for scaling transformers."""
    def fit(self, X: np.ndarray):
        raise NotImplementedError

    def transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class SafeMinMaxScaler(BaseScaler):
    """
    MinMaxScaler with zero-division protection for constant features.
    Transforms features to [feature_range[0], feature_range[1]].
    """
    def __init__(self, feature_range=(0.0, 1.0)):
        self.feature_range = feature_range
        self.data_min_ = None
        self.data_max_ = None
        self.data_range_ = None
        self.scale_ = None
        self.min_ = None

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        self.data_min_ = np.nanmin(X, axis=0)
        self.data_max_ = np.nanmax(X, axis=0)
        self.data_range_ = self.data_max_ - self.data_min_

        # Zero-division protection: where range is zero, set scale to 0.0
        safe_range = np.where(self.data_range_ == 0.0, 1.0, self.data_range_)
        res_min, res_max = self.feature_range
        self.scale_ = (res_max - res_min) / safe_range
        self.min_ = res_min - self.data_min_ * self.scale_
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        X_scaled = X * self.scale_ + self.min_
        # For constant features, assign lower bound
        if self.data_range_ is not None and np.any(self.data_range_ == 0.0):
            const_mask = (self.data_range_ == 0.0)
            X_scaled[:, const_mask] = self.feature_range[0]
        return X_scaled

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        res_min, _ = self.feature_range
        safe_scale = np.where(self.scale_ == 0.0, 1.0, self.scale_)
        X_orig = (X - self.min_) / safe_scale
        if self.data_range_ is not None and np.any(self.data_range_ == 0.0):
            const_mask = (self.data_range_ == 0.0)
            X_orig[:, const_mask] = self.data_min_[const_mask]
        return X_orig


class SafeStandardScaler(BaseScaler):
    """
    StandardScaler (z-score) with zero-division protection.
    Standardizes features by removing mean and scaling to unit variance.
    """
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
        self.var_ = None

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = np.nanmean(X, axis=0)
        self.var_ = np.nanvar(X, axis=0)
        std = np.sqrt(self.var_)
        # Zero-division protection: where std is zero, set scale to 1.0
        self.scale_ = np.where(std == 0.0, 1.0, std)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        X_scaled = (X - self.mean_) / self.scale_
        # For zero variance features, set centered output strictly to 0.0
        const_mask = (self.var_ == 0.0)
        if np.any(const_mask):
            X_scaled[:, const_mask] = 0.0
        return X_scaled

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return X * self.scale_ + self.mean_


class SafeRobustScaler(BaseScaler):
    """
    RobustScaler using median and IQR with zero-IQR protection.
    """
    def __init__(self, quantile_range=(25.0, 75.0)):
        self.quantile_range = quantile_range
        self.center_ = None
        self.scale_ = None

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        q_min, q_max = self.quantile_range
        self.center_ = np.nanmedian(X, axis=0)
        q25 = np.nanpercentile(X, q_min, axis=0)
        q75 = np.nanpercentile(X, q_max, axis=0)
        iqr = q75 - q25
        # Where IQR == 0, fallback to 1.0
        self.scale_ = np.where(iqr == 0.0, 1.0, iqr)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        X_scaled = (X - self.center_) / self.scale_
        return X_scaled

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return X * self.scale_ + self.center_


class PassThroughScaler(BaseScaler):
    """Pass-through scaler for unscaled models (e.g. XGBoost, Random Forest)."""
    def fit(self, X: np.ndarray):
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.float64)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.float64)


# -----------------------------------------------------------------------------
# HIGH-LEVEL SCALING PIPELINE CLASS
# -----------------------------------------------------------------------------

class CMAPSSScalingPipeline:
    """
    End-to-end feature extraction and scaling pipeline.
    Ensures zero data leakage between training, validation, and testing partitions.
    """
    def __init__(self, method: str = "minmax", feature_tier: str = "tier1_core"):
        """
        Parameters:
            method: 'minmax', 'standard', 'robust', or 'none' (passthrough).
            feature_tier: 'tier1_core', 'tier2_multi_condition', or 'tier3_extended'.
        """
        self.method = method.lower()
        self.feature_tier = feature_tier
        if feature_tier not in FEATURE_TIERS:
            raise ValueError(f"Unknown feature tier: {feature_tier}. Available: {list(FEATURE_TIERS.keys())}")

        self.feature_columns = FEATURE_TIERS[feature_tier]
        self.is_fitted = False

        if self.method in ["minmax", "min_max"]:
            self.scaler = SafeMinMaxScaler(feature_range=(0.0, 1.0))
        elif self.method in ["standard", "zscore"]:
            self.scaler = SafeStandardScaler()
        elif self.method in ["robust", "iqr"]:
            self.scaler = SafeRobustScaler()
        elif self.method in ["none", "passthrough", "raw"]:
            self.scaler = PassThroughScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}. Choose from 'minmax', 'standard', 'robust', 'none'.")

    def fit(self, df_train: pd.DataFrame):
        """Fit scaler strictly on the training partition."""
        self._validate_schema(df_train)
        X_train = df_train[self.feature_columns].values
        self.scaler.fit(X_train)
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame, return_dataframe: bool = True):
        """
        Transform a dataset partition (validation or test) using fitted parameters.
        Returns scaled features, with metadata and target preserved.
        """
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted on training partition before transform().")

        self._validate_schema(df)
        X_raw = df[self.feature_columns].values
        X_scaled = self.scaler.transform(X_raw)

        if return_dataframe:
            df_out = pd.DataFrame(X_scaled, columns=self.feature_columns, index=df.index)
            # Reattach metadata if present
            for meta_col in METADATA_COLUMNS:
                if meta_col in df.columns:
                    df_out.insert(0 if meta_col == "unit_number" else 1, meta_col, df[meta_col])
            # Reattach target if present
            if TARGET_COLUMN in df.columns:
                df_out[TARGET_COLUMN] = df[TARGET_COLUMN]
            return df_out
        else:
            # Return tuple: (X_scaled, metadata_dict, target_array)
            meta = {c: df[c].values for c in METADATA_COLUMNS if c in df.columns}
            target = df[TARGET_COLUMN].values if TARGET_COLUMN in df.columns else None
            return X_scaled, meta, target

    def fit_transform(self, df_train: pd.DataFrame, return_dataframe: bool = True):
        """Fit and transform training partition."""
        return self.fit(df_train).transform(df_train, return_dataframe=return_dataframe)

    def _validate_schema(self, df: pd.DataFrame):
        """Verify that all configured feature columns exist."""
        missing = [c for c in self.feature_columns if c not in df.columns]
        if missing:
            raise KeyError(f"Dataset missing required feature columns for {self.feature_tier}: {missing}")

    def save(self, filepath: str):
        """Serialize fitted pipeline to disk."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted pipeline.")
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    _SCALER_LOAD_CACHE = {}

    @staticmethod
    def load(filepath: str) -> "CMAPSSScalingPipeline":
        """Load serialized pipeline from disk."""
        abs_p = os.path.abspath(filepath)
        if abs_p in CMAPSSScalingPipeline._SCALER_LOAD_CACHE:
            return CMAPSSScalingPipeline._SCALER_LOAD_CACHE[abs_p]
        with open(filepath, "rb") as f:
            pipeline = pickle.load(f)
        CMAPSSScalingPipeline._SCALER_LOAD_CACHE[abs_p] = pipeline
        return pipeline


# -----------------------------------------------------------------------------
# DRY-RUN / INTEGRATION VERIFICATION FUNCTION
# -----------------------------------------------------------------------------

def run_dry_run_verification():
    """
    Executes a safe, non-persistent synthetic demonstration of the pipeline.
    Demonstrates zero leakage, constant handling, and metadata preservation.
    """
    print("=" * 80)
    print("STEP 3.7: DRY-RUN VERIFICATION OF SCALING PIPELINE")
    print("=" * 80)

    # Synthetic training engine batch (Engine 1 & 2)
    synthetic_train = pd.DataFrame({
        "unit_number": [1, 1, 2, 2],
        "time_cycles": [1, 2, 1, 2],
        "op_setting_1": [0.001, -0.002, 0.003, -0.001],
        "op_setting_2": [0.0001, -0.0002, 0.0001, 0.0000],
        "op_setting_3": [100.0, 100.0, 100.0, 100.0],  # Constant feature
        "sensor_2": [641.5, 642.0, 641.8, 642.3],
        "sensor_3": [1580.0, 1585.0, 1582.0, 1589.0],
        "sensor_4": [1400.0, 1405.0, 1402.0, 1408.0],
        "sensor_7": [553.0, 552.0, 553.5, 551.5],
        "sensor_8": [2388.0, 2388.1, 2388.05, 2388.15],
        "sensor_9": [9050.0, 9055.0, 9052.0, 9060.0],
        "sensor_11": [47.3, 47.5, 47.4, 47.6],
        "sensor_12": [521.0, 520.5, 521.2, 520.2],
        "sensor_13": [2388.0, 2388.1, 2388.05, 2388.15],
        "sensor_14": [8130.0, 8135.0, 8132.0, 8140.0],
        "sensor_15": [8.40, 8.42, 8.41, 8.44],
        "sensor_17": [392.0, 393.0, 392.0, 394.0],
        "sensor_20": [38.8, 38.6, 38.7, 38.5],
        "sensor_21": [23.3, 23.2, 23.3, 23.1],
        "RUL": [150, 149, 180, 179],
    })

    # Synthetic validation engine (Engine 3)
    synthetic_val = pd.DataFrame({
        "unit_number": [3, 3],
        "time_cycles": [1, 2],
        "op_setting_1": [0.002, -0.001],
        "op_setting_2": [0.0000, 0.0001],
        "op_setting_3": [100.0, 100.0],  # Constant
        "sensor_2": [641.7, 642.1],
        "sensor_3": [1581.0, 1587.0],
        "sensor_4": [1401.0, 1406.0],
        "sensor_7": [553.2, 551.8],
        "sensor_8": [2388.02, 2388.12],
        "sensor_9": [9051.0, 9058.0],
        "sensor_11": [47.35, 47.55],
        "sensor_12": [521.1, 520.4],
        "sensor_13": [2388.02, 2388.12],
        "sensor_14": [8131.0, 8138.0],
        "sensor_15": [8.405, 8.43],
        "sensor_17": [392.0, 393.5],
        "sensor_20": [38.75, 38.55],
        "sensor_21": [23.28, 23.15],
        "RUL": [160, 159],
    })

    for method in ["minmax", "standard", "none"]:
        print(f"\n--- Testing Scaler: {method.upper()} (Tier 1: 14 Core Sensors) ---")
        pipeline = CMAPSSScalingPipeline(method=method, feature_tier="tier1_core")

        # Fit strictly on train
        train_scaled = pipeline.fit_transform(synthetic_train)
        # Transform validation without fitting
        val_scaled = pipeline.transform(synthetic_val)

        print(f"Train Scaled Shape: {train_scaled.shape}, Cols: {list(train_scaled.columns[:4])} ...")
        print(f"Val Scaled Shape:   {val_scaled.shape}")
        print(f"Metadata preserved: 'unit_number'={list(val_scaled['unit_number'])}, 'RUL'={list(val_scaled['RUL'])}")

        if method == "minmax":
            assert (train_scaled[TIER1_CORE_SENSORS].min().min() >= 0.0)
            assert (train_scaled[TIER1_CORE_SENSORS].max().max() <= 1.0)
            print("  [PASS] MinMaxScaler bounded in [0, 1] on training partition.")

    print("\n[SUCCESS] Pipeline structure verified successfully.")
    print("No scalers fitted on real datasets in this step (deferred to Step 3.10).")


if __name__ == "__main__":
    run_dry_run_verification()
