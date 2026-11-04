"""
=============================================================================
AIRCRAFT ENGINE AI - HEALTH MONITORING & PREDICTIVE MAINTENANCE
Phase 4: Machine Learning and Model Evaluation
Step 4.2: Train and Evaluate XGBoost Regression Model
=============================================================================

This script implements and quantitatively evaluates XGBoost regression models
across all four NASA C-MAPSS subsets (FD001, FD002, FD003, FD004) and across
three feature tiers (Tier 1, Tier 2, Tier 3).

Total Configurations: Exactly 40 distinct model configurations
(10 per subset x 4 subsets):
  1-3.   XGBoost - Final Observation (Tier 1, Tier 2, Tier 3)
  4-6.   XGBoost - Flattened Window (Tier 1, Tier 2, Tier 3)
  7-9.   XGBoost - Summary Statistics (Tier 1, Tier 2, Tier 3)
  10.    XGBoost - Tuned / Regularized Summary Statistics (Best Val Tier)

Strict Protocol:
  - Training target: Capped at 125 cycles: y_train = min(raw_RUL, 125)
  - Primary validation: Capped target: y_val_capped = min(raw_RUL, 125)
  - Additional diagnostic validation: Uncapped target: y_val_uncapped = raw_RUL
  - Official test benchmark: Uncapped NASA ground-truth RUL (y_test_final)
  - Raw model predictions are evaluated (never automatically clipped to 125)
  - Per-engine validation metrics aggregated across all 142 validation engines
  - Subgroup breakdown: Unpadded (690 engines) vs Padded (17 engines)
  - Exclusive winner selection criterion: Minimum validation capped RMSE
  - Zero test-set leakage: Test data is completely isolated until after selection
=============================================================================
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import xgboost as xgb

# Project root path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from preprocessing.scaling_pipeline import (
    TIER1_CORE_SENSORS,
    TIER2_MULTI_CONDITION,
    TIER3_EXTENDED_ALL,
)

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
RUL_CAP = 125.0
RANDOM_STATE = 42

# Column index mappings from TIER3_EXTENDED_ALL
TIER_FEATURE_MAP = {
    "Tier1_Core14": {
        "features": TIER1_CORE_SENSORS,
        "indices": [TIER3_EXTENDED_ALL.index(c) for c in TIER1_CORE_SENSORS],
        "dim_final": 14,
        "dim_flattened": 420,
        "dim_summary": 70,
    },
    "Tier2_Operational17": {
        "features": TIER2_MULTI_CONDITION,
        "indices": [TIER3_EXTENDED_ALL.index(c) for c in TIER2_MULTI_CONDITION],
        "dim_final": 17,
        "dim_flattened": 510,
        "dim_summary": 85,
    },
    "Tier3_Extended24": {
        "features": TIER3_EXTENDED_ALL,
        "indices": [TIER3_EXTENDED_ALL.index(c) for c in TIER3_EXTENDED_ALL],
        "dim_final": 24,
        "dim_flattened": 720,
        "dim_summary": 120,
    },
}


# =============================================================================
# FEATURE TRANSFORMATIONS
# =============================================================================

def extract_final_observation(X: np.ndarray, col_indices: list[int]) -> np.ndarray:
    """Extract features from the final observation cycle (t = W-1)."""
    return X[:, -1, col_indices].astype(np.float32)


def extract_flattened_window(X: np.ndarray, col_indices: list[int]) -> np.ndarray:
    """Flatten the 30-cycle temporal window for selected features."""
    X_sub = X[:, :, col_indices]
    N, W, K = X_sub.shape
    return X_sub.reshape(N, W * K).astype(np.float32)


def compute_window_summary_statistics(X: np.ndarray, col_indices: list[int]) -> np.ndarray:
    """
    Compute 5 summary statistics across the 30-cycle window for selected features:
      1. Mean: (N, K)
      2. Standard Deviation: (N, K)
      3. Minimum: (N, K)
      4. Maximum: (N, K)
      5. Linear Trend Slope: (N, K)
    Returns:
      Array of shape (N, 5 * K)
    """
    X_sub = X[:, :, col_indices]
    N, W, K = X_sub.shape

    mean_val = np.mean(X_sub, axis=1)
    std_val = np.std(X_sub, axis=1)
    min_val = np.min(X_sub, axis=1)
    max_val = np.max(X_sub, axis=1)

    # Linear OLS slope over time indices t in [0, ..., W-1]
    t = np.arange(W, dtype=np.float32)
    t_mean = (W - 1) / 2.0
    t_diff = (t - t_mean).reshape(1, W, 1)
    t_ss = np.sum((t - t_mean) ** 2)  # constant 2247.5 for W=30

    x_diff = X_sub - mean_val.reshape(N, 1, K)
    slope_val = np.sum(t_diff * x_diff, axis=1) / t_ss

    summary_feats = np.concatenate([mean_val, std_val, min_val, max_val, slope_val], axis=1)
    return summary_feats.astype(np.float32)


# =============================================================================
# METRIC FUNCTIONS
# =============================================================================

def compute_nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute official NASA asymmetric scoring function with overflow protection."""
    d = y_pred - y_true
    d_clipped = np.clip(d, -500.0, 500.0)
    s = np.where(d_clipped < 0, np.exp(-d_clipped / 13.0) - 1.0, np.exp(d_clipped / 10.0) - 1.0)
    return float(np.sum(s))


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute standard regression metrics (RMSE, MAE, R2, NASA Score)."""
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))

    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    ss_res = float(np.sum(err ** 2))
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

    nasa_score = compute_nasa_score(y_true, y_pred)
    return {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "r2": round(r2, 4),
        "score": round(nasa_score, 2),
    }


# =============================================================================
# MANIFEST METADATA GENERATORS
# =============================================================================

def build_project_configuration() -> dict:
    """Build standardized project configuration metadata for Step 4.6 compliance."""
    return {
        "global_random_seed": 42,
        "target_definition": {
            "target_name": "Remaining Useful Life (RUL)",
            "target_unit": "operational_cycles",
            "training_target_capping": 125.0,
            "primary_validation_capping": 125.0,
            "diagnostic_validation_target": "uncapped",
            "official_test_target": "uncapped",
            "prediction_policy": "raw_unclipped",
            "scoring_error_clipping": [-500.0, 500.0],
            "scoring_error_clipping_purpose": "numerical_protection_against_overflow",
        },
        "temporal_window": {
            "window_size": 30,
            "stride": 1,
            "short_test_engine_padding": "left-edge replication of initial observation (cycle t=1 repeated backwards)",
        },
        "feature_tiers": {
            "Tier1_Core14": list(TIER1_CORE_SENSORS),
            "Tier2_Operational17": list(TIER2_MULTI_CONDITION),
            "Tier3_Extended24": list(TIER3_EXTENDED_ALL),
        },
        "summary_statistics_definition": {
            "window_length": 30,
            "statistics": ["mean", "std", "min", "max", "ols_linear_trend_slope"],
            "formula_notes": "ols_linear_trend_slope computed over discrete time index t in [0, 29] via closed-form OLS formula: sum((t - t_mean)*(x - x_mean)) / sum((t - t_mean)**2)",
        },
    }


def get_environment_metadata() -> dict:
    """Dynamically extract environment and package versions for Step 4.6 compliance."""
    import platform
    import scipy
    import sklearn
    import xgboost
    import torch
    import matplotlib

    return {
        "python_version": platform.python_version(),
        "python_build": platform.python_build()[0] + f" ({platform.python_build()[1]}) [{platform.python_compiler()}]",
        "platform": platform.platform(),
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "torch": torch.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "recorded_for": "NASA C-MAPSS Phase 4 Roadmap Step 4.6 Compliance",
    }


# =============================================================================
# MAIN XGBOOST BENCHMARK RUNNER
# =============================================================================

def main():
    print("=" * 80)
    print("PHASE 4 - STEP 4.2: TRAIN AND EVALUATE XGBOOST REGRESSION MODELS")
    print(f"Configurations: Exactly 40 (10 per subset x 4 subsets) | Random State: {RANDOM_STATE}")
    print("=" * 80)

    seq_dir = os.path.join(PROJECT_ROOT, "dataset", "processed", "sequences")
    reports_dir = os.path.join(PROJECT_ROOT, "reports", "models")
    xgboost_dir = os.path.join(PROJECT_ROOT, "models", "xgboost")
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(xgboost_dir, exist_ok=True)

    # Verification of feature dimension specifications
    print("\n[VERIFICATION] Programmatic Feature Dimension Check:")
    for tier_name, tier_info in TIER_FEATURE_MAP.items():
        k = len(tier_info["indices"])
        dim_f = tier_info["dim_final"]
        dim_fl = tier_info["dim_flattened"]
        dim_st = tier_info["dim_summary"]
        assert k == dim_f, f"Mismatch in {tier_name} final dimension"
        assert k * 30 == dim_fl, f"Mismatch in {tier_name} flattened dimension"
        assert k * 5 == dim_st, f"Mismatch in {tier_name} summary stats dimension"
        print(f"  - {tier_name:20s}: Final={dim_f:2d} | Flattened={dim_fl:3d} | SummaryStats={dim_st:3d} [OK]")

    performance_records = []
    per_engine_records = []
    manifest_records = {}

    config_idx = 0
    start_total_time = time.time()

    for subset in SUBSETS:
        print("\n" + "#" * 80)
        print(f"SUBSET: {subset}")
        print("#" * 80)

        # 1. Load sequence arrays and metadata
        X_train_raw = np.load(os.path.join(seq_dir, f"X_train_{subset}.npy"))
        y_train_raw = np.load(os.path.join(seq_dir, f"y_train_{subset}.npy"))

        X_val_raw = np.load(os.path.join(seq_dir, f"X_validation_{subset}.npy"))
        y_val_raw = np.load(os.path.join(seq_dir, f"y_validation_{subset}.npy"))
        df_val_meta = pd.read_csv(os.path.join(seq_dir, f"validation_metadata_{subset}.csv"))

        X_test_final_raw = np.load(os.path.join(seq_dir, f"X_test_final_{subset}.npy"))
        y_test_final_raw = np.load(os.path.join(seq_dir, f"y_test_final_{subset}.npy"))
        df_test_final_meta = pd.read_csv(os.path.join(seq_dir, f"test_final_metadata_{subset}.csv"))

        # Capped and uncapped targets
        y_train_capped = np.minimum(y_train_raw, RUL_CAP).astype(np.float32)
        y_val_capped = np.minimum(y_val_raw, RUL_CAP).astype(np.float32)
        y_val_uncapped = y_val_raw.astype(np.float32)
        y_test_final_uncapped = y_test_final_raw.astype(np.float32)

        val_units = df_val_meta["unit_number"].unique()
        num_val_engines = len(val_units)
        is_padded_test = df_test_final_meta["is_padded"].values

        # Find final window index for each validation engine
        last_val_indices = df_val_meta.groupby("unit_number").tail(1).index.values

        print(f"[{subset}] Data loaded: Train={X_train_raw.shape}, Val={X_val_raw.shape} ({num_val_engines} engines), Test={X_test_final_raw.shape}")

        # Precompute transformed representations for all tiers to optimize runtime
        print(f"[{subset}] Precomputing feature transformations...")
        transformed_data = {}
        for tier_name, tier_info in TIER_FEATURE_MAP.items():
            col_idx = tier_info["indices"]
            transformed_data[tier_name] = {
                "final": {
                    "train": extract_final_observation(X_train_raw, col_idx),
                    "val": extract_final_observation(X_val_raw, col_idx),
                    "test": extract_final_observation(X_test_final_raw, col_idx),
                },
                "flattened": {
                    "train": extract_flattened_window(X_train_raw, col_idx),
                    "val": extract_flattened_window(X_val_raw, col_idx),
                    "test": extract_flattened_window(X_test_final_raw, col_idx),
                },
                "summary": {
                    "train": compute_window_summary_statistics(X_train_raw, col_idx),
                    "val": compute_window_summary_statistics(X_val_raw, col_idx),
                    "test": compute_window_summary_statistics(X_test_final_raw, col_idx),
                },
            }

        # Stage 1: Build the 9 default configurations
        base_configs = []

        # Configs 1-3: Final Observation
        for tier in ["Tier1_Core14", "Tier2_Operational17", "Tier3_Extended24"]:
            base_configs.append({
                "model_family": "XGBoost",
                "model_variant": "XGBoost_FinalObservation",
                "repr_type": "final",
                "tier_name": tier,
                "feature_dim": TIER_FEATURE_MAP[tier]["dim_final"],
                "is_tuned": False,
                "get_model": lambda: xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.0,
                    reg_lambda=1.0,
                    objective="reg:squarederror",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            })

        # Configs 4-6: Flattened Window
        for tier in ["Tier1_Core14", "Tier2_Operational17", "Tier3_Extended24"]:
            base_configs.append({
                "model_family": "XGBoost",
                "model_variant": "XGBoost_FlattenedWindow",
                "repr_type": "flattened",
                "tier_name": tier,
                "feature_dim": TIER_FEATURE_MAP[tier]["dim_flattened"],
                "is_tuned": False,
                "get_model": lambda: xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.0,
                    reg_lambda=1.0,
                    objective="reg:squarederror",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            })

        # Configs 7-9: Summary Statistics
        for tier in ["Tier1_Core14", "Tier2_Operational17", "Tier3_Extended24"]:
            base_configs.append({
                "model_family": "XGBoost",
                "model_variant": "XGBoost_SummaryStats",
                "repr_type": "summary",
                "tier_name": tier,
                "feature_dim": TIER_FEATURE_MAP[tier]["dim_summary"],
                "is_tuned": False,
                "get_model": lambda: xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.0,
                    reg_lambda=1.0,
                    objective="reg:squarederror",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            })

        # Run Stage 1 (Configs 1-9) to identify best SummaryStats tier via validation capped RMSE
        subset_summary_results = {}
        for cfg in base_configs:
            config_idx += 1
            t_start = time.time()
            m_family = cfg["model_family"]
            m_variant = cfg["model_variant"]
            tier = cfg["tier_name"]
            f_dim = cfg["feature_dim"]
            repr_type = cfg["repr_type"]

            print(f"[{config_idx:02d}/40] [{subset}] Running: {m_variant} | {tier} (Dim={f_dim})...", end=" ", flush=True)

            X_tr = transformed_data[tier][repr_type]["train"]
            X_v = transformed_data[tier][repr_type]["val"]
            X_te = transformed_data[tier][repr_type]["test"]

            # Train standard XGBoost without early stopping
            model = cfg["get_model"]()
            model.fit(X_tr, y_train_capped, verbose=False)

            val_preds = model.predict(X_v)
            test_preds = model.predict(X_te)
            elapsed = time.time() - t_start

            # 1. Primary Validation (Capped Targets)
            m_val_capped = evaluate_predictions(y_val_capped, val_preds)

            # 2. Diagnostic Validation (Uncapped Targets)
            m_val_uncapped = evaluate_predictions(y_val_uncapped, val_preds)

            # Store summary stats validation performance for selecting tier of Config 10
            if m_variant == "XGBoost_SummaryStats":
                subset_summary_results[tier] = m_val_capped["rmse"]

            # 3. Per-Engine Validation Reporting
            engine_rmses = []
            engine_maes = []
            for u in val_units:
                u_mask = (df_val_meta["unit_number"] == u).values
                u_err = val_preds[u_mask] - y_val_capped[u_mask]
                e_rmse = float(np.sqrt(np.mean(u_err ** 2)))
                e_mae = float(np.mean(np.abs(u_err)))
                engine_rmses.append(e_rmse)
                engine_maes.append(e_mae)

                per_engine_records.append({
                    "subset": subset,
                    "model_variant": m_variant,
                    "feature_tier": tier,
                    "unit_number": int(u),
                    "windows_count": int(np.sum(u_mask)),
                    "per_engine_rmse_capped": round(e_rmse, 4),
                    "per_engine_mae_capped": round(e_mae, 4),
                })

            mean_engine_rmse = round(float(np.mean(engine_rmses)), 4)
            std_engine_rmse = round(float(np.std(engine_rmses)), 4)

            # Validation final-cycle RMSE
            val_final_preds = val_preds[last_val_indices]
            val_final_true = y_val_capped[last_val_indices]
            val_final_cycle_rmse = round(float(np.sqrt(np.mean((val_final_preds - val_final_true) ** 2))), 4)

            # 4. Official Test Benchmark (Uncapped NASA Ground Truth)
            m_test_full = evaluate_predictions(y_test_final_uncapped, test_preds)

            # Test subgroups
            unpadded_mask = (~is_padded_test)
            padded_mask = is_padded_test

            if np.any(unpadded_mask):
                m_test_unpadded = evaluate_predictions(
                    y_test_final_uncapped[unpadded_mask], test_preds[unpadded_mask]
                )
                test_unpadded_rmse = m_test_unpadded["rmse"]
                test_unpadded_score = m_test_unpadded["score"]
            else:
                test_unpadded_rmse = None
                test_unpadded_score = None

            if np.any(padded_mask):
                m_test_padded = evaluate_predictions(
                    y_test_final_uncapped[padded_mask], test_preds[padded_mask]
                )
                test_padded_rmse = m_test_padded["rmse"]
                test_padded_score = m_test_padded["score"]
            else:
                test_padded_rmse = "N/A"
                test_padded_score = "N/A"

            print(f"Val(Cap) RMSE={m_val_capped['rmse']:5.2f}, R2={m_val_capped['r2']:6.3f} | Test(Uncap) RMSE={m_test_full['rmse']:5.2f}, Score={m_test_full['score']:8.1f} ({elapsed:.1f}s)")

            # Master tabular record
            performance_records.append({
                "subset": subset,
                "model_family": m_family,
                "model_variant": m_variant,
                "feature_tier": tier,
                "feature_dim": f_dim,
                "val_capped_rmse": m_val_capped["rmse"],
                "val_capped_mae": m_val_capped["mae"],
                "val_capped_r2": m_val_capped["r2"],
                "val_capped_score": m_val_capped["score"],
                "val_uncapped_rmse": m_val_uncapped["rmse"],
                "val_uncapped_mae": m_val_uncapped["mae"],
                "val_uncapped_r2": m_val_uncapped["r2"],
                "val_uncapped_score": m_val_uncapped["score"],
                "val_engines_count": num_val_engines,
                "val_mean_engine_rmse": mean_engine_rmse,
                "val_std_engine_rmse": std_engine_rmse,
                "val_final_cycle_rmse": val_final_cycle_rmse,
                "test_uncapped_rmse": m_test_full["rmse"],
                "test_uncapped_mae": m_test_full["mae"],
                "test_uncapped_score": m_test_full["score"],
                "test_unpadded_rmse": test_unpadded_rmse,
                "test_unpadded_score": test_unpadded_score,
                "test_padded_rmse": test_padded_rmse,
                "test_padded_score": test_padded_score,
                "training_time_sec": round(elapsed, 2),
            })

            manifest_records[f"{subset}_{m_variant}_{tier}"] = {
                "subset": subset,
                "model_family": m_family,
                "model_variant": m_variant,
                "feature_tier": tier,
                "feature_dim": f_dim,
                "hyperparameters": {
                    "n_estimators": 100,
                    "max_depth": 6,
                    "learning_rate": 0.1,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "reg_alpha": 0.0,
                    "reg_lambda": 1.0,
                    "random_state": RANDOM_STATE,
                    "early_stopping": False,
                },
                "val_capped_metrics": m_val_capped,
                "val_uncapped_metrics": m_val_uncapped,
                "test_uncapped_metrics": m_test_full,
                "val_mean_engine_rmse": mean_engine_rmse,
                "val_final_cycle_rmse": val_final_cycle_rmse,
                "training_time_sec": round(elapsed, 2),
            }

        # Stage 2: Config 10 - Tuned / Regularized Summary Statistics
        # Pick the tier with lowest validation capped RMSE among SummaryStats
        best_tier_for_tuning = min(subset_summary_results, key=subset_summary_results.get)
        print(f"[{subset}] Best SummaryStats Tier for Tuned Model: {best_tier_for_tuning} (Val Capped RMSE: {subset_summary_results[best_tier_for_tuning]:.4f})")

        config_idx += 1
        t_start = time.time()
        m_family = "XGBoost"
        m_variant = "XGBoost_Tuned_SummaryStats"
        tier = best_tier_for_tuning
        f_dim = TIER_FEATURE_MAP[tier]["dim_summary"]

        print(f"[{config_idx:02d}/40] [{subset}] Running: {m_variant} | {tier} (Dim={f_dim}, EarlyStopping=20)...", end=" ", flush=True)

        X_tr = transformed_data[tier]["summary"]["train"]
        X_v = transformed_data[tier]["summary"]["val"]
        X_te = transformed_data[tier]["summary"]["test"]

        # Tuned XGBoost with shallow depth, shrinkage, L2 leaf weight regularization, and early stopping
        model_tuned = xgb.XGBRegressor(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=5.0,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            early_stopping_rounds=20,
            eval_metric="rmse",
        )

        # Early stopping evaluates strictly on validation capped targets (NEVER on test data)
        model_tuned.fit(X_tr, y_train_capped, eval_set=[(X_v, y_val_capped)], verbose=False)
        best_iter = int(model_tuned.best_iteration)

        val_preds = model_tuned.predict(X_v)
        test_preds = model_tuned.predict(X_te)
        elapsed = time.time() - t_start

        # 1. Primary Validation (Capped Targets)
        m_val_capped = evaluate_predictions(y_val_capped, val_preds)

        # 2. Diagnostic Validation (Uncapped Targets)
        m_val_uncapped = evaluate_predictions(y_val_uncapped, val_preds)

        # 3. Per-Engine Validation Reporting
        engine_rmses = []
        engine_maes = []
        for u in val_units:
            u_mask = (df_val_meta["unit_number"] == u).values
            u_err = val_preds[u_mask] - y_val_capped[u_mask]
            e_rmse = float(np.sqrt(np.mean(u_err ** 2)))
            e_mae = float(np.mean(np.abs(u_err)))
            engine_rmses.append(e_rmse)
            engine_maes.append(e_mae)

            per_engine_records.append({
                "subset": subset,
                "model_variant": m_variant,
                "feature_tier": tier,
                "unit_number": int(u),
                "windows_count": int(np.sum(u_mask)),
                "per_engine_rmse_capped": round(e_rmse, 4),
                "per_engine_mae_capped": round(e_mae, 4),
            })

        mean_engine_rmse = round(float(np.mean(engine_rmses)), 4)
        std_engine_rmse = round(float(np.std(engine_rmses)), 4)

        # Validation final-cycle RMSE
        val_final_preds = val_preds[last_val_indices]
        val_final_true = y_val_capped[last_val_indices]
        val_final_cycle_rmse = round(float(np.sqrt(np.mean((val_final_preds - val_final_true) ** 2))), 4)

        # 4. Official Test Benchmark (Uncapped NASA Ground Truth)
        m_test_full = evaluate_predictions(y_test_final_uncapped, test_preds)

        # Test subgroups
        unpadded_mask = (~is_padded_test)
        padded_mask = is_padded_test

        if np.any(unpadded_mask):
            m_test_unpadded = evaluate_predictions(
                y_test_final_uncapped[unpadded_mask], test_preds[unpadded_mask]
            )
            test_unpadded_rmse = m_test_unpadded["rmse"]
            test_unpadded_score = m_test_unpadded["score"]
        else:
            test_unpadded_rmse = None
            test_unpadded_score = None

        if np.any(padded_mask):
            m_test_padded = evaluate_predictions(
                y_test_final_uncapped[padded_mask], test_preds[padded_mask]
            )
            test_padded_rmse = m_test_padded["rmse"]
            test_padded_score = m_test_padded["score"]
        else:
            test_padded_rmse = "N/A"
            test_padded_score = "N/A"

        print(f"Val(Cap) RMSE={m_val_capped['rmse']:5.2f}, R2={m_val_capped['r2']:6.3f} (BestIter={best_iter}) | Test(Uncap) RMSE={m_test_full['rmse']:5.2f}, Score={m_test_full['score']:8.1f} ({elapsed:.1f}s)")

        # Master tabular record
        performance_records.append({
            "subset": subset,
            "model_family": m_family,
            "model_variant": m_variant,
            "feature_tier": tier,
            "feature_dim": f_dim,
            "val_capped_rmse": m_val_capped["rmse"],
            "val_capped_mae": m_val_capped["mae"],
            "val_capped_r2": m_val_capped["r2"],
            "val_capped_score": m_val_capped["score"],
            "val_uncapped_rmse": m_val_uncapped["rmse"],
            "val_uncapped_mae": m_val_uncapped["mae"],
            "val_uncapped_r2": m_val_uncapped["r2"],
            "val_uncapped_score": m_val_uncapped["score"],
            "val_engines_count": num_val_engines,
            "val_mean_engine_rmse": mean_engine_rmse,
            "val_std_engine_rmse": std_engine_rmse,
            "val_final_cycle_rmse": val_final_cycle_rmse,
            "test_uncapped_rmse": m_test_full["rmse"],
            "test_uncapped_mae": m_test_full["mae"],
            "test_uncapped_score": m_test_full["score"],
            "test_unpadded_rmse": test_unpadded_rmse,
            "test_unpadded_score": test_unpadded_score,
            "test_padded_rmse": test_padded_rmse,
            "test_padded_score": test_padded_score,
            "training_time_sec": round(elapsed, 2),
        })

        manifest_records[f"{subset}_{m_variant}_{tier}"] = {
            "subset": subset,
            "model_family": m_family,
            "model_variant": m_variant,
            "feature_tier": tier,
            "feature_dim": f_dim,
            "hyperparameters": {
                "n_estimators": 250,
                "max_depth": 4,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 5.0,
                "random_state": RANDOM_STATE,
                "early_stopping_rounds": 20,
                "best_iteration": best_iter,
            },
            "val_capped_metrics": m_val_capped,
            "val_uncapped_metrics": m_val_uncapped,
            "test_uncapped_metrics": m_test_full,
            "val_mean_engine_rmse": mean_engine_rmse,
            "val_final_cycle_rmse": val_final_cycle_rmse,
            "training_time_sec": round(elapsed, 2),
        }

    total_time = time.time() - start_total_time
    print("\n" + "=" * 80)
    print(f"ALL 40 XGBOOST CONFIGURATIONS COMPLETED IN {total_time:.1f} SECONDS")
    print("=" * 80)

    # Save master performance CSV
    df_perf = pd.DataFrame(performance_records)
    perf_csv_path = os.path.join(reports_dir, "xgboost_model_performance.csv")
    df_perf.to_csv(perf_csv_path, index=False)
    print(f"[SAVE] Master XGBoost performance table: {perf_csv_path}")

    # Save per-engine validation CSV
    df_engine = pd.DataFrame(per_engine_records)
    engine_csv_path = os.path.join(reports_dir, "xgboost_per_engine_validation.csv")
    df_engine.to_csv(engine_csv_path, index=False)
    print(f"[SAVE] Per-engine validation metrics: {engine_csv_path}")

    # Save manifest JSON with standardized Step 4.6 metadata
    manifest_path = os.path.join(xgboost_dir, "xgboost_manifest.json")
    final_manifest = {
        "_project_configuration": build_project_configuration(),
        **manifest_records,
        "_environment_metadata": get_environment_metadata(),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(final_manifest, f, indent=2)
    print(f"[SAVE] XGBoost execution manifest: {manifest_path}")

    # Display summary of best XGBoost model per subset (selected strictly by Validation Capped RMSE)
    print("\n" + "=" * 80)
    print("TOP PERFORMING XGBOOST MODEL PER SUBSET (SELECTED BY VALIDATION CAPPED RMSE):")
    print("=" * 80)
    for subset in SUBSETS:
        sub_df = df_perf[df_perf["subset"] == subset]
        best_row = sub_df.sort_values("val_capped_rmse").iloc[0]
        print(f"  [{subset}] Best: {best_row['model_variant']} ({best_row['feature_tier']})")
        print(f"         Val Capped RMSE: {best_row['val_capped_rmse']:.4f} | Official Test RMSE: {best_row['test_uncapped_rmse']:.4f} | Test Score: {best_row['test_uncapped_score']:.2f}")


if __name__ == "__main__":
    main()
