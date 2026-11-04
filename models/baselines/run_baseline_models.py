"""
=============================================================================
AIRCRAFT ENGINE AI - HEALTH MONITORING & PREDICTIVE MAINTENANCE
Phase 4: Machine Learning and Model Evaluation
Step 4.2: Implement and Evaluate Baseline Models
=============================================================================

This script implements and quantitatively evaluates non-deep-learning baseline
models across all four NASA C-MAPSS subsets (FD001, FD002, FD003, FD004)
and across three feature tiers (Tier 1, Tier 2, Tier 3).

Total Configurations: Exactly 52 distinct model configurations
(13 per subset x 4 subsets):
  1. Constant Training-Mean Regressor (1 per subset = 4)
  2. Ridge Regression - Final Observation (3 tiers x 4 subsets = 12)
  3. Ridge Regression - Flattened Window (3 tiers x 4 subsets = 12)
  4. Random Forest - Flattened Window (3 tiers x 4 subsets = 12)
  5. Random Forest - Summary Statistics (3 tiers x 4 subsets = 12)

Strict Protocol:
  - Training target: Capped at 125 cycles: y_train = min(raw_RUL, 125)
  - Primary validation: Capped target: y_val_capped = min(raw_RUL, 125)
  - Additional diagnostic validation: Uncapped target: y_val_uncapped = raw_RUL
  - Official test benchmark: Uncapped NASA ground-truth RUL (y_test_final)
  - Raw model predictions are evaluated (never automatically clipped to 125)
  - Per-engine validation metrics aggregated across all 142 validation engines
  - Subgroup breakdown: Unpadded (690 engines) vs Padded (17 engines)
=============================================================================
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

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
    # X shape: (N, 30, 24) -> slice: (N, len(col_indices))
    return X[:, -1, col_indices].astype(np.float32)


def extract_flattened_window(X: np.ndarray, col_indices: list[int]) -> np.ndarray:
    """Flatten the 30-cycle temporal window for selected features."""
    # X shape: (N, 30, 24) -> slice: (N, 30, K) -> reshape: (N, 30 * K)
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
# BASELINE MODEL CLASSES
# =============================================================================

class ConstantMeanRegressor:
    """Baseline predictor that predicts the training-target scalar mean."""
    def __init__(self):
        self.mean_value_ = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.mean_value_ = float(np.mean(y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.mean_value_, dtype=np.float32)


# =============================================================================
# MAIN BENCHMARK RUNNER
# =============================================================================

def main():
    print("=" * 80)
    print("PHASE 4 - STEP 4.2: IMPLEMENT AND EVALUATE BASELINE MODELS")
    print(f"Configurations: Exactly 52 (13 per subset x 4 subsets) | Random State: {RANDOM_STATE}")
    print("=" * 80)

    seq_dir = os.path.join(PROJECT_ROOT, "dataset", "processed", "sequences")
    reports_dir = os.path.join(PROJECT_ROOT, "reports", "models")
    baselines_dir = os.path.join(PROJECT_ROOT, "models", "baselines")
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(baselines_dir, exist_ok=True)

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

        # Define the 13 configurations for this subset
        configs_to_run = []

        # Config 1: Constant Mean
        configs_to_run.append({
            "model_family": "ConstantMean",
            "model_variant": "TrainingMean",
            "repr_type": "none",
            "tier_name": "Tier3_Extended24",  # Evaluated on full data
            "feature_dim": 0,
            "get_model": lambda: ConstantMeanRegressor(),
        })

        # Configs 2-4: Ridge - Final Observation
        for tier in ["Tier1_Core14", "Tier2_Operational17", "Tier3_Extended24"]:
            configs_to_run.append({
                "model_family": "Ridge",
                "model_variant": "Ridge_FinalObservation",
                "repr_type": "final",
                "tier_name": tier,
                "feature_dim": TIER_FEATURE_MAP[tier]["dim_final"],
                "get_model": lambda: Ridge(alpha=1.0, random_state=RANDOM_STATE),
            })

        # Configs 5-7: Ridge - Flattened Window
        for tier in ["Tier1_Core14", "Tier2_Operational17", "Tier3_Extended24"]:
            configs_to_run.append({
                "model_family": "Ridge",
                "model_variant": "Ridge_FlattenedWindow",
                "repr_type": "flattened",
                "tier_name": tier,
                "feature_dim": TIER_FEATURE_MAP[tier]["dim_flattened"],
                "get_model": lambda: Ridge(alpha=1.0, random_state=RANDOM_STATE),
            })

        # Configs 8-10: Random Forest - Flattened Window
        for tier in ["Tier1_Core14", "Tier2_Operational17", "Tier3_Extended24"]:
            configs_to_run.append({
                "model_family": "RandomForest",
                "model_variant": "RandomForest_FlattenedWindow",
                "repr_type": "flattened",
                "tier_name": tier,
                "feature_dim": TIER_FEATURE_MAP[tier]["dim_flattened"],
                "get_model": lambda: RandomForestRegressor(
                    n_estimators=100,
                    max_depth=15,
                    min_samples_split=10,
                    min_samples_leaf=4,
                    max_features="sqrt",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            })

        # Configs 11-13: Random Forest - Summary Statistics
        for tier in ["Tier1_Core14", "Tier2_Operational17", "Tier3_Extended24"]:
            configs_to_run.append({
                "model_family": "RandomForest",
                "model_variant": "RandomForest_SummaryStats",
                "repr_type": "summary",
                "tier_name": tier,
                "feature_dim": TIER_FEATURE_MAP[tier]["dim_summary"],
                "get_model": lambda: RandomForestRegressor(
                    n_estimators=100,
                    max_depth=15,
                    min_samples_split=10,
                    min_samples_leaf=4,
                    max_features="sqrt",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            })

        assert len(configs_to_run) == 13, f"Expected 13 configs for {subset}, got {len(configs_to_run)}"

        # Execute configurations
        for cfg in configs_to_run:
            config_idx += 1
            t_start = time.time()
            m_family = cfg["model_family"]
            m_variant = cfg["model_variant"]
            tier = cfg["tier_name"]
            f_dim = cfg["feature_dim"]
            repr_type = cfg["repr_type"]

            print(f"[{config_idx:02d}/52] [{subset}] Running: {m_variant} | {tier} (Dim={f_dim})...", end=" ", flush=True)

            # Extract X matrices
            if repr_type == "none":
                X_tr = np.zeros((len(y_train_capped), 1), dtype=np.float32)
                X_v = np.zeros((len(y_val_capped), 1), dtype=np.float32)
                X_te = np.zeros((len(y_test_final_uncapped), 1), dtype=np.float32)
            else:
                X_tr = transformed_data[tier][repr_type]["train"]
                X_v = transformed_data[tier][repr_type]["val"]
                X_te = transformed_data[tier][repr_type]["test"]

            # Train model
            model = cfg["get_model"]()
            model.fit(X_tr, y_train_capped)

            # Predictions
            val_preds = model.predict(X_v)
            test_preds = model.predict(X_te)
            elapsed = time.time() - t_start

            # 1. Primary Validation (Capped Targets)
            m_val_capped = evaluate_predictions(y_val_capped, val_preds)

            # 2. Additional Diagnostic Validation (Uncapped Targets)
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

                # Record per-engine detail
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

            # Validation final-cycle RMSE (only the last observed window of each validation engine)
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

            # Summary print
            print(f"Val(Cap) RMSE={m_val_capped['rmse']:5.2f}, R2={m_val_capped['r2']:6.3f} | Test(Uncap) RMSE={m_test_full['rmse']:5.2f}, Score={m_test_full['score']:8.1f} ({elapsed:.1f}s)")

            # Master tabular record
            performance_records.append({
                "subset": subset,
                "model_family": m_family,
                "model_variant": m_variant,
                "feature_tier": tier,
                "feature_dim": f_dim,
                # Primary Validation (Capped Target)
                "val_capped_rmse": m_val_capped["rmse"],
                "val_capped_mae": m_val_capped["mae"],
                "val_capped_r2": m_val_capped["r2"],
                "val_capped_score": m_val_capped["score"],
                # Additional Diagnostic Validation (Uncapped Target)
                "val_uncapped_rmse": m_val_uncapped["rmse"],
                "val_uncapped_mae": m_val_uncapped["mae"],
                "val_uncapped_r2": m_val_uncapped["r2"],
                "val_uncapped_score": m_val_uncapped["score"],
                # Per-Engine Validation Aggregations
                "val_engines_count": num_val_engines,
                "val_mean_engine_rmse": mean_engine_rmse,
                "val_std_engine_rmse": std_engine_rmse,
                "val_final_cycle_rmse": val_final_cycle_rmse,
                # Official Test Benchmark (Uncapped NASA GT)
                "test_uncapped_rmse": m_test_full["rmse"],
                "test_uncapped_mae": m_test_full["mae"],
                "test_uncapped_score": m_test_full["score"],
                # Test Subgroups
                "test_unpadded_rmse": test_unpadded_rmse,
                "test_unpadded_score": test_unpadded_score,
                "test_padded_rmse": test_padded_rmse,
                "test_padded_score": test_padded_score,
                "training_time_sec": round(elapsed, 2),
            })

            # Manifest configuration record
            manifest_records[f"{subset}_{m_variant}_{tier}"] = {
                "subset": subset,
                "model_family": m_family,
                "model_variant": m_variant,
                "feature_tier": tier,
                "feature_dim": f_dim,
                "val_capped_metrics": m_val_capped,
                "val_uncapped_metrics": m_val_uncapped,
                "test_uncapped_metrics": m_test_full,
                "val_mean_engine_rmse": mean_engine_rmse,
                "val_final_cycle_rmse": val_final_cycle_rmse,
                "training_time_sec": round(elapsed, 2),
            }

    total_time = time.time() - start_total_time
    print("\n" + "=" * 80)
    print(f"ALL 52 BASELINE CONFIGURATIONS COMPLETED IN {total_time:.1f} SECONDS")
    print("=" * 80)

    # Save master performance CSV
    df_perf = pd.DataFrame(performance_records)
    perf_csv_path = os.path.join(reports_dir, "baseline_model_performance.csv")
    df_perf.to_csv(perf_csv_path, index=False)
    print(f"[SAVE] Master baseline performance table: {perf_csv_path}")

    # Save per-engine validation CSV
    df_engine = pd.DataFrame(per_engine_records)
    engine_csv_path = os.path.join(reports_dir, "baseline_per_engine_validation.csv")
    df_engine.to_csv(engine_csv_path, index=False)
    print(f"[SAVE] Per-engine validation metrics: {engine_csv_path}")

    # Save baseline manifest JSON
    manifest_path = os.path.join(baselines_dir, "baseline_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest_records, f, indent=2)
    print(f"[SAVE] Baseline execution manifest: {manifest_path}")

    # Display summary of best models per subset (selected strictly by Validation Capped RMSE)
    print("\n" + "=" * 80)
    print("TOP PERFORMING BASELINE MODEL PER SUBSET (SELECTED BY VALIDATION CAPPED RMSE):")
    print("=" * 80)
    for subset in SUBSETS:
        sub_df = df_perf[df_perf["subset"] == subset]
        best_row = sub_df.sort_values("val_capped_rmse").iloc[0]
        print(f"  [{subset}] Best: {best_row['model_variant']} ({best_row['feature_tier']})")
        print(f"         Val Capped RMSE: {best_row['val_capped_rmse']:.4f} | Official Test RMSE: {best_row['test_uncapped_rmse']:.4f} | Test Score: {best_row['test_uncapped_score']:.2f}")


if __name__ == "__main__":
    main()
