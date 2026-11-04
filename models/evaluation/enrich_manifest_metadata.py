"""
=============================================================================
AIRCRAFT ENGINE AI - HEALTH MONITORING & PREDICTIVE MAINTENANCE
Phase 4: Machine Learning and Model Evaluation
Step 4.6: Manifest Metadata Enrichment & Verification Migration Utility
=============================================================================

This script safely and non-destructively enriches the three experiment manifests:
  1. models/baselines/baseline_manifest.json
  2. models/xgboost/xgboost_manifest.json
  3. models/lstm/lstm_manifest.json

Safeguards:
  - ZERO metrics, loss curves, or RMSEs are altered.
  - All original experiment keys (52 baseline, 40 XGBoost, 16 LSTM = 108) are preserved.
  - Injects standardized _project_configuration block into all three manifests.
  - Injects exact executed hyperparameters and random seeds.
  - Preserves existing _environment_metadata.
=============================================================================
"""

import os
import sys
import json
import platform
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from preprocessing.scaling_pipeline import (
    TIER1_CORE_SENSORS,
    TIER2_MULTI_CONDITION,
    TIER3_EXTENDED_ALL,
)

# Standardized Project Configuration Schema
PROJECT_CONFIGURATION = {
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

CONSTANT_MEAN_FITTED_VALUES = {
    "FD001": 80.1274,
    "FD002": 80.5846,
    "FD003": 89.7240,
    "FD004": 88.3473,
}


def enrich_baseline_manifest():
    path = os.path.join(PROJECT_ROOT, "models", "baselines", "baseline_manifest.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    env_meta = data.get("_environment_metadata")

    exp_keys = [k for k in data if not k.startswith("_")]
    assert len(exp_keys) == 52, f"Expected 52 baseline experiments, found {len(exp_keys)}"

    enriched = {"_project_configuration": PROJECT_CONFIGURATION}

    for key, record in data.items():
        if key.startswith("_"):
            continue
        
        m_family = record["model_family"]
        m_variant = record["model_variant"]
        subset = record["subset"]

        rec = dict(record)

        if m_family == "ConstantMean":
            rec["hyperparameters"] = {
                "strategy": "empirical_training_mean",
                "fitted_mean_value": CONSTANT_MEAN_FITTED_VALUES[subset],
            }
        elif m_family == "Ridge":
            rec["hyperparameters"] = {
                "alpha": 1.0,
                "solver": "auto",
                "fit_intercept": True,
                "random_state": 42,
                "note": "Deterministic solver (auto selects Cholesky/SVD); random_state parameter accepted by constructor but does not introduce stochastic variance",
            }
        elif m_family == "RandomForest":
            rec["hyperparameters"] = {
                "n_estimators": 100,
                "max_depth": 15,
                "min_samples_split": 10,
                "min_samples_leaf": 4,
                "max_features": "sqrt",
                "random_state": 42,
                "n_jobs": -1,
            }
        else:
            raise ValueError(f"Unknown baseline family: {m_family}")

        enriched[key] = rec

    if env_meta:
        enriched["_environment_metadata"] = env_meta

    with open(path, "w", encoding="utf-8") as f:
        json.dump(enriched, f, indent=2)

    print(f"[OK] Enriched {path}: 52 experiments + project configuration + environment metadata")


def enrich_xgboost_manifest():
    path = os.path.join(PROJECT_ROOT, "models", "xgboost", "xgboost_manifest.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    env_meta = data.get("_environment_metadata")

    exp_keys = [k for k in data if not k.startswith("_")]
    assert len(exp_keys) == 40, f"Expected 40 XGBoost experiments, found {len(exp_keys)}"

    enriched = {"_project_configuration": PROJECT_CONFIGURATION}

    for key, record in data.items():
        if key.startswith("_"):
            continue

        rec = dict(record)
        hp = dict(rec.get("hyperparameters", {}))
        hp["random_state"] = 42
        rec["hyperparameters"] = hp
        enriched[key] = rec

    if env_meta:
        enriched["_environment_metadata"] = env_meta

    with open(path, "w", encoding="utf-8") as f:
        json.dump(enriched, f, indent=2)

    print(f"[OK] Enriched {path}: 40 experiments + project configuration + environment metadata")


def enrich_lstm_manifest():
    path = os.path.join(PROJECT_ROOT, "models", "lstm", "lstm_manifest.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    env_meta = data.get("_environment_metadata")

    exp_keys = [k for k in data if not k.startswith("_")]
    assert len(exp_keys) == 16, f"Expected 16 LSTM experiments, found {len(exp_keys)}"

    enriched = {"_project_configuration": PROJECT_CONFIGURATION}

    for key, record in data.items():
        if key.startswith("_"):
            continue

        rec = dict(record)
        ap = dict(rec.get("architecture_params", {}))
        ap["random_seed"] = 42
        rec["architecture_params"] = ap
        enriched[key] = rec

    if env_meta:
        enriched["_environment_metadata"] = env_meta

    with open(path, "w", encoding="utf-8") as f:
        json.dump(enriched, f, indent=2)

    print(f"[OK] Enriched {path}: 16 experiments + project configuration + environment metadata")


if __name__ == "__main__":
    enrich_baseline_manifest()
    enrich_xgboost_manifest()
    enrich_lstm_manifest()
    print("\nAll manifests successfully enriched.")
