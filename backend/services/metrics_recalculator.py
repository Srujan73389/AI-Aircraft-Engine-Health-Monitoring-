"""
Metrics Recalculator Service
Phase 9 - Step 9.4: "Recalculate evaluation metrics from saved predictions and confirm the reported values"
Roadmap Reference: Master Project Roadmap PDF, Section 10, Step 9.4 & Section 1, Page 2

Provides rigorous metric recalculation utilities:
1. Root Mean Squared Error (RMSE)
2. Mean Absolute Error (MAE)
3. NASA Operational Asymmetric Scoring Function S:
   d_i = y_hat_i - y_true_i
   s_i = exp(-d_i / 13) - 1   if d_i < 0 (early prediction)
   s_i = exp(d_i / 10) - 1    if d_i >= 0 (late prediction)
   S = sum(s_i)
4. Comprehensive subset evaluation across XGBoost, Bi-LSTM, and Consensus ensemble.
"""

import os
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from digital_twin.trajectory_loader import TrajectoryLoader
from digital_twin.model_inference import PrognosticsModelManager
from digital_twin.twin_simulator import ORDERED_SCALER_FEATURES, CORE_INDICES_IN_24
from preprocessing.scaling_pipeline import CMAPSSScalingPipeline


DEFAULT_DATA_DIR = os.path.join("dataset", "raw", "C-MAPSS")
DEFAULT_SCALERS_DIR = os.path.join("models", "scalers")
DEFAULT_CHECKPOINTS_DIR = os.path.join("models", "checkpoints")

SUBSET_TEST_COUNTS = {
    "FD001": 100,
    "FD002": 259,
    "FD003": 100,
    "FD004": 248,
}

REPORTED_BENCHMARK_RMSE = {
    "FD001": {"xgboost": 13.4770, "lstm": 14.8409},
    "FD002": {"xgboost": 30.0483, "lstm": 29.0728},
    "FD003": {"xgboost": 14.2476, "lstm": 17.4228},
    "FD004": {"xgboost": 29.7334, "lstm": 30.3794},
}


def calculate_rmse(
    y_true: Union[List[float], np.ndarray],
    y_pred: Union[List[float], np.ndarray],
) -> float:
    """Computes Root Mean Squared Error between true and predicted arrays."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if yt.shape != yp.shape:
        raise ValueError(f"Shape mismatch: y_true {yt.shape} vs y_pred {yp.shape}")
    if len(yt) == 0:
        raise ValueError("Cannot calculate RMSE of empty arrays.")
    return float(np.sqrt(np.mean((yp - yt) ** 2)))


def calculate_mae(
    y_true: Union[List[float], np.ndarray],
    y_pred: Union[List[float], np.ndarray],
) -> float:
    """Computes Mean Absolute Error between true and predicted arrays."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if yt.shape != yp.shape:
        raise ValueError(f"Shape mismatch: y_true {yt.shape} vs y_pred {yp.shape}")
    if len(yt) == 0:
        raise ValueError("Cannot calculate MAE of empty arrays.")
    return float(np.mean(np.abs(yp - yt)))


def calculate_nasa_score(
    y_true: Union[List[float], np.ndarray],
    y_pred: Union[List[float], np.ndarray],
    clip_bounds: Tuple[float, float] = (-500.0, 500.0),
) -> float:
    """
    Computes the official NASA C-MAPSS Asymmetric Scoring Function S.
    
    Penalizes late predictions (d >= 0) more severely than early predictions (d < 0):
        d_i = y_pred_i - y_true_i
        s_i = exp(-d_i / 13) - 1 for d_i < 0
        s_i = exp(d_i / 10) - 1  for d_i >= 0
        S = sum(s_i)
    
    Residuals are clipped to clip_bounds (default [-500, 500]) for numerical safety against float overflow.
    """
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if yt.shape != yp.shape:
        raise ValueError(f"Shape mismatch: y_true {yt.shape} vs y_pred {yp.shape}")
    if len(yt) == 0:
        raise ValueError("Cannot calculate NASA Score of empty arrays.")

    d = yp - yt
    d_clipped = np.clip(d, clip_bounds[0], clip_bounds[1])

    scores = np.where(
        d_clipped < 0.0,
        np.exp(-d_clipped / 13.0) - 1.0,
        np.exp(d_clipped / 10.0) - 1.0,
    )
    return float(np.sum(scores))


def evaluate_subset_test_metrics(
    subset: str,
    data_dir: Optional[str] = None,
    scalers_dir: Optional[str] = None,
    checkpoints_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Executes batch inference across all test engines for a given subset at their final cycle T_i,
    and recalculates official benchmark metrics (RMSE, MAE, NASA Score) for XGBoost, Bi-LSTM,
    and Consensus ensemble.
    """
    sub_upper = subset.upper()
    if sub_upper not in SUBSET_TEST_COUNTS:
        raise ValueError(f"Invalid subset '{subset}'. Supported: {list(SUBSET_TEST_COUNTS.keys())}.")

    base_data = data_dir or DEFAULT_DATA_DIR
    base_scalers = scalers_dir or DEFAULT_SCALERS_DIR
    base_checkpoints = checkpoints_dir or DEFAULT_CHECKPOINTS_DIR

    # 1. Ingest true labels
    rul_file = os.path.join(base_data, f"RUL_{sub_upper}.txt")
    if not os.path.exists(rul_file):
        raise FileNotFoundError(f"Missing ground-truth RUL file: {rul_file}")

    with open(rul_file, "r", encoding="utf-8") as f:
        y_true = np.array([float(line.strip()) for line in f if line.strip()], dtype=np.float64)

    expected_len = SUBSET_TEST_COUNTS[sub_upper]
    if len(y_true) != expected_len:
        raise ValueError(f"Expected {expected_len} ground-truth labels for {sub_upper}, got {len(y_true)}.")

    # 2. Setup loader, scaling pipeline, and prognostics manager
    loader = TrajectoryLoader(data_dir=base_data)
    scaler_path = os.path.join(base_scalers, f"scaler_{sub_upper}_tier3.pkl")
    scaling_pipe = CMAPSSScalingPipeline.load(scaler_path)
    mgr = PrognosticsModelManager(sub_upper, checkpoint_dir=base_checkpoints)

    # 3. Batch inference across all test engines
    xgb_preds = []
    lstm_preds = []

    for u in range(1, expected_len + 1):
        df = loader.load_engine_trajectory(sub_upper, u, split="test")
        raw_24 = df[ORDERED_SCALER_FEATURES].values

        if len(raw_24) < 30:
            pad = np.repeat(raw_24[:1], 30 - len(raw_24), axis=0)
            raw_30 = np.vstack([pad, raw_24])
        else:
            raw_30 = raw_24[-30:]

        norm_30 = scaling_pipe.scaler.transform(raw_30)

        if sub_upper == "FD001":
            seq_tensor = norm_30[:, CORE_INDICES_IN_24].astype(np.float32)
        else:
            seq_tensor = norm_30.astype(np.float32)

        xp, lp, _ = mgr.predict_prognostics(seq_tensor)
        xgb_preds.append(float(xp))
        lstm_preds.append(float(lp))

    xgb_arr = np.array(xgb_preds, dtype=np.float64)
    lstm_arr = np.array(lstm_preds, dtype=np.float64)
    cons_arr = (xgb_arr + lstm_arr) / 2.0

    # 4. Metric recalculation
    metrics = {
        "subset": sub_upper,
        "num_engines": expected_len,
        "xgboost": {
            "rmse": round(calculate_rmse(y_true, xgb_arr), 4),
            "mae": round(calculate_mae(y_true, xgb_arr), 4),
            "nasa_score": round(calculate_nasa_score(y_true, xgb_arr), 2),
            "reported_benchmark_rmse": REPORTED_BENCHMARK_RMSE[sub_upper]["xgboost"],
        },
        "lstm": {
            "rmse": round(calculate_rmse(y_true, lstm_arr), 4),
            "mae": round(calculate_mae(y_true, lstm_arr), 4),
            "nasa_score": round(calculate_nasa_score(y_true, lstm_arr), 2),
            "reported_benchmark_rmse": REPORTED_BENCHMARK_RMSE[sub_upper]["lstm"],
        },
        "consensus": {
            "rmse": round(calculate_rmse(y_true, cons_arr), 4),
            "mae": round(calculate_mae(y_true, cons_arr), 4),
            "nasa_score": round(calculate_nasa_score(y_true, cons_arr), 2),
        },
    }

    return metrics
