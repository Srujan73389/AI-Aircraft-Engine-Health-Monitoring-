"""
Ground Truth RUL Comparator Service
Phase 9 - Step 9.3: "Compare RUL predictions with the corresponding ground-truth labels"
Roadmap Reference: Master Project Roadmap PDF, Section 10, Step 9.3 & Section 1, Page 2

Provides utilities to:
1. Ingest true remaining useful life from official NASA C-MAPSS label files (RUL_FD00*.txt).
2. Compute ground-truth RUL for any trajectory cycle t using: y*(t) = y*_final + (T_i - t).
3. Execute dual-model inference (XGBoost & Bi-LSTM champions) and compute comparative metrics:
   - Point predictions (y_hat_xgb, y_hat_lstm)
   - Consensus RUL and inter-model epistemic spread
   - Absolute prediction errors (e_xgb, e_lstm, e_consensus)
   - Bounded error envelopes and confidence scores
"""

import os
from typing import Any, Dict, Optional, Tuple
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


def get_ground_truth_filepath(subset: str, data_dir: Optional[str] = None) -> str:
    """Resolves the path to the ground-truth RUL label file for a subset."""
    base_dir = data_dir or DEFAULT_DATA_DIR
    sub_upper = subset.upper()
    if sub_upper not in SUBSET_TEST_COUNTS:
        raise ValueError(f"Invalid subset '{subset}'. Supported: {list(SUBSET_TEST_COUNTS.keys())}.")
    filename = f"RUL_{sub_upper}.txt"
    filepath = os.path.join(base_dir, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Ground-truth RUL file not found at: {filepath}")
    return filepath


def get_ground_truth_rul(
    subset: str,
    unit: int,
    cycle: Optional[int] = None,
    data_dir: Optional[str] = None,
) -> float:
    """
    Retrieves the authentic ground-truth RUL for a test engine at cycle t.
    
    Formula:
        y*(t) = y*_final + (T_i - t)
    where y*_final is the label from RUL_FD00*.txt, T_i is the total recorded cycles for unit i.
    If cycle is None, returns y*_final.
    """
    sub_upper = subset.upper()
    filepath = get_ground_truth_filepath(sub_upper, data_dir)
    max_units = SUBSET_TEST_COUNTS[sub_upper]

    if unit < 1 or unit > max_units:
        raise ValueError(f"Invalid test engine unit {unit} for subset {sub_upper}. Valid range: 1 to {max_units}.")

    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if len(lines) != max_units:
        raise ValueError(
            f"Ground-truth file {filepath} contains {len(lines)} records; expected {max_units} for {sub_upper}."
        )

    final_rul = float(lines[unit - 1])

    if cycle is None:
        return final_rul

    # Retrieve total recorded cycles T_i for test engine
    loader = TrajectoryLoader(data_dir=data_dir or DEFAULT_DATA_DIR)
    df = loader.load_engine_trajectory(sub_upper, unit, split="test")
    total_cycles = int(df["time_cycles"].max())

    if cycle < 1 or cycle > total_cycles:
        raise ValueError(
            f"Cycle {cycle} out of bounds for {sub_upper} test engine {unit} (1 to {total_cycles} cycles)."
        )

    # Linear degradation truth: y*(t) = y*_final + (T_i - t)
    cycle_rul = final_rul + float(total_cycles - cycle)
    return cycle_rul


def extract_30cycle_sequence_tensor(
    subset: str,
    unit: int,
    cycle: int,
    data_dir: Optional[str] = None,
    scalers_dir: Optional[str] = None,
) -> np.ndarray:
    """
    Extracts and normalizes the 30-cycle temporal window up to cycle t for model inference.
    Applies left-edge replication padding if t < 30 in accordance with Phase 3 & Phase 5 specifications.
    """
    sub_upper = subset.upper()
    loader = TrajectoryLoader(data_dir=data_dir or DEFAULT_DATA_DIR)
    df = loader.load_engine_trajectory(sub_upper, unit, split="test")

    # Slice trajectory up to cycle
    df_upto = df[df["time_cycles"] <= cycle]
    if len(df_upto) == 0:
        raise ValueError(f"No telemetry found up to cycle {cycle} for {sub_upper} unit {unit}.")

    raw_24 = df_upto[ORDERED_SCALER_FEATURES].values

    # Left-edge replication padding if history < 30
    if len(raw_24) < 30:
        pad_len = 30 - len(raw_24)
        pad = np.repeat(raw_24[:1], pad_len, axis=0)
        raw_30 = np.vstack([pad, raw_24])
    else:
        raw_30 = raw_24[-30:]

    # Scale using Phase 3 fitted scaler
    s_dir = scalers_dir or DEFAULT_SCALERS_DIR
    scaler_path = os.path.join(s_dir, f"scaler_{sub_upper}_tier3.pkl")
    scaling_pipe = CMAPSSScalingPipeline.load(scaler_path)
    norm_30 = scaling_pipe.scaler.transform(raw_30)

    # Subset channel selection
    if sub_upper == "FD001":
        seq_tensor = norm_30[:, CORE_INDICES_IN_24].astype(np.float32)
    else:
        seq_tensor = norm_30.astype(np.float32)

    return seq_tensor


def compare_engine_rul(
    subset: str,
    unit: int,
    cycle: Optional[int] = None,
    data_dir: Optional[str] = None,
    scalers_dir: Optional[str] = None,
    checkpoints_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Executes live dual-model inference and compares predictions against authentic ground truth.
    
    Returns:
        Structured dictionary containing:
        - metadata: subset, unit, cycle, total_cycles
        - predictions: predicted_rul_xgboost, predicted_rul_lstm, consensus_rul
        - ground_truth: ground_truth_rul, final_cycle_ground_truth
        - errors: error_xgboost, error_lstm, error_consensus
        - uncertainty: prognostic_spread, confidence_score
    """
    sub_upper = subset.upper()
    loader = TrajectoryLoader(data_dir=data_dir or DEFAULT_DATA_DIR)
    df = loader.load_engine_trajectory(sub_upper, unit, split="test")
    total_cycles = int(df["time_cycles"].max())
    eval_cycle = total_cycles if cycle is None else cycle

    # 1. Ground truth
    gt_rul = get_ground_truth_rul(sub_upper, unit, eval_cycle, data_dir=data_dir)
    final_gt_rul = get_ground_truth_rul(sub_upper, unit, None, data_dir=data_dir)

    # 2. Extract 30-cycle temporal tensor
    seq_tensor = extract_30cycle_sequence_tensor(
        sub_upper, unit, eval_cycle, data_dir=data_dir, scalers_dir=scalers_dir
    )

    # 3. Dual-model inference
    c_dir = checkpoints_dir or DEFAULT_CHECKPOINTS_DIR
    mgr = PrognosticsModelManager(sub_upper, checkpoint_dir=c_dir)
    xgb_pred, lstm_pred, spread = mgr.predict_prognostics(seq_tensor)

    # 4. Consensus & metrics
    consensus_pred = float((xgb_pred + lstm_pred) / 2.0)
    confidence = float(max(0.0, min(1.0, 1.0 - (spread / max(50.0, consensus_pred)))))

    # Absolute errors
    err_xgb = abs(xgb_pred - gt_rul)
    err_lstm = abs(lstm_pred - gt_rul)
    err_consensus = abs(consensus_pred - gt_rul)

    return {
        "subset": sub_upper,
        "unit": unit,
        "cycle": eval_cycle,
        "total_cycles": total_cycles,
        "ground_truth_rul": round(gt_rul, 2),
        "final_cycle_ground_truth": round(final_gt_rul, 2),
        "predicted_rul_xgboost": round(float(xgb_pred), 2),
        "predicted_rul_lstm": round(float(lstm_pred), 2),
        "consensus_rul": round(consensus_pred, 2),
        "prognostic_spread": round(float(spread), 2),
        "confidence_score": round(confidence, 4),
        "error_xgboost": round(float(err_xgb), 2),
        "error_lstm": round(float(err_lstm), 2),
        "error_consensus": round(float(err_consensus), 2),
    }
