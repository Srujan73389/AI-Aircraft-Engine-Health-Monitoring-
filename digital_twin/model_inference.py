"""
Model Inference Integration Module - Phase 5 Digital Twin Prognostics
Couples Phase 4 champion machine-learning models (XGBoost and LSTM) with the Digital Twin
to generate cycle-by-cycle Remaining Useful Life (RUL) estimates and model consensus spread.

Roadmap Reference: Phase 5 - Step 5.6
Phase 4 Reference: Models from Step 4.2 (XGBoost) and Step 4.3 (LSTM) champions
"""

import math
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import xgboost as xgb

from digital_twin.engine_state import EngineState
from models.lstm.run_lstm_models import SingleLSTMRegressor, StackedLSTMRegressor


# Default directory paths
DEFAULT_CHECKPOINT_DIR = os.path.join("models", "checkpoints")

# Champion model specifications matching checkpoint_manifest.json
CHAMPION_MODEL_REGISTRY = {
    "FD001": {
        "xgb": {
            "checkpoint_file": "xgb_FD001_champion.json",
            "variant": "XGBoost_Tuned_SummaryStats",
            "feature_tier": "Tier1_Core14",
            "feature_type": "summary",
            "feature_dim": 70,
            "raw_channels": 14,
        },
        "lstm": {
            "checkpoint_file": "lstm_FD001_champion.pt",
            "variant": "Stacked_LSTM",
            "architecture": "stacked",
            "feature_tier": "Tier1_Core14",
            "feature_dim": 14,
            "hidden_dim1": 64,
            "hidden_dim2": 32,
            "dropout": 0.20,
        },
    },
    "FD002": {
        "xgb": {
            "checkpoint_file": "xgb_FD002_champion.json",
            "variant": "XGBoost_FlattenedWindow",
            "feature_tier": "Tier3_Extended24",
            "feature_type": "flattened",
            "feature_dim": 720,
            "raw_channels": 24,
        },
        "lstm": {
            "checkpoint_file": "lstm_FD002_champion.pt",
            "variant": "Stacked_LSTM",
            "architecture": "stacked",
            "feature_tier": "Tier3_Extended24",
            "feature_dim": 24,
            "hidden_dim1": 64,
            "hidden_dim2": 32,
            "dropout": 0.20,
        },
    },
    "FD003": {
        "xgb": {
            "checkpoint_file": "xgb_FD003_champion.json",
            "variant": "XGBoost_Tuned_SummaryStats",
            "feature_tier": "Tier3_Extended24",
            "feature_type": "summary",
            "feature_dim": 120,
            "raw_channels": 24,
        },
        "lstm": {
            "checkpoint_file": "lstm_FD003_champion.pt",
            "variant": "Single_LSTM",
            "architecture": "single",
            "feature_tier": "Tier3_Extended24",
            "feature_dim": 24,
            "hidden_dim": 48,
            "dropout": 0.20,
        },
    },
    "FD004": {
        "xgb": {
            "checkpoint_file": "xgb_FD004_champion.json",
            "variant": "XGBoost_FlattenedWindow",
            "feature_tier": "Tier3_Extended24",
            "feature_type": "flattened",
            "feature_dim": 720,
            "raw_channels": 24,
        },
        "lstm": {
            "checkpoint_file": "lstm_FD004_champion.pt",
            "variant": "Stacked_LSTM",
            "architecture": "stacked",
            "feature_tier": "Tier3_Extended24",
            "feature_dim": 24,
            "hidden_dim1": 64,
            "hidden_dim2": 32,
            "dropout": 0.20,
        },
    },
}


def compute_window_summary_statistics(
    X_seq: np.ndarray, col_indices: Optional[List[int]] = None
) -> np.ndarray:
    """
    Computes 5 summary statistics (mean, std, min, max, linear slope) across the 30-cycle window.
    Exact reproduction of Phase 4 summary statistics feature engineering.

    Args:
        X_seq: 2D array of shape (30, K) or 3D array of shape (N, 30, K).
        col_indices: Optional column index filter.

    Returns:
        Summary features of shape (N, 5 * K_sub) as float32 array.
    """
    if X_seq.ndim == 2:
        X_seq = X_seq[np.newaxis, :, :]

    if col_indices is not None:
        X_sub = X_seq[:, :, col_indices]
    else:
        X_sub = X_seq

    N, W, K = X_sub.shape
    if W != 30:
        raise ValueError(f"Summary statistics require window length W=30, got {W}.")

    if not np.all(np.isfinite(X_sub)):
        raise ValueError("Input sequence contains non-finite values (NaN or Inf).")

    mean_val = np.mean(X_sub, axis=1)
    std_val = np.std(X_sub, axis=1)
    min_val = np.min(X_sub, axis=1)
    max_val = np.max(X_sub, axis=1)

    t = np.arange(W, dtype=np.float32)
    t_mean = (W - 1) / 2.0
    t_diff = (t - t_mean).reshape(1, W, 1)
    t_ss = np.sum((t - t_mean) ** 2)

    x_diff = X_sub - mean_val.reshape(N, 1, K)
    slope_val = np.sum(t_diff * x_diff, axis=1) / t_ss

    summary_feats = np.concatenate([mean_val, std_val, min_val, max_val, slope_val], axis=1)
    return summary_feats.astype(np.float32)


def extract_flattened_window(
    X_seq: np.ndarray, col_indices: Optional[List[int]] = None
) -> np.ndarray:
    """
    Flattens a 30-cycle temporal window matrix into a 1D feature representation.
    Exact reproduction of Phase 4 flattened window feature engineering.

    Args:
        X_seq: 2D array of shape (30, K) or 3D array of shape (N, 30, K).
        col_indices: Optional column index filter.

    Returns:
        Flattened features of shape (N, 30 * K_sub) as float32 array.
    """
    if X_seq.ndim == 2:
        X_seq = X_seq[np.newaxis, :, :]

    if col_indices is not None:
        X_sub = X_seq[:, :, col_indices]
    else:
        X_sub = X_seq

    N, W, K = X_sub.shape
    if W != 30:
        raise ValueError(f"Flattened window requires window length W=30, got {W}.")

    if not np.all(np.isfinite(X_sub)):
        raise ValueError("Input sequence contains non-finite values (NaN or Inf).")

    return X_sub.reshape(N, W * K).astype(np.float32)


# Process-level in-memory cache to prevent redundant disk I/O on identical checkpoints
_CHAMPION_MODEL_CACHE: Dict[Tuple[str, str], Tuple[xgb.XGBRegressor, nn.Module]] = {}


class PrognosticsModelManager:
    """
    Manages loading, feature preparation, and inference for the 8 Phase 4 champion models.
    Supports all four C-MAPSS subsets (FD001 - FD004) with exact architectural fidelity.
    """

    def __init__(
        self,
        subset: str = "FD001",
        checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
    ):
        if subset not in CHAMPION_MODEL_REGISTRY:
            raise ValueError(
                f"Unknown subset '{subset}'. Supported subsets: {list(CHAMPION_MODEL_REGISTRY.keys())}"
            )
        self.subset = subset
        self.checkpoint_dir = checkpoint_dir
        self.config = CHAMPION_MODEL_REGISTRY[subset]

        # Model instances
        self.xgb_model: Optional[xgb.XGBRegressor] = None
        self.lstm_model: Optional[nn.Module] = None
        self.is_loaded: bool = False

        # Load models immediately
        self.load_models()

    def load_models(self) -> None:
        """
        Loads the champion XGBoost and LSTM checkpoints from disk.
        """
        cache_key = (self.subset, os.path.abspath(self.checkpoint_dir))
        if cache_key in _CHAMPION_MODEL_CACHE:
            self.xgb_model, self.lstm_model = _CHAMPION_MODEL_CACHE[cache_key]
            self.is_loaded = True
            return

        xgb_cfg = self.config["xgb"]
        lstm_cfg = self.config["lstm"]

        xgb_path = os.path.join(self.checkpoint_dir, xgb_cfg["checkpoint_file"])
        lstm_path = os.path.join(self.checkpoint_dir, lstm_cfg["checkpoint_file"])

        if not os.path.exists(xgb_path):
            raise FileNotFoundError(
                f"XGBoost checkpoint not found at: {xgb_path}. "
                "Ensure Phase 4 champion checkpoints are present."
            )
        if not os.path.exists(lstm_path):
            raise FileNotFoundError(
                f"LSTM checkpoint not found at: {lstm_path}. "
                "Ensure Phase 4 champion checkpoints are present."
            )

        # 1. Load XGBoost champion
        self.xgb_model = xgb.XGBRegressor()
        self.xgb_model.load_model(xgb_path)

        # 2. Instantiate and load LSTM champion architecture
        if lstm_cfg["architecture"] == "single":
            self.lstm_model = SingleLSTMRegressor(
                input_dim=lstm_cfg["feature_dim"],
                hidden_dim=lstm_cfg["hidden_dim"],
                dropout=lstm_cfg["dropout"],
            )
        else:
            self.lstm_model = StackedLSTMRegressor(
                input_dim=lstm_cfg["feature_dim"],
                hidden_dim1=lstm_cfg["hidden_dim1"],
                hidden_dim2=lstm_cfg["hidden_dim2"],
                dropout=lstm_cfg["dropout"],
            )

        payload = torch.load(lstm_path, weights_only=False, map_location="cpu")
        state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
        self.lstm_model.load_state_dict(state_dict)
        self.lstm_model.eval()

        self.is_loaded = True
        _CHAMPION_MODEL_CACHE[cache_key] = (self.xgb_model, self.lstm_model)

    def validate_sequence_input(self, seq_tensor: np.ndarray) -> np.ndarray:
        """
        Validates the 30-cycle temporal sequence matrix before feature extraction or inference.

        Strict Verification Rules:
            1. Input must be a NumPy array.
            2. Input must be strictly 2-dimensional (ndim == 2).
            3. Time steps dimension must be exactly 30 (shape[0] == 30).
            4. Channel dimension must match subset requirement:
               - FD001: exactly 14 channels (Tier 1 Core 14)
               - FD002, FD003, FD004: exactly 24 channels (Tier 3 Extended 24)
            5. Rejects NaN, positive infinity (+inf), and negative infinity (-inf) with clear ValueError messages.

        Args:
            seq_tensor: Candidate sequence matrix.

        Returns:
            Validated numpy array.

        Raises:
            TypeError: If seq_tensor is not a numpy ndarray.
            ValueError: If dimensionality, window length, channel count, or finiteness checks fail.
        """
        if not isinstance(seq_tensor, np.ndarray):
            raise TypeError(
                f"Sequence input must be a numpy ndarray, got {type(seq_tensor).__name__}."
            )

        if seq_tensor.ndim != 2:
            raise ValueError(
                f"Sequence input must be strictly 2-dimensional (30, D), got {seq_tensor.ndim}D "
                f"array with shape {seq_tensor.shape}."
            )

        if seq_tensor.shape[0] != 30:
            raise ValueError(
                f"Sequence input requires exactly 30 time steps, got {seq_tensor.shape[0]}."
            )

        expected_channels = 14 if self.subset == "FD001" else 24
        if seq_tensor.shape[1] != expected_channels:
            raise ValueError(
                f"Mismatched channel count for {self.subset}: Expected {expected_channels} channels, "
                f"got {seq_tensor.shape[1]}."
            )

        if np.isnan(seq_tensor).any():
            raise ValueError("Sequence input contains NaN values; strictly finite numbers required.")

        if np.isposinf(seq_tensor).any():
            raise ValueError("Sequence input contains positive infinity (+inf); strictly finite numbers required.")

        if np.isneginf(seq_tensor).any():
            raise ValueError("Sequence input contains negative infinity (-inf); strictly finite numbers required.")

        return seq_tensor

    def prepare_xgboost_features(self, seq_tensor: np.ndarray) -> np.ndarray:
        """
        Transforms a validated 30-cycle temporal sequence tensor into XGBoost input features.

        Args:
            seq_tensor: Sequence tensor of shape (30, D).

        Returns:
            2D numpy array of shape (1, expected_dim) as float32.
        """
        self.validate_sequence_input(seq_tensor)

        xgb_cfg = self.config["xgb"]
        expected_dim = xgb_cfg["feature_dim"]

        if xgb_cfg["feature_type"] == "summary":
            feats = compute_window_summary_statistics(seq_tensor)
        elif xgb_cfg["feature_type"] == "flattened":
            feats = extract_flattened_window(seq_tensor)
        else:
            raise ValueError(f"Unknown feature transformation type: {xgb_cfg['feature_type']}")

        if feats.shape[1] != expected_dim:
            raise ValueError(
                f"Mismatched XGBoost feature dimension for {self.subset}: "
                f"Expected {expected_dim}, got {feats.shape[1]}."
            )
        return feats

    def prepare_lstm_tensor(self, seq_tensor: np.ndarray) -> torch.Tensor:
        """
        Prepares a validated 30-cycle sequence tensor for LSTM inference.

        Args:
            seq_tensor: Sequence tensor of shape (30, D).

        Returns:
            PyTorch float tensor of shape (1, 30, D).
        """
        self.validate_sequence_input(seq_tensor)

        lstm_cfg = self.config["lstm"]
        expected_dim = lstm_cfg["feature_dim"]

        if seq_tensor.shape[1] != expected_dim:
            raise ValueError(
                f"Mismatched LSTM input channel dimension for {self.subset}: "
                f"Expected {expected_dim}, got {seq_tensor.shape[1]}."
            )

        tensor = torch.from_numpy(seq_tensor).unsqueeze(0).float()
        return tensor

    def predict_xgboost(self, seq_tensor: np.ndarray) -> float:
        """
        Executes inference using the champion XGBoost model.

        Args:
            seq_tensor: Sequence tensor of shape (30, D).

        Returns:
            Point Remaining Useful Life prediction in cycles (uncapped float).
        """
        if not self.is_loaded or self.xgb_model is None:
            raise RuntimeError("XGBoost model is not loaded.")

        feats = self.prepare_xgboost_features(seq_tensor)
        pred_raw = self.xgb_model.predict(feats)

        if pred_raw.size != 1:
            raise ValueError(f"Expected scalar XGBoost prediction, got shape {pred_raw.shape}.")

        val = float(pred_raw.item())
        if not math.isfinite(val):
            raise ValueError(f"XGBoost returned non-finite prediction: {val}")

        return val

    def predict_lstm(self, seq_tensor: np.ndarray) -> float:
        """
        Executes sequence inference using the champion LSTM model.

        Args:
            seq_tensor: Sequence tensor of shape (30, D).

        Returns:
            Sequence Remaining Useful Life prediction in cycles (uncapped float).
        """
        if not self.is_loaded or self.lstm_model is None:
            raise RuntimeError("LSTM model is not loaded.")

        x_tensor = self.prepare_lstm_tensor(seq_tensor)
        with torch.no_grad():
            out = self.lstm_model(x_tensor)

        if out.numel() != 1:
            raise ValueError(f"Expected scalar LSTM prediction, got shape {out.shape}.")

        val = float(out.item())
        if not math.isfinite(val):
            raise ValueError(f"LSTM returned non-finite prediction: {val}")

        return val

    def predict_prognostics(self, seq_tensor: np.ndarray) -> Tuple[float, float, float]:
        """
        Generates dual-model RUL predictions and inter-model consensus spread.

        Args:
            seq_tensor: 30-cycle temporal sequence tensor G31.

        Returns:
            Tuple of (pred_xgb, pred_lstm, prognostic_spread).
        """
        self.validate_sequence_input(seq_tensor)

        pred_xgb = self.predict_xgboost(seq_tensor)
        pred_lstm = self.predict_lstm(seq_tensor)
        spread = abs(pred_xgb - pred_lstm)

        if not math.isfinite(spread):
            raise ValueError(f"Calculated prognostic spread is non-finite: {spread}")

        return float(pred_xgb), float(pred_lstm), float(spread)

    def update_state_prognostics(self, state: EngineState) -> EngineState:
        """
        Updates an EngineState with champion ML predictions (G35, G36, G37).

        Args:
            state: EngineState instance with populated temporal_seq_tensor (G31).

        Returns:
            The same EngineState instance with populated G35, G36, and G37.
        """
        if state.temporal_seq_tensor is None:
            raise ValueError(
                f"EngineState at cycle {state.time_cycles} lacks a temporal_seq_tensor (G31). "
                "Ensure state advancement has constructed the 30-cycle window."
            )

        pred_xgb, pred_lstm, spread = self.predict_prognostics(state.temporal_seq_tensor)

        state.predicted_rul_xgboost = pred_xgb
        state.predicted_rul_lstm = pred_lstm
        state.prognostic_spread = spread

        return state
