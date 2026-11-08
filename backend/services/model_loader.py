"""
Model Registry & Preprocessing Pipeline Loader Service
Phase 8 - Step 8.1: Backend API Model Loader Service

Loads, caches, and verifies all Phase 3 preprocessing scalers (FD001 - FD004)
and Phase 4 champion prognostic & anomaly models (XGBoost, Bi-LSTM, Isolation Forest, PCA-SPE).
"""

from dataclasses import dataclass, field
import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import xgboost as xgb

from preprocessing.scaling_pipeline import CMAPSSScalingPipeline
from models.lstm.run_lstm_models import SingleLSTMRegressor, StackedLSTMRegressor
from safety.disclaimers import ACADEMIC_SIMULATION_DISCLAIMER


# Default paths relative to project root
DEFAULT_SCALERS_DIR = os.path.join("models", "scalers")
DEFAULT_CHECKPOINTS_DIR = os.path.join("models", "checkpoints")
DEFAULT_ANOMALY_MANIFEST = os.path.join("models", "anomaly_detection", "anomaly_manifest.json")
DEFAULT_CHECKPOINT_MANIFEST = os.path.join("models", "checkpoints", "checkpoint_manifest.json")

SUPPORTED_SUBSETS = ["FD001", "FD002", "FD003", "FD004"]

# Champion model architectures matching checkpoint_manifest.json
CHAMPION_SPECS: Dict[str, Dict[str, Any]] = {
    "FD001": {
        "xgb": {
            "checkpoint_file": "xgb_FD001_champion.json",
            "variant": "XGBoost_Tuned_SummaryStats",
            "feature_tier": "Tier1_Core14",
            "feature_dim": 70,
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
            "feature_dim": 720,
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
            "feature_dim": 120,
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
            "feature_dim": 720,
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


@dataclass
class AnomalyDetector:
    """
    Representation of calibrated Phase 4 Anomaly Detection Models (M1 iForest, M2 PCA-SPE).
    """
    subset: str
    model_type: str  # "isolation_forest" or "pca_spe"
    model_name: str
    threshold: float
    feature_columns: List[str]
    operating_regime_mode: str
    is_loaded: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_anomaly(self, score: float) -> bool:
        """Determines whether a given anomaly score breaches the calibrated 95th-percentile threshold."""
        return float(score) > self.threshold

    def get_threshold(self) -> float:
        return self.threshold

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subset": self.subset,
            "model_type": self.model_type,
            "model_name": self.model_name,
            "threshold": self.threshold,
            "feature_count": len(self.feature_columns),
            "feature_columns": self.feature_columns,
            "operating_regime_mode": self.operating_regime_mode,
            "is_loaded": self.is_loaded,
        }


class ModelRegistry:
    """
    In-memory Model and Preprocessing Pipeline Registry for the Backend API.
    Maintains cached instances of Phase 3 scalers and Phase 4 machine learning models.
    """

    def __init__(
        self,
        scalers_dir: str = DEFAULT_SCALERS_DIR,
        checkpoints_dir: str = DEFAULT_CHECKPOINTS_DIR,
        anomaly_manifest_path: str = DEFAULT_ANOMALY_MANIFEST,
        auto_load: bool = False,
    ):
        self.scalers_dir = scalers_dir
        self.checkpoints_dir = checkpoints_dir
        self.anomaly_manifest_path = anomaly_manifest_path

        # Internal in-memory stores
        self._scalers: Dict[str, CMAPSSScalingPipeline] = {}
        self._xgb_models: Dict[str, xgb.XGBRegressor] = {}
        self._lstm_models: Dict[str, nn.Module] = {}
        self._anomaly_models: Dict[str, Dict[str, AnomalyDetector]] = {
            s: {} for s in SUPPORTED_SUBSETS
        }

        self._is_ready: bool = False
        self._artifact_inventory: Dict[str, Any] = {}

        if auto_load:
            self.load_all()

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    @property
    def scalers_count(self) -> int:
        return len(self._scalers)

    @property
    def models_count(self) -> int:
        # 4 XGBoost + 4 LSTM + 8 Anomaly detectors (2 per subset)
        anomaly_count = sum(len(dets) for dets in self._anomaly_models.values())
        return len(self._xgb_models) + len(self._lstm_models) + anomaly_count

    def discover_artifacts(self) -> Dict[str, Any]:
        """
        Discovers and validates paths for all scalers and checkpoint files on disk.
        """
        inventory: Dict[str, Any] = {
            "scalers": {},
            "checkpoints": {},
            "anomaly_manifest_present": os.path.exists(self.anomaly_manifest_path),
        }

        for sub in SUPPORTED_SUBSETS:
            # Scalers: primary scaler_{sub}_tier3.pkl, fallback scaler_{sub}.pkl
            p_primary = os.path.join(self.scalers_dir, f"scaler_{sub}_tier3.pkl")
            p_alt = os.path.join(self.scalers_dir, f"scaler_{sub}.pkl")
            chosen_scaler = p_primary if os.path.exists(p_primary) else (p_alt if os.path.exists(p_alt) else None)
            inventory["scalers"][sub] = {
                "path": chosen_scaler,
                "exists": chosen_scaler is not None,
            }

            # Checkpoints
            xgb_file = CHAMPION_SPECS[sub]["xgb"]["checkpoint_file"]
            lstm_file = CHAMPION_SPECS[sub]["lstm"]["checkpoint_file"]
            p_xgb = os.path.join(self.checkpoints_dir, xgb_file)
            p_lstm = os.path.join(self.checkpoints_dir, lstm_file)
            inventory["checkpoints"][sub] = {
                "xgb": {"path": p_xgb, "exists": os.path.exists(p_xgb)},
                "lstm": {"path": p_lstm, "exists": os.path.exists(p_lstm)},
            }

        self._artifact_inventory = inventory
        return inventory

    def load_scalers(self) -> None:
        """
        Loads all 4 Phase 3 CMAPSS subset scalers (FD001 - FD004) into memory.
        """
        for sub in SUPPORTED_SUBSETS:
            p_primary = os.path.join(self.scalers_dir, f"scaler_{sub}_tier3.pkl")
            p_alt = os.path.join(self.scalers_dir, f"scaler_{sub}.pkl")
            scaler_path = p_primary if os.path.exists(p_primary) else p_alt

            if not os.path.exists(scaler_path):
                raise FileNotFoundError(
                    f"Scaler file for {sub} not found. Searched: {p_primary}, {p_alt}"
                )

            pipeline = CMAPSSScalingPipeline.load(scaler_path)
            self._scalers[sub] = pipeline

    def load_xgboost_models(self) -> None:
        """
        Loads all 4 champion XGBoost models (JSON format) from checkpoints.
        """
        for sub in SUPPORTED_SUBSETS:
            cfg = CHAMPION_SPECS[sub]["xgb"]
            ckpt_path = os.path.join(self.checkpoints_dir, cfg["checkpoint_file"])
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"XGBoost checkpoint for {sub} not found at {ckpt_path}")

            model = xgb.XGBRegressor()
            model.load_model(ckpt_path)
            self._xgb_models[sub] = model

    def load_lstm_models(self) -> None:
        """
        Instantiates and loads all 4 champion PyTorch LSTM models (.pt format).
        """
        for sub in SUPPORTED_SUBSETS:
            cfg = CHAMPION_SPECS[sub]["lstm"]
            ckpt_path = os.path.join(self.checkpoints_dir, cfg["checkpoint_file"])
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"LSTM checkpoint for {sub} not found at {ckpt_path}")

            if cfg["architecture"] == "single":
                model: nn.Module = SingleLSTMRegressor(
                    input_dim=cfg["feature_dim"],
                    hidden_dim=cfg["hidden_dim"],
                    dropout=cfg["dropout"],
                )
            else:
                model = StackedLSTMRegressor(
                    input_dim=cfg["feature_dim"],
                    hidden_dim1=cfg["hidden_dim1"],
                    hidden_dim2=cfg["hidden_dim2"],
                    dropout=cfg["dropout"],
                )

            payload = torch.load(ckpt_path, weights_only=False, map_location="cpu")
            state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
            model.load_state_dict(state_dict)
            model.eval()
            self._lstm_models[sub] = model

    def load_anomaly_models(self) -> None:
        """
        Loads calibrated Isolation Forest (M1) and PCA-SPE (M2) anomaly detectors from manifest.
        """
        if not os.path.exists(self.anomaly_manifest_path):
            raise FileNotFoundError(f"Anomaly manifest not found at: {self.anomaly_manifest_path}")

        with open(self.anomaly_manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)

        experiments = manifest_data.get("experiments", {})

        for sub in SUPPORTED_SUBSETS:
            # 1. Isolation Forest (M1)
            m1_key = f"{sub}_M1_RegimeAware_iForest"
            if m1_key in experiments:
                exp = experiments[m1_key]
                thresh = float(exp["threshold_calibration"]["threshold_value"])
                cols = list(exp.get("feature_columns", []))
                mode = exp.get("operating_regime_mode", "Regime-Normalized")
                detector = AnomalyDetector(
                    subset=sub,
                    model_type="isolation_forest",
                    model_name=exp.get("model_name", "Regime-Aware Isolation Forest"),
                    threshold=thresh,
                    feature_columns=cols,
                    operating_regime_mode=mode,
                    metadata=exp.get("metrics", {}),
                )
                self._anomaly_models[sub]["isolation_forest"] = detector

            # 2. PCA-SPE (M2)
            m2_key = f"{sub}_M2_PCA_SPE"
            if m2_key in experiments:
                exp = experiments[m2_key]
                thresh = float(exp["threshold_calibration"]["threshold_value"])
                cols = list(exp.get("feature_columns", []))
                mode = exp.get("operating_regime_mode", "Regime-Normalized")
                detector = AnomalyDetector(
                    subset=sub,
                    model_type="pca_spe",
                    model_name=exp.get("model_name", "PCA Squared Prediction Error"),
                    threshold=thresh,
                    feature_columns=cols,
                    operating_regime_mode=mode,
                    metadata=exp.get("metrics", {}),
                )
                self._anomaly_models[sub]["pca_spe"] = detector

    def load_all(self) -> None:
        """
        Orchestrates full synchronous loading of all scalers, prognostics, and anomaly detectors.
        """
        self.discover_artifacts()
        self.load_scalers()
        self.load_xgboost_models()
        self.load_lstm_models()
        self.load_anomaly_models()
        self._is_ready = True

    def get_scaler(self, subset: str) -> CMAPSSScalingPipeline:
        """
        Retrieves the loaded scaling pipeline for a specific subset.
        """
        s_upper = subset.upper()
        if s_upper not in self._scalers:
            raise KeyError(
                f"Scaler for subset '{subset}' not found. Available: {list(self._scalers.keys())}"
            )
        return self._scalers[s_upper]

    def get_model(self, subset: str, model_type: str) -> Any:
        """
        Retrieves a loaded model or anomaly detector.

        Args:
            subset: C-MAPSS subset identifier ("FD001" - "FD004").
            model_type: "xgboost" ("xgb"), "lstm" ("bi_lstm"), "isolation_forest" ("iforest"),
                        or "pca_spe" ("pca").
        """
        s_upper = subset.upper()
        m_norm = model_type.lower().replace("-", "_")

        if s_upper not in SUPPORTED_SUBSETS:
            raise KeyError(f"Unsupported subset '{subset}'. Must be one of {SUPPORTED_SUBSETS}")

        if m_norm in ("xgboost", "xgb"):
            if s_upper not in self._xgb_models:
                raise KeyError(f"XGBoost model for {s_upper} not loaded.")
            return self._xgb_models[s_upper]

        elif m_norm in ("lstm", "bi_lstm", "bilstm", "pytorch"):
            if s_upper not in self._lstm_models:
                raise KeyError(f"LSTM model for {s_upper} not loaded.")
            return self._lstm_models[s_upper]

        elif m_norm in ("isolation_forest", "iforest", "m1"):
            dets = self._anomaly_models.get(s_upper, {})
            if "isolation_forest" not in dets:
                raise KeyError(f"Isolation Forest detector for {s_upper} not loaded.")
            return dets["isolation_forest"]

        elif m_norm in ("pca_spe", "pca", "spe", "m2"):
            dets = self._anomaly_models.get(s_upper, {})
            if "pca_spe" not in dets:
                raise KeyError(f"PCA-SPE detector for {s_upper} not loaded.")
            return dets["pca_spe"]

        else:
            raise ValueError(
                f"Unknown model_type '{model_type}'. Supported: ['xgboost', 'lstm', 'isolation_forest', 'pca_spe']"
            )

    def get_status(self) -> Dict[str, Any]:
        """
        Returns a complete system inventory and status summary.
        """
        scalers_status = {}
        for sub in SUPPORTED_SUBSETS:
            loaded = sub in self._scalers
            scalers_status[sub] = {
                "loaded": loaded,
                "feature_count": 24 if loaded else 0,
                "type": "SafeMinMaxScaler",
            }

        models_status = {}
        for sub in SUPPORTED_SUBSETS:
            models_status[sub] = {
                "xgboost": {
                    "loaded": sub in self._xgb_models,
                    "variant": CHAMPION_SPECS[sub]["xgb"]["variant"],
                    "feature_dim": CHAMPION_SPECS[sub]["xgb"]["feature_dim"],
                    "checkpoint": CHAMPION_SPECS[sub]["xgb"]["checkpoint_file"],
                },
                "lstm": {
                    "loaded": sub in self._lstm_models,
                    "variant": CHAMPION_SPECS[sub]["lstm"]["variant"],
                    "architecture": CHAMPION_SPECS[sub]["lstm"]["architecture"],
                    "feature_dim": CHAMPION_SPECS[sub]["lstm"]["feature_dim"],
                    "checkpoint": CHAMPION_SPECS[sub]["lstm"]["checkpoint_file"],
                },
                "isolation_forest": {
                    "loaded": "isolation_forest" in self._anomaly_models.get(sub, {}),
                    "threshold": self._anomaly_models[sub]["isolation_forest"].threshold
                    if "isolation_forest" in self._anomaly_models.get(sub, {}) else None,
                },
                "pca_spe": {
                    "loaded": "pca_spe" in self._anomaly_models.get(sub, {}),
                    "threshold": self._anomaly_models[sub]["pca_spe"].threshold
                    if "pca_spe" in self._anomaly_models.get(sub, {}) else None,
                },
            }

        return {
            "status": "ready" if self._is_ready else "not_loaded",
            "ready": self._is_ready,
            "scalers_loaded_count": self.scalers_count,
            "models_loaded_count": self.models_count,
            "champion_checkpoints_loaded": len(self._xgb_models) + len(self._lstm_models),
            "scalers": scalers_status,
            "models": models_status,
            "disclaimer": ACADEMIC_SIMULATION_DISCLAIMER,
        }


# Global singleton instance
_GLOBAL_REGISTRY: Optional[ModelRegistry] = None


def get_model_registry(auto_load: bool = True) -> ModelRegistry:
    """
    Returns the process-wide ModelRegistry singleton.
    """
    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None:
        _GLOBAL_REGISTRY = ModelRegistry(auto_load=auto_load)
    elif auto_load and not _GLOBAL_REGISTRY.is_ready:
        _GLOBAL_REGISTRY.load_all()
    return _GLOBAL_REGISTRY
