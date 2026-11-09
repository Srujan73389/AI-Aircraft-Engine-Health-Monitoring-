"""
Prognostics Agent Module - Phase 6 Step 6.4
Autonomous Prognostics Agent for the AI-Based Aircraft Engine Digital Twin Multi-Agent System.

Roadmap Reference: Phase 6 - Step 6.4:
"Implement the Prognostics Agent using the saved RUL model and preprocessing pipeline"

Capabilities:
1. Frozen Checkpoint Loader:
   - Ingests pre-trained Phase 4 champion models (XGBoost & PyTorch LSTM) for all 4 C-MAPSS subsets (FD001–FD004).
   - Strictly read-only inference mode with torch.no_grad() and eval() mode for PyTorch architectures.
2. Dual-Model RUL & Ensemble Forecasting:
   - Generates point estimates: predicted_rul_xgboost (G35) and predicted_rul_lstm (G36).
   - Enforces strict non-negativity (RUL >= 0.0) across all predictions even under extrapolation.
   - Produces balanced or weighted ensemble RUL forecasts.
3. Epistemic Uncertainty Quantification & Classification:
   - Computes prognostic consensus spread: G37 = |G35 - G36|.
   - Categorizes uncertainty into standard aerospace operational tiers:
     * LOW_UNCERTAINTY (spread <= 15.0 cycles)
     * MODERATE_UNCERTAINTY (15.0 < spread <= 30.0 cycles)
     * HIGH_UNCERTAINTY (spread > 30.0 cycles)
4. Virtual Engine Health Index (G38):
   - Computes continuous normalized degradation health index H(t) strictly bounded in [0.0, 1.0].
5. Operational Caution Projection:
   - Estimates cycles remaining before reaching mission-critical maintenance threshold (e.g., 45 cycles).
6. Standardized Type-Safe Schema Generation:
   - Ingests PrognosticsInput or Digital Twin EngineState.
   - Produces validated PrognosticsOutput with execution timing metadata (AgentMetadata).
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
import os
import sys
import time
from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np
import torch

from agents.schemas import (
    AgentExecutionStatus,
    AgentMetadata,
    PrognosticsInput,
    PrognosticsOutput,
)
from digital_twin.model_inference import (
    CHAMPION_MODEL_REGISTRY,
    DEFAULT_CHECKPOINT_DIR,
    PrognosticsModelManager,
)


class PrognosticsAgent:
    """
    Autonomous Aerospace Prognostics Agent.
    Executes deep learning and gradient boosted ensemble inference to forecast
    Remaining Useful Life (RUL), quantify epistemic spread, and assess virtual engine health.
    """

    # Uncertainty tier thresholds (cycles)
    LOW_UNCERTAINTY_LIMIT: float = 15.0
    MODERATE_UNCERTAINTY_LIMIT: float = 30.0

    # Default operational caution threshold (cycles)
    DEFAULT_CAUTION_THRESHOLD: float = 45.0

    # Phase 4 piecewise RUL maximum plateau (cycles)
    DEFAULT_RUL_MAX_PLATEAU: float = 125.0

    def __init__(
        self,
        agent_id: str = "prognostics_agent_v1",
        subset: str = "FD001",
        checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
        caution_threshold: float = DEFAULT_CAUTION_THRESHOLD,
        rul_max_plateau: float = DEFAULT_RUL_MAX_PLATEAU,
        ensemble_weights: Tuple[float, float] = (0.5, 0.5),  # (xgb_weight, lstm_weight)
        temporal_transformer: Optional[Any] = None,
    ):
        """
        Initializes the Prognostics Agent.

        Args:
            agent_id: Unique agent identifier for metadata tracking.
            subset: Default C-MAPSS dataset subset ("FD001", "FD002", "FD003", "FD004").
            checkpoint_dir: Directory path containing Phase 4 champion model checkpoints.
            caution_threshold: Operational cycle threshold triggering maintenance caution (default: 45 cycles).
            rul_max_plateau: Piecewise linear RUL ceiling for health index normalization (default: 125 cycles).
            ensemble_weights: Tuple of (weight_xgb, weight_lstm), must sum to 1.0.
            temporal_transformer: Optional Temporal Transformer self-attention model for transformer-based RUL forecasting.
        """
        self.agent_id = agent_id
        self.default_subset = subset.upper()
        self.checkpoint_dir = checkpoint_dir
        self.caution_threshold = float(caution_threshold)
        self.rul_max_plateau = max(1.0, float(rul_max_plateau))
        self.temporal_transformer = temporal_transformer

        # Normalize ensemble weights
        w_xgb, w_lstm = ensemble_weights
        total_w = w_xgb + w_lstm
        if total_w <= 0.0:
            raise ValueError(f"Ensemble weights must sum to > 0, got {ensemble_weights}.")
        self.weight_xgb = float(w_xgb / total_w)
        self.weight_lstm = float(w_lstm / total_w)

        # Cache of PrognosticsModelManagers per subset
        self._model_managers: Dict[str, PrognosticsModelManager] = {}

        # Preload the default subset manager
        self.get_model_manager(self.default_subset)

    def get_model_manager(self, subset: str) -> PrognosticsModelManager:
        """
        Retrieves or initializes the cached PrognosticsModelManager for a specific subset.

        Args:
            subset: SubsetName ("FD001", "FD002", "FD003", "FD004").

        Returns:
            Configured and loaded PrognosticsModelManager instance.
        """
        sub = subset.upper()
        if sub not in CHAMPION_MODEL_REGISTRY:
            raise ValueError(
                f"Unknown subset '{sub}'. Supported subsets: {list(CHAMPION_MODEL_REGISTRY.keys())}"
            )

        if sub not in self._model_managers:
            self._model_managers[sub] = PrognosticsModelManager(
                subset=sub, checkpoint_dir=self.checkpoint_dir
            )
        return self._model_managers[sub]

    def classify_uncertainty(self, spread: float) -> str:
        """
        Classifies epistemic uncertainty spread into operational tiers.

        Args:
            spread: Inter-model absolute spread |RUL_xgb - RUL_lstm| in cycles.

        Returns:
            "LOW_UNCERTAINTY", "MODERATE_UNCERTAINTY", or "HIGH_UNCERTAINTY".
        """
        if spread <= self.LOW_UNCERTAINTY_LIMIT:
            return "LOW_UNCERTAINTY"
        elif spread <= self.MODERATE_UNCERTAINTY_LIMIT:
            return "MODERATE_UNCERTAINTY"
        else:
            return "HIGH_UNCERTAINTY"

    def compute_virtual_health_index(self, ensemble_rul: float) -> float:
        """
        Computes the normalized virtual engine health index H(t) in [0.0, 1.0].
        1.0 represents pristine operational health; 0.0 represents complete life exhaustion.

        Args:
            ensemble_rul: Predicted remaining useful life in cycles.

        Returns:
            Health index float strictly bounded in [0.0, 1.0].
        """
        h_val = ensemble_rul / self.rul_max_plateau
        return float(min(1.0, max(0.0, h_val)))

    def compute_cycles_to_caution(self, ensemble_rul: float) -> float:
        """
        Estimates the number of flight cycles remaining before reaching the operational caution threshold.

        Args:
            ensemble_rul: Predicted remaining useful life in cycles.

        Returns:
            Non-negative float cycles until caution threshold is breached.
        """
        if ensemble_rul <= self.caution_threshold:
            return 0.0
        return float(ensemble_rul - self.caution_threshold)

    def process_input(
        self,
        prognostics_input: Union[PrognosticsInput, Dict[str, Any]],
    ) -> PrognosticsOutput:
        """
        Executes prognostics inference on a strongly-typed PrognosticsInput payload.

        Args:
            prognostics_input: PrognosticsInput instance or serialized dictionary.

        Returns:
            Type-safe, fully validated PrognosticsOutput instance.
        """
        start_time = time.perf_counter()
        timestamp = datetime.now(timezone.utc).isoformat()

        # Ingestion & validation
        if isinstance(prognostics_input, dict):
            inp = PrognosticsInput.from_dict(prognostics_input)
        elif isinstance(prognostics_input, PrognosticsInput):
            inp = prognostics_input
        else:
            raise TypeError(f"Expected PrognosticsInput or dict, got {type(prognostics_input)}")

        inp.validate()

        unit_number = inp.unit_number
        time_cycles = inp.time_cycles
        subset = inp.subset.upper()

        # Convert sequence tensor to numpy float32 array
        seq_tensor = np.array(inp.temporal_seq_tensor, dtype=np.float32)

        # Retrieve model manager for the target subset
        manager = self.get_model_manager(subset)

        # Execute dual-model inference
        raw_pred_xgb, raw_pred_lstm, _ = manager.predict_prognostics(seq_tensor)

        # Enforce non-negativity constraint (RUL >= 0.0)
        pred_xgb = max(0.0, float(raw_pred_xgb))
        pred_lstm = max(0.0, float(raw_pred_lstm))

        # Compute ensemble forecast
        ensemble_rul = max(0.0, float(self.weight_xgb * pred_xgb + self.weight_lstm * pred_lstm))

        # Compute epistemic uncertainty spread |G35 - G36|
        prognostic_spread = abs(pred_xgb - pred_lstm)

        # Classify uncertainty tier
        uncertainty_class = self.classify_uncertainty(prognostic_spread)

        # Compute virtual health index G38 in [0.0, 1.0]
        virtual_health_index = self.compute_virtual_health_index(ensemble_rul)

        # Compute cycles to caution threshold
        cycles_to_caution = self.compute_cycles_to_caution(ensemble_rul)

        # Assemble summary message
        summary = (
            f"Engine Unit {unit_number} cycle {time_cycles} ({subset}): "
            f"Ensemble RUL forecast is {ensemble_rul:.1f} cycles "
            f"(XGBoost: {pred_xgb:.1f} cycles, LSTM: {pred_lstm:.1f} cycles). "
            f"Epistemic spread: {prognostic_spread:.1f} cycles [{uncertainty_class}]. "
            f"Virtual Health Index H(t): {virtual_health_index:.2f}. "
            f"Cycles to caution threshold ({self.caution_threshold:.0f} cycles): {cycles_to_caution:.1f} cycles."
        )

        exec_time_ms = (time.perf_counter() - start_time) * 1000.0
        meta = AgentMetadata(
            agent_id=self.agent_id,
            execution_timestamp=timestamp,
            execution_time_ms=exec_time_ms,
            status=AgentExecutionStatus.SUCCESS,
            version="1.0.0",
        )

        output = PrognosticsOutput(
            unit_number=unit_number,
            time_cycles=time_cycles,
            subset=subset,
            predicted_rul_xgboost=pred_xgb,
            predicted_rul_lstm=pred_lstm,
            ensemble_rul_estimate=ensemble_rul,
            prognostic_spread=prognostic_spread,
            virtual_health_index=virtual_health_index,
            uncertainty_classification=uncertainty_class,
            cycles_to_caution_threshold=cycles_to_caution,
            prognostic_summary=summary,
            metadata=meta,
        )

        output.validate()
        return output

    def process_engine_state(self, state: Any) -> PrognosticsOutput:
        """
        Adapter method to ingest a Digital Twin EngineState instance.

        Args:
            state: EngineState instance possessing unit_number, time_cycles, and temporal_seq_tensor.

        Returns:
            Type-safe PrognosticsOutput instance.
        """
        required_attrs = ("unit_number", "time_cycles", "temporal_seq_tensor")
        for attr in required_attrs:
            if not hasattr(state, attr):
                raise TypeError(f"Provided state lacks required EngineState attribute '{attr}'")

        if state.temporal_seq_tensor is None:
            raise ValueError(
                f"EngineState at cycle {state.time_cycles} has no temporal_seq_tensor (G31). "
                "Sequential ML prognostics require a valid 30-cycle temporal window."
            )

        seq_tensor = state.temporal_seq_tensor
        if isinstance(seq_tensor, np.ndarray):
            seq_list = seq_tensor.tolist()
        elif isinstance(seq_tensor, list):
            seq_list = seq_tensor
        else:
            raise TypeError(f"Unexpected temporal_seq_tensor type: {type(seq_tensor)}")

        norm_vec = getattr(state, "normalized_feature_vec", None)
        if norm_vec is not None and isinstance(norm_vec, np.ndarray):
            vec_list = norm_vec.tolist()
        elif norm_vec is not None and isinstance(norm_vec, list):
            vec_list = norm_vec
        else:
            # Fall back to the latest cycle in the sequence tensor
            vec_list = seq_list[-1]

        subset = getattr(state, "subset", self.default_subset)
        tier = getattr(state, "feature_tier", "Tier1_Core14")
        padded = getattr(state, "is_padded", False)

        inp = PrognosticsInput(
            unit_number=int(state.unit_number),
            time_cycles=int(state.time_cycles),
            subset=subset,
            normalized_feature_vec=vec_list,
            temporal_seq_tensor=seq_list,
            feature_tier=tier,
            is_padded=bool(padded),
        )

        return self.process_input(inp)

    def predict_transformer_rul(self, sequence: Any, enable_mc_dropout: bool = False) -> Optional[Dict[str, Any]]:
        """Evaluates sequence window with integrated Temporal Transformer self-attention model."""
        if self.temporal_transformer is None:
            return None
        return self.temporal_transformer.predict_rul(sequence, enable_mc_dropout=enable_mc_dropout)
