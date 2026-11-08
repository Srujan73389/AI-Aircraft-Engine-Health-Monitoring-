"""
Prognostics & Remaining Useful Life (RUL) Prediction API Router
Phase 8 - Step 8.2: Endpoints serving dual-model XGBoost & Bi-LSTM RUL predictions.
"""

from typing import Any, Dict
import numpy as np
from fastapi import APIRouter, HTTPException

from backend.schemas.api_schemas import (
    RULPredictionRequest,
    RULPredictionResponse,
)
from digital_twin.model_inference import PrognosticsModelManager
from digital_twin.twin_simulator import DigitalTwinSimulator
from safety.disclaimers import ACADEMIC_SIMULATION_DISCLAIMER

router = APIRouter(tags=["Prognostics & RUL Forecasting"])


@router.post("/predict/rul", response_model=RULPredictionResponse)
def predict_rul(req: RULPredictionRequest) -> RULPredictionResponse:
    """
    Executes champion XGBoost and Bi-LSTM Remaining Useful Life inference,
    computing the inter-model spread |RUL_xgb - RUL_lstm| and consensus RUL.
    """
    sub_upper = req.subset.upper()
    if sub_upper not in ("FD001", "FD002", "FD003", "FD004"):
        raise HTTPException(
            status_code=404,
            detail=f"Subset '{req.subset}' not found. Supported subsets: ['FD001', 'FD002', 'FD003', 'FD004']"
        )

    try:
        # Case A: Direct 30-cycle temporal sequence tensor provided
        seq_input = req.sequence_tensor if req.sequence_tensor is not None else req.features_sequence
        if seq_input is not None:
            try:
                seq_arr = np.array(seq_input, dtype=np.float32)
            except Exception as e:
                raise HTTPException(
                    status_code=422,
                    detail=f"Malformed sequence tensor input: unable to convert to float32 matrix. {str(e)}"
                )
            if seq_arr.ndim not in (2, 3):
                raise HTTPException(
                    status_code=400,
                    detail=f"Malformed sequence tensor: expected 2D or 3D temporal matrix of shape (30, D), got {seq_arr.ndim}D array with shape {seq_arr.shape}."
                )
            time_dim = seq_arr.shape[-2]
            if time_dim != 30:
                raise HTTPException(
                    status_code=400,
                    detail=f"Malformed sequence tensor: required window length is 30 cycles, got {time_dim} cycles."
                )
            if not np.all(np.isfinite(seq_arr)):
                raise HTTPException(
                    status_code=400,
                    detail="Malformed sequence tensor: contains non-finite values (NaN or Inf)."
                )

            mgr = PrognosticsModelManager(subset=sub_upper)
            rul_xgb, rul_lstm, spread = mgr.predict_prognostics(seq_arr)
            unit_id = req.unit_number
            cycle_idx = req.cycle

        # Case B: Engine unit number and cycle specified
        else:
            unit_id = req.unit_number if req.unit_number is not None else 1
            sim = DigitalTwinSimulator(subset=sub_upper, enable_ml_inference=True)
            available = sim.loader.get_available_units(subset=sub_upper, split="train")
            if unit_id not in available:
                raise HTTPException(
                    status_code=404,
                    detail=f"Engine unit {unit_id} not found in subset '{sub_upper}'. Valid engine ID range is {min(available)} to {max(available)} (total {len(available)} engines)."
                )

            sim.load_engine(unit_number=unit_id, split="train")
            max_cycle = sim.loader.load_engine_trajectory(sub_upper, unit_id, split="train")["time_cycles"].max()
            if req.cycle is not None:
                if req.cycle > max_cycle:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Requested cycle {req.cycle} exceeds engine {unit_id} maximum cycle {max_cycle}."
                    )
                while sim.current_state and sim.current_state.time_cycles < req.cycle:
                    if sim.step() is None:
                        break
                cycle_idx = req.cycle
            else:
                cycle_idx = sim.current_state.time_cycles

            state = sim.current_state
            if state is None or state.predicted_rul_xgboost is None or state.predicted_rul_lstm is None:
                raise HTTPException(
                    status_code=500,
                    detail="Prognostic inference unpopulated: champion model execution failed to compute RUL values."
                )
            rul_xgb = float(state.predicted_rul_xgboost)
            rul_lstm = float(state.predicted_rul_lstm)
            spread = abs(rul_xgb - rul_lstm)

        consensus_rul = float(np.mean([rul_xgb, rul_lstm]))
        # Spread-based confidence metric in [0, 1]
        confidence_score = float(max(0.0, min(1.0, 1.0 - (spread / max(50.0, consensus_rul)))))

        return RULPredictionResponse(
            subset=sub_upper,
            unit_number=unit_id,
            cycle=cycle_idx,
            predicted_rul_xgboost=round(rul_xgb, 2),
            predicted_rul_lstm=round(rul_lstm, 2),
            prognostic_spread=round(spread, 2),
            consensus_rul=round(consensus_rul, 2),
            confidence_score=round(confidence_score, 4),
            disclaimer=ACADEMIC_SIMULATION_DISCLAIMER,
        )
    except HTTPException:
        raise
    except FileNotFoundError as fnf:
        raise HTTPException(
            status_code=503,
            detail=f"MODEL_UNAVAILABLE: Required champion checkpoint or scaler file is missing or inaccessible. {str(fnf)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prognostic prediction failed for {sub_upper}: {str(e)}"
        )
