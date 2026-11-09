"""
Simulation What-If API Router
Phase 8 - Step 8.2: Endpoints serving interactive What-If scenario perturbations
and closed-loop Digital Twin updates.
"""

from typing import Any, Dict, Optional
import math
from fastapi import APIRouter, HTTPException

from backend.schemas.api_schemas import (
    SimulationWhatIfRequest,
    SimulationWhatIfResponse,
)
from digital_twin.twin_simulator import DigitalTwinSimulator
from safety.verification import SimulationAction, ActionType
from safety.closed_loop import ClosedLoopSafetyGovernor
from safety.disclaimers import ACADEMIC_SIMULATION_DISCLAIMER

router = APIRouter(tags=["What-If Simulation"])


@router.post("/simulation/what-if", response_model=SimulationWhatIfResponse)
def run_simulation_what_if(req: SimulationWhatIfRequest) -> SimulationWhatIfResponse:
    """
    Executes a closed-loop What-If parameter perturbation on the Digital Twin,
    strictly gated behind the safety verification layer.
    """
    sub_upper = req.subset.upper()

    try:
        # 1. Resolve ActionType enum and defensive validation
        action_type_str = req.action_type.upper().strip()
        if action_type_str == "SET_SENSOR_BIAS":
            a_type = ActionType.SENSOR_BIAS_INJECTION
        elif action_type_str in ActionType.__members__:
            a_type = ActionType[action_type_str]
        else:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid action_type '{req.action_type}'. Must be one of {[a.value for a in ActionType]}",
            )

        if not req.target_variable or not req.target_variable.strip():
            raise HTTPException(
                status_code=422,
                detail="target_variable cannot be empty or whitespace.",
            )

        if isinstance(req.proposed_value, float) and (math.isnan(req.proposed_value) or math.isinf(req.proposed_value)):
            raise HTTPException(
                status_code=422,
                detail="proposed_value cannot be NaN or Infinite.",
            )
        if isinstance(req.delta_value, float) and (math.isnan(req.delta_value) or math.isinf(req.delta_value)):
            raise HTTPException(
                status_code=422,
                detail="delta_value cannot be NaN or Infinite.",
            )

        action = SimulationAction(
            action_id=f"act_whatif_{sub_upper}_u{req.unit_number}_c{req.cycle}",
            action_type=a_type,
            target_variable=req.target_variable,
            proposed_value=req.proposed_value,
            delta_value=req.delta_value,
        )

        # 2. Instantiate simulator and advance to requested flight cycle
        sim = DigitalTwinSimulator(subset=sub_upper, enable_ml_inference=True)
        sim.load_engine(unit_number=req.unit_number, split="train")
        if req.cycle:
            sim.advance_to_cycle(req.cycle)

        # 3. Apply safe action through closed-loop governor
        governor = ClosedLoopSafetyGovernor()
        res = governor.apply_simulation_action(
            simulator_or_state=sim,
            action=action,
            operator_approval=req.operator_approval,
        )

        rec_dict = res.recovery_plan.to_dict() if res.recovery_plan else None

        def _normalize(d: Optional[Dict[str, Any]]) -> Dict[str, Any]:
            if not d:
                return {}
            out = dict(d)
            for k, v in d.items():
                if isinstance(k, str) and "_" in k and k.startswith("G"):
                    prefix, rest = k.split("_", 1)
                    if prefix[1:].isdigit() and rest not in out:
                        out[rest] = v
            return out

        return SimulationWhatIfResponse(
            action_id=res.action_id,
            decision_status=res.decision_status.value,
            update_applied=res.update_applied,
            operator_approved=res.operator_approved,
            state_before=_normalize(res.state_before),
            state_after=_normalize(res.state_after),
            reason_summary=res.reason_summary,
            recovery_plan=rec_dict,
            disclaimer=ACADEMIC_SIMULATION_DISCLAIMER,
        )
    except HTTPException:
        raise
    except (FileNotFoundError, KeyError) as err:
        raise HTTPException(
            status_code=503,
            detail=f"MODEL_UNAVAILABLE: Checkpoint or scaler required for simulation is missing or inaccessible. {str(err)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"What-If simulation execution failed: {str(e)}"
        )
