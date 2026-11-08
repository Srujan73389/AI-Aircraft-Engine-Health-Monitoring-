"""
Simulation What-If API Router
Phase 8 - Step 8.2: Endpoints serving interactive What-If scenario perturbations
and closed-loop Digital Twin updates.
"""

from typing import Any, Dict
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
        # 1. Resolve ActionType enum
        action_type_str = req.action_type.upper()
        try:
            a_type = ActionType[action_type_str]
        except KeyError:
            a_type = ActionType.THERMAL_PERTURBATION

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
            while sim.current_state and sim.current_state.time_cycles < req.cycle:
                if sim.step() is None:
                    break

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
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"What-If simulation execution failed: {str(e)}"
        )
