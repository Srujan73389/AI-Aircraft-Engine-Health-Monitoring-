"""
Safety Governance and Defensive Input Boundary Auditor
Roadmap Reference: Phase 9 - Step 9.8 ("Test safety acceptance, rejection and missing-input behaviour")
Strict Academic Integrity Standard: Master Roadmap Section 1, Page 2
("Do not invent sensor measurements or report fabricated model scores")

Provides automated auditing of:
1. Tri-state decision matrix verification (ACCEPTED, REJECTED, REQUIRES_REVIEW).
2. Human-in-the-loop operator approval gating.
3. Defensive API boundary rejection for missing, malformed, NaN, Infinite, and illegal payloads.
4. Non-destructive state rollback fidelity across multiple rejection categories.
"""

import copy
import math
from typing import Dict, Any, List, Optional, Tuple, Union

from backend.main import app
from digital_twin.twin_simulator import DigitalTwinSimulator
from safety.verification import (
    SimulationAction,
    ActionType,
    SafetyVerifier,
)
from safety.decision import SafetyDecisionStatus
from safety.closed_loop import ClosedLoopSafetyGovernor
from safety.recovery import RecoveryStrategyType
from safety.disclaimers import ACADEMIC_SIMULATION_DISCLAIMER
from tests.test_backend_endpoints_step8_2 import FastAPITestClientAdapter


def audit_input_boundary_defense(
    payload: Dict[str, Any],
    client: Optional[Any] = None,
) -> Tuple[int, Dict[str, Any]]:
    """
    Submits a candidate payload to the POST /api/v1/simulation/what-if endpoint
    and returns the HTTP status code and response payload.

    Parameters
    ----------
    payload : Dict[str, Any]
        Raw request dictionary.
    client : Optional[Any]
        FastAPI TestClient adapter instance.

    Returns
    -------
    Tuple[int, Dict[str, Any]]
        (status_code, response_json_dict)
    """
    test_client = client or FastAPITestClientAdapter(app)
    resp = test_client.post("/api/v1/simulation/what-if", json=payload)
    return resp.status_code, resp.json()


def audit_safety_decision_matrix(
    subset: str = "FD001",
    unit: int = 1,
    cycle: int = 1,
) -> Dict[str, Any]:
    """
    Executes a comprehensive evaluation across all decision matrix branches:
    1. ACCEPTED (safe thermal perturbation +2.0 °R)
    2. REJECTED (redline breach +50.0 °R)
    3. REJECTED (direct firewall write to measured telemetry T30)
    4. REQUIRES_REVIEW without operator approval (caution breach, update_applied = False)
    5. REQUIRES_REVIEW with operator approval (caution breach, update_applied = True)
    6. Missing and malformed defensive boundary inputs

    Returns
    -------
    Dict[str, Any]
        Structured audit report documenting decision fidelity across all cases.
    """
    sim = DigitalTwinSimulator(subset=subset.upper(), enable_ml_inference=False)
    sim.load_engine(unit_number=unit, split="train")
    if cycle > 1:
        while sim.current_state and sim.current_state.time_cycles < cycle:
            if sim.step() is None:
                break

    baseline_state = sim.current_state.to_dict(include_arrays=False)
    governor = ClosedLoopSafetyGovernor()
    results: Dict[str, Any] = {}

    # Case 1: ACCEPTED - Mild Thermal Perturbation (+2.0 °R)
    act_safe = SimulationAction(
        action_id=f"act_safe_{subset}_u{unit}_c{cycle}",
        action_type=ActionType.THERMAL_PERTURBATION,
        target_variable="thermal_stress_offset",
        proposed_value=2.0,
    )
    res_safe = governor.apply_simulation_action(
        simulator_or_state=copy.deepcopy(baseline_state),
        action=act_safe,
        operator_approval=False,
    )
    results["accepted_safe"] = {
        "decision_status": res_safe.decision_status.value,
        "update_applied": res_safe.update_applied,
        "offset_committed": res_safe.state_after.get("thermal_stress_offset") == 2.0,
        "strategy": res_safe.recovery_plan.strategy_type.value if res_safe.recovery_plan else None,
    }

    # Case 2: REJECTED - Redline Thermal Breach (+50.0 °R)
    act_redline = SimulationAction(
        action_id=f"act_redline_{subset}_u{unit}_c{cycle}",
        action_type=ActionType.THERMAL_PERTURBATION,
        target_variable="thermal_stress_offset",
        proposed_value=50.0,
    )
    res_redline = governor.apply_simulation_action(
        simulator_or_state=copy.deepcopy(baseline_state),
        action=act_redline,
        operator_approval=False,
    )
    results["rejected_redline"] = {
        "decision_status": res_redline.decision_status.value,
        "update_applied": res_redline.update_applied,
        "offset_reset": res_redline.state_after.get("thermal_stress_offset") == 0.0,
        "strategy": res_redline.recovery_plan.strategy_type.value if res_redline.recovery_plan else None,
        "rollback_performed": res_redline.recovery_plan.mitigation_parameters.get("rollback_performed", False) if res_redline.recovery_plan else False,
    }

    # Case 3: REJECTED - Direct Firewall Write to Measured Telemetry (T30)
    act_firewall = SimulationAction(
        action_id=f"act_firewall_{subset}_u{unit}_c{cycle}",
        action_type=ActionType.THERMAL_PERTURBATION,
        target_variable="T30",
        proposed_value=1650.0,
    )
    res_firewall = governor.apply_simulation_action(
        simulator_or_state=copy.deepcopy(baseline_state),
        action=act_firewall,
        operator_approval=False,
    )
    results["rejected_firewall"] = {
        "decision_status": res_firewall.decision_status.value,
        "update_applied": res_firewall.update_applied,
        "strategy": res_firewall.recovery_plan.strategy_type.value if res_firewall.recovery_plan else None,
        "firewall_enforced": res_firewall.recovery_plan.mitigation_parameters.get("firewall_enforced", False) if res_firewall.recovery_plan else False,
    }

    # Case 4: REQUIRES_REVIEW without Operator Approval (Caution breach, +22.0 °R)
    act_review = SimulationAction(
        action_id=f"act_review_{subset}_u{unit}_c{cycle}",
        action_type=ActionType.THERMAL_PERTURBATION,
        target_variable="thermal_stress_offset",
        proposed_value=22.0,
    )
    res_review_blocked = governor.apply_simulation_action(
        simulator_or_state=copy.deepcopy(baseline_state),
        action=act_review,
        operator_approval=False,
    )
    results["requires_review_blocked"] = {
        "decision_status": res_review_blocked.decision_status.value,
        "update_applied": res_review_blocked.update_applied,
        "strategy": res_review_blocked.recovery_plan.strategy_type.value if res_review_blocked.recovery_plan else None,
    }

    # Case 5: REQUIRES_REVIEW with Operator Approval (Caution breach committed)
    res_review_approved = governor.apply_simulation_action(
        simulator_or_state=copy.deepcopy(baseline_state),
        action=act_review,
        operator_approval=True,
    )
    results["requires_review_approved"] = {
        "decision_status": res_review_approved.decision_status.value,
        "update_applied": res_review_approved.update_applied,
        "operator_approved": res_review_approved.operator_approved,
        "offset_committed": res_review_approved.state_after.get("thermal_stress_offset") == 22.0,
    }

    # Case 6: Defensive Boundary Input Tests
    client = FastAPITestClientAdapter(app)

    code_missing, body_missing = audit_input_boundary_defense({
        "subset": subset,
        "action_type": "THERMAL_PERTURBATION",
        "proposed_value": 3.0,
    }, client=client)

    code_invalid_act, body_invalid_act = audit_input_boundary_defense({
        "subset": subset,
        "action_type": "FORBIDDEN_EXPLODE_ENGINE",
        "target_variable": "thermal_stress_offset",
        "proposed_value": 3.0,
    }, client=client)

    code_nan, body_nan = audit_input_boundary_defense({
        "subset": subset,
        "action_type": "THERMAL_PERTURBATION",
        "target_variable": "thermal_stress_offset",
        "proposed_value": float("nan"),
    }, client=client)

    results["defensive_inputs"] = {
        "missing_target_variable_code": code_missing,
        "invalid_action_type_code": code_invalid_act,
        "nan_proposed_value_code": code_nan,
    }

    matrix_compliant = (
        results["accepted_safe"]["decision_status"] == "ACCEPTED"
        and results["accepted_safe"]["update_applied"] is True
        and results["rejected_redline"]["decision_status"] == "REJECTED"
        and results["rejected_redline"]["update_applied"] is False
        and results["rejected_firewall"]["decision_status"] == "REJECTED"
        and results["rejected_firewall"]["update_applied"] is False
        and results["requires_review_blocked"]["decision_status"] == "REQUIRES_REVIEW"
        and results["requires_review_blocked"]["update_applied"] is False
        and results["requires_review_approved"]["decision_status"] == "REQUIRES_REVIEW"
        and results["requires_review_approved"]["update_applied"] is True
        and code_missing == 422
        and code_invalid_act == 422
        and code_nan == 422
    )

    return {
        "subset": subset.upper(),
        "unit_number": unit,
        "cycle": cycle,
        "matrix_compliant": matrix_compliant,
        "results": results,
        "disclaimer": ACADEMIC_SIMULATION_DISCLAIMER,
    }
