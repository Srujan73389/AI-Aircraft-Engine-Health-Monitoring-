"""
Safety Verification & Operational Governance API Router
Phase 8 - Step 8.2: Endpoints serving pre-execution constraint verification,
closed-loop state gating, and append-only audit trail queries.
"""

from typing import Any, Dict, List, Optional
import os
import json
from fastapi import APIRouter, HTTPException, Query

from backend.schemas.api_schemas import (
    SafetyVerificationRequest,
    SafetyVerificationResponse,
    SimulationWhatIfRequest,
    SimulationWhatIfResponse,
    SafetyAuditTrailResponse,
)
from digital_twin.twin_simulator import DigitalTwinSimulator
from safety.verification import (
    SafetyVerifier,
    SimulationAction,
    ActionType,
)
from safety.closed_loop import ClosedLoopSafetyGovernor
from safety.audit import SafetyAuditLogger
from safety.disclaimers import ACADEMIC_SIMULATION_DISCLAIMER

router = APIRouter(tags=["Safety Verification & Governance"])

# Shared audit logger instance
GLOBAL_AUDIT_LOGGER = SafetyAuditLogger(
    log_dir=os.path.join("reports", "safety_logs"),
    log_filename="api_safety_audit.jsonl",
)


@router.post("/safety/verify", response_model=SafetyVerificationResponse)
def verify_safety_action(req: SafetyVerificationRequest) -> SafetyVerificationResponse:
    """
    Evaluates proposed simulation action against operating constraints,
    aerothermal limits, and physical invariants before any state commitment.
    """
    try:
        action_type_str = req.action_type.upper()
        try:
            a_type = ActionType[action_type_str]
        except KeyError:
            a_type = ActionType.THERMAL_PERTURBATION

        action = SimulationAction(
            action_id=req.action_id or f"act_verify_{req.target_variable}",
            action_type=a_type,
            target_variable=req.target_variable,
            proposed_value=req.proposed_value,
            delta_value=req.delta_value,
        )

        # Baseline state resolution
        if req.current_state:
            state_dict = dict(req.current_state)
        else:
            sim = DigitalTwinSimulator(subset=req.subset or "FD001")
            sim.load_engine(unit_number=req.unit_number or 1, split="train")
            if req.cycle:
                while sim.current_state and sim.current_state.time_cycles < req.cycle:
                    if sim.step() is None:
                        break
            state_dict = sim.current_state.to_dict(include_arrays=False)

        verifier = SafetyVerifier(strict_firewall=True, audit_logger=GLOBAL_AUDIT_LOGGER)
        decision = verifier.evaluate_action_decision(
            current_state=state_dict,
            action=action,
            attach_recovery_plan=True,
        )

        violations_data = [v.to_dict() for v in decision.violations]
        plan_dict = decision.recovery_plan.to_dict() if decision.recovery_plan else None

        return SafetyVerificationResponse(
            action_id=decision.action_id,
            status=decision.status.value,
            is_safe=decision.is_accepted,
            summary_message=decision.summary_message,
            caution_count=decision.caution_count,
            critical_count=decision.critical_count,
            hard_limit_count=decision.hard_limit_count,
            violations=violations_data,
            recovery_plan=plan_dict,
            disclaimer=ACADEMIC_SIMULATION_DISCLAIMER,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Safety verification evaluation failed: {str(e)}"
        )


@router.post("/safety/execute-gated", response_model=SimulationWhatIfResponse)
def execute_gated_action(req: SimulationWhatIfRequest) -> SimulationWhatIfResponse:
    """
    Applies an action via ClosedLoopSafetyGovernor: commits state if ACCEPTED,
    gates behind operator approval if REQUIRES_REVIEW, and executes rollback if REJECTED.
    """
    sub_upper = req.subset.upper()

    try:
        action_type_str = req.action_type.upper()
        try:
            a_type = ActionType[action_type_str]
        except KeyError:
            a_type = ActionType.THERMAL_PERTURBATION

        action = SimulationAction(
            action_id=f"act_gated_{sub_upper}_u{req.unit_number}_c{req.cycle}",
            action_type=a_type,
            target_variable=req.target_variable,
            proposed_value=req.proposed_value,
            delta_value=req.delta_value,
            initiator_id=f"user_{req.unit_number}",
            description=f"Gated action on {req.target_variable}",
        )

        sim = DigitalTwinSimulator(subset=sub_upper, enable_ml_inference=True)
        sim.load_engine(unit_number=req.unit_number, split="train")
        if req.cycle:
            while sim.current_state and sim.current_state.time_cycles < req.cycle:
                if sim.step() is None:
                    break

        verifier = SafetyVerifier(strict_firewall=True, audit_logger=GLOBAL_AUDIT_LOGGER)
        governor = ClosedLoopSafetyGovernor(verifier=verifier)
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
            detail=f"Gated action execution failed: {str(e)}"
        )


@router.get("/safety/audit-trail", response_model=SafetyAuditTrailResponse)
def get_audit_trail(
    limit: int = Query(default=50, ge=1, le=500),
    decision_status: Optional[str] = Query(default=None),
    target_variable: Optional[str] = Query(default=None),
) -> SafetyAuditTrailResponse:
    """
    Retrieves append-only safety decision audit trail records.
    """
    records = []
    # 1. Read from in-memory logger
    in_mem = [r.to_dict() for r in GLOBAL_AUDIT_LOGGER.get_recent_records(100)]
    records.extend(in_mem)

    # 2. Also read from JSONL file if present
    log_path = GLOBAL_AUDIT_LOGGER.log_path
    if os.path.exists(log_path):
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line_str = line.strip()
                    if line_str:
                        rec = json.loads(line_str)
                        if rec not in records:
                            records.append(rec)
        except Exception:
            pass

    # Filter
    filtered = records
    if decision_status:
        filtered = [r for r in filtered if r.get("decision_status") == decision_status.upper()]
    if target_variable:
        filtered = [r for r in filtered if r.get("target_variable") == target_variable]

    slice_records = filtered[-limit:]

    return SafetyAuditTrailResponse(
        total_records=len(filtered),
        returned_records=len(slice_records),
        records=slice_records,
        disclaimer=ACADEMIC_SIMULATION_DISCLAIMER,
    )
