"""
Simulated Recovery Strategy Subsystem
Defines automated recovery recommendations—including physical inspections,
maintenance reviews, parameter clamping, and scenario rollbacks.

Roadmap Reference: Phase 7 - Step 7.6
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import math
from typing import Dict, Any, List, Optional, Union
from uuid import uuid4

from safety.decision import (
    SafetyDecisionStatus,
    ActionSafetyDecision,
)
from safety.verification import (
    ActionType,
    SimulationAction,
)
from safety.disclaimers import (
    ACADEMIC_SIMULATION_DISCLAIMER,
    assert_non_commanding_recommendation,
)


class RecoveryStrategyType(str, Enum):
    """
    Standardized classification of recovery strategies for digital twin and engine operations.
    """
    SCENARIO_ROLLBACK = "SCENARIO_ROLLBACK"
    MAINTENANCE_REVIEW = "MAINTENANCE_REVIEW"
    PHYSICAL_INSPECTION = "PHYSICAL_INSPECTION"
    PARAMETER_CLAMP = "PARAMETER_CLAMP"
    CONTINUE_MONITORING = "CONTINUE_MONITORING"


class RecoveryPriority(str, Enum):
    """
    Operational urgency assigned to a recovery recommendation.
    """
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class SimulatedRecoveryPlan:
    """
    Actionable recovery plan providing specific operational and simulated remediation steps.
    """
    plan_id: str
    action_id: str
    decision_status: SafetyDecisionStatus
    strategy_type: RecoveryStrategyType
    priority: RecoveryPriority
    recommended_actions: List[str]
    mitigation_parameters: Dict[str, Any]
    is_simulation_only: bool = True
    is_maintenance_authority: bool = False
    is_flight_control_command: bool = False
    disclaimer_notice: str = ACADEMIC_SIMULATION_DISCLAIMER
    created_timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "action_id": self.action_id,
            "decision_status": self.decision_status.value,
            "strategy_type": self.strategy_type.value,
            "priority": self.priority.value,
            "recommended_actions": list(self.recommended_actions),
            "mitigation_parameters": dict(self.mitigation_parameters),
            "is_simulation_only": self.is_simulation_only,
            "is_maintenance_authority": self.is_maintenance_authority,
            "is_flight_control_command": self.is_flight_control_command,
            "disclaimer_notice": self.disclaimer_notice,
            "created_timestamp": self.created_timestamp,
        }


def _finalize_plan(plan: SimulatedRecoveryPlan) -> SimulatedRecoveryPlan:
    """Helper to verify non-commanding compliance before returning plan."""
    assert_non_commanding_recommendation(plan)
    return plan


def generate_recovery_recommendation(
    decision: ActionSafetyDecision,
    action: SimulationAction,
    current_state: Optional[Any] = None,
) -> SimulatedRecoveryPlan:
    """
    Formulates a targeted recovery recommendation based on the safety decision,
    action type, and breached operational constraints.
    """
    plan_id = f"REC_{uuid4().hex[:12]}"
    violated_ids = [v.constraint_id for v in decision.violations]

    # =========================================================================
    # 1. NOMINAL CASE: Action is ACCEPTED -> CONTINUE_MONITORING (LOW)
    # =========================================================================
    if decision.status == SafetyDecisionStatus.ACCEPTED:
        return _finalize_plan(
            SimulatedRecoveryPlan(
                plan_id=plan_id,
                action_id=action.action_id,
                decision_status=decision.status,
                strategy_type=RecoveryStrategyType.CONTINUE_MONITORING,
                priority=RecoveryPriority.LOW,
                recommended_actions=[
                    f"Action '{action.action_id}' is ACCEPTED; zero operating constraints or physical invariants breached.",
                    "Maintain standard nominal continuous health monitoring and telemetry logging.",
                    "Log state transition into operational baseline history.",
                ],
                mitigation_parameters={
                    "action_permitted": True,
                    "monitoring_cadence_cycles": 1,
                    "operational_status": "NOMINAL",
                },
            )
        )

    # =========================================================================
    # 2. CAUTION CASE: Action REQUIRES_REVIEW -> MAINTENANCE_REVIEW (MEDIUM)
    # =========================================================================
    if decision.status == SafetyDecisionStatus.REQUIRES_REVIEW:
        return _finalize_plan(
            SimulatedRecoveryPlan(
                plan_id=plan_id,
                action_id=action.action_id,
                decision_status=decision.status,
                strategy_type=RecoveryStrategyType.MAINTENANCE_REVIEW,
                priority=RecoveryPriority.MEDIUM,
                recommended_actions=[
                    f"Flag action '{action.action_id}' for ground maintenance engineering review upon turnaround.",
                    "Initiate high-frequency trend monitoring on HPC discharge temperature (T30) and thermal creep.",
                    "Cross-examine combustor fuel flow ratio (phi) and overall pressure ratio (OPR) for incipient degradation.",
                    "Recommend line maintenance diagnostic inspection prior to next flight release.",
                ],
                mitigation_parameters={
                    "action_permitted": False,
                    "review_category": "THERMAL_CREEP_INSPECTION",
                    "caution_target": action.target_variable,
                    "caution_threshold": 1605.0,
                    "warning_threshold": 1625.0,
                    "recommended_turnaround_window_cycles": 5,
                },
            )
        )

    # =========================================================================
    # 3. REJECTED CASES: Disallowed / Severe Limit Breaches
    # =========================================================================

    # A. Safety Firewall Breach (direct write to G6-G26)
    if "FIREWALL_MEASURED_IMMUTABILITY" in violated_ids or "firewall" in decision.summary_message.lower():
        return _finalize_plan(
            SimulatedRecoveryPlan(
                plan_id=plan_id,
                action_id=action.action_id,
                decision_status=decision.status,
                strategy_type=RecoveryStrategyType.SCENARIO_ROLLBACK,
                priority=RecoveryPriority.HIGH,
                recommended_actions=[
                    f"Reject direct modification of MEASURED telemetry '{action.target_variable}': prohibited by Safety Firewall.",
                    "Execute scenario rollback to preserve bit-for-bit physical sensor invariance (|Δ| = 0.0).",
                    "Re-route hypothetical simulation perturbation via virtual offset slots: thermal_stress_offset (G39) or sensor_bias_delta (G40).",
                ],
                mitigation_parameters={
                    "rollback_performed": True,
                    "rejected_target": action.target_variable,
                    "rejected_value": str(action.proposed_value),
                    "firewall_enforced": True,
                    "allowed_alternatives": ["thermal_stress_offset (G39)", "sensor_bias_delta (G40)"],
                },
            )
        )

    # B. Thermodynamic Ordering Inversions (T30 <= T24, P30 <= P15, Nc <= Nf)
    if any("INVARIANT" in cid for cid in violated_ids):
        return _finalize_plan(
            SimulatedRecoveryPlan(
                plan_id=plan_id,
                action_id=action.action_id,
                decision_status=decision.status,
                strategy_type=RecoveryStrategyType.PHYSICAL_INSPECTION,
                priority=RecoveryPriority.CRITICAL,
                recommended_actions=[
                    "Immediate simulation halt: proposed action creates thermodynamic stage ordering inversion.",
                    "Mandatory physical inspection of compressor rotor blades, stator vanes, and bypass ducts for aerodynamic stall/surge damage.",
                    "Execute complete scenario rollback to last verified nominal aerothermal flight state.",
                    "Calibrate inter-stage total temperature and pressure telemetry rakes prior to subsequent testing.",
                ],
                mitigation_parameters={
                    "rollback_performed": True,
                    "inspection_type": "AEROTHERMAL_STAGE_INTEGRITY_INSPECTION",
                    "affected_stages": ["LPC", "HPC", "BYPASS_DUCT"],
                    "reset_offsets": {"thermal_stress_offset": 0.0, "sensor_bias_delta": {}},
                },
            )
        )

    # C. Metallurgical Hard Limit Breach (T30 > 1625 °R) or Casing Overpressure (P30 > 700 psia)
    if "CONST_TEMP_T30_MAX_WARNING" in violated_ids or "CONST_PRESS_P30" in violated_ids:
        safe_ceiling = 1625.0 if "CONST_TEMP_T30_MAX_WARNING" in violated_ids else 700.0
        return _finalize_plan(
            SimulatedRecoveryPlan(
                plan_id=plan_id,
                action_id=action.action_id,
                decision_status=decision.status,
                strategy_type=RecoveryStrategyType.PHYSICAL_INSPECTION,
                priority=RecoveryPriority.CRITICAL,
                recommended_actions=[
                    f"Immediate simulated abort: action breaches catastrophic structural hard limit ({decision.summary_message}).",
                    "Mandate optical borescope inspection of High-Pressure Compressor (HPC) stage disks and combustor liner.",
                    "Inspect high-pressure turbine nozzle guide vanes and blade leading edges for thermal barrier coating spallation.",
                    "Roll back scenario parameters and clamp thermal stress offset to zero (thermal_stress_offset = 0.0).",
                ],
                mitigation_parameters={
                    "rollback_performed": True,
                    "inspection_type": "OPTICAL_BORESCOPE_HPC_COMBUSTOR",
                    "structural_hard_limit_variable": action.target_variable,
                    "safe_threshold": safe_ceiling,
                    "reset_offsets": {"thermal_stress_offset": 0.0},
                },
            )
        )

    # D. Prognostics Causality Violation (Negative RUL, H(t) out of [0, 1])
    if any(cid in ("CONST_PROG_RUL_NON_NEGATIVE", "CONST_PROG_HEALTH_INDEX") for cid in violated_ids):
        # Calculate clamped values
        mitig_params: Dict[str, Any] = {"clamp_applied": True}
        is_rul = "rul" in action.target_variable.lower() or any("RUL" in cid for cid in violated_ids)
        is_hi = "health" in action.target_variable.lower() or any("HEALTH" in cid for cid in violated_ids)

        if is_rul:
            try:
                raw_v = float(action.proposed_value)
                mitig_params["original_value"] = raw_v
                mitig_params["clamped_rul"] = max(0.0, raw_v)
            except (ValueError, TypeError):
                mitig_params["clamped_rul"] = 0.0

        if is_hi:
            try:
                raw_v = float(action.proposed_value)
                mitig_params["original_value"] = raw_v
                mitig_params["clamped_health_index"] = max(0.0, min(1.0, raw_v))
            except (ValueError, TypeError):
                mitig_params["clamped_health_index"] = 0.0

        return _finalize_plan(
            SimulatedRecoveryPlan(
                plan_id=plan_id,
                action_id=action.action_id,
                decision_status=decision.status,
                strategy_type=RecoveryStrategyType.PARAMETER_CLAMP,
                priority=RecoveryPriority.HIGH,
                recommended_actions=[
                    f"Apply mathematical parameter clamp on '{action.target_variable}' to enforce physical causality.",
                    "Clamp predicted Remaining Useful Life non-negativity constraint (RUL >= 0.0 cycles).",
                    "Clamp virtual engine health index to closed unit interval [0.0, 1.0].",
                    "Audit ML model preprocessing and regression inference pipeline for out-of-distribution input sequences.",
                ],
                mitigation_parameters=mitig_params,
            )
        )

    # E. Aerodynamic Flight Envelope Exceedance (Altitude > 45 kft, Mach > 0.90)
    if any(cid in ("CONST_FLIGHT_ALTITUDE", "CONST_FLIGHT_MACH", "CONST_FLIGHT_TRA") for cid in violated_ids):
        safe_limit = 45000.0 if "alt" in action.target_variable.lower() else 0.90
        return _finalize_plan(
            SimulatedRecoveryPlan(
                plan_id=plan_id,
                action_id=action.action_id,
                decision_status=decision.status,
                strategy_type=RecoveryStrategyType.SCENARIO_ROLLBACK,
                priority=RecoveryPriority.HIGH,
                recommended_actions=[
                    f"Reject flight condition reconfiguration: proposed '{action.target_variable}' exceeds certified aircraft envelope.",
                    f"Roll back operational flight profile to certified service ceiling and Mach limit (ceiling: {safe_limit}).",
                    "Notify flight mission profile simulator of aerodynamic boundary constraint.",
                ],
                mitigation_parameters={
                    "rollback_performed": True,
                    "target_variable": action.target_variable,
                    "proposed_value": action.proposed_value,
                    "certified_envelope_limit": safe_limit,
                },
            )
        )

    # F. General Rejection Fallback
    plan = SimulatedRecoveryPlan(
        plan_id=plan_id,
        action_id=action.action_id,
        decision_status=decision.status,
        strategy_type=RecoveryStrategyType.SCENARIO_ROLLBACK,
        priority=RecoveryPriority.HIGH,
        recommended_actions=[
            f"Roll back simulation action '{action.action_id}' due to operational constraint violations.",
            "Restore virtual engine state vector to pre-action baseline.",
        ],
        mitigation_parameters={
            "rollback_performed": True,
            "violated_constraints": violated_ids,
        },
    )
    return _finalize_plan(plan)
