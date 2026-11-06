"""
Closed-Loop Simulation Governor and Safe State Mutation Engine
Ensures simulation state updates are committed if and only if authorized by the safety layer,
enforcing rollbacks, parameter clamping, and telemetry immutability.

Roadmap Reference: Phase 7 - Step 7.8
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
from typing import Dict, Any, List, Optional, Union

from safety.decision import (
    SafetyDecisionStatus,
    ActionSafetyDecision,
)
from safety.verification import (
    ActionType,
    SimulationAction,
    SafetyVerifier,
)
from safety.recovery import (
    RecoveryStrategyType,
    RecoveryPriority,
    SimulatedRecoveryPlan,
    generate_recovery_recommendation,
)
from safety.taxonomy import (
    VariableCategory,
    MEASURED_VARIABLE_NAMES,
    verify_provenance_immutability,
)


@dataclass
class ClosedLoopUpdateResult:
    """
    Structured outcome of a closed-loop simulation action evaluation and state mutation attempt.
    """
    action_id: str
    decision_status: SafetyDecisionStatus
    update_applied: bool
    state_before: Dict[str, Any]
    state_after: Dict[str, Any]
    reason_summary: str
    recovery_plan: Optional[SimulatedRecoveryPlan] = None
    operator_approved: bool = False
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_id": self.action_id,
            "decision_status": self.decision_status.value,
            "update_applied": self.update_applied,
            "state_before": dict(self.state_before),
            "state_after": dict(self.state_after),
            "reason_summary": self.reason_summary,
            "recovery_plan": self.recovery_plan.to_dict() if self.recovery_plan else None,
            "operator_approved": self.operator_approved,
            "timestamp": self.timestamp,
        }


class ClosedLoopSafetyGovernor:
    """
    Master closed-loop governor regulating state transitions in the Digital Twin.
    Gates execution behind SafetyVerifier decisions and manages rollback/clamping remediations.
    """

    def __init__(
        self,
        verifier: Optional[SafetyVerifier] = None,
    ):
        self.verifier = verifier or SafetyVerifier()

    def _extract_state_dict(self, target: Any) -> Dict[str, Any]:
        """Normalize target simulator or state to an inspectable dictionary."""
        if hasattr(target, "current_state") and getattr(target, "current_state") is not None:
            c_state = target.current_state
            if hasattr(c_state, "to_dict"):
                return c_state.to_dict(include_arrays=False)
            elif isinstance(c_state, dict):
                return dict(c_state)
        elif hasattr(target, "to_dict"):
            return target.to_dict(include_arrays=False)
        elif isinstance(target, dict):
            return dict(target)
        elif hasattr(target, "__dict__"):
            return {k: v for k, v in vars(target).items() if not k.startswith("_") and not callable(v)}
        return {}

    def _get_target_state_object(self, target: Any) -> Any:
        """Resolve the mutable state object from a simulator or state container."""
        if hasattr(target, "current_state") and getattr(target, "current_state") is not None:
            return target.current_state
        return target

    def apply_simulation_action(
        self,
        simulator_or_state: Any,
        action: SimulationAction,
        operator_approval: bool = False,
    ) -> ClosedLoopUpdateResult:
        """
        Gated closed-loop simulation step:
        1. Evaluates proposed action against safety verification engine.
        2. Generates targeted recovery plan.
        3. If ACCEPTED -> commits candidate state.
        4. If REQUIRES_REVIEW -> commits only if operator_approval is True; otherwise holds state unchanged.
        5. If REJECTED -> aborts update, executes rollback / clamping remediation, preserves telemetry immutability.
        """
        # Step 0: Capture pre-action baseline
        state_before = self._extract_state_dict(simulator_or_state)
        target_state = self._get_target_state_object(simulator_or_state)

        # Step 1: Evaluate action safety decision
        decision: ActionSafetyDecision = self.verifier.evaluate_action_decision(
            simulator_or_state,
            action,
            attach_recovery_plan=True,
        )

        # Step 2: Generate or retrieve recovery plan
        recovery_plan = decision.recovery_plan or generate_recovery_recommendation(
            decision, action, simulator_or_state
        )

        update_applied = False
        reason_summary = decision.summary_message

        # Step 3: Closed-Loop Gating Logic

        # =====================================================================
        # Branch A: ACCEPTED -> Apply candidate state
        # =====================================================================
        if decision.status == SafetyDecisionStatus.ACCEPTED:
            update_applied = True
            self._commit_candidate_action(target_state, action, decision)
            reason_summary = f"Action ACCEPTED: {decision.summary_message}"

        # =====================================================================
        # Branch B: REQUIRES_REVIEW -> Gated by operator_approval
        # =====================================================================
        elif decision.status == SafetyDecisionStatus.REQUIRES_REVIEW:
            if operator_approval:
                update_applied = True
                self._commit_candidate_action(target_state, action, decision)
                reason_summary = (
                    f"Action REQUIRES_REVIEW committed under manual operator override: {decision.summary_message}"
                )
            else:
                update_applied = False
                reason_summary = (
                    f"Action REQUIRES_REVIEW blocked: Gated by safety layer; holds state pending operator approval. "
                    f"Primary: {decision.summary_message}"
                )

        # =====================================================================
        # Branch C: REJECTED -> Abort update & apply recovery remediation
        # =====================================================================
        elif decision.status == SafetyDecisionStatus.REJECTED:
            update_applied = False
            self._apply_recovery_remediation(target_state, action, recovery_plan)
            reason_summary = (
                f"Action REJECTED by safety layer: Update blocked and recovery remediation applied. "
                f"Primary: {decision.summary_message}"
            )

        # Step 4: Capture post-action state and verify measured telemetry immutability
        state_after = self._extract_state_dict(simulator_or_state)

        # Assert zero corruption on MEASURED physical telemetry
        verify_provenance_immutability(state_before, state_after)

        return ClosedLoopUpdateResult(
            action_id=action.action_id,
            decision_status=decision.status,
            update_applied=update_applied,
            state_before=state_before,
            state_after=state_after,
            reason_summary=reason_summary,
            recovery_plan=recovery_plan,
            operator_approved=operator_approval,
        )

    def _commit_candidate_action(
        self,
        target_state: Any,
        action: SimulationAction,
        decision: ActionSafetyDecision,
    ) -> None:
        """
        Commits an authorized simulation action to the target state object.
        Guarantees that measured telemetry slots (G6 - G26) are never directly mutated.
        """
        # 1. Thermal Perturbation: updates G39 thermal_stress_offset
        if action.action_type == ActionType.THERMAL_PERTURBATION:
            offset = float(action.proposed_value if action.proposed_value is not None else action.delta_value or 0.0)
            if hasattr(target_state, "thermal_stress_offset"):
                target_state.thermal_stress_offset = offset
            if isinstance(target_state, dict):
                target_state["thermal_stress_offset"] = offset
                target_state["G39_thermal_stress_offset"] = offset

        # 2. Sensor Bias Injection: updates G40 sensor_bias_delta
        elif action.action_type == ActionType.SENSOR_BIAS_INJECTION:
            biases = dict(getattr(target_state, "sensor_bias_delta", {}) or {})
            if isinstance(action.proposed_value, dict):
                biases.update(action.proposed_value)
            else:
                biases[action.target_variable] = float(action.proposed_value or 0.0)

            if hasattr(target_state, "sensor_bias_delta"):
                target_state.sensor_bias_delta = biases
            if isinstance(target_state, dict):
                target_state["sensor_bias_delta"] = biases
                target_state["G40_sensor_bias_delta"] = biases

        # 3. Operational Reconfiguration: updates G3 - G5 flight conditions
        elif action.action_type == ActionType.OPERATIONAL_RECONFIGURATION:
            if hasattr(target_state, action.target_variable):
                setattr(target_state, action.target_variable, action.proposed_value)
            if isinstance(target_state, dict):
                target_state[action.target_variable] = action.proposed_value

        # 4. State Enrichment: updates ML prognostics / health index
        elif action.action_type == ActionType.STATE_ENRICHMENT:
            if isinstance(action.proposed_value, dict):
                for k, v in action.proposed_value.items():
                    if hasattr(target_state, k):
                        setattr(target_state, k, v)
                    if isinstance(target_state, dict):
                        target_state[k] = v
            else:
                target_var = action.target_variable
                val = action.proposed_value
                if target_var in ("predicted_rul", "rul"):
                    if hasattr(target_state, "predicted_rul_xgboost"):
                        target_state.predicted_rul_xgboost = val
                    if hasattr(target_state, "predicted_rul_lstm"):
                        target_state.predicted_rul_lstm = val
                    if hasattr(target_state, "predicted_rul"):
                        target_state.predicted_rul = val
                elif target_var in ("health_index", "virtual_health_index", "hi"):
                    if hasattr(target_state, "virtual_health_index"):
                        target_state.virtual_health_index = val
                    if hasattr(target_state, "health_index"):
                        target_state.health_index = val
                else:
                    if hasattr(target_state, target_var):
                        setattr(target_state, target_var, val)
                if isinstance(target_state, dict):
                    target_state[target_var] = val

        # 5. Recovery Action
        elif action.action_type == ActionType.RECOVERY_ACTION:
            if hasattr(target_state, "thermal_stress_offset"):
                target_state.thermal_stress_offset = 0.0
            if hasattr(target_state, "sensor_bias_delta"):
                target_state.sensor_bias_delta = {}
            if isinstance(target_state, dict):
                target_state["thermal_stress_offset"] = 0.0
                target_state["sensor_bias_delta"] = {}

    def _apply_recovery_remediation(
        self,
        target_state: Any,
        action: SimulationAction,
        recovery_plan: SimulatedRecoveryPlan,
    ) -> None:
        """
        Applies non-destructive recovery remediation upon rejection.
        """
        mitig = recovery_plan.mitigation_parameters

        # A. Parameter Clamp Remediation (Prognostics)
        if recovery_plan.strategy_type == RecoveryStrategyType.PARAMETER_CLAMP:
            if "clamped_rul" in mitig:
                clamped_rul = mitig["clamped_rul"]
                if hasattr(target_state, "predicted_rul_xgboost"):
                    target_state.predicted_rul_xgboost = clamped_rul
                if hasattr(target_state, "predicted_rul_lstm"):
                    target_state.predicted_rul_lstm = clamped_rul
                if isinstance(target_state, dict):
                    target_state["predicted_rul"] = clamped_rul
                    target_state["predicted_rul_xgboost"] = clamped_rul
                    target_state["predicted_rul_lstm"] = clamped_rul

            if "clamped_health_index" in mitig:
                clamped_hi = mitig["clamped_health_index"]
                if hasattr(target_state, "virtual_health_index"):
                    target_state.virtual_health_index = clamped_hi
                if isinstance(target_state, dict):
                    target_state["virtual_health_index"] = clamped_hi

        # B. Scenario Rollback / Hardware Hard Limit Over-temp Remediation
        elif recovery_plan.strategy_type in (
            RecoveryStrategyType.SCENARIO_ROLLBACK,
            RecoveryStrategyType.PHYSICAL_INSPECTION,
        ):
            # Reset perturbation offsets to clean baseline
            if hasattr(target_state, "thermal_stress_offset"):
                target_state.thermal_stress_offset = 0.0
            if hasattr(target_state, "sensor_bias_delta"):
                target_state.sensor_bias_delta = {}
            if isinstance(target_state, dict):
                target_state["thermal_stress_offset"] = 0.0
                target_state["sensor_bias_delta"] = {}


def safe_simulation_step(
    simulator_or_state: Any,
    action: SimulationAction,
    governor: Optional[ClosedLoopSafetyGovernor] = None,
    operator_approval: bool = False,
) -> ClosedLoopUpdateResult:
    """
    Convenience functional interface to execute a closed-loop safe simulation step.
    """
    gov = governor or ClosedLoopSafetyGovernor()
    return gov.apply_simulation_action(
        simulator_or_state=simulator_or_state,
        action=action,
        operator_approval=operator_approval,
    )
