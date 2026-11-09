"""
Safety Verification Engine and Action Validation Subsystem
Evaluates proposed simulation actions, perturbations, and state enrichments against
aerospace operating constraints and telemetry immutability boundaries.

Roadmap Reference: Phase 7 - Step 7.3
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import math
from typing import Dict, Any, List, Optional, Union, Tuple

from safety.constraints import (
    ConstraintSource,
    ConstraintSeverity,
    OperatingConstraint,
    ConstraintViolation,
    ConstraintEvaluationReport,
    OPERATING_CONSTRAINTS,
    evaluate_constraints,
    _extract_variable,
)
from safety.taxonomy import (
    VariableCategory,
    VariableProvenance,
    VARIABLE_TAXONOMY,
    MEASURED_VARIABLE_NAMES,
    get_variable_provenance,
)
from safety.disclaimers import FORBIDDEN_COMMAND_PATTERNS
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from safety.decision import ActionSafetyDecision


class ActionType(str, Enum):
    """Classification of proposed digital twin simulation actions."""
    THERMAL_PERTURBATION = "THERMAL_PERTURBATION"
    SENSOR_BIAS_INJECTION = "SENSOR_BIAS_INJECTION"
    OPERATIONAL_RECONFIGURATION = "OPERATIONAL_RECONFIGURATION"
    STATE_ENRICHMENT = "STATE_ENRICHMENT"
    RECOVERY_ACTION = "RECOVERY_ACTION"


@dataclass
class SimulationAction:
    """
    Formal representation of a proposed simulation action or state mutation.
    """
    action_id: str
    action_type: ActionType
    target_variable: str
    proposed_value: Any
    delta_value: Optional[float] = None
    initiator_id: str = "simulation_harness"
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_id": self.action_id,
            "action_type": self.action_type.value,
            "target_variable": self.target_variable,
            "proposed_value": self.proposed_value,
            "delta_value": self.delta_value,
            "initiator_id": self.initiator_id,
            "description": self.description,
        }


@dataclass
class SafetyCheckResult:
    """
    Structured outcome of a safety verification evaluation.
    """
    action_id: str
    target_variable: str
    is_safe: bool
    provenance_allowed: bool
    constraints_evaluated_count: int
    violations: List[ConstraintViolation] = field(default_factory=list)
    projected_state: Optional[Dict[str, Any]] = None
    evaluation_timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_id": self.action_id,
            "target_variable": self.target_variable,
            "is_safe": self.is_safe,
            "provenance_allowed": self.provenance_allowed,
            "constraints_evaluated_count": self.constraints_evaluated_count,
            "violations": [v.to_dict() for v in self.violations],
            "projected_state": self.projected_state,
            "evaluation_timestamp": self.evaluation_timestamp,
        }


class SafetyVerifier:
    """
    Safety Verification Engine enforcing Digital Twin physical firewalls
    and aerothermal operating constraints prior to action execution.
    """

    def __init__(
        self,
        constraints: Optional[Dict[str, OperatingConstraint]] = None,
        taxonomy: Optional[Dict[str, VariableProvenance]] = None,
        strict_firewall: bool = True,
        audit_logger: Optional[Any] = None,
    ):
        self.constraints = constraints or OPERATING_CONSTRAINTS
        self.taxonomy = taxonomy or VARIABLE_TAXONOMY
        self.strict_firewall = strict_firewall
        self.audit_logger = audit_logger

    def is_target_immutable(self, target_variable: str) -> bool:
        """
        Check if a target variable corresponds to immutable MEASURED physical telemetry.
        """
        prov = get_variable_provenance(target_variable)
        if prov is not None:
            return prov.category == VariableCategory.MEASURED

        if target_variable in MEASURED_VARIABLE_NAMES:
            return True

        # Raw C-MAPSS sensor column names sensor_1 through sensor_21
        if target_variable.startswith("sensor_"):
            suffix = target_variable[7:]
            if suffix.isdigit() and 1 <= int(suffix) <= 21:
                return True

        # Check G6 through G26
        if target_variable.startswith("G") and "_" in target_variable:
            prefix = target_variable.split("_")[0]
            if prefix in self.taxonomy and self.taxonomy[prefix].category == VariableCategory.MEASURED:
                return True

        return False

    def _update_sensor_value(self, state_dict: Dict[str, Any], var_name: str, new_val: Any) -> None:
        """
        Helper to set a variable and keep aliased G-prefixed keys in sync.
        """
        state_dict[var_name] = new_val
        for k in list(state_dict.keys()):
            if k.endswith(f"_{var_name}"):
                state_dict[k] = new_val

    def project_state(
        self,
        current_state_dict: Dict[str, Any],
        action: SimulationAction,
    ) -> Dict[str, Any]:
        """
        Calculates the projected candidate state resulting from the proposed action.
        """
        projected = dict(current_state_dict)

        # 1. Thermal Perturbation: injects thermal_stress_offset and updates effective T30
        if action.action_type == ActionType.THERMAL_PERTURBATION:
            offset = (
                float(action.proposed_value)
                if action.proposed_value is not None
                else float(action.delta_value or 0.0)
            )
            self._update_sensor_value(projected, "thermal_stress_offset", offset)

            # Baseline measured T30
            t30_base = _extract_variable(projected, "T30")
            if t30_base is None:
                t30_base = 1585.0
            effective_t30 = float(t30_base) + offset
            self._update_sensor_value(projected, "T30", effective_t30)

            # Recompute derived hpc_temp_rise
            t24 = _extract_variable(projected, "T24")
            if t24 is None:
                t24 = 640.0
            self._update_sensor_value(projected, "hpc_temp_rise", effective_t30 - float(t24))

        # 2. Sensor Bias Injection: injects sensor_bias_delta and perturbs effective sensor readings
        elif action.action_type == ActionType.SENSOR_BIAS_INJECTION:
            current_biases = dict(projected.get("sensor_bias_delta") or {})
            if isinstance(action.proposed_value, dict):
                current_biases.update(action.proposed_value)
                for s_key, b_val in action.proposed_value.items():
                    base = _extract_variable(projected, s_key)
                    if base is not None:
                        new_val = float(base) + float(b_val)
                        self._update_sensor_value(projected, s_key, new_val)
                    else:
                        self._update_sensor_value(projected, s_key, float(b_val))
            else:
                bias_val = float(action.proposed_value if action.proposed_value is not None else action.delta_value or 0.0)
                current_biases[action.target_variable] = bias_val
                base = _extract_variable(projected, action.target_variable)
                if base is not None:
                    new_val = float(base) + bias_val
                    self._update_sensor_value(projected, action.target_variable, new_val)

            projected["sensor_bias_delta"] = current_biases

            # If P30 or P2 perturbed, recompute OPR
            p30 = _extract_variable(projected, "P30")
            p2 = _extract_variable(projected, "P2")
            if p30 is not None and p2 is not None and float(p2) > 0:
                self._update_sensor_value(projected, "overall_pressure_ratio", float(p30) / float(p2))

        # 3. Operational Reconfiguration: updates flight condition context
        elif action.action_type == ActionType.OPERATIONAL_RECONFIGURATION:
            self._update_sensor_value(projected, action.target_variable, action.proposed_value)

        # 4. State Enrichment: updates prognostics / health index
        elif action.action_type == ActionType.STATE_ENRICHMENT:
            if isinstance(action.proposed_value, dict):
                for k, v in action.proposed_value.items():
                    self._update_sensor_value(projected, k, v)
            else:
                self._update_sensor_value(projected, action.target_variable, action.proposed_value)

        # 5. Recovery Action: resets offsets and returns toward nominal
        elif action.action_type == ActionType.RECOVERY_ACTION:
            self._update_sensor_value(projected, "thermal_stress_offset", 0.0)
            projected["sensor_bias_delta"] = {}
            if isinstance(action.proposed_value, dict):
                for k, v in action.proposed_value.items():
                    self._update_sensor_value(projected, k, v)

        return projected

    def verify_action(
        self,
        current_state: Union[Dict[str, Any], Any],
        action: SimulationAction,
    ) -> SafetyCheckResult:
        """
        Evaluates a proposed simulation action against physical immutability rules
        and aerothermal operating constraints.

        Parameters
        ----------
        current_state : Union[Dict[str, Any], EngineState]
            Baseline state vector before the proposed action.
        action : SimulationAction
            The proposed action specification.

        Returns
        -------
        SafetyCheckResult
            Comprehensive safety verdict, provenance validation, and violations list.
        """
        # 1. Normalize current state to dictionary
        if hasattr(current_state, "current_state") and getattr(current_state, "current_state") is not None:
            c_s = getattr(current_state, "current_state")
            if hasattr(c_s, "to_dict"):
                state_dict = c_s.to_dict(include_arrays=True)
            elif isinstance(c_s, dict):
                state_dict = dict(c_s)
            elif hasattr(c_s, "__dict__"):
                state_dict = {
                    k: v
                    for k, v in vars(c_s).items()
                    if not k.startswith("_") and not callable(v)
                }
            else:
                state_dict = {}
        elif hasattr(current_state, "to_dict"):
            state_dict = current_state.to_dict(include_arrays=True)
        elif isinstance(current_state, dict):
            state_dict = dict(current_state)
        elif hasattr(current_state, "__dict__"):
            state_dict = {
                k: v
                for k, v in vars(current_state).items()
                if not k.startswith("_") and not callable(v)
            }
        else:
            state_dict = {}

        # 2. Check for forbidden aircraft control or certified maintenance commanding patterns
        action_text = f"{action.action_id} {action.description} {action.target_variable} {action.action_type.value}".upper()
        for pattern in FORBIDDEN_COMMAND_PATTERNS:
            pattern_norm = pattern.replace("_", " ").upper()
            if pattern in action_text or pattern_norm in action_text:
                violation = ConstraintViolation(
                    constraint_id="COMMAND_AUTHORITY_VIOLATION",
                    target_variable=action.target_variable,
                    observed_value=pattern,
                    min_value=None,
                    max_value=None,
                    units="",
                    severity=ConstraintSeverity.HARD_LIMIT,
                    source=ConstraintSource.ACADEMIC_SIMULATION_ASSUMPTION,
                    description="Academic non-authority rule: direct aircraft commanding is prohibited.",
                    message=(
                        f"Command Authority Rejection: Proposed action '{action.action_id}' uses prohibited "
                        f"command verb '{pattern}'. This platform is strictly an exploratory academic simulation "
                        f"and cannot issue real aircraft flight-control commands or replace certified maintenance manuals."
                    ),
                )
                return SafetyCheckResult(
                    action_id=action.action_id,
                    target_variable=action.target_variable,
                    is_safe=False,
                    provenance_allowed=False,
                    constraints_evaluated_count=0,
                    violations=[violation],
                    projected_state=None,
                )

        # 3. Strict Safety Firewall: Reject direct mutation of MEASURED telemetry
        if self.strict_firewall and self.is_target_immutable(action.target_variable):
            violation = ConstraintViolation(
                constraint_id="FIREWALL_MEASURED_IMMUTABILITY",
                target_variable=action.target_variable,
                observed_value=action.proposed_value,
                min_value=None,
                max_value=None,
                units="",
                severity=ConstraintSeverity.HARD_LIMIT,
                source=ConstraintSource.PHYSICAL_THERMODYNAMICS,
                description="Digital Twin safety firewall: MEASURED telemetry is immutable.",
                message=(
                    f"Safety Firewall Rejection: Proposed action '{action.action_id}' directly targets "
                    f"MEASURED sensor '{action.target_variable}'. Direct telemetry modification is forbidden. "
                    f"Use THERMAL_PERTURBATION (G39) or SENSOR_BIAS_INJECTION (G40) for simulated perturbations."
                ),
            )
            return SafetyCheckResult(
                action_id=action.action_id,
                target_variable=action.target_variable,
                is_safe=False,
                provenance_allowed=False,
                constraints_evaluated_count=0,
                violations=[violation],
                projected_state=None,
            )

        # 3. Project Candidate State
        projected_state = self.project_state(state_dict, action)

        # 4. Evaluate Operating Constraints and Aerothermal Invariants
        eval_report: ConstraintEvaluationReport = evaluate_constraints(projected_state)

        is_safe = eval_report.is_valid

        return SafetyCheckResult(
            action_id=action.action_id,
            target_variable=action.target_variable,
            is_safe=is_safe,
            provenance_allowed=True,
            constraints_evaluated_count=eval_report.total_evaluated,
            violations=eval_report.violations,
            projected_state=projected_state,
        )

    def evaluate_action_decision(
        self,
        current_state: Union[Dict[str, Any], Any],
        action: SimulationAction,
        attach_recovery_plan: bool = False,
    ) -> "ActionSafetyDecision":
        """
        Evaluates a simulation action and returns a formal tri-state ActionSafetyDecision
        (ACCEPTED, REJECTED, or REQUIRES_REVIEW).
        Automatically logs to attached audit logger if present.
        """
        from safety.decision import classify_safety_decision
        from safety.recovery import generate_recovery_recommendation
        check_result = self.verify_action(current_state, action)
        decision = classify_safety_decision(check_result)
        if attach_recovery_plan:
            decision.recovery_plan = generate_recovery_recommendation(decision, action, current_state)
        if self.audit_logger is not None:
            self.audit_logger.log_decision(decision, action, current_state)
        return decision

    def generate_recovery_plan(
        self,
        decision: "ActionSafetyDecision",
        action: SimulationAction,
        current_state: Optional[Any] = None,
    ) -> Any:
        """
        Generates an actionable simulated recovery plan for an evaluated decision.
        """
        from safety.recovery import generate_recovery_recommendation
        return generate_recovery_recommendation(decision, action, current_state)


def verify_simulation_action(
    current_state: Union[Dict[str, Any], Any],
    action: SimulationAction,
    verifier: Optional[SafetyVerifier] = None,
) -> SafetyCheckResult:
    """
    Convenience functional interface to verify a proposed simulation action.
    """
    v = verifier or SafetyVerifier()
    return v.verify_action(current_state, action)


def evaluate_simulation_decision(
    current_state: Union[Dict[str, Any], Any],
    action: SimulationAction,
    verifier: Optional[SafetyVerifier] = None,
    attach_recovery_plan: bool = False,
) -> "ActionSafetyDecision":
    """
    Convenience functional interface returning an ActionSafetyDecision for a proposed action.
    """
    v = verifier or SafetyVerifier()
    return v.evaluate_action_decision(current_state, action, attach_recovery_plan=attach_recovery_plan)


def generate_simulation_recovery_plan(
    decision: "ActionSafetyDecision",
    action: SimulationAction,
    current_state: Optional[Any] = None,
    verifier: Optional[SafetyVerifier] = None,
) -> Any:
    """
    Convenience functional interface to generate a recovery plan.
    """
    v = verifier or SafetyVerifier()
    return v.generate_recovery_plan(decision, action, current_state)

