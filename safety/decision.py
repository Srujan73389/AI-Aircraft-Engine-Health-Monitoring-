"""
Safety Decision Classification Subsystem
Evaluates simulation action safety verification results into clear tri-state verdicts:
ACCEPTED, REJECTED, or REQUIRES_REVIEW.

Roadmap Reference: Phase 7 - Step 7.4
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Any, List, Optional, Union

from safety.constraints import (
    ConstraintSeverity,
    ConstraintSource,
    ConstraintViolation,
)
from safety.verification import (
    SafetyCheckResult,
    SimulationAction,
)


class SafetyDecisionStatus(str, Enum):
    """
    Tri-state decision outcome for proposed simulation actions.
    """
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    REQUIRES_REVIEW = "REQUIRES_REVIEW"


@dataclass
class ActionSafetyDecision:
    """
    Structured outcome of an actionable safety decision.
    """
    action_id: str
    status: SafetyDecisionStatus
    target_variable: str
    summary_message: str
    caution_count: int
    critical_count: int
    hard_limit_count: int
    violations: List[ConstraintViolation] = field(default_factory=list)
    projected_state: Optional[Dict[str, Any]] = None
    recovery_plan: Optional[Any] = None
    decision_timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def is_accepted(self) -> bool:
        return self.status == SafetyDecisionStatus.ACCEPTED

    @property
    def is_rejected(self) -> bool:
        return self.status == SafetyDecisionStatus.REJECTED

    @property
    def requires_review(self) -> bool:
        return self.status == SafetyDecisionStatus.REQUIRES_REVIEW

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_id": self.action_id,
            "status": self.status.value,
            "target_variable": self.target_variable,
            "summary_message": self.summary_message,
            "caution_count": self.caution_count,
            "critical_count": self.critical_count,
            "hard_limit_count": self.hard_limit_count,
            "violations": [v.to_dict() for v in self.violations],
            "projected_state": self.projected_state,
            "recovery_plan": (
                self.recovery_plan.to_dict()
                if hasattr(self.recovery_plan, "to_dict")
                else self.recovery_plan
            ),
            "decision_timestamp": self.decision_timestamp,
        }


def classify_safety_decision(safety_result: SafetyCheckResult) -> ActionSafetyDecision:
    """
    Determines the formal tri-state safety decision from a SafetyCheckResult.

    Decision Matrix:
    1. REJECTED:
       - Provenance disallowed (breaches telemetry immutability firewall G6-G26).
       - Any HARD_LIMIT violation (burst pressure, speed disk burst, negative RUL, thermodynamic inversion).
       - Any CRITICAL violation (service ceiling FL450, Mach 0.90, turbine temp limit).
    2. REQUIRES_REVIEW:
       - Zero critical or hard limit violations, but one or more CAUTION violations
         (e.g., T30 thermal creep between 1605 °R and 1625 °R).
    3. ACCEPTED:
       - Zero violations of any severity level; all physical invariants and operational envelopes satisfied.

    Parameters
    ----------
    safety_result : SafetyCheckResult
        Result from SafetyVerifier.verify_action().

    Returns
    -------
    ActionSafetyDecision
        Structured decision record with status, counts, and explanation.
    """
    # 1. Count violation severities
    caution_count = 0
    critical_count = 0
    hard_limit_count = 0

    for v in safety_result.violations:
        if v.severity == ConstraintSeverity.CAUTION:
            caution_count += 1
        elif v.severity == ConstraintSeverity.CRITICAL:
            critical_count += 1
        elif v.severity == ConstraintSeverity.HARD_LIMIT:
            hard_limit_count += 1

    # 2. Evaluate Provenance Firewall Rejection
    if not safety_result.provenance_allowed:
        return ActionSafetyDecision(
            action_id=safety_result.action_id,
            status=SafetyDecisionStatus.REJECTED,
            target_variable=safety_result.target_variable,
            summary_message=(
                f"Action '{safety_result.action_id}' REJECTED: Direct modification of MEASURED "
                f"physical telemetry '{safety_result.target_variable}' is forbidden by the Digital Twin safety firewall."
            ),
            caution_count=caution_count,
            critical_count=critical_count,
            hard_limit_count=hard_limit_count or 1,
            violations=safety_result.violations,
            projected_state=None,
        )

    # 3. Evaluate Hard Limit / Critical Violations -> REJECTED
    if hard_limit_count > 0 or critical_count > 0:
        severe_msgs = [
            f"[{v.severity.value}] {v.constraint_id}: {v.message}"
            for v in safety_result.violations
            if v.severity in (ConstraintSeverity.HARD_LIMIT, ConstraintSeverity.CRITICAL)
        ]
        summary_msg = (
            f"Action '{safety_result.action_id}' REJECTED: {hard_limit_count} hard limit and "
            f"{critical_count} critical violation(s) detected. Primary: {severe_msgs[0]}"
        )
        return ActionSafetyDecision(
            action_id=safety_result.action_id,
            status=SafetyDecisionStatus.REJECTED,
            target_variable=safety_result.target_variable,
            summary_message=summary_msg,
            caution_count=caution_count,
            critical_count=critical_count,
            hard_limit_count=hard_limit_count,
            violations=safety_result.violations,
            projected_state=safety_result.projected_state,
        )

    # 4. Evaluate Caution Violations -> REQUIRES_REVIEW
    if caution_count > 0:
        caution_msgs = [
            f"[{v.severity.value}] {v.constraint_id}: {v.message}"
            for v in safety_result.violations
            if v.severity == ConstraintSeverity.CAUTION
        ]
        summary_msg = (
            f"Action '{safety_result.action_id}' REQUIRES_REVIEW: {caution_count} operational caution limit(s) "
            f"breached without exceeding structural hard limits. Operational review advised. Primary: {caution_msgs[0]}"
        )
        return ActionSafetyDecision(
            action_id=safety_result.action_id,
            status=SafetyDecisionStatus.REQUIRES_REVIEW,
            target_variable=safety_result.target_variable,
            summary_message=summary_msg,
            caution_count=caution_count,
            critical_count=0,
            hard_limit_count=0,
            violations=safety_result.violations,
            projected_state=safety_result.projected_state,
        )

    # 5. Zero Violations -> ACCEPTED
    return ActionSafetyDecision(
        action_id=safety_result.action_id,
        status=SafetyDecisionStatus.ACCEPTED,
        target_variable=safety_result.target_variable,
        summary_message=(
            f"Action '{safety_result.action_id}' ACCEPTED: All {safety_result.constraints_evaluated_count} "
            f"operating constraints, aerothermal invariants, and provenance boundaries fully satisfied."
        ),
        caution_count=0,
        critical_count=0,
        hard_limit_count=0,
        violations=[],
        projected_state=safety_result.projected_state,
    )
