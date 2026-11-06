"""
Safety Verification and Simulated Recovery Subsystem
Phase 7 - Safety Verification and Operating Constraints
"""

from safety.constraints import (
    ConstraintSource,
    ConstraintSeverity,
    OperatingConstraint,
    ConstraintViolation,
    ConstraintEvaluationReport,
    OPERATING_CONSTRAINTS,
    evaluate_constraints,
)

from safety.taxonomy import (
    VariableCategory,
    VariableProvenance,
    StateVariableClassificationReport,
    ProvenanceViolationError,
    VARIABLE_TAXONOMY,
    get_variable_provenance,
    classify_state_variables,
    verify_provenance_immutability,
)

from safety.verification import (
    ActionType,
    SimulationAction,
    SafetyCheckResult,
    SafetyVerifier,
    verify_simulation_action,
    evaluate_simulation_decision,
    generate_simulation_recovery_plan,
)

from safety.decision import (
    SafetyDecisionStatus,
    ActionSafetyDecision,
    classify_safety_decision,
)

from safety.audit import (
    SafetyReasonCode,
    SafetyAuditRecord,
    SafetyAuditLogger,
    infer_reason_code,
)

from safety.recovery import (
    RecoveryStrategyType,
    RecoveryPriority,
    SimulatedRecoveryPlan,
    generate_recovery_recommendation,
)

from safety.disclaimers import (
    ACADEMIC_SIMULATION_DISCLAIMER,
    FORBIDDEN_COMMAND_PATTERNS,
    CommandAuthorityViolationError,
    assert_non_commanding_recommendation,
    apply_academic_disclaimer,
)

from safety.closed_loop import (
    ClosedLoopUpdateResult,
    ClosedLoopSafetyGovernor,
    safe_simulation_step,
)

__all__ = [
    # Step 7.1 Constraints
    "ConstraintSource",
    "ConstraintSeverity",
    "OperatingConstraint",
    "ConstraintViolation",
    "ConstraintEvaluationReport",
    "OPERATING_CONSTRAINTS",
    "evaluate_constraints",
    # Step 7.2 Variable Taxonomy & Provenance
    "VariableCategory",
    "VariableProvenance",
    "StateVariableClassificationReport",
    "ProvenanceViolationError",
    "VARIABLE_TAXONOMY",
    "get_variable_provenance",
    "classify_state_variables",
    "verify_provenance_immutability",
    # Step 7.3 Safety Verification Engine
    "ActionType",
    "SimulationAction",
    "SafetyCheckResult",
    "SafetyVerifier",
    "verify_simulation_action",
    "evaluate_simulation_decision",
    "generate_simulation_recovery_plan",
    # Step 7.4 Tri-State Decision Classification
    "SafetyDecisionStatus",
    "ActionSafetyDecision",
    "classify_safety_decision",
    # Step 7.5 Safety Decision Audit Logging
    "SafetyReasonCode",
    "SafetyAuditRecord",
    "SafetyAuditLogger",
    "infer_reason_code",
    # Step 7.6 Simulated Recovery Strategy
    "RecoveryStrategyType",
    "RecoveryPriority",
    "SimulatedRecoveryPlan",
    "generate_recovery_recommendation",
    # Step 7.7 Academic Simulation & Non-Authority Disclaimers
    "ACADEMIC_SIMULATION_DISCLAIMER",
    "FORBIDDEN_COMMAND_PATTERNS",
    "CommandAuthorityViolationError",
    "assert_non_commanding_recommendation",
    "apply_academic_disclaimer",
    # Step 7.8 Closed-Loop Simulation Governor
    "ClosedLoopUpdateResult",
    "ClosedLoopSafetyGovernor",
    "safe_simulation_step",
]
