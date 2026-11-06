"""
Academic Simulation and Non-Authority Disclaimer Subsystem
Guarantees that digital twin outputs, recommendations, and simulated recovery plans
never issue real-time aircraft control commands or supersede certified maintenance manuals.

Roadmap Reference: Phase 7 - Step 7.7
"""

from typing import Dict, Any, List, Optional, Union


# =============================================================================
# STANDARD ACADEMIC NON-AUTHORITY DISCLAIMER
# =============================================================================
ACADEMIC_SIMULATION_DISCLAIMER: str = (
    "RESEARCH & ACADEMIC SIMULATION ONLY. This digital twin platform and its AI agents provide exploratory "
    "prognostics and diagnostic recommendations for academic research using NASA C-MAPSS data. They DO NOT "
    "issue real-time aircraft commands, modify flight-control logic, or supersede certified maintenance manuals "
    "(AMM/CMM) or airworthiness directives issued by certified authorities (FAA, EASA, or OEM)."
)


# =============================================================================
# PROHIBITED FLIGHT-CONTROL AND MAINTENANCE COMMAND PATTERNS
# =============================================================================
FORBIDDEN_COMMAND_PATTERNS: List[str] = [
    "EXECUTE_FADEC_OVERRIDE",
    "FADEC_OVERRIDE",
    "FADEC_COMMAND",
    "FLIGHT_CONTROL_DISPATCH",
    "AUTHORIZE_FLIGHT_RELEASE",
    "OVERRIDE_MEL",
    "REPLACE_MAINTENANCE_MANUAL",
    "CLEAR_AIRWORTHINESS_FAULT",
    "ISSUE_FLIGHT_CLEARANCE",
    "EXECUTE_ACTUATOR_COMMAND",
    "DISPATCH_AIRCRAFT",
    "SUPERSEDE_AMM",
    "SUPERSEDE_CMM",
]


class CommandAuthorityViolationError(ValueError):
    """
    Raised when an action, text, or recovery recommendation attempts to assert
    real aircraft flight-control authority or replace certified maintenance procedures.
    """
    pass


def assert_non_commanding_recommendation(
    recommendation_text_or_plan: Union[str, Any],
) -> bool:
    """
    Scans a string, dictionary, or SimulatedRecoveryPlan for prohibited command verbs
    or assertions of real aircraft authority.

    Parameters
    ----------
    recommendation_text_or_plan : Union[str, SimulatedRecoveryPlan, Dict[str, Any]]
        The object or text to inspect.

    Returns
    -------
    bool
        True if the recommendation is verified to be non-commanding.

    Raises
    ------
    CommandAuthorityViolationError
        If any prohibited commanding pattern or authority claim is detected.
    """
    # 1. Inspect direct string
    if isinstance(recommendation_text_or_plan, str):
        _scan_text(recommendation_text_or_plan)
        return True

    # 2. Inspect SimulatedRecoveryPlan or Object
    if hasattr(recommendation_text_or_plan, "is_maintenance_authority"):
        if getattr(recommendation_text_or_plan, "is_maintenance_authority", False):
            raise CommandAuthorityViolationError(
                "Authority Violation: Simulated recovery plan falsely claims maintenance authority."
            )

    if hasattr(recommendation_text_or_plan, "is_flight_control_command"):
        if getattr(recommendation_text_or_plan, "is_flight_control_command", False):
            raise CommandAuthorityViolationError(
                "Authority Violation: Simulated recovery plan falsely claims flight control authority."
            )

    # Inspect recommended_actions list
    if hasattr(recommendation_text_or_plan, "recommended_actions"):
        for action_text in getattr(recommendation_text_or_plan, "recommended_actions", []):
            _scan_text(str(action_text))

    # Inspect mitigation_parameters
    if hasattr(recommendation_text_or_plan, "mitigation_parameters"):
        mitig = getattr(recommendation_text_or_plan, "mitigation_parameters", {})
        _scan_dict(mitig)

    # Inspect dictionary representation
    if isinstance(recommendation_text_or_plan, dict):
        if recommendation_text_or_plan.get("is_maintenance_authority") is True:
            raise CommandAuthorityViolationError(
                "Authority Violation: Dictionary claims maintenance authority."
            )
        if recommendation_text_or_plan.get("is_flight_control_command") is True:
            raise CommandAuthorityViolationError(
                "Authority Violation: Dictionary claims flight control authority."
            )
        _scan_dict(recommendation_text_or_plan)

    return True


def _scan_text(text: str) -> None:
    """Helper to scan a string against forbidden command patterns."""
    text_upper = text.upper()
    for pattern in FORBIDDEN_COMMAND_PATTERNS:
        # Match pattern as substring or with spaces
        pattern_norm = pattern.replace("_", " ").upper()
        if pattern in text_upper or pattern_norm in text_upper:
            raise CommandAuthorityViolationError(
                f"Command Authority Violation: Prohibited pattern '{pattern}' detected in advisory output. "
                f"Digital Twin is strictly an academic exploratory simulation."
            )


def _scan_dict(d: Dict[str, Any]) -> None:
    """Helper to recursively scan dictionary keys and values."""
    for k, v in d.items():
        _scan_text(str(k))
        if isinstance(v, str):
            _scan_text(v)
        elif isinstance(v, list):
            for item in v:
                _scan_text(str(item))
        elif isinstance(v, dict):
            _scan_dict(v)


def apply_academic_disclaimer(
    plan_or_dict: Union[Any, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Attaches formal academic non-authority disclaimer metadata to a dictionary
    or SimulatedRecoveryPlan.
    """
    if hasattr(plan_or_dict, "to_dict"):
        data = plan_or_dict.to_dict()
    elif isinstance(plan_or_dict, dict):
        data = dict(plan_or_dict)
    else:
        data = {"content": str(plan_or_dict)}

    data["is_simulation_only"] = True
    data["is_maintenance_authority"] = False
    data["is_flight_control_command"] = False
    data["disclaimer_notice"] = ACADEMIC_SIMULATION_DISCLAIMER
    data["regulatory_status"] = "EXPLORATORY_PROTOTYPE_NOT_FOR_FLIGHT"

    return data
