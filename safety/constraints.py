"""
Operating Constraints and Aerothermal Safety Foundations
Defines operational boundaries, thermodynamic limits, and physical invariants
governing the turbofan digital twin and autonomous agents.

Roadmap Reference: Phase 7 - Step 7.1
"""

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Dict, Any, List, Optional, Union, Tuple


class ConstraintSource(str, Enum):
    """Origin and theoretical justification for the operating boundary."""
    PHYSICAL_THERMODYNAMICS = "PHYSICAL_THERMODYNAMICS"
    OPERATIONAL_FLIGHT_ENVELOPE = "OPERATIONAL_FLIGHT_ENVELOPE"
    EMPIRICAL_BASELINE = "EMPIRICAL_BASELINE"
    ACADEMIC_SIMULATION_ASSUMPTION = "ACADEMIC_SIMULATION_ASSUMPTION"


class ConstraintSeverity(str, Enum):
    """Enforcement severity level when an operating limit is exceeded."""
    CAUTION = "CAUTION"
    CRITICAL = "CRITICAL"
    HARD_LIMIT = "HARD_LIMIT"


@dataclass(frozen=True)
class OperatingConstraint:
    """
    Formal representation of a single operating constraint or physical invariant.
    """
    constraint_id: str
    target_variable: str
    min_value: Optional[float]
    max_value: Optional[float]
    units: str
    source: ConstraintSource
    severity: ConstraintSeverity
    description: str
    aerothermal_assumption: str
    is_ordering_invariant: bool = False


@dataclass
class ConstraintViolation:
    """Record of an evaluated constraint violation."""
    constraint_id: str
    target_variable: str
    observed_value: Any
    min_value: Optional[float]
    max_value: Optional[float]
    units: str
    severity: ConstraintSeverity
    source: ConstraintSource
    description: str
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "constraint_id": self.constraint_id,
            "target_variable": self.target_variable,
            "observed_value": self.observed_value,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "units": self.units,
            "severity": self.severity.value,
            "source": self.source.value,
            "description": self.description,
            "message": self.message,
        }


@dataclass
class ConstraintEvaluationReport:
    """Structured report returned by evaluate_constraints."""
    total_evaluated: int
    passed_count: int
    violation_count: int
    is_valid: bool
    violations: List[ConstraintViolation] = field(default_factory=list)
    passed_constraints: List[str] = field(default_factory=list)
    violated_constraints: List[str] = field(default_factory=list)

    @property
    def has_violations(self) -> bool:
        return self.violation_count > 0

    def filter_by_severity(self, severity: ConstraintSeverity) -> List[ConstraintViolation]:
        return [v for v in self.violations if v.severity == severity]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_evaluated": self.total_evaluated,
            "passed_count": self.passed_count,
            "violation_count": self.violation_count,
            "is_valid": self.is_valid,
            "violations": [v.to_dict() for v in self.violations],
            "passed_constraints": list(self.passed_constraints),
            "violated_constraints": list(self.violated_constraints),
        }


# =============================================================================
# OPERATING CONSTRAINTS REGISTRY
# =============================================================================
OPERATING_CONSTRAINTS: Dict[str, OperatingConstraint] = {
    # -------------------------------------------------------------------------
    # 1. Ambient Flight Envelope
    # -------------------------------------------------------------------------
    "CONST_FLIGHT_ALTITUDE": OperatingConstraint(
        constraint_id="CONST_FLIGHT_ALTITUDE",
        target_variable="altitude",
        min_value=0.0,
        max_value=45000.0,
        units="ft",
        source=ConstraintSource.OPERATIONAL_FLIGHT_ENVELOPE,
        severity=ConstraintSeverity.CRITICAL,
        description="Operational pressure altitude ceiling for subsonic commercial turbofan.",
        aerothermal_assumption="Standard commercial turbofan service ceiling (FL450) dictated by cabin pressurization and aerodynamic lift boundaries.",
    ),
    "CONST_FLIGHT_MACH": OperatingConstraint(
        constraint_id="CONST_FLIGHT_MACH",
        target_variable="mach_number",
        min_value=0.0,
        max_value=0.90,
        units="dimensionless",
        source=ConstraintSource.OPERATIONAL_FLIGHT_ENVELOPE,
        severity=ConstraintSeverity.CRITICAL,
        description="Flight Mach number operating envelope.",
        aerothermal_assumption="Transonic drag divergence Mach boundary (M <= 0.90) for high-bypass civil aircraft nacelles.",
    ),
    "CONST_FLIGHT_TRA": OperatingConstraint(
        constraint_id="CONST_FLIGHT_TRA",
        target_variable="throttle_resolver_angle",
        min_value=0.0,
        max_value=100.0,
        units="deg",
        source=ConstraintSource.OPERATIONAL_FLIGHT_ENVELOPE,
        severity=ConstraintSeverity.CRITICAL,
        description="Power lever Throttle Resolver Angle (TRA) command travel.",
        aerothermal_assumption="Mechanical resolver authority bounded between flight idle (0 deg) and full forward takeoff rating (100 deg).",
    ),

    # -------------------------------------------------------------------------
    # 2. Core Gas-Path Temperatures
    # -------------------------------------------------------------------------
    "CONST_TEMP_T2": OperatingConstraint(
        constraint_id="CONST_TEMP_T2",
        target_variable="T2",
        min_value=400.0,
        max_value=600.0,
        units="°R",
        source=ConstraintSource.PHYSICAL_THERMODYNAMICS,
        severity=ConstraintSeverity.CRITICAL,
        description="Fan inlet total stagnation temperature.",
        aerothermal_assumption="Total stagnation temperature bounded by ambient ISA standard lapse rates across high-altitude cruise and sea-level hot day.",
    ),
    "CONST_TEMP_T24": OperatingConstraint(
        constraint_id="CONST_TEMP_T24",
        target_variable="T24",
        min_value=450.0,
        max_value=750.0,
        units="°R",
        source=ConstraintSource.PHYSICAL_THERMODYNAMICS,
        severity=ConstraintSeverity.CRITICAL,
        description="LPC discharge total temperature.",
        aerothermal_assumption="Low-Pressure Compressor compression heat addition governed by isentropic efficiency and low-pressure ratio.",
    ),
    "CONST_TEMP_T30_CAUTION": OperatingConstraint(
        constraint_id="CONST_TEMP_T30_CAUTION",
        target_variable="T30",
        min_value=1100.0,
        max_value=1605.0,
        units="°R",
        source=ConstraintSource.EMPIRICAL_BASELINE,
        severity=ConstraintSeverity.CAUTION,
        description="HPC discharge total temperature operational caution boundary.",
        aerothermal_assumption="Empirically calibrated thermal creep threshold where HPC blade degradation accelerates beyond nominal fleet baseline.",
    ),
    "CONST_TEMP_T30_MAX_WARNING": OperatingConstraint(
        constraint_id="CONST_TEMP_T30_MAX_WARNING",
        target_variable="T30",
        min_value=1100.0,
        max_value=1625.0,
        units="°R",
        source=ConstraintSource.PHYSICAL_THERMODYNAMICS,
        severity=ConstraintSeverity.HARD_LIMIT,
        description="HPC discharge total temperature structural hard limit / maximum warning.",
        aerothermal_assumption="Upper metallurgical safety threshold preventing catastrophic high-pressure compressor stage disk thermal distortion.",
    ),
    "CONST_TEMP_T50": OperatingConstraint(
        constraint_id="CONST_TEMP_T50",
        target_variable="T50",
        min_value=1000.0,
        max_value=1600.0,
        units="°R",
        source=ConstraintSource.PHYSICAL_THERMODYNAMICS,
        severity=ConstraintSeverity.CRITICAL,
        description="LPT discharge exhaust total temperature.",
        aerothermal_assumption="Turbine exhaust gas temperature (EGT) survivability envelope balancing expansion work against turbine material creep.",
    ),

    # -------------------------------------------------------------------------
    # 3. Core Gas-Path Pressures
    # -------------------------------------------------------------------------
    "CONST_PRESS_P2": OperatingConstraint(
        constraint_id="CONST_PRESS_P2",
        target_variable="P2",
        min_value=2.5,
        max_value=20.0,
        units="psia",
        source=ConstraintSource.PHYSICAL_THERMODYNAMICS,
        severity=ConstraintSeverity.CRITICAL,
        description="Fan inlet total pressure.",
        aerothermal_assumption="Atmospheric ambient total pressure recovery spanning high-altitude cruise (FL450) to sea-level high-Mach ram stagnation.",
    ),
    "CONST_PRESS_P15": OperatingConstraint(
        constraint_id="CONST_PRESS_P15",
        target_variable="P15",
        min_value=5.0,
        max_value=30.0,
        units="psia",
        source=ConstraintSource.PHYSICAL_THERMODYNAMICS,
        severity=ConstraintSeverity.CRITICAL,
        description="Fan bypass duct total pressure.",
        aerothermal_assumption="Fan pressure ratio (FPR) baseline compression providing bypass stream propulsive thrust.",
    ),
    "CONST_PRESS_P30": OperatingConstraint(
        constraint_id="CONST_PRESS_P30",
        target_variable="P30",
        min_value=100.0,
        max_value=700.0,
        units="psia",
        source=ConstraintSource.PHYSICAL_THERMODYNAMICS,
        severity=ConstraintSeverity.HARD_LIMIT,
        description="HPC delivery total pressure.",
        aerothermal_assumption="Combustor casing pneumatic structural containment pressure limit under full compressor stage work addition.",
    ),
    "CONST_PRESS_PS30": OperatingConstraint(
        constraint_id="CONST_PRESS_PS30",
        target_variable="Ps30",
        min_value=5.0,
        max_value=65.0,
        units="psia",
        source=ConstraintSource.PHYSICAL_THERMODYNAMICS,
        severity=ConstraintSeverity.CRITICAL,
        description="HPC delivery static pressure.",
        aerothermal_assumption="Static diffuser inlet thermodynamic pressure feeding combustor primary fuel-air mixing zone.",
    ),
    "CONST_DERIVED_OPR": OperatingConstraint(
        constraint_id="CONST_DERIVED_OPR",
        target_variable="overall_pressure_ratio",
        min_value=15.0,
        max_value=55.0,
        units="dimensionless",
        source=ConstraintSource.PHYSICAL_THERMODYNAMICS,
        severity=ConstraintSeverity.CRITICAL,
        description="Overall engine compressor pressure ratio (P30 / P2).",
        aerothermal_assumption="Thermodynamic Brayton cycle cycle design envelope ensuring stable aerodynamic operation without compressor stall.",
    ),

    # -------------------------------------------------------------------------
    # 4. Rotor Speeds
    # -------------------------------------------------------------------------
    "CONST_SPEED_NF": OperatingConstraint(
        constraint_id="CONST_SPEED_NF",
        target_variable="Nf",
        min_value=1500.0,
        max_value=3000.0,
        units="rpm",
        source=ConstraintSource.PHYSICAL_THERMODYNAMICS,
        severity=ConstraintSeverity.CRITICAL,
        description="Low-pressure fan spool mechanical rotational speed.",
        aerothermal_assumption="Fan tip aerodynamic Mach shock containment and shaft torsional resonance safety boundaries.",
    ),
    "CONST_SPEED_NC": OperatingConstraint(
        constraint_id="CONST_SPEED_NC",
        target_variable="Nc",
        min_value=7000.0,
        max_value=10500.0,
        units="rpm",
        source=ConstraintSource.PHYSICAL_THERMODYNAMICS,
        severity=ConstraintSeverity.CRITICAL,
        description="High-pressure core spool mechanical rotational speed.",
        aerothermal_assumption="HP compressor disk centrifugal hoop stress limits avoiding plastic burst yielding.",
    ),

    # -------------------------------------------------------------------------
    # 5. Physical Health Index and RUL Boundaries
    # -------------------------------------------------------------------------
    "CONST_PROG_HEALTH_INDEX": OperatingConstraint(
        constraint_id="CONST_PROG_HEALTH_INDEX",
        target_variable="virtual_health_index",
        min_value=0.0,
        max_value=1.0,
        units="dimensionless",
        source=ConstraintSource.ACADEMIC_SIMULATION_ASSUMPTION,
        severity=ConstraintSeverity.HARD_LIMIT,
        description="Digital Twin virtual engine health index H(t).",
        aerothermal_assumption="Normalized continuous degradation metric normalized on closed unit interval [0.0, 1.0], where 1.0 is nominal mint and 0.0 is failure.",
    ),
    "CONST_PROG_RUL_NON_NEGATIVE": OperatingConstraint(
        constraint_id="CONST_PROG_RUL_NON_NEGATIVE",
        target_variable="predicted_rul",
        min_value=0.0,
        max_value=None,
        units="cycles",
        source=ConstraintSource.ACADEMIC_SIMULATION_ASSUMPTION,
        severity=ConstraintSeverity.HARD_LIMIT,
        description="Predicted Remaining Useful Life non-negativity constraint.",
        aerothermal_assumption="Physical causality requirement: an engine cannot exhibit negative operational flight cycles remaining.",
    ),

    # -------------------------------------------------------------------------
    # 6. Aerothermal Ordering Invariants
    # -------------------------------------------------------------------------
    "CONST_INVARIANT_TEMP_ORDER": OperatingConstraint(
        constraint_id="CONST_INVARIANT_TEMP_ORDER",
        target_variable="temperature_ordering",
        min_value=None,
        max_value=None,
        units="°R",
        source=ConstraintSource.PHYSICAL_THERMODYNAMICS,
        severity=ConstraintSeverity.HARD_LIMIT,
        description="Aerothermal temperature ordering invariant: T30 > T24 > T2.",
        aerothermal_assumption="Second Law of Thermodynamics: successive compression work across LPC and HPC must monotonically raise gas stagnation temperature.",
        is_ordering_invariant=True,
    ),
    "CONST_INVARIANT_PRESS_ORDER": OperatingConstraint(
        constraint_id="CONST_INVARIANT_PRESS_ORDER",
        target_variable="pressure_ordering",
        min_value=None,
        max_value=None,
        units="psia",
        source=ConstraintSource.PHYSICAL_THERMODYNAMICS,
        severity=ConstraintSeverity.HARD_LIMIT,
        description="Aerothermal pressure ordering invariant: P30 > P15 > P2.",
        aerothermal_assumption="Compressor positive pressure gradient: stage energy addition requires P30 > P15 > P2 under forward flow.",
        is_ordering_invariant=True,
    ),
    "CONST_INVARIANT_SPOOL_ORDER": OperatingConstraint(
        constraint_id="CONST_INVARIANT_SPOOL_ORDER",
        target_variable="spool_speed_ordering",
        min_value=None,
        max_value=None,
        units="rpm",
        source=ConstraintSource.PHYSICAL_THERMODYNAMICS,
        severity=ConstraintSeverity.HARD_LIMIT,
        description="Spool speed hierarchy invariant: Nc > Nf.",
        aerothermal_assumption="Mechanical design geometry: small-diameter high-pressure spool rotates substantially faster than large-diameter low-pressure fan spool.",
        is_ordering_invariant=True,
    ),
}


# =============================================================================
# EXTRACTION & EVALUATION HELPERS
# =============================================================================
def _extract_variable(state_dict: Dict[str, Any], variable_name: str) -> Optional[float]:
    """
    Extract a variable value from a state dictionary or object, resolving aliases.
    """
    # Direct match
    if variable_name in state_dict:
        val = state_dict[variable_name]
        return float(val) if val is not None and not (isinstance(val, float) and math.isnan(val)) else None

    # Canonical aliases
    aliases: Dict[str, List[str]] = {
        "altitude": ["G3_altitude", "op_setting_1", "alt"],
        "mach_number": ["G4_mach_number", "op_setting_2", "mach"],
        "throttle_resolver_angle": ["G5_throttle_resolver_angle", "op_setting_3", "tra", "TRA"],
        "T2": ["G6_T2", "sensor_1"],
        "T24": ["G7_T24", "sensor_2"],
        "T30": ["G8_T30", "sensor_3"],
        "T50": ["G9_T50", "sensor_4"],
        "P2": ["G10_P2", "sensor_5"],
        "P15": ["G11_P15", "sensor_6"],
        "P30": ["G12_P30", "sensor_7"],
        "Ps30": ["G16_Ps30", "sensor_11"],
        "overall_pressure_ratio": ["G28_overall_pressure_ratio", "OPR", "opr"],
        "Nf": ["G13_Nf", "sensor_8"],
        "Nc": ["G14_Nc", "sensor_9"],
        "virtual_health_index": ["G38_virtual_health_index", "health_index", "H_t", "H"],
        "predicted_rul": [
            "predicted_rul",
            "predicted_rul_xgboost",
            "predicted_rul_lstm",
            "ensemble_rul_estimate",
            "G35_predicted_rul_xgboost",
            "G36_predicted_rul_lstm",
            "rul",
            "RUL",
        ],
    }

    if variable_name in aliases:
        for alias in aliases[variable_name]:
            if alias in state_dict:
                val = state_dict[alias]
                if val is not None and not (isinstance(val, float) and math.isnan(val)):
                    return float(val)

    # Derived fallback for OPR
    if variable_name == "overall_pressure_ratio":
        p30 = _extract_variable(state_dict, "P30")
        p2 = _extract_variable(state_dict, "P2")
        if p30 is not None and p2 is not None and p2 > 0:
            return float(p30 / p2)

    return None


def evaluate_constraints(state_dict_or_engine_state: Union[Dict[str, Any], Any]) -> ConstraintEvaluationReport:
    """
    Evaluate all registered operating constraints and aerothermal ordering invariants
    against an EngineState or dictionary representation.

    Parameters
    ----------
    state_dict_or_engine_state : Union[Dict[str, Any], EngineState]
        Engine state vector or dictionary containing telemetry and/or model predictions.

    Returns
    -------
    ConstraintEvaluationReport
        Structured report detailing total evaluated, violations, passed IDs, and violation IDs.
    """
    # 1. Normalize input to dictionary
    if hasattr(state_dict_or_engine_state, "to_dict"):
        state_dict = state_dict_or_engine_state.to_dict()
    elif isinstance(state_dict_or_engine_state, dict):
        state_dict = dict(state_dict_or_engine_state)
    elif hasattr(state_dict_or_engine_state, "__dict__"):
        state_dict = dict(vars(state_dict_or_engine_state))
    else:
        state_dict = {}

    total_evaluated = 0
    passed_constraints: List[str] = []
    violated_constraints: List[str] = []
    violations: List[ConstraintViolation] = []

    # 2. Evaluate each registered constraint
    for c_id, constraint in OPERATING_CONSTRAINTS.items():
        total_evaluated += 1

        # A. Ordering Invariants
        if constraint.is_ordering_invariant:
            is_passed, observed, msg = _evaluate_ordering_invariant(state_dict, constraint)
            if is_passed:
                passed_constraints.append(c_id)
            else:
                violated_constraints.append(c_id)
                violations.append(
                    ConstraintViolation(
                        constraint_id=c_id,
                        target_variable=constraint.target_variable,
                        observed_value=observed,
                        min_value=constraint.min_value,
                        max_value=constraint.max_value,
                        units=constraint.units,
                        severity=constraint.severity,
                        source=constraint.source,
                        description=constraint.description,
                        message=msg,
                    )
                )
            continue

        # B. Prognostics RUL Non-Negativity (evaluate all present RUL predictions)
        if constraint.target_variable == "predicted_rul":
            rul_vals = []
            for r_key in ["predicted_rul", "predicted_rul_xgboost", "predicted_rul_lstm", "ensemble_rul_estimate", "rul", "RUL"]:
                if r_key in state_dict and state_dict[r_key] is not None:
                    rul_vals.append((r_key, float(state_dict[r_key])))

            if not rul_vals:
                # If RUL fields are not present in state, considered passed (not evaluated / not in failure)
                passed_constraints.append(c_id)
                continue

            rul_violation = None
            for key_name, r_val in rul_vals:
                if r_val < 0.0:
                    rul_violation = (key_name, r_val)
                    break

            if rul_violation is None:
                passed_constraints.append(c_id)
            else:
                key_name, bad_val = rul_violation
                violated_constraints.append(c_id)
                violations.append(
                    ConstraintViolation(
                        constraint_id=c_id,
                        target_variable=f"{constraint.target_variable} ({key_name})",
                        observed_value=bad_val,
                        min_value=0.0,
                        max_value=None,
                        units=constraint.units,
                        severity=constraint.severity,
                        source=constraint.source,
                        description=constraint.description,
                        message=f"Negative RUL violation: {key_name} = {bad_val:.2f} cycles < 0.0 cycles.",
                    )
                )
            continue

        # C. Standard Numerical Range Constraints
        val = _extract_variable(state_dict, constraint.target_variable)

        # If variable is absent or None, check if it's optional (e.g. virtual_health_index before prognostics)
        if val is None:
            # Not observed in state: pass gracefully
            passed_constraints.append(c_id)
            continue

        # Normalize Altitude: C-MAPSS dataset provides altitude in kft (e.g. 0 to 42 kft).
        # Standard constraint is in ft (0 - 45,000 ft).
        eval_val = val
        if constraint.target_variable == "altitude":
            if abs(val) <= 50.0:  # Provided in kft
                eval_val = val * 1000.0

        # Numerical bounds check with aerospace simulation noise margin
        # For sea-level nominal C-MAPSS data, sensor noise may read e.g. -0.7 ft or -0.0004 Mach
        noise_margin = 0.0
        if constraint.target_variable == "altitude":
            noise_margin = 50.0  # Allow -50 ft sea-level ground effect / sensor noise
        elif constraint.target_variable == "mach_number":
            noise_margin = 0.01  # Allow -0.01 sensor zero-bias noise

        is_violated = False
        violation_msg = ""

        if constraint.min_value is not None and eval_val < (constraint.min_value - noise_margin):
            is_violated = True
            violation_msg = (
                f"Lower limit breach: {constraint.target_variable} = {eval_val:.2f} {constraint.units} "
                f"< min allowed {constraint.min_value:.2f} {constraint.units}."
            )
        elif constraint.max_value is not None and eval_val > constraint.max_value:
            is_violated = True
            violation_msg = (
                f"Upper limit breach: {constraint.target_variable} = {eval_val:.2f} {constraint.units} "
                f"> max allowed {constraint.max_value:.2f} {constraint.units}."
            )

        if is_violated:
            violated_constraints.append(c_id)
            violations.append(
                ConstraintViolation(
                    constraint_id=c_id,
                    target_variable=constraint.target_variable,
                    observed_value=eval_val,
                    min_value=constraint.min_value,
                    max_value=constraint.max_value,
                    units=constraint.units,
                    severity=constraint.severity,
                    source=constraint.source,
                    description=constraint.description,
                    message=violation_msg,
                )
            )
        else:
            passed_constraints.append(c_id)

    is_valid = len(violations) == 0

    return ConstraintEvaluationReport(
        total_evaluated=total_evaluated,
        passed_count=len(passed_constraints),
        violation_count=len(violations),
        is_valid=is_valid,
        violations=violations,
        passed_constraints=passed_constraints,
        violated_constraints=violated_constraints,
    )


def _evaluate_ordering_invariant(
    state_dict: Dict[str, Any],
    constraint: OperatingConstraint,
) -> Tuple[bool, Any, str]:
    """
    Evaluate aerodynamic and thermodynamic stage ordering invariants.
    """
    # 1. Temperature Ordering: T30 > T24 > T2
    if constraint.constraint_id == "CONST_INVARIANT_TEMP_ORDER":
        t2 = _extract_variable(state_dict, "T2")
        t24 = _extract_variable(state_dict, "T24")
        t30 = _extract_variable(state_dict, "T30")

        if t2 is None or t24 is None or t30 is None:
            return True, None, "Insufficient temperature variables to evaluate ordering."

        observed = {"T2": t2, "T24": t24, "T30": t30}
        if t30 <= t24:
            return False, observed, f"Inversion detected: T30 ({t30:.1f} °R) <= T24 ({t24:.1f} °R)."
        if t24 <= t2:
            return False, observed, f"Inversion detected: T24 ({t24:.1f} °R) <= T2 ({t2:.1f} °R)."
        return True, observed, "Temperature ordering invariant satisfied (T30 > T24 > T2)."

    # 2. Pressure Ordering: P30 > P15 > P2
    if constraint.constraint_id == "CONST_INVARIANT_PRESS_ORDER":
        p2 = _extract_variable(state_dict, "P2")
        p15 = _extract_variable(state_dict, "P15")
        p30 = _extract_variable(state_dict, "P30")

        if p2 is None or p15 is None or p30 is None:
            return True, None, "Insufficient pressure variables to evaluate ordering."

        observed = {"P2": p2, "P15": p15, "P30": p30}
        if p30 <= p15:
            return False, observed, f"Inversion detected: P30 ({p30:.1f} psia) <= P15 ({p15:.1f} psia)."
        if p15 <= p2:
            return False, observed, f"Inversion detected: P15 ({p15:.1f} psia) <= P2 ({p2:.1f} psia)."
        return True, observed, "Pressure ordering invariant satisfied (P30 > P15 > P2)."

    # 3. Spool Speed Ordering: Nc > Nf
    if constraint.constraint_id == "CONST_INVARIANT_SPOOL_ORDER":
        nf = _extract_variable(state_dict, "Nf")
        nc = _extract_variable(state_dict, "Nc")

        if nf is None or nc is None:
            return True, None, "Insufficient spool speed variables to evaluate ordering."

        observed = {"Nf": nf, "Nc": nc}
        if nc <= nf:
            return False, observed, f"Inversion detected: Core speed Nc ({nc:.1f} rpm) <= Fan speed Nf ({nf:.1f} rpm)."
        return True, observed, "Spool speed invariant satisfied (Nc > Nf)."

    return True, None, "Unknown invariant."
