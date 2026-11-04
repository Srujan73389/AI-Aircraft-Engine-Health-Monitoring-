"""
Physics Engine Module - Simplified Turbofan Aerothermal and Mechanical Calculations
Evaluates defensible thermodynamic and mechanical indicators for the Digital Twin.

Roadmap Reference: Phase 5 - Step 5.5
"""

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Union
import numpy as np


# Standard Thermodynamic Constants for Dry Air
STANDARD_GAMMA_AIR = 1.400                 # Ratio of specific heats for dry air
STANDARD_CP_AIR = 0.240                    # Specific heat at constant pressure in BTU/(lbm * deg R)
STANDARD_T_STD_RANKINE = 518.67            # Sea-level standard stagnation temperature (518.67 deg R = 59 deg F = 15 deg C)


def _validate_finite_positive(val: Any, name: str) -> float:
    """
    Validates that a numeric input is a real number, strictly finite (not NaN, not Inf),
    and strictly positive (> 0.0).

    Args:
        val: Input value to validate.
        name: Parameter name for informative error messages.

    Returns:
        Validated float value.

    Raises:
        TypeError: If val is not an int, float, or numpy numeric scalar.
        ValueError: If val is NaN, Inf, or <= 0.0.
    """
    if val is None:
        raise TypeError(f"{name} cannot be None; must be a finite positive number.")
    if not isinstance(val, (int, float, np.number)):
        raise TypeError(f"{name} must be a numeric type (int, float), got {type(val).__name__}.")
    val_f = float(val)
    if not math.isfinite(val_f):
        raise ValueError(f"{name} must be finite (not NaN or Inf), got {val_f}.")
    if val_f <= 0.0:
        raise ValueError(f"{name} must be strictly positive (> 0.0), got {val_f}.")
    return val_f


@dataclass
class PhysicsIndicators:
    """
    Data container holding defensible physics-based aerothermal and mechanical indicators.
    """
    delta_t_hpc: float                     # T30 - T24 (deg R)
    delta_t_booster: float                 # T24 - T2 (deg R)
    total_temp_ratio: float                # T30 / T2 (dimensionless)
    overall_pressure_ratio: float          # P30 / P2 (dimensionless)
    bypass_pressure_ratio: float           # P15 / P2 (dimensionless)
    comp_isentropic_eff: float             # Overall compression isentropic efficiency (unclipped float)
    comp_polytropic_eff: float             # Overall compression polytropic efficiency (unclipped float)
    comp_specific_work: float              # Approximate compression-stream specific enthalpy rise (BTU/lbm)
    spool_speed_ratio_phys: float          # Nc / Nf (dimensionless)
    spool_speed_ratio_corr: float          # NRc / NRf (dimensionless)
    fan_speed_theoretical_corr: float      # Nf / sqrt(T2 / 518.67) (rpm)
    fan_speed_corr_residual: float         # NRf - fan_speed_theoretical_corr (rpm)
    core_speed_theoretical_corr: float     # Nc / sqrt(T24 / 518.67) (rpm)
    core_speed_corr_residual: float        # NRc - core_speed_theoretical_corr (rpm)
    warnings: List[str]                    # Any thermodynamic boundary or plausibility warnings

    def to_dict(self) -> Dict[str, Any]:
        """Convert indicators to dictionary."""
        return asdict(self)


def compute_compression_isentropic_efficiency(
    p2: float,
    p30: float,
    t2: float,
    t30: float,
    gamma: float = STANDARD_GAMMA_AIR,
) -> float:
    """
    Calculates overall compression system isentropic efficiency (Fan hub + Booster + HPC)
    as a lumped-system aerothermal indicator.

    Formula:
        eta_is = ((P30 / P2) ** ((gamma - 1) / gamma) - 1.0) / ((T30 / T2) - 1.0)

    Assumptions & Scope:
        - Assumes ideal gas with constant ratio of specific heats gamma = 1.400.
        - Models overall compression train as a single lumped stage; does not isolate
          individual component efficiencies.
        - The calculated value is an unclipped mathematical result. Values outside the
          nominal turbofan physical range [0.75, 0.98] reflect thermodynamic inconsistency,
          extreme degradation, or anomalous telemetry and must be flagged via diagnostic
          warnings rather than silently clipped.

    Args:
        p2: Fan inlet total pressure (psia, finite and > 0).
        p30: HPC exit total pressure (psia, finite and > p2).
        t2: Fan inlet total temperature (deg R, finite and > 0).
        t30: HPC exit total temperature (deg R, finite and > t2).
        gamma: Ratio of specific heats (finite and > 1.0, default: 1.400).

    Returns:
        Isentropic efficiency as an unclipped float.

    Raises:
        TypeError: If any input is not numeric or is None.
        ValueError: If inputs are non-finite, non-positive, or violate thermodynamic bounds (P30 <= P2, T30 <= T2).
    """
    p2_f = _validate_finite_positive(p2, "p2")
    p30_f = _validate_finite_positive(p30, "p30")
    t2_f = _validate_finite_positive(t2, "t2")
    t30_f = _validate_finite_positive(t30, "t30")
    gamma_f = _validate_finite_positive(gamma, "gamma")

    if gamma_f <= 1.0:
        raise ValueError(f"Ratio of specific heats gamma must be > 1.0, got {gamma_f}.")
    if p30_f <= p2_f:
        raise ValueError(f"Compression requires P30 > P2. Got P30={p30_f}, P2={p2_f}.")
    if t30_f <= t2_f:
        raise ValueError(f"Compression work requires T30 > T2. Got T30={t30_f}, T2={t2_f}.")

    exponent = (gamma_f - 1.0) / gamma_f
    opr = p30_f / p2_f
    t_ratio = t30_f / t2_f

    eta_is = (math.pow(opr, exponent) - 1.0) / (t_ratio - 1.0)
    return float(eta_is)


def compute_compression_polytropic_efficiency(
    p2: float,
    p30: float,
    t2: float,
    t30: float,
    gamma: float = STANDARD_GAMMA_AIR,
) -> float:
    """
    Calculates overall compression system polytropic (infinitesimal-stage) efficiency
    as a lumped-system aerothermal indicator.

    Formula:
        eta_poly = (((gamma - 1) / gamma) * ln(P30 / P2)) / ln(T30 / T2)

    Assumptions & Scope:
        - Assumes ideal gas with constant ratio of specific heats gamma = 1.400.
        - Unclipped mathematical result; nominal values for modern turbofan compression
          systems lie in [0.80, 0.99].
        - Values outside nominal bounds trigger diagnostic warnings without silent clipping.

    Args:
        p2: Fan inlet total pressure (psia, finite and > 0).
        p30: HPC exit total pressure (psia, finite and > p2).
        t2: Fan inlet total temperature (deg R, finite and > 0).
        t30: HPC exit total temperature (deg R, finite and > t2).
        gamma: Ratio of specific heats (finite and > 1.0, default: 1.400).

    Returns:
        Polytropic efficiency as an unclipped float.

    Raises:
        TypeError: If any input is not numeric or is None.
        ValueError: If inputs are non-finite, non-positive, or violate thermodynamic bounds (P30 <= P2, T30 <= T2).
    """
    p2_f = _validate_finite_positive(p2, "p2")
    p30_f = _validate_finite_positive(p30, "p30")
    t2_f = _validate_finite_positive(t2, "t2")
    t30_f = _validate_finite_positive(t30, "t30")
    gamma_f = _validate_finite_positive(gamma, "gamma")

    if gamma_f <= 1.0:
        raise ValueError(f"Ratio of specific heats gamma must be > 1.0, got {gamma_f}.")
    if p30_f <= p2_f:
        raise ValueError(f"Compression requires P30 > P2. Got P30={p30_f}, P2={p2_f}.")
    if t30_f <= t2_f:
        raise ValueError(f"Compression work requires T30 > T2. Got T30={t30_f}, T2={t2_f}.")

    exponent = (gamma_f - 1.0) / gamma_f
    numerator = exponent * math.log(p30_f / p2_f)
    denominator = math.log(t30_f / t2_f)

    eta_poly = numerator / denominator
    return float(eta_poly)


def compute_compression_specific_work(
    t2: float,
    t30: float,
    cp: float = STANDARD_CP_AIR,
) -> float:
    """
    Calculates the approximate compression-stream specific enthalpy rise across the turbofan
    compression system (Fan hub + Booster + HPC).

    Formula:
        delta_h_comp = cp * (T30 - T2)

    Physical Interpretation & Assumptions:
        - This provides an approximate measure of specific enthalpy added to the gas path,
          assuming an ideal gas with constant specific heat cp = 0.240 BTU/(lbm * deg R)
          (1004.83 J/(kg * K)).
        - This is NOT an independently measured mechanical shaft work value, as turbofan shaft
          torque and mechanical power are not instrumented in C-MAPSS.
        - Under idealized adiabatic conditions with zero heat loss, compression stream specific
          enthalpy rise equals specific shaft work input: w_comp ≈ delta_h_comp.

    Limitations:
        - Neglects temperature dependence of specific heat cp(T), which increases by ~15%
          over the temperature range from 518.67 deg R (ambient) to > 1600 deg R (HPC exit).
        - Neglects inter-stage customer and turbine cooling bleed air extractions (W31, W32).
        - Neglects casing heat transfer and non-adiabatic thermal losses.

    Args:
        t2: Fan inlet total temperature (deg R, finite and > 0).
        t30: HPC exit total temperature (deg R, finite and >= t2).
        cp: Specific heat at constant pressure in BTU/(lbm * deg R) (finite and > 0, default: 0.240).

    Returns:
        Approximate compression-stream specific enthalpy rise in BTU/lbm (unclipped float).

    Raises:
        TypeError: If any input is not numeric or is None.
        ValueError: If inputs are non-finite, non-positive, or T30 < T2.
    """
    t2_f = _validate_finite_positive(t2, "t2")
    t30_f = _validate_finite_positive(t30, "t30")
    cp_f = _validate_finite_positive(cp, "cp")

    if t30_f < t2_f:
        raise ValueError(f"Compression work requires T30 >= T2. Got T30={t30_f}, T2={t2_f}.")

    return float(cp_f * (t30_f - t2_f))


def compute_spool_speed_ratios(
    nc: float,
    nf: float,
    nrc: float,
    nrf: float,
) -> Dict[str, float]:
    """
    Calculates mechanical and aerothermal spool speed ratios.

    Args:
        nc: Physical core speed (rpm, finite and > 0).
        nf: Physical fan speed (rpm, finite and > 0).
        nrc: Corrected core speed (rpm, finite and > 0).
        nrf: Corrected fan speed (rpm, finite and > 0).

    Returns:
        Dictionary with 'speed_ratio_phys' (Nc / Nf) and 'speed_ratio_corr' (NRc / NRf).

    Raises:
        TypeError: If any input is not numeric or is None.
        ValueError: If any speed is non-finite or non-positive (<= 0.0).
    """
    nc_f = _validate_finite_positive(nc, "nc")
    nf_f = _validate_finite_positive(nf, "nf")
    nrc_f = _validate_finite_positive(nrc, "nrc")
    nrf_f = _validate_finite_positive(nrf, "nrf")

    return {
        "speed_ratio_phys": float(nc_f / nf_f),
        "speed_ratio_corr": float(nrc_f / nrf_f),
    }


def compute_aerothermal_speed_corrections(
    nf: float,
    t2: float,
    nc: float,
    t24: float,
    nrf: float,
    nrc: float,
    t_std: float = STANDARD_T_STD_RANKINE,
) -> Dict[str, float]:
    """
    Evaluates aerothermal speed correction equations and calculates residual errors against telemetry.

    Formulas:
        NRf_theory = Nf / sqrt(T2 / T_std)
        NRc_theory = Nc / sqrt(T24 / T_std)

    Args:
        nf: Physical fan speed (rpm, finite and > 0).
        t2: Fan inlet temperature (deg R, finite and > 0).
        nc: Physical core speed (rpm, finite and > 0).
        t24: LPC outlet / core booster temperature (deg R, finite and > 0).
        nrf: Measured corrected fan speed (rpm, finite and > 0).
        nrc: Measured corrected core speed (rpm, finite and > 0).
        t_std: Sea-level standard temperature (finite and > 0, default: 518.67 deg R).

    Returns:
        Dictionary containing theoretical corrected speeds and residual errors (measured - theoretical).

    Raises:
        TypeError: If any input is not numeric or is None.
        ValueError: If any input is non-finite or non-positive.
    """
    nf_f = _validate_finite_positive(nf, "nf")
    t2_f = _validate_finite_positive(t2, "t2")
    nc_f = _validate_finite_positive(nc, "nc")
    t24_f = _validate_finite_positive(t24, "t24")
    nrf_f = _validate_finite_positive(nrf, "nrf")
    nrc_f = _validate_finite_positive(nrc, "nrc")
    t_std_f = _validate_finite_positive(t_std, "t_std")

    theta_2 = t2_f / t_std_f
    theta_24 = t24_f / t_std_f

    nrf_theory = nf_f / math.sqrt(theta_2)
    nrc_theory = nc_f / math.sqrt(theta_24)

    return {
        "fan_speed_theoretical_corr": float(nrf_theory),
        "fan_speed_corr_residual": float(nrf_f - nrf_theory),
        "core_speed_theoretical_corr": float(nrc_theory),
        "core_speed_corr_residual": float(nrc_f - nrc_theory),
    }


def evaluate_engine_physics(telemetry: Union[Dict[str, Any], Any]) -> PhysicsIndicators:
    """
    Evaluates all defensible physics-based indicators from an EngineState object or telemetry dictionary.
    Calculates unclipped mathematical values and raises structured diagnostic warnings for any
    plausibility boundary violations.

    Args:
        telemetry: EngineState instance or dictionary mapping variable names / C-MAPSS column names.

    Returns:
        PhysicsIndicators dataclass with unclipped indicators and diagnostic warnings.

    Raises:
        KeyError: If a required telemetry variable is missing.
        TypeError: If a telemetry value is of an invalid type.
        ValueError: If a telemetry value is None or non-finite.
    """
    if telemetry is None:
        raise TypeError("telemetry input cannot be None.")

    warnings: List[str] = []

    # Extract and validate required variables supporting both EngineState attributes and dictionary keys
    def _get(keys: List[str]) -> float:
        found = False
        val = None
        for k in keys:
            if hasattr(telemetry, k):
                val = getattr(telemetry, k)
                found = True
                break
            if isinstance(telemetry, dict) and k in telemetry:
                val = telemetry[k]
                found = True
                break

        if not found:
            raise KeyError(f"Telemetry missing required variable (keys checked: {keys})")
        if val is None:
            raise ValueError(f"Telemetry variable {keys[0]} is None; strictly finite number required.")
        if not isinstance(val, (int, float, np.number)):
            raise TypeError(f"Telemetry variable {keys[0]} must be numeric, got {type(val).__name__}.")
        val_f = float(val)
        if not math.isfinite(val_f):
            raise ValueError(f"Telemetry variable {keys[0]} must be finite (got {val_f}).")
        if val_f <= 0.0:
            raise ValueError(f"Telemetry variable {keys[0]} must be strictly positive (> 0.0), got {val_f}.")
        return val_f

    t2 = _get(["T2", "t2", "sensor_1"])
    t24 = _get(["T24", "t24", "sensor_2"])
    t30 = _get(["T30", "t30", "sensor_3"])
    p2 = _get(["P2", "p2", "sensor_5"])
    p15 = _get(["P15", "p15", "sensor_6"])
    p30 = _get(["P30", "p30", "sensor_7"])
    nf = _get(["Nf", "nf", "sensor_8"])
    nc = _get(["Nc", "nc", "sensor_9"])
    nrf = _get(["NRf", "nrf", "sensor_13"])
    nrc = _get(["NRc", "nrc", "sensor_14"])

    # Basic thermodynamic consistency checks
    if t30 <= t24:
        warnings.append(f"HPC temperature rise violation: T30 ({t30:.2f}) <= T24 ({t24:.2f})")
    if t24 <= t2:
        warnings.append(f"Booster temperature rise violation: T24 ({t24:.2f}) <= T2 ({t2:.2f})")
    if p30 <= p2:
        warnings.append(f"Overall pressure ratio violation: P30 ({p30:.2f}) <= P2 ({p2:.2f})")
    if p15 <= p2:
        warnings.append(f"Bypass pressure ratio violation: P15 ({p15:.2f}) <= P2 ({p2:.2f})")
    if nc <= nf:
        warnings.append(f"Shaft speed hierarchy violation: Nc ({nc:.2f}) <= Nf ({nf:.2f})")

    # Compute individual indicators
    delta_t_hpc = t30 - t24
    delta_t_booster = t24 - t2
    total_temp_ratio = t30 / t2 if t2 > 0 else float("nan")
    opr = p30 / p2 if p2 > 0 else float("nan")
    bypass_pr = p15 / p2 if p2 > 0 else float("nan")

    # Isentropic and polytropic efficiencies (unclipped mathematical evaluation)
    comp_is = compute_compression_isentropic_efficiency(p2, p30, t2, t30)
    if comp_is < 0.75 or comp_is > 0.98:
        warnings.append(
            f"Isentropic efficiency {comp_is:.4f} outside nominal turbofan physical range [0.75, 0.98]"
        )

    comp_poly = compute_compression_polytropic_efficiency(p2, p30, t2, t30)
    if comp_poly < 0.80 or comp_poly > 0.99:
        warnings.append(
            f"Polytropic efficiency {comp_poly:.4f} outside nominal turbofan physical range [0.80, 0.99]"
        )

    # Specific work (approximate compression-stream specific enthalpy rise)
    comp_w = compute_compression_specific_work(t2, t30)

    # Spool speed ratios
    ratios = compute_spool_speed_ratios(nc, nf, nrc, nrf)
    phys_ratio = ratios["speed_ratio_phys"]
    corr_ratio = ratios["speed_ratio_corr"]

    # Aerothermal speed corrections
    corrections = compute_aerothermal_speed_corrections(nf, t2, nc, t24, nrf, nrc)
    fan_theory = corrections["fan_speed_theoretical_corr"]
    fan_res = corrections["fan_speed_corr_residual"]
    core_theory = corrections["core_speed_theoretical_corr"]
    core_res = corrections["core_speed_corr_residual"]
    if abs(fan_res) > 2.0:
        warnings.append(f"Large fan speed correction residual: {fan_res:.2f} rpm")
    if abs(core_res) > 50.0:
        warnings.append(f"Large core speed correction residual: {core_res:.2f} rpm")

    return PhysicsIndicators(
        delta_t_hpc=float(delta_t_hpc),
        delta_t_booster=float(delta_t_booster),
        total_temp_ratio=float(total_temp_ratio),
        overall_pressure_ratio=float(opr),
        bypass_pressure_ratio=float(bypass_pr),
        comp_isentropic_eff=float(comp_is),
        comp_polytropic_eff=float(comp_poly),
        comp_specific_work=float(comp_w),
        spool_speed_ratio_phys=float(phys_ratio),
        spool_speed_ratio_corr=float(corr_ratio),
        fan_speed_theoretical_corr=float(fan_theory),
        fan_speed_corr_residual=float(fan_res),
        core_speed_theoretical_corr=float(core_theory),
        core_speed_corr_residual=float(core_res),
        warnings=warnings,
    )
