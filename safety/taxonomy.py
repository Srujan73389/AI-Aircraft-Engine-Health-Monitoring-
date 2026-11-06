"""
Variable Taxonomy and Provenance Classification Subsystem
Defines formal classifications for measured, predicted, simulated, and manually configured variables,
enforcing provenance tracking and the Digital Twin safety firewall.

Roadmap Reference: Phase 7 - Step 7.2
"""

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Dict, Any, List, Optional, Union, Tuple


class VariableCategory(str, Enum):
    """
    Formal operational provenance category of an engine state variable.
    """
    MEASURED = "MEASURED"
    PREDICTED = "PREDICTED"
    SIMULATED = "SIMULATED"
    MANUALLY_CONFIGURED = "MANUALLY_CONFIGURED"


class ProvenanceViolationError(ValueError):
    """
    Raised when an immutable telemetry variable (MEASURED) is overwritten,
    tampered with, or mutated by a simulated or predictive process.
    """
    pass


@dataclass(frozen=True)
class VariableProvenance:
    """
    Metadata specification describing the provenance, mathematical formulation,
    and immutability guarantees of a specific state variable.
    """
    variable_id: str
    canonical_name: str
    category: VariableCategory
    units: str
    description: str
    is_immutable_in_twin: bool
    associated_models_or_equations: Optional[str] = None


@dataclass
class StateVariableClassificationReport:
    """
    Structured categorization report partitioning state variables into their
    respective provenance categories.
    """
    measured: Dict[str, Any] = field(default_factory=dict)
    predicted: Dict[str, Any] = field(default_factory=dict)
    simulated: Dict[str, Any] = field(default_factory=dict)
    manually_configured: Dict[str, Any] = field(default_factory=dict)
    unclassified: Dict[str, Any] = field(default_factory=dict)
    total_variables: int = 0

    def __getitem__(self, item: str) -> Dict[str, Any]:
        lookup = {
            "measured": self.measured,
            "predicted": self.predicted,
            "simulated": self.simulated,
            "manually_configured": self.manually_configured,
            "unclassified": self.unclassified,
        }
        if item in lookup:
            return lookup[item]
        raise KeyError(f"Invalid category key: '{item}'. Valid keys: {list(lookup.keys())}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "measured": dict(self.measured),
            "predicted": dict(self.predicted),
            "simulated": dict(self.simulated),
            "manually_configured": dict(self.manually_configured),
            "unclassified": dict(self.unclassified),
            "total_variables": self.total_variables,
            "counts": {
                "MEASURED": len(self.measured),
                "PREDICTED": len(self.predicted),
                "SIMULATED": len(self.simulated),
                "MANUALLY_CONFIGURED": len(self.manually_configured),
                "UNCLASSIFIED": len(self.unclassified),
            },
        }


# =============================================================================
# COMPREHENSIVE VARIABLE TAXONOMY REGISTRY (G1 - G40 + AUXILIARIES)
# =============================================================================
VARIABLE_TAXONOMY: Dict[str, VariableProvenance] = {
    # -------------------------------------------------------------------------
    # Group A: Operational Tracking Metadata & Flight Conditions (MANUALLY_CONFIGURED)
    # -------------------------------------------------------------------------
    "G1": VariableProvenance(
        variable_id="G1",
        canonical_name="unit_number",
        category=VariableCategory.MANUALLY_CONFIGURED,
        units="dimensionless",
        description="Discrete engine unit/tail serial identifier.",
        is_immutable_in_twin=False,
        associated_models_or_equations="Fleet registration database index",
    ),
    "G2": VariableProvenance(
        variable_id="G2",
        canonical_name="time_cycles",
        category=VariableCategory.MANUALLY_CONFIGURED,
        units="cycles",
        description="Discrete operational flight cycle clock counter.",
        is_immutable_in_twin=False,
        associated_models_or_equations="Flight log cycle tracking",
    ),
    "G3": VariableProvenance(
        variable_id="G3",
        canonical_name="altitude",
        category=VariableCategory.MANUALLY_CONFIGURED,
        units="kft",
        description="Flight pressure altitude operating condition (op_setting_1).",
        is_immutable_in_twin=False,
        associated_models_or_equations="Flight mission profile schedule",
    ),
    "G4": VariableProvenance(
        variable_id="G4",
        canonical_name="mach_number",
        category=VariableCategory.MANUALLY_CONFIGURED,
        units="dimensionless",
        description="Flight Mach number operating condition (op_setting_2).",
        is_immutable_in_twin=False,
        associated_models_or_equations="Flight management system target speed",
    ),
    "G5": VariableProvenance(
        variable_id="G5",
        canonical_name="throttle_resolver_angle",
        category=VariableCategory.MANUALLY_CONFIGURED,
        units="TRA %",
        description="Throttle Resolver Angle pilot command setting (op_setting_3).",
        is_immutable_in_twin=False,
        associated_models_or_equations="Cockpit power lever position command",
    ),

    # -------------------------------------------------------------------------
    # Group B: Measured Gas-Path Sensors (MEASURED - IMMUTABLE)
    # -------------------------------------------------------------------------
    "G6": VariableProvenance(
        variable_id="G6",
        canonical_name="T2",
        category=VariableCategory.MEASURED,
        units="°R",
        description="Total temperature at fan inlet station (sensor_1).",
        is_immutable_in_twin=True,
        associated_models_or_equations="Thermocouple / RTD telemetry probe",
    ),
    "G7": VariableProvenance(
        variable_id="G7",
        canonical_name="T24",
        category=VariableCategory.MEASURED,
        units="°R",
        description="Total temperature at LPC outlet station (sensor_2).",
        is_immutable_in_twin=True,
        associated_models_or_equations="Dual-junction thermocouple probe",
    ),
    "G8": VariableProvenance(
        variable_id="G8",
        canonical_name="T30",
        category=VariableCategory.MEASURED,
        units="°R",
        description="Total temperature at HPC outlet station (sensor_3).",
        is_immutable_in_twin=True,
        associated_models_or_equations="High-temperature thermocouple probe",
    ),
    "G9": VariableProvenance(
        variable_id="G9",
        canonical_name="T50",
        category=VariableCategory.MEASURED,
        units="°R",
        description="Total temperature at LPT outlet station (sensor_4).",
        is_immutable_in_twin=True,
        associated_models_or_equations="Exhaust gas temperature (EGT) pyrometer/rake",
    ),
    "G10": VariableProvenance(
        variable_id="G10",
        canonical_name="P2",
        category=VariableCategory.MEASURED,
        units="psia",
        description="Total pressure at fan inlet station (sensor_5).",
        is_immutable_in_twin=True,
        associated_models_or_equations="Inlet pitot-static pressure transducer",
    ),
    "G11": VariableProvenance(
        variable_id="G11",
        canonical_name="P15",
        category=VariableCategory.MEASURED,
        units="psia",
        description="Total pressure in fan bypass duct station (sensor_6).",
        is_immutable_in_twin=True,
        associated_models_or_equations="Bypass duct total pressure transducer",
    ),
    "G12": VariableProvenance(
        variable_id="G12",
        canonical_name="P30",
        category=VariableCategory.MEASURED,
        units="psia",
        description="Total pressure at HPC outlet station (sensor_7).",
        is_immutable_in_twin=True,
        associated_models_or_equations="Combustor casing total pressure transducer",
    ),
    "G13": VariableProvenance(
        variable_id="G13",
        canonical_name="Nf",
        category=VariableCategory.MEASURED,
        units="rpm",
        description="Physical low-pressure fan spool speed (sensor_8).",
        is_immutable_in_twin=True,
        associated_models_or_equations="Variable reluctance / eddy current speed pickup",
    ),
    "G14": VariableProvenance(
        variable_id="G14",
        canonical_name="Nc",
        category=VariableCategory.MEASURED,
        units="rpm",
        description="Physical high-pressure core spool speed (sensor_9).",
        is_immutable_in_twin=True,
        associated_models_or_equations="Core shaft phonic wheel speed sensor",
    ),
    "G15": VariableProvenance(
        variable_id="G15",
        canonical_name="epr",
        category=VariableCategory.MEASURED,
        units="dimensionless",
        description="Engine Pressure Ratio P50/P2 (sensor_10).",
        is_immutable_in_twin=True,
        associated_models_or_equations="Differential pressure ratio transmitter",
    ),
    "G16": VariableProvenance(
        variable_id="G16",
        canonical_name="Ps30",
        category=VariableCategory.MEASURED,
        units="psia",
        description="Static pressure at HPC outlet station (sensor_11).",
        is_immutable_in_twin=True,
        associated_models_or_equations="HPC exit wall static pressure tap transducer",
    ),
    "G17": VariableProvenance(
        variable_id="G17",
        canonical_name="phi",
        category=VariableCategory.MEASURED,
        units="pps/psia",
        description="Ratio of fuel flow to static pressure Ps30 (sensor_12).",
        is_immutable_in_twin=True,
        associated_models_or_equations="Fuel flow meter signal scaled by Ps30",
    ),
    "G18": VariableProvenance(
        variable_id="G18",
        canonical_name="NRf",
        category=VariableCategory.MEASURED,
        units="rpm",
        description="Corrected low-pressure fan spool speed (sensor_13).",
        is_immutable_in_twin=True,
        associated_models_or_equations="Nf / sqrt(theta_2)",
    ),
    "G19": VariableProvenance(
        variable_id="G19",
        canonical_name="NRc",
        category=VariableCategory.MEASURED,
        units="rpm",
        description="Corrected high-pressure core spool speed (sensor_14).",
        is_immutable_in_twin=True,
        associated_models_or_equations="Nc / sqrt(theta_25)",
    ),
    "G20": VariableProvenance(
        variable_id="G20",
        canonical_name="BPR",
        category=VariableCategory.MEASURED,
        units="dimensionless",
        description="Bypass ratio of bypass air to core air (sensor_15).",
        is_immutable_in_twin=True,
        associated_models_or_equations="Aerodynamic mass flow split telemetry",
    ),
    "G21": VariableProvenance(
        variable_id="G21",
        canonical_name="farB",
        category=VariableCategory.MEASURED,
        units="dimensionless",
        description="Burner fuel-to-air mass ratio (sensor_16).",
        is_immutable_in_twin=True,
        associated_models_or_equations="Combustor stoichiometric fuel-air metering",
    ),
    "G22": VariableProvenance(
        variable_id="G22",
        canonical_name="htBleed",
        category=VariableCategory.MEASURED,
        units="dimensionless ratio",
        description="Bleed air enthalpy parameter (sensor_17).",
        is_immutable_in_twin=True,
        associated_models_or_equations="Customer / thermal management bleed enthalpy",
    ),
    "G23": VariableProvenance(
        variable_id="G23",
        canonical_name="Nf_dmd",
        category=VariableCategory.MEASURED,
        units="rpm",
        description="FADEC demanded fan spool speed (sensor_18).",
        is_immutable_in_twin=True,
        associated_models_or_equations="FADEC electronic control unit demand",
    ),
    "G24": VariableProvenance(
        variable_id="G24",
        canonical_name="PCNfR_dmd",
        category=VariableCategory.MEASURED,
        units="%",
        description="Demanded corrected fan speed percentage (sensor_19).",
        is_immutable_in_twin=True,
        associated_models_or_equations="FADEC closed-loop reference setpoint",
    ),
    "G25": VariableProvenance(
        variable_id="G25",
        canonical_name="W31",
        category=VariableCategory.MEASURED,
        units="lbm/s",
        description="HPT coolant bleed air mass flow rate (sensor_20).",
        is_immutable_in_twin=True,
        associated_models_or_equations="Turbine cooling valve flow telemetry",
    ),
    "G26": VariableProvenance(
        variable_id="G26",
        canonical_name="W32",
        category=VariableCategory.MEASURED,
        units="lbm/s",
        description="LPT coolant bleed air mass flow rate (sensor_21).",
        is_immutable_in_twin=True,
        associated_models_or_equations="LPT cooling modulation telemetry",
    ),

    # -------------------------------------------------------------------------
    # Group C: Simulated / Derived Macroscopic Indicators (SIMULATED)
    # -------------------------------------------------------------------------
    "G27": VariableProvenance(
        variable_id="G27",
        canonical_name="hpc_temp_rise",
        category=VariableCategory.SIMULATED,
        units="°R",
        description="Compression temperature rise across HPC.",
        is_immutable_in_twin=False,
        associated_models_or_equations="delta_T = T30 - T24",
    ),
    "G28": VariableProvenance(
        variable_id="G28",
        canonical_name="overall_pressure_ratio",
        category=VariableCategory.SIMULATED,
        units="dimensionless",
        description="Overall engine compressor pressure ratio (OPR).",
        is_immutable_in_twin=False,
        associated_models_or_equations="OPR = P30 / P2",
    ),
    "G29": VariableProvenance(
        variable_id="G29",
        canonical_name="spool_speed_ratio",
        category=VariableCategory.SIMULATED,
        units="dimensionless",
        description="Ratio of core speed to fan speed.",
        is_immutable_in_twin=False,
        associated_models_or_equations="spool_ratio = Nc / Nf",
    ),
    "G30": VariableProvenance(
        variable_id="G30",
        canonical_name="normalized_feature_vec",
        category=VariableCategory.SIMULATED,
        units="normalized [0, 1]",
        description="SafeMinMaxScaler normalized feature representation.",
        is_immutable_in_twin=False,
        associated_models_or_equations="SafeMinMaxScaler(G3 - G26)",
    ),
    "G31": VariableProvenance(
        variable_id="G31",
        canonical_name="temporal_seq_tensor",
        category=VariableCategory.SIMULATED,
        units="normalized [0, 1]",
        description="Rolling 30-cycle temporal sliding window tensor.",
        is_immutable_in_twin=False,
        associated_models_or_equations="Sliding window buffer [30 x D]",
    ),
    "G39": VariableProvenance(
        variable_id="G39",
        canonical_name="thermal_stress_offset",
        category=VariableCategory.SIMULATED,
        units="°R",
        description="What-if simulated thermal degradation offset delta T.",
        is_immutable_in_twin=False,
        associated_models_or_equations="Synthetic perturbation: T30_sim = T30_meas + delta_T",
    ),
    "G40": VariableProvenance(
        variable_id="G40",
        canonical_name="sensor_bias_delta",
        category=VariableCategory.SIMULATED,
        units="sensor units",
        description="What-if simulated sensor drift and calibration bias dictionary.",
        is_immutable_in_twin=False,
        associated_models_or_equations="Synthetic perturbation: S_sim = S_meas + delta_bias",
    ),

    # -------------------------------------------------------------------------
    # Group D: Predicted Machine Learning & Health States (PREDICTED)
    # -------------------------------------------------------------------------
    "G32": VariableProvenance(
        variable_id="G32",
        canonical_name="anomaly_score_m1",
        category=VariableCategory.PREDICTED,
        units="anomaly score",
        description="Unsupervised anomaly score from Isolation Forest ensemble (Model 1).",
        is_immutable_in_twin=False,
        associated_models_or_equations="IsolationForest.score_samples()",
    ),
    "G33": VariableProvenance(
        variable_id="G33",
        canonical_name="anomaly_score_m2",
        category=VariableCategory.PREDICTED,
        units="residual energy (Q-statistic)",
        description="PCA Squared Prediction Error (SPE) reconstruction residual (Model 2).",
        is_immutable_in_twin=False,
        associated_models_or_equations="SPE = ||x - x_reconstructed||^2",
    ),
    "G34": VariableProvenance(
        variable_id="G34",
        canonical_name="anomaly_alert_flag",
        category=VariableCategory.PREDICTED,
        units="binary flag {0, 1}",
        description="Multi-model 3-cycle persistent anomaly alarm flag.",
        is_immutable_in_twin=False,
        associated_models_or_equations="Temporal persistence filter: sum_{t-2}^t (alert) >= 3",
    ),
    "G35": VariableProvenance(
        variable_id="G35",
        canonical_name="predicted_rul_xgboost",
        category=VariableCategory.PREDICTED,
        units="flight cycles",
        description="Remaining Useful Life predicted by Phase 4 XGBoost champion model.",
        is_immutable_in_twin=False,
        associated_models_or_equations="XGBRegressor.predict(features)",
    ),
    "G36": VariableProvenance(
        variable_id="G36",
        canonical_name="predicted_rul_lstm",
        category=VariableCategory.PREDICTED,
        units="flight cycles",
        description="Remaining Useful Life predicted by Phase 4 PyTorch LSTM champion model.",
        is_immutable_in_twin=False,
        associated_models_or_equations="BiDirectionalLSTM.forward(seq_tensor)",
    ),
    "G37": VariableProvenance(
        variable_id="G37",
        canonical_name="prognostic_spread",
        category=VariableCategory.PREDICTED,
        units="flight cycles",
        description="Epistemic prognostic uncertainty spread |RUL_xgb - RUL_lstm|.",
        is_immutable_in_twin=False,
        associated_models_or_equations="|G35 - G36|",
    ),
    "G38": VariableProvenance(
        variable_id="G38",
        canonical_name="virtual_health_index",
        category=VariableCategory.PREDICTED,
        units="index [0, 1]",
        description="Virtual continuous health index degradation state H(t).",
        is_immutable_in_twin=False,
        associated_models_or_equations="H(t) = 1.0 - (1.0 - RUL_norm)^alpha",
    ),

    # -------------------------------------------------------------------------
    # Auxiliary Operational Thresholds & Configuration (MANUALLY_CONFIGURED)
    # -------------------------------------------------------------------------
    "CFG_SAMPLING_FREQ": VariableProvenance(
        variable_id="CFG_SAMPLING_FREQ",
        canonical_name="sampling_frequency_hz",
        category=VariableCategory.MANUALLY_CONFIGURED,
        units="Hz",
        description="Flight data acquisition sampling frequency.",
        is_immutable_in_twin=False,
        associated_models_or_equations="FADEC data bus configuration",
    ),
    "CFG_SUBSET": VariableProvenance(
        variable_id="CFG_SUBSET",
        canonical_name="baseline_subset",
        category=VariableCategory.MANUALLY_CONFIGURED,
        units="string",
        description="Operating sub-dataset profile (FD001, FD002, FD003, FD004).",
        is_immutable_in_twin=False,
        associated_models_or_equations="Fleet deployment configuration",
    ),
    "CFG_T30_CAUTION": VariableProvenance(
        variable_id="CFG_T30_CAUTION",
        canonical_name="caution_threshold_t30",
        category=VariableCategory.MANUALLY_CONFIGURED,
        units="°R",
        description="Configured operational caution threshold for HPC exit temperature.",
        is_immutable_in_twin=False,
        associated_models_or_equations="Airline maintenance policy threshold",
    ),
    "CFG_T30_WARNING": VariableProvenance(
        variable_id="CFG_T30_WARNING",
        canonical_name="warning_threshold_t30",
        category=VariableCategory.MANUALLY_CONFIGURED,
        units="°R",
        description="Configured operational warning hard limit for HPC exit temperature.",
        is_immutable_in_twin=False,
        associated_models_or_equations="OEM engine operations manual limit",
    ),
    "M1": VariableProvenance(
        variable_id="M1",
        canonical_name="isolation_forest_anomaly_score",
        category=VariableCategory.PREDICTED,
        units="anomaly score",
        description="Model 1 Isolation Forest anomaly score alias.",
        is_immutable_in_twin=False,
        associated_models_or_equations="Isolation Forest anomaly ensemble",
    ),
    "M2": VariableProvenance(
        variable_id="M2",
        canonical_name="pca_spe_residual_energy",
        category=VariableCategory.PREDICTED,
        units="residual energy (Q-stat)",
        description="Model 2 PCA-SPE reconstruction residual energy alias.",
        is_immutable_in_twin=False,
        associated_models_or_equations="PCA reconstruction error ||x - x_hat||^2",
    ),
}

# Fast lookup map from canonical name to provenance
_CANONICAL_LOOKUP: Dict[str, VariableProvenance] = {
    prov.canonical_name: prov for prov in VARIABLE_TAXONOMY.values()
}

# Immutable measured variable names set (G6 - G26)
MEASURED_VARIABLE_NAMES = {
    prov.canonical_name for prov in VARIABLE_TAXONOMY.values() if prov.category == VariableCategory.MEASURED
}


def get_variable_provenance(key: str) -> Optional[VariableProvenance]:
    """
    Retrieve provenance metadata by variable_id (e.g. 'G8', 'G35') or canonical name (e.g. 'T30', 'predicted_rul_xgboost').
    """
    if key in VARIABLE_TAXONOMY:
        return VARIABLE_TAXONOMY[key]
    if key in _CANONICAL_LOOKUP:
        return _CANONICAL_LOOKUP[key]

    # Handle prefixed keys like "G8_T30"
    if "_" in key:
        prefix = key.split("_")[0]
        if prefix in VARIABLE_TAXONOMY:
            return VARIABLE_TAXONOMY[prefix]
        rest = key[len(prefix) + 1 :]
        if rest in _CANONICAL_LOOKUP:
            return _CANONICAL_LOOKUP[rest]

    return None


# =============================================================================
# PROVENANCE INSPECTION & VERIFICATION HELPERS
# =============================================================================
def classify_state_variables(state_dict_or_engine_state: Union[Dict[str, Any], Any]) -> StateVariableClassificationReport:
    """
    Inspects an engine state representation and partitions its populated variables
    into Measured, Predicted, Simulated, and Manually Configured subsets.

    Parameters
    ----------
    state_dict_or_engine_state : Union[Dict[str, Any], EngineState]
        Engine state vector or dictionary.

    Returns
    -------
    StateVariableClassificationReport
        Partitioned variables with attribute and dictionary access.
    """
    # 1. Normalize input to dictionary
    if hasattr(state_dict_or_engine_state, "to_dict"):
        state_dict = state_dict_or_engine_state.to_dict(include_arrays=True)
    elif isinstance(state_dict_or_engine_state, dict):
        state_dict = dict(state_dict_or_engine_state)
    elif hasattr(state_dict_or_engine_state, "__dict__"):
        state_dict = {
            k: v
            for k, v in vars(state_dict_or_engine_state).items()
            if not k.startswith("_") and not callable(v)
        }
    else:
        state_dict = {}

    report = StateVariableClassificationReport()

    for raw_key, value in state_dict.items():
        prov = get_variable_provenance(raw_key)

        canonical = prov.canonical_name if prov else raw_key

        if prov is None:
            # Check heuristic aliases
            if raw_key in ["unit_number", "time_cycles", "altitude", "mach_number", "throttle_resolver_angle"]:
                report.manually_configured[canonical] = value
            elif raw_key.startswith("sensor_") or raw_key in MEASURED_VARIABLE_NAMES:
                report.measured[canonical] = value
            elif "rul" in raw_key.lower() or "score" in raw_key.lower() or "health" in raw_key.lower():
                report.predicted[canonical] = value
            elif "temp_rise" in raw_key.lower() or "ratio" in raw_key.lower() or "offset" in raw_key.lower():
                report.simulated[canonical] = value
            else:
                report.unclassified[canonical] = value
            continue

        if prov.category == VariableCategory.MEASURED:
            report.measured[canonical] = value
        elif prov.category == VariableCategory.PREDICTED:
            report.predicted[canonical] = value
        elif prov.category == VariableCategory.SIMULATED:
            report.simulated[canonical] = value
        elif prov.category == VariableCategory.MANUALLY_CONFIGURED:
            report.manually_configured[canonical] = value
        else:
            report.unclassified[canonical] = value

    report.total_variables = (
        len(report.measured)
        + len(report.predicted)
        + len(report.simulated)
        + len(report.manually_configured)
        + len(report.unclassified)
    )

    return report


def verify_provenance_immutability(
    original_state: Union[Dict[str, Any], Any],
    updated_state: Union[Dict[str, Any], Any],
) -> bool:
    """
    Asserts that no MEASURED physical variable (G6 through G26) was overwritten,
    mutated, or tampered with between an original state and an updated state.

    Parameters
    ----------
    original_state : Union[Dict[str, Any], EngineState]
        Pre-enrichment / pre-agent baseline state vector.
    updated_state : Union[Dict[str, Any], EngineState]
        Post-enrichment state vector or dictionary.

    Returns
    -------
    bool
        True if all measured variables remain identical.

    Raises
    ------
    ProvenanceViolationError
        If any MEASURED sensor variable exhibits a difference.
    """
    orig_rep = classify_state_variables(original_state)
    upd_rep = classify_state_variables(updated_state)

    orig_meas = orig_rep.measured
    upd_meas = upd_rep.measured

    # Check all measured variables present in the original state
    for var_name, orig_val in orig_meas.items():
        if var_name not in upd_meas:
            continue

        upd_val = upd_meas[var_name]

        if orig_val is None and upd_val is None:
            continue

        if orig_val is None or upd_val is None:
            raise ProvenanceViolationError(
                f"Provenance immutability violation: Measured variable '{var_name}' existence mutated: "
                f"original={orig_val}, updated={upd_val}."
            )

        # Numeric comparison
        try:
            val_orig_f = float(orig_val)
            val_upd_f = float(upd_val)
            if not math.isclose(val_orig_f, val_upd_f, rel_tol=1e-9, abs_tol=1e-9):
                raise ProvenanceViolationError(
                    f"Provenance immutability violation: Measured telemetry '{var_name}' altered! "
                    f"Original: {val_orig_f}, Updated: {val_upd_f} (|diff| = {abs(val_orig_f - val_upd_f):.6e})."
                )
        except (ValueError, TypeError):
            if orig_val != upd_val:
                raise ProvenanceViolationError(
                    f"Provenance immutability violation: Measured telemetry '{var_name}' altered! "
                    f"Original: {orig_val}, Updated: {upd_val}."
                )

    return True
