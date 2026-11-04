"""
EngineState Class - Virtual Engine State Vector Representation (G1 - G40)
Defines the 40-entry state vector for the Aircraft Engine Digital Twin.

Roadmap Reference: Phase 5 - Steps 5.1, 5.2, 5.3, 5.4
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, List
import numpy as np


class StateClassification(str, Enum):
    MEASURED = "Measured (M)"
    DERIVED = "Derived (D)"
    PREDICTED = "Predicted (P)"
    SIMULATED = "Simulated (S)"


# Complete 40-Variable Canonical State Taxonomy (G1 - G40)
STATE_TAXONOMY: Dict[str, Dict[str, Any]] = {
    "G1": {
        "field_name": "unit_number",
        "name": "Engine Unit ID",
        "classification": StateClassification.MEASURED,
        "category": "Operational Tracking Metadata",
        "unit": "dimensionless",
        "c_mapss_col": "unit_number",
        "is_simulation_only": False,
        "description": "Discrete engine unit trajectory identifier.",
    },
    "G2": {
        "field_name": "time_cycles",
        "name": "Operational Flight Cycle",
        "classification": StateClassification.MEASURED,
        "category": "Operational Tracking Metadata",
        "unit": "cycles",
        "c_mapss_col": "time_cycles",
        "is_simulation_only": False,
        "description": "Discrete operational cycle clock index.",
    },
    "G3": {
        "field_name": "altitude",
        "name": "Pressure Altitude",
        "classification": StateClassification.MEASURED,
        "category": "Operating Context & Environmental Settings",
        "unit": "kft",
        "c_mapss_col": "op_setting_1",
        "is_simulation_only": False,
        "description": "Flight pressure altitude (0 to 42 kft).",
    },
    "G4": {
        "field_name": "mach_number",
        "name": "Flight Mach Number",
        "classification": StateClassification.MEASURED,
        "category": "Operating Context & Environmental Settings",
        "unit": "dimensionless",
        "c_mapss_col": "op_setting_2",
        "is_simulation_only": False,
        "description": "Flight Mach number (0.0 to 0.84).",
    },
    "G5": {
        "field_name": "throttle_resolver_angle",
        "name": "Throttle Resolver Angle",
        "classification": StateClassification.MEASURED,
        "category": "Operating Context & Environmental Settings",
        "unit": "TRA %",
        "c_mapss_col": "op_setting_3",
        "is_simulation_only": False,
        "description": "Power lever resolver angle (20% to 100%).",
    },
    "G6": {
        "field_name": "T2",
        "name": "Fan Inlet Total Temperature",
        "classification": StateClassification.MEASURED,
        "category": "Gas-Path Thermodynamic Sensors",
        "unit": "°R",
        "c_mapss_col": "sensor_1",
        "is_simulation_only": False,
        "description": "Total temperature at fan inlet station.",
    },
    "G7": {
        "field_name": "T24",
        "name": "LPC Outlet Total Temperature",
        "classification": StateClassification.MEASURED,
        "category": "Gas-Path Thermodynamic Sensors",
        "unit": "°R",
        "c_mapss_col": "sensor_2",
        "is_simulation_only": False,
        "description": "Total temperature at Low-Pressure Compressor outlet.",
    },
    "G8": {
        "field_name": "T30",
        "name": "HPC Outlet Total Temperature",
        "classification": StateClassification.MEASURED,
        "category": "Gas-Path Thermodynamic Sensors",
        "unit": "°R",
        "c_mapss_col": "sensor_3",
        "is_simulation_only": False,
        "description": "Total temperature at High-Pressure Compressor outlet.",
    },
    "G9": {
        "field_name": "T50",
        "name": "LPT Outlet Total Temperature",
        "classification": StateClassification.MEASURED,
        "category": "Gas-Path Thermodynamic Sensors",
        "unit": "°R",
        "c_mapss_col": "sensor_4",
        "is_simulation_only": False,
        "description": "Total temperature at Low-Pressure Turbine outlet.",
    },
    "G10": {
        "field_name": "P2",
        "name": "Fan Inlet Total Pressure",
        "classification": StateClassification.MEASURED,
        "category": "Gas-Path Thermodynamic Sensors",
        "unit": "psia",
        "c_mapss_col": "sensor_5",
        "is_simulation_only": False,
        "description": "Total pressure at fan inlet station.",
    },
    "G11": {
        "field_name": "P15",
        "name": "Bypass Duct Total Pressure",
        "classification": StateClassification.MEASURED,
        "category": "Gas-Path Thermodynamic Sensors",
        "unit": "psia",
        "c_mapss_col": "sensor_6",
        "is_simulation_only": False,
        "description": "Total pressure in fan bypass duct station.",
    },
    "G12": {
        "field_name": "P30",
        "name": "HPC Outlet Total Pressure",
        "classification": StateClassification.MEASURED,
        "category": "Gas-Path Thermodynamic Sensors",
        "unit": "psia",
        "c_mapss_col": "sensor_7",
        "is_simulation_only": False,
        "description": "Total pressure at High-Pressure Compressor outlet.",
    },
    "G13": {
        "field_name": "Nf",
        "name": "Physical Fan Speed",
        "classification": StateClassification.MEASURED,
        "category": "Gas-Path Thermodynamic Sensors",
        "unit": "rpm",
        "c_mapss_col": "sensor_8",
        "is_simulation_only": False,
        "description": "Physical rotational speed of the low-pressure fan spool.",
    },
    "G14": {
        "field_name": "Nc",
        "name": "Physical Core Speed",
        "classification": StateClassification.MEASURED,
        "category": "Gas-Path Thermodynamic Sensors",
        "unit": "rpm",
        "c_mapss_col": "sensor_9",
        "is_simulation_only": False,
        "description": "Physical rotational speed of the high-pressure core spool.",
    },
    "G15": {
        "field_name": "epr",
        "name": "Engine Pressure Ratio",
        "classification": StateClassification.MEASURED,
        "category": "Gas-Path Thermodynamic Sensors",
        "unit": "dimensionless",
        "c_mapss_col": "sensor_10",
        "is_simulation_only": False,
        "description": "Engine pressure ratio (P50/P2).",
    },
    "G16": {
        "field_name": "Ps30",
        "name": "HPC Outlet Static Pressure",
        "classification": StateClassification.MEASURED,
        "category": "Gas-Path Thermodynamic Sensors",
        "unit": "psia",
        "c_mapss_col": "sensor_11",
        "is_simulation_only": False,
        "description": "Static pressure at High-Pressure Compressor outlet station.",
    },
    "G17": {
        "field_name": "phi",
        "name": "Fuel Flow to Ps30 Ratio",
        "classification": StateClassification.MEASURED,
        "category": "Gas-Path Thermodynamic Sensors",
        "unit": "pps/psia",
        "c_mapss_col": "sensor_12",
        "is_simulation_only": False,
        "description": "Ratio of fuel mass flow rate to Ps30.",
    },
    "G18": {
        "field_name": "NRf",
        "name": "Corrected Fan Speed",
        "classification": StateClassification.MEASURED,
        "category": "Gas-Path Thermodynamic Sensors",
        "unit": "rpm",
        "c_mapss_col": "sensor_13",
        "is_simulation_only": False,
        "description": "Corrected low-pressure fan spool speed.",
    },
    "G19": {
        "field_name": "NRc",
        "name": "Corrected Core Speed",
        "classification": StateClassification.MEASURED,
        "category": "Gas-Path Thermodynamic Sensors",
        "unit": "rpm",
        "c_mapss_col": "sensor_14",
        "is_simulation_only": False,
        "description": "Corrected high-pressure core spool speed.",
    },
    "G20": {
        "field_name": "BPR",
        "name": "Bypass Ratio",
        "classification": StateClassification.MEASURED,
        "category": "Gas-Path Thermodynamic Sensors",
        "unit": "dimensionless",
        "c_mapss_col": "sensor_15",
        "is_simulation_only": False,
        "description": "Ratio of bypass duct airflow to core gas-path airflow.",
    },
    "G21": {
        "field_name": "farB",
        "name": "Burner Fuel-Air Ratio",
        "classification": StateClassification.MEASURED,
        "category": "Gas-Path Thermodynamic Sensors",
        "unit": "dimensionless",
        "c_mapss_col": "sensor_16",
        "is_simulation_only": False,
        "description": "Combustor fuel-to-air mass ratio.",
    },
    "G22": {
        "field_name": "htBleed",
        "name": "Bleed Enthalpy",
        "classification": StateClassification.MEASURED,
        "category": "Gas-Path Thermodynamic Sensors",
        "unit": "dimensionless ratio",
        "c_mapss_col": "sensor_17",
        "is_simulation_only": False,
        "description": "Bleed air enthalpy parameter.",
    },
    "G23": {
        "field_name": "Nf_dmd",
        "name": "Demanded Fan Speed",
        "classification": StateClassification.MEASURED,
        "category": "Gas-Path Thermodynamic Sensors",
        "unit": "rpm",
        "c_mapss_col": "sensor_18",
        "is_simulation_only": False,
        "description": "Flight control system demanded fan spool speed.",
    },
    "G24": {
        "field_name": "PCNfR_dmd",
        "name": "Demanded Corrected Fan Speed",
        "classification": StateClassification.MEASURED,
        "category": "Gas-Path Thermodynamic Sensors",
        "unit": "%",
        "c_mapss_col": "sensor_19",
        "is_simulation_only": False,
        "description": "Flight control system demanded corrected fan speed percentage.",
    },
    "G25": {
        "field_name": "W31",
        "name": "HPT Coolant Bleed Flow",
        "classification": StateClassification.MEASURED,
        "category": "Gas-Path Thermodynamic Sensors",
        "unit": "lbm/s",
        "c_mapss_col": "sensor_20",
        "is_simulation_only": False,
        "description": "High-Pressure Turbine cooling bleed air mass flow rate.",
    },
    "G26": {
        "field_name": "W32",
        "name": "LPT Coolant Bleed Flow",
        "classification": StateClassification.MEASURED,
        "category": "Gas-Path Thermodynamic Sensors",
        "unit": "lbm/s",
        "c_mapss_col": "sensor_21",
        "is_simulation_only": False,
        "description": "Low-Pressure Turbine cooling bleed air mass flow rate.",
    },
    "G27": {
        "field_name": "hpc_temp_rise",
        "name": "HPC Temperature Rise",
        "classification": StateClassification.DERIVED,
        "category": "Derived Consistency Indicators",
        "unit": "°R",
        "c_mapss_col": "sensor_3 - sensor_2",
        "is_simulation_only": False,
        "description": "Compression temperature rise delta T = T30 - T24 across HPC.",
    },
    "G28": {
        "field_name": "overall_pressure_ratio",
        "name": "Overall Pressure Ratio (OPR)",
        "classification": StateClassification.DERIVED,
        "category": "Derived Consistency Indicators",
        "unit": "dimensionless",
        "c_mapss_col": "sensor_7 / sensor_5",
        "is_simulation_only": False,
        "description": "Total engine compression pressure ratio P30 / P2.",
    },
    "G29": {
        "field_name": "spool_speed_ratio",
        "name": "Spool Speed Ratio",
        "classification": StateClassification.DERIVED,
        "category": "Derived Consistency Indicators",
        "unit": "dimensionless",
        "c_mapss_col": "sensor_9 / sensor_8",
        "is_simulation_only": False,
        "description": "Ratio of high-pressure core speed to low-pressure fan speed Nc / Nf.",
    },
    "G30": {
        "field_name": "normalized_feature_vec",
        "name": "Normalized Feature Vector",
        "classification": StateClassification.DERIVED,
        "category": "Composite Preprocessing Representations",
        "unit": "normalized [0, 1]",
        "c_mapss_col": "SafeMinMaxScaler(G3-G26)",
        "is_simulation_only": False,
        "description": "14-dim (FD001) or 24-dim (FD002-FD004) scaled feature vector.",
    },
    "G31": {
        "field_name": "temporal_seq_tensor",
        "name": "Temporal Sequence Tensor",
        "classification": StateClassification.DERIVED,
        "category": "Composite Preprocessing Representations",
        "unit": "normalized [0, 1]",
        "c_mapss_col": "Rolling 30-cycle buffer",
        "is_simulation_only": False,
        "description": "30-cycle temporal sliding window tensor for LSTM and sequential ML inference.",
    },
    "G32": {
        "field_name": "anomaly_score_m1",
        "name": "Isolation Forest Anomaly Score",
        "classification": StateClassification.PREDICTED,
        "category": "Phase 4 Machine Learning Anomaly States",
        "unit": "anomaly score",
        "c_mapss_col": None,
        "is_simulation_only": False,
        "description": "Unsupervised anomaly score from Isolation Forest ensemble.",
    },
    "G33": {
        "field_name": "anomaly_score_m2",
        "name": "PCA-SPE Reconstruction Residual",
        "classification": StateClassification.PREDICTED,
        "category": "Phase 4 Machine Learning Anomaly States",
        "unit": "residual energy (Q-statistic)",
        "c_mapss_col": None,
        "is_simulation_only": False,
        "description": "Squared Prediction Error (SPE) reconstruction residual from PCA model.",
    },
    "G34": {
        "field_name": "anomaly_alert_flag",
        "name": "Persistence Alert Flag",
        "classification": StateClassification.PREDICTED,
        "category": "Phase 4 Machine Learning Anomaly States",
        "unit": "binary flag {0, 1}",
        "c_mapss_col": None,
        "is_simulation_only": False,
        "description": "3-cycle persistent multi-model threshold crossing alarm flag.",
    },
    "G35": {
        "field_name": "predicted_rul_xgboost",
        "name": "XGBoost Prognostic RUL",
        "classification": StateClassification.PREDICTED,
        "category": "Phase 4 Machine Learning Prognostic States",
        "unit": "flight cycles",
        "c_mapss_col": None,
        "is_simulation_only": False,
        "description": "Remaining Useful Life predicted by Phase 4 XGBoost champion model.",
    },
    "G36": {
        "field_name": "predicted_rul_lstm",
        "name": "LSTM Prognostic RUL",
        "classification": StateClassification.PREDICTED,
        "category": "Phase 4 Machine Learning Prognostic States",
        "unit": "flight cycles",
        "c_mapss_col": None,
        "is_simulation_only": False,
        "description": "Remaining Useful Life predicted by Phase 4 LSTM champion model.",
    },
    "G37": {
        "field_name": "prognostic_spread",
        "name": "Prognostic Uncertainty Spread",
        "classification": StateClassification.DERIVED,
        "category": "Phase 4 Machine Learning Prognostic States",
        "unit": "flight cycles",
        "c_mapss_col": None,
        "is_simulation_only": False,
        "description": "Absolute inter-model spread |G35 - G36| capturing epistemic prognostic uncertainty.",
    },
    "G38": {
        "field_name": "virtual_health_index",
        "name": "Virtual Engine Health Index",
        "classification": StateClassification.DERIVED,
        "category": "Phase 4 Machine Learning Prognostic States",
        "unit": "index [0, 1]",
        "c_mapss_col": None,
        "is_simulation_only": False,
        "description": "Proposed continuous health degradation state H(t) in [0, 1].",
    },
    "G39": {
        "field_name": "thermal_stress_offset",
        "name": "Hypothetical Thermal Stress Offset",
        "classification": StateClassification.SIMULATED,
        "category": "What-If Simulation Perturbation States",
        "unit": "°R",
        "c_mapss_col": None,
        "is_simulation_only": True,
        "description": "Hypothetical temperature offset delta T applied during What-If simulation.",
    },
    "G40": {
        "field_name": "sensor_bias_delta",
        "name": "Hypothetical Sensor Bias Delta",
        "classification": StateClassification.SIMULATED,
        "category": "What-If Simulation Perturbation States",
        "unit": "sensor units (psia, °R, rpm)",
        "c_mapss_col": None,
        "is_simulation_only": True,
        "description": "Hypothetical sensor calibration bias delta applied during What-If simulation.",
    },
}


@dataclass
class EngineState:
    """
    Virtual Engine State Vector S(t) capturing the 40 state entries G1 - G40.
    Separates measured telemetry, derived indicators, ML predictions, and simulated scenario parameters.
    """

    # -------------------------------------------------------------------------
    # Group A: Operational Tracking Metadata (G1 - G2) [Measured / Admin]
    # -------------------------------------------------------------------------
    unit_number: int  # G1: Engine serial/trajectory identifier (dimensionless)
    time_cycles: int  # G2: Discrete flight cycle counter (cycles)

    # -------------------------------------------------------------------------
    # Group B: Operating Context & Environmental Settings (G3 - G5) [Measured]
    # -------------------------------------------------------------------------
    altitude: float  # G3: Pressure altitude (kft, 0 to 42 kft) [op_setting_1]
    mach_number: float  # G4: Flight Mach number (dimensionless, 0.0 to 0.84) [op_setting_2]
    throttle_resolver_angle: float  # G5: Throttle resolver angle (TRA %, 20% to 100%) [op_setting_3]

    # -------------------------------------------------------------------------
    # Group C: Gas-Path Thermodynamic Sensors (G6 - G26) [Measured]
    # -------------------------------------------------------------------------
    T2: float  # G6: Total temp at fan inlet (°R) [sensor_1]
    T24: float  # G7: Total temp at LPC outlet (°R) [sensor_2]
    T30: float  # G8: Total temp at HPC outlet (°R) [sensor_3]
    T50: float  # G9: Total temp at LPT outlet (°R) [sensor_4]
    P2: float  # G10: Total pressure at fan inlet (psia) [sensor_5]
    P15: float  # G11: Total pressure in bypass duct (psia) [sensor_6]
    P30: float  # G12: Total pressure at HPC outlet (psia) [sensor_7]
    Nf: float  # G13: Physical fan spool speed (rpm) [sensor_8]
    Nc: float  # G14: Physical core spool speed (rpm) [sensor_9]
    epr: float  # G15: Engine pressure ratio P50/P2 (dimensionless) [sensor_10]
    Ps30: float  # G16: Static pressure at HPC outlet (psia) [sensor_11]
    phi: float  # G17: Fuel flow ratio Wf/Ps30 (pps/psia) [sensor_12]
    NRf: float  # G18: Corrected fan speed (rpm) [sensor_13]
    NRc: float  # G19: Corrected core speed (rpm) [sensor_14]
    BPR: float  # G20: Bypass ratio (dimensionless) [sensor_15]
    farB: float  # G21: Burner fuel-air ratio (dimensionless) [sensor_16]
    htBleed: float  # G22: Bleed enthalpy (dimensionless ratio) [sensor_17]
    Nf_dmd: float  # G23: Demanded fan speed (rpm) [sensor_18]
    PCNfR_dmd: float  # G24: Demanded corrected fan speed (%) [sensor_19]
    W31: float  # G25: HPT coolant bleed (lbm/s) [sensor_20]
    W32: float  # G26: LPT coolant bleed (lbm/s) [sensor_21]

    # -------------------------------------------------------------------------
    # Group D: Derived Macroscopic Consistency Indicators (G27 - G29) [Derived]
    # -------------------------------------------------------------------------
    hpc_temp_rise: float  # G27: T30 - T24 (°R)
    overall_pressure_ratio: float  # G28: P30 / P2 (dimensionless OPR)
    spool_speed_ratio: float  # G29: Nc / Nf (dimensionless)

    # -------------------------------------------------------------------------
    # Group E: Composite Preprocessing Representations (G30 - G31) [Derived]
    # -------------------------------------------------------------------------
    normalized_feature_vec: np.ndarray  # G30: Normalized feature vector (14-dim or 24-dim)
    temporal_seq_tensor: np.ndarray  # G31: 30-cycle temporal sequence tensor (30x14 or 30x24)

    # -------------------------------------------------------------------------
    # Group F: Phase 4 Machine Learning Anomaly States (G32 - G34) [Predicted / Derived]
    # -------------------------------------------------------------------------
    anomaly_score_m1: Optional[float] = None  # G32: Isolation Forest anomaly score (Step 5.6)
    anomaly_score_m2: Optional[float] = None  # G33: PCA-SPE score (Step 5.6)
    anomaly_alert_flag: Optional[int] = None  # G34: Binary 3-cycle persistence flag {0, 1}

    # -------------------------------------------------------------------------
    # Group G: Phase 4 Machine Learning Prognostic States (G35 - G38) [Predicted / Derived]
    # -------------------------------------------------------------------------
    predicted_rul_xgboost: Optional[float] = None  # G35: XGBoost RUL estimate (Step 5.6)
    predicted_rul_lstm: Optional[float] = None  # G36: LSTM RUL estimate (Step 5.6)
    prognostic_spread: Optional[float] = None  # G37: |RUL_xgb - RUL_lstm| (Step 5.6)
    virtual_health_index: Optional[float] = None  # G38: Derived health index [0, 1] (Step 5.5)

    # -------------------------------------------------------------------------
    # Group H: What-If Simulation Perturbation States (G39 - G40) [Simulated]
    # -------------------------------------------------------------------------
    thermal_stress_offset: Optional[float] = None  # G39: Hypothetical delta T (°R) (Step 5.7)
    sensor_bias_delta: Optional[Dict[str, float]] = None  # G40: Hypothetical sensor bias (Step 5.7)

    # -------------------------------------------------------------------------
    # Metadata & Tracking Provenance
    # -------------------------------------------------------------------------
    subset: str = "FD001"
    feature_tier: str = "Tier1_Core14"
    is_padded: bool = False
    padded_cycles_count: int = 0
    history_window_length: int = 1
    scenario_id: Optional[str] = None
    scenario_audit: Optional[Dict[str, Any]] = None

    def to_dict(self, include_arrays: bool = True) -> Dict[str, Any]:
        """Convert state vector to dictionary format for inspection or serialization."""
        data = {
            "G1_unit_number": self.unit_number,
            "G2_time_cycles": self.time_cycles,
            "G3_altitude": self.altitude,
            "G4_mach_number": self.mach_number,
            "G5_throttle_resolver_angle": self.throttle_resolver_angle,
            "G6_T2": self.T2,
            "G7_T24": self.T24,
            "G8_T30": self.T30,
            "G9_T50": self.T50,
            "G10_P2": self.P2,
            "G11_P15": self.P15,
            "G12_P30": self.P30,
            "G13_Nf": self.Nf,
            "G14_Nc": self.Nc,
            "G15_epr": self.epr,
            "G16_Ps30": self.Ps30,
            "G17_phi": self.phi,
            "G18_NRf": self.NRf,
            "G19_NRc": self.NRc,
            "G20_BPR": self.BPR,
            "G21_farB": self.farB,
            "G22_htBleed": self.htBleed,
            "G23_Nf_dmd": self.Nf_dmd,
            "G24_PCNfR_dmd": self.PCNfR_dmd,
            "G25_W31": self.W31,
            "G26_W32": self.W32,
            "G27_hpc_temp_rise": self.hpc_temp_rise,
            "G28_overall_pressure_ratio": self.overall_pressure_ratio,
            "G29_spool_speed_ratio": self.spool_speed_ratio,
            "G32_anomaly_score_m1": self.anomaly_score_m1,
            "G33_anomaly_score_m2": self.anomaly_score_m2,
            "G34_anomaly_alert_flag": self.anomaly_alert_flag,
            "G35_predicted_rul_xgboost": self.predicted_rul_xgboost,
            "G36_predicted_rul_lstm": self.predicted_rul_lstm,
            "G37_prognostic_spread": self.prognostic_spread,
            "G38_virtual_health_index": self.virtual_health_index,
            "G39_thermal_stress_offset": self.thermal_stress_offset,
            "G40_sensor_bias_delta": self.sensor_bias_delta,
            "scenario_audit": self.scenario_audit,
            "provenance": {
                "subset": self.subset,
                "feature_tier": self.feature_tier,
                "is_padded": self.is_padded,
                "padded_cycles_count": self.padded_cycles_count,
                "history_window_length": self.history_window_length,
                "scenario_id": self.scenario_id,
            },
        }
        if include_arrays:
            data["G30_normalized_feature_vec"] = self.normalized_feature_vec.tolist() if self.normalized_feature_vec is not None else None
            data["G31_temporal_seq_tensor_shape"] = list(self.temporal_seq_tensor.shape) if self.temporal_seq_tensor is not None else None
        return data

    @classmethod
    def get_taxonomy(cls) -> Dict[str, Dict[str, Any]]:
        """Return the canonical 40-variable state taxonomy specification."""
        return STATE_TAXONOMY

    @classmethod
    def get_classification_map(cls) -> Dict[str, StateClassification]:
        """Return mapping of state variable IDs (G1 - G40) to their formal StateClassification."""
        return {gid: info["classification"] for gid, info in STATE_TAXONOMY.items()}

    def to_labeled_dict(self, include_arrays: bool = True) -> Dict[str, Any]:
        """
        Convert state vector to dictionary format with explicit provenance, classification,
        and engineering unit annotations for each state variable.
        """
        labeled: Dict[str, Any] = {}
        for gid, info in STATE_TAXONOMY.items():
            field_name = info["field_name"]
            val = getattr(self, field_name, None)
            if gid == "G30" and val is not None and not include_arrays:
                val = f"<array shape {val.shape}>"
            elif gid == "G30" and val is not None and include_arrays:
                val = val.tolist()
            elif gid == "G31" and val is not None and not include_arrays:
                val = f"<tensor shape {val.shape}>"
            elif gid == "G31" and val is not None and include_arrays:
                val = list(val.shape)

            labeled[f"{gid}_{field_name}"] = {
                "value": val,
                "classification": info["classification"].value,
                "category": info["category"],
                "unit": info["unit"],
                "is_simulation_only": info["is_simulation_only"],
            }

        labeled["scenario_audit"] = self.scenario_audit
        labeled["provenance"] = {
            "subset": self.subset,
            "feature_tier": self.feature_tier,
            "is_padded": self.is_padded,
            "padded_cycles_count": self.padded_cycles_count,
            "history_window_length": self.history_window_length,
            "scenario_id": self.scenario_id,
        }
        return labeled

    def check_thermodynamic_sanity(self) -> Dict[str, bool]:
        """
        Evaluate macroscopic sanity consistency checks validated in Step 5.2.
        Returns dictionary of boolean pass/fail checks.
        """
        return {
            "T30_gt_T24": bool(self.T30 > self.T24),
            "T24_gt_T2": bool(self.T24 > self.T2),
            "P30_gt_P2": bool(self.P30 > self.P2),
            "P30_gt_P15": bool(self.P30 > self.P15),
            "Nc_gt_Nf": bool(self.Nc > self.Nf),
            "overall_sanity_pass": bool(
                self.T30 > self.T24 and self.T24 > self.T2 and self.P30 > self.P2 and self.Nc > self.Nf
            ),
        }
