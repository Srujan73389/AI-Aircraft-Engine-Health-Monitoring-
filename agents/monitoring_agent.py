"""
Monitoring Agent Module - Phase 6 Step 6.2
Autonomous Monitoring Agent for the AI-Based Aircraft Engine Digital Twin Multi-Agent System.

Roadmap Reference: Phase 6 - Step 6.2:
"Implement and test the Monitoring Agent independently"

Capabilities:
1. Physical & Operational Threshold Evaluation:
   - Configurable sensor operating bands (Caution & Warning levels).
   - Hard thermodynamic engineering boundaries (Phase 2 EDA and Phase 5 Digital Twin physics).
   - Macroscopic consistency checks (OPR, Nc/Nf, delta T_HPC, thermodynamic order inversions).
2. Machine Learning Anomaly Score Evaluation:
   - Evaluates Isolation Forest (M1) and PCA-SPE (M2) scores against 95th-percentile calibrated thresholds.
3. Multi-Cycle Persistence Tracking:
   - Tracks consecutive anomalous cycles per engine trajectory.
   - Triggers persistence_alert=True strictly when persistence_count >= 3.
4. Operational State & Severity Classification:
   - Classifies engine health: OperationalState.HEALTHY, DEGRADED, ANOMALOUS, CRITICAL.
   - Assigns SeverityLevel: NOMINAL, ADVISORY, CAUTION, WARNING.
5. Standardized Type-Safe Schema Generation:
   - Ingests MonitoringInput or EngineState, producing fully validated MonitoringOutput.
   - Measures execution latency and attaches AgentMetadata.
6. Trajectory Isolation & Reset:
   - Independent tracking per engine unit with explicit reset_trajectory() API.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
import os
import sys
import time
from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np

from agents.schemas import (
    AgentExecutionStatus,
    SeverityLevel,
    OperationalState,
    AgentMetadata,
    ThresholdBreach,
    MonitoringInput,
    MonitoringOutput,
)

# Optional Digital Twin imports (graceful handling if imported elsewhere)
try:
    from digital_twin.engine_state import EngineState
except ImportError:
    EngineState = None  # type: ignore


# -----------------------------------------------------------------------------
# 1. Physical Sensor Aliases & Mappings
# -----------------------------------------------------------------------------
SENSOR_ALIAS_MAP: Dict[str, str] = {
    # C-MAPSS raw columns to physical symbols
    "sensor_1": "T2",
    "sensor_2": "T24",
    "sensor_3": "T30",
    "sensor_4": "T50",
    "sensor_5": "P2",
    "sensor_6": "P15",
    "sensor_7": "P30",
    "sensor_8": "Nf",
    "sensor_9": "Nc",
    "sensor_10": "epr",
    "sensor_11": "Ps30",
    "sensor_12": "phi",
    "sensor_13": "NRf",
    "sensor_14": "NRc",
    "sensor_15": "BPR",
    "sensor_16": "farB",
    "sensor_17": "htBleed",
    "sensor_18": "Nf_dmd",
    "sensor_19": "PCNfR_dmd",
    "sensor_20": "W31",
    "sensor_21": "W32",
    # Operating settings aliases
    "op_setting_1": "altitude",
    "op_setting_2": "mach_number",
    "op_setting_3": "throttle_resolver_angle",
    # Derived indicator aliases
    "delta_T": "hpc_temp_rise",
    "delta_T_hpc": "hpc_temp_rise",
    "OPR": "overall_pressure_ratio",
    "opr": "overall_pressure_ratio",
    "spool_ratio": "spool_speed_ratio",
}


# -----------------------------------------------------------------------------
# 2. Defensible Thermodynamic Boundaries (Hard Limits across all flight regimes)
# -----------------------------------------------------------------------------
GLOBAL_THERMODYNAMIC_BOUNDS: Dict[str, Dict[str, float]] = {
    # Temperatures (°R)
    "T2": {"min": 400.0, "max": 600.0},
    "T24": {"min": 450.0, "max": 750.0},
    "T30": {"min": 1100.0, "max": 1850.0},
    "T50": {"min": 1000.0, "max": 1600.0},
    # Pressures (psia)
    "P2": {"min": 2.5, "max": 20.0},
    "P15": {"min": 5.0, "max": 30.0},
    "P30": {"min": 100.0, "max": 700.0},
    "Ps30": {"min": 5.0, "max": 65.0},
    # Spool Speeds (rpm)
    "Nf": {"min": 1500.0, "max": 3000.0},
    "Nc": {"min": 7000.0, "max": 10500.0},
    # Macroscopic Derived Indicators
    "hpc_temp_rise": {"min": 500.0, "max": 1250.0},
    "overall_pressure_ratio": {"min": 15.0, "max": 55.0},
    "spool_speed_ratio": {"min": 3.0, "max": 5.5},
}


# -----------------------------------------------------------------------------
# 3. Default Operational Threshold Bands (Subsystem / Subset Baseline)
# -----------------------------------------------------------------------------
# Calibrated against Phase 2 EDA nominal envelopes and Phase 4 validation trajectories.
DEFAULT_FD001_OPERATING_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "T30": {
        "caution_high": 1605.0,
        "warning_high": 1625.0,
        "caution_low": 1550.0,
        "warning_low": 1500.0,
    },
    "P30": {
        "caution_high": 560.0,
        "warning_high": 575.0,
        "caution_low": 530.0,
        "warning_low": 500.0,
    },
    "T50": {
        "caution_high": 1435.0,
        "warning_high": 1460.0,
        "caution_low": 1370.0,
        "warning_low": 1330.0,
    },
    "overall_pressure_ratio": {
        "caution_high": 40.0,
        "warning_high": 45.0,
        "caution_low": 35.0,
        "warning_low": 30.0,
    },
    "spool_speed_ratio": {
        "caution_high": 3.95,
        "warning_high": 4.10,
        "caution_low": 3.70,
        "warning_low": 3.55,
    },
    "hpc_temp_rise": {
        "caution_high": 990.0,
        "warning_high": 1025.0,
        "caution_low": 910.0,
        "warning_low": 880.0,
    },
}

# 95th Percentile Decision Thresholds (calibrated in Phase 4 Step 4.4 on healthy Cohort B)
DEFAULT_ANOMALY_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "FD001": {"isolation_forest": -0.031165, "pca_spe": 1.610083},
    "FD002": {"isolation_forest": -0.045051, "pca_spe": 1.274029},
    "FD003": {"isolation_forest": -0.038731, "pca_spe": 1.191610},
    "FD004": {"isolation_forest": -0.054126, "pca_spe": 1.868511},
}


class MonitoringAgent:
    """
    Autonomous Monitoring Agent.
    Evaluates physical sensor readings, thermodynamic consistency indicators,
    and statistical anomaly scores to track fleet health and alert on persistent degradation.
    """

    def __init__(
        self,
        agent_id: str = "monitoring_agent_v1",
        subset: str = "FD001",
        sensor_thresholds: Optional[Dict[str, Dict[str, float]]] = None,
        anomaly_thresholds: Optional[Dict[str, float]] = None,
        thermodynamic_bounds: Optional[Dict[str, Dict[str, float]]] = None,
        persistence_alert_window: int = 3,
        lstm_ae_detector: Optional[Any] = None,
    ):
        """
        Initializes the Monitoring Agent with configurable operating thresholds.

        Args:
            agent_id: Unique agent identifier for provenance tracking.
            subset: C-MAPSS dataset subset identifier ("FD001", "FD002", etc.).
            sensor_thresholds: Custom dictionary of caution and warning thresholds per sensor.
            anomaly_thresholds: Custom decision thresholds for Isolation Forest and PCA-SPE.
            thermodynamic_bounds: Custom absolute thermodynamic limits.
            persistence_alert_window: Consecutive anomalous cycle count required to trigger persistence_alert (default: 3).
            lstm_ae_detector: Optional LSTM-AE anomaly detector instance for deep sequence reconstruction.
        """
        self.agent_id = agent_id
        self.subset = subset.upper()
        self.persistence_alert_window = max(1, persistence_alert_window)
        self.lstm_ae_detector = lstm_ae_detector

        # Set sensor operating thresholds (merging custom over defaults)
        self.sensor_thresholds = dict(DEFAULT_FD001_OPERATING_THRESHOLDS)
        if sensor_thresholds is not None:
            for k, v in sensor_thresholds.items():
                norm_k = self._normalize_key(k)
                self.sensor_thresholds[norm_k] = dict(v)

        # Set absolute thermodynamic bounds
        self.thermodynamic_bounds = dict(GLOBAL_THERMODYNAMIC_BOUNDS)
        if thermodynamic_bounds is not None:
            for k, v in thermodynamic_bounds.items():
                norm_k = self._normalize_key(k)
                self.thermodynamic_bounds[norm_k] = dict(v)

        # Set statistical anomaly thresholds
        default_sub_anom = DEFAULT_ANOMALY_THRESHOLDS.get(self.subset, DEFAULT_ANOMALY_THRESHOLDS["FD001"])
        self.anomaly_thresholds = dict(default_sub_anom)
        if anomaly_thresholds is not None:
            self.anomaly_thresholds.update(anomaly_thresholds)

        # Trajectory persistence state machine: {unit_number: consecutive_anomalous_cycles}
        self._persistence_counts: Dict[int, int] = {}

    @staticmethod
    def _normalize_key(key: str) -> str:
        """Normalizes sensor/indicator names using alias map."""
        clean_k = key.strip()
        return SENSOR_ALIAS_MAP.get(clean_k, clean_k)

    def reset_trajectory(self, unit_number: Optional[int] = None) -> None:
        """
        Resets multi-cycle persistence tracking state.

        Args:
            unit_number: If provided, resets tracking for this specific engine ID.
                         If None, clears tracking state across all engines.
        """
        if unit_number is not None:
            if unit_number in self._persistence_counts:
                self._persistence_counts[unit_number] = 0
        else:
            self._persistence_counts.clear()

    def get_persistence_count(self, unit_number: int) -> int:
        """Retrieves the current consecutive anomalous cycle count for an engine."""
        return self._persistence_counts.get(unit_number, 0)

    def process_input(self, monitoring_input: Union[MonitoringInput, Dict[str, Any]]) -> MonitoringOutput:
        """
        Processes a single flight cycle monitoring input, evaluates thresholds,
        checks thermodynamic consistency, updates persistence counters, and classifies operational state.

        Args:
            monitoring_input: Strongly-typed MonitoringInput or serialized dictionary.

        Returns:
            Type-safe MonitoringOutput with complete diagnostic breakdown.
        """
        start_time = time.perf_counter()
        timestamp = datetime.now(timezone.utc).isoformat()

        # Ingest and validate input payload
        if isinstance(monitoring_input, dict):
            inp = MonitoringInput.from_dict(monitoring_input)
        elif isinstance(monitoring_input, MonitoringInput):
            inp = monitoring_input
        else:
            raise TypeError(f"Expected MonitoringInput or dict, got {type(monitoring_input)}")

        inp.validate()

        unit_number = inp.unit_number
        time_cycles = inp.time_cycles

        # Normalize telemetry and derived indicator keys
        telemetry: Dict[str, float] = {
            self._normalize_key(k): float(v) for k, v in inp.raw_telemetry.items()
        }
        derived: Dict[str, float] = {
            self._normalize_key(k): float(v) for k, v in inp.derived_indicators.items()
        }

        # ---------------------------------------------------------------------
        # 1. Physical & Operating Threshold Evaluation
        # ---------------------------------------------------------------------
        breaches: List[ThresholdBreach] = []
        combined_telemetry = {**telemetry, **derived}

        # A. Check against operating thresholds (Caution / Warning)
        for sensor_name, bands in self.sensor_thresholds.items():
            if sensor_name not in combined_telemetry:
                continue

            val = combined_telemetry[sensor_name]

            # High breaches
            if "warning_high" in bands and val > bands["warning_high"]:
                breaches.append(
                    ThresholdBreach(
                        sensor_name=sensor_name,
                        observed_value=val,
                        threshold_limit=bands["warning_high"],
                        direction="HIGH",
                        breach_margin=val - bands["warning_high"],
                        severity=SeverityLevel.WARNING,
                    )
                )
            elif "caution_high" in bands and val > bands["caution_high"]:
                breaches.append(
                    ThresholdBreach(
                        sensor_name=sensor_name,
                        observed_value=val,
                        threshold_limit=bands["caution_high"],
                        direction="HIGH",
                        breach_margin=val - bands["caution_high"],
                        severity=SeverityLevel.CAUTION,
                    )
                )

            # Low breaches
            if "warning_low" in bands and val < bands["warning_low"]:
                breaches.append(
                    ThresholdBreach(
                        sensor_name=sensor_name,
                        observed_value=val,
                        threshold_limit=bands["warning_low"],
                        direction="LOW",
                        breach_margin=bands["warning_low"] - val,
                        severity=SeverityLevel.WARNING,
                    )
                )
            elif "caution_low" in bands and val < bands["caution_low"]:
                breaches.append(
                    ThresholdBreach(
                        sensor_name=sensor_name,
                        observed_value=val,
                        threshold_limit=bands["caution_low"],
                        direction="LOW",
                        breach_margin=bands["caution_low"] - val,
                        severity=SeverityLevel.CAUTION,
                    )
                )

        # B. Check against global thermodynamic boundaries (Physical Safety Limits)
        for sensor_name, bounds in self.thermodynamic_bounds.items():
            if sensor_name not in combined_telemetry:
                continue

            val = combined_telemetry[sensor_name]
            min_val = bounds.get("min")
            max_val = bounds.get("max")

            if max_val is not None and val > max_val:
                # Avoid duplicate breach if already flagged as WARNING
                already_flagged = any(
                    b.sensor_name == sensor_name and b.direction == "HIGH" and b.severity == SeverityLevel.WARNING
                    for b in breaches
                )
                if not already_flagged:
                    breaches.append(
                        ThresholdBreach(
                            sensor_name=sensor_name,
                            observed_value=val,
                            threshold_limit=max_val,
                            direction="HIGH",
                            breach_margin=val - max_val,
                            severity=SeverityLevel.WARNING,
                        )
                    )

            if min_val is not None and val < min_val:
                already_flagged = any(
                    b.sensor_name == sensor_name and b.direction == "LOW" and b.severity == SeverityLevel.WARNING
                    for b in breaches
                )
                if not already_flagged:
                    breaches.append(
                        ThresholdBreach(
                            sensor_name=sensor_name,
                            observed_value=val,
                            threshold_limit=min_val,
                            direction="LOW",
                            breach_margin=min_val - val,
                            severity=SeverityLevel.WARNING,
                        )
                    )

        # C. Macroscopic Physical Consistency & Thermodynamic Inversion Checks
        # Aerothermal order: T30 > T24 > T2
        t2 = combined_telemetry.get("T2")
        t24 = combined_telemetry.get("T24")
        t30 = combined_telemetry.get("T30")
        if t2 is not None and t24 is not None and t24 <= t2:
            breaches.append(
                ThresholdBreach(
                    sensor_name="T24_T2_INVERSION",
                    observed_value=t24,
                    threshold_limit=t2,
                    direction="LOW",
                    breach_margin=t2 - t24,
                    severity=SeverityLevel.WARNING,
                )
            )
        if t24 is not None and t30 is not None and t30 <= t24:
            breaches.append(
                ThresholdBreach(
                    sensor_name="T30_T24_INVERSION",
                    observed_value=t30,
                    threshold_limit=t24,
                    direction="LOW",
                    breach_margin=t24 - t30,
                    severity=SeverityLevel.WARNING,
                )
            )

        # Compression pressure order: P30 > P15 > P2 and P2 > 0
        p2 = combined_telemetry.get("P2")
        p30 = combined_telemetry.get("P30")
        if p2 is not None and p2 <= 0.0:
            breaches.append(
                ThresholdBreach(
                    sensor_name="P2_NON_POSITIVE",
                    observed_value=p2,
                    threshold_limit=0.0,
                    direction="LOW",
                    breach_margin=0.0 - p2,
                    severity=SeverityLevel.WARNING,
                )
            )
        if p2 is not None and p30 is not None and p30 <= p2:
            breaches.append(
                ThresholdBreach(
                    sensor_name="P30_P2_INVERSION",
                    observed_value=p30,
                    threshold_limit=p2,
                    direction="LOW",
                    breach_margin=p2 - p30,
                    severity=SeverityLevel.WARNING,
                )
            )

        # Spool speed order: Nc > Nf and Nf > 0
        nf = combined_telemetry.get("Nf")
        nc = combined_telemetry.get("Nc")
        if nf is not None and nf <= 0.0:
            breaches.append(
                ThresholdBreach(
                    sensor_name="NF_NON_POSITIVE",
                    observed_value=nf,
                    threshold_limit=0.0,
                    direction="LOW",
                    breach_margin=0.0 - nf,
                    severity=SeverityLevel.WARNING,
                )
            )
        if nf is not None and nc is not None and nc <= nf:
            breaches.append(
                ThresholdBreach(
                    sensor_name="NC_NF_INVERSION",
                    observed_value=nc,
                    threshold_limit=nf,
                    direction="LOW",
                    breach_margin=nf - nc,
                    severity=SeverityLevel.WARNING,
                )
            )

        # ---------------------------------------------------------------------
        # 2. Machine Learning Anomaly Score Evaluation
        # ---------------------------------------------------------------------
        if_score = inp.anomaly_scores.get("isolation_forest")
        if if_score is None:
            if_score = inp.anomaly_scores.get("anomaly_score_m1")
        if if_score is None:
            if_score = inp.anomaly_scores.get("m1")

        pca_score = inp.anomaly_scores.get("pca_spe")
        if pca_score is None:
            pca_score = inp.anomaly_scores.get("anomaly_score_m2")
        if pca_score is None:
            pca_score = inp.anomaly_scores.get("m2")

        if_tau = self.anomaly_thresholds.get("isolation_forest", -0.031165)
        pca_tau = self.anomaly_thresholds.get("pca_spe", 1.610083)

        isolation_forest_breach = False
        if if_score is not None and math.isfinite(if_score):
            if if_score > if_tau:
                isolation_forest_breach = True

        pca_spe_breach = False
        if pca_score is not None and math.isfinite(pca_score):
            if pca_score > pca_tau:
                pca_spe_breach = True

        lstm_score = inp.anomaly_scores.get("lstm_ae")
        if lstm_score is None:
            lstm_score = inp.anomaly_scores.get("anomaly_score_lstm_ae")
        lstm_tau = self.anomaly_thresholds.get("lstm_ae", 0.05)
        lstm_ae_breach = False
        if lstm_score is not None and math.isfinite(lstm_score):
            if lstm_score > lstm_tau:
                lstm_ae_breach = True

        model_anomaly_flag = isolation_forest_breach or pca_spe_breach or lstm_ae_breach

        # ---------------------------------------------------------------------
        # 3. Multi-Cycle Persistence Tracking
        # ---------------------------------------------------------------------
        # A cycle is deemed anomalous if it has physical threshold breaches OR statistical anomaly breach
        is_anomalous_cycle = (len(breaches) > 0) or model_anomaly_flag

        current_count = self._persistence_counts.get(unit_number, 0)
        if is_anomalous_cycle:
            new_count = current_count + 1
        else:
            new_count = 0

        self._persistence_counts[unit_number] = new_count
        persistence_alert = (new_count >= self.persistence_alert_window)

        # ---------------------------------------------------------------------
        # 4. Operational State & Anomaly Severity Classification
        # ---------------------------------------------------------------------
        has_warning_breach = any(b.severity == SeverityLevel.WARNING for b in breaches)
        has_caution_breach = any(b.severity == SeverityLevel.CAUTION for b in breaches)

        if has_warning_breach:
            operational_state = OperationalState.CRITICAL
            anomaly_severity = SeverityLevel.WARNING
        elif persistence_alert:
            operational_state = OperationalState.DEGRADED
            anomaly_severity = SeverityLevel.CAUTION
        elif is_anomalous_cycle:
            operational_state = OperationalState.ANOMALOUS
            anomaly_severity = SeverityLevel.CAUTION if has_caution_breach else SeverityLevel.ADVISORY
        else:
            operational_state = OperationalState.HEALTHY
            anomaly_severity = SeverityLevel.NOMINAL

        # ---------------------------------------------------------------------
        # 5. Diagnostic Summary Generation
        # ---------------------------------------------------------------------
        summary_parts = []
        if operational_state == OperationalState.HEALTHY:
            summary_parts.append(f"Engine {unit_number} cycle {time_cycles} operating within nominal aerothermal envelope.")
        elif operational_state == OperationalState.CRITICAL:
            warning_names = [b.sensor_name for b in breaches if b.severity == SeverityLevel.WARNING]
            summary_parts.append(
                f"CRITICAL: Flight safety boundary or severe threshold violation detected in: {', '.join(warning_names)}."
            )
        elif operational_state == OperationalState.DEGRADED:
            summary_parts.append(
                f"DEGRADED: Multi-cycle persistent degradation confirmed ({new_count} consecutive anomalous cycles)."
            )
        elif operational_state == OperationalState.ANOMALOUS:
            reasons = []
            if len(breaches) > 0:
                reasons.append(f"{len(breaches)} sensor threshold breach(es)")
            if isolation_forest_breach:
                reasons.append(f"iForest score ({if_score:.4f} > {if_tau:.4f})")
            if pca_spe_breach:
                reasons.append(f"PCA-SPE score ({pca_score:.4f} > {pca_tau:.4f})")
            summary_parts.append(f"ANOMALOUS: Transient anomaly detected ({', '.join(reasons)}).")

        summary_message = " ".join(summary_parts)

        # ---------------------------------------------------------------------
        # 6. Metadata & Output Construction
        # ---------------------------------------------------------------------
        execution_time_ms = (time.perf_counter() - start_time) * 1000.0
        meta = AgentMetadata(
            agent_id=self.agent_id,
            execution_timestamp=timestamp,
            execution_time_ms=execution_time_ms,
            status=AgentExecutionStatus.SUCCESS,
            version="1.0.0",
        )

        output = MonitoringOutput(
            unit_number=unit_number,
            time_cycles=time_cycles,
            operational_state=operational_state,
            threshold_breaches=breaches,
            model_anomaly_flag=model_anomaly_flag,
            isolation_forest_breach=isolation_forest_breach,
            pca_spe_breach=pca_spe_breach,
            persistence_count=new_count,
            persistence_alert=persistence_alert,
            anomaly_severity=anomaly_severity,
            summary_message=summary_message,
            metadata=meta,
        )

        output.validate()
        return output

    def process_engine_state(self, state: Any) -> MonitoringOutput:
        """
        Convenience adapter to ingest a Digital Twin EngineState instance.

        Args:
            state: EngineState instance from digital_twin.engine_state.

        Returns:
            MonitoringOutput generated by the Monitoring Agent.
        """
        # Validate that state is an EngineState or possesses the required attributes
        required_attrs = ("unit_number", "time_cycles", "altitude", "mach_number", "T30", "P30")
        for attr in required_attrs:
            if not hasattr(state, attr):
                raise TypeError(f"Provided state lacks required EngineState attribute '{attr}'")

        operating_context = {
            "altitude": float(state.altitude),
            "mach_number": float(state.mach_number),
            "throttle_resolver_angle": float(state.throttle_resolver_angle),
        }

        # Extract 21 C-MAPSS gas-path sensors
        sensor_fields = [
            "T2", "T24", "T30", "T50", "P2", "P15", "P30", "Nf", "Nc", "epr",
            "Ps30", "phi", "NRf", "NRc", "BPR", "farB", "htBleed", "Nf_dmd",
            "PCNfR_dmd", "W31", "W32"
        ]
        raw_telemetry: Dict[str, float] = {}
        for sf in sensor_fields:
            val = getattr(state, sf, None)
            if val is not None and math.isfinite(val):
                raw_telemetry[sf] = float(val)

        derived_indicators = {
            "hpc_temp_rise": float(state.hpc_temp_rise),
            "overall_pressure_ratio": float(state.overall_pressure_ratio),
            "spool_speed_ratio": float(state.spool_speed_ratio),
        }

        anomaly_scores: Dict[str, Optional[float]] = {}
        if hasattr(state, "anomaly_score_m1"):
            anomaly_scores["isolation_forest"] = state.anomaly_score_m1
        if hasattr(state, "anomaly_score_m2"):
            anomaly_scores["pca_spe"] = state.anomaly_score_m2

        inp = MonitoringInput(
            unit_number=int(state.unit_number),
            time_cycles=int(state.time_cycles),
            operating_context=operating_context,
            raw_telemetry=raw_telemetry,
            derived_indicators=derived_indicators,
            anomaly_scores=anomaly_scores,
        )

        return self.process_input(inp)

    def evaluate_sequence_with_lstm_ae(self, sequence: Any) -> Optional[Dict[str, Any]]:
        """Evaluates multivariate sensor sequence with integrated LSTM-AE detector."""
        if self.lstm_ae_detector is None:
            return None
        return self.lstm_ae_detector.evaluate_sequence(sequence)

    evaluate = process_input
