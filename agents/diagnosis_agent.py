"""
Diagnosis Agent Module - Phase 6 Step 6.3
Autonomous Diagnostic Root-Cause Agent for the AI-Based Aircraft Engine Multi-Agent System.

Roadmap Reference: Phase 6 - Step 6.3:
"Implement the Diagnosis Agent using documented rules or validated anomaly evidence"

Capabilities:
1. Baseline Reference Model:
   - Configurable nominal reference values per operating regime / subset to compute deviation deltas.
2. Evidence Synthesizer:
   - Converts MonitoringOutput threshold breaches and anomalous sensor readings into structured FaultEvidence instances.
3. Diagnostic Rule Engine matching physical signatures to fault modes:
   - THERMAL_DISTRESS: Elevated T30, excessive delta T_HPC, elevated T50.
   - HPC_EFFICIENCY_DEGRADATION: Elevated T30 combined with depressed OPR or depressed Ps30.
   - SENSOR_CALIBRATION_BIAS: Isolated sensor breach while thermodynamic companion sensors remain nominal.
   - GENERAL_CORE_DEGRADATION: Multi-sensor gradual divergence with persistent multi-cycle alert.
   - NOMINAL_OPERATION: Clean monitoring output with zero breaches and nominal state.
4. Calibrated Confidence Scorer:
   - Evaluates symptom match coherence, breach severity, and persistence history.
   - Strictly enforces confidence bounded in [0.0, 0.95] (never claims 1.0 certainty, reserving ground truth for teardown).
5. Maintenance Action Recommender:
   - Provides actionable, non-commanding maintenance recommendations for flight crews and ground engineers.
6. Standardized Type-Safe Schema Generation:
   - Ingests DiagnosisInput or MonitoringOutput, producing fully validated DiagnosisOutput with AgentMetadata.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
import os
import sys
import time
from typing import Dict, Any, List, Optional, Tuple, Union

from agents.schemas import (
    AgentExecutionStatus,
    SeverityLevel,
    OperationalState,
    AgentMetadata,
    ThresholdBreach,
    MonitoringInput,
    MonitoringOutput,
    FaultEvidence,
    FaultCandidate,
    DiagnosisInput,
    DiagnosisOutput,
)
from agents.monitoring_agent import SENSOR_ALIAS_MAP


# -----------------------------------------------------------------------------
# 1. Nominal Baseline Reference Envelopes (FD001 Sea-Level Static Baseline)
# -----------------------------------------------------------------------------
DEFAULT_FD001_BASELINES: Dict[str, float] = {
    # Temperatures (°R)
    "T2": 518.67,
    "T24": 642.35,
    "T30": 1585.00,
    "T50": 1400.00,
    # Pressures (psia)
    "P2": 14.62,
    "P15": 21.61,
    "P30": 553.50,
    "Ps30": 47.45,
    # Spool Speeds (rpm)
    "Nf": 2388.05,
    "Nc": 9045.00,
    # Ratios & Bleeds
    "epr": 1.30,
    "phi": 521.80,
    "NRf": 2388.08,
    "NRc": 8130.00,
    "BPR": 8.42,
    "farB": 0.03,
    "htBleed": 392.00,
    "Nf_dmd": 2388.00,
    "PCNfR_dmd": 100.00,
    "W31": 38.85,
    "W32": 23.35,
    # Macroscopic Derived Indicators
    "hpc_temp_rise": 942.65,
    "overall_pressure_ratio": 37.86,
    "spool_speed_ratio": 3.79,
}

# Thermodynamic companion groupings for sensor bias discrimination
SENSOR_COMPANION_GROUPS: Dict[str, List[str]] = {
    "P30": ["P2", "P15", "Ps30", "overall_pressure_ratio"],
    "Ps30": ["P30", "P2", "P15"],
    "T30": ["T24", "T50", "hpc_temp_rise"],
    "T24": ["T2", "T30"],
    "T50": ["T30", "T24"],
    "Nf": ["Nc", "spool_speed_ratio", "NRf"],
    "Nc": ["Nf", "spool_speed_ratio", "NRc"],
}


class DiagnosisAgent:
    """
    Autonomous Aerospace Diagnostic Agent.
    Synthesizes monitoring evidence, evaluates physical fault signatures,
    ranks root-cause fault candidates with bounded confidence, and recommends maintenance actions.
    """

    # Maximum confidence limit enforced by aerospace safety protocol
    MAX_CONFIDENCE_BOUND: float = 0.95

    def __init__(
        self,
        agent_id: str = "diagnosis_agent_v1",
        subset: str = "FD001",
        baseline_references: Optional[Dict[str, float]] = None,
        confidence_ceiling: float = 0.95,
        cnn_fault_classifier: Optional[Any] = None,
    ):
        """
        Initializes the Diagnosis Agent.

        Args:
            agent_id: Unique agent identifier for metadata tracking.
            subset: C-MAPSS dataset subset identifier ("FD001", "FD002", etc.).
            baseline_references: Nominal sensor reference dictionary. Defaults to FD001 nominal baselines.
            confidence_ceiling: Maximum allowed confidence score in [0.0, 1.0], strictly bounded <= 0.95.
            cnn_fault_classifier: Optional 1D-CNN deep fault classifier for temporal sequence pattern diagnosis.
        """
        self.agent_id = agent_id
        self.subset = subset.upper()
        self.confidence_ceiling = min(float(confidence_ceiling), self.MAX_CONFIDENCE_BOUND)
        self.cnn_fault_classifier = cnn_fault_classifier

        # Baseline reference table
        self.baseline_references = dict(DEFAULT_FD001_BASELINES)
        if baseline_references is not None:
            for k, v in baseline_references.items():
                norm_k = self._normalize_key(k)
                self.baseline_references[norm_k] = float(v)

    @staticmethod
    def _normalize_key(key: str) -> str:
        """Normalizes sensor and indicator names using alias map."""
        clean_k = key.strip()
        return SENSOR_ALIAS_MAP.get(clean_k, clean_k)

    def synthesize_evidence(
        self,
        monitoring_output: MonitoringOutput,
        telemetry_snapshot: Optional[Dict[str, float]] = None,
    ) -> List[FaultEvidence]:
        """
        Converts MonitoringOutput threshold breaches and telemetry readings into structured FaultEvidence instances.

        Args:
            monitoring_output: Output from the Monitoring Agent.
            telemetry_snapshot: Optional dictionary of raw sensor readings.

        Returns:
            List of structured FaultEvidence instances with delta against baseline references.
        """
        evidence_list: List[FaultEvidence] = []
        recorded_sensors = set()

        # 1. Synthesize evidence from active threshold breaches
        for breach in monitoring_output.threshold_breaches:
            norm_name = self._normalize_key(breach.sensor_name)
            observed = breach.observed_value
            base_ref = self.baseline_references.get(norm_name, breach.threshold_limit)
            delta = observed - base_ref
            direction = breach.direction

            desc = (
                f"{norm_name} threshold breach ({direction}): observed {observed:.2f} "
                f"vs baseline {base_ref:.2f} (delta: {delta:+.2f}, margin: {breach.breach_margin:.2f})"
            )
            evidence_list.append(
                FaultEvidence(
                    sensor_name=norm_name,
                    observed_value=observed,
                    baseline_reference=base_ref,
                    delta=delta,
                    direction=direction,
                    description=desc,
                )
            )
            recorded_sensors.add(norm_name)

        # 2. Add telemetry deviations that may not have breached absolute thresholds
        if telemetry_snapshot:
            for k, v in telemetry_snapshot.items():
                norm_k = self._normalize_key(k)
                if norm_k in recorded_sensors or norm_k not in self.baseline_references:
                    continue

                if not math.isfinite(v):
                    continue

                base_ref = self.baseline_references[norm_k]
                delta = v - base_ref

                # Only include noticeable deviations (> 2% from baseline)
                if abs(base_ref) > 1e-5:
                    pct_diff = abs(delta) / abs(base_ref)
                else:
                    pct_diff = abs(delta)

                if pct_diff >= 0.02:
                    dir_str = "HIGH" if delta > 0 else "LOW"
                    desc = f"{norm_k} deviation: observed {v:.2f} vs baseline {base_ref:.2f} (delta: {delta:+.2f})"
                    evidence_list.append(
                        FaultEvidence(
                            sensor_name=norm_k,
                            observed_value=float(v),
                            baseline_reference=base_ref,
                            delta=delta,
                            direction=dir_str,
                            description=desc,
                        )
                    )
                    recorded_sensors.add(norm_k)

        # Validate all synthesized evidences
        for ev in evidence_list:
            ev.validate()

        return evidence_list

    def _evaluate_symptom_coherence(
        self,
        candidate_type: str,
        evidence: List[FaultEvidence],
        telemetry: Dict[str, float],
        persistence_count: int,
    ) -> Tuple[float, List[FaultEvidence], str]:
        """
        Matches evidence against physical aerothermal fault mode signatures,
        returning (raw_confidence, contributing_evidence, recommended_action).
        """
        ev_map = {e.sensor_name: e for e in evidence}
        contributing: List[FaultEvidence] = []

        if candidate_type == "SENSOR_CALIBRATION_BIAS":
            # An isolated sensor breach where thermodynamic companion sensors show zero anomaly
            # Identify the isolated breach
            breached_names = list(ev_map.keys())
            if len(breached_names) == 1:
                target_sensor = breached_names[0]
                companions = SENSOR_COMPANION_GROUPS.get(target_sensor, [])
                companion_breaches = [c for c in companions if c in ev_map]

                if len(companion_breaches) == 0:
                    contributing.append(ev_map[target_sensor])
                    # High symptom coherence for isolated breach
                    raw_conf = 0.85
                    if persistence_count >= 3:
                        raw_conf += 0.05
                    action = (
                        f"Perform bench calibration and loop-check on sensor '{target_sensor}'. "
                        "Inspect pneumatic sense lines and thermocouple wiring before engine removal."
                    )
                    return raw_conf, contributing, action

            return 0.0, [], ""

        elif candidate_type == "HPC_EFFICIENCY_DEGRADATION":
            # Aerodynamic signature: T30 elevated (or delta_T elevated) COMBINED WITH depressed OPR or depressed Ps30
            has_t30_high = False
            has_opr_low = False
            has_ps30_low = False

            if "T30" in ev_map and ev_map["T30"].direction == "HIGH":
                has_t30_high = True
                contributing.append(ev_map["T30"])
            elif "hpc_temp_rise" in ev_map and ev_map["hpc_temp_rise"].direction == "HIGH":
                has_t30_high = True
                contributing.append(ev_map["hpc_temp_rise"])

            # Check OPR low
            if "overall_pressure_ratio" in ev_map and ev_map["overall_pressure_ratio"].direction == "LOW":
                has_opr_low = True
                contributing.append(ev_map["overall_pressure_ratio"])
            elif "overall_pressure_ratio" in telemetry:
                opr_val = telemetry["overall_pressure_ratio"]
                opr_base = self.baseline_references.get("overall_pressure_ratio", 37.86)
                if opr_val < opr_base - 0.5:
                    has_opr_low = True

            # Check Ps30 or P30 low
            if "Ps30" in ev_map and ev_map["Ps30"].direction == "LOW":
                has_ps30_low = True
                contributing.append(ev_map["Ps30"])
            elif "P30" in ev_map and ev_map["P30"].direction == "LOW":
                has_ps30_low = True
                contributing.append(ev_map["P30"])

            if has_t30_high and (has_opr_low or has_ps30_low):
                # Strong aerodynamic coupled signature
                raw_conf = 0.82
                if persistence_count >= 3:
                    raw_conf += 0.08
                action = (
                    "Perform borescope inspection of HPC rotor blade tips and variable stator vanes (VSV). "
                    "Execute engine core compressor detergent wash."
                )
                return raw_conf, contributing, action
            elif has_t30_high:
                # Partial symptom
                return 0.40, contributing, "Inspect compressor exit aerothermal instrumentation."

            return 0.0, [], ""

        elif candidate_type == "THERMAL_DISTRESS":
            # Elevated T30, excessive delta T_HPC, elevated T50
            has_t30 = "T30" in ev_map and ev_map["T30"].direction == "HIGH"
            has_t50 = "T50" in ev_map and ev_map["T50"].direction == "HIGH"
            has_dt = "hpc_temp_rise" in ev_map and ev_map["hpc_temp_rise"].direction == "HIGH"

            if has_t30:
                contributing.append(ev_map["T30"])
            if has_t50:
                contributing.append(ev_map["T50"])
            if has_dt:
                contributing.append(ev_map["hpc_temp_rise"])

            matched_count = sum([has_t30, has_t50, has_dt])
            if matched_count >= 2:
                raw_conf = 0.80 + (0.05 if matched_count == 3 else 0.0)
                if persistence_count >= 3:
                    raw_conf += 0.05
                action = (
                    "Inspect combustor fuel nozzles for streaking/coking; perform borescope inspection "
                    "of High-Pressure Turbine (HPT) stage 1 nozzle guide vanes and blade leading edges."
                )
                return raw_conf, contributing, action
            elif matched_count == 1:
                return 0.50, contributing, "Monitor hot-section EGT margins and verify bleed valve operation."

            return 0.0, [], ""

        elif candidate_type == "GENERAL_CORE_DEGRADATION":
            # Broad multi-sensor drift with persistent degradation alert
            if persistence_count >= 3 or len(evidence) >= 3:
                for ev in evidence:
                    if ev not in contributing:
                        contributing.append(ev)
                raw_conf = 0.75 + min(0.12, 0.03 * len(evidence))
                action = (
                    "Schedule detailed gas turbine module health assessment; monitor remaining useful life "
                    "and review fleet maintenance interval adjustments."
                )
                return raw_conf, contributing, action

            return 0.0, [], ""

        return 0.0, [], ""

    def diagnose(
        self,
        diagnosis_input: Union[DiagnosisInput, MonitoringOutput, Dict[str, Any]],
        telemetry_snapshot: Optional[Dict[str, float]] = None,
    ) -> DiagnosisOutput:
        """
        Executes autonomous diagnostic reasoning on the input monitoring and telemetry evidence.

        Args:
            diagnosis_input: DiagnosisInput instance, MonitoringOutput instance, or serialized dict.
            telemetry_snapshot: Optional dictionary of sensor telemetry if MonitoringOutput was passed.

        Returns:
            Type-safe, fully validated DiagnosisOutput instance.
        """
        start_time = time.perf_counter()
        timestamp = datetime.now(timezone.utc).isoformat()

        # ---------------------------------------------------------------------
        # 1. Ingestion & Input Adaption
        # ---------------------------------------------------------------------
        if isinstance(diagnosis_input, DiagnosisInput):
            d_inp = diagnosis_input
            d_inp.validate()
            unit_number = d_inp.unit_number
            time_cycles = d_inp.time_cycles
            mon_out = d_inp.monitoring_output
            telemetry = {self._normalize_key(k): float(v) for k, v in d_inp.telemetry_snapshot.items()}
            persistence_count = max(d_inp.historical_persistence_count, mon_out.persistence_count)

        elif isinstance(diagnosis_input, MonitoringOutput):
            mon_out = diagnosis_input
            mon_out.validate()
            unit_number = mon_out.unit_number
            time_cycles = mon_out.time_cycles
            telemetry = {}
            if telemetry_snapshot is not None:
                telemetry = {self._normalize_key(k): float(v) for k, v in telemetry_snapshot.items()}
            persistence_count = mon_out.persistence_count

        elif isinstance(diagnosis_input, dict):
            if "monitoring_output" in diagnosis_input:
                d_inp = DiagnosisInput.from_dict(diagnosis_input)
                d_inp.validate()
                unit_number = d_inp.unit_number
                time_cycles = d_inp.time_cycles
                mon_out = d_inp.monitoring_output
                telemetry = {self._normalize_key(k): float(v) for k, v in d_inp.telemetry_snapshot.items()}
                persistence_count = max(d_inp.historical_persistence_count, mon_out.persistence_count)
            else:
                mon_out = MonitoringOutput.from_dict(diagnosis_input)
                mon_out.validate()
                unit_number = mon_out.unit_number
                time_cycles = mon_out.time_cycles
                telemetry = {}
                if telemetry_snapshot is not None:
                    telemetry = {self._normalize_key(k): float(v) for k, v in telemetry_snapshot.items()}
                persistence_count = mon_out.persistence_count
        else:
            raise TypeError(f"Expected DiagnosisInput, MonitoringOutput, or dict, got {type(diagnosis_input)}")

        # Validate finite inputs
        for k, v in telemetry.items():
            if not math.isfinite(v):
                raise ValueError(f"DiagnosisAgent: non-finite telemetry value for {k}: {v}")

        # ---------------------------------------------------------------------
        # 2. Case A: Nominal Operation
        # ---------------------------------------------------------------------
        is_healthy = (
            mon_out.operational_state == OperationalState.HEALTHY
            and len(mon_out.threshold_breaches) == 0
            and not mon_out.model_anomaly_flag
            and mon_out.persistence_count == 0
        )

        if is_healthy:
            exec_time = (time.perf_counter() - start_time) * 1000.0
            meta = AgentMetadata(
                agent_id=self.agent_id,
                execution_timestamp=timestamp,
                execution_time_ms=exec_time,
                status=AgentExecutionStatus.SUCCESS,
                version="1.0.0",
            )
            out = DiagnosisOutput(
                unit_number=unit_number,
                time_cycles=time_cycles,
                diagnostic_hypotheses=[],
                fault_candidates=[],
                primary_fault=None,
                confidence_score=0.0,
                contributing_evidence=[],
                severity=SeverityLevel.NOMINAL,
                requires_maintenance_action=False,
                diagnostic_summary="Nominal engine operation. All gas-path aerothermal parameters within baseline tolerances.",
                metadata=meta,
            )
            out.validate()
            return out

        # ---------------------------------------------------------------------
        # 3. Evidence Synthesis
        # ---------------------------------------------------------------------
        evidence = self.synthesize_evidence(mon_out, telemetry)

        # ---------------------------------------------------------------------
        # 4. Diagnostic Rule Engine Evaluation
        # ---------------------------------------------------------------------
        candidates: List[FaultCandidate] = []
        fault_types_to_evaluate = [
            ("SENSOR_CALIBRATION_BIAS", "SENSOR"),
            ("HPC_EFFICIENCY_DEGRADATION", "HPC"),
            ("THERMAL_DISTRESS", "HOT_SECTION"),
            ("GENERAL_CORE_DEGRADATION", "ENGINE_CORE"),
        ]

        for f_type, comp in fault_types_to_evaluate:
            raw_conf, contrib_ev, action = self._evaluate_symptom_coherence(
                f_type, evidence, telemetry, persistence_count
            )
            if raw_conf > 0.0:
                # Bounded confidence: strictly in [0.0, confidence_ceiling]
                bounded_conf = min(float(raw_conf), self.confidence_ceiling)
                candidate = FaultCandidate(
                    fault_type=f_type,
                    component=comp,
                    likelihood=bounded_conf,
                    contributing_evidence=contrib_ev,
                    recommended_action=action,
                )
                candidate.validate()
                candidates.append(candidate)

        # Sort candidates descending by likelihood
        candidates.sort(key=lambda c: c.likelihood, reverse=True)

        # ---------------------------------------------------------------------
        # 5. Fault Selection & Maintenance Recommendation
        # ---------------------------------------------------------------------
        if len(candidates) > 0:
            top_candidate = candidates[0]
            primary_fault = top_candidate.fault_type
            confidence_score = top_candidate.likelihood
            hypotheses = [c.fault_type for c in candidates]
            all_contributing = top_candidate.contributing_evidence
            requires_action = True

            # Determine diagnostic severity
            if mon_out.operational_state == OperationalState.CRITICAL:
                diag_severity = SeverityLevel.WARNING
            elif mon_out.operational_state == OperationalState.DEGRADED:
                diag_severity = SeverityLevel.CAUTION
            elif mon_out.operational_state == OperationalState.ANOMALOUS:
                diag_severity = SeverityLevel.CAUTION if top_candidate.likelihood >= 0.70 else SeverityLevel.ADVISORY
            else:
                diag_severity = SeverityLevel.ADVISORY

            summary = (
                f"Diagnostic assessment indicates {primary_fault} in {top_candidate.component} "
                f"(confidence: {confidence_score * 100:.1f}%). {top_candidate.recommended_action}"
            )
        else:
            # Anomaly exists in monitoring but does not match a specific component signature
            primary_fault = "UNSPECIFIED_ANOMALOUS_CONDITION"
            confidence_score = 0.35
            hypotheses = ["UNSPECIFIED_ANOMALOUS_CONDITION"]
            all_contributing = evidence
            requires_action = False
            diag_severity = SeverityLevel.ADVISORY
            summary = (
                f"Unclassified aerothermal anomaly detected with {len(evidence)} evidence points. "
                "Recommend continuous monitoring on subsequent flight cycles."
            )

        # ---------------------------------------------------------------------
        # 6. Metadata & Output Assembly
        # ---------------------------------------------------------------------
        exec_time = (time.perf_counter() - start_time) * 1000.0
        meta = AgentMetadata(
            agent_id=self.agent_id,
            execution_timestamp=timestamp,
            execution_time_ms=exec_time,
            status=AgentExecutionStatus.SUCCESS,
            version="1.0.0",
        )

        output = DiagnosisOutput(
            unit_number=unit_number,
            time_cycles=time_cycles,
            diagnostic_hypotheses=hypotheses,
            fault_candidates=candidates,
            primary_fault=primary_fault,
            confidence_score=confidence_score,
            contributing_evidence=all_contributing,
            severity=diag_severity,
            requires_maintenance_action=requires_action,
            diagnostic_summary=summary,
            metadata=meta,
        )

        output.validate()
        return output

    def diagnose_with_cnn(self, sequence: Any, compute_saliency: bool = False) -> Optional[Dict[str, Any]]:
        """Evaluates multivariate sensor sequence with integrated 1D-CNN fault classifier."""
        if self.cnn_fault_classifier is None:
            return None
        return self.cnn_fault_classifier.diagnose_sequence(sequence, compute_saliency=compute_saliency)
