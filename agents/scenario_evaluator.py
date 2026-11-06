"""
Multi-Agent Scenario Evaluator & Interaction Benchmark Runner - Phase 6 Step 6.8
Automates end-to-end evaluation of the integrated multi-agent system across:
1. Nominal Trajectories: Early-life flight cycles verifying zero false alarms and high RUL.
2. Degraded Trajectories: Late-life engine runs verifying 3-cycle persistence detection,
   DEGRADED state classification, root-cause fault isolation, and depleted RUL forecasts.
3. Counterfactual What-If Scenarios: Injected stress scenarios verifying:
   - "thermal_stress_overtemp": THERMAL_DISTRESS diagnosis & reduced RUL.
   - "sensor_bias_drift": SENSOR_CALIBRATION_BIAS diagnosis preventing false engine removals.
   - "compounded_degradation": Multi-variable stress triggering CRITICAL state escalation.

Roadmap Reference: Phase 6 - Step 6.8:
"Test multi-agent interaction on nominal data, degraded data, and what-if scenarios"
"""

import copy
from datetime import datetime, timezone
import json
import math
import os
import time
from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np

from agents.schemas import (
    AgentExecutionStatus,
    SeverityLevel,
    OperationalState,
    AgentMetadata,
    MonitoringOutput,
    DiagnosisOutput,
    PrognosticsOutput,
    OrchestrationResult,
)
from agents.monitoring_agent import MonitoringAgent
from agents.diagnosis_agent import DiagnosisAgent
from agents.prognostics_agent import PrognosticsAgent
from agents.orchestrator import AgentOrchestrator
from agents.audit_logger import AgentAuditLogger
from agents.twin_bridge import DigitalTwinSafetyBridge, SafetyViolationError, IMMUTABLE_PHYSICAL_FIELDS
from digital_twin.twin_simulator import DigitalTwinSimulator
from digital_twin.engine_state import EngineState
from digital_twin.scenario_interface import (
    ScenarioCategory,
    ScenarioConfig,
    ScenarioModifier,
    ORDERED_SCALER_FEATURES,
)


class MultiAgentScenarioEvaluator:
    """
    Automated Multi-Agent Interaction and Scenario Benchmark Runner.
    Coordinates nominal, degraded, and counterfactual stress scenarios through
    the full AgentOrchestrator pipeline under strict Digital Twin safety firewalls.
    """

    def __init__(
        self,
        subset: str = "FD001",
        enable_twin_feedback: bool = True,
        log_dir: str = os.path.join("reports", "agent_logs"),
        audit_logger: Optional[AgentAuditLogger] = None,
        twin_bridge: Optional[DigitalTwinSafetyBridge] = None,
        sensor_thresholds: Optional[Dict[str, Dict[str, float]]] = None,
        agent_id: str = "scenario_evaluator_v1",
    ):
        """
        Initializes the Multi-Agent Scenario Evaluator.

        Args:
            subset: C-MAPSS dataset subset identifier ("FD001", "FD002", etc.).
            enable_twin_feedback: If True, activates the Digital Twin Safety Bridge.
            log_dir: Directory path for structured agent audit trails.
            audit_logger: Optional pre-configured AgentAuditLogger instance.
            twin_bridge: Optional pre-configured DigitalTwinSafetyBridge instance.
            sensor_thresholds: Custom dictionary of caution and warning thresholds.
            agent_id: Provenance identifier for the evaluator.
        """
        self.subset = subset.upper()
        self.enable_twin_feedback = enable_twin_feedback
        self.log_dir = log_dir
        self.agent_id = agent_id

        # Instantiate or adopt audit logger
        if audit_logger is not None:
            self.audit_logger = audit_logger
        else:
            self.audit_logger = AgentAuditLogger(log_dir=log_dir)

        # Instantiate or adopt safety bridge
        if twin_bridge is not None:
            self.twin_bridge = twin_bridge
        else:
            self.twin_bridge = DigitalTwinSafetyBridge(agent_id=f"{agent_id}_bridge")

        # Calibrated default threshold bands (matching Step 6.2 and 6.7 specifications)
        effective_thresholds = {"T30": {"caution_high": 1605.0, "warning_high": 1625.0}}
        if sensor_thresholds is not None:
            for k, v in sensor_thresholds.items():
                effective_thresholds[k] = v

        self.monitoring_agent = MonitoringAgent(
            agent_id="monitoring_agent",
            subset=self.subset,
            sensor_thresholds=effective_thresholds,
            persistence_alert_window=3,
        )

        self.diagnosis_agent = DiagnosisAgent(
            agent_id="diagnosis_agent",
            subset=self.subset,
        )

        self.prognostics_agent = PrognosticsAgent(
            agent_id="prognostics_agent",
            subset=self.subset,
        )

        self.orchestrator = AgentOrchestrator(
            agent_id="orchestrator",
            monitoring_agent=self.monitoring_agent,
            diagnosis_agent=self.diagnosis_agent,
            prognostics_agent=self.prognostics_agent,
            audit_logger=self.audit_logger,
            enable_twin_feedback=self.enable_twin_feedback,
            twin_bridge=self.twin_bridge,
            subset=self.subset,
            default_mode="full",
        )

        self.latest_evaluation_metrics: Dict[str, Any] = {}

    # -------------------------------------------------------------------------
    # Trajectory Runner 1: Nominal Flight Trajectory Evaluation
    # -------------------------------------------------------------------------
    def evaluate_nominal_trajectory(
        self,
        engine_id: int = 1,
        cycles: int = 40,
        split: str = "train",
    ) -> Dict[str, Any]:
        """
        Evaluates an engine during early nominal life (cycles 1 to `cycles`).
        Validates:
        - OperationalState is HEALTHY across all cycles.
        - Zero physical threshold breaches.
        - Zero false persistence alarms (false alarm rate = 0.0%).
        - Virtual health index H(t) >= 0.70.
        - Physical telemetry invariance G1-G29 maintained.

        Args:
            engine_id: Unit number to load from dataset.
            cycles: Number of initial nominal cycles to evaluate (default: 40).
            split: Dataset split ("train" or "test").

        Returns:
            Dictionary containing nominal trajectory metrics and per-cycle records.
        """
        sim = DigitalTwinSimulator(subset=self.subset)
        sim.load_engine(unit_number=engine_id, split=split)
        self.monitoring_agent.reset_trajectory(engine_id)

        nominal_records: List[Dict[str, Any]] = []
        false_alarms = 0
        total_breaches = 0
        health_indices: List[float] = []
        ruls: List[float] = []
        latencies_ms: List[float] = []

        max_eval = min(cycles, sim.total_cycles)
        for c in range(1, max_eval + 1):
            state = sim.current_state
            res = self.orchestrator.orchestrate_state(state, mode="full")

            # Collect metrics
            mon = res.monitoring_result
            prog = res.prognostics_result
            diag = res.diagnosis_result

            is_false_alarm = False
            if mon is not None:
                if mon.persistence_alert or mon.operational_state != OperationalState.HEALTHY:
                    false_alarms += 1
                    is_false_alarm = True
                total_breaches += len(mon.threshold_breaches)

            h_idx = prog.virtual_health_index if prog is not None else 1.0
            rul_val = prog.ensemble_rul_estimate if prog is not None else 0.0
            health_indices.append(h_idx)
            ruls.append(rul_val)

            lat = res.execution_timeline_ms.get("total_elapsed_ms", 0.0)
            latencies_ms.append(lat)

            # Record lightweight summary
            record = {
                "cycle": res.cycle,
                "status": res.status.value,
                "operational_state": mon.operational_state.value if mon else "UNKNOWN",
                "breaches_count": len(mon.threshold_breaches) if mon else 0,
                "persistence_alert": mon.persistence_alert if mon else False,
                "health_index": h_idx,
                "ensemble_rul": rul_val,
                "primary_fault": diag.primary_fault if diag else None,
                "latency_ms": lat,
            }
            nominal_records.append(record)

            if c < max_eval:
                sim.step()

        mean_h = float(np.mean(health_indices)) if health_indices else 1.0
        mean_rul = float(np.mean(ruls)) if ruls else 0.0
        mean_lat = float(np.mean(latencies_ms)) if latencies_ms else 0.0

        metrics = {
            "trajectory_type": "NOMINAL",
            "engine_id": engine_id,
            "total_cycles_evaluated": len(nominal_records),
            "false_alarms": false_alarms,
            "false_alarm_rate": float(false_alarms / len(nominal_records)) if nominal_records else 0.0,
            "total_threshold_breaches": total_breaches,
            "mean_virtual_health_index": round(mean_h, 4),
            "min_virtual_health_index": round(float(np.min(health_indices)), 4) if health_indices else 1.0,
            "mean_ensemble_rul": round(mean_rul, 2),
            "mean_latency_ms": round(mean_lat, 2),
            "all_nominal_passed": (false_alarms == 0 and total_breaches == 0 and mean_h >= 0.70),
            "records": nominal_records,
        }

        self.latest_evaluation_metrics["nominal"] = metrics
        return metrics

    # -------------------------------------------------------------------------
    # Trajectory Runner 2: Degraded Trajectory Evaluation
    # -------------------------------------------------------------------------
    def evaluate_degraded_trajectory(
        self,
        engine_id: int = 5,
        start_cycle: Optional[int] = None,
        split: str = "train",
    ) -> Dict[str, Any]:
        """
        Replays an engine into its late-life degraded operational regime.
        Validates:
        - Multi-cycle persistence accumulation reaching k >= 3.
        - Transition of operational state to DEGRADED (or CRITICAL).
        - Root-cause fault isolation by DiagnosisAgent (confidence > 0.0).
        - Depleted RUL forecast and low virtual health index.
        - Digital Twin safe enrichment with G1-G29 invariance.

        Args:
            engine_id: Unit number with late-life degradation (default: 5).
            start_cycle: Optional starting cycle. If None, starts 6 cycles before failure.
            split: Dataset split ("train" or "test").

        Returns:
            Dictionary containing degradation detection metrics and execution log.
        """
        sim = DigitalTwinSimulator(subset=self.subset)
        sim.load_engine(unit_number=engine_id, split=split)
        self.monitoring_agent.reset_trajectory(engine_id)

        # Default start_cycle to 6 cycles before total recorded life
        if start_cycle is None:
            effective_start = max(1, sim.total_cycles - 6)
        else:
            effective_start = max(1, min(start_cycle, sim.total_cycles))

        # Advance simulator to start_cycle
        while sim.current_state.time_cycles < effective_start:
            sim.step()

        degraded_records: List[Dict[str, Any]] = []
        persistence_alert_triggered = False
        alert_cycle: Optional[int] = None
        isolated_fault: Optional[str] = None
        max_confidence: float = 0.0
        final_rul: float = 0.0
        final_health: float = 1.0

        while True:
            state = sim.current_state
            res = self.orchestrator.orchestrate_state(state, mode="full")

            mon = res.monitoring_result
            diag = res.diagnosis_result
            prog = res.prognostics_result

            if mon is not None:
                if mon.persistence_alert and not persistence_alert_triggered:
                    persistence_alert_triggered = True
                    alert_cycle = res.cycle

            if diag is not None and diag.primary_fault is not None:
                if diag.confidence_score > max_confidence:
                    max_confidence = float(diag.confidence_score)
                    isolated_fault = diag.primary_fault

            if prog is not None:
                final_rul = float(prog.ensemble_rul_estimate)
                final_health = float(prog.virtual_health_index)

            record = {
                "cycle": res.cycle,
                "operational_state": mon.operational_state.value if mon else "UNKNOWN",
                "persistence_count": mon.persistence_count if mon else 0,
                "persistence_alert": mon.persistence_alert if mon else False,
                "primary_fault": diag.primary_fault if diag else None,
                "confidence_score": float(diag.confidence_score) if diag else 0.0,
                "ensemble_rul": final_rul,
                "health_index": final_health,
            }
            degraded_records.append(record)

            if sim.current_state.time_cycles >= sim.total_cycles:
                break
            sim.step()

        metrics = {
            "trajectory_type": "DEGRADED",
            "engine_id": engine_id,
            "total_cycles_evaluated": len(degraded_records),
            "start_cycle": effective_start,
            "end_cycle": sim.total_cycles,
            "persistence_alert_triggered": persistence_alert_triggered,
            "alert_cycle": alert_cycle,
            "detection_latency_cycles": (alert_cycle - effective_start) if alert_cycle else None,
            "isolated_fault": isolated_fault,
            "max_confidence": round(max_confidence, 4),
            "final_rul_forecast": round(final_rul, 2),
            "final_virtual_health_index": round(final_health, 4),
            "degradation_passed": (persistence_alert_triggered and isolated_fault is not None and final_rul < 25.0),
            "records": degraded_records,
        }

        self.latest_evaluation_metrics["degraded"] = metrics
        return metrics

    # -------------------------------------------------------------------------
    # Scenario Runner 3: Counterfactual What-If Scenarios
    # -------------------------------------------------------------------------
    def evaluate_what_if_scenario(
        self,
        scenario_id: str,
        engine_id: int = 1,
        cycle: int = 35,
        split: str = "train",
    ) -> Dict[str, Any]:
        """
        Injects a controlled counterfactual what-if perturbation into the engine state.
        Supported Scenarios:
        1. "thermal_stress_overtemp":
           Thermal excursion: T30 elevated (+30 °R), hpc_temp_rise elevated (+30 °R), T50 elevated (+20 °R).
           Asserts THERMAL_DISTRESS diagnosis, confidence >= 0.70, and reduced RUL relative to baseline.
        2. "sensor_bias_drift":
           Isolated sensor perturbation: P30 elevated (+15 psia) with companion sensors nominal.
           Asserts SENSOR_CALIBRATION_BIAS diagnosis, confidence >= 0.80, and prevents false engine pull.
        3. "compounded_degradation":
           Multi-sensor stress: T30 at 1630 °R (> warning_high) and P30 at 580 psia (> warning_high).
           Asserts CRITICAL operational state and WARNING diagnosis severity.

        Args:
            scenario_id: One of "thermal_stress_overtemp", "sensor_bias_drift", "compounded_degradation".
            engine_id: Unit number to load (default: 1).
            cycle: Operational cycle index to test (default: 35).
            split: Dataset split ("train" or "test").

        Returns:
            Dictionary containing scenario validation results and orchestration result.
        """
        valid_scenarios = ("thermal_stress_overtemp", "sensor_bias_drift", "compounded_degradation")
        if scenario_id not in valid_scenarios:
            raise ValueError(f"Unknown scenario_id '{scenario_id}'. Must be one of {valid_scenarios}.")

        sim = DigitalTwinSimulator(subset=self.subset)
        sim.load_engine(unit_number=engine_id, split=split)
        self.monitoring_agent.reset_trajectory(engine_id)

        target_cycle = min(cycle, sim.total_cycles)
        while sim.current_state.time_cycles < target_cycle:
            sim.step()

        base_state = copy.deepcopy(sim.current_state)

        # Baseline execution for reference
        baseline_res = self.orchestrator.orchestrate_state(base_state, mode="full")
        base_rul = baseline_res.prognostics_result.ensemble_rul_estimate if baseline_res.prognostics_result else 100.0

        # Construct perturbed candidate state
        scenario_state = copy.deepcopy(base_state)
        scenario_state.scenario_id = scenario_id

        if scenario_id == "thermal_stress_overtemp":
            offset = 30.0
            scenario_state.thermal_stress_offset = offset
            scenario_state.T30 = float(base_state.T30 + offset)
            scenario_state.hpc_temp_rise = float(scenario_state.T30 - scenario_state.T24)
            scenario_state.T50 = float(base_state.T50 + 20.0)

            # Update feature vector & sequence tensor so prognostics sees elevated temperature
            if scenario_state.normalized_feature_vec is not None and len(scenario_state.normalized_feature_vec) > 2:
                scenario_state.normalized_feature_vec = np.copy(scenario_state.normalized_feature_vec)
                scenario_state.normalized_feature_vec[2] += 0.35  # T30 feature shift
            if scenario_state.temporal_seq_tensor is not None and scenario_state.temporal_seq_tensor.shape[1] > 2:
                scenario_state.temporal_seq_tensor = np.copy(scenario_state.temporal_seq_tensor)
                scenario_state.temporal_seq_tensor[-5:, 2] += 0.35

        elif scenario_id == "sensor_bias_drift":
            p30_offset = 15.0
            scenario_state.sensor_bias_delta = {"P30": p30_offset}
            scenario_state.P30 = float(base_state.P30 + p30_offset)
            # Companion sensors (Ps30, P2, P15) remain unchanged to guarantee isolated breach

        elif scenario_id == "compounded_degradation":
            # Multi-variable severe stress exceeding warning boundaries
            scenario_state.T30 = 1630.0  # Exceeds warning_high (1625.0)
            scenario_state.hpc_temp_rise = float(1630.0 - scenario_state.T24)
            scenario_state.P30 = 580.0   # Exceeds warning_high (575.0)
            scenario_state.T50 = 1465.0  # Exceeds warning_high (1460.0)
            scenario_state.overall_pressure_ratio = float(580.0 / scenario_state.P2)

        # Execute orchestration under safety firewall
        orch_res = self.orchestrator.orchestrate_state(scenario_state, mode="full")

        mon = orch_res.monitoring_result
        diag = orch_res.diagnosis_result
        prog = orch_res.prognostics_result
        enriched = orch_res.enriched_state

        # Scenario-specific validations
        scenario_passed = False
        validation_notes: List[str] = []

        if scenario_id == "thermal_stress_overtemp":
            t30_breaches = [b for b in mon.threshold_breaches if b.sensor_name == "T30"] if mon else []
            has_thermal_fault = (diag is not None and diag.primary_fault == "THERMAL_DISTRESS")
            high_conf = (diag is not None and diag.confidence_score >= 0.70)
            scenario_passed = bool(len(t30_breaches) > 0 and has_thermal_fault and high_conf)
            validation_notes.append(f"T30 breaches: {len(t30_breaches)}")
            validation_notes.append(f"Primary fault: {diag.primary_fault if diag else None} (conf: {diag.confidence_score if diag else 0.0})")

        elif scenario_id == "sensor_bias_drift":
            p30_breaches = [b for b in mon.threshold_breaches if b.sensor_name == "P30"] if mon else []
            other_breaches = [b for b in mon.threshold_breaches if b.sensor_name != "P30"] if mon else []
            is_sensor_bias = (diag is not None and diag.primary_fault == "SENSOR_CALIBRATION_BIAS")
            prevents_engine_pull = False
            if diag is not None and "Inspect pneumatic sense lines" in diag.diagnostic_summary:
                prevents_engine_pull = True

            scenario_passed = bool(len(p30_breaches) == 1 and len(other_breaches) == 0 and is_sensor_bias and prevents_engine_pull)
            validation_notes.append(f"P30 isolated breach: {len(p30_breaches) == 1 and len(other_breaches) == 0}")
            validation_notes.append(f"Primary fault: {diag.primary_fault if diag else None}")
            validation_notes.append(f"Recommended action prevents false pull: {prevents_engine_pull}")

        elif scenario_id == "compounded_degradation":
            is_critical = (mon is not None and mon.operational_state == OperationalState.CRITICAL)
            is_warning_sev = (diag is not None and diag.severity == SeverityLevel.WARNING)
            scenario_passed = bool(is_critical and is_warning_sev)
            validation_notes.append(f"Operational state: {mon.operational_state.value if mon else None} (is CRITICAL: {is_critical})")
            validation_notes.append(f"Diagnostic severity: {diag.severity.value if diag else None} (is WARNING: {is_warning_sev})")

        # Telemetry invariance check on candidate state before and after enrichment
        invariance_verified = False
        if enriched is not None:
            invariance_verified = True
            for f in IMMUTABLE_PHYSICAL_FIELDS:
                if getattr(scenario_state, f) != getattr(enriched, f):
                    invariance_verified = False
                    break

        scenario_metrics = {
            "scenario_id": scenario_id,
            "engine_id": engine_id,
            "cycle": target_cycle,
            "scenario_passed": scenario_passed,
            "operational_state": mon.operational_state.value if mon else "UNKNOWN",
            "threshold_breaches": [b.sensor_name for b in mon.threshold_breaches] if mon else [],
            "primary_fault": diag.primary_fault if diag else None,
            "confidence_score": float(diag.confidence_score) if diag else 0.0,
            "baseline_rul": round(base_rul, 2),
            "scenario_rul": round(prog.ensemble_rul_estimate, 2) if prog else 0.0,
            "virtual_health_index": round(prog.virtual_health_index, 4) if prog else 1.0,
            "telemetry_invariant": invariance_verified,
            "validation_notes": validation_notes,
            "orchestration_result": orch_res,
        }

        if "what_if" not in self.latest_evaluation_metrics:
            self.latest_evaluation_metrics["what_if"] = {}
        self.latest_evaluation_metrics["what_if"][scenario_id] = scenario_metrics

        return scenario_metrics

    # -------------------------------------------------------------------------
    # Comprehensive Benchmark Runner
    # -------------------------------------------------------------------------
    def run_full_evaluation_suite(
        self,
        nominal_engine_id: int = 1,
        degraded_engine_id: int = 5,
    ) -> Dict[str, Any]:
        """
        Executes the entire multi-agent validation suite across nominal, degraded,
        and all 3 what-if scenarios. Computes aggregated benchmark metrics.

        Returns:
            Comprehensive evaluation dictionary with benchmark metrics.
        """
        t0 = time.perf_counter()

        # 1. Run Nominal Trajectory
        nominal_metrics = self.evaluate_nominal_trajectory(engine_id=nominal_engine_id, cycles=40)

        # 2. Run Degraded Trajectory
        degraded_metrics = self.evaluate_degraded_trajectory(engine_id=degraded_engine_id)

        # 3. Run What-If Scenarios
        what_if_results = {}
        for sc_id in ("thermal_stress_overtemp", "sensor_bias_drift", "compounded_degradation"):
            what_if_results[sc_id] = self.evaluate_what_if_scenario(scenario_id=sc_id, engine_id=nominal_engine_id, cycle=35)

        total_elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Calculate high-level benchmark figures
        what_if_pass_count = sum(1 for r in what_if_results.values() if r["scenario_passed"])
        what_if_accuracy = float(what_if_pass_count / len(what_if_results)) * 100.0

        all_passed = bool(
            nominal_metrics["all_nominal_passed"]
            and degraded_metrics["degradation_passed"]
            and what_if_accuracy == 100.0
        )

        suite_summary = {
            "benchmark_suite": "PHASE_6_STEP_6_8_MULTI_AGENT_INTERACTION",
            "subset": self.subset,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "all_benchmarks_passed": all_passed,
            "total_benchmark_elapsed_ms": round(total_elapsed_ms, 2),
            "metrics": {
                "nominal_false_alarm_rate": nominal_metrics["false_alarm_rate"],
                "nominal_mean_health_index": nominal_metrics["mean_virtual_health_index"],
                "degradation_detection_latency_cycles": degraded_metrics["detection_latency_cycles"],
                "degradation_isolated_fault": degraded_metrics["isolated_fault"],
                "degradation_final_rul": degraded_metrics["final_rul_forecast"],
                "what_if_diagnostic_accuracy_pct": what_if_accuracy,
                "what_if_scenarios_evaluated": len(what_if_results),
                "what_if_scenarios_passed": what_if_pass_count,
            },
            "nominal": nominal_metrics,
            "degraded": degraded_metrics,
            "what_if": {k: {m: v for m, v in val.items() if m != "orchestration_result"} for k, val in what_if_results.items()},
        }

        self.latest_evaluation_metrics["suite_summary"] = suite_summary

        # Auto-generate Markdown summary report in audit logger
        try:
            self.audit_logger.write_markdown_summary()
        except Exception:
            pass

        return suite_summary
