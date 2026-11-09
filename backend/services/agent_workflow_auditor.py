"""
Multi-Agent Workflow and Pipeline Auditor
Roadmap Reference: Phase 9 - Step 9.7 ("Test each agent independently and then test the full orchestrated workflow")
Strict Academic Integrity Standard: Master Roadmap Section 1, Page 2
("Display anomaly flags and diagnosis evidence separately from confirmed fault claims")

Provides independent agent execution harnesses and end-to-end multi-agent orchestration auditing:
1. Isolated verification of MonitoringAgent, DiagnosisAgent, and PrognosticsAgent.
2. Context-preserving sequential pipeline verification (Monitoring -> Diagnosis -> Prognostics).
3. Stage-by-stage execution latency profiling and performance envelope verification.
4. Cross-subset (FD001-FD004) and split (train/test) workflow fidelity.
"""

import math
import time
from typing import Dict, Any, List, Optional, Union

from agents.schemas import (
    AgentExecutionStatus,
    OperationalState,
    SeverityLevel,
    MonitoringInput,
    MonitoringOutput,
    DiagnosisInput,
    DiagnosisOutput,
    PrognosticsInput,
    PrognosticsOutput,
    OrchestrationResult,
)
from agents.monitoring_agent import MonitoringAgent
from agents.diagnosis_agent import DiagnosisAgent
from agents.prognostics_agent import PrognosticsAgent
from agents.orchestrator import AgentOrchestrator
from digital_twin.twin_simulator import DigitalTwinSimulator
from safety.disclaimers import ACADEMIC_SIMULATION_DISCLAIMER


def audit_independent_agents(
    subset: str = "FD001",
    unit: int = 1,
    cycle: int = 30,
    split: str = "train",
) -> Dict[str, Any]:
    """
    Executes and audits MonitoringAgent, DiagnosisAgent, and PrognosticsAgent
    in strict isolation, evaluating input validation, telemetry processing,
    and output schema validity.

    Parameters
    ----------
    subset : str
        C-MAPSS subset identifier ("FD001" to "FD004")
    unit : int
        Engine serial identifier
    cycle : int
        Operating flight cycle
    split : str
        Data partition ("train" or "test")

    Returns
    -------
    Dict[str, Any]
        Structured audit report for independent agent evaluations.
    """
    subset_upper = subset.upper()
    sim = DigitalTwinSimulator(subset=subset_upper, enable_ml_inference=True)
    sim.load_engine(unit_number=unit, split=split.lower())
    target_c = min(cycle, sim.total_cycles)
    while sim.current_state and sim.current_state.time_cycles < target_c:
        if sim.step() is None:
            break
    state = sim.current_state
    if state is None:
        raise RuntimeError(f"Failed to advance simulator to cycle {cycle} for unit {unit}")

    # 1. Independent Monitoring Agent
    t0_mon = time.perf_counter()
    mon_agent = MonitoringAgent(subset=subset_upper)
    mon_out = mon_agent.process_engine_state(state)
    mon_time_ms = (time.perf_counter() - t0_mon) * 1000.0

    # 2. Independent Diagnosis Agent
    t0_diag = time.perf_counter()
    diag_agent = DiagnosisAgent(subset=subset_upper)
    telemetry_snap: Dict[str, float] = {}
    sensor_fields = [
        "T2", "T24", "T30", "T50", "P2", "P15", "P30", "Nf", "Nc", "epr",
        "Ps30", "phi", "NRf", "NRc", "BPR", "farB", "htBleed", "Nf_dmd",
        "PCNfR_dmd", "W31", "W32", "hpc_temp_rise", "overall_pressure_ratio",
        "spool_speed_ratio",
    ]
    for sf in sensor_fields:
        val = getattr(state, sf, None)
        if val is not None and isinstance(val, (int, float)) and math.isfinite(val):
            telemetry_snap[sf] = float(val)

    diag_out = diag_agent.diagnose(mon_out, telemetry_snapshot=telemetry_snap)
    diag_time_ms = (time.perf_counter() - t0_diag) * 1000.0

    # 3. Independent Prognostics Agent
    t0_prog = time.perf_counter()
    prog_agent = PrognosticsAgent(subset=subset_upper)
    prog_out = prog_agent.process_engine_state(state)
    prog_time_ms = (time.perf_counter() - t0_prog) * 1000.0

    all_successful = (
        mon_out.metadata.status == AgentExecutionStatus.SUCCESS
        and diag_out.metadata.status == AgentExecutionStatus.SUCCESS
        and prog_out.metadata.status == AgentExecutionStatus.SUCCESS
    )

    return {
        "subset": subset_upper,
        "unit_number": unit,
        "cycle": target_c,
        "split": split.lower(),
        "all_successful": all_successful,
        "monitoring": {
            "agent_id": mon_agent.agent_id,
            "status": mon_out.metadata.status.value,
            "operational_state": mon_out.operational_state.value,
            "breaches_count": len(mon_out.threshold_breaches),
            "breached_sensors": [b.sensor_name for b in mon_out.threshold_breaches],
            "persistence_count": mon_out.persistence_count,
            "persistence_alert_triggered": mon_out.persistence_alert,
            "execution_time_ms": mon_time_ms,
        },
        "diagnosis": {
            "agent_id": diag_agent.agent_id,
            "status": diag_out.metadata.status.value,
            "primary_fault": diag_out.primary_fault,
            "confidence_score": diag_out.confidence_score,
            "evidence_count": len(diag_out.contributing_evidence),
            "candidate_count": len(diag_out.fault_candidates),
            "requires_maintenance": diag_out.requires_maintenance_action,
            "disclaimer": ACADEMIC_SIMULATION_DISCLAIMER,
            "execution_time_ms": diag_time_ms,
        },
        "prognostics": {
            "agent_id": prog_agent.agent_id,
            "status": prog_out.metadata.status.value,
            "predicted_rul_xgboost": prog_out.predicted_rul_xgboost,
            "predicted_rul_lstm": prog_out.predicted_rul_lstm,
            "ensemble_rul": prog_out.ensemble_rul_estimate,
            "prognostic_spread": prog_out.prognostic_spread,
            "uncertainty_classification": prog_out.uncertainty_classification,
            "virtual_health_index": prog_out.virtual_health_index,
            "execution_time_ms": prog_time_ms,
        },
        "disclaimer": ACADEMIC_SIMULATION_DISCLAIMER,
    }


def audit_orchestrated_pipeline(
    subset: str = "FD001",
    unit: int = 1,
    cycle: int = 30,
    split: str = "train",
    mode: str = "full",
) -> Dict[str, Any]:
    """
    Executes and audits the complete multi-agent pipeline through AgentOrchestrator,
    validating sequential context passing, latency profiles, and data flow integrity.

    Parameters
    ----------
    subset : str
        C-MAPSS subset identifier
    unit : int
        Engine serial identifier
    cycle : int
        Operating flight cycle
    split : str
        Data partition ("train" or "test")
    mode : str
        Orchestrator execution mode ("full", "monitoring_only", "prognostics_only", "dry_run")

    Returns
    -------
    Dict[str, Any]
        Structured audit report for orchestrated pipeline execution.
    """
    subset_upper = subset.upper()
    sim = DigitalTwinSimulator(subset=subset_upper, enable_ml_inference=True)
    sim.load_engine(unit_number=unit, split=split.lower())
    target_c = min(cycle, sim.total_cycles)
    while sim.current_state and sim.current_state.time_cycles < target_c:
        if sim.step() is None:
            break
    state = sim.current_state
    if state is None:
        raise RuntimeError(f"Failed to advance simulator to cycle {cycle} for unit {unit}")

    orchestrator = AgentOrchestrator(subset=subset_upper, default_mode=mode)
    t_start = time.perf_counter()
    orch_res: OrchestrationResult = orchestrator.orchestrate_state(state, mode=mode)
    total_measured_wall_ms = (time.perf_counter() - t_start) * 1000.0

    # Data flow integrity validation
    data_flow_intact = True
    context_matches = (
        orch_res.engine_id == unit
        and orch_res.cycle == target_c
        and orch_res.subset == subset_upper
    )
    if not context_matches:
        data_flow_intact = False

    if mode == "full":
        if orch_res.monitoring_result is None:
            data_flow_intact = False
        if orch_res.diagnosis_result is None:
            data_flow_intact = False
        if orch_res.prognostics_result is None:
            data_flow_intact = False

    timeline = dict(orch_res.execution_timeline_ms)
    mon_ms = timeline.get("monitoring_ms", 0.0)
    diag_ms = timeline.get("diagnosis_ms", 0.0)
    prog_ms = timeline.get("prognostics_ms", 0.0)
    pipeline_ms = timeline.get("total_elapsed_ms", 0.0)

    # Performance envelope assertions (sub-15ms per agent, sub-50ms pipeline)
    latencies_compliant = (
        mon_ms < 15.0
        and diag_ms < 15.0
        and prog_ms < 30.0
        and pipeline_ms < 50.0
    )

    return {
        "subset": subset_upper,
        "unit_number": unit,
        "cycle": target_c,
        "split": split.lower(),
        "mode": mode,
        "run_id": orch_res.run_id,
        "status": orch_res.status.value,
        "data_flow_intact": data_flow_intact,
        "latencies_compliant": latencies_compliant,
        "execution_timeline_ms": timeline,
        "wall_clock_time_ms": total_measured_wall_ms,
        "monitoring": {
            "operational_state": orch_res.monitoring_result.operational_state.value if orch_res.monitoring_result else None,
            "breaches_count": len(orch_res.monitoring_result.threshold_breaches) if orch_res.monitoring_result else 0,
            "breached_sensors": [b.sensor_name for b in orch_res.monitoring_result.threshold_breaches] if orch_res.monitoring_result else [],
        },
        "diagnosis": {
            "primary_fault": orch_res.diagnosis_result.primary_fault if orch_res.diagnosis_result else None,
            "confidence_score": orch_res.diagnosis_result.confidence_score if orch_res.diagnosis_result else 0.0,
            "evidence_count": len(orch_res.diagnosis_result.contributing_evidence) if orch_res.diagnosis_result else 0,
            "requires_maintenance": orch_res.diagnosis_result.requires_maintenance_action if orch_res.diagnosis_result else False,
        },
        "prognostics": {
            "predicted_rul_xgboost": orch_res.prognostics_result.predicted_rul_xgboost if orch_res.prognostics_result else None,
            "predicted_rul_lstm": orch_res.prognostics_result.predicted_rul_lstm if orch_res.prognostics_result else None,
            "ensemble_rul": orch_res.prognostics_result.ensemble_rul_estimate if orch_res.prognostics_result else None,
            "prognostic_spread": orch_res.prognostics_result.prognostic_spread if orch_res.prognostics_result else None,
            "virtual_health_index": orch_res.prognostics_result.virtual_health_index if orch_res.prognostics_result else None,
        },
        "disclaimer": ACADEMIC_SIMULATION_DISCLAIMER,
    }
