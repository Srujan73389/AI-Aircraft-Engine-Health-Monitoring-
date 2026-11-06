"""
Multi-Agent Orchestrator Module - Phase 6 Step 6.5
Coordinates and sequences autonomous agents (Monitoring, Diagnosis, Prognostics)
in the AI-Based Aircraft Engine Digital Twin multi-agent platform.

Roadmap Reference: Phase 6 - Step 6.5:
"Implement the Orchestrator to invoke agents in the required sequence"

Capabilities:
1. Sequenced Multi-Agent Execution Pipeline:
   - Ingests OrchestrationRequest or Digital Twin EngineState.
   - Stage 1: MonitoringAgent evaluates telemetry against physical thresholds and statistical anomaly models.
   - Stage 2: DiagnosisAgent synthesizes monitoring evidence and isolates root-cause fault modes.
   - Stage 3: PrognosticsAgent executes champion ML inference (XGBoost & LSTM) for RUL and health index forecast.
   - Aggregates results into a type-safe OrchestrationResult.
2. Configurable Execution Modes:
   - "full": Complete end-to-end execution of all three agents.
   - "monitoring_only": Executes only aerothermal surveillance and persistence tracking.
   - "prognostics_only": Executes only sequential ML RUL forecasting and health index estimation.
   - "dry_run": Validates requests and returns skipped/empty results with pipeline timing.
3. Execution Timeline & Latency Profiling:
   - Measures wall-clock millisecond latency per agent and total elapsed pipeline runtime.
4. Robust Fault Containment:
   - Catches downstream agent exceptions, records structured error diagnostics, and sets
     status=FAILED or WARNING without crashing the orchestration harness.
5. Trajectory State Management:
   - Exposes reset_engine(unit_number) to cleanly manage multi-cycle persistence state transitions.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
import os
import sys
import time
import uuid
from typing import Dict, Any, List, Optional, Tuple, Union

from agents.schemas import (
    AgentExecutionStatus,
    SeverityLevel,
    OperationalState,
    AgentMetadata,
    OrchestrationRequest,
    OrchestrationResult,
    MonitoringInput,
    MonitoringOutput,
    DiagnosisInput,
    DiagnosisOutput,
    PrognosticsInput,
    PrognosticsOutput,
)
from agents.monitoring_agent import MonitoringAgent
from agents.diagnosis_agent import DiagnosisAgent
from agents.prognostics_agent import PrognosticsAgent
from agents.audit_logger import AgentAuditLogger
from agents.twin_bridge import DigitalTwinSafetyBridge, SafetyViolationError

# Optional Digital Twin simulator import
try:
    from digital_twin.twin_simulator import DigitalTwinSimulator
    from digital_twin.engine_state import EngineState
except ImportError:
    DigitalTwinSimulator = None  # type: ignore
    EngineState = None  # type: ignore


class AgentOrchestrator:
    """
    Master Multi-Agent Orchestrator.
    Manages agent lifecycle, executes sequenced diagnostic-prognostic pipelines,
    profiles stage latencies, and guarantees fault-contained operational results.
    """

    VALID_MODES = ("full", "monitoring_only", "prognostics_only", "dry_run")

    def __init__(
        self,
        monitoring_agent: Optional[MonitoringAgent] = None,
        diagnosis_agent: Optional[DiagnosisAgent] = None,
        prognostics_agent: Optional[PrognosticsAgent] = None,
        audit_logger: Optional[AgentAuditLogger] = None,
        enable_twin_feedback: bool = False,
        twin_bridge: Optional[DigitalTwinSafetyBridge] = None,
        subset: str = "FD001",
        default_mode: str = "full",
        agent_id: str = "orchestrator_v1",
    ):
        """
        Initializes the Multi-Agent Orchestrator.

        Args:
            monitoring_agent: Optional pre-configured MonitoringAgent instance.
            diagnosis_agent: Optional pre-configured DiagnosisAgent instance.
            prognostics_agent: Optional pre-configured PrognosticsAgent instance.
            audit_logger: Optional pre-configured AgentAuditLogger instance for structured logging.
            enable_twin_feedback: If True, enriches the input EngineState with validated agent outputs.
            twin_bridge: Optional pre-configured DigitalTwinSafetyBridge instance.
            subset: Default C-MAPSS subset identifier ("FD001", "FD002", etc.).
            default_mode: Default execution mode ("full", "monitoring_only", "prognostics_only", "dry_run").
            agent_id: Provenance identifier for metadata tracking.
        """
        self.agent_id = agent_id
        self.subset = subset.upper()
        if default_mode not in self.VALID_MODES:
            raise ValueError(f"Invalid default_mode '{default_mode}'. Must be one of {self.VALID_MODES}")
        self.default_mode = default_mode
        self.audit_logger = audit_logger
        self.enable_twin_feedback = enable_twin_feedback
        self.twin_bridge = twin_bridge or (DigitalTwinSafetyBridge() if enable_twin_feedback else None)

        # Initialize agents
        self.monitoring_agent = monitoring_agent or MonitoringAgent(
            agent_id="monitoring_agent_sub", subset=self.subset
        )
        self.diagnosis_agent = diagnosis_agent or DiagnosisAgent(
            agent_id="diagnosis_agent_sub", subset=self.subset
        )
        self.prognostics_agent = prognostics_agent or PrognosticsAgent(
            agent_id="prognostics_agent_sub", subset=self.subset
        )

    def reset_engine(self, unit_number: Optional[int] = None) -> None:
        """
        Resets multi-cycle tracking and persistence counters across agents.

        Args:
            unit_number: Engine serial identifier. If None, resets state across all engines.
        """
        self.monitoring_agent.reset_trajectory(unit_number)

    def _extract_telemetry_snapshot(self, state: Any) -> Dict[str, float]:
        """Extracts available gas-path sensors and derived indicators from EngineState."""
        telemetry: Dict[str, float] = {}
        sensor_keys = [
            "T2", "T24", "T30", "T50", "P2", "P15", "P30", "Nf", "Nc", "epr",
            "Ps30", "phi", "NRf", "NRc", "BPR", "farB", "htBleed", "Nf_dmd",
            "PCNfR_dmd", "W31", "W32", "hpc_temp_rise", "overall_pressure_ratio",
            "spool_speed_ratio",
        ]
        for k in sensor_keys:
            val = getattr(state, k, None)
            if val is not None and isinstance(val, (int, float)) and math.isfinite(val):
                telemetry[k] = float(val)
        return telemetry

    def orchestrate_state(
        self,
        state: Any,
        mode: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> OrchestrationResult:
        """
        Executes multi-agent orchestration directly on an EngineState instance.

        Args:
            state: EngineState instance from DigitalTwinSimulator.
            mode: Optional execution mode override ("full", "monitoring_only", "prognostics_only", "dry_run").
            run_id: Optional tracking identifier.

        Returns:
            Type-safe OrchestrationResult.
        """
        req_mode = mode or self.default_mode
        if req_mode not in self.VALID_MODES:
            raise ValueError(f"Invalid execution mode '{req_mode}'. Must be one of {self.VALID_MODES}")

        # Validate state attributes
        required_attrs = ("unit_number", "time_cycles")
        for attr in required_attrs:
            if not hasattr(state, attr):
                raise TypeError(f"Provided state lacks required attribute '{attr}'")

        unit_number = int(state.unit_number)
        time_cycles = int(state.time_cycles)
        subset = getattr(state, "subset", self.subset).upper()

        if unit_number <= 0:
            raise ValueError(f"AgentOrchestrator: unit_number must be > 0 ({unit_number}).")
        if time_cycles <= 0:
            raise ValueError(f"AgentOrchestrator: time_cycles must be > 0 ({time_cycles}).")

        rid = run_id or f"run_{unit_number}_{time_cycles}_{uuid.uuid4().hex[:8]}"
        pipeline_start = time.perf_counter()
        timestamp = datetime.now(timezone.utc).isoformat()
        timeline_ms: Dict[str, float] = {}

        mon_result: Optional[MonitoringOutput] = None
        diag_result: Optional[DiagnosisOutput] = None
        prog_result: Optional[PrognosticsOutput] = None
        pipeline_status = AgentExecutionStatus.SUCCESS
        error_msgs: List[str] = []

        # ---------------------------------------------------------------------
        # Execution Mode: DRY RUN
        # ---------------------------------------------------------------------
        if req_mode == "dry_run":
            total_elapsed = (time.perf_counter() - pipeline_start) * 1000.0
            timeline_ms["total_elapsed_ms"] = total_elapsed
            meta = AgentMetadata(
                agent_id=self.agent_id,
                execution_timestamp=timestamp,
                execution_time_ms=total_elapsed,
                status=AgentExecutionStatus.SUCCESS,
                version="1.0.0",
            )
            out = OrchestrationResult(
                run_id=rid,
                engine_id=unit_number,
                cycle=time_cycles,
                subset=subset,
                status=AgentExecutionStatus.SUCCESS,
                monitoring_result=None,
                diagnosis_result=None,
                prognostics_result=None,
                execution_timeline_ms=timeline_ms,
                metadata=meta,
            )
            out.validate()
            if self.audit_logger is not None:
                self.audit_logger.log_orchestration(
                    run_id=rid,
                    engine_id=unit_number,
                    cycle=time_cycles,
                    orchestration_result=out,
                )
            return out

        # ---------------------------------------------------------------------
        # Stage 1: Monitoring Agent Execution
        # ---------------------------------------------------------------------
        if req_mode in ("full", "monitoring_only"):
            t0 = time.perf_counter()
            try:
                mon_result = self.monitoring_agent.process_engine_state(state)
                timeline_ms["monitoring_ms"] = (time.perf_counter() - t0) * 1000.0
                if self.audit_logger is not None and mon_result is not None:
                    self.audit_logger.log_monitoring(
                        run_id=rid,
                        engine_id=unit_number,
                        cycle=time_cycles,
                        monitoring_output=mon_result,
                    )
            except Exception as e:
                timeline_ms["monitoring_ms"] = (time.perf_counter() - t0) * 1000.0
                error_msgs.append(f"MonitoringAgent failed: {str(e)}")
                pipeline_status = AgentExecutionStatus.FAILED

        # ---------------------------------------------------------------------
        # Stage 2: Diagnosis Agent Execution
        # ---------------------------------------------------------------------
        if req_mode == "full" and mon_result is not None:
            t0 = time.perf_counter()
            try:
                telemetry = self._extract_telemetry_snapshot(state)
                diag_result = self.diagnosis_agent.diagnose(mon_result, telemetry_snapshot=telemetry)
                timeline_ms["diagnosis_ms"] = (time.perf_counter() - t0) * 1000.0
                if self.audit_logger is not None and diag_result is not None:
                    self.audit_logger.log_diagnosis(
                        run_id=rid,
                        engine_id=unit_number,
                        cycle=time_cycles,
                        diagnosis_output=diag_result,
                    )
            except Exception as e:
                timeline_ms["diagnosis_ms"] = (time.perf_counter() - t0) * 1000.0
                error_msgs.append(f"DiagnosisAgent failed: {str(e)}")
                pipeline_status = AgentExecutionStatus.WARNING

        # ---------------------------------------------------------------------
        # Stage 3: Prognostics Agent Execution
        # ---------------------------------------------------------------------
        if req_mode in ("full", "prognostics_only"):
            t0 = time.perf_counter()
            try:
                prog_result = self.prognostics_agent.process_engine_state(state)
                timeline_ms["prognostics_ms"] = (time.perf_counter() - t0) * 1000.0
                if self.audit_logger is not None and prog_result is not None:
                    self.audit_logger.log_prognostics(
                        run_id=rid,
                        engine_id=unit_number,
                        cycle=time_cycles,
                        prognostics_output=prog_result,
                    )
            except Exception as e:
                timeline_ms["prognostics_ms"] = (time.perf_counter() - t0) * 1000.0
                error_msgs.append(f"PrognosticsAgent failed: {str(e)}")
                if pipeline_status != AgentExecutionStatus.FAILED:
                    pipeline_status = AgentExecutionStatus.WARNING

        # ---------------------------------------------------------------------
        # Stage 4: Aggregation & Provenance Assembly
        # ---------------------------------------------------------------------
        total_elapsed = (time.perf_counter() - pipeline_start) * 1000.0
        timeline_ms["total_elapsed_ms"] = total_elapsed

        err_str = "; ".join(error_msgs) if error_msgs else None
        meta = AgentMetadata(
            agent_id=self.agent_id,
            execution_timestamp=timestamp,
            execution_time_ms=total_elapsed,
            status=pipeline_status,
            error_message=err_str,
            version="1.0.0",
        )

        result = OrchestrationResult(
            run_id=rid,
            engine_id=unit_number,
            cycle=time_cycles,
            subset=subset,
            status=pipeline_status,
            monitoring_result=mon_result,
            diagnosis_result=diag_result,
            prognostics_result=prog_result,
            execution_timeline_ms=timeline_ms,
            metadata=meta,
        )

        result.validate()

        # ---------------------------------------------------------------------
        # Optional Stage 5: Digital Twin Safety Bridge Feedback
        # ---------------------------------------------------------------------
        if self.enable_twin_feedback and self.twin_bridge is not None and isinstance(state, EngineState):
            t_bridge = time.perf_counter()
            try:
                enriched = self.twin_bridge.enrich_state_with_agent_outputs(state, result)
                result.enriched_state = enriched
                result.execution_timeline_ms["twin_bridge_ms"] = (time.perf_counter() - t_bridge) * 1000.0
            except SafetyViolationError as sve:
                result.execution_timeline_ms["twin_bridge_ms"] = (time.perf_counter() - t_bridge) * 1000.0
                result.status = AgentExecutionStatus.WARNING
                if result.metadata:
                    curr_err = result.metadata.error_message
                    result.metadata.error_message = f"{curr_err}; SafetyBridge: {str(sve)}" if curr_err else f"SafetyBridge: {str(sve)}"

        if self.audit_logger is not None:
            self.audit_logger.log_orchestration(
                run_id=rid,
                engine_id=unit_number,
                cycle=time_cycles,
                orchestration_result=result,
            )
        return result

    def run(
        self,
        request: Union[OrchestrationRequest, Dict[str, Any]],
        state: Optional[Any] = None,
    ) -> OrchestrationResult:
        """
        Executes multi-agent orchestration based on an OrchestrationRequest.
        If an EngineState is not provided, the orchestrator advances a DigitalTwinSimulator instance.

        Args:
            request: Strongly-typed OrchestrationRequest or serialized dict.
            state: Optional EngineState instance.

        Returns:
            Type-safe OrchestrationResult.
        """
        if isinstance(request, dict):
            req = OrchestrationRequest.from_dict(request)
        elif isinstance(request, OrchestrationRequest):
            req = request
        else:
            raise TypeError(f"Expected OrchestrationRequest or dict, got {type(request)}")

        req.validate()

        # If state is provided directly, dispatch to orchestrate_state
        if state is not None:
            return self.orchestrate_state(state=state, mode=req.execution_mode)

        # Otherwise, instantiate DigitalTwinSimulator to step to target cycle
        if DigitalTwinSimulator is None:
            raise RuntimeError("DigitalTwinSimulator is not available to load engine trajectory.")

        sim = DigitalTwinSimulator(subset=req.subset)
        sim.load_engine(unit_number=req.engine_id, split="train")

        # Step forward until target cycle
        target_cycle = min(req.flight_cycle, sim.total_cycles)
        for _ in range(target_cycle - 1):
            sim.step()

        target_state = sim.current_state
        return self.orchestrate_state(state=target_state, mode=req.execution_mode)

    def execute(
        self,
        request: Union[OrchestrationRequest, Dict[str, Any]],
        state: Optional[Any] = None,
    ) -> OrchestrationResult:
        """Alias for run()."""
        return self.run(request, state=state)
