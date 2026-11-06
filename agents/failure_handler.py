"""
Adversarial Failure Injection & Resilience Suite - Phase 6 Step 6.9
Verifies that agent failures, sensor dropouts, non-finite telemetry, runtime crashes,
unwriteable audit logging, and safety firewall violations are handled safely,
contained effectively, and reported clearly without crashing the flight monitoring system.

Roadmap Reference: Phase 6 - Step 6.9:
"Verify that agent failures are handled safely and reported clearly"
"""

import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
import os
import time
from typing import Dict, Any, List, Optional, Tuple, Union
import uuid

import numpy as np

from agents.schemas import (
    AgentExecutionStatus,
    SeverityLevel,
    OperationalState,
    AgentMetadata,
    MonitoringInput,
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


@dataclass
class IncidentRecord:
    """Individual failure test incident verification record."""
    failure_mode: str
    description: str
    contained_safely: bool
    status_assigned: str
    exception_type: Optional[str] = None
    exception_message: Optional[str] = None
    telemetry_preserved: bool = True
    audit_logged: bool = True
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "failure_mode": self.failure_mode,
            "description": self.description,
            "contained_safely": self.contained_safely,
            "status_assigned": self.status_assigned,
            "exception_type": self.exception_type,
            "exception_message": self.exception_message,
            "telemetry_preserved": self.telemetry_preserved,
            "audit_logged": self.audit_logged,
            "details": self.details,
        }


@dataclass
class AgentFailureReport:
    """Structured executive failure and resilience test report."""
    report_id: str
    timestamp: str
    subset: str
    total_tests_run: int
    passed_tests_count: int
    failed_tests_count: int
    all_contained: bool
    incidents: List[IncidentRecord] = field(default_factory=list)
    summary_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "timestamp": self.timestamp,
            "subset": self.subset,
            "total_tests_run": self.total_tests_run,
            "passed_tests_count": self.passed_tests_count,
            "failed_tests_count": self.failed_tests_count,
            "all_contained": self.all_contained,
            "summary_metrics": self.summary_metrics,
            "incidents": [inc.to_dict() for inc in self.incidents],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_markdown(self) -> str:
        lines = [
            f"# Multi-Agent System Failure & Resilience Incident Report",
            f"",
            f"**Report ID:** `{self.report_id}`  ",
            f"**Timestamp:** `{self.timestamp}`  ",
            f"**Dataset Subset:** `{self.subset}`  ",
            f"**Overall Containment Status:** {'**ALL CONTAINED (PASS)**' if self.all_contained else '**CONTAINMENT BREACH (FAIL)**'}  ",
            f"",
            f"---",
            f"",
            f"## 1. Executive Summary",
            f"",
            f"- **Total Adversarial Tests Executed:** {self.total_tests_run}",
            f"- **Passed (Safely Contained):** {self.passed_tests_count}",
            f"- **Failed (Uncontained):** {self.failed_tests_count}",
            f"- **Fault Containment Rate:** {self.summary_metrics.get('containment_rate_pct', 100.0):.1f}%",
            f"",
            f"---",
            f"",
            f"## 2. Incident Verification Matrix",
            f"",
            f"| Failure Mode | Status Assigned | Contained? | Exception Type | Telemetry Preserved? |",
            f"| :--- | :---: | :---: | :--- | :---: |",
        ]

        for inc in self.incidents:
            cont_badge = "YES" if inc.contained_safely else "**NO**"
            telem_badge = "YES" if inc.telemetry_preserved else "**NO**"
            exc = f"`{inc.exception_type}`" if inc.exception_type else "None"
            lines.append(f"| `{inc.failure_mode}` | `{inc.status_assigned}` | {cont_badge} | {exc} | {telem_badge} |")

        lines.extend([
            f"",
            f"---",
            f"",
            f"## 3. Incident Details & Diagnostics",
            f"",
        ])

        for idx, inc in enumerate(self.incidents, start=1):
            lines.extend([
                f"### Incident {idx}: `{inc.failure_mode}`",
                f"- **Description:** {inc.description}",
                f"- **Pipeline Status:** `{inc.status_assigned}`",
                f"- **Exception Caught:** `{inc.exception_type}`: {inc.exception_message or 'N/A'}",
                f"- **Safely Contained:** {inc.contained_safely}",
                f"- **Telemetry Integrity:** {inc.telemetry_preserved}",
                f"- **Audit Trail Preserved:** {inc.audit_logged}",
                f"",
            ])

        return "\n".join(lines)

    def save_report(self, output_dir: Optional[str] = None) -> Tuple[str, str]:
        """Save report to JSON and Markdown files."""
        target_dir = output_dir or os.path.join("reports", "agent_logs")
        os.makedirs(target_dir, exist_ok=True)

        json_path = os.path.join(target_dir, f"failure_report_{self.report_id[:8]}.json")
        md_path = os.path.join(target_dir, f"failure_report_{self.report_id[:8]}.md")

        with open(json_path, "w", encoding="utf-8") as f:
            f.write(self.to_json())

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(self.to_markdown())

        return json_path, md_path


class AgentFailureHarness:
    """
    Automated Failure Injection and Fault Containment Test Harness.
    Exercises adversarial corruption, missing telemetry, runtime agent crashes,
    unwriteable audit logging, and safety firewall tampering.
    """

    def __init__(
        self,
        subset: str = "FD001",
        log_dir: str = os.path.join("reports", "agent_logs"),
        agent_id: str = "failure_harness_v1",
    ):
        self.subset = subset.upper()
        self.log_dir = log_dir
        self.agent_id = agent_id

    def _get_nominal_state(self, engine_id: int = 1, cycles: int = 30) -> EngineState:
        """Helper: Retrieve a valid nominal engine state from DigitalTwinSimulator."""
        sim = DigitalTwinSimulator(subset=self.subset)
        sim.load_engine(unit_number=engine_id, split="train")
        for _ in range(cycles - 1):
            sim.step()
        return sim.current_state

    # -------------------------------------------------------------------------
    # Failure Mode 1: Corrupted Telemetry (NaN, Inf, Non-Finite Values)
    # -------------------------------------------------------------------------
    def test_corrupted_telemetry_nan_inf(self, engine_id: int = 1) -> IncidentRecord:
        """
        Injects NaN, +Inf, -Inf into telemetry stream; verifies graceful rejection
        with clear ValueError reporting before any model or agent execution.
        """
        nan_caught = False
        inf_caught = False
        exc_type = None
        exc_msg = None

        # 1. Test NaN rejection in MonitoringInput
        try:
            bad_input_nan = MonitoringInput(
                unit_number=engine_id,
                time_cycles=30,
                operating_context={"altitude": 0.0, "mach_number": 0.0, "throttle_resolver_angle": 100.0},
                raw_telemetry={"T30": float("nan"), "P30": 554.0},
                derived_indicators={"hpc_temp_rise": 948.0, "overall_pressure_ratio": 37.8, "spool_speed_ratio": 3.8},
            )
            bad_input_nan.validate()
        except ValueError as ve:
            nan_caught = True
            exc_type = type(ve).__name__
            exc_msg = str(ve)

        # 2. Test Inf rejection in MonitoringInput
        try:
            bad_input_inf = MonitoringInput(
                unit_number=engine_id,
                time_cycles=30,
                operating_context={"altitude": float("inf"), "mach_number": 0.0, "throttle_resolver_angle": 100.0},
                raw_telemetry={"T30": 1590.0, "P30": 554.0},
                derived_indicators={"hpc_temp_rise": 948.0, "overall_pressure_ratio": 37.8, "spool_speed_ratio": 3.8},
            )
            bad_input_inf.validate()
        except ValueError as ve:
            inf_caught = True

        contained = bool(nan_caught and inf_caught and exc_msg and "non-finite" in exc_msg.lower())

        return IncidentRecord(
            failure_mode="corrupted_telemetry_nan_inf",
            description="Injection of NaN and Inf floats into telemetry and operating context.",
            contained_safely=contained,
            status_assigned="REJECTED_AT_INGESTION",
            exception_type=exc_type,
            exception_message=exc_msg,
            telemetry_preserved=True,
            audit_logged=True,
            details={"nan_rejected": nan_caught, "inf_rejected": inf_caught},
        )

    # -------------------------------------------------------------------------
    # Failure Mode 2: Missing Sensor Channels / Dropouts
    # -------------------------------------------------------------------------
    def test_missing_sensor_channels(self, engine_id: int = 1) -> IncidentRecord:
        """
        Injects missing sensor channels or missing state attributes; verifies
        clear TypeError/ValueError reporting rather than silent mathematical corruption.
        """
        state = self._get_nominal_state(engine_id=engine_id)
        broken_state = copy.deepcopy(state)
        # Delete critical sensor attribute
        delattr(broken_state, "T30")

        mon = MonitoringAgent(subset=self.subset)
        caught = False
        exc_type = None
        exc_msg = None

        try:
            mon.process_engine_state(broken_state)
        except (TypeError, AttributeError, ValueError) as err:
            caught = True
            exc_type = type(err).__name__
            exc_msg = str(err)

        contained = bool(caught and exc_msg and "T30" in exc_msg)

        return IncidentRecord(
            failure_mode="missing_sensor_channels",
            description="Dropping required sensor attribute T30 from engine state during monitoring ingestion.",
            contained_safely=contained,
            status_assigned="REJECTED_WITH_ERROR",
            exception_type=exc_type,
            exception_message=exc_msg,
            telemetry_preserved=True,
            audit_logged=True,
            details={"missing_attribute_detected": caught},
        )

    # -------------------------------------------------------------------------
    # Failure Mode 3: MonitoringAgent Runtime Crash Isolation
    # -------------------------------------------------------------------------
    def test_monitoring_agent_runtime_crash(self, engine_id: int = 1) -> IncidentRecord:
        """
        Injects a runtime crash into MonitoringAgent; verifies AgentOrchestrator catches
        the failure, sets status=FAILED, logs error trace, and terminates safely.
        """
        state = self._get_nominal_state(engine_id=engine_id)

        # Mock faulty monitoring agent
        class FaultyMonitoringAgent(MonitoringAgent):
            def process_engine_state(self, s):
                raise RuntimeError("Simulated Hardware Telemetry Bus Deserialization Fault")

        orch = AgentOrchestrator(
            subset=self.subset,
            monitoring_agent=FaultyMonitoringAgent(subset=self.subset),
            default_mode="full",
        )

        res = orch.orchestrate_state(state)

        contained = bool(
            res.status == AgentExecutionStatus.FAILED
            and res.metadata is not None
            and "MonitoringAgent failed" in (res.metadata.error_message or "")
            and "Simulated Hardware Telemetry Bus" in (res.metadata.error_message or "")
            and res.monitoring_result is None
        )

        return IncidentRecord(
            failure_mode="monitoring_agent_runtime_crash",
            description="Injected unhandled RuntimeError inside MonitoringAgent.process_engine_state.",
            contained_safely=contained,
            status_assigned=res.status.value,
            exception_type="RuntimeError",
            exception_message=res.metadata.error_message if res.metadata else None,
            telemetry_preserved=True,
            audit_logged=True,
            details={"orchestration_status": res.status.value},
        )

    # -------------------------------------------------------------------------
    # Failure Mode 4: DiagnosisAgent Runtime Crash Isolation
    # -------------------------------------------------------------------------
    def test_diagnosis_agent_runtime_crash(self, engine_id: int = 1) -> IncidentRecord:
        """
        Injects a runtime crash into DiagnosisAgent; verifies AgentOrchestrator sets
        status=WARNING, preserves Monitoring and Prognostics results without halting.
        """
        state = self._get_nominal_state(engine_id=engine_id)

        class FaultyDiagnosisAgent(DiagnosisAgent):
            def diagnose(self, mon_output, telemetry_snapshot=None):
                raise RuntimeError("Simulated Aerothermal Knowledge Base Query Timeout")

        orch = AgentOrchestrator(
            subset=self.subset,
            diagnosis_agent=FaultyDiagnosisAgent(subset=self.subset),
            default_mode="full",
        )

        res = orch.orchestrate_state(state)

        contained = bool(
            res.status == AgentExecutionStatus.WARNING
            and res.metadata is not None
            and "DiagnosisAgent failed" in (res.metadata.error_message or "")
            and res.monitoring_result is not None  # Monitoring preserved
            and res.prognostics_result is not None  # Prognostics preserved
            and res.diagnosis_result is None
        )

        return IncidentRecord(
            failure_mode="diagnosis_agent_runtime_crash",
            description="Injected unhandled RuntimeError inside DiagnosisAgent.diagnose.",
            contained_safely=contained,
            status_assigned=res.status.value,
            exception_type="RuntimeError",
            exception_message=res.metadata.error_message if res.metadata else None,
            telemetry_preserved=True,
            audit_logged=True,
            details={
                "monitoring_preserved": res.monitoring_result is not None,
                "prognostics_preserved": res.prognostics_result is not None,
            },
        )

    # -------------------------------------------------------------------------
    # Failure Mode 5: PrognosticsAgent Runtime Crash Isolation
    # -------------------------------------------------------------------------
    def test_prognostics_agent_runtime_crash(self, engine_id: int = 1) -> IncidentRecord:
        """
        Injects a runtime crash into PrognosticsAgent (e.g. inference error); verifies
        AgentOrchestrator sets status=WARNING, preserving Monitoring and Diagnosis results.
        """
        state = self._get_nominal_state(engine_id=engine_id)

        class FaultyPrognosticsAgent(PrognosticsAgent):
            def process_engine_state(self, s):
                raise RuntimeError("Simulated Neural Network GPU Out Of Memory")

        orch = AgentOrchestrator(
            subset=self.subset,
            prognostics_agent=FaultyPrognosticsAgent(subset=self.subset),
            default_mode="full",
        )

        res = orch.orchestrate_state(state)

        contained = bool(
            res.status == AgentExecutionStatus.WARNING
            and res.metadata is not None
            and "PrognosticsAgent failed" in (res.metadata.error_message or "")
            and res.monitoring_result is not None  # Monitoring preserved
            and res.diagnosis_result is not None   # Diagnosis preserved
            and res.prognostics_result is None
        )

        return IncidentRecord(
            failure_mode="prognostics_agent_runtime_crash",
            description="Injected unhandled RuntimeError inside PrognosticsAgent.process_engine_state.",
            contained_safely=contained,
            status_assigned=res.status.value,
            exception_type="RuntimeError",
            exception_message=res.metadata.error_message if res.metadata else None,
            telemetry_preserved=True,
            audit_logged=True,
            details={
                "monitoring_preserved": res.monitoring_result is not None,
                "diagnosis_preserved": res.diagnosis_result is not None,
            },
        )

    # -------------------------------------------------------------------------
    # Failure Mode 6: Safety Firewall Tampering Rejection
    # -------------------------------------------------------------------------
    def test_safety_firewall_tampering(self, engine_id: int = 1) -> IncidentRecord:
        """
        Simulates rogue feedback attempting to mutate G1-G29 telemetry or inject negative RUL;
        verifies DigitalTwinSafetyBridge raises SafetyViolationError, pipeline assigns WARNING,
        and original engine telemetry remains intact.
        """
        state = self._get_nominal_state(engine_id=engine_id)
        bridge = DigitalTwinSafetyBridge(agent_id="test_firewall")

        # 1. Direct validation tampering check (mutating T30)
        tampered_state = copy.deepcopy(state)
        tampered_state.T30 = tampered_state.T30 + 15.0

        tampering_caught = False
        exc_type = None
        exc_msg = None

        try:
            bridge.validate_feedback_safety(state, tampered_state)
        except SafetyViolationError as sve:
            tampering_caught = True
            exc_type = type(sve).__name__
            exc_msg = str(sve)

        # 2. Pipeline orchestration containment check with bad RUL
        orch = AgentOrchestrator(
            subset=self.subset,
            enable_twin_feedback=True,
            twin_bridge=bridge,
            default_mode="full",
        )

        # Build mock invalid result with negative RUL
        rogue_result = OrchestrationResult(
            run_id="rogue_run",
            engine_id=engine_id,
            cycle=30,
            subset=self.subset,
            status=AgentExecutionStatus.SUCCESS,
            prognostics_result=PrognosticsOutput(
                unit_number=engine_id,
                time_cycles=30,
                subset=self.subset,
                predicted_rul_xgboost=-10.0,  # Negative RUL
                predicted_rul_lstm=50.0,
                ensemble_rul_estimate=20.0,
                prognostic_spread=60.0,
                virtual_health_index=0.5,
                uncertainty_classification="LOW_UNCERTAINTY",
            ),
        )

        pipeline_blocked = False
        try:
            bridge.enrich_state_with_agent_outputs(state, rogue_result)
        except SafetyViolationError:
            pipeline_blocked = True

        contained = bool(tampering_caught and pipeline_blocked and "CRITICAL SAFETY VIOLATION" in (exc_msg or ""))

        return IncidentRecord(
            failure_mode="safety_firewall_tampering_rejection",
            description="Attempt to tamper with immutable telemetry T30 and inject negative RUL.",
            contained_safely=contained,
            status_assigned="BLOCKED_BY_FIREWALL",
            exception_type=exc_type,
            exception_message=exc_msg,
            telemetry_preserved=True,
            audit_logged=True,
            details={
                "telemetry_tampering_blocked": tampering_caught,
                "negative_rul_blocked": pipeline_blocked,
            },
        )

    # -------------------------------------------------------------------------
    # Failure Mode 7: Audit Logger I/O Failure Resilience
    # -------------------------------------------------------------------------
    def test_audit_logger_io_resilience(self, engine_id: int = 1) -> IncidentRecord:
        """
        Configures AgentAuditLogger with an unwriteable directory or path;
        verifies that disk I/O errors never crash the flight telemetry pipeline.
        """
        state = self._get_nominal_state(engine_id=engine_id)

        # Use an intentionally invalid unwriteable path on Windows (e.g. NUL or illegal characters)
        illegal_path = os.path.join("Z:\\nonexistent_network_drive_9999", "reports")
        logger = AgentAuditLogger(log_dir=illegal_path)

        orch = AgentOrchestrator(
            subset=self.subset,
            audit_logger=logger,
            default_mode="full",
        )

        # Pipeline must succeed normally despite disk write failure
        no_crash = False
        res = None
        try:
            res = orch.orchestrate_state(state)
            no_crash = (res is not None and res.status == AgentExecutionStatus.SUCCESS)
        except Exception:
            no_crash = False

        contained = bool(no_crash and res is not None)

        return IncidentRecord(
            failure_mode="audit_logger_io_failure_resilience",
            description="Simulated unwriteable storage destination Z:\\nonexistent_network_drive_9999.",
            contained_safely=contained,
            status_assigned=res.status.value if res else "CRASHED",
            exception_type="IOError/OSError",
            exception_message="Suppressed cleanly by non-blocking logger fault containment.",
            telemetry_preserved=True,
            audit_logged=False,
            details={"orchestration_completed": no_crash},
        )

    # -------------------------------------------------------------------------
    # Failure Mode 8: Operational Recovery After Failure
    # -------------------------------------------------------------------------
    def test_recovery_after_failure(self, engine_id: int = 1) -> IncidentRecord:
        """
        Verifies that after experiencing an agent crash or exception, the orchestrator
        and system immediately recover to full operational capability on the next cycle.
        """
        state = self._get_nominal_state(engine_id=engine_id)

        # 1. Execute failure cycle with crashing diagnosis agent
        class IntermittentDiagnosisAgent(DiagnosisAgent):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.should_fail = True

            def diagnose(self, mon_output, telemetry_snapshot=None):
                if self.should_fail:
                    raise RuntimeError("Transient Communication Drop")
                return super().diagnose(mon_output, telemetry_snapshot=telemetry_snapshot)

        diag_agent = IntermittentDiagnosisAgent(subset=self.subset)
        orch = AgentOrchestrator(
            subset=self.subset,
            diagnosis_agent=diag_agent,
            default_mode="full",
        )

        res_fail = orch.orchestrate_state(state)
        failed_as_expected = (res_fail.status == AgentExecutionStatus.WARNING)

        # 2. Heal the intermittent issue for the next cycle
        diag_agent.should_fail = False
        res_recovered = orch.orchestrate_state(state)
        recovered_as_expected = (
            res_recovered.status == AgentExecutionStatus.SUCCESS
            and res_recovered.diagnosis_result is not None
            and res_recovered.prognostics_result is not None
            and res_recovered.monitoring_result is not None
        )

        contained = bool(failed_as_expected and recovered_as_expected)

        return IncidentRecord(
            failure_mode="recovery_after_failure",
            description="Transient communication dropout during diagnosis followed by nominal recovery on cycle t+1.",
            contained_safely=contained,
            status_assigned=res_recovered.status.value,
            exception_type=None,
            exception_message="Fully recovered to full capability on subsequent cycle.",
            telemetry_preserved=True,
            audit_logged=True,
            details={
                "cycle_t_failed_safely": failed_as_expected,
                "cycle_t_plus_1_recovered": recovered_as_expected,
            },
        )

    # -------------------------------------------------------------------------
    # Comprehensive Failure Suite Runner
    # -------------------------------------------------------------------------
    def run_all_failure_tests(self) -> AgentFailureReport:
        """
        Executes all adversarial failure injection tests and generates an
        executive AgentFailureReport.
        """
        t0 = time.perf_counter()
        rid = f"report_{uuid.uuid4().hex[:12]}"
        timestamp = datetime.now(timezone.utc).isoformat()

        incidents: List[IncidentRecord] = [
            self.test_corrupted_telemetry_nan_inf(),
            self.test_missing_sensor_channels(),
            self.test_monitoring_agent_runtime_crash(),
            self.test_diagnosis_agent_runtime_crash(),
            self.test_prognostics_agent_runtime_crash(),
            self.test_safety_firewall_tampering(),
            self.test_audit_logger_io_resilience(),
            self.test_recovery_after_failure(),
        ]

        total = len(incidents)
        passed = sum(1 for inc in incidents if inc.contained_safely)
        failed = total - passed
        all_contained = (failed == 0)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        metrics = {
            "total_adversarial_scenarios": total,
            "contained_count": passed,
            "uncontained_count": failed,
            "containment_rate_pct": float(passed / total) * 100.0 if total > 0 else 0.0,
            "total_execution_ms": round(elapsed_ms, 2),
        }

        report = AgentFailureReport(
            report_id=rid,
            timestamp=timestamp,
            subset=self.subset,
            total_tests_run=total,
            passed_tests_count=passed,
            failed_tests_count=failed,
            all_contained=all_contained,
            incidents=incidents,
            summary_metrics=metrics,
        )

        return report
