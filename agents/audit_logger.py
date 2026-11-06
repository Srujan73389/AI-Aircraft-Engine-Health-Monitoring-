"""
Structured Audit Logging Module - Phase 6 Step 6.6
High-fidelity audit logging and provenance subsystem for the AI-Based Aircraft Engine Multi-Agent System.

Roadmap Reference: Phase 6 - Step 6.6:
"Add structured logs showing each agent's input, output, timestamp, and status"

Capabilities:
1. Standardized Audit Record Schema:
   - Captures log_id (UUID), run_id, engine_id, cycle, timestamp (ISO 8601 UTC).
   - Records agent_id, agent_type ("MONITORING", "DIAGNOSIS", "PROGNOSTICS", "ORCHESTRATOR").
   - Sanitized input and output summaries with key telemetry, breaches, fault modes, RUL, and health index.
   - Status tracking (AgentExecutionStatus string), execution latency (ms), and optional error diagnostics.
2. Dual Output Storage:
   - Appendable, machine-readable JSON-Lines audit trail (reports/agent_logs/agent_audit_trail.jsonl).
   - Session-based Markdown summary report generator.
3. Fast Query & Retrieval API:
   - get_logs_by_run(run_id)
   - get_logs_by_engine(engine_id, max_records)
   - get_logs_by_status(status)
4. Non-Blocking Fault Containment:
   - Disk and file I/O operations are fully wrapped in exception containment; logging failures
     will never crash or interrupt the mission-critical agent pipeline.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import json
import math
import os
import sys
import threading
import time
import uuid
from typing import Dict, Any, List, Optional, Tuple, Union

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


DEFAULT_LOG_DIR = os.path.join("reports", "agent_logs")
DEFAULT_JSONL_FILE = "agent_audit_trail.jsonl"
DEFAULT_MD_FILE = "agent_audit_summary.md"


@dataclass
class AuditLogRecord:
    """Standardized structured audit record capturing agent execution provenance."""
    log_id: str
    run_id: str
    engine_id: int
    cycle: int
    timestamp: str
    agent_id: str
    agent_type: str  # "MONITORING", "DIAGNOSIS", "PROGNOSTICS", "ORCHESTRATOR"
    input_summary: Dict[str, Any]
    output_summary: Dict[str, Any]
    status: str      # Value of AgentExecutionStatus (e.g. "SUCCESS", "WARNING", "FAILED")
    execution_time_ms: float
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "log_id": self.log_id,
            "run_id": self.run_id,
            "engine_id": int(self.engine_id),
            "cycle": int(self.cycle),
            "timestamp": self.timestamp,
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "input_summary": self.input_summary,
            "output_summary": self.output_summary,
            "status": self.status,
            "execution_time_ms": float(self.execution_time_ms),
            "error_message": self.error_message,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuditLogRecord":
        return cls(
            log_id=data["log_id"],
            run_id=data["run_id"],
            engine_id=int(data["engine_id"]),
            cycle=int(data["cycle"]),
            timestamp=data["timestamp"],
            agent_id=data["agent_id"],
            agent_type=data["agent_type"],
            input_summary=dict(data.get("input_summary", {})),
            output_summary=dict(data.get("output_summary", {})),
            status=data["status"],
            execution_time_ms=float(data["execution_time_ms"]),
            error_message=data.get("error_message"),
        )


class AgentAuditLogger:
    """
    Subsystem for capturing, appending, and querying structured multi-agent audit logs.
    Guarantees thread-safe writes and non-crashing execution.
    """

    def __init__(
        self,
        log_dir: str = DEFAULT_LOG_DIR,
        jsonl_filename: str = DEFAULT_JSONL_FILE,
        md_filename: str = DEFAULT_MD_FILE,
        in_memory_only: bool = False,
    ):
        """
        Initializes the AgentAuditLogger.

        Args:
            log_dir: Directory where log files are stored (default: reports/agent_logs/).
            jsonl_filename: Filename for the JSON-Lines audit trail.
            md_filename: Filename for the markdown session summary report.
            in_memory_only: If True, skips disk writes (useful for hermetic test execution).
        """
        self.log_dir = log_dir
        self.jsonl_filename = jsonl_filename
        self.md_filename = md_filename
        self.in_memory_only = in_memory_only

        self.jsonl_path = os.path.join(self.log_dir, self.jsonl_filename)
        self.md_path = os.path.join(self.log_dir, self.md_filename)

        self._lock = threading.Lock()
        self._in_memory_records: List[Dict[str, Any]] = []

        if not self.in_memory_only:
            try:
                os.makedirs(self.log_dir, exist_ok=True)
            except Exception:
                pass  # Fault containment for directory creation

    def log_record(
        self,
        run_id: str,
        engine_id: int,
        cycle: int,
        agent_id: str,
        agent_type: str,
        input_summary: Dict[str, Any],
        output_summary: Dict[str, Any],
        status: Union[str, AgentExecutionStatus],
        execution_time_ms: float,
        error_message: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Constructs and persists a single structured audit log record.

        Returns:
            Dictionary representation of the recorded log entry.
        """
        status_str = status.value if isinstance(status, AgentExecutionStatus) else str(status)
        timestamp = datetime.now(timezone.utc).isoformat()
        log_id = str(uuid.uuid4())

        record = AuditLogRecord(
            log_id=log_id,
            run_id=run_id,
            engine_id=engine_id,
            cycle=cycle,
            timestamp=timestamp,
            agent_id=agent_id,
            agent_type=agent_type.upper(),
            input_summary=input_summary,
            output_summary=output_summary,
            status=status_str,
            execution_time_ms=float(execution_time_ms),
            error_message=error_message,
        )

        record_dict = record.to_dict()

        with self._lock:
            # Always cache in memory for query API
            self._in_memory_records.append(record_dict)

            # Persist to disk if not in-memory-only
            if not self.in_memory_only:
                try:
                    os.makedirs(self.log_dir, exist_ok=True)
                    with open(self.jsonl_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(record_dict) + "\n")
                except Exception as e:
                    # Non-blocking fault containment: Never crash caller on write failure
                    pass

        return record_dict

    def log_monitoring(
        self,
        run_id: str,
        engine_id: int,
        cycle: int,
        monitoring_output: MonitoringOutput,
        input_summary: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Specialized logger for MonitoringAgent outputs."""
        inp_sum = input_summary or {"engine_id": engine_id, "cycle": cycle}
        out_sum = {
            "operational_state": monitoring_output.operational_state.value,
            "threshold_breaches_count": len(monitoring_output.threshold_breaches),
            "breached_sensors": [b.sensor_name for b in monitoring_output.threshold_breaches],
            "model_anomaly_flag": monitoring_output.model_anomaly_flag,
            "persistence_count": monitoring_output.persistence_count,
            "persistence_alert": monitoring_output.persistence_alert,
            "anomaly_severity": monitoring_output.anomaly_severity.value,
        }
        meta = monitoring_output.metadata
        agent_id = meta.agent_id if meta else "monitoring_agent"
        status = meta.status if meta else AgentExecutionStatus.SUCCESS
        exec_ms = meta.execution_time_ms if meta else 0.0

        return self.log_record(
            run_id=run_id,
            engine_id=engine_id,
            cycle=cycle,
            agent_id=agent_id,
            agent_type="MONITORING",
            input_summary=inp_sum,
            output_summary=out_sum,
            status=status,
            execution_time_ms=exec_ms,
            error_message=meta.error_message if meta else None,
        )

    def log_diagnosis(
        self,
        run_id: str,
        engine_id: int,
        cycle: int,
        diagnosis_output: DiagnosisOutput,
        input_summary: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Specialized logger for DiagnosisAgent outputs."""
        inp_sum = input_summary or {"engine_id": engine_id, "cycle": cycle}
        out_sum = {
            "primary_fault": diagnosis_output.primary_fault,
            "confidence_score": float(diagnosis_output.confidence_score),
            "fault_candidates_count": len(diagnosis_output.fault_candidates),
            "candidates": [c.fault_type for c in diagnosis_output.fault_candidates],
            "severity": diagnosis_output.severity.value,
            "requires_maintenance_action": diagnosis_output.requires_maintenance_action,
        }
        meta = diagnosis_output.metadata
        agent_id = meta.agent_id if meta else "diagnosis_agent"
        status = meta.status if meta else AgentExecutionStatus.SUCCESS
        exec_ms = meta.execution_time_ms if meta else 0.0

        return self.log_record(
            run_id=run_id,
            engine_id=engine_id,
            cycle=cycle,
            agent_id=agent_id,
            agent_type="DIAGNOSIS",
            input_summary=inp_sum,
            output_summary=out_sum,
            status=status,
            execution_time_ms=exec_ms,
            error_message=meta.error_message if meta else None,
        )

    def log_prognostics(
        self,
        run_id: str,
        engine_id: int,
        cycle: int,
        prognostics_output: PrognosticsOutput,
        input_summary: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Specialized logger for PrognosticsAgent outputs."""
        inp_sum = input_summary or {"engine_id": engine_id, "cycle": cycle}
        out_sum = {
            "predicted_rul_xgboost": float(prognostics_output.predicted_rul_xgboost),
            "predicted_rul_lstm": float(prognostics_output.predicted_rul_lstm),
            "ensemble_rul_estimate": float(prognostics_output.ensemble_rul_estimate),
            "prognostic_spread": float(prognostics_output.prognostic_spread),
            "virtual_health_index": float(prognostics_output.virtual_health_index),
            "uncertainty_classification": prognostics_output.uncertainty_classification,
            "cycles_to_caution_threshold": prognostics_output.cycles_to_caution_threshold,
        }
        meta = prognostics_output.metadata
        agent_id = meta.agent_id if meta else "prognostics_agent"
        status = meta.status if meta else AgentExecutionStatus.SUCCESS
        exec_ms = meta.execution_time_ms if meta else 0.0

        return self.log_record(
            run_id=run_id,
            engine_id=engine_id,
            cycle=cycle,
            agent_id=agent_id,
            agent_type="PROGNOSTICS",
            input_summary=inp_sum,
            output_summary=out_sum,
            status=status,
            execution_time_ms=exec_ms,
            error_message=meta.error_message if meta else None,
        )

    def log_orchestration(
        self,
        run_id: str,
        engine_id: int,
        cycle: int,
        orchestration_result: OrchestrationResult,
        input_summary: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Specialized logger for overall AgentOrchestrator pipeline completion."""
        inp_sum = input_summary or {"engine_id": engine_id, "cycle": cycle, "subset": orchestration_result.subset}
        out_sum = {
            "pipeline_status": orchestration_result.status.value,
            "execution_timeline_ms": orchestration_result.execution_timeline_ms,
            "has_monitoring": orchestration_result.monitoring_result is not None,
            "has_diagnosis": orchestration_result.diagnosis_result is not None,
            "has_prognostics": orchestration_result.prognostics_result is not None,
        }
        meta = orchestration_result.metadata
        agent_id = meta.agent_id if meta else "agent_orchestrator"
        status = orchestration_result.status
        exec_ms = meta.execution_time_ms if meta else orchestration_result.execution_timeline_ms.get("total_elapsed_ms", 0.0)

        return self.log_record(
            run_id=run_id,
            engine_id=engine_id,
            cycle=cycle,
            agent_id=agent_id,
            agent_type="ORCHESTRATOR",
            input_summary=inp_sum,
            output_summary=out_sum,
            status=status,
            execution_time_ms=exec_ms,
            error_message=meta.error_message if meta else None,
        )

    def get_all_logs(self) -> List[Dict[str, Any]]:
        """Retrieves all log records currently in memory or persisted on disk."""
        with self._lock:
            if len(self._in_memory_records) > 0:
                return list(self._in_memory_records)

            if not self.in_memory_only and os.path.exists(self.jsonl_path):
                records = []
                try:
                    with open(self.jsonl_path, "r", encoding="utf-8") as f:
                        for line in f:
                            clean = line.strip()
                            if clean:
                                records.append(json.loads(clean))
                    return records
                except Exception:
                    return []
            return []

    def get_logs_by_run(self, run_id: str) -> List[Dict[str, Any]]:
        """Retrieves all log records matching a specific pipeline run_id."""
        return [r for r in self.get_all_logs() if r.get("run_id") == run_id]

    def get_logs_by_engine(self, engine_id: int, max_records: int = 100) -> List[Dict[str, Any]]:
        """Retrieves log records for a specific engine identifier."""
        matched = [r for r in self.get_all_logs() if r.get("engine_id") == engine_id]
        return matched[-max_records:]

    def get_logs_by_status(self, status: Union[str, AgentExecutionStatus]) -> List[Dict[str, Any]]:
        """Retrieves log records matching an execution status string."""
        st_str = status.value if isinstance(status, AgentExecutionStatus) else str(status)
        return [r for r in self.get_all_logs() if r.get("status") == st_str]

    def write_markdown_summary(self, output_path: Optional[str] = None) -> str:
        """
        Generates and saves a publication-grade markdown summary table of recent logs.

        Args:
            output_path: Optional destination path. Defaults to reports/agent_logs/agent_audit_summary.md.

        Returns:
            Rendered markdown string.
        """
        dest = output_path or self.md_path
        logs = self.get_all_logs()

        lines = [
            "# Multi-Agent Audit Trail Summary",
            "",
            f"**Generated:** {datetime.now(timezone.utc).isoformat()}  ",
            f"**Total Audit Records:** {len(logs)}  ",
            "",
            "| Timestamp | Run ID | Engine ID | Cycle | Agent Type | Status | Latency (ms) | Output Summary |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ]

        # Take last 50 entries
        recent = logs[-50:] if len(logs) > 50 else logs
        for r in recent:
            ts = r.get("timestamp", "")[:19]
            rid = r.get("run_id", "")[:16]
            eid = r.get("engine_id", 0)
            cyc = r.get("cycle", 0)
            atype = r.get("agent_type", "")
            st = r.get("status", "")
            lat = f"{r.get('execution_time_ms', 0.0):.2f}"
            out_s = json.dumps(r.get("output_summary", {}))
            if len(out_s) > 40:
                out_s = out_s[:37] + "..."
            # Escape pipe symbols for markdown tables
            out_s = out_s.replace("|", "/")
            lines.append(f"| {ts} | `{rid}` | {eid} | {cyc} | **{atype}** | `{st}` | {lat} | `{out_s}` |")

        md_content = "\n".join(lines) + "\n"

        if not self.in_memory_only:
            try:
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                with open(dest, "w", encoding="utf-8") as f:
                    f.write(md_content)
            except Exception:
                pass

        return md_content
