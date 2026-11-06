"""
Safety Decision Audit Logging Subsystem
Captures, classifies, and records the explicit reason for every safety decision
into structured, append-only audit trails.

Roadmap Reference: Phase 7 - Step 7.5
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
import os
import threading
from typing import Dict, Any, List, Optional, Union
from uuid import uuid4

from safety.decision import (
    SafetyDecisionStatus,
    ActionSafetyDecision,
)
from safety.verification import SimulationAction


class SafetyReasonCode(str, Enum):
    """
    Standardized aerospace domain reason codes explaining safety verdicts.
    """
    NOMINAL_EXECUTION = "NOMINAL_EXECUTION"
    TELEMETRY_FIREWALL_BREACH = "TELEMETRY_FIREWALL_BREACH"
    THERMAL_CREEP_ADVISORY = "THERMAL_CREEP_ADVISORY"
    METALLURGICAL_HARD_LIMIT_BREACH = "METALLURGICAL_HARD_LIMIT_BREACH"
    PNEUMATIC_CASING_OVERPRESSURE = "PNEUMATIC_CASING_OVERPRESSURE"
    STAGE_ORDERING_INVERSION = "STAGE_ORDERING_INVERSION"
    AERODYNAMIC_ENVELOPE_EXCEEDANCE = "AERODYNAMIC_ENVELOPE_EXCEEDANCE"
    PROGNOSTIC_CAUSALITY_VIOLATION = "PROGNOSTIC_CAUSALITY_VIOLATION"
    GENERAL_CONSTRAINT_VIOLATION = "GENERAL_CONSTRAINT_VIOLATION"


@dataclass
class SafetyAuditRecord:
    """
    Structured record documenting an actionable safety decision and its exact rationale.
    """
    record_id: str
    action_id: str
    decision_status: SafetyDecisionStatus
    target_variable: str
    reason_code: SafetyReasonCode
    summary_reason: str
    detailed_reasons: List[str] = field(default_factory=list)
    violated_constraints: List[str] = field(default_factory=list)
    initiator_id: str = "simulation_harness"
    flight_cycle: Optional[int] = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "action_id": self.action_id,
            "decision_status": self.decision_status.value,
            "target_variable": self.target_variable,
            "reason_code": self.reason_code.value,
            "summary_reason": self.summary_reason,
            "detailed_reasons": list(self.detailed_reasons),
            "violated_constraints": list(self.violated_constraints),
            "initiator_id": self.initiator_id,
            "flight_cycle": self.flight_cycle,
            "timestamp": self.timestamp,
            "metadata": dict(self.metadata),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SafetyAuditRecord":
        return cls(
            record_id=data["record_id"],
            action_id=data["action_id"],
            decision_status=SafetyDecisionStatus(data["decision_status"]),
            target_variable=data["target_variable"],
            reason_code=SafetyReasonCode(data["reason_code"]),
            summary_reason=data["summary_reason"],
            detailed_reasons=data.get("detailed_reasons", []),
            violated_constraints=data.get("violated_constraints", []),
            initiator_id=data.get("initiator_id", "simulation_harness"),
            flight_cycle=data.get("flight_cycle"),
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
            metadata=data.get("metadata", {}),
        )


def infer_reason_code(decision: ActionSafetyDecision, action: SimulationAction) -> SafetyReasonCode:
    """
    Infers the primary domain SafetyReasonCode based on the decision status and violations.
    """
    # 1. Nominal Accepted
    if decision.status == SafetyDecisionStatus.ACCEPTED:
        return SafetyReasonCode.NOMINAL_EXECUTION

    violated_ids = [v.constraint_id for v in decision.violations]

    # 2. Firewall Breaches
    if any(v.constraint_id == "FIREWALL_MEASURED_IMMUTABILITY" for v in decision.violations) or (
        decision.is_rejected and not decision.summary_message.find("firewall") == -1
    ):
        return SafetyReasonCode.TELEMETRY_FIREWALL_BREACH

    # 3. Aerothermal Stage Ordering Inversions
    if any("INVARIANT" in cid for cid in violated_ids):
        return SafetyReasonCode.STAGE_ORDERING_INVERSION

    # 4. Metallurgical Hard Limit (T30 > 1625 °R)
    if "CONST_TEMP_T30_MAX_WARNING" in violated_ids:
        return SafetyReasonCode.METALLURGICAL_HARD_LIMIT_BREACH

    # 5. Pneumatic Casing Overpressure (P30 > 700 psia)
    if "CONST_PRESS_P30" in violated_ids or "CONST_PRESS_PS30" in violated_ids:
        return SafetyReasonCode.PNEUMATIC_CASING_OVERPRESSURE

    # 6. Aerodynamic Flight Envelope Exceedance
    if any(cid in ("CONST_FLIGHT_ALTITUDE", "CONST_FLIGHT_MACH", "CONST_FLIGHT_TRA") for cid in violated_ids):
        return SafetyReasonCode.AERODYNAMIC_ENVELOPE_EXCEEDANCE

    # 7. Prognostic Causality Violations
    if any(cid in ("CONST_PROG_RUL_NON_NEGATIVE", "CONST_PROG_HEALTH_INDEX") for cid in violated_ids):
        return SafetyReasonCode.PROGNOSTIC_CAUSALITY_VIOLATION

    # 8. Thermal Creep Caution Advisory
    if decision.status == SafetyDecisionStatus.REQUIRES_REVIEW or "CONST_TEMP_T30_CAUTION" in violated_ids:
        return SafetyReasonCode.THERMAL_CREEP_ADVISORY

    return SafetyReasonCode.GENERAL_CONSTRAINT_VIOLATION


class SafetyAuditLogger:
    """
    Thread-safe, append-only logger maintaining the immutable safety decision audit trail.
    """

    def __init__(
        self,
        log_dir: str = "reports/safety_logs",
        log_filename: str = "safety_audit_trail.jsonl",
    ):
        self.log_dir = log_dir
        self.log_filename = log_filename
        self.log_path = os.path.join(self.log_dir, self.log_filename)
        self._lock = threading.Lock()
        self._in_memory_records: List[SafetyAuditRecord] = []

        # Ensure target logging directory exists
        try:
            os.makedirs(self.log_dir, exist_ok=True)
        except Exception:
            pass

    def log_decision(
        self,
        decision: ActionSafetyDecision,
        action: SimulationAction,
        current_state: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SafetyAuditRecord:
        """
        Constructs a SafetyAuditRecord and appends it to the in-memory log and JSONL file.
        Gracefully handles file I/O exceptions so logging never interrupts simulation.
        """
        # Extract flight cycle if available
        flight_cycle = None
        if current_state is not None:
            if hasattr(current_state, "time_cycles"):
                flight_cycle = int(current_state.time_cycles)
            elif isinstance(current_state, dict):
                flight_cycle = current_state.get("time_cycles") or current_state.get("G2_time_cycles")
                if flight_cycle is not None:
                    try:
                        flight_cycle = int(flight_cycle)
                    except (ValueError, TypeError):
                        flight_cycle = None

        reason_code = infer_reason_code(decision, action)
        detailed_reasons = (
            [v.message for v in decision.violations]
            if decision.violations
            else [decision.summary_message]
        )
        violated_constraints = [v.constraint_id for v in decision.violations]

        rec_meta = dict(metadata or {})
        rec_meta["action_type"] = action.action_type.value
        rec_meta["proposed_value"] = str(action.proposed_value)
        if action.delta_value is not None:
            rec_meta["delta_value"] = action.delta_value

        record = SafetyAuditRecord(
            record_id=f"SAR_{uuid4().hex[:12]}",
            action_id=action.action_id,
            decision_status=decision.status,
            target_variable=action.target_variable,
            reason_code=reason_code,
            summary_reason=decision.summary_message,
            detailed_reasons=detailed_reasons,
            violated_constraints=violated_constraints,
            initiator_id=action.initiator_id,
            flight_cycle=flight_cycle,
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata=rec_meta,
        )

        with self._lock:
            self._in_memory_records.append(record)

            # Append to file
            try:
                os.makedirs(self.log_dir, exist_ok=True)
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(record.to_json() + "\n")
            except Exception:
                # Never crash simulation harness on I/O issues
                pass

        return record

    def get_recent_records(self, n: int = 10) -> List[SafetyAuditRecord]:
        """Returns the most recent n audit records."""
        with self._lock:
            return list(self._in_memory_records[-n:])

    def filter_by_status(self, status: Union[SafetyDecisionStatus, str]) -> List[SafetyAuditRecord]:
        """Filters in-memory records by SafetyDecisionStatus."""
        status_val = status.value if isinstance(status, SafetyDecisionStatus) else str(status)
        with self._lock:
            return [r for r in self._in_memory_records if r.decision_status.value == status_val]

    def filter_by_reason_code(self, reason_code: Union[SafetyReasonCode, str]) -> List[SafetyAuditRecord]:
        """Filters in-memory records by SafetyReasonCode."""
        rc_val = reason_code.value if isinstance(reason_code, SafetyReasonCode) else str(reason_code)
        with self._lock:
            return [r for r in self._in_memory_records if r.reason_code.value == rc_val]

    def clear_in_memory(self) -> None:
        """Clears in-memory audit records without deleting disk files."""
        with self._lock:
            self._in_memory_records.clear()
