"""
Multi-Agent Schemas Module - Phase 6 Step 6.1
Defines structured, strongly-typed input and output schemas for all monitoring,
diagnosis, and prognostics agents in the AI Aircraft Engine Digital Twin multi-agent system.

Roadmap Reference: Phase 6 - Step 6.1:
"Define the input and output schema for every agent"
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
import json
import math
import os
import sys
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np


# -----------------------------------------------------------------------------
# Common Enumerations & Status Indicators
# -----------------------------------------------------------------------------
class AgentExecutionStatus(str, Enum):
    """Execution status for agent invocations."""
    PENDING = "PENDING"
    SUCCESS = "SUCCESS"
    WARNING = "WARNING"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class SeverityLevel(str, Enum):
    """Operational anomaly and fault severity levels."""
    NOMINAL = "NOMINAL"
    ADVISORY = "ADVISORY"
    CAUTION = "CAUTION"
    WARNING = "WARNING"


class OperationalState(str, Enum):
    """Engine macro-level operational health states."""
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    ANOMALOUS = "ANOMALOUS"
    CRITICAL = "CRITICAL"


# -----------------------------------------------------------------------------
# Common Metadata Schema
# -----------------------------------------------------------------------------
@dataclass
class AgentMetadata:
    """Standardized metadata tracking agent execution provenance."""
    agent_id: str
    execution_timestamp: str
    execution_time_ms: float
    status: AgentExecutionStatus
    error_message: Optional[str] = None
    version: str = "1.0.0"

    def validate(self) -> None:
        """Validate metadata consistency."""
        if not self.agent_id:
            raise ValueError("AgentMetadata: agent_id cannot be empty.")
        if self.execution_time_ms < 0.0:
            raise ValueError(f"AgentMetadata: execution_time_ms cannot be negative ({self.execution_time_ms}).")
        if not isinstance(self.status, AgentExecutionStatus):
            raise TypeError(f"AgentMetadata: status must be AgentExecutionStatus enum, got {type(self.status)}.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "execution_timestamp": self.execution_timestamp,
            "execution_time_ms": float(self.execution_time_ms),
            "status": self.status.value,
            "error_message": self.error_message,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentMetadata":
        status_val = data["status"]
        if isinstance(status_val, str):
            status_enum = AgentExecutionStatus(status_val)
        else:
            status_enum = status_val
        return cls(
            agent_id=data["agent_id"],
            execution_timestamp=data["execution_timestamp"],
            execution_time_ms=float(data["execution_time_ms"]),
            status=status_enum,
            error_message=data.get("error_message"),
            version=data.get("version", "1.0.0"),
        )


# -----------------------------------------------------------------------------
# 1. Monitoring Agent Schemas
# -----------------------------------------------------------------------------
@dataclass
class ThresholdBreach:
    """Represents a specific physical sensor threshold violation."""
    sensor_name: str
    observed_value: float
    threshold_limit: float
    direction: str  # "HIGH" or "LOW"
    breach_margin: float
    severity: SeverityLevel

    def validate(self) -> None:
        if not math.isfinite(self.observed_value):
            raise ValueError(f"ThresholdBreach: observed_value must be finite ({self.observed_value}).")
        if self.direction not in ("HIGH", "LOW"):
            raise ValueError(f"ThresholdBreach: direction must be 'HIGH' or 'LOW', got {self.direction}.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sensor_name": self.sensor_name,
            "observed_value": float(self.observed_value),
            "threshold_limit": float(self.threshold_limit),
            "direction": self.direction,
            "breach_margin": float(self.breach_margin),
            "severity": self.severity.value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ThresholdBreach":
        sev = data["severity"]
        sev_enum = SeverityLevel(sev) if isinstance(sev, str) else sev
        return cls(
            sensor_name=data["sensor_name"],
            observed_value=float(data["observed_value"]),
            threshold_limit=float(data["threshold_limit"]),
            direction=data["direction"],
            breach_margin=float(data["breach_margin"]),
            severity=sev_enum,
        )


@dataclass
class MonitoringInput:
    """Input payload delivered to the Monitoring Agent."""
    unit_number: int
    time_cycles: int
    operating_context: Dict[str, float]  # altitude (G3), mach_number (G4), throttle_resolver_angle (G5)
    raw_telemetry: Dict[str, float]      # T2 through W32 (G6 - G26)
    derived_indicators: Dict[str, float] # G27 (delta_T), G28 (OPR), G29 (spool_ratio)
    anomaly_scores: Dict[str, Optional[float]] = field(default_factory=dict) # G32, G33

    def validate(self) -> None:
        if self.unit_number <= 0:
            raise ValueError(f"MonitoringInput: unit_number must be > 0 ({self.unit_number}).")
        if self.time_cycles <= 0:
            raise ValueError(f"MonitoringInput: time_cycles must be > 0 ({self.time_cycles}).")
        for k, v in self.operating_context.items():
            if not math.isfinite(v):
                raise ValueError(f"MonitoringInput: non-finite operating context for {k}: {v}")
        for k, v in self.raw_telemetry.items():
            if not math.isfinite(v):
                raise ValueError(f"MonitoringInput: non-finite telemetry for {k}: {v}")
        for k, v in self.derived_indicators.items():
            if not math.isfinite(v):
                raise ValueError(f"MonitoringInput: non-finite derived indicator for {k}: {v}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "unit_number": int(self.unit_number),
            "time_cycles": int(self.time_cycles),
            "operating_context": {k: float(v) for k, v in self.operating_context.items()},
            "raw_telemetry": {k: float(v) for k, v in self.raw_telemetry.items()},
            "derived_indicators": {k: float(v) for k, v in self.derived_indicators.items()},
            "anomaly_scores": {k: (float(v) if v is not None else None) for k, v in self.anomaly_scores.items()},
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MonitoringInput":
        return cls(
            unit_number=int(data["unit_number"]),
            time_cycles=int(data["time_cycles"]),
            operating_context=dict(data["operating_context"]),
            raw_telemetry=dict(data["raw_telemetry"]),
            derived_indicators=dict(data["derived_indicators"]),
            anomaly_scores=dict(data.get("anomaly_scores", {})),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "MonitoringInput":
        return cls.from_dict(json.loads(json_str))


@dataclass
class MonitoringOutput:
    """Output diagnostics produced by the Monitoring Agent."""
    unit_number: int
    time_cycles: int
    operational_state: OperationalState
    threshold_breaches: List[ThresholdBreach] = field(default_factory=list)
    model_anomaly_flag: bool = False
    isolation_forest_breach: bool = False
    pca_spe_breach: bool = False
    persistence_count: int = 0
    persistence_alert: bool = False
    anomaly_severity: SeverityLevel = SeverityLevel.NOMINAL
    summary_message: str = ""
    metadata: Optional[AgentMetadata] = None

    def validate(self) -> None:
        if self.unit_number <= 0:
            raise ValueError(f"MonitoringOutput: unit_number must be > 0 ({self.unit_number}).")
        if self.time_cycles <= 0:
            raise ValueError(f"MonitoringOutput: time_cycles must be > 0 ({self.time_cycles}).")
        if self.persistence_count < 0:
            raise ValueError(f"MonitoringOutput: persistence_count cannot be negative ({self.persistence_count}).")
        for b in self.threshold_breaches:
            b.validate()
        if self.metadata is not None:
            self.metadata.validate()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "unit_number": int(self.unit_number),
            "time_cycles": int(self.time_cycles),
            "operational_state": self.operational_state.value,
            "threshold_breaches": [b.to_dict() for b in self.threshold_breaches],
            "model_anomaly_flag": bool(self.model_anomaly_flag),
            "isolation_forest_breach": bool(self.isolation_forest_breach),
            "pca_spe_breach": bool(self.pca_spe_breach),
            "persistence_count": int(self.persistence_count),
            "persistence_alert": bool(self.persistence_alert),
            "anomaly_severity": self.anomaly_severity.value,
            "summary_message": self.summary_message,
            "metadata": self.metadata.to_dict() if self.metadata else None,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MonitoringOutput":
        op_state = data["operational_state"]
        op_state_enum = OperationalState(op_state) if isinstance(op_state, str) else op_state
        sev = data["anomaly_severity"]
        sev_enum = SeverityLevel(sev) if isinstance(sev, str) else sev
        breaches = [ThresholdBreach.from_dict(b) for b in data.get("threshold_breaches", [])]
        meta = AgentMetadata.from_dict(data["metadata"]) if data.get("metadata") else None

        return cls(
            unit_number=int(data["unit_number"]),
            time_cycles=int(data["time_cycles"]),
            operational_state=op_state_enum,
            threshold_breaches=breaches,
            model_anomaly_flag=bool(data.get("model_anomaly_flag", False)),
            isolation_forest_breach=bool(data.get("isolation_forest_breach", False)),
            pca_spe_breach=bool(data.get("pca_spe_breach", False)),
            persistence_count=int(data.get("persistence_count", 0)),
            persistence_alert=bool(data.get("persistence_alert", False)),
            anomaly_severity=sev_enum,
            summary_message=data.get("summary_message", ""),
            metadata=meta,
        )

    @classmethod
    def from_json(cls, json_str: str) -> "MonitoringOutput":
        return cls.from_dict(json.loads(json_str))


# -----------------------------------------------------------------------------
# 2. Diagnosis Agent Schemas
# -----------------------------------------------------------------------------
@dataclass
class FaultEvidence:
    """Specific sensor anomaly contributing to a diagnostic hypothesis."""
    sensor_name: str
    observed_value: float
    baseline_reference: float
    delta: float
    direction: str  # "HIGH", "LOW", "DRIFT", "OSCILLATING"
    description: str

    def validate(self) -> None:
        if not math.isfinite(self.observed_value):
            raise ValueError(f"FaultEvidence: observed_value must be finite ({self.observed_value}).")
        if not math.isfinite(self.delta):
            raise ValueError(f"FaultEvidence: delta must be finite ({self.delta}).")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sensor_name": self.sensor_name,
            "observed_value": float(self.observed_value),
            "baseline_reference": float(self.baseline_reference),
            "delta": float(self.delta),
            "direction": self.direction,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FaultEvidence":
        return cls(
            sensor_name=data["sensor_name"],
            observed_value=float(data["observed_value"]),
            baseline_reference=float(data["baseline_reference"]),
            delta=float(data["delta"]),
            direction=data["direction"],
            description=data["description"],
        )


@dataclass
class FaultCandidate:
    """Hypothesized fault mode with likelihood assessment."""
    fault_type: str  # e.g., "THERMAL_DISTRESS", "HPC_EFFICIENCY_DEGRADATION", "SENSOR_BIAS_DRIFT"
    component: str   # e.g., "HPC", "HPT", "COMBUSTOR", "FAN", "SENSOR"
    likelihood: float  # [0.0, 1.0]
    contributing_evidence: List[FaultEvidence] = field(default_factory=list)
    recommended_action: str = ""

    def validate(self) -> None:
        if not (0.0 <= self.likelihood <= 1.0):
            raise ValueError(f"FaultCandidate: likelihood must be in [0.0, 1.0], got {self.likelihood}.")
        for ev in self.contributing_evidence:
            ev.validate()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fault_type": self.fault_type,
            "component": self.component,
            "likelihood": float(self.likelihood),
            "contributing_evidence": [e.to_dict() for e in self.contributing_evidence],
            "recommended_action": self.recommended_action,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FaultCandidate":
        evidences = [FaultEvidence.from_dict(e) for e in data.get("contributing_evidence", [])]
        return cls(
            fault_type=data["fault_type"],
            component=data["component"],
            likelihood=float(data["likelihood"]),
            contributing_evidence=evidences,
            recommended_action=data.get("recommended_action", ""),
        )


@dataclass
class DiagnosisInput:
    """Input payload delivered to the Diagnosis Agent."""
    unit_number: int
    time_cycles: int
    monitoring_output: MonitoringOutput
    telemetry_snapshot: Dict[str, float]
    flight_regime: Optional[int] = None
    operating_context: Dict[str, float] = field(default_factory=dict)
    historical_persistence_count: int = 0

    def validate(self) -> None:
        if self.unit_number <= 0:
            raise ValueError(f"DiagnosisInput: unit_number must be > 0 ({self.unit_number}).")
        if self.time_cycles <= 0:
            raise ValueError(f"DiagnosisInput: time_cycles must be > 0 ({self.time_cycles}).")
        self.monitoring_output.validate()
        for k, v in self.telemetry_snapshot.items():
            if not math.isfinite(v):
                raise ValueError(f"DiagnosisInput: non-finite telemetry in snapshot for {k}: {v}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "unit_number": int(self.unit_number),
            "time_cycles": int(self.time_cycles),
            "monitoring_output": self.monitoring_output.to_dict(),
            "telemetry_snapshot": {k: float(v) for k, v in self.telemetry_snapshot.items()},
            "flight_regime": self.flight_regime,
            "operating_context": {k: float(v) for k, v in self.operating_context.items()},
            "historical_persistence_count": int(self.historical_persistence_count),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DiagnosisInput":
        mon_out = MonitoringOutput.from_dict(data["monitoring_output"])
        return cls(
            unit_number=int(data["unit_number"]),
            time_cycles=int(data["time_cycles"]),
            monitoring_output=mon_out,
            telemetry_snapshot=dict(data["telemetry_snapshot"]),
            flight_regime=data.get("flight_regime"),
            operating_context=dict(data.get("operating_context", {})),
            historical_persistence_count=int(data.get("historical_persistence_count", 0)),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "DiagnosisInput":
        return cls.from_dict(json.loads(json_str))


@dataclass
class DiagnosisOutput:
    """Diagnostic root-cause hypothesis produced by the Diagnosis Agent."""
    unit_number: int
    time_cycles: int
    diagnostic_hypotheses: List[str] = field(default_factory=list)
    fault_candidates: List[FaultCandidate] = field(default_factory=list)
    primary_fault: Optional[str] = None
    confidence_score: float = 0.0  # [0.0, 1.0] - Strictly bounded
    contributing_evidence: List[FaultEvidence] = field(default_factory=list)
    severity: SeverityLevel = SeverityLevel.NOMINAL
    requires_maintenance_action: bool = False
    diagnostic_summary: str = ""
    metadata: Optional[AgentMetadata] = None

    def validate(self) -> None:
        if self.unit_number <= 0:
            raise ValueError(f"DiagnosisOutput: unit_number must be > 0 ({self.unit_number}).")
        if self.time_cycles <= 0:
            raise ValueError(f"DiagnosisOutput: time_cycles must be > 0 ({self.time_cycles}).")
        if not (0.0 <= self.confidence_score <= 1.0):
            raise ValueError(f"DiagnosisOutput: confidence_score must be in [0.0, 1.0], got {self.confidence_score}.")
        for fc in self.fault_candidates:
            fc.validate()
        for ev in self.contributing_evidence:
            ev.validate()
        if self.metadata is not None:
            self.metadata.validate()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "unit_number": int(self.unit_number),
            "time_cycles": int(self.time_cycles),
            "diagnostic_hypotheses": list(self.diagnostic_hypotheses),
            "fault_candidates": [c.to_dict() for c in self.fault_candidates],
            "primary_fault": self.primary_fault,
            "confidence_score": float(self.confidence_score),
            "contributing_evidence": [e.to_dict() for e in self.contributing_evidence],
            "severity": self.severity.value,
            "requires_maintenance_action": bool(self.requires_maintenance_action),
            "diagnostic_summary": self.diagnostic_summary,
            "metadata": self.metadata.to_dict() if self.metadata else None,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DiagnosisOutput":
        sev = data["severity"]
        sev_enum = SeverityLevel(sev) if isinstance(sev, str) else sev
        candidates = [FaultCandidate.from_dict(c) for c in data.get("fault_candidates", [])]
        evidences = [FaultEvidence.from_dict(e) for e in data.get("contributing_evidence", [])]
        meta = AgentMetadata.from_dict(data["metadata"]) if data.get("metadata") else None

        return cls(
            unit_number=int(data["unit_number"]),
            time_cycles=int(data["time_cycles"]),
            diagnostic_hypotheses=list(data.get("diagnostic_hypotheses", [])),
            fault_candidates=candidates,
            primary_fault=data.get("primary_fault"),
            confidence_score=float(data["confidence_score"]),
            contributing_evidence=evidences,
            severity=sev_enum,
            requires_maintenance_action=bool(data.get("requires_maintenance_action", False)),
            diagnostic_summary=data.get("diagnostic_summary", ""),
            metadata=meta,
        )

    @classmethod
    def from_json(cls, json_str: str) -> "DiagnosisOutput":
        return cls.from_dict(json.loads(json_str))


# -----------------------------------------------------------------------------
# 3. Prognostics Agent Schemas
# -----------------------------------------------------------------------------
@dataclass
class PrognosticsInput:
    """Input payload delivered to the Prognostics Agent."""
    unit_number: int
    time_cycles: int
    subset: str
    normalized_feature_vec: List[float]       # G30 (14-dim or 24-dim)
    temporal_seq_tensor: List[List[float]]     # G31 (30 x 14 or 30 x 24)
    feature_tier: str = "Tier1_Core14"
    is_padded: bool = False

    def validate(self) -> None:
        if self.unit_number <= 0:
            raise ValueError(f"PrognosticsInput: unit_number must be > 0 ({self.unit_number}).")
        if self.time_cycles <= 0:
            raise ValueError(f"PrognosticsInput: time_cycles must be > 0 ({self.time_cycles}).")
        if len(self.normalized_feature_vec) not in (14, 24):
            raise ValueError(f"PrognosticsInput: normalized_feature_vec must have 14 or 24 elements, got {len(self.normalized_feature_vec)}.")
        if len(self.temporal_seq_tensor) != 30:
            raise ValueError(f"PrognosticsInput: temporal_seq_tensor must have 30 time steps, got {len(self.temporal_seq_tensor)}.")
        for step in self.temporal_seq_tensor:
            if len(step) not in (14, 24):
                raise ValueError(f"PrognosticsInput: temporal_seq_tensor channels must be 14 or 24, got {len(step)}.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "unit_number": int(self.unit_number),
            "time_cycles": int(self.time_cycles),
            "subset": self.subset.upper(),
            "normalized_feature_vec": [float(x) for x in self.normalized_feature_vec],
            "temporal_seq_tensor": [[float(x) for x in row] for row in self.temporal_seq_tensor],
            "feature_tier": self.feature_tier,
            "is_padded": bool(self.is_padded),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PrognosticsInput":
        vec = data["normalized_feature_vec"]
        if isinstance(vec, np.ndarray):
            vec = vec.tolist()
        tensor = data["temporal_seq_tensor"]
        if isinstance(tensor, np.ndarray):
            tensor = tensor.tolist()

        return cls(
            unit_number=int(data["unit_number"]),
            time_cycles=int(data["time_cycles"]),
            subset=data["subset"].upper(),
            normalized_feature_vec=vec,
            temporal_seq_tensor=tensor,
            feature_tier=data.get("feature_tier", "Tier1_Core14"),
            is_padded=bool(data.get("is_padded", False)),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "PrognosticsInput":
        return cls.from_dict(json.loads(json_str))


@dataclass
class PrognosticsOutput:
    """Prognostic RUL and health index forecast produced by the Prognostics Agent."""
    unit_number: int
    time_cycles: int
    subset: str
    predicted_rul_xgboost: float     # G35 (cycles)
    predicted_rul_lstm: float        # G36 (cycles)
    ensemble_rul_estimate: float    # Combined forecast (cycles)
    prognostic_spread: float        # G37 = |G35 - G36| (epistemic uncertainty)
    virtual_health_index: float     # G38 in [0.0, 1.0]
    uncertainty_classification: str # "LOW_UNCERTAINTY", "MODERATE_UNCERTAINTY", "HIGH_UNCERTAINTY"
    cycles_to_caution_threshold: Optional[float] = None
    prognostic_summary: str = ""
    metadata: Optional[AgentMetadata] = None

    def validate(self) -> None:
        if self.unit_number <= 0:
            raise ValueError(f"PrognosticsOutput: unit_number must be > 0 ({self.unit_number}).")
        if self.time_cycles <= 0:
            raise ValueError(f"PrognosticsOutput: time_cycles must be > 0 ({self.time_cycles}).")
        if self.predicted_rul_xgboost < 0.0 or not math.isfinite(self.predicted_rul_xgboost):
            raise ValueError(f"PrognosticsOutput: predicted_rul_xgboost must be non-negative finite ({self.predicted_rul_xgboost}).")
        if self.predicted_rul_lstm < 0.0 or not math.isfinite(self.predicted_rul_lstm):
            raise ValueError(f"PrognosticsOutput: predicted_rul_lstm must be non-negative finite ({self.predicted_rul_lstm}).")
        if self.ensemble_rul_estimate < 0.0 or not math.isfinite(self.ensemble_rul_estimate):
            raise ValueError(f"PrognosticsOutput: ensemble_rul_estimate must be non-negative finite ({self.ensemble_rul_estimate}).")
        if self.prognostic_spread < 0.0 or not math.isfinite(self.prognostic_spread):
            raise ValueError(f"PrognosticsOutput: prognostic_spread must be non-negative finite ({self.prognostic_spread}).")
        if not (0.0 <= self.virtual_health_index <= 1.0) or not math.isfinite(self.virtual_health_index):
            raise ValueError(f"PrognosticsOutput: virtual_health_index must be in [0.0, 1.0], got {self.virtual_health_index}.")
        if self.metadata is not None:
            self.metadata.validate()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "unit_number": int(self.unit_number),
            "time_cycles": int(self.time_cycles),
            "subset": self.subset.upper(),
            "predicted_rul_xgboost": float(self.predicted_rul_xgboost),
            "predicted_rul_lstm": float(self.predicted_rul_lstm),
            "ensemble_rul_estimate": float(self.ensemble_rul_estimate),
            "prognostic_spread": float(self.prognostic_spread),
            "virtual_health_index": float(self.virtual_health_index),
            "uncertainty_classification": self.uncertainty_classification,
            "cycles_to_caution_threshold": float(self.cycles_to_caution_threshold) if self.cycles_to_caution_threshold is not None else None,
            "prognostic_summary": self.prognostic_summary,
            "metadata": self.metadata.to_dict() if self.metadata else None,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PrognosticsOutput":
        meta = AgentMetadata.from_dict(data["metadata"]) if data.get("metadata") else None
        caution = data.get("cycles_to_caution_threshold")
        return cls(
            unit_number=int(data["unit_number"]),
            time_cycles=int(data["time_cycles"]),
            subset=data["subset"].upper(),
            predicted_rul_xgboost=float(data["predicted_rul_xgboost"]),
            predicted_rul_lstm=float(data["predicted_rul_lstm"]),
            ensemble_rul_estimate=float(data["ensemble_rul_estimate"]),
            prognostic_spread=float(data["prognostic_spread"]),
            virtual_health_index=float(data["virtual_health_index"]),
            uncertainty_classification=data.get("uncertainty_classification", "LOW_UNCERTAINTY"),
            cycles_to_caution_threshold=float(caution) if caution is not None else None,
            prognostic_summary=data.get("prognostic_summary", ""),
            metadata=meta,
        )

    @classmethod
    def from_json(cls, json_str: str) -> "PrognosticsOutput":
        return cls.from_dict(json.loads(json_str))


# -----------------------------------------------------------------------------
# 4. Orchestration Schemas
# -----------------------------------------------------------------------------
@dataclass
class OrchestrationRequest:
    """Request payload sent to the Multi-Agent Orchestrator."""
    engine_id: int
    flight_cycle: int
    subset: str = "FD001"
    execution_mode: str = "full"  # "full", "monitoring_only", "prognostics_only", "dry_run"
    enable_what_if: bool = False
    what_if_scenario_id: Optional[str] = None

    def validate(self) -> None:
        if self.engine_id <= 0:
            raise ValueError(f"OrchestrationRequest: engine_id must be > 0 ({self.engine_id}).")
        if self.flight_cycle <= 0:
            raise ValueError(f"OrchestrationRequest: flight_cycle must be > 0 ({self.flight_cycle}).")
        valid_subsets = ("FD001", "FD002", "FD003", "FD004")
        if self.subset.upper() not in valid_subsets:
            raise ValueError(f"OrchestrationRequest: subset must be in {valid_subsets}, got {self.subset}.")
        valid_modes = ("full", "monitoring_only", "prognostics_only", "dry_run")
        if self.execution_mode not in valid_modes:
            raise ValueError(f"OrchestrationRequest: execution_mode must be in {valid_modes}, got {self.execution_mode}.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "engine_id": int(self.engine_id),
            "flight_cycle": int(self.flight_cycle),
            "subset": self.subset.upper(),
            "execution_mode": self.execution_mode,
            "enable_what_if": bool(self.enable_what_if),
            "what_if_scenario_id": self.what_if_scenario_id,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OrchestrationRequest":
        return cls(
            engine_id=int(data["engine_id"]),
            flight_cycle=int(data["flight_cycle"]),
            subset=data.get("subset", "FD001").upper(),
            execution_mode=data.get("execution_mode", "full"),
            enable_what_if=bool(data.get("enable_what_if", False)),
            what_if_scenario_id=data.get("what_if_scenario_id"),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "OrchestrationRequest":
        return cls.from_dict(json.loads(json_str))


@dataclass
class OrchestrationResult:
    """Comprehensive multi-agent pipeline result produced by the Orchestrator."""
    run_id: str
    engine_id: int
    cycle: int
    subset: str
    status: AgentExecutionStatus
    monitoring_result: Optional[MonitoringOutput] = None
    diagnosis_result: Optional[DiagnosisOutput] = None
    prognostics_result: Optional[PrognosticsOutput] = None
    execution_timeline_ms: Dict[str, float] = field(default_factory=dict)
    metadata: Optional[AgentMetadata] = None
    enriched_state: Optional[Any] = None

    def validate(self) -> None:
        if not self.run_id:
            raise ValueError("OrchestrationResult: run_id cannot be empty.")
        if self.engine_id <= 0:
            raise ValueError(f"OrchestrationResult: engine_id must be > 0 ({self.engine_id}).")
        if self.cycle <= 0:
            raise ValueError(f"OrchestrationResult: cycle must be > 0 ({self.cycle}).")
        if self.monitoring_result is not None:
            self.monitoring_result.validate()
        if self.diagnosis_result is not None:
            self.diagnosis_result.validate()
        if self.prognostics_result is not None:
            self.prognostics_result.validate()
        if self.metadata is not None:
            self.metadata.validate()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "engine_id": int(self.engine_id),
            "cycle": int(self.cycle),
            "subset": self.subset.upper(),
            "status": self.status.value,
            "monitoring_result": self.monitoring_result.to_dict() if self.monitoring_result else None,
            "diagnosis_result": self.diagnosis_result.to_dict() if self.diagnosis_result else None,
            "prognostics_result": self.prognostics_result.to_dict() if self.prognostics_result else None,
            "execution_timeline_ms": {k: float(v) for k, v in self.execution_timeline_ms.items()},
            "metadata": self.metadata.to_dict() if self.metadata else None,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OrchestrationResult":
        st = data["status"]
        st_enum = AgentExecutionStatus(st) if isinstance(st, str) else st
        mon = MonitoringOutput.from_dict(data["monitoring_result"]) if data.get("monitoring_result") else None
        diag = DiagnosisOutput.from_dict(data["diagnosis_result"]) if data.get("diagnosis_result") else None
        prog = PrognosticsOutput.from_dict(data["prognostics_result"]) if data.get("prognostics_result") else None
        meta = AgentMetadata.from_dict(data["metadata"]) if data.get("metadata") else None

        return cls(
            run_id=data["run_id"],
            engine_id=int(data["engine_id"]),
            cycle=int(data["cycle"]),
            subset=data["subset"].upper(),
            status=st_enum,
            monitoring_result=mon,
            diagnosis_result=diag,
            prognostics_result=prog,
            execution_timeline_ms=dict(data.get("execution_timeline_ms", {})),
            metadata=meta,
        )

    @classmethod
    def from_json(cls, json_str: str) -> "OrchestrationResult":
        return cls.from_dict(json.loads(json_str))
