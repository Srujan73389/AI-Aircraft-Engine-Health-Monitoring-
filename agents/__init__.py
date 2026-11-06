"""
Multi-Agent Package - Phase 6
Modular multi-agent architecture for AI-based aircraft engine health monitoring,
fault diagnosis, and predictive maintenance.

Roadmap Reference: Phase 6 - Production Multi-Agent System
"""

from agents.schemas import (
    # Enums
    AgentExecutionStatus,
    SeverityLevel,
    OperationalState,
    # Common
    AgentMetadata,
    OrchestrationRequest,
    OrchestrationResult,
    # Monitoring
    ThresholdBreach,
    MonitoringInput,
    MonitoringOutput,
    # Diagnosis
    FaultEvidence,
    FaultCandidate,
    DiagnosisInput,
    DiagnosisOutput,
    # Prognostics
    PrognosticsInput,
    PrognosticsOutput,
)
from agents.monitoring_agent import MonitoringAgent
from agents.diagnosis_agent import DiagnosisAgent
from agents.prognostics_agent import PrognosticsAgent
from agents.orchestrator import AgentOrchestrator
from agents.audit_logger import AgentAuditLogger
from agents.twin_bridge import DigitalTwinSafetyBridge, SafetyViolationError
from agents.scenario_evaluator import MultiAgentScenarioEvaluator
from agents.failure_handler import AgentFailureHarness, AgentFailureReport

__all__ = [
    # Enums
    "AgentExecutionStatus",
    "SeverityLevel",
    "OperationalState",
    # Common
    "AgentOrchestrator",
    "AgentAuditLogger",
    "DigitalTwinSafetyBridge",
    "SafetyViolationError",
    "MultiAgentScenarioEvaluator",
    "AgentFailureHarness",
    "AgentFailureReport",
    "AgentMetadata",
    "OrchestrationRequest",
    "OrchestrationResult",
    # Monitoring
    "MonitoringAgent",
    "ThresholdBreach",
    "MonitoringInput",
    "MonitoringOutput",
    # Diagnosis
    "DiagnosisAgent",
    "FaultEvidence",
    "FaultCandidate",
    "DiagnosisInput",
    "DiagnosisOutput",
    # Prognostics
    "PrognosticsAgent",
    "PrognosticsInput",
    "PrognosticsOutput",
]
