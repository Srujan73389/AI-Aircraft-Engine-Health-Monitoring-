"""
Pydantic API Schemas for Aircraft Engine AI REST Service
Phase 8 - Step 8.2: Endpoints for fleet selection, monitoring, RUL prediction,
simulation what-if, and safety verification.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from safety.disclaimers import ACADEMIC_SIMULATION_DISCLAIMER


# =============================================================================
# 1. Fleet & Engine Schemas
# =============================================================================

class FleetSubsetInfo(BaseModel):
    subset: str = Field(..., description="C-MAPSS subset identifier ('FD001' to 'FD004')")
    operating_conditions: str = Field(..., description="Environmental regime mode (Single Sea-Level or Six Regimes)")
    fault_modes: List[str] = Field(..., description="Simulated hardware degradation modes")
    training_engines_count: int = Field(..., description="Total training run-to-failure engine units")
    test_engines_count: int = Field(..., description="Total validation/test engine units")
    description: str = Field(..., description="Detailed aerothermal simulation description")


class EngineSummary(BaseModel):
    subset: str = Field(..., description="C-MAPSS subset")
    unit_number: int = Field(..., description="Discrete engine unit ID")
    max_cycles: int = Field(..., description="Total recorded trajectory cycles")
    operating_conditions: str = Field(default="Sea Level", description="Operating context summary")


class EngineCycleState(BaseModel):
    subset: str = Field(..., description="C-MAPSS subset")
    unit_number: int = Field(..., description="Engine unit number")
    cycle: int = Field(..., description="Current operational flight cycle")
    state_vector: Dict[str, Any] = Field(..., description="Complete 40-variable state vector S(t) (G1 to G40)")
    disclaimer: str = Field(default=ACADEMIC_SIMULATION_DISCLAIMER, description="Academic simulation non-authority disclaimer")


# =============================================================================
# 2. Monitoring & Anomaly Detection Schemas
# =============================================================================

class SensorTrendsResponse(BaseModel):
    subset: str
    unit_number: int
    total_cycles: int
    cycle_history: List[int]
    sensor_names: List[str]
    trends: Dict[str, List[float]]
    disclaimer: str = Field(default=ACADEMIC_SIMULATION_DISCLAIMER)


class AnomalyCheckRequest(BaseModel):
    subset: str = Field(default="FD001", description="C-MAPSS subset")
    unit_number: Optional[int] = Field(default=None, description="Optional engine ID for recorded lookup")
    cycle: Optional[int] = Field(default=None, description="Optional flight cycle for recorded lookup")
    sensor_readings: Optional[Dict[str, Any]] = Field(default=None, description="Current cycle sensor telemetry map")
    anomaly_scores: Optional[Dict[str, Any]] = Field(default=None, description="Pre-computed anomaly scores")


class AnomalyCheckResponse(BaseModel):
    subset: str
    is_anomaly: bool
    isolation_forest_breached: bool
    pca_spe_breached: bool
    isolation_forest_score: Optional[float] = None
    pca_spe_score: Optional[float] = None
    thresholds: Dict[str, float]
    severity: str  # "NOMINAL", "CAUTION", "CRITICAL"
    summary: str
    consecutive_breaches: int = Field(default=0, description="Consecutive cycle anomaly breach counter")
    persistence_alarm: bool = Field(default=False, description="3-cycle sustained anomaly persistence alarm")
    disclaimer: str = Field(default=ACADEMIC_SIMULATION_DISCLAIMER)


# =============================================================================
# 3. Prognostics & RUL Prediction Schemas
# =============================================================================

class RULPredictionRequest(BaseModel):
    subset: str = Field(default="FD001", description="C-MAPSS subset identifier")
    unit_number: Optional[int] = Field(default=None, description="Optional engine ID to predict from trajectory")
    cycle: Optional[int] = Field(default=None, description="Operational cycle index (defaults to latest)")
    sequence_tensor: Optional[List[List[float]]] = Field(
        default=None,
        description="Direct (30, D) temporal sequence matrix (14 channels for FD001, 24 for FD002-FD004)"
    )
    features_sequence: Optional[List[List[float]]] = Field(
        default=None,
        description="Direct (30, D) temporal sequence matrix alias"
    )
    features: Optional[Dict[str, float]] = Field(
        default=None,
        description="Optional single-cycle feature dictionary"
    )


class RULPredictionResponse(BaseModel):
    subset: str
    unit_number: Optional[int] = None
    cycle: Optional[int] = None
    predicted_rul_xgboost: float
    predicted_rul_lstm: float
    prognostic_spread: float
    consensus_rul: float
    confidence_score: float
    disclaimer: str = Field(default=ACADEMIC_SIMULATION_DISCLAIMER)


# =============================================================================
# 4. Simulation What-If Schemas
# =============================================================================

class SimulationWhatIfRequest(BaseModel):
    subset: str = Field(default="FD001", description="C-MAPSS subset")
    unit_number: int = Field(default=1, description="Engine unit ID")
    cycle: int = Field(default=1, description="Cycle to perturb from")
    action_type: str = Field(..., description="Simulation action category")
    target_variable: str = Field(..., description="Target variable name or G-index")
    proposed_value: Any = Field(..., description="Hypothetical value to inject")
    delta_value: Optional[float] = Field(default=None, description="Optional relative delta")
    parameters: Optional[Dict[str, Any]] = Field(default=None, description="Action parameter payload")
    operator_approval: bool = Field(default=False, description="Manual operator override signoff for reviewable states")


class SimulationWhatIfResponse(BaseModel):
    action_id: str
    decision_status: str  # "ACCEPTED", "REJECTED", "REQUIRES_REVIEW"
    update_applied: bool
    operator_approved: bool
    state_before: Dict[str, Any]
    state_after: Dict[str, Any]
    reason_summary: str
    recovery_plan: Optional[Dict[str, Any]] = None
    disclaimer: str = Field(default=ACADEMIC_SIMULATION_DISCLAIMER)


# =============================================================================
# 5. Safety Verification & Governance Schemas
# =============================================================================

class SafetyVerificationRequest(BaseModel):
    action_id: Optional[str] = Field(default=None, description="Optional unique action identifier")
    action_type: str = Field(default="THERMAL_PERTURBATION", description="Action type")
    target_variable: str = Field(default="thermal_stress_offset", description="Target state variable")
    proposed_value: Any = Field(default=4.0, description="Proposed value")
    delta_value: Optional[float] = Field(default=None, description="Proposed delta offset")
    current_state: Optional[Dict[str, Any]] = Field(default=None, description="State vector dictionary to evaluate against")
    subset: Optional[str] = Field(default="FD001", description="Engine subset for fallback baseline state")
    unit_number: Optional[int] = Field(default=1, description="Engine unit number")
    cycle: Optional[int] = Field(default=1, description="Flight cycle")


class SafetyVerificationResponse(BaseModel):
    action_id: str
    status: str  # "ACCEPTED", "REJECTED", "REQUIRES_REVIEW"
    is_safe: bool
    summary_message: str
    caution_count: int
    critical_count: int
    hard_limit_count: int
    violations: List[Dict[str, Any]]
    recovery_plan: Optional[Dict[str, Any]] = None
    disclaimer: str = Field(default=ACADEMIC_SIMULATION_DISCLAIMER)


class SafetyAuditTrailQuery(BaseModel):
    limit: int = Field(default=50, ge=1, le=500, description="Maximum records to return")
    decision_status: Optional[str] = Field(default=None, description="Filter by decision verdict")
    target_variable: Optional[str] = Field(default=None, description="Filter by target variable")


class SafetyAuditTrailResponse(BaseModel):
    total_records: int
    returned_records: int
    records: List[Dict[str, Any]]
    disclaimer: str = Field(default=ACADEMIC_SIMULATION_DISCLAIMER)


# =============================================================================
# 6. Defensive Error Response Schemas (Phase 8 Step 8.10)
# =============================================================================

class APIErrorDetail(BaseModel):
    loc: Optional[List[str]] = Field(default=None, description="Location of error in request")
    msg: str = Field(..., description="Error message")
    type: Optional[str] = Field(default=None, description="Error type identifier")


class APIErrorResponse(BaseModel):
    status_code: int = Field(..., description="HTTP status code")
    error_code: str = Field(..., description="Machine-readable error classification code (e.g. INVALID_ENGINE_ID, MISSING_INPUT, MODEL_UNAVAILABLE)")
    message: str = Field(..., description="High-level human-readable error summary")
    detail: Optional[Any] = Field(default=None, description="Detailed itemized errors or bounds context")
    suggestion: Optional[str] = Field(default=None, description="Actionable remediation guidance")
    disclaimer: str = Field(default=ACADEMIC_SIMULATION_DISCLAIMER, description="Academic simulation non-authority disclaimer")
