"""
Aircraft Engine Digital Twin Module
NASA C-MAPSS Turbofan Engine Health Monitoring & Simulation
"""

from digital_twin.engine_state import EngineState, StateClassification, STATE_TAXONOMY
from digital_twin.trajectory_loader import TrajectoryLoader
from digital_twin.twin_simulator import DigitalTwinSimulator
from digital_twin.physics_engine import (
    PhysicsIndicators,
    compute_compression_isentropic_efficiency,
    compute_compression_polytropic_efficiency,
    compute_compression_specific_work,
    compute_spool_speed_ratios,
    compute_aerothermal_speed_corrections,
    evaluate_engine_physics,
)
from digital_twin.model_inference import (
    PrognosticsModelManager,
    CHAMPION_MODEL_REGISTRY,
    compute_window_summary_statistics,
    extract_flattened_window,
)
from digital_twin.scenario_interface import (
    ScenarioCategory,
    ScenarioConfig,
    ScenarioValidator,
    ScenarioModifier,
    create_thermal_stress_scenario,
    create_sensor_bias_scenario,
    create_accelerated_aging_scenario,
    THERMAL_STRESS_TARGETS,
    SENSOR_BIAS_LIMITS,
    ACCELERATED_AGING_LIMITS,
)
from digital_twin.scenario_comparison import (
    ScenarioComparisonRunner,
    ScenarioComparisonResult,
)
from digital_twin.recorded_data_comparison import (
    RecordedDataComparator,
    TelemetrySummary,
    DerivedIndicatorsSummary,
    PreprocessingSummary,
    RULPrognosticsSummary,
)
from digital_twin.stability_validator import (
    TwinStabilityValidator,
    PhysicalRangeCheckResult,
    LongTrajectoryStabilityResult,
    RepeatabilityResult,
    EdgeCaseResilienceResult,
    THERMODYNAMIC_BOUNDS,
)
from digital_twin.verify_digital_twin_contracts import (
    DigitalTwinContractVerifier,
    TaxonomyContractResult,
    UnperturbedDefaultsResult,
    ProvenanceLabelingResult,
    ScenarioPerturbationContractResult,
)

__all__ = [
    "EngineState",
    "StateClassification",
    "STATE_TAXONOMY",
    "TrajectoryLoader",
    "DigitalTwinSimulator",
    "PhysicsIndicators",
    "compute_compression_isentropic_efficiency",
    "compute_compression_polytropic_efficiency",
    "compute_compression_specific_work",
    "compute_spool_speed_ratios",
    "compute_aerothermal_speed_corrections",
    "evaluate_engine_physics",
    "PrognosticsModelManager",
    "CHAMPION_MODEL_REGISTRY",
    "compute_window_summary_statistics",
    "extract_flattened_window",
    "ScenarioCategory",
    "ScenarioConfig",
    "ScenarioValidator",
    "ScenarioModifier",
    "create_thermal_stress_scenario",
    "create_sensor_bias_scenario",
    "create_accelerated_aging_scenario",
    "THERMAL_STRESS_TARGETS",
    "SENSOR_BIAS_LIMITS",
    "ACCELERATED_AGING_LIMITS",
    "ScenarioComparisonRunner",
    "ScenarioComparisonResult",
    "RecordedDataComparator",
    "TelemetrySummary",
    "DerivedIndicatorsSummary",
    "PreprocessingSummary",
    "RULPrognosticsSummary",
    "TwinStabilityValidator",
    "PhysicalRangeCheckResult",
    "LongTrajectoryStabilityResult",
    "RepeatabilityResult",
    "EdgeCaseResilienceResult",
    "THERMODYNAMIC_BOUNDS",
    "DigitalTwinContractVerifier",
    "TaxonomyContractResult",
    "UnperturbedDefaultsResult",
    "ProvenanceLabelingResult",
    "ScenarioPerturbationContractResult",
]
