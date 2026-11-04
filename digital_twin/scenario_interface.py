"""
ScenarioInterface Module - Controlled What-If Simulation Scenarios
Provides reproducible, validated what-if scenario configurations for the Digital Twin.

Roadmap Reference: Phase 5 - Step 5.7
Scope: Academic simulation interface for hypothetical scenario analysis.
Does NOT modify raw datasets, Phase 4 models, or physical hardware telemetry.
"""

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Dict, Any, Optional, Tuple, List, Union
import numpy as np

# The canonical 24 scaler feature ordering established in Phase 3
ORDERED_SCALER_FEATURES: List[str] = [
    "op_setting_1", "op_setting_2", "op_setting_3",
    "sensor_1", "sensor_2", "sensor_3", "sensor_4", "sensor_5",
    "sensor_6", "sensor_7", "sensor_8", "sensor_9", "sensor_10",
    "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
    "sensor_16", "sensor_17", "sensor_18", "sensor_19", "sensor_20", "sensor_21",
]


class ScenarioCategory(str, Enum):
    """Supported what-if scenario categories."""
    THERMAL_STRESS = "thermal_stress"
    SENSOR_BIAS = "sensor_bias"
    ACCELERATED_AGING = "accelerated_aging"


# Channel mappings between thermodynamic symbols and C-MAPSS feature columns
TARGET_TO_FEATURE_MAP: Dict[str, str] = {
    # Operating settings
    "op_setting_1": "op_setting_1",
    "altitude": "op_setting_1",
    "op_setting_2": "op_setting_2",
    "mach_number": "op_setting_2",
    "op_setting_3": "op_setting_3",
    "TRA": "op_setting_3",
    "throttle_resolver_angle": "op_setting_3",
    # Gas-path temperatures
    "sensor_1": "sensor_1",
    "T2": "sensor_1",
    "sensor_2": "sensor_2",
    "T24": "sensor_2",
    "sensor_3": "sensor_3",
    "T30": "sensor_3",
    "sensor_4": "sensor_4",
    "T50": "sensor_4",
    # Pressures
    "sensor_5": "sensor_5",
    "P2": "sensor_5",
    "sensor_6": "sensor_6",
    "P15": "sensor_6",
    "sensor_7": "sensor_7",
    "P30": "sensor_7",
    "sensor_11": "sensor_11",
    "Ps30": "sensor_11",
    # Rotational speeds
    "sensor_8": "sensor_8",
    "Nf": "sensor_8",
    "sensor_9": "sensor_9",
    "Nc": "sensor_9",
    "sensor_13": "sensor_13",
    "NRf": "sensor_13",
    "sensor_14": "sensor_14",
    "NRc": "sensor_14",
    "sensor_18": "sensor_18",
    "Nf_dmd": "sensor_18",
    # Ratios, bleed flows, and demands
    "sensor_10": "sensor_10",
    "epr": "sensor_10",
    "sensor_12": "sensor_12",
    "phi": "sensor_12",
    "sensor_15": "sensor_15",
    "BPR": "sensor_15",
    "sensor_16": "sensor_16",
    "farB": "sensor_16",
    "sensor_17": "sensor_17",
    "htBleed": "sensor_17",
    "sensor_19": "sensor_19",
    "PCNfR_dmd": "sensor_19",
    "sensor_20": "sensor_20",
    "W31": "sensor_20",
    "sensor_21": "sensor_21",
    "W32": "sensor_21",
}

# Permitted parameters, units, and validation ranges for Thermal Stress
THERMAL_STRESS_TARGETS: Dict[str, Tuple[str, Tuple[float, float]]] = {
    "T2": ("°R", (-100.0, 200.0)),
    "T24": ("°R", (-100.0, 200.0)),
    "T30": ("°R", (-100.0, 200.0)),
    "T50": ("°R", (-100.0, 200.0)),
    "sensor_1": ("°R", (-100.0, 200.0)),
    "sensor_2": ("°R", (-100.0, 200.0)),
    "sensor_3": ("°R", (-100.0, 200.0)),
    "sensor_4": ("°R", (-100.0, 200.0)),
}

# Permitted parameters, units, and validation ranges for Sensor Bias
SENSOR_BIAS_LIMITS: Dict[str, Tuple[List[str], Tuple[float, float]]] = {
    # Temperatures (°R)
    "T2": (["°R", "Rankine", "degR"], (-100.0, 200.0)),
    "T24": (["°R", "Rankine", "degR"], (-100.0, 200.0)),
    "T30": (["°R", "Rankine", "degR"], (-100.0, 200.0)),
    "T50": (["°R", "Rankine", "degR"], (-100.0, 200.0)),
    "sensor_1": (["°R", "Rankine", "degR"], (-100.0, 200.0)),
    "sensor_2": (["°R", "Rankine", "degR"], (-100.0, 200.0)),
    "sensor_3": (["°R", "Rankine", "degR"], (-100.0, 200.0)),
    "sensor_4": (["°R", "Rankine", "degR"], (-100.0, 200.0)),
    # Pressures (psia)
    "P2": (["psia"], (-20.0, 50.0)),
    "P15": (["psia"], (-20.0, 50.0)),
    "P30": (["psia"], (-50.0, 100.0)),
    "Ps30": (["psia"], (-50.0, 100.0)),
    "sensor_5": (["psia"], (-20.0, 50.0)),
    "sensor_6": (["psia"], (-20.0, 50.0)),
    "sensor_7": (["psia"], (-50.0, 100.0)),
    "sensor_11": (["psia"], (-50.0, 100.0)),
    # Rotational speeds (rpm)
    "Nf": (["rpm"], (-500.0, 500.0)),
    "Nc": (["rpm"], (-500.0, 500.0)),
    "NRf": (["rpm"], (-500.0, 500.0)),
    "NRc": (["rpm"], (-500.0, 500.0)),
    "Nf_dmd": (["rpm"], (-500.0, 500.0)),
    "sensor_8": (["rpm"], (-500.0, 500.0)),
    "sensor_9": (["rpm"], (-500.0, 500.0)),
    "sensor_13": (["rpm"], (-500.0, 500.0)),
    "sensor_14": (["rpm"], (-500.0, 500.0)),
    "sensor_18": (["rpm"], (-500.0, 500.0)),
    # Operating settings
    "op_setting_1": (["kft"], (-10.0, 10.0)),
    "altitude": (["kft"], (-10.0, 10.0)),
    "op_setting_2": (["Mach", "dimensionless", ""], (-0.2, 0.2)),
    "mach_number": (["Mach", "dimensionless", ""], (-0.2, 0.2)),
    "op_setting_3": (["%", "percent"], (-20.0, 20.0)),
    "TRA": (["%", "percent"], (-20.0, 20.0)),
    "throttle_resolver_angle": (["%", "percent"], (-20.0, 20.0)),
    # Ratios, Demands, Bleeds
    "epr": (["dimensionless", ""], (-0.5, 0.5)),
    "sensor_10": (["dimensionless", ""], (-0.5, 0.5)),
    "phi": (["pps/psia"], (-100.0, 100.0)),
    "sensor_12": (["pps/psia"], (-100.0, 100.0)),
    "BPR": (["dimensionless", ""], (-2.0, 2.0)),
    "sensor_15": (["dimensionless", ""], (-2.0, 2.0)),
    "farB": (["dimensionless", ""], (-0.01, 0.01)),
    "sensor_16": (["dimensionless", ""], (-0.01, 0.01)),
    "htBleed": (["dimensionless", ""], (-50.0, 50.0)),
    "sensor_17": (["dimensionless", ""], (-50.0, 50.0)),
    "PCNfR_dmd": (["%", "percent"], (-20.0, 20.0)),
    "sensor_19": (["%", "percent"], (-20.0, 20.0)),
    "W31": (["lbm/s"], (-10.0, 10.0)),
    "sensor_20": (["lbm/s"], (-10.0, 10.0)),
    "W32": (["lbm/s"], (-10.0, 10.0)),
    "sensor_21": (["lbm/s"], (-10.0, 10.0)),
}

# Accelerated aging parameter limits
ACCELERATED_AGING_LIMITS: Tuple[float, float] = (1.0, 3.0)


@dataclass
class ScenarioConfig:
    """
    Validated serializable dataclass defining a controlled what-if simulation scenario.
    Captures target parameter, modification value, units, validation bounds, and metadata.
    Enforces strict canonical bounds, numerical finiteness, and simulation-only designation.
    """
    scenario_id: str
    category: Union[ScenarioCategory, str]
    name: str
    target_parameter: str
    modification_value: float
    unit: str
    validation_range: Tuple[float, float]
    description: str
    is_simulation_only: bool = True
    baseline_value: Optional[float] = None
    simulated_value: Optional[float] = None

    def __post_init__(self):
        # 1. Enforce simulation-only designation
        if type(self.is_simulation_only) is not bool:
            raise ValueError(
                f"is_simulation_only must be an actual Python boolean (True), got {type(self.is_simulation_only).__name__} ({self.is_simulation_only!r}). "
                "Strings, integers, and non-boolean representations are strictly rejected."
            )
        if self.is_simulation_only is not True:
            raise ValueError(
                f"is_simulation_only must be True. Got {self.is_simulation_only}. "
                "Operational hardware control or live injection is strictly prohibited."
            )

        # 2. Normalize category
        if isinstance(self.category, str):
            try:
                self.category = ScenarioCategory(self.category.lower())
            except ValueError:
                raise ValueError(
                    f"Unsupported scenario category '{self.category}'. "
                    f"Must be one of {[c.value for c in ScenarioCategory]}."
                )

        # 3. Validate configuration integrity immediately upon initialization
        ScenarioValidator.validate_config(self)

    def to_dict(self) -> Dict[str, Any]:
        """Convert scenario configuration to serializable dictionary."""
        return {
            "scenario_id": self.scenario_id,
            "category": self.category.value if isinstance(self.category, ScenarioCategory) else str(self.category),
            "name": self.name,
            "target_parameter": self.target_parameter,
            "modification_value": float(self.modification_value),
            "unit": self.unit,
            "validation_range": [float(self.validation_range[0]), float(self.validation_range[1])],
            "description": self.description,
            "is_simulation_only": bool(self.is_simulation_only),
            "baseline_value": float(self.baseline_value) if self.baseline_value is not None else None,
            "simulated_value": float(self.simulated_value) if self.simulated_value is not None else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScenarioConfig":
        """Instantiate ScenarioConfig from dictionary representation."""
        required_keys = [
            "scenario_id", "category", "name", "target_parameter",
            "modification_value", "unit", "validation_range", "description"
        ]
        for key in required_keys:
            if key not in data:
                raise KeyError(f"Missing required ScenarioConfig field: '{key}'.")

        v_range = data["validation_range"]
        if not (isinstance(v_range, (list, tuple)) and len(v_range) == 2):
            raise ValueError(f"validation_range must be a 2-element tuple/list, got {v_range}.")

        # Strict boolean validation for is_simulation_only
        if "is_simulation_only" in data:
            sim_flag = data["is_simulation_only"]
            if type(sim_flag) is not bool:
                raise ValueError(
                    f"is_simulation_only must be an actual Python boolean (True), got {type(sim_flag).__name__} ({sim_flag!r}). "
                    "String representations ('false', 'true', '0', 'yes'), integers (0, 1), and non-boolean values are strictly rejected."
                )
            if sim_flag is not True:
                raise ValueError(
                    "is_simulation_only must be True. Operational hardware control or live injection is strictly prohibited."
                )
        else:
            sim_flag = True

        return cls(
            scenario_id=str(data["scenario_id"]),
            category=str(data["category"]),
            name=str(data["name"]),
            target_parameter=str(data["target_parameter"]),
            modification_value=float(data["modification_value"]),
            unit=str(data["unit"]),
            validation_range=(float(v_range[0]), float(v_range[1])),
            description=str(data["description"]),
            is_simulation_only=sim_flag,
            baseline_value=float(data["baseline_value"]) if data.get("baseline_value") is not None else None,
            simulated_value=float(data["simulated_value"]) if data.get("simulated_value") is not None else None,
        )


class ScenarioValidator:
    """
    Validates scenario configurations, parameters, bounds, and thermodynamic constraints.
    Enforces strict safety: rejects NaN, +/-inf, unphysical values, and unsupported categories.
    Never silently clips values.
    """

    @staticmethod
    def validate_config(config: ScenarioConfig) -> None:
        """
        Validate all fields of a ScenarioConfig.
        Enforces canonical bounds, strict finiteness, simulation-only flag, and thermodynamic safety.
        Rejects any attempt to widen bounds through custom validation ranges.
        """
        # 1. Enforce simulation-only designation
        if type(config.is_simulation_only) is not bool:
            raise ValueError(
                f"is_simulation_only must be an actual Python boolean True, got {type(config.is_simulation_only).__name__} ({config.is_simulation_only!r}). "
                "String representations, integers, and non-boolean values are strictly rejected."
            )
        if config.is_simulation_only is not True:
            raise ValueError(
                f"is_simulation_only must be True, got {config.is_simulation_only}. "
                "Hardware control or non-simulation execution is strictly prohibited."
            )

        # 2. String identifiers
        if not config.scenario_id or not isinstance(config.scenario_id, str):
            raise ValueError("scenario_id must be a non-empty string.")
        if not config.name or not isinstance(config.name, str):
            raise ValueError("name must be a non-empty string.")
        if not config.target_parameter or not isinstance(config.target_parameter, str):
            raise ValueError("target_parameter must be a non-empty string.")

        # 3. Finiteness of numerical modification value
        val = config.modification_value
        if not isinstance(val, (int, float)) or not math.isfinite(val):
            raise ValueError(f"modification_value must be a finite numeric value, got {val}.")

        # 4. Optional baseline_value and simulated_value finiteness checks
        if config.baseline_value is not None:
            if not isinstance(config.baseline_value, (int, float)) or not math.isfinite(config.baseline_value):
                raise ValueError(f"baseline_value must be a finite numeric value, got {config.baseline_value}.")
        if config.simulated_value is not None:
            if not isinstance(config.simulated_value, (int, float)) or not math.isfinite(config.simulated_value):
                raise ValueError(f"simulated_value must be a finite numeric value, got {config.simulated_value}.")

        # 5. Validation range integrity
        v_low, v_high = config.validation_range
        if not (math.isfinite(v_low) and math.isfinite(v_high)):
            raise ValueError(f"validation_range values must be finite, got ({v_low}, {v_high}).")
        if v_low > v_high:
            raise ValueError(f"Invalid validation_range: lower bound {v_low} > upper bound {v_high}.")

        # 6. Category-specific canonical bound enforcement & target validation
        category = config.category
        target = config.target_parameter

        if category == ScenarioCategory.THERMAL_STRESS:
            if target not in THERMAL_STRESS_TARGETS:
                raise ValueError(
                    f"Invalid target '{target}' for Thermal Stress. "
                    f"Must be one of {sorted(list(THERMAL_STRESS_TARGETS.keys()))}."
                )
            expected_unit, canonical_bounds = THERMAL_STRESS_TARGETS[target]

            # Reject custom validation range that deviates from canonical range
            if tuple(config.validation_range) != canonical_bounds:
                raise ValueError(
                    f"Custom validation_range {config.validation_range} is prohibited for Thermal Stress target '{target}'. "
                    f"Must match canonical range {canonical_bounds}."
                )

            if config.unit not in ["°R", "Rankine", "degR"]:
                raise ValueError(f"Invalid unit '{config.unit}' for Thermal Stress. Expected '{expected_unit}'.")

            if val < canonical_bounds[0] or val > canonical_bounds[1]:
                raise ValueError(
                    f"modification_value {val} for parameter '{target}' is outside "
                    f"canonical validation range [{canonical_bounds[0]}, {canonical_bounds[1]}]."
                )

        elif category == ScenarioCategory.SENSOR_BIAS:
            if target not in SENSOR_BIAS_LIMITS:
                raise ValueError(
                    f"Invalid target '{target}' for Sensor Bias. "
                    f"Must be a valid C-MAPSS channel or physical alias."
                )
            valid_units, canonical_bounds = SENSOR_BIAS_LIMITS[target]

            # Reject custom validation range that deviates from canonical range
            if tuple(config.validation_range) != canonical_bounds:
                raise ValueError(
                    f"Custom validation_range {config.validation_range} is prohibited for Sensor Bias target '{target}'. "
                    f"Must match canonical range {canonical_bounds}."
                )

            if config.unit not in valid_units and valid_units != [""]:
                raise ValueError(
                    f"Invalid unit '{config.unit}' for target '{target}'. Expected one of {valid_units}."
                )

            if val < canonical_bounds[0] or val > canonical_bounds[1]:
                raise ValueError(
                    f"modification_value {val} for parameter '{target}' is outside "
                    f"canonical validation range [{canonical_bounds[0]}, {canonical_bounds[1]}]."
                )

        elif category == ScenarioCategory.ACCELERATED_AGING:
            valid_targets = ["aging_multiplier", "cycle_aging_rate", "cycle_progression_multiplier"]
            if target not in valid_targets:
                raise ValueError(
                    f"Invalid target '{target}' for Accelerated Aging. "
                    f"Must be one of {valid_targets}."
                )
            canonical_bounds = ACCELERATED_AGING_LIMITS

            # Reject custom validation range that deviates from canonical range
            if tuple(config.validation_range) != canonical_bounds:
                raise ValueError(
                    f"Custom validation_range {config.validation_range} is prohibited for Accelerated Aging. "
                    f"Must match canonical range {canonical_bounds}."
                )

            if val < canonical_bounds[0] or val > canonical_bounds[1]:
                raise ValueError(
                    f"modification_value {val} for parameter '{target}' is outside "
                    f"canonical validation range [{canonical_bounds[0]}, {canonical_bounds[1]}]."
                )

        else:
            raise ValueError(f"Unsupported scenario category '{category}'.")

    @staticmethod
    def validate_simulated_telemetry(channel_name: str, baseline_val: float, sim_val: float) -> None:
        """
        Verify that resulting simulated telemetry satisfies physical constraints.
        Temperatures, pressures, speeds, flows, and ratios must remain strictly positive.
        """
        if not math.isfinite(sim_val):
            raise ValueError(f"Simulated telemetry for '{channel_name}' produced non-finite value: {sim_val}.")

        # Positive-definite physical variables
        # sensor_1..4 (temps), sensor_5..7 (pressures), sensor_8..9 (speeds),
        # sensor_10..15 (epr, Ps30, phi, NRf, NRc, BPR), sensor_16..21 (farB, htBleed, Nf_dmd, PCNfR_dmd, W31, W32)
        if channel_name.startswith("sensor_") or channel_name in [
            "T2", "T24", "T30", "T50", "P2", "P15", "P30", "Ps30", "Nf", "Nc",
            "epr", "phi", "NRf", "NRc", "BPR", "farB", "htBleed", "Nf_dmd", "PCNfR_dmd", "W31", "W32"
        ]:
            if sim_val <= 0.0:
                raise ValueError(
                    f"Simulated value {sim_val} for physical channel '{channel_name}' violates "
                    f"strictly-positive physical boundary (baseline was {baseline_val})."
                )


class ScenarioModifier:
    """
    Applies validated scenario modifications to raw telemetry without mutating baseline data.
    Maintains clean separation between measured telemetry, simulated inputs, and ML inference.
    """

    @staticmethod
    def resolve_feature_column(target: str) -> str:
        """Resolve a physical parameter alias or sensor name to an ORDERED_SCALER_FEATURES column."""
        if target in TARGET_TO_FEATURE_MAP:
            return TARGET_TO_FEATURE_MAP[target]
        if target in ORDERED_SCALER_FEATURES:
            return target
        raise ValueError(f"Cannot resolve target '{target}' to a recognized C-MAPSS feature column.")

    @staticmethod
    def apply_scenario(
        raw_24: np.ndarray,
        config: ScenarioConfig
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply a scenario perturbation to a 24-element raw telemetry vector.
        
        Args:
            raw_24: Original 24-element raw telemetry array (G3 - G26).
            config: Validated ScenarioConfig.
            
        Returns:
            Tuple of:
                sim_raw_24: New 24-element array containing the modified simulation inputs.
                audit_record: Metadata documenting baseline, modification, and simulated values.
        """
        if not isinstance(raw_24, np.ndarray) or raw_24.shape != (24,):
            raise ValueError(f"raw_24 must be a 1D numpy array of shape (24,), got {type(raw_24)} shape {getattr(raw_24, 'shape', None)}.")

        # Re-validate config safeguards
        ScenarioValidator.validate_config(config)

        # Create an isolated copy to guarantee the original telemetry is untouched
        sim_raw_24 = np.copy(raw_24)

        if config.category in [ScenarioCategory.THERMAL_STRESS, ScenarioCategory.SENSOR_BIAS]:
            col_name = ScenarioModifier.resolve_feature_column(config.target_parameter)
            col_idx = ORDERED_SCALER_FEATURES.index(col_name)

            baseline_val = float(raw_24[col_idx])
            sim_val = baseline_val + float(config.modification_value)

            # Validate physical feasibility of the simulated telemetry
            ScenarioValidator.validate_simulated_telemetry(col_name, baseline_val, sim_val)

            sim_raw_24[col_idx] = sim_val

            audit_record = {
                "scenario_id": config.scenario_id,
                "category": config.category.value,
                "target_parameter": config.target_parameter,
                "feature_column": col_name,
                "channel_index": col_idx,
                "baseline_value": baseline_val,
                "original_baseline_value": baseline_val,
                "applied_modification": float(config.modification_value),
                "applied_delta": float(config.modification_value),
                "simulated_value": sim_val,
                "resulting_simulated_value": sim_val,
                "unit": config.unit,
                "is_simulation_only": True,
            }

        elif config.category == ScenarioCategory.ACCELERATED_AGING:
            # Accelerated aging models an operational cycle degradation progression multiplier.
            # Physical execution in simulator is formally deferred to prevent temporal gaps in ML inference.
            audit_record = {
                "scenario_id": config.scenario_id,
                "category": config.category.value,
                "target_parameter": config.target_parameter,
                "baseline_value": 1.0,
                "original_baseline_value": 1.0,
                "applied_modification": float(config.modification_value),
                "applied_delta": float(config.modification_value) - 1.0,
                "simulated_value": float(config.modification_value),
                "resulting_simulated_value": float(config.modification_value),
                "aging_multiplier": float(config.modification_value),
                "unit": config.unit,
                "is_simulation_only": True,
                "internal_wear_laws_deferred": True,
                "limitation_note": (
                    "Accelerated aging execution is formally deferred. Temporal subsampling of recorded "
                    "C-MAPSS cycles produces invalid discontinuous sequences for Phase 4 ML prognostics (LSTM/XGBoost). "
                    "Internal physical component degradation modifiers are unobservable in C-MAPSS."
                ),
            }

        else:
            raise ValueError(f"Unhandled scenario category: '{config.category}'.")

        return sim_raw_24, audit_record


# -----------------------------------------------------------------------------
# Convenience Factory Functions
# -----------------------------------------------------------------------------
def create_thermal_stress_scenario(
    scenario_id: str,
    target: str,
    offset_degR: float,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> ScenarioConfig:
    """Create and validate a Thermal Stress scenario."""
    bounds = (-100.0, 200.0)
    if target in THERMAL_STRESS_TARGETS:
        _, bounds = THERMAL_STRESS_TARGETS[target]

    desc = description or f"Hypothetical thermal stress offset of {offset_degR:+.1f} °R applied to {target}."
    sc_name = name or f"Thermal Stress ({target} {offset_degR:+.1f} °R)"

    return ScenarioConfig(
        scenario_id=scenario_id,
        category=ScenarioCategory.THERMAL_STRESS,
        name=sc_name,
        target_parameter=target,
        modification_value=offset_degR,
        unit="°R",
        validation_range=bounds,
        description=desc,
        is_simulation_only=True,
    )


def create_sensor_bias_scenario(
    scenario_id: str,
    target: str,
    bias: float,
    unit: str,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> ScenarioConfig:
    """Create and validate a Sensor Bias scenario."""
    if target not in SENSOR_BIAS_LIMITS:
        raise ValueError(f"Target '{target}' not in known sensor bias limits dictionary.")

    _, bounds = SENSOR_BIAS_LIMITS[target]
    desc = description or f"Hypothetical sensor bias of {bias:+.2f} {unit} applied to {target}."
    sc_name = name or f"Sensor Bias ({target} {bias:+.2f} {unit})"

    return ScenarioConfig(
        scenario_id=scenario_id,
        category=ScenarioCategory.SENSOR_BIAS,
        name=sc_name,
        target_parameter=target,
        modification_value=bias,
        unit=unit,
        validation_range=bounds,
        description=desc,
        is_simulation_only=True,
    )


def create_accelerated_aging_scenario(
    scenario_id: str,
    aging_multiplier: float,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> ScenarioConfig:
    """
    Create and validate an Accelerated Aging operational scenario.
    Degradation rate multiplier in [1.0, 3.0].
    Microscopic component wear equations are formally deferred.
    """
    desc = description or (
        f"Hypothetical accelerated operational aging scenario with progression multiplier {aging_multiplier:.2f}x. "
        f"Internal component wear laws are formally deferred."
    )
    sc_name = name or f"Accelerated Aging ({aging_multiplier:.2f}x)"

    return ScenarioConfig(
        scenario_id=scenario_id,
        category=ScenarioCategory.ACCELERATED_AGING,
        name=sc_name,
        target_parameter="aging_multiplier",
        modification_value=aging_multiplier,
        unit="multiplier",
        validation_range=ACCELERATED_AGING_LIMITS,
        description=desc,
        is_simulation_only=True,
    )
