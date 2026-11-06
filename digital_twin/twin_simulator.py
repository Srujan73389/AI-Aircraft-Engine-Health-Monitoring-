"""
DigitalTwinSimulator Class - Aircraft Engine Digital Twin Time-Step Advancement
Advances virtual engine state cycle-by-cycle using recorded C-MAPSS trajectories.

Roadmap Reference: Phase 5 - Steps 5.1, 5.2, 5.3, 5.4
"""

import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from digital_twin.engine_state import EngineState, StateClassification
from digital_twin.trajectory_loader import TrajectoryLoader, C_MAPSS_COLUMNS
from digital_twin.model_inference import PrognosticsModelManager
from digital_twin.scenario_interface import (
    ScenarioConfig,
    ScenarioCategory,
    ScenarioModifier,
    ScenarioValidator,
)
from preprocessing.scaling_pipeline import CMAPSSScalingPipeline


ORDERED_SCALER_FEATURES = [
    "op_setting_1", "op_setting_2", "op_setting_3",
    "sensor_1", "sensor_2", "sensor_3", "sensor_4", "sensor_5",
    "sensor_6", "sensor_7", "sensor_8", "sensor_9", "sensor_10",
    "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
    "sensor_16", "sensor_17", "sensor_18", "sensor_19", "sensor_20", "sensor_21",
]

TIER1_CORE_SENSORS = [
    "sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_8",
    "sensor_9", "sensor_11", "sensor_12", "sensor_13", "sensor_14",
    "sensor_15", "sensor_17", "sensor_20", "sensor_21"
]

CORE_INDICES_IN_24 = [ORDERED_SCALER_FEATURES.index(s) for s in TIER1_CORE_SENSORS]


class DigitalTwinSimulator:
    """
    Virtual Engine Digital Twin time-step advancement simulator.
    Loads engine trajectories, maintains a 30-cycle rolling history, applies left-edge
    replication padding, transforms telemetry through Phase 3 scalers, and constructs
    the 40-variable virtual state vector S(t) at each discrete flight cycle.
    """

    def __init__(
        self,
        subset: str = "FD001",
        data_dir: str = os.path.join("dataset", "raw", "C-MAPSS"),
        scalers_dir: str = os.path.join("models", "scalers"),
        checkpoint_dir: str = os.path.join("models", "checkpoints"),
        window_size: int = 30,
        enable_ml_inference: bool = False,
        scenario: Optional[ScenarioConfig] = None,
    ):
        self.subset = subset.upper()
        if self.subset not in ["FD001", "FD002", "FD003", "FD004"]:
            raise ValueError(f"Invalid subset '{subset}'. Must be one of ['FD001', 'FD002', 'FD003', 'FD004'].")

        self.window_size = window_size
        self.data_dir = data_dir
        self.scalers_dir = scalers_dir
        self.checkpoint_dir = checkpoint_dir
        self.enable_ml_inference = enable_ml_inference
        self.loader = TrajectoryLoader(data_dir=self.data_dir)

        # What-if scenario state
        self.active_scenario: Optional[ScenarioConfig] = None
        self.current_scenario_audit: Optional[Dict[str, Any]] = None
        self.scenario_audit_history: List[Dict[str, Any]] = []
        if scenario is not None:
            self.set_scenario(scenario)

        # Feature tier assignment per roadmap and Phase 4 model interfaces:
        # FD001 uses Tier 1 Core 14; FD002, FD003, FD004 use Tier 3 Extended 24.
        self.feature_tier = "Tier1_Core14" if self.subset == "FD001" else "Tier3_Extended24"

        # Load subset-specific scaler fitted exclusively on training partition (Phase 3 Step 3.8)
        scaler_filename = f"scaler_{self.subset}_tier3.pkl"
        scaler_path = os.path.join(self.scalers_dir, scaler_filename)
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Required Phase 3 scaler not found at: {scaler_path}")

        self.scaling_pipeline = CMAPSSScalingPipeline.load(scaler_path)

        # Prognostics Model Manager (Phase 5 Step 5.6)
        self.prognostics_manager: Optional[PrognosticsModelManager] = None
        if self.enable_ml_inference:
            self.prognostics_manager = PrognosticsModelManager(
                subset=self.subset, checkpoint_dir=self.checkpoint_dir
            )

        # Internal simulation buffers
        self.active_unit: Optional[int] = None
        self.active_split: Optional[str] = None
        self.trajectory_df: Optional[pd.DataFrame] = None
        self.current_step_idx: int = -1
        self.total_cycles: int = 0

        # Rolling history buffers
        self.raw_history: List[np.ndarray] = []  # List of 24-dim raw feature arrays
        self.norm_history: List[np.ndarray] = []  # List of 24-dim normalized feature arrays
        self.state_history: List[EngineState] = []

    def load_engine(self, unit_number: int, split: str = "train") -> EngineState:
        """
        Load an engine trajectory and initialize its virtual state at cycle 1.
        Returns the initial EngineState.
        """
        self.active_unit = unit_number
        self.active_split = split.lower()
        self.trajectory_df = self.loader.load_engine_trajectory(
            subset=self.subset, unit_number=unit_number, split=self.active_split
        )
        self.total_cycles = len(self.trajectory_df)
        self.current_step_idx = -1

        # Clear history buffers
        self.raw_history = []
        self.norm_history = []
        self.state_history = []
        self.current_scenario_audit = None
        self.scenario_audit_history = []

        # Advance to cycle 1 (initial state)
        initial_state = self.step()
        if initial_state is None:
            raise RuntimeError(f"Failed to initialize virtual state for engine {unit_number}.")
        return initial_state

    def set_scenario(self, scenario: Optional[ScenarioConfig]) -> None:
        """
        Configure or update the active what-if simulation scenario.
        Enforces strict validation before assignment.
        Rejects accelerated aging execution to prevent invalid discontinuous sequence inputs to ML models.
        """
        if scenario is not None:
            ScenarioValidator.validate_config(scenario)
            if scenario.category == ScenarioCategory.ACCELERATED_AGING:
                raise NotImplementedError(
                    "Accelerated aging execution is formally deferred. Temporal subsampling of recorded "
                    "C-MAPSS cycles produces invalid discontinuous sequences for Phase 4 ML models "
                    "(LSTM/XGBoost), and physical component wear laws are unobservable in C-MAPSS telemetry."
                )
        self.active_scenario = scenario

    def clear_scenario(self) -> None:
        """Clear active what-if simulation scenario, restoring baseline simulation."""
        self.active_scenario = None
        self.current_scenario_audit = None

    def step(self, scenario: Optional[ScenarioConfig] = None) -> Optional[EngineState]:
        """
        Advance the virtual engine by one discrete operational cycle.
        Applies active or provided what-if scenario perturbation if configured.
        Strictly preserves consecutive-cycle sequence progression (no cycle skipping).
        Returns the updated EngineState, or None if the trajectory has completed.
        """
        if self.trajectory_df is None:
            raise RuntimeError("No engine trajectory loaded. Call load_engine() first.")

        # Determine effective scenario for this step
        eff_scenario = scenario if scenario is not None else self.active_scenario
        if eff_scenario is not None:
            ScenarioValidator.validate_config(eff_scenario)
            if eff_scenario.category == ScenarioCategory.ACCELERATED_AGING:
                raise NotImplementedError(
                    "Accelerated aging execution is formally deferred. Temporal subsampling of recorded "
                    "C-MAPSS cycles produces invalid discontinuous sequences for Phase 4 ML models "
                    "(LSTM/XGBoost), and physical component wear laws are unobservable in C-MAPSS telemetry."
                )

        # Strictly advance consecutive recorded cycles (zero temporal gaps)
        next_idx = self.current_step_idx + 1
        if next_idx >= self.total_cycles:
            return None  # Trajectory complete

        self.current_step_idx = next_idx
        row = self.trajectory_df.iloc[self.current_step_idx]

        # Extract metadata
        unit_id = int(row["unit_number"])
        cycle_idx = int(row["time_cycles"])

        # Extract 24 physical telemetry values (G3 - G26) - Unmodified ground truth
        raw_24 = np.array([float(row[c]) for c in ORDERED_SCALER_FEATURES], dtype=np.float64)

        # Baseline normalization via Phase 3 SafeMinMaxScaler
        baseline_norm_24 = self.scaling_pipeline.scaler.transform(raw_24.reshape(1, -1))[0]

        # Always preserve unmodified recorded ground truth in raw history
        self.raw_history.append(raw_24)

        # Determine simulation inputs under what-if scenario
        thermal_stress_offset: Optional[float] = None
        sensor_bias_delta: Optional[Dict[str, float]] = None
        scenario_id: Optional[str] = None
        scenario_audit: Optional[Dict[str, Any]] = None

        if eff_scenario is None:
            norm_24 = baseline_norm_24
            self.current_scenario_audit = None
        else:
            scenario_id = eff_scenario.scenario_id
            sim_raw_24, audit = ScenarioModifier.apply_scenario(raw_24, eff_scenario)
            norm_24 = self.scaling_pipeline.scaler.transform(sim_raw_24.reshape(1, -1))[0]
            scenario_audit = audit
            self.current_scenario_audit = audit
            self.scenario_audit_history.append(audit)

            if eff_scenario.category == ScenarioCategory.THERMAL_STRESS:
                thermal_stress_offset = float(eff_scenario.modification_value)
            elif eff_scenario.category == ScenarioCategory.SENSOR_BIAS:
                sensor_bias_delta = {str(eff_scenario.target_parameter): float(eff_scenario.modification_value)}

        self.norm_history.append(norm_24)

        # Derived consistency indicators (G27 - G29)
        # G27: T30 - T24 (HPC temp rise)
        hpc_temp_rise = float(row["sensor_3"] - row["sensor_2"])
        # G28: P30 / P2 (Overall Compression Ratio)
        p2 = float(row["sensor_5"])
        p30 = float(row["sensor_7"])
        overall_pressure_ratio = float(p30 / p2) if p2 > 0 else 0.0
        # G29: Nc / Nf (Spool speed ratio)
        nf = float(row["sensor_8"])
        nc = float(row["sensor_9"])
        spool_speed_ratio = float(nc / nf) if nf > 0 else 0.0

        # Construct tier-aware representations G30 and G31
        current_history_len = len(self.norm_history)
        history_matrix = np.array(self.norm_history, dtype=np.float64)  # Shape: (current_len, 24)

        # Apply left-edge replication padding if history < window_size (Phase 3 protocol)
        if current_history_len < self.window_size:
            pad_len = self.window_size - current_history_len
            pad_prefix = np.repeat(history_matrix[:1], pad_len, axis=0)
            window_matrix_24 = np.vstack([pad_prefix, history_matrix])  # Shape: (30, 24)
            is_padded = True
            padded_count = pad_len
        else:
            window_matrix_24 = history_matrix[-self.window_size:]  # Shape: (30, 24)
            is_padded = False
            padded_count = 0

        # Subset-specific feature slicing for G30 and G31
        if self.feature_tier == "Tier1_Core14":
            # FD001 slices the 14 core prognostic sensors
            g30_vector = norm_24[CORE_INDICES_IN_24]  # Shape: (14,)
            g31_tensor = window_matrix_24[:, CORE_INDICES_IN_24]  # Shape: (30, 14)
        else:
            # FD002, FD003, FD004 retain the full 24 features
            g30_vector = norm_24  # Shape: (24,)
            g31_tensor = window_matrix_24  # Shape: (30, 24)

        # Assemble EngineState (G1 - G40)
        state = EngineState(
            # G1 - G2
            unit_number=unit_id,
            time_cycles=cycle_idx,
            # G3 - G5
            altitude=float(row["op_setting_1"]),
            mach_number=float(row["op_setting_2"]),
            throttle_resolver_angle=float(row["op_setting_3"]),
            # G6 - G26
            T2=float(row["sensor_1"]),
            T24=float(row["sensor_2"]),
            T30=float(row["sensor_3"]),
            T50=float(row["sensor_4"]),
            P2=float(row["sensor_5"]),
            P15=float(row["sensor_6"]),
            P30=float(row["sensor_7"]),
            Nf=float(row["sensor_8"]),
            Nc=float(row["sensor_9"]),
            epr=float(row["sensor_10"]),
            Ps30=float(row["sensor_11"]),
            phi=float(row["sensor_12"]),
            NRf=float(row["sensor_13"]),
            NRc=float(row["sensor_14"]),
            BPR=float(row["sensor_15"]),
            farB=float(row["sensor_16"]),
            htBleed=float(row["sensor_17"]),
            Nf_dmd=float(row["sensor_18"]),
            PCNfR_dmd=float(row["sensor_19"]),
            W31=float(row["sensor_20"]),
            W32=float(row["sensor_21"]),
            # G27 - G29
            hpc_temp_rise=hpc_temp_rise,
            overall_pressure_ratio=overall_pressure_ratio,
            spool_speed_ratio=spool_speed_ratio,
            # G30 - G31
            normalized_feature_vec=g30_vector,
            temporal_seq_tensor=g31_tensor,
            # G32 - G38: ML predictions and health index remain unpopulated (Step 5.6)
            anomaly_score_m1=None,
            anomaly_score_m2=None,
            anomaly_alert_flag=None,
            predicted_rul_xgboost=None,
            predicted_rul_lstm=None,
            prognostic_spread=None,
            virtual_health_index=None,
            # G39 - G40: What-if parameters (Step 5.7)
            thermal_stress_offset=thermal_stress_offset,
            sensor_bias_delta=sensor_bias_delta,
            # Provenance & Audit
            subset=self.subset,
            feature_tier=self.feature_tier,
            is_padded=is_padded,
            padded_cycles_count=padded_count,
            history_window_length=current_history_len,
            scenario_id=scenario_id,
            scenario_audit=scenario_audit,
        )

        # Step 5.6: If prognostics inference is enabled, compute G35, G36, and G37
        if self.enable_ml_inference and self.prognostics_manager is not None:
            state = self.prognostics_manager.update_state_prognostics(state)

        self.state_history.append(state)
        return state

    def enable_prognostics(self) -> None:
        """Enables machine-learning model inference for downstream steps."""
        if self.prognostics_manager is None:
            self.prognostics_manager = PrognosticsModelManager(
                subset=self.subset, checkpoint_dir=self.checkpoint_dir
            )
        self.enable_ml_inference = True

    def run_all(self, scenario: Optional[ScenarioConfig] = None) -> List[EngineState]:
        """Advance through all remaining cycles of the loaded trajectory."""
        while self.step(scenario=scenario) is not None:
            pass
        return self.state_history

    @property
    def current_state(self) -> Optional[EngineState]:
        """Return the most recently computed EngineState."""
        return self.state_history[-1] if self.state_history else None

    def get_history_summary_df(self) -> pd.DataFrame:
        """Export tabular summary of all advanced cycles."""
        if not self.state_history:
            return pd.DataFrame()
        records = [s.to_dict(include_arrays=False) for s in self.state_history]
        return pd.DataFrame(records)

    def apply_safe_action(
        self,
        action: Any,
        governor: Optional[Any] = None,
        operator_approval: bool = False,
    ) -> Any:
        """
        Executes a closed-loop safe simulation action on the digital twin.
        Gated by the Phase 7 ClosedLoopSafetyGovernor.
        """
        from safety.closed_loop import ClosedLoopSafetyGovernor
        gov = governor or ClosedLoopSafetyGovernor()
        return gov.apply_simulation_action(
            simulator_or_state=self,
            action=action,
            operator_approval=operator_approval,
        )
