"""
ScenarioComparison Module - Controlled Baseline vs Altered Scenario Execution
Executes and compares Digital Twin trajectories under identical initial conditions and recorded telemetry.

Roadmap Reference: Phase 5 - Step 5.8
Evaluates prognostic model sensitivity to supported what-if scenarios (Thermal Stress, Sensor Bias).
"""

from dataclasses import dataclass, field
import json
import math
import os
import tempfile
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), ".mplconfig")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from digital_twin.engine_state import EngineState
from digital_twin.scenario_interface import (
    ScenarioCategory,
    ScenarioConfig,
    ScenarioValidator,
    ScenarioModifier,
    ORDERED_SCALER_FEATURES,
)
from digital_twin.twin_simulator import DigitalTwinSimulator


@dataclass
class ScenarioComparisonResult:
    """Encapsulates the full execution and comparison results of a what-if experiment."""
    experiment_name: str
    subset: str
    unit_number: int
    split: str
    scenario_config: Optional[ScenarioConfig]
    total_cycles: int
    baseline_records: List[EngineState]
    altered_records: List[EngineState]
    comparison_df: pd.DataFrame
    summary_metrics: Dict[str, Any]
    fair_comparison_checks: Dict[str, bool]


class ScenarioComparisonRunner:
    """
    Executes controlled baseline-versus-altered what-if scenarios using identical initial conditions.
    Maintains two independent simulator instances, verifies fair comparison conditions,
    and quantifies prognostic sensitivity across discrete flight cycles.
    """

    SENSOR_FIELDS: List[str] = [
        "T2", "T24", "T30", "T50", "P2", "P15", "P30", "Nf", "Nc",
        "epr", "Ps30", "phi", "NRf", "NRc", "BPR", "farB", "htBleed",
        "Nf_dmd", "PCNfR_dmd", "W31", "W32"
    ]

    MACRO_FIELDS: List[str] = [
        "hpc_temp_rise", "overall_pressure_ratio", "spool_speed_ratio"
    ]

    def __init__(
        self,
        data_dir: str = os.path.join("dataset", "raw", "C-MAPSS"),
        scalers_dir: str = os.path.join("models", "scalers"),
        checkpoint_dir: str = os.path.join("models", "checkpoints"),
    ):
        self.data_dir = data_dir
        self.scalers_dir = scalers_dir
        self.checkpoint_dir = checkpoint_dir

    def run_comparison(
        self,
        subset: str = "FD001",
        unit_number: int = 1,
        split: str = "train",
        scenario: Optional[ScenarioConfig] = None,
        experiment_name: str = "Experiment",
        max_cycles: Optional[int] = None,
    ) -> ScenarioComparisonResult:
        """
        Execute an unperturbed baseline simulator alongside an altered scenario simulator
        across the identical recorded engine trajectory.
        """
        if scenario is not None:
            ScenarioValidator.validate_config(scenario)
            if scenario.category == ScenarioCategory.ACCELERATED_AGING:
                raise NotImplementedError(
                    "Accelerated aging execution is formally deferred. Temporal subsampling of recorded "
                    "C-MAPSS cycles produces invalid discontinuous sequences for Phase 4 ML models "
                    "(LSTM/XGBoost), and physical component wear laws are unobservable in C-MAPSS telemetry."
                )

        # 1. Initialize two independent simulators with ML prognostics enabled
        sim_baseline = DigitalTwinSimulator(
            subset=subset,
            data_dir=self.data_dir,
            scalers_dir=self.scalers_dir,
            checkpoint_dir=self.checkpoint_dir,
            enable_ml_inference=True,
            scenario=None,
        )

        sim_altered = DigitalTwinSimulator(
            subset=subset,
            data_dir=self.data_dir,
            scalers_dir=self.scalers_dir,
            checkpoint_dir=self.checkpoint_dir,
            enable_ml_inference=True,
            scenario=scenario,
        )

        # 2. Load the identical engine trajectory
        s_base_init = sim_baseline.load_engine(unit_number=unit_number, split=split)
        s_alt_init = sim_altered.load_engine(unit_number=unit_number, split=split)

        # 3. Verify identical starting conditions
        init_equal = (
            s_base_init.unit_number == s_alt_init.unit_number and
            s_base_init.time_cycles == s_alt_init.time_cycles and
            s_base_init.altitude == s_alt_init.altitude and
            s_base_init.mach_number == s_alt_init.mach_number and
            s_base_init.throttle_resolver_angle == s_alt_init.throttle_resolver_angle
        )
        for s_f in self.SENSOR_FIELDS:
            if getattr(s_base_init, s_f) != getattr(s_alt_init, s_f):
                init_equal = False
                break

        if not init_equal:
            raise RuntimeError("Fair comparison violation: Simulators did not start from identical conditions.")

        # Verification trackers
        all_telemetry_equal = True
        all_macro_equal = True
        all_baseline_fields_unset = True
        all_altered_fields_correct = True
        all_audit_verified = True
        all_outputs_finite = True

        baseline_cycle_ids: List[int] = []
        altered_cycle_ids: List[int] = []
        comparison_records: List[Dict[str, Any]] = []

        expected_cycles = max_cycles if max_cycles is not None else sim_baseline.total_cycles

        total_steps = 0
        s_base = s_base_init
        s_alt = s_alt_init

        while True:
            total_steps += 1
            cycle_idx = s_base.time_cycles
            baseline_cycle_ids.append(s_base.time_cycles)
            altered_cycle_ids.append(s_alt.time_cycles)

            # Check identical cycle indexing
            if s_base.time_cycles != s_alt.time_cycles:
                raise RuntimeError(
                    f"Cycle desynchronization: baseline cycle {s_base.time_cycles} != altered cycle {s_alt.time_cycles}."
                )

            # Check original measured telemetry is never overwritten
            for fld in self.SENSOR_FIELDS:
                if getattr(s_base, fld) != getattr(s_alt, fld):
                    all_telemetry_equal = False

            # Check macroscopic consistency indicators remain calculated from measured telemetry
            for fld in self.MACRO_FIELDS:
                if getattr(s_base, fld) != getattr(s_alt, fld):
                    all_macro_equal = False

            # Check baseline scenario fields remain unset
            if (s_base.thermal_stress_offset is not None or
                s_base.sensor_bias_delta is not None or
                s_base.scenario_id is not None or
                s_base.scenario_audit is not None):
                all_baseline_fields_unset = False

            # Check altered scenario fields and scenario audit
            if scenario is not None:
                if s_alt.scenario_id != scenario.scenario_id:
                    all_altered_fields_correct = False
                if s_alt.scenario_audit is None:
                    all_altered_fields_correct = False
                    all_audit_verified = False
                else:
                    audit = s_alt.scenario_audit
                    target_param = audit.get("target_parameter")
                    feat_col = audit.get("feature_column")
                    ch_idx = audit.get("channel_index")
                    base_val = audit.get("baseline_value")
                    app_mod = audit.get("applied_modification")
                    sim_val = audit.get("simulated_value")

                    expected_feat_col = ScenarioModifier.resolve_feature_column(scenario.target_parameter)
                    expected_ch_idx = ORDERED_SCALER_FEATURES.index(expected_feat_col)

                    if (target_param != scenario.target_parameter or
                        feat_col != expected_feat_col or
                        ch_idx != expected_ch_idx or
                        base_val is None or not math.isfinite(base_val) or
                        app_mod is None or not math.isclose(app_mod, scenario.modification_value, rel_tol=1e-7, abs_tol=1e-7) or
                        sim_val is None or not math.isfinite(sim_val) or
                        not math.isclose(sim_val, base_val + scenario.modification_value, rel_tol=1e-7, abs_tol=1e-7)):
                        all_audit_verified = False

                if scenario.category == ScenarioCategory.THERMAL_STRESS:
                    if s_alt.thermal_stress_offset != scenario.modification_value:
                        all_altered_fields_correct = False
                elif scenario.category == ScenarioCategory.SENSOR_BIAS:
                    if s_alt.sensor_bias_delta != {scenario.target_parameter: scenario.modification_value}:
                        all_altered_fields_correct = False

            # Extract ML predictions
            base_xgb = s_base.predicted_rul_xgboost
            alt_xgb = s_alt.predicted_rul_xgboost
            base_lstm = s_base.predicted_rul_lstm
            alt_lstm = s_alt.predicted_rul_lstm
            base_spread = s_base.prognostic_spread
            alt_spread = s_alt.prognostic_spread

            # Prediction deltas
            delta_xgb = (alt_xgb - base_xgb) if (alt_xgb is not None and base_xgb is not None) else float("nan")
            delta_lstm = (alt_lstm - base_lstm) if (alt_lstm is not None and base_lstm is not None) else float("nan")
            delta_spread = (alt_spread - base_spread) if (alt_spread is not None and base_spread is not None) else float("nan")

            # Check finiteness for predictions, spreads, and deltas
            for val in [base_xgb, alt_xgb, base_lstm, alt_lstm, base_spread, alt_spread, delta_xgb, delta_lstm, delta_spread]:
                if val is None or not math.isfinite(val):
                    all_outputs_finite = False

            # Check finiteness for normalized feature vectors (G30)
            if s_base.normalized_feature_vec is None or not np.all(np.isfinite(s_base.normalized_feature_vec)):
                all_outputs_finite = False
            if s_alt.normalized_feature_vec is None or not np.all(np.isfinite(s_alt.normalized_feature_vec)):
                all_outputs_finite = False

            # Check finiteness for temporal sequence tensors (G31)
            if s_base.temporal_seq_tensor is None or not np.all(np.isfinite(s_base.temporal_seq_tensor)):
                all_outputs_finite = False
            if s_alt.temporal_seq_tensor is None or not np.all(np.isfinite(s_alt.temporal_seq_tensor)):
                all_outputs_finite = False

            rec: Dict[str, Any] = {
                "cycle": cycle_idx,
                "unit_number": unit_number,
                "baseline_rul_xgb": base_xgb,
                "altered_rul_xgb": alt_xgb,
                "delta_rul_xgb": delta_xgb,
                "baseline_rul_lstm": base_lstm,
                "altered_rul_lstm": alt_lstm,
                "delta_rul_lstm": delta_lstm,
                "baseline_spread": base_spread,
                "altered_spread": alt_spread,
                "delta_spread": delta_spread,
                "target_parameter": scenario.target_parameter if scenario else None,
                "modification_value": scenario.modification_value if scenario else 0.0,
                "unit": scenario.unit if scenario else None,
            }

            if scenario and s_alt.scenario_audit:
                rec["baseline_val"] = s_alt.scenario_audit.get("original_baseline_value")
                rec["simulated_val"] = s_alt.scenario_audit.get("resulting_simulated_value")
            else:
                rec["baseline_val"] = None
                rec["simulated_val"] = None

            comparison_records.append(rec)

            if max_cycles is not None and total_steps >= max_cycles:
                break

            s_base_next = sim_baseline.step()
            s_alt_next = sim_altered.step()

            if s_base_next is None or s_alt_next is None:
                # Both should finish at the exact same cycle
                if s_base_next != s_alt_next:
                    raise RuntimeError("Trajectory termination mismatch between baseline and altered simulator.")
                break

            s_base = s_base_next
            s_alt = s_alt_next

        comparison_df = pd.DataFrame(comparison_records)

        # 4. Compute summary metrics
        delta_xgb_series = comparison_df["delta_rul_xgb"].dropna()
        delta_lstm_series = comparison_df["delta_rul_lstm"].dropna()
        delta_spread_series = comparison_df["delta_spread"].dropna()

        summary_metrics = {
            "experiment_name": experiment_name,
            "subset": subset,
            "unit_number": unit_number,
            "total_cycles_evaluated": len(comparison_df),
            "target_parameter": scenario.target_parameter if scenario else None,
            "category": scenario.category.value if scenario else "none",
            "modification_value": scenario.modification_value if scenario else 0.0,
            "unit": scenario.unit if scenario else None,
            "delta_rul_xgb": {
                "mean": float(delta_xgb_series.mean()),
                "std": float(delta_xgb_series.std()) if len(delta_xgb_series) > 1 else 0.0,
                "min": float(delta_xgb_series.min()),
                "max": float(delta_xgb_series.max()),
                "mean_abs_delta": float(delta_xgb_series.abs().mean()),
            },
            "delta_rul_lstm": {
                "mean": float(delta_lstm_series.mean()),
                "std": float(delta_lstm_series.std()) if len(delta_lstm_series) > 1 else 0.0,
                "min": float(delta_lstm_series.min()),
                "max": float(delta_lstm_series.max()),
                "mean_abs_delta": float(delta_lstm_series.abs().mean()),
            },
            "delta_spread": {
                "mean": float(delta_spread_series.mean()),
                "std": float(delta_spread_series.std()) if len(delta_spread_series) > 1 else 0.0,
                "min": float(delta_spread_series.min()),
                "max": float(delta_spread_series.max()),
            },
        }

        all_cycle_ids_match = (baseline_cycle_ids == altered_cycle_ids)
        all_cycles_evaluated = (
            len(comparison_records) == expected_cycles and
            len(sim_baseline.state_history) == expected_cycles and
            len(sim_altered.state_history) == expected_cycles
        )

        fair_checks = {
            "identical_initial_conditions": bool(init_equal),
            "identical_cycle_count": bool(len(sim_baseline.state_history) == len(sim_altered.state_history)),
            "identical_cycle_ids": bool(all_cycle_ids_match),
            "all_requested_cycles_evaluated": bool(all_cycles_evaluated),
            "unmodified_measured_telemetry": bool(all_telemetry_equal),
            "unmodified_macroscopic_indicators": bool(all_macro_equal),
            "baseline_scenario_fields_unset": bool(all_baseline_fields_unset),
            "altered_scenario_fields_populated": bool(all_altered_fields_correct),
            "scenario_audit_verified": bool(all_audit_verified),
            "all_outputs_finite": bool(all_outputs_finite),
        }

        failed_checks = [k for k, v in fair_checks.items() if not v]
        if failed_checks:
            raise RuntimeError(
                f"Fair comparison failure: One or more fairness checks failed: {failed_checks}. "
                f"Experiment cannot be treated as successful."
            )

        return ScenarioComparisonResult(
            experiment_name=experiment_name,
            subset=subset,
            unit_number=unit_number,
            split=split,
            scenario_config=scenario,
            total_cycles=len(comparison_df),
            baseline_records=sim_baseline.state_history,
            altered_records=sim_altered.state_history,
            comparison_df=comparison_df,
            summary_metrics=summary_metrics,
            fair_comparison_checks=fair_checks,
        )

    @staticmethod
    def export_artifacts(
        result: ScenarioComparisonResult,
        output_dir: str = os.path.join("reports", "digital_twin", "step5_8"),
        file_prefix: str = "comparison",
    ) -> Dict[str, str]:
        """Export tabular CSVs, JSON summary, and comparison figures to disk."""
        os.makedirs(output_dir, exist_ok=True)

        saved_files = {}

        # 1. Comparison DataFrame
        comp_path = os.path.join(output_dir, f"{file_prefix}.csv")
        result.comparison_df.to_csv(comp_path, index=False)
        saved_files["comparison_csv"] = comp_path

        # 2. Baseline summary DataFrame
        base_df = pd.DataFrame([s.to_dict(include_arrays=False) for s in result.baseline_records])
        base_path = os.path.join(output_dir, f"baseline_{file_prefix}.csv")
        base_df.to_csv(base_path, index=False)
        saved_files["baseline_csv"] = base_path

        # 3. Altered summary DataFrame
        alt_df = pd.DataFrame([s.to_dict(include_arrays=False) for s in result.altered_records])
        alt_path = os.path.join(output_dir, f"altered_{file_prefix}.csv")
        alt_df.to_csv(alt_path, index=False)
        saved_files["altered_csv"] = alt_path

        # 4. Summary metrics JSON
        json_path = os.path.join(output_dir, f"{file_prefix}_summary.json")
        summary_payload = {
            "summary_metrics": result.summary_metrics,
            "fair_comparison_checks": result.fair_comparison_checks,
            "scenario_config": result.scenario_config.to_dict() if result.scenario_config else None,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2)
        saved_files["summary_json"] = json_path

        # 5. High-resolution comparison plot
        plot_path = os.path.join(output_dir, f"{file_prefix}_plot.png")
        ScenarioComparisonRunner.generate_comparison_plot(result, plot_path)
        saved_files["plot_png"] = plot_path

        return saved_files

    @staticmethod
    def generate_comparison_plot(result: ScenarioComparisonResult, output_path: str) -> None:
        """Generate high-resolution 3-panel figure comparing baseline and altered predictions."""
        df = result.comparison_df
        cycles = df["cycle"]

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

        sc_name = result.scenario_config.name if result.scenario_config else "No Scenario"

        # Panel 1: XGBoost Predictions
        ax1.plot(cycles, df["baseline_rul_xgb"], label="Baseline XGBoost RUL", color="navy", linewidth=1.8)
        ax1.plot(cycles, df["altered_rul_xgb"], label=f"Altered XGBoost RUL ({sc_name})", color="crimson", linestyle="--", linewidth=1.8)
        ax1.set_ylabel("Predicted RUL (cycles)", fontsize=11)
        ax1.set_title(f"XGBoost Prognostics: Baseline vs. {result.experiment_name}", fontsize=12, fontweight="bold")
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(loc="upper right")

        # Panel 2: LSTM Predictions
        ax2.plot(cycles, df["baseline_rul_lstm"], label="Baseline LSTM RUL", color="teal", linewidth=1.8)
        ax2.plot(cycles, df["altered_rul_lstm"], label=f"Altered LSTM RUL ({sc_name})", color="darkorange", linestyle="--", linewidth=1.8)
        ax2.set_ylabel("Predicted RUL (cycles)", fontsize=11)
        ax2.set_title(f"LSTM Prognostics: Baseline vs. {result.experiment_name}", fontsize=12, fontweight="bold")
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend(loc="upper right")

        # Panel 3: Prediction Delta (Altered - Baseline)
        ax3.plot(cycles, df["delta_rul_xgb"], label=r"$\Delta$ RUL (XGBoost)", color="crimson", linewidth=1.5)
        ax3.plot(cycles, df["delta_rul_lstm"], label=r"$\Delta$ RUL (LSTM)", color="darkorange", linewidth=1.5)
        ax3.axhline(0, color="black", linestyle="-", linewidth=0.8, alpha=0.7)
        ax3.set_xlabel("Flight Cycle (t)", fontsize=11)
        ax3.set_ylabel(r"$\Delta$ RUL (Altered - Base)", fontsize=11)
        ax3.set_title(r"Prognostic Sensitivity: $\Delta RUL$ Over Flight Cycles", fontsize=12, fontweight="bold")
        ax3.grid(True, linestyle=":", alpha=0.6)
        ax3.legend(loc="upper right")

        fig.tight_layout()
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
