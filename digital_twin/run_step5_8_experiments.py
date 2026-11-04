"""
Step 5.8 Experiment Execution Script
Executes Experiment A (Thermal Stress: T30 +35 °R) and Experiment B (Sensor Bias: P30 +15 psia)
on NASA C-MAPSS FD001 Engine Unit 1 across its entire run-to-failure trajectory (192 cycles).
Exports comparison tables, summary metrics, and visual comparison plots.
"""

import json
import os
from typing import Dict, Any

from digital_twin.scenario_interface import (
    create_thermal_stress_scenario,
    create_sensor_bias_scenario,
)
from digital_twin.scenario_comparison import (
    ScenarioComparisonRunner,
    ScenarioComparisonResult,
)


def run_all_step5_8_experiments(
    output_dir: str = os.path.join("reports", "digital_twin", "step5_8"),
) -> Dict[str, Any]:
    """Execute both Step 5.8 experiments and export all artifacts."""
    runner = ScenarioComparisonRunner()
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("PHASE 5 - STEP 5.8: CONTROLLED BASELINE VS. ALTERED SCENARIO EXECUTION")
    print("Dataset: NASA C-MAPSS FD001 | Engine Unit: 1 (Train Split: 192 cycles)")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Experiment A: Thermal Stress (+35.0 °R on T30)
    # -------------------------------------------------------------------------
    print("\n[1/2] Executing Experiment A: Thermal Stress (+35.0 °R on T30)...")
    sc_thermal = create_thermal_stress_scenario(
        scenario_id="SCEN_EXP_A_TS_T30",
        target="T30",
        offset_degR=35.0,
        name="Thermal Stress (+35.0 °R on T30)",
        description="Hypothetical elevated HPC exit temperature excursion (+35.0 °R)",
    )

    res_thermal = runner.run_comparison(
        subset="FD001",
        unit_number=1,
        split="train",
        scenario=sc_thermal,
        experiment_name="Thermal Stress (+35.0 °R on T30)",
    )

    artifacts_thermal = ScenarioComparisonRunner.export_artifacts(
        result=res_thermal,
        output_dir=output_dir,
        file_prefix="thermal_stress_T30_plus35",
    )
    print(f"  -> Total cycles evaluated: {res_thermal.total_cycles}")
    print(f"  -> Delta RUL XGBoost Mean: {res_thermal.summary_metrics['delta_rul_xgb']['mean']:+.2f} cycles")
    print(f"  -> Delta RUL LSTM Mean:    {res_thermal.summary_metrics['delta_rul_lstm']['mean']:+.2f} cycles")
    print(f"  -> Delta Spread Mean:      {res_thermal.summary_metrics['delta_spread']['mean']:+.2f} cycles")
    print(f"  -> Artifacts saved to: {list(artifacts_thermal.values())}")

    # -------------------------------------------------------------------------
    # Experiment B: Sensor Bias (+15.0 psia on P30)
    # -------------------------------------------------------------------------
    print("\n[2/2] Executing Experiment B: Sensor Bias (+15.0 psia on P30)...")
    sc_bias = create_sensor_bias_scenario(
        scenario_id="SCEN_EXP_B_SB_P30",
        target="P30",
        bias=15.0,
        unit="psia",
        name="Sensor Bias (+15.0 psia on P30)",
        description="Hypothetical HPC exit total pressure transducer bias (+15.0 psia)",
    )

    res_bias = runner.run_comparison(
        subset="FD001",
        unit_number=1,
        split="train",
        scenario=sc_bias,
        experiment_name="Sensor Bias (+15.0 psia on P30)",
    )

    artifacts_bias = ScenarioComparisonRunner.export_artifacts(
        result=res_bias,
        output_dir=output_dir,
        file_prefix="sensor_bias_P30_plus15",
    )
    print(f"  -> Total cycles evaluated: {res_bias.total_cycles}")
    print(f"  -> Delta RUL XGBoost Mean: {res_bias.summary_metrics['delta_rul_xgb']['mean']:+.2f} cycles")
    print(f"  -> Delta RUL LSTM Mean:    {res_bias.summary_metrics['delta_rul_lstm']['mean']:+.2f} cycles")
    print(f"  -> Delta Spread Mean:      {res_bias.summary_metrics['delta_spread']['mean']:+.2f} cycles")
    print(f"  -> Artifacts saved to: {list(artifacts_bias.values())}")

    # -------------------------------------------------------------------------
    # Combined Summary Payload
    # -------------------------------------------------------------------------
    combined_summary = {
        "step": "Phase 5 Step 5.8: Baseline vs Altered Scenario Execution",
        "dataset_subset": "FD001",
        "engine_unit": 1,
        "split": "train",
        "total_trajectory_cycles": 192,
        "experiment_A_thermal_stress": {
            "summary_metrics": res_thermal.summary_metrics,
            "fair_comparison_checks": res_thermal.fair_comparison_checks,
            "artifacts": artifacts_thermal,
        },
        "experiment_B_sensor_bias": {
            "summary_metrics": res_bias.summary_metrics,
            "fair_comparison_checks": res_bias.fair_comparison_checks,
            "artifacts": artifacts_bias,
        },
    }

    combined_summary_path = os.path.join(output_dir, "step5_8_experiments_summary.json")
    with open(combined_summary_path, "w", encoding="utf-8") as f:
        json.dump(combined_summary, f, indent=2)

    print(f"\n[OK] Combined experiments summary written to: {combined_summary_path}")
    return combined_summary


if __name__ == "__main__":
    run_all_step5_8_experiments()
