"""
Step 5.9 Execution Script - Compare Digital Twin Outputs with Recorded Data
Executes comparisons across all 4 C-MAPSS subsets for both training and validation trajectories.
Generates tabular CSVs, JSON summary, markdown report, and diagnostic plots in reports/digital_twin/step5_9/.

Roadmap Reference: Phase 5 - Step 5.9
"""

import json
import os
import sys
from typing import Dict, Any

from digital_twin.recorded_data_comparison import RecordedDataComparator


def run_step5_9_comparisons(
    output_dir: str = os.path.join("reports", "digital_twin", "step5_9"),
) -> Dict[str, Any]:
    """Execute all Step 5.9 comparisons and generate reports/artifacts."""
    print("=" * 80)
    print("PHASE 5 - STEP 5.9: COMPARE DIGITAL TWIN OUTPUTS WITH RECORDED DATA")
    print("Subsets: FD001, FD002, FD003, FD004 | Categories A, B, C, D, E")
    print("=" * 80)

    comparator = RecordedDataComparator()
    artifacts = comparator.run_all_comparisons(output_dir=output_dir)

    summary = artifacts["summary_results"]

    # -------------------------------------------------------------------------
    # Display Summary Results to Console
    # -------------------------------------------------------------------------
    print("\n[CATEGORY A: Direct Measured Telemetry Pass-Through (G1–G26)]")
    print(f"{'Subset':<8} {'Split':<8} {'Unit':<8} {'Cycles':<8} {'Variables':<10} {'Matches':<10} {'Mismatches':<12} {'Max Diff':<12} {'Status'}")
    print("-" * 88)
    for sub, res in summary["categories"]["A_direct_telemetry"].items():
        status = "EXACT MATCH (100%)" if res["all_exact_match"] else "MISMATCH"
        print(f"{sub:<8} {res['split']:<8} Unit {res['unit_number']:<3} {res['records_compared']:<8} "
              f"{res['variables_compared']:<10} {res['exact_matches']:<10} {res['mismatches']:<12} "
              f"{res['max_abs_diff']:<12.1e} {status}")

    print("\n[CATEGORY B: Independently Recomputed Derived Indicators (G27–G29)]")
    print(f"{'Subset':<8} {'Unit':<8} {'Cycles':<8} {'G27 Max Diff':<14} {'G28 Max Diff':<14} {'G29 Max Diff':<14} {'All Finite':<12} {'G27 Match':<10} {'G28 Match':<10} {'G29 Match':<10}")
    print("-" * 108)
    for sub, res in summary["categories"]["B_derived_indicators"].items():
        print(f"{sub:<8} Unit {res['unit_number']:<3} {res['records_compared']:<8} "
              f"{res['max_abs_diff_g27']:<14.1e} {res['max_abs_diff_g28']:<14.1e} {res['max_abs_diff_g29']:<14.1e} "
              f"{str(res['all_finite']):<12} {res['exact_matches_g27']:<10} {res['exact_matches_g28']:<10} {res['exact_matches_g29']:<10}")

    print("\n[CATEGORY C: Preprocessing and Sequence Feature Representations (G30–G31)]")
    print(f"{'Subset':<8} {'Unit':<8} {'G30 Cycles':<12} {'Features':<10} {'G30 Max Diff':<14} {'G31 Windows':<14} {'G31 Max Diff':<14} {'Excluded (t<30)':<16}")
    print("-" * 96)
    for sub, res in summary["categories"]["C_preprocessing_and_sequences"].items():
        print(f"{sub:<8} Unit {res['unit_number']:<3} {res['g30_records_compared']:<12} "
              f"{res['g30_features_per_record']:<10} {res['g30_max_abs_diff']:<14.1e} "
              f"{res['g31_records_compared']:<14} {res['g31_max_abs_diff']:<14.1e} "
              f"{res['excluded_padded_cycles']:<16}")

    print("\n[CATEGORY D: Machine Learning Prognostic RUL Outputs (G35–G36)]")
    print(f"{'Subset':<8} {'Unit':<8} {'Windows':<10} {'XGB MAE (Cap)':<15} {'LSTM MAE (Cap)':<15} {'XGB RMSE (Cap)':<15} {'LSTM RMSE (Cap)':<15} {'XGB vs P4 MAE':<15} {'LSTM vs P4 MAE'}")
    print("-" * 118)
    for sub, res in summary["categories"]["D_ml_prognostics"].items():
        print(f"{sub:<8} Unit {res['unit_number']:<3} {res['records_compared']:<10} "
              f"{res['xgb_mae_capped']:<15.2f} {res['lstm_mae_capped']:<15.2f} "
              f"{res['xgb_rmse_capped']:<15.2f} {res['lstm_rmse_capped']:<15.2f} "
              f"{res['xgb_mae_vs_phase4_recorded']:<15.4f} {res['lstm_mae_vs_phase4_recorded']:<15.4f}")

    print("\n[CATEGORY E: Explicit Classification of Unobservable Variables]")
    for var, reason in summary["categories"]["E_unobservable_variables"].items():
        print(f"  • {var}: {reason}")

    print("\n" + "=" * 80)
    print(f"[OK] Step 5.9 Artifacts successfully written to: {output_dir}")
    print("  1. Telemetry Comparison:       telemetry_comparison.csv")
    print("  2. Derived Indicators:         derived_indicator_comparison.csv")
    print("  3. Preprocessing/Sequences:    preprocessing_comparison.csv")
    print("  4. RUL Prognostics:            rul_comparison.csv")
    print("  5. Summary JSON:               comparison_summary.json")
    print("  6. Summary Markdown:           comparison_summary.md")
    print("  7. Diagnostic Plots:           4 high-resolution figures (.png)")
    print("=" * 80)

    return artifacts


if __name__ == "__main__":
    run_step5_9_comparisons()
