"""
Execution Script for Phase 5 Step 5.10: Validate Numerical Stability, Repeatability, and Edge Robustness
Runs comprehensive validation suite across all four C-MAPSS subsets (FD001 - FD004)
and exports summary JSON, markdown report, and diagnostic plots.
"""

import os
import sys

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from digital_twin.stability_validator import TwinStabilityValidator


def main():
    print("=" * 80)
    print("Executing Phase 5 Step 5.10: Numerical Stability, Repeatability & Edge Robustness")
    print("=" * 80)

    validator = TwinStabilityValidator()
    output_dir = os.path.join(REPO_ROOT, "reports", "digital_twin", "step5_10")
    
    print(f"Output directory: {output_dir}")
    print("Running multi-subset validation suite...")
    
    summary = validator.run_full_validation_suite(output_dir=output_dir)

    print("\n--- Long-Trajectory Numerical Stability (>300 Cycles) ---")
    for key, val in summary["long_trajectory_stability"].items():
        print(f"  {key}: {val['total_cycles']} cycles | Shape invariant: {val['fifo_buffer_shape_invariant']} | All finite: {val['all_states_finite']} | Stable: {val['is_stable']}")

    print("\n--- Deterministic Multi-Run Repeatability ---")
    for sub, val in summary["deterministic_repeatability"].items():
        print(f"  {sub}: Max Diff G1-G29 = {val['max_diff_g1_g29']:.1e} | Max Diff G30 = {val['max_diff_g30']:.1e} | Exact: {val['exact_match']}")

    print("\n--- What-If Scenario Repeatability ---")
    for sc_name, val in summary["scenario_repeatability"].items():
        print(f"  {sc_name}: Max Diff G1-G29 = {val['max_diff_g1_g29']:.1e} | Exact: {val['exact_match']}")

    print("\n--- Physical Thermodynamic Range Validation ---")
    for sub, val in summary["physical_range_validation"].items():
        print(f"  {sub}: Checked {val['cycles_checked']} cycles | Overall Valid: {val['is_overall_valid']}")

    print("\n--- Edge-Case Error Resilience ---")
    edge = summary["edge_case_resilience"]
    print(f"  All Edge Tests Passed: {edge['all_passed']}")
    for k, v in edge["details"].items():
        print(f"    - {k}: {v}")

    print("\n" + "=" * 80)
    print(f"Phase 5 Step 5.10 Validation Completed Successfully.")
    print(f"Artifacts saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
