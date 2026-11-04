"""
=============================================================================
AIRCRAFT ENGINE AI - HEALTH MONITORING & PREDICTIVE MAINTENANCE
Phase 3: Data Preprocessing and Feature Engineering
Step 3.3: Calculate and Append Test Remaining Useful Life (RUL) Labels
=============================================================================

This script processes all four NASA C-MAPSS test subsets:
  - test_FD001.txt + RUL_FD001.txt -> test_FD001_with_RUL.csv
  - test_FD002.txt + RUL_FD002.txt -> test_FD002_with_RUL.csv
  - test_FD003.txt + RUL_FD003.txt -> test_FD003_with_RUL.csv
  - test_FD004.txt + RUL_FD004.txt -> test_FD004_with_RUL.csv

Mathematical formulation:
  For each test engine unit i at operational flight cycle t:
    RUL(i, t) = RUL_GT(i) + (t_cutoff(i) - t)

Where:
  - RUL_GT(i) is the true remaining useful life after the last observed cycle,
    provided in NASA ground-truth file RUL_FD00x.txt for unit i.
  - t_cutoff(i) is the maximum observed operational cycle for test unit i.
  - t is the current recorded operational cycle (1 <= t <= t_cutoff(i)).

All original 26 columns and their row order are strictly preserved.
No feature scaling, normalization, feature selection, or RUL capping is applied.
=============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd

# Define the 26 standard C-MAPSS column names
ORIGINAL_COLUMNS = [
    "unit_number",
    "time_cycles",
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
    "sensor_1",
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_5",
    "sensor_6",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_10",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_16",
    "sensor_17",
    "sensor_18",
    "sensor_19",
    "sensor_20",
    "sensor_21",
]

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]


def load_raw_test_data(raw_dir: str, subset: str) -> pd.DataFrame:
    """Load raw test trajectory file with standard 26 column names."""
    file_path = os.path.join(raw_dir, f"test_{subset}.txt")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Raw test file not found: {file_path}")

    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        names=ORIGINAL_COLUMNS,
        dtype={"unit_number": int, "time_cycles": int},
    )
    return df


def load_ground_truth_rul(raw_dir: str, subset: str) -> pd.Series:
    """
    Load ground-truth RUL vector for the test engines.
    NASA convention: line k (1-indexed) corresponds to test unit k.
    """
    file_path = os.path.join(raw_dir, f"RUL_{subset}.txt")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Ground-truth RUL file not found: {file_path}")

    df_rul = pd.read_csv(file_path, sep=r"\s+", header=None)
    rul_series = df_rul[0].astype(int)
    # Re-index starting at 1 so unit_number matches the series index directly
    rul_series.index = range(1, len(rul_series) + 1)
    rul_series.index.name = "unit_number"
    rul_series.name = "RUL_GT"
    return rul_series


def calculate_test_rul(df_test: pd.DataFrame, rul_gt: pd.Series) -> pd.DataFrame:
    """
    Calculate raw linear Remaining Useful Life (RUL) for each test trajectory observation.

    Formula:
        RUL(i, t) = RUL_GT(i) + (t_cutoff(i) - t)

    Returns a new DataFrame with all original 26 columns preserved and 'RUL' appended.
    """
    df_out = df_test.copy()

    # Determine cutoff cycle (max observed cycle) per engine unit
    cutoff_cycles = df_out.groupby("unit_number")["time_cycles"].transform("max")

    # Map ground truth RUL to each observation by unit_number
    gt_mapped = df_out["unit_number"].map(rul_gt)

    # Compute raw linear RUL
    df_out["RUL"] = (gt_mapped + (cutoff_cycles - df_out["time_cycles"])).astype(int)

    return df_out


def verify_test_subset(
    df_raw: pd.DataFrame, df_processed: pd.DataFrame, rul_gt: pd.Series, subset: str
) -> dict:
    """
    Run 10 mandatory verification checks on the processed test subset.
    """
    results = {}

    # Check 1: Row count matches
    raw_rows = len(df_raw)
    proc_rows = len(df_processed)
    results["row_count_match"] = (raw_rows == proc_rows, f"{raw_rows} == {proc_rows}")

    # Check 2: Original 26 columns preserved in original order
    cols_match = list(df_processed.columns[:26]) == ORIGINAL_COLUMNS
    results["original_cols_preserved"] = (cols_match, "First 26 columns match exactly")

    # Check 3: Exactly one additional column named 'RUL'
    has_rul = (
        len(df_processed.columns) == 27
        and df_processed.columns[-1] == "RUL"
    )
    results["rul_column_added"] = (has_rul, f"Columns: {len(df_processed.columns)} (Expected 27)")

    # Check 4: Test engine count matches number of RUL ground-truth values
    num_test_engines = df_processed["unit_number"].nunique()
    num_gt_values = len(rul_gt)
    results["engine_count_matches_gt"] = (
        num_test_engines == num_gt_values,
        f"{num_test_engines} test engines == {num_gt_values} GT values",
    )

    # Check 5: Engine IDs and ground-truth values are aligned correctly (strictly 1..N)
    engine_ids = sorted(df_processed["unit_number"].unique())
    expected_ids = list(range(1, num_test_engines + 1))
    gt_indices = list(rul_gt.index)
    aligned = (engine_ids == expected_ids) and (gt_indices == expected_ids)
    results["engine_alignment"] = (
        aligned,
        f"Engine IDs strictly 1..{num_test_engines}, 1:1 match with GT indices",
    )

    # Check 6: At final observed cycle, calculated RUL equals NASA ground-truth value
    terminal_rows = df_processed.groupby("unit_number").last()
    terminal_matches_gt = (terminal_rows["RUL"] == rul_gt).all()
    results["terminal_matches_gt"] = (
        terminal_matches_gt,
        f"All {len(terminal_rows)} engines match GT at cutoff cycle",
    )

    # Check 7: RUL values are non-negative integers
    non_negative = (df_processed["RUL"] >= 0).all()
    is_integer = np.issubdtype(df_processed["RUL"].dtype, np.integer)
    results["non_negative_integer"] = (
        non_negative and is_integer,
        f"Non-negative: {non_negative}, Integer type: {is_integer} ({df_processed['RUL'].dtype})",
    )

    # Check 8: RUL decreases by exactly 1 for consecutive cycles within each engine
    rul_diffs = df_processed.groupby("unit_number")["RUL"].diff()
    valid_diffs = rul_diffs.dropna()
    strictly_decrements_by_one = (valid_diffs == -1).all()
    results["stepwise_decrement_one"] = (
        strictly_decrements_by_one,
        f"All {len(valid_diffs)} consecutive step diffs == -1",
    )

    # Check 9: No missing or infinite RUL values
    no_null = df_processed["RUL"].isna().sum() == 0
    no_inf = np.isinf(df_processed["RUL"]).sum() == 0
    results["no_missing_or_inf"] = (
        no_null and no_inf,
        f"Missing: {df_processed['RUL'].isna().sum()}, Inf: {np.isinf(df_processed['RUL']).sum()}",
    )

    # Check 10: First-cycle RUL == ground truth + cutoff cycle - 1
    first_rows = df_processed.groupby("unit_number").first()
    cutoff_per_unit = df_processed.groupby("unit_number")["time_cycles"].max()
    expected_first_rul = rul_gt + cutoff_per_unit - 1
    first_rul_correct = (first_rows["RUL"] == expected_first_rul).all()
    results["first_cycle_rul_match"] = (
        first_rul_correct,
        f"All {len(first_rows)} initial RUL values match (GT + t_cutoff - 1)",
    )

    return results


def main():
    print("=" * 80)
    print("PHASE 3 - STEP 3.3: CALCULATE AND APPEND TEST RUL LABELS")
    print("=" * 80)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(base_dir, "dataset", "raw", "C-MAPSS")
    proc_dir = os.path.join(base_dir, "dataset", "processed")

    os.makedirs(proc_dir, exist_ok=True)

    summary_records = []
    all_passed = True

    for subset in SUBSETS:
        print(f"\n[{subset}] Processing test data...")
        df_raw = load_raw_test_data(raw_dir, subset)
        rul_gt = load_ground_truth_rul(raw_dir, subset)

        df_processed = calculate_test_rul(df_raw, rul_gt)

        # Output file path
        output_filename = f"test_{subset}_with_RUL.csv"
        output_path = os.path.join(proc_dir, output_filename)

        print(f"[{subset}] Saving processed dataset to: {output_path}")
        df_processed.to_csv(output_path, index=False)

        # Verification
        print(f"[{subset}] Running 10 verification checks...")
        verif = verify_test_subset(df_raw, df_processed, rul_gt, subset)

        subset_passed = True
        for check_name, (status, detail) in verif.items():
            status_str = "PASS" if status else "FAIL"
            if not status:
                subset_passed = False
                all_passed = False
            print(f"  - [{status_str}] {check_name}: {detail}")

        # Compute summary stats for the report
        summary_records.append({
            "Subset": subset,
            "Raw Rows": len(df_raw),
            "Processed Rows": len(df_processed),
            "Engines": df_processed["unit_number"].nunique(),
            "Min RUL": df_processed["RUL"].min(),
            "Max RUL": df_processed["RUL"].max(),
            "Mean RUL": round(df_processed["RUL"].mean(), 2),
            "Median RUL": round(df_processed["RUL"].median(), 2),
            "Std RUL": round(df_processed["RUL"].std(), 2),
            "All Checks Passed": subset_passed,
        })

    # Summary table
    summary_df = pd.DataFrame(summary_records)
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY REPORT ACROSS ALL TEST SUBSETS")
    print("=" * 80)
    print(summary_df.to_string(index=False))

    # Print sample rows for FD001..FD004
    print("\n" + "=" * 80)
    print("SAMPLE ROWS FOR TEST SUBSETS (First engine: head and tail)")
    print("=" * 80)
    for subset in SUBSETS:
        df_sub = pd.read_csv(os.path.join(proc_dir, f"test_{subset}_with_RUL.csv"))
        unit1 = df_sub[df_sub["unit_number"] == 1]
        print(f"\n--- {subset} Engine 1 (Total cycles: {len(unit1)}) ---")
        cols_to_show = ["unit_number", "time_cycles", "op_setting_1", "sensor_1", "sensor_2", "sensor_21", "RUL"]
        print("First 3 cycles:")
        print(unit1[cols_to_show].head(3).to_string(index=False))
        print("Final 3 cycles (approaching cutoff):")
        print(unit1[cols_to_show].tail(3).to_string(index=False))

    if all_passed:
        print("\nAll 10 verification checks PASSED successfully for all 4 test subsets.")
        sys.exit(0)
    else:
        print("\nOne or more verification checks FAILED. Please review the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
