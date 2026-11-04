"""
=============================================================================
AIRCRAFT ENGINE AI - HEALTH MONITORING & PREDICTIVE MAINTENANCE
Phase 3: Data Preprocessing and Feature Engineering
Step 3.2: Calculate and Append Training Remaining Useful Life (RUL) Labels
=============================================================================

This script processes all four NASA C-MAPSS training subsets:
  - train_FD001.txt -> train_FD001_with_RUL.csv
  - train_FD002.txt -> train_FD002_with_RUL.csv
  - train_FD003.txt -> train_FD003_with_RUL.csv
  - train_FD004.txt -> train_FD004_with_RUL.csv

Mathematical formulation:
  For each engine unit i at operational flight cycle t:
    RUL(i, t) = maximum_cycle_of_engine_i - current_cycle_t

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


def load_raw_training_data(raw_dir: str, subset: str) -> pd.DataFrame:
    """Load raw training file with the 26 standard column names."""
    file_path = os.path.join(raw_dir, f"train_{subset}.txt")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Raw file not found: {file_path}")

    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        names=ORIGINAL_COLUMNS,
        dtype={"unit_number": int, "time_cycles": int},
    )
    return df


def calculate_training_rul(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate raw linear Remaining Useful Life (RUL) for each engine trajectory.

    Formula:
        RUL(i, t) = max_cycle_i - t

    Returns a new DataFrame with all original 26 columns preserved and 'RUL' appended.
    """
    df_out = df.copy()

    # Determine maximum cycle per engine unit
    max_cycles = df_out.groupby("unit_number")["time_cycles"].transform("max")

    # Compute raw linear RUL
    df_out["RUL"] = (max_cycles - df_out["time_cycles"]).astype(int)

    return df_out


def verify_processed_subset(
    df_raw: pd.DataFrame, df_processed: pd.DataFrame, subset: str
) -> dict:
    """
    Run 9 mandatory verification checks on the processed subset.
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

    # Check 4: Terminal cycle RUL == 0 for every engine
    # Terminal cycle is the last record of each engine
    terminal_rows = df_processed.groupby("unit_number").last()
    terminal_rul_zero = (terminal_rows["RUL"] == 0).all()
    results["terminal_rul_zero"] = (
        terminal_rul_zero,
        f"All {len(terminal_rows)} engines terminate with RUL = 0",
    )

    # Check 5: RUL values are non-negative integers
    non_negative = (df_processed["RUL"] >= 0).all()
    is_integer = np.issubdtype(df_processed["RUL"].dtype, np.integer)
    results["non_negative_integer"] = (
        non_negative and is_integer,
        f"Non-negative: {non_negative}, Integer type: {is_integer} ({df_processed['RUL'].dtype})",
    )

    # Check 6: RUL decreases by exactly 1 for consecutive cycles within each engine
    rul_diffs = df_processed.groupby("unit_number")["RUL"].diff()
    # For the first row of each engine, diff is NaN. For subsequent rows, diff must be -1.
    valid_diffs = rul_diffs.dropna()
    strictly_decrements_by_one = (valid_diffs == -1).all()
    results["stepwise_decrement_one"] = (
        strictly_decrements_by_one,
        f"All {len(valid_diffs)} consecutive step diffs == -1",
    )

    # Check 7: No missing or infinite RUL values
    no_null = df_processed["RUL"].isna().sum() == 0
    no_inf = np.isinf(df_processed["RUL"]).sum() == 0
    results["no_missing_or_inf"] = (
        no_null and no_inf,
        f"Missing: {df_processed['RUL'].isna().sum()}, Inf: {np.isinf(df_processed['RUL']).sum()}",
    )

    # Check 8: First-cycle RUL == max_cycle - 1 for each engine
    first_rows = df_processed.groupby("unit_number").first()
    max_cycles_per_unit = df_processed.groupby("unit_number")["time_cycles"].max()
    first_rul_expected = max_cycles_per_unit - 1
    first_rul_correct = (first_rows["RUL"] == first_rul_expected).all()
    results["first_cycle_rul_match"] = (
        first_rul_correct,
        f"All {len(first_rows)} initial RUL values match (max_cycle - 1)",
    )

    # Check 9: Number of unique engines unchanged
    raw_units = df_raw["unit_number"].nunique()
    proc_units = df_processed["unit_number"].nunique()
    results["engine_count_preserved"] = (
        raw_units == proc_units,
        f"{raw_units} raw engines == {proc_units} processed engines",
    )

    return results


def main():
    print("=" * 80)
    print("PHASE 3 - STEP 3.2: CALCULATE AND APPEND TRAINING RUL LABELS")
    print("=" * 80)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(base_dir, "dataset", "raw", "C-MAPSS")
    proc_dir = os.path.join(base_dir, "dataset", "processed")

    os.makedirs(proc_dir, exist_ok=True)

    summary_records = []
    all_passed = True

    for subset in SUBSETS:
        print(f"\n[{subset}] Processing training data...")
        df_raw = load_raw_training_data(raw_dir, subset)
        df_processed = calculate_training_rul(df_raw)

        # Output file path
        output_filename = f"train_{subset}_with_RUL.csv"
        output_path = os.path.join(proc_dir, output_filename)

        print(f"[{subset}] Saving processed dataset to: {output_path}")
        df_processed.to_csv(output_path, index=False)

        # Verification
        print(f"[{subset}] Running 9 verification checks...")
        verif = verify_processed_subset(df_raw, df_processed, subset)

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
    print("VERIFICATION SUMMARY REPORT ACROSS ALL TRAINING SUBSETS")
    print("=" * 80)
    print(summary_df.to_string(index=False))

    if all_passed:
        print("\nAll 9 verification checks PASSED successfully for all 4 subsets.")
        sys.exit(0)
    else:
        print("\nOne or more verification checks FAILED. Please review the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
