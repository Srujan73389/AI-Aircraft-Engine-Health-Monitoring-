"""
=============================================================================
AIRCRAFT ENGINE AI - HEALTH MONITORING & PREDICTIVE MAINTENANCE
Phase 2: Dataset Understanding & Exploratory Data Analysis (EDA)
Step 2.5: Comprehensive Data Quality & Types Audit
=============================================================================
This script programmatically inspects all 8 trajectory files for:
1. Actual inferred data types for every one of the 26 columns.
2. Missing / null value counts for every one of the 26 columns.
3. Non-numeric token detection with exact row/column tracking.
4. Infinite values (+inf, -inf).
5. Invalid engine unit IDs (must be integer >= 1).
6. Invalid operational cycle numbers (must be integer >= 1).
7. Total rows containing any invalid record.
=============================================================================
"""

import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join("dataset", "raw", "C-MAPSS")

FILES_TO_CHECK = [
    "train_FD001.txt", "train_FD002.txt", "train_FD003.txt", "train_FD004.txt",
    "test_FD001.txt",  "test_FD002.txt",  "test_FD003.txt",  "test_FD004.txt"
]

COLUMN_NAMES = [
    "unit_number", "time_cycles",
    "op_setting_1", "op_setting_2", "op_setting_3",
    "sensor_1", "sensor_2", "sensor_3", "sensor_4", "sensor_5",
    "sensor_6", "sensor_7", "sensor_8", "sensor_9", "sensor_10",
    "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
    "sensor_16", "sensor_17", "sensor_18", "sensor_19", "sensor_20", "sensor_21"
]


def audit_file(filename):
    filepath = os.path.join(DATA_DIR, filename)
    print("\n" + "=" * 105)
    print(f"FILE: {filename}")
    print("=" * 105)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    # -------------------------------------------------------------------------
    # 1. READ RAW STRING DATA TO DETECT NON-NUMERIC TOKENS
    # -------------------------------------------------------------------------
    df_str = pd.read_csv(filepath, sep=r"\s+", header=None, names=COLUMN_NAMES, dtype=str)
    total_rows, total_cols = df_str.shape

    non_numeric_details = []
    non_numeric_counts = {col: 0 for col in COLUMN_NAMES}

    for col in COLUMN_NAMES:
        series_str = df_str[col]
        # Try converting to numeric; genuine text strings become NaN
        converted = pd.to_numeric(series_str, errors="coerce")
        # Mask where conversion failed but original string was not empty/null
        bad_mask = converted.isna() & series_str.notna()
        bad_count = bad_mask.sum()
        non_numeric_counts[col] = int(bad_count)
        if bad_count > 0:
            bad_rows = df_str[bad_mask].index.tolist()
            for r_idx in bad_rows:
                non_numeric_details.append((col, r_idx, series_str.iloc[r_idx]))

    total_non_numeric = sum(non_numeric_counts.values())

    # -------------------------------------------------------------------------
    # 2. LOAD NUMERIC DATAFRAME TO INSPECT ACTUAL DTYPES & VALUES
    # -------------------------------------------------------------------------
    df_num = pd.read_csv(filepath, sep=r"\s+", header=None, names=COLUMN_NAMES)

    # Missing / null values per column
    missing_counts = df_num.isna().sum()
    total_missing = int(missing_counts.sum())

    # Infinite values
    inf_mask = np.isinf(df_num.select_dtypes(include=[np.number]).values)
    total_inf = int(inf_mask.sum())

    # Invalid engine IDs
    unit_col = df_num["unit_number"]
    invalid_units = int(((unit_col < 1) | (unit_col % 1 != 0)).sum())

    # Invalid cycle numbers
    cycle_col = df_num["time_cycles"]
    invalid_cycles = int(((cycle_col < 1) | (cycle_col % 1 != 0)).sum())

    # Rows with any invalid record
    has_nan_or_inf = df_num.isna().any(axis=1) | np.isinf(df_num.select_dtypes(include=[np.number]).values).any(axis=1)
    has_bad_ids = (unit_col < 1) | (unit_col % 1 != 0) | (cycle_col < 1) | (cycle_col % 1 != 0)
    invalid_rows_total = int((has_nan_or_inf | has_bad_ids).sum())

    # -------------------------------------------------------------------------
    # 3. DISPLAY PER-COLUMN BREAKDOWN (ACTUAL DTYPES & MISSING COUNTS)
    # -------------------------------------------------------------------------
    print(f"\n[PER-COLUMN AUDIT: {filename}] (Total Rows: {total_rows:,} | Total Columns: {total_cols})")
    col_header = (
        f"{'Col #':<5} | {'Column Name':<14} | {'Actual Dtype':<13} | "
        f"{'Data Kind':<16} | {'Missing (NaN)':<13} | {'Non-Numeric':<11}"
    )
    print("-" * len(col_header))
    print(col_header)
    print("-" * len(col_header))

    int_cols_count = 0
    float_cols_count = 0

    for idx, col in enumerate(COLUMN_NAMES, start=1):
        actual_dtype = str(df_num[col].dtype)
        if "int" in actual_dtype:
            kind = "Integer"
            int_cols_count += 1
        elif "float" in actual_dtype:
            kind = "Floating-point"
            float_cols_count += 1
        else:
            kind = "Non-numeric / Other"

        m_count = int(missing_counts[col])
        nn_count = non_numeric_counts[col]

        print(
            f"{idx:>5} | {col:<14} | {actual_dtype:<13} | "
            f"{kind:<16} | {m_count:<13} | {nn_count:<11}"
        )

    print("-" * len(col_header))
    print(f"Column Types Summary: {int_cols_count} Integer columns, {float_cols_count} Floating-point columns.")

    # -------------------------------------------------------------------------
    # 4. FILE-LEVEL QUALITY CHECKS
    # -------------------------------------------------------------------------
    print(f"\n[FILE-LEVEL QUALITY CHECKS: {filename}]")
    print(f"  Total Missing Values across all columns: {total_missing}")
    print(f"  Total Non-Numeric Tokens found:          {total_non_numeric}")
    print(f"  Total Infinite Values (+/- inf):         {total_inf}")
    print(f"  Invalid Engine IDs (must be int >= 1):   {invalid_units}")
    print(f"  Invalid Cycle Values (must be int >= 1): {invalid_cycles}")
    print(f"  Total Rows Containing Any Invalid Data:  {invalid_rows_total}")

    if total_non_numeric > 0:
        print("  [!] NON-NUMERIC TOKENS DETECTED:")
        for col_name, row_no, val in non_numeric_details[:10]:
            print(f"      - Column '{col_name}', Row {row_no}: '{val}'")
    else:
        print("  -> Non-numeric verification: 0 non-numeric tokens (all values are valid numbers).")

    if invalid_rows_total == 0:
        print("  -> Quality Verdict: 100% CLEAN.")
    else:
        print("  -> Quality Verdict: ISSUES DETECTED.")

    return {
        "filename": filename,
        "rows": total_rows,
        "cols": total_cols,
        "int_cols": int_cols_count,
        "float_cols": float_cols_count,
        "missing_total": total_missing,
        "non_numeric_total": total_non_numeric,
        "infinite_total": total_inf,
        "invalid_units": invalid_units,
        "invalid_cycles": invalid_cycles,
        "invalid_rows": invalid_rows_total,
    }


def main():
    print("=" * 105)
    print("PHASE 2, STEP 2.5: COMPREHENSIVE DATA QUALITY & ACTUAL DATA TYPES AUDIT")
    print("=" * 105)

    summary_records = []
    for fname in FILES_TO_CHECK:
        rec = audit_file(fname)
        summary_records.append(rec)

    # -------------------------------------------------------------------------
    # 5. FINAL COMPARISON TABLE ACROSS ALL 8 FILES
    # -------------------------------------------------------------------------
    print("\n" + "=" * 105)
    print("FINAL DATA QUALITY SUMMARY COMPARISON TABLE (ALL 8 FILES)")
    print("=" * 105)
    tbl_header = (
        f"{'Filename':<17} | {'Rows':<8} | {'Cols':<5} | {'Int / Float':<11} | "
        f"{'Missing':<7} | {'Non-Numeric':<11} | {'Infinite':<8} | {'Bad Units':<9} | {'Bad Cycles':<10} | {'Status':<8}"
    )
    print(tbl_header)
    print("-" * len(tbl_header))

    all_clean = True
    for r in summary_records:
        status_str = "CLEAN" if r["invalid_rows"] == 0 else "ISSUES"
        if r["invalid_rows"] > 0:
            all_clean = False
        types_ratio = f"{r['int_cols']}i / {r['float_cols']}f"
        print(
            f"{r['filename']:<17} | {r['rows']:<8} | {r['cols']:<5} | {types_ratio:<11} | "
            f"{r['missing_total']:<7} | {r['non_numeric_total']:<11} | {r['infinite_total']:<8} | "
            f"{r['invalid_units']:<9} | {r['invalid_cycles']:<10} | {status_str:<8}"
        )

    print("=" * 105)
    if all_clean:
        print("OVERALL VERDICT: ALL 8 FILES ARE 100% CLEAN. NO MISSING, NON-NUMERIC, OR INVALID RECORDS.")
    else:
        print("OVERALL VERDICT: DATA QUALITY ISSUES FOUND.")
    print("=" * 105)


if __name__ == "__main__":
    main()
