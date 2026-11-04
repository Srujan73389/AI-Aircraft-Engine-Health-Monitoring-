"""
=============================================================================
AIRCRAFT ENGINE AI - HEALTH MONITORING & PREDICTIVE MAINTENANCE
Phase 3: Data Preprocessing and Feature Engineering
Step 3.5: Handle Invalid or Missing Values (Data Quality & Validation)
=============================================================================

This script validates the data quality and structural integrity of all eight
processed C-MAPSS datasets (train and test for FD001, FD002, FD003, FD004):
  - dataset/processed/train_FD001_with_RUL.csv
  - dataset/processed/train_FD002_with_RUL.csv
  - dataset/processed/train_FD003_with_RUL.csv
  - dataset/processed/train_FD004_with_RUL.csv
  - dataset/processed/test_FD001_with_RUL.csv
  - dataset/processed/test_FD002_with_RUL.csv
  - dataset/processed/test_FD003_with_RUL.csv
  - dataset/processed/test_FD004_with_RUL.csv

Checks performed:
  1. Dimension check (row count matches Phase 2/3, col count == 27)
  2. Column schema and ordering check
  3. Missing values check (NaN / null count across all columns)
  4. Infinite values check (+inf / -inf across all numeric columns)
  5. Data type validity check (integer and float expectations)
  6. Engine ID integrity (strictly 1..N, contiguous, positive)
  7. Cycle number integrity (starts at 1, strictly monotonic +1 steps)
  8. Sensor & setting finite bounds check
  9. RUL target integrity (finite, non-negative, monotonic -1 decrement, terminal alignment)
 10. Duplicate record check (zero duplicate rows)

No cleaning or modification is performed unless an actual corrupted record is found.
=============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd

EXPECTED_COLUMNS = [
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
    "RUL",
]

DATASET_SPECS = {
    ("train", "FD001"): {"rows": 20631, "engines": 100},
    ("train", "FD002"): {"rows": 53759, "engines": 260},
    ("train", "FD003"): {"rows": 24720, "engines": 100},
    ("train", "FD004"): {"rows": 61249, "engines": 249},
    ("test", "FD001"): {"rows": 13096, "engines": 100},
    ("test", "FD002"): {"rows": 33991, "engines": 259},
    ("test", "FD003"): {"rows": 16596, "engines": 100},
    ("test", "FD004"): {"rows": 41214, "engines": 248},
}


def load_gt_rul(raw_dir: str, subset: str) -> pd.Series:
    """Load ground-truth RUL vector for test verification."""
    rul_path = os.path.join(raw_dir, f"RUL_{subset}.txt")
    df_rul = pd.read_csv(rul_path, sep=r"\s+", header=None)
    series = df_rul[0].astype(int)
    series.index = range(1, len(series) + 1)
    return series


def validate_dataset(df: pd.DataFrame, split: str, subset: str, raw_dir: str) -> tuple[bool, list[dict]]:
    """Run comprehensive quality and validation checks on a single dataset."""
    specs = DATASET_SPECS[(split, subset)]
    expected_rows = specs["rows"]
    expected_engines = specs["engines"]

    checks = []

    # 1. Dimension Check
    rows_ok = len(df) == expected_rows
    cols_ok = len(df.columns) == 27
    checks.append({
        "check": "Dimensions (Rows & Cols)",
        "pass": rows_ok and cols_ok,
        "detail": f"Shape: {df.shape} (Expected: ({expected_rows}, 27))",
    })

    # 2. Schema Check
    cols_match = list(df.columns) == EXPECTED_COLUMNS
    checks.append({
        "check": "Column Names and Order",
        "pass": cols_match,
        "detail": "26 original features + 'RUL' in exact order",
    })

    # 3. Missing Values Check
    null_count = df.isna().sum().sum()
    checks.append({
        "check": "Missing / NaN Values",
        "pass": null_count == 0,
        "detail": f"Total null values across all cells: {null_count}",
    })

    # 4. Infinite Values Check
    inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
    checks.append({
        "check": "Infinite Values (+/- Inf)",
        "pass": inf_count == 0,
        "detail": f"Total infinite values across all cells: {inf_count}",
    })

    # 5. Data Types Check (All 27 columns must be numeric; IDs, cycles, and RUL must be integers)
    int_cols = ["unit_number", "time_cycles", "RUL"]
    int_types_ok = all(np.issubdtype(df[col].dtype, np.integer) for col in int_cols)
    all_numeric_ok = all(np.issubdtype(df[col].dtype, np.number) for col in EXPECTED_COLUMNS)
    non_numeric_cols = [c for c in EXPECTED_COLUMNS if not np.issubdtype(df[c].dtype, np.number)]
    checks.append({
        "check": "Numeric Data Types Integrity",
        "pass": int_types_ok and all_numeric_ok and len(non_numeric_cols) == 0,
        "detail": f"All 27 columns strictly numeric (Integer keys: {int_types_ok}, Non-numeric cols: {len(non_numeric_cols)})",
    })

    # 6. Engine ID Integrity
    unique_engines = sorted(df["unit_number"].unique())
    engine_count_ok = len(unique_engines) == expected_engines
    contiguous_ok = unique_engines == list(range(1, expected_engines + 1))
    checks.append({
        "check": "Engine IDs (Contiguous 1..N)",
        "pass": engine_count_ok and contiguous_ok,
        "detail": f"Count: {len(unique_engines)}, Contiguous 1..{expected_engines}: {contiguous_ok}",
    })

    # 7. Cycle Number Integrity
    cycles_positive = (df["time_cycles"] >= 1).all()
    # Check starts at 1 and increments by 1
    cycle_starts_1 = (df.groupby("unit_number")["time_cycles"].first() == 1).all()
    cycle_diffs = df.groupby("unit_number")["time_cycles"].diff().dropna()
    cycle_diffs_one = (cycle_diffs == 1).all()
    checks.append({
        "check": "Cycle Number Sequence",
        "pass": cycles_positive and cycle_starts_1 and cycle_diffs_one,
        "detail": f"Starts at 1: {cycle_starts_1}, Strictly monotonic +1: {cycle_diffs_one}",
    })

    # 8. Sensor & Setting Finite Bounds Check
    sensor_cols = [c for c in EXPECTED_COLUMNS if c.startswith("sensor_") or c.startswith("op_setting_")]
    all_finite = np.isfinite(df[sensor_cols]).all().all()
    checks.append({
        "check": "Sensors & Settings Finite",
        "pass": bool(all_finite),
        "detail": f"All {len(sensor_cols)} features strictly finite numeric values",
    })

    # 9. RUL Target Integrity
    rul_non_neg = (df["RUL"] >= 0).all()
    rul_diffs = df.groupby("unit_number")["RUL"].diff().dropna()
    rul_decrement_one = (rul_diffs == -1).all()

    if split == "train":
        terminal_rul = df.groupby("unit_number")["RUL"].last()
        terminal_ok = (terminal_rul == 0).all()
        term_detail = f"Terminal RUL == 0: {terminal_ok}"
    else:
        gt_rul = load_gt_rul(raw_dir, subset)
        terminal_rul = df.groupby("unit_number")["RUL"].last()
        terminal_ok = (terminal_rul == gt_rul).all()
        term_detail = f"Terminal RUL matches GT: {terminal_ok}"

    checks.append({
        "check": "RUL Target Validity",
        "pass": rul_non_neg and rul_decrement_one and terminal_ok,
        "detail": f"Non-negative: {rul_non_neg}, Monotonic -1: {rul_decrement_one}, {term_detail}",
    })

    # 10. Duplicate Records Check
    dup_count = df.duplicated().sum()
    checks.append({
        "check": "Duplicate Rows",
        "pass": dup_count == 0,
        "detail": f"Duplicate rows detected: {dup_count}",
    })

    dataset_pass = all(c["pass"] for c in checks)
    return dataset_pass, checks


def main():
    print("=" * 80)
    print("PHASE 3 - STEP 3.5: PROCESSED DATASET QUALITY & VALIDATION REPORT")
    print("=" * 80)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    proc_dir = os.path.join(base_dir, "dataset", "processed")
    raw_dir = os.path.join(base_dir, "dataset", "raw", "C-MAPSS")

    all_passed = True
    summary_table = []

    for (split, subset) in sorted(DATASET_SPECS.keys()):
        filename = f"{split}_{subset}_with_RUL.csv"
        filepath = os.path.join(proc_dir, filename)

        print(f"\n[{split.upper()} {subset}] Inspecting: {filename}")
        if not os.path.exists(filepath):
            print(f"  [ERROR] File missing: {filepath}")
            all_passed = False
            continue

        df = pd.read_csv(filepath)
        dataset_pass, checks = validate_dataset(df, split, subset, raw_dir)

        if not dataset_pass:
            all_passed = False

        for c in checks:
            status = "PASS" if c["pass"] else "FAIL"
            print(f"  - [{status}] {c['check']}: {c['detail']}")

        summary_table.append({
            "Subset": subset,
            "Split": split.upper(),
            "Rows": len(df),
            "Cols": len(df.columns),
            "Nulls": df.isna().sum().sum(),
            "Infs": np.isinf(df.select_dtypes(include=[np.number])).sum().sum(),
            "Duplicates": df.duplicated().sum(),
            "Status": "PASS" if dataset_pass else "FAIL",
        })

    summary_df = pd.DataFrame(summary_table)
    print("\n" + "=" * 80)
    print("OVERALL QUALITY VALIDATION SUMMARY (ALL 8 DATASETS)")
    print("=" * 80)
    print(summary_df.to_string(index=False))

    total_rows = summary_df["Rows"].sum()
    print(f"\nTotal verified records across all 8 datasets: {total_rows:,} rows.")
    print("Expected benchmark count: 265,256 rows.")

    if all_passed and total_rows == 265256:
        print("\nCONCLUSION: ALL CHECKS PASSED. ZERO MISSING OR INVALID VALUES FOUND.")
        print("NO CLEANING, REMOVAL, OR IMPUTATION IS REQUIRED.")
        sys.exit(0)
    else:
        print("\nWARNING: One or more validation checks failed. Review details above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
