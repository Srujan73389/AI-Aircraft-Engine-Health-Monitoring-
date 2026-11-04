"""
=============================================================================
AIRCRAFT ENGINE AI - HEALTH MONITORING & PREDICTIVE MAINTENANCE
Phase 3: Data Preprocessing and Feature Engineering
Step 3.10: Engine-Level Training and Validation Split
=============================================================================

This script performs a strictly LEAKAGE-FREE, reproducible engine-level split
for all four NASA C-MAPSS training datasets:
  - train_FD001_with_RUL.csv -> train_FD001_split.csv & validation_FD001_split.csv
  - train_FD002_with_RUL.csv -> train_FD002_split.csv & validation_FD002_split.csv
  - train_FD003_with_RUL.csv -> train_FD003_split.csv & validation_FD003_split.csv
  - train_FD004_with_RUL.csv -> train_FD004_split.csv & validation_FD004_split.csv

Key Architectural Properties:
  1. Engine-Level Granularity: Engines are assigned ENTIRELY to either training
     or validation. No cycles from the same engine are ever split across groups.
  2. Reproducibility: Seed is fixed at 42.
  3. Stratification Ratio: Target 80% engines for training, 20% for validation.
  4. Chronological & Data Integrity: Preserves all 27 original columns, row
     order, cycle sequences, and RUL values without modification.
  5. Audit Manifest: Saves train_val_split_manifest.json containing exact engine
     assignments and verification metrics.
=============================================================================
"""

import os
import sys
import json
import numpy as np
import pandas as pd

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
SPLIT_RATIO = 0.80  # 80% training engines, 20% validation engines
RANDOM_SEED = 42

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


def split_engine_subset(df: pd.DataFrame, subset: str, seed: int = RANDOM_SEED) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Split a training dataset into training and validation sets at the engine level.
    """
    # 1. Identify unique engine IDs in ascending order
    unique_engines = np.array(sorted(df["unit_number"].unique()))
    total_engines = len(unique_engines)

    # 2. Reproducibly shuffle engine IDs
    rng = np.random.RandomState(seed)
    shuffled_engines = rng.permutation(unique_engines)

    # 3. Partition engines (80% train, 20% val)
    n_train_engines = int(round(total_engines * SPLIT_RATIO))
    train_engines = sorted(shuffled_engines[:n_train_engines].tolist())
    val_engines = sorted(shuffled_engines[n_train_engines:].tolist())

    # 4. Extract corresponding rows
    df_train = df[df["unit_number"].isin(train_engines)].copy()
    df_val = df[df["unit_number"].isin(val_engines)].copy()

    # 5. Metadata and statistics
    split_info = {
        "subset": subset,
        "random_seed": seed,
        "target_split_ratio": SPLIT_RATIO,
        "total_engines": total_engines,
        "train_engines_count": len(train_engines),
        "val_engines_count": len(val_engines),
        "actual_engine_split_pct_train": round((len(train_engines) / total_engines) * 100, 2),
        "actual_engine_split_pct_val": round((len(val_engines) / total_engines) * 100, 2),
        "total_rows": len(df),
        "train_rows_count": len(df_train),
        "val_rows_count": len(df_val),
        "actual_row_split_pct_train": round((len(df_train) / len(df)) * 100, 2),
        "actual_row_split_pct_val": round((len(df_val) / len(df)) * 100, 2),
        "train_engine_ids": [int(x) for x in train_engines],
        "val_engine_ids": [int(x) for x in val_engines],
    }

    return df_train, df_val, split_info


def verify_split(df_orig: pd.DataFrame, df_train: pd.DataFrame, df_val: pd.DataFrame, split_info: dict) -> dict:
    """Run comprehensive verification on the partitioned data."""
    checks = {}

    train_engines = set(split_info["train_engine_ids"])
    val_engines = set(split_info["val_engine_ids"])
    orig_engines = set(df_orig["unit_number"].unique())

    # Check 1: Zero engine overlap
    overlap = train_engines.intersection(val_engines)
    checks["zero_engine_overlap"] = (
        len(overlap) == 0,
        f"Overlap count: {len(overlap)}",
    )

    # Check 2: All original engines accounted for
    engine_union = train_engines.union(val_engines)
    checks["all_engines_accounted"] = (
        engine_union == orig_engines,
        f"{len(engine_union)} engines partitioned == {len(orig_engines)} original engines",
    )

    # Check 3: Row counts sum exactly
    row_sum_ok = (len(df_train) + len(df_val)) == len(df_orig)
    checks["row_count_sum_exact"] = (
        row_sum_ok,
        f"Train ({len(df_train)}) + Val ({len(df_val)}) == Orig ({len(df_orig)})",
    )

    # Check 4: No duplicate rows in train or val
    train_dups = df_train.duplicated().sum()
    val_dups = df_val.duplicated().sum()
    checks["no_duplicate_rows"] = (
        train_dups == 0 and val_dups == 0,
        f"Train dups: {train_dups}, Val dups: {val_dups}",
    )

    # Check 5: Column names and order preserved exactly
    cols_train_ok = list(df_train.columns) == EXPECTED_COLUMNS
    cols_val_ok = list(df_val.columns) == EXPECTED_COLUMNS
    checks["column_schema_preserved"] = (
        cols_train_ok and cols_val_ok,
        f"27 columns in exact original order (Train: {cols_train_ok}, Val: {cols_val_ok})",
    )

    # Check 6: Chronological cycle ordering intact within each engine
    train_diffs = df_train.groupby("unit_number")["time_cycles"].diff().dropna()
    val_diffs = df_val.groupby("unit_number")["time_cycles"].diff().dropna()
    train_cycles_ordered = (train_diffs == 1).all()
    val_cycles_ordered = (val_diffs == 1).all()
    checks["cycle_ordering_preserved"] = (
        train_cycles_ordered and val_cycles_ordered,
        f"Strictly monotonic +1 step (Train: {train_cycles_ordered}, Val: {val_cycles_ordered})",
    )

    # Check 7: RUL target validity preserved (train ends at 0, non-negative, step -1)
    train_term_rul = df_train.groupby("unit_number")["RUL"].last()
    val_term_rul = df_val.groupby("unit_number")["RUL"].last()
    train_term_zero = (train_term_rul == 0).all()
    val_term_zero = (val_term_rul == 0).all()
    checks["rul_targets_preserved"] = (
        train_term_zero and val_term_zero and (df_train["RUL"] >= 0).all() and (df_val["RUL"] >= 0).all(),
        f"Terminal RUL == 0 (Train: {train_term_zero}, Val: {val_term_zero})",
    )

    return checks


def main():
    print("=" * 80)
    print("PHASE 3 - STEP 3.10: ENGINE-LEVEL TRAINING & VALIDATION SPLIT")
    print("=" * 80)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    proc_dir = os.path.join(base_dir, "dataset", "processed")

    all_passed = True
    manifest = {}
    summary_table = []

    for subset in SUBSETS:
        input_filename = f"train_{subset}_with_RUL.csv"
        input_path = os.path.join(proc_dir, input_filename)

        print(f"\n[{subset}] Loading: {input_filename}")
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        df_orig = pd.read_csv(input_path)

        # Perform split
        df_train, df_val, split_info = split_engine_subset(df_orig, subset, seed=RANDOM_SEED)

        # Save output split CSVs
        train_out_path = os.path.join(proc_dir, f"train_{subset}_split.csv")
        val_out_path = os.path.join(proc_dir, f"validation_{subset}_split.csv")

        print(f"[{subset}] Saving training partition to:   {train_out_path}")
        df_train.to_csv(train_out_path, index=False)

        print(f"[{subset}] Saving validation partition to: {val_out_path}")
        df_val.to_csv(val_out_path, index=False)

        # Verification
        print(f"[{subset}] Running verification checks...")
        verif = verify_split(df_orig, df_train, df_val, split_info)

        subset_passed = True
        for check_name, (status, detail) in verif.items():
            status_str = "PASS" if status else "FAIL"
            if not status:
                subset_passed = False
                all_passed = False
            print(f"  - [{status_str}] {check_name}: {detail}")

        manifest[subset] = split_info

        summary_table.append({
            "Subset": subset,
            "Total Engines": split_info["total_engines"],
            "Train Engines": split_info["train_engines_count"],
            "Val Engines": split_info["val_engines_count"],
            "Engine Split %": f"{split_info['actual_engine_split_pct_train']}% / {split_info['actual_engine_split_pct_val']}%",
            "Total Rows": split_info["total_rows"],
            "Train Rows": split_info["train_rows_count"],
            "Val Rows": split_info["val_rows_count"],
            "Row Split %": f"{split_info['actual_row_split_pct_train']}% / {split_info['actual_row_split_pct_val']}%",
            "Status": "PASS" if subset_passed else "FAIL",
        })

    # Save manifest
    manifest_path = os.path.join(proc_dir, "train_val_split_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[MANIFEST] Saved reproducible split manifest to: {manifest_path}")

    # Print summary table
    summary_df = pd.DataFrame(summary_table)
    print("\n" + "=" * 80)
    print("FINAL ENGINE-LEVEL SPLIT VERIFICATION TABLE (ALL 4 SUBSETS)")
    print("=" * 80)
    print(summary_df.to_string(index=False))

    total_orig_rows = summary_df["Total Rows"].sum()
    total_train_rows = summary_df["Train Rows"].sum()
    total_val_rows = summary_df["Val Rows"].sum()
    print(f"\nTotal Dataset Summary:")
    print(f"  - Total Original Training Rows:   {total_orig_rows:,}")
    print(f"  - Total Partitioned Train Rows:    {total_train_rows:,} ({total_train_rows/total_orig_rows*100:.2f}%)")
    print(f"  - Total Partitioned Val Rows:      {total_val_rows:,} ({total_val_rows/total_orig_rows*100:.2f}%)")
    print(f"  - Row Preservation Match:          {total_train_rows + total_val_rows == total_orig_rows}")

    if all_passed:
        print("\nALL 4 SUBSETS SUCCESSFULLY PARTITIONED AND VERIFIED.")
        sys.exit(0)
    else:
        print("\nONE OR MORE VERIFICATION CHECKS FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
