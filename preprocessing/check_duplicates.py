"""
=============================================================================
AIRCRAFT ENGINE AI - HEALTH MONITORING & PREDICTIVE MAINTENANCE
Phase 2: Dataset Understanding & Exploratory Data Analysis (EDA)
Step 2.6: Duplicate Rows & Duplicate Engine-Cycle Combinations Audit
=============================================================================
This script inspects all 8 trajectory files (train_FD001 to train_FD004 and
test_FD001 to test_FD004) for:
1. Full-row duplicates: Completely identical rows across all 26 columns.
2. Duplicate engine-cycle combinations: Multiple rows sharing the same
   (unit_number, time_cycles) key, regardless of sensor values.
=============================================================================
"""

import os
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


def audit_duplicates(filename):
    filepath = os.path.join(DATA_DIR, filename)
    print("\n" + "=" * 95)
    print(f"AUDITING DUPLICATES FOR: {filename}")
    print("=" * 95)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    # Load dataset using pandas
    df = pd.read_csv(filepath, sep=r"\s+", header=None, names=COLUMN_NAMES)
    total_rows = len(df)

    # -------------------------------------------------------------------------
    # 1. FULL-ROW DUPLICATE CHECK (Across all 26 columns)
    # -------------------------------------------------------------------------
    # df.duplicated() returns True for rows that are exact duplicates of an earlier row
    full_row_dup_mask = df.duplicated()
    full_row_dup_count = int(full_row_dup_mask.sum())

    print(f"Total Rows Ingested:                    {total_rows:,}")
    print(f"Full-Row Duplicate Count (All 26 Cols): {full_row_dup_count}")

    if full_row_dup_count > 0:
        print("  [!] FULL-ROW DUPLICATES FOUND:")
        dup_indices = df[full_row_dup_mask].index.tolist()
        print(f"      Affected row indices (first 10): {dup_indices[:10]}")
        print("      Sample duplicate rows:")
        print(df[df.duplicated(keep=False)].head(6))
    else:
        print("  -> Full-row status: ZERO duplicates across all 26 columns.")

    # -------------------------------------------------------------------------
    # 2. DUPLICATE ENGINE-CYCLE COMBINATIONS CHECK
    # -------------------------------------------------------------------------
    # In time-series degradation data, (unit_number, time_cycles) must serve as a
    # unique composite primary key. No engine should have two records for cycle t.
    key_dup_mask = df.duplicated(subset=["unit_number", "time_cycles"])
    key_dup_count = int(key_dup_mask.sum())

    print(f"\nDuplicate (unit_number, time_cycles) Keys: {key_dup_count}")

    if key_dup_count > 0:
        print("  [!] DUPLICATE ENGINE-CYCLE COMBINATIONS FOUND:")
        bad_keys_df = df[df.duplicated(subset=["unit_number", "time_cycles"], keep=False)]
        print(f"      Total records involved in key collisions: {len(bad_keys_df)}")
        print("      Sample collision rows (unit_number, time_cycles, row index):")
        for idx, row in bad_keys_df.head(10).iterrows():
            print(f"        Row {idx}: Engine {int(row['unit_number'])}, Cycle {int(row['time_cycles'])}")
    else:
        print("  -> Engine-cycle key status: ZERO key collisions (each engine has exactly 1 row per cycle).")

    # -------------------------------------------------------------------------
    # 3. OVERALL FILE VERDICT
    # -------------------------------------------------------------------------
    status = "CLEAN" if (full_row_dup_count == 0 and key_dup_count == 0) else "DUPLICATES DETECTED"
    print(f"\nFile Verdict: {status}")

    return {
        "filename": filename,
        "rows": total_rows,
        "full_row_duplicates": full_row_dup_count,
        "key_duplicates": key_dup_count,
        "status": status
    }


def main():
    print("=" * 105)
    print("PHASE 2, STEP 2.6: DUPLICATE ROWS & DUPLICATE ENGINE-CYCLE AUDIT")
    print("=" * 105)

    results = []
    for fname in FILES_TO_CHECK:
        res = audit_duplicates(fname)
        results.append(res)

    # -------------------------------------------------------------------------
    # SUMMARY COMPARISON TABLE
    # -------------------------------------------------------------------------
    print("\n" + "=" * 105)
    print("SUMMARY COMPARISON TABLE: DUPLICATE AUDIT ACROSS ALL 8 FILES")
    print("=" * 105)
    header = (
        f"{'Filename':<17} | {'Total Rows':<11} | {'Full-Row Dups':<15} | "
        f"{'Engine-Cycle Key Dups':<22} | {'Overall Status':<15}"
    )
    print(header)
    print("-" * len(header))

    all_clean = True
    for r in results:
        if r["status"] != "CLEAN":
            all_clean = False
        row_str = (
            f"{r['filename']:<17} | {r['rows']:<11,d} | {r['full_row_duplicates']:<15} | "
            f"{r['key_duplicates']:<22} | {r['status']:<15}"
        )
        print(row_str)

    print("=" * 105)
    if all_clean:
        print("FINAL VERDICT: ALL 8 FILES ARE 100% UNIQUE. NO DUPLICATE ROWS OR KEY COLLISIONS EXIST.")
    else:
        print("FINAL VERDICT: DUPLICATE ISSUES DETECTED. DO NOT DELETE AUTOMATICALLY.")
    print("=" * 105)


if __name__ == "__main__":
    main()
