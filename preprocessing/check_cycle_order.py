"""
=============================================================================
AIRCRAFT ENGINE AI - HEALTH MONITORING & PREDICTIVE MAINTENANCE
Phase 2: Dataset Understanding & Exploratory Data Analysis (EDA)
Step 2.7: Cycle Ordering & Sequence Continuity Audit
=============================================================================
This script inspects all 8 trajectory files (train_FD001 to train_FD004 and
test_FD001 to test_FD004) for:
1. Exact consecutive cycle difference: (curr_c - prev_c == 1).
   - If diff <= 0: Flag repeated or reversed cycle.
   - If diff > 1: Flag jump and identify specific missing cycles in between.
2. Separate full-range missing-cycle audit across each engine's lifespan.
3. Engine record grouping (records must be contiguous and ordered 1 to N).
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


def audit_cycle_order(filename):
    filepath = os.path.join(DATA_DIR, filename)
    print("\n" + "=" * 105)
    print(f"AUDITING CYCLE ORDER & SEQUENCE CONTINUITY FOR: {filename}")
    print("=" * 105)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    # Load dataset in its original row order (DO NOT sort)
    df = pd.read_csv(filepath, sep=r"\s+", header=None, names=COLUMN_NAMES)
    total_rows = len(df)
    unique_engines = df["unit_number"].nunique()

    # -------------------------------------------------------------------------
    # 1. ENGINE GROUPING & CONTIGUITY CHECK
    # -------------------------------------------------------------------------
    grouping_violations = []
    seen_engines = []
    cur_engine = None

    for row_idx, unit in enumerate(df["unit_number"]):
        unit = int(unit)
        if unit != cur_engine:
            if unit in seen_engines:
                grouping_violations.append({
                    "engine_id": unit,
                    "row_index": row_idx,
                    "issue": f"Engine {unit} reappeared at row {row_idx} after previous block ended"
                })
            else:
                seen_engines.append(unit)
            cur_engine = unit

    expected_engine_sequence = list(range(1, unique_engines + 1))
    sequence_matches = (seen_engines == expected_engine_sequence)

    print(f"Total Rows:                         {total_rows:,}")
    print(f"Total Unique Engines:               {unique_engines}")
    print(f"Contiguous Engine Blocks:           {'YES (No engines split or interleaved)' if len(grouping_violations) == 0 else 'NO'}")
    print(f"Engine Blocks Sequence (1 to N):    {'YES (Strictly 1, 2, ..., ' + str(unique_engines) + ')' if sequence_matches else 'NO'}")
    print(f"Grouping Violations Detected:       {len(grouping_violations)}")

    # -------------------------------------------------------------------------
    # 2. CONSECUTIVE CYCLE STEP AUDIT (curr_c - prev_c == 1)
    # -------------------------------------------------------------------------
    # Explicitly check whether step difference between consecutive rows equals 1.
    consecutive_step_violations = []
    engines_with_gaps = []
    total_missing_cycles = 0

    # Group by engine without sorting to preserve exact file row order
    grouped = df.groupby("unit_number", sort=False)

    for engine_id, group in grouped:
        cycles_list = group["time_cycles"].tolist()
        row_indices = group.index.tolist()

        # Check that engine begins at cycle 1
        if cycles_list[0] != 1:
            consecutive_step_violations.append({
                "engine_id": engine_id,
                "row_index": row_indices[0],
                "prev_cycle": None,
                "curr_cycle": cycles_list[0],
                "diff": None,
                "issue": f"Engine {engine_id} does not start at cycle 1 (starts at cycle {cycles_list[0]})"
            })

        # Check consecutive cycle differences: curr_c - prev_c
        for i in range(1, len(cycles_list)):
            prev_c = cycles_list[i - 1]
            curr_c = cycles_list[i]
            diff = curr_c - prev_c

            if diff != 1:
                if diff <= 0:
                    issue_desc = (
                        f"Repeated cycle {curr_c} at row {row_indices[i]}" if diff == 0
                        else f"Reversed cycle at row {row_indices[i]}: cycle {curr_c} < previous cycle {prev_c} (diff: {diff})"
                    )
                else:  # diff > 1
                    missing_between = list(range(prev_c + 1, curr_c))
                    issue_desc = (
                        f"Gap at row {row_indices[i]}: jumped from cycle {prev_c} to {curr_c} "
                        f"(diff: {diff}, missing cycles: {missing_between})"
                    )

                consecutive_step_violations.append({
                    "engine_id": engine_id,
                    "row_index": row_indices[i],
                    "prev_cycle": prev_c,
                    "curr_cycle": curr_c,
                    "diff": diff,
                    "issue": issue_desc
                })

        # ---------------------------------------------------------------------
        # 3. SEPARATE FULL-RANGE MISSING CYCLE AUDIT
        # ---------------------------------------------------------------------
        min_c = min(cycles_list)
        max_c = max(cycles_list)
        expected_cycles = set(range(min_c, max_c + 1))
        actual_cycles = set(cycles_list)
        missing_cycles = sorted(list(expected_cycles - actual_cycles))

        if len(missing_cycles) > 0:
            engines_with_gaps.append({
                "engine_id": engine_id,
                "missing_cycles": missing_cycles,
                "count": len(missing_cycles)
            })
            total_missing_cycles += len(missing_cycles)

    print(f"\nConsecutive Step Violations (diff != 1): {len(consecutive_step_violations)}")
    if len(consecutive_step_violations) > 0:
        print("  [!] STEP VIOLATIONS FOUND:")
        for v in consecutive_step_violations[:10]:
            print(f"      - Engine {v['engine_id']} at row index {v['row_index']}: {v['issue']}")
    else:
        print("  -> Consecutive cycle difference check: (curr_c - prev_c == 1) holds 100% across all rows.")

    print(f"Engines with Sequence Gaps:         {len(engines_with_gaps)}")
    print(f"Total Missing Cycles (Gaps):        {total_missing_cycles}")
    if len(engines_with_gaps) > 0:
        print("  [!] SEQUENCE GAPS FOUND:")
        for g in engines_with_gaps[:5]:
            print(f"      - Engine {g['engine_id']}: Missing cycles {g['missing_cycles']}")
    else:
        print("  -> Sequence gap check: ZERO gaps (every integer cycle from 1 to cutoff is present).")

    # Overall Verdict
    is_clean = (len(grouping_violations) == 0 and len(consecutive_step_violations) == 0 and total_missing_cycles == 0)
    status = "CLEAN" if is_clean else "ISSUES DETECTED"
    print(f"\nOverall Verdict for {filename}: {status}")

    return {
        "filename": filename,
        "total_engines": unique_engines,
        "step_violations": len(consecutive_step_violations),
        "engines_with_gaps": len(engines_with_gaps),
        "total_missing_cycles": total_missing_cycles,
        "grouping_violations": len(grouping_violations),
        "status": status
    }


def main():
    print("=" * 105)
    print("PHASE 2, STEP 2.7: CONSECUTIVE CYCLE DIFFERENCE & CONTINUITY AUDIT")
    print("=" * 105)

    results = []
    for fname in FILES_TO_CHECK:
        res = audit_cycle_order(fname)
        results.append(res)

    # -------------------------------------------------------------------------
    # SUMMARY COMPARISON TABLE
    # -------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("SUMMARY COMPARISON TABLE: CYCLE ORDER & CONTINUITY ACROSS ALL 8 FILES")
    print("=" * 115)
    header = (
        f"{'Filename':<17} | {'Total Engines':<14} | {'Step Violations':<17} | "
        f"{'Engines w/ Gaps':<16} | {'Missing Cycles':<15} | {'Grouping Errors':<16} | {'Status':<8}"
    )
    print(header)
    print("-" * len(header))

    all_clean = True
    for r in results:
        if r["status"] != "CLEAN":
            all_clean = False
        row_str = (
            f"{r['filename']:<17} | {r['total_engines']:<14} | {r['step_violations']:<17} | "
            f"{r['engines_with_gaps']:<16} | {r['total_missing_cycles']:<15} | "
            f"{r['grouping_violations']:<16} | {r['status']:<8}"
        )
        print(row_str)

    print("=" * 115)
    if all_clean:
        print("FINAL VERDICT: ALL 8 FILES HAVE STRICT (curr_c - prev_c == 1) INCREMENTS, ZERO GAPS, AND CONTIGUOUS ENGINE BLOCKS.")
    else:
        print("FINAL VERDICT: ISSUES FOUND. DO NOT SORT OR MODIFY FILES.")
    print("=" * 115)


if __name__ == "__main__":
    main()
