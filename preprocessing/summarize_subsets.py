"""
=============================================================================
AIRCRAFT ENGINE AI - HEALTH MONITORING & PREDICTIVE MAINTENANCE
Phase 2: Dataset Understanding & Exploratory Data Analysis (EDA)
Step 2.4: Subset Statistical Summary (Rows, Columns, Engines, Cycles)
=============================================================================
This script computes comprehensive statistics across all four C-MAPSS subsets
(FD001, FD002, FD003, FD004) for both training and testing datasets, as well
as the corresponding ground-truth RUL files.
=============================================================================
"""

import os
import pandas as pd

DATA_DIR = os.path.join("dataset", "raw", "C-MAPSS")
SUBSETS = ["FD001", "FD002", "FD003", "FD004"]

COLUMN_NAMES = [
    "unit_number", "time_cycles",
    "op_setting_1", "op_setting_2", "op_setting_3",
    "sensor_1", "sensor_2", "sensor_3", "sensor_4", "sensor_5",
    "sensor_6", "sensor_7", "sensor_8", "sensor_9", "sensor_10",
    "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
    "sensor_16", "sensor_17", "sensor_18", "sensor_19", "sensor_20", "sensor_21"
]


def analyze_trajectory_file(filename):
    """Loads a train or test file and computes row, column, engine, and cycle statistics."""
    filepath = os.path.join(DATA_DIR, filename)
    df = pd.read_csv(filepath, sep=r"\s+", header=None, names=COLUMN_NAMES)
    
    rows, cols = df.shape
    unique_engines = df["unit_number"].nunique()
    min_cycle_overall = df["time_cycles"].min()
    max_cycle_overall = df["time_cycles"].max()
    
    # Calculate cycle duration for each engine (the max cycle reached by each unit)
    cycles_per_engine = df.groupby("unit_number")["time_cycles"].max()
    min_engine_cycles = cycles_per_engine.min()
    max_engine_cycles = cycles_per_engine.max()
    avg_engine_cycles = cycles_per_engine.mean()
    
    return {
        "file": filename,
        "rows": rows,
        "cols": cols,
        "unique_engines": unique_engines,
        "min_cycle_num": min_cycle_overall,
        "max_cycle_num": max_cycle_overall,
        "min_engine_cycles": min_engine_cycles,
        "max_engine_cycles": max_engine_cycles,
        "avg_engine_cycles": avg_engine_cycles
    }


def analyze_rul_file(filename):
    """Loads an RUL file and returns the total count of ground-truth values."""
    filepath = os.path.join(DATA_DIR, filename)
    df = pd.read_csv(filepath, sep=r"\s+", header=None, names=["RUL"])
    return {
        "file": filename,
        "rul_count": len(df),
        "min_rul": df["RUL"].min(),
        "max_rul": df["RUL"].max(),
        "avg_rul": df["RUL"].mean()
    }


def main():
    print("=" * 110)
    print("STEP 2.4: NASA C-MAPSS SUBSET STATISTICAL SUMMARY (FD001 - FD004)")
    print("=" * 110)

    train_stats = []
    test_stats = []
    rul_stats = {}

    for sub in SUBSETS:
        train_file = f"train_{sub}.txt"
        test_file = f"test_{sub}.txt"
        rul_file = f"RUL_{sub}.txt"

        print(f"\nProcessing subset {sub}...")
        train_stats.append(analyze_trajectory_file(train_file))
        test_stats.append(analyze_trajectory_file(test_file))
        rul_stats[sub] = analyze_rul_file(rul_file)

    # -------------------------------------------------------------------------
    # 1. DISPLAY DETAILED PER-FILE RESULTS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 110)
    print("DETAILED STATISTICS PER SUBSET")
    print("=" * 110)

    for i, sub in enumerate(SUBSETS):
        tr = train_stats[i]
        te = test_stats[i]
        ru = rul_stats[sub]

        print(f"\n--- SUBSET: {sub} ---")
        print(f"  [TRAINING SET - {tr['file']}]")
        print(f"    Total Rows:            {tr['rows']:,}")
        print(f"    Total Columns:         {tr['cols']}")
        print(f"    Unique Engines:        {tr['unique_engines']}")
        print(f"    Cycle Range (All):     Cycle {tr['min_cycle_num']} to Cycle {tr['max_cycle_num']}")
        print(f"    Shortest Engine Life:  {tr['min_engine_cycles']} cycles")
        print(f"    Longest Engine Life:   {tr['max_engine_cycles']} cycles")
        print(f"    Average Engine Life:   {tr['avg_engine_cycles']:.2f} cycles")
        
        print(f"  [TESTING SET - {te['file']}]")
        print(f"    Total Rows:            {te['rows']:,}")
        print(f"    Total Columns:         {te['cols']}")
        print(f"    Unique Engines:        {te['unique_engines']}")
        print(f"    Cycle Range (All):     Cycle {te['min_cycle_num']} to Cycle {te['max_cycle_num']}")
        print(f"    Shortest Test Cutoff:  {te['min_engine_cycles']} cycles")
        print(f"    Longest Test Cutoff:   {te['max_engine_cycles']} cycles")
        print(f"    Average Test Cutoff:   {te['avg_engine_cycles']:.2f} cycles")

        print(f"  [GROUND-TRUTH RUL - {ru['file']}]")
        print(f"    Total Target Values:   {ru['rul_count']} (Matches Test Engines: {ru['rul_count'] == te['unique_engines']})")
        print(f"    RUL Range:             {ru['min_rul']} to {ru['max_rul']} cycles (Mean: {ru['avg_rul']:.2f} cycles)")

    # -------------------------------------------------------------------------
    # 2. COMPARISON TABLE
    # -------------------------------------------------------------------------
    print("\n" + "=" * 110)
    print("SUMMARY COMPARISON TABLE (TRAINING VS. TESTING)")
    print("=" * 110)
    header = (
        f"{'Subset':<7} | {'Split':<6} | {'Rows':<8} | {'Cols':<5} | {'Engines':<8} | "
        f"{'Cycle Range':<13} | {'Min Engine':<11} | {'Max Engine':<11} | {'Avg Engine':<11} | {'RUL Count':<9}"
    )
    print(header)
    print("-" * len(header))

    for i, sub in enumerate(SUBSETS):
        tr = train_stats[i]
        te = test_stats[i]
        ru = rul_stats[sub]

        tr_row = (
            f"{sub:<7} | {'Train':<6} | {tr['rows']:<8} | {tr['cols']:<5} | {tr['unique_engines']:<8} | "
            f"{tr['min_cycle_num']}-{tr['max_cycle_num']:<11} | {tr['min_engine_cycles']:<11} | "
            f"{tr['max_engine_cycles']:<11} | {tr['avg_engine_cycles']:<11.2f} | {'N/A (Fails)':<9}"
        )
        te_row = (
            f"{sub:<7} | {'Test':<6} | {te['rows']:<8} | {te['cols']:<5} | {te['unique_engines']:<8} | "
            f"{te['min_cycle_num']}-{te['max_cycle_num']:<11} | {te['min_engine_cycles']:<11} | "
            f"{te['max_engine_cycles']:<11} | {te['avg_engine_cycles']:<11.2f} | {ru['rul_count']:<9}"
        )
        print(tr_row)
        print(te_row)
        if i < len(SUBSETS) - 1:
            print("-" * len(header))

    print("=" * 110)
    print("STEP 2.4 COMPLETED SUCCESSFULLY.")
    print("=" * 110)


if __name__ == "__main__":
    main()
