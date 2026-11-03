"""
=============================================================================
AIRCRAFT ENGINE AI - HEALTH MONITORING & PREDICTIVE MAINTENANCE
Phase 2: Dataset Understanding & Exploratory Data Analysis (EDA)
Step 2.8: Comprehensive Summary Statistics & Variability Audit
=============================================================================
This script computes descriptive statistics (count, mean, std, min, Q1, median,
Q3, max) for every column across all 8 trajectory files (train_FD001 to train_FD004
and test_FD001 to test_FD004).

Key features:
1. Calculates Coefficient of Variation (CV% = std / |mean| * 100) for scale-free
   relative variability assessment.
2. Programmatically classifies sensors into strict thresholds:
   - Zero standard deviation (std == 0.0)
   - Low standard deviation (0.0 < std < 0.05)
   - High standard deviation (std > 50.0)
3. Compares both Mean and Standard Deviation across ALL 21 sensors between
   Train and Test within each FD subset separately.
4. Saves complete summary tables to reports/eda/.
=============================================================================
"""

import os
import pandas as pd

DATA_DIR = os.path.join("dataset", "raw", "C-MAPSS")
OUTPUT_DIR = os.path.join("reports", "eda")

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]

COLUMN_NAMES = [
    "unit_number", "time_cycles",
    "op_setting_1", "op_setting_2", "op_setting_3",
    "sensor_1", "sensor_2", "sensor_3", "sensor_4", "sensor_5",
    "sensor_6", "sensor_7", "sensor_8", "sensor_9", "sensor_10",
    "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
    "sensor_16", "sensor_17", "sensor_18", "sensor_19", "sensor_20", "sensor_21"
]

SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]


def compute_summary_statistics(filename):
    """Loads a trajectory file, computes 8 descriptive statistics plus CV%,
    saves results to CSV in reports/eda/, and returns the DataFrame."""
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    df = pd.read_csv(filepath, sep=r"\s+", header=None, names=COLUMN_NAMES)
    
    # Compute base statistics
    stats_df = df.describe(percentiles=[0.25, 0.50, 0.75]).T
    stats_df = stats_df[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
    stats_df.index.name = "column_name"
    stats_df = stats_df.reset_index()

    # Calculate Coefficient of Variation (CV% = std / |mean| * 100) where mean != 0
    cv_list = []
    for _, row in stats_df.iterrows():
        m = row["mean"]
        s = row["std"]
        if abs(m) > 1e-12:
            cv_list.append((s / abs(m)) * 100.0)
        else:
            cv_list.append(0.0)
    stats_df["cv_pct"] = cv_list

    # Save to reports/eda/
    base_name = os.path.splitext(filename)[0]
    out_csv = os.path.join(OUTPUT_DIR, f"{base_name}_summary.csv")
    stats_df.to_csv(out_csv, index=False)
    
    return df, stats_df, out_csv


def print_formatted_summary(filename, stats_df):
    """Displays a clean summary table in the terminal."""
    print("\n" + "=" * 120)
    print(f"DESCRIPTIVE STATISTICS: {filename}")
    print("=" * 120)
    header = (
        f"{'Column':<14} | {'Count':<7} | {'Mean':<10} | {'Std Dev':<10} | {'CV (%)':<8} | "
        f"{'Min':<10} | {'Median':<10} | {'Max':<10}"
    )
    print(header)
    print("-" * len(header))

    for _, row in stats_df.iterrows():
        c_name = row["column_name"]
        cnt = int(row["count"])
        mean_val = f"{row['mean']:.4f}"
        std_val = f"{row['std']:.4f}"
        cv_val = f"{row['cv_pct']:.3f}%"
        min_val = f"{row['min']:.4f}"
        med_val = f"{row['50%']:.4f}"
        max_val = f"{row['max']:.4f}"

        print(
            f"{c_name:<14} | {cnt:<7} | {mean_val:<10} | {std_val:<10} | {cv_val:<8} | "
            f"{min_val:<10} | {med_val:<10} | {max_val:<10}"
        )


def analyze_sensor_variability(stats_df):
    """Strictly categorizes sensors based on exact mathematical criteria."""
    sensor_stats = stats_df[stats_df["column_name"].isin(SENSOR_COLS)]
    
    # Exact threshold filters
    zero_std = sensor_stats[sensor_stats["std"] == 0.0]["column_name"].tolist()
    low_std = sensor_stats[(sensor_stats["std"] > 0.0) & (sensor_stats["std"] < 0.05)]["column_name"].tolist()
    high_std = sensor_stats[sensor_stats["std"] > 50.0]["column_name"].tolist()
    mod_std = sensor_stats[(sensor_stats["std"] >= 0.05) & (sensor_stats["std"] <= 50.0)]["column_name"].tolist()

    return zero_std, low_std, mod_std, high_std


def main():
    print("=" * 120)
    print("PHASE 2, STEP 2.8: DESCRIPTIVE SUMMARY STATISTICS & SENSOR AUDIT")
    print("=" * 120)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_stats = {}
    csv_paths = {}

    for sub in SUBSETS:
        for split in ["train", "test"]:
            fname = f"{split}_{sub}.txt"
            df, stats_df, out_csv = compute_summary_statistics(fname)
            all_stats[fname] = stats_df
            csv_paths[fname] = out_csv
            print_formatted_summary(fname, stats_df)

    # -------------------------------------------------------------------------
    # 1. STRICT SENSOR VARIABILITY CLASSIFICATION (PER DATASET)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 120)
    print("STRICT SENSOR VARIABILITY CLASSIFICATION (EXACT THRESHOLDS)")
    print("  - Zero Standard Deviation: std == 0.0")
    print("  - Low Standard Deviation:  0.0 < std < 0.05")
    print("  - Moderate Std Deviation:  0.05 <= std <= 50.0")
    print("  - High Standard Deviation: std > 50.0")
    print("=" * 120)

    for fname, stats_df in all_stats.items():
        zero_s, low_s, mod_s, high_s = analyze_sensor_variability(stats_df)
        print(f"\n--- {fname} ---")
        print(f"  Zero Std (std == 0.0):         Count: {len(zero_s)} -> {zero_s if zero_s else 'None'}")
        print(f"  Low Std (0.0 < std < 0.05):    Count: {len(low_s)} -> {low_s if low_s else 'None'}")
        print(f"  Moderate Std (0.05 <= std <= 50): Count: {len(mod_s)} -> {mod_s}")
        print(f"  High Std (std > 50.0):         Count: {len(high_s)} -> {high_s if high_s else 'None'}")

    # -------------------------------------------------------------------------
    # 2. COMPLETE TRAIN VS. TEST COMPARISON (ALL 21 SENSORS PER SUBSET)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 120)
    print("COMPLETE TRAIN VS. TEST COMPARISON FOR ALL 21 SENSORS (MEAN & STD)")
    print("=" * 120)

    for sub in SUBSETS:
        tr_stats = all_stats[f"train_{sub}.txt"].set_index("column_name")
        te_stats = all_stats[f"test_{sub}.txt"].set_index("column_name")

        print(f"\n[SUBSET: {sub} - Full 21-Sensor Train vs. Test Comparison]")
        comp_header = (
            f"{'Sensor':<11} | {'Train Mean':<11} | {'Test Mean':<11} | {'Mean Diff':<11} | "
            f"{'Train Std':<11} | {'Test Std':<11} | {'Std Diff':<11}"
        )
        print(comp_header)
        print("-" * len(comp_header))

        for s in SENSOR_COLS:
            tr_m = tr_stats.loc[s, "mean"]
            te_m = te_stats.loc[s, "mean"]
            diff_m = te_m - tr_m

            tr_s = tr_stats.loc[s, "std"]
            te_s = te_stats.loc[s, "std"]
            diff_s = te_s - tr_s

            print(
                f"{s:<11} | {tr_m:<11.4f} | {te_m:<11.4f} | {diff_m:<+11.4f} | "
                f"{tr_s:<11.4f} | {te_s:<11.4f} | {diff_s:<+11.4f}"
            )

    # -------------------------------------------------------------------------
    # 3. CSV FILES INVENTORY
    # -------------------------------------------------------------------------
    print("\n" + "=" * 120)
    print("GENERATED SUMMARY CSV FILES INVENTORY")
    print("=" * 120)
    for fname, p in csv_paths.items():
        size_bytes = os.path.getsize(p)
        print(f"  - {os.path.basename(p):<25} ({size_bytes:>5} bytes) -> {p}")

    print("\n" + "=" * 120)
    print("STEP 2.8 CORRECTIONS COMPLETED.")
    print("=" * 120)


if __name__ == "__main__":
    main()
