"""
=============================================================================
AIRCRAFT ENGINE AI - HEALTH MONITORING & PREDICTIVE MAINTENANCE
Phase 2: Dataset Understanding & Exploratory Data Analysis (EDA)
Step 2.3: Inspect First and Last Five Rows of All 12 Dataset Files
=============================================================================
This script loads all 12 NASA C-MAPSS dataset files using pandas:
- 4 Training files (train_FD001 to train_FD004)
- 4 Testing files (test_FD001 to test_FD004)
- 4 Ground-Truth RUL files (RUL_FD001 to RUL_FD004)

For each file, it displays:
1. Filename and file type (Training / Testing / Ground-Truth RUL)
2. Total dimensions (shape)
3. First five rows (head)
4. Last five rows (tail)
=============================================================================
"""

import os
import pandas as pd

# Set display options to clearly see all 26 columns in terminal
pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 1000)

DATA_DIR = os.path.join("dataset", "raw", "C-MAPSS")

COLUMN_NAMES = [
    "unit_number", "time_cycles",
    "op_setting_1", "op_setting_2", "op_setting_3",
    "sensor_1", "sensor_2", "sensor_3", "sensor_4", "sensor_5",
    "sensor_6", "sensor_7", "sensor_8", "sensor_9", "sensor_10",
    "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
    "sensor_16", "sensor_17", "sensor_18", "sensor_19", "sensor_20", "sensor_21"
]

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]


def inspect_file(filename, file_type):
    filepath = os.path.join(DATA_DIR, filename)
    print("=" * 100)
    print(f"FILE: {filename}  |  TYPE: {file_type}")
    print("=" * 100)
    
    if not os.path.exists(filepath):
        print(f"ERROR: File not found at {filepath}")
        return

    if file_type == "Ground-Truth Remaining Useful Life (RUL)":
        df = pd.read_csv(filepath, sep=r"\s+", header=None, names=["RUL_cycles"])
        print(f"Shape: {df.shape[0]} test engines x {df.shape[1]} column\n")
        print("--- First 5 Ground-Truth RUL Values (Test Engines 1 to 5) ---")
        print(df.head())
        print("\n--- Last 5 Ground-Truth RUL Values (Final 5 Test Engines) ---")
        print(df.tail())
    else:
        df = pd.read_csv(filepath, sep=r"\s+", header=None, names=COLUMN_NAMES)
        unique_engines = df["unit_number"].nunique()
        print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns | Unique Engines: {unique_engines}\n")
        print("--- First 5 Rows (Engine #1, initial cycles 1 to 5) ---")
        print(df.head())
        print("\n--- Last 5 Rows (Final engine, final recorded cycles) ---")
        print(df.tail())
    print("\n")


def main():
    print("\n" + "#" * 100)
    print("SECTION 1: TRAINING DATASETS (Run-to-Failure Trajectories)")
    print("#" * 100)
    for sub in SUBSETS:
        inspect_file(f"train_{sub}.txt", "Training Trajectory (Run-to-Failure)")

    print("\n" + "#" * 100)
    print("SECTION 2: TESTING DATASETS (Truncated Trajectories)")
    print("#" * 100)
    for sub in SUBSETS:
        inspect_file(f"test_{sub}.txt", "Testing Trajectory (Truncated Prior to Failure)")

    print("\n" + "#" * 100)
    print("SECTION 3: GROUND-TRUTH RUL FILES")
    print("#" * 100)
    for sub in SUBSETS:
        inspect_file(f"RUL_{sub}.txt", "Ground-Truth Remaining Useful Life (RUL)")


if __name__ == "__main__":
    main()
