"""
=============================================================================
AIRCRAFT ENGINE AI - HEALTH MONITORING & PREDICTIVE MAINTENANCE
Phase 2: Dataset Understanding & Exploratory Data Analysis (EDA)
Step 2.1: Dataset Inspection Script
=============================================================================

This script loads the raw NASA C-MAPSS FD001 training dataset using pandas,
assigns meaningful column names to all 26 variables, and displays fundamental
structural information (shape, column names, head, tail, and info).

Dataset location: dataset/raw/C-MAPSS/train_FD001.txt
=============================================================================
"""

import os
import pandas as pd

# Set pandas display options so all 26 columns can be seen clearly in terminal
pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 1000)


def main():
    print("=" * 80)
    print("STEP 2.1: NASA C-MAPSS DATASET INSPECTION (train_FD001.txt)")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. DEFINE PATH TO THE DATASET
    # -------------------------------------------------------------------------
    # We construct the path relative to the project root directory.
    # dataset/raw/C-MAPSS/ contains the original, untouched dataset files.
    file_path = os.path.join("dataset", "raw", "C-MAPSS", "train_FD001.txt")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")

    print(f"\n[1] Loading dataset from: {file_path}")

    # -------------------------------------------------------------------------
    # 2. DEFINE MEANINGFUL COLUMN NAMES
    # -------------------------------------------------------------------------
    # The C-MAPSS dataset has 26 space-separated columns:
    #   - Col 1: Unit / Engine identifier (unit_number)
    #   - Col 2: Time in flight cycles (time_cycles)
    #   - Col 3-5: Operational settings (op_setting_1, op_setting_2, op_setting_3)
    #   - Col 6-26: 21 thermodynamic sensor measurements (sensor_1 to sensor_21)
    column_names = [
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

    # -------------------------------------------------------------------------
    # 3. LOAD DATASET USING PANDAS
    # -------------------------------------------------------------------------
    # We use sep=r'\s+' because the text file has one or more spaces between numbers
    # header=None tells pandas that the raw file does not have a header row
    # names=column_names assigns our 26 descriptive column names
    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        names=column_names
    )

    print("Dataset loaded successfully!")

    # -------------------------------------------------------------------------
    # 4. DATASET SHAPE (ROWS AND COLUMNS)
    # -------------------------------------------------------------------------
    rows, cols = df.shape
    print("\n" + "-" * 80)
    print(f"[2] DATASET SHAPE: {rows} rows x {cols} columns")
    print("-" * 80)
    print(f"Total operational cycle records (rows): {rows:,}")
    print(f"Total measured variables (columns):    {cols}")

    # -------------------------------------------------------------------------
    # 5. COLUMN NAMES LIST
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("[3] COLUMN NAMES (26 variables):")
    print("-" * 80)
    for i, col in enumerate(df.columns, start=1):
        print(f"  Column {i:>2}: {col}")

    # -------------------------------------------------------------------------
    # 6. FIRST FIVE ROWS (HEAD)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("[4] FIRST 5 ROWS (df.head()):")
    print("-" * 80)
    print(df.head())

    # -------------------------------------------------------------------------
    # 7. LAST FIVE ROWS (TAIL)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("[5] LAST 5 ROWS (df.tail()):")
    print("-" * 80)
    print(df.tail())

    # -------------------------------------------------------------------------
    # 8. DATASET INFORMATION (INFO)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("[6] DATASET INFORMATION (df.info()):")
    print("-" * 80)
    df.info()

    print("\n" + "=" * 80)
    print("STEP 2.1 COMPLETED SUCCESSFULLY: train_FD001.txt loaded & inspected.")
    print("=" * 80)


if __name__ == "__main__":
    main()
