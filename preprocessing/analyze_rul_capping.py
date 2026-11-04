"""
=============================================================================
AIRCRAFT ENGINE AI - HEALTH MONITORING & PREDICTIVE MAINTENANCE
Phase 3: Data Preprocessing and Feature Engineering
Step 3.4: Evaluate and Compare RUL Capping Strategies
=============================================================================

This script evaluates three RUL target formulations across all 8 processed
C-MAPSS datasets (train and test for FD001, FD002, FD003, FD004):
  1. Raw linear RUL (no cap)
  2. Piecewise capped RUL with cap = 125 cycles
  3. Piecewise capped RUL with cap = 130 cycles

Formula:
  RUL_capped = min(RUL_raw, RUL_cap)

Outputs generated:
  - reports/eda/rul_capping_comparison.csv

Strict restrictions:
  - Does NOT modify processed datasets.
  - Does NOT overwrite raw RUL labels.
  - Does NOT create capped datasets yet.
=============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
SPLITS = ["train", "test"]
CAPS = [None, 125, 130]


def analyze_capping(base_dir: str) -> pd.DataFrame:
    proc_dir = os.path.join(base_dir, "dataset", "processed")
    eda_dir = os.path.join(base_dir, "reports", "eda")
    os.makedirs(eda_dir, exist_ok=True)

    records = []

    for subset in SUBSETS:
        for split in SPLITS:
            filename = f"{split}_{subset}_with_RUL.csv"
            filepath = os.path.join(proc_dir, filename)

            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Processed file not found: {filepath}")

            df = pd.read_csv(filepath, usecols=["unit_number", "time_cycles", "RUL"])
            rul_raw = df["RUL"]
            total_obs = len(rul_raw)
            total_engines = df["unit_number"].nunique()

            for cap in CAPS:
                if cap is None:
                    strategy_name = "Raw Linear (No Cap)"
                    rul_eval = rul_raw
                    affected_count = 0
                    affected_pct = 0.0
                    cap_val = np.nan
                else:
                    strategy_name = f"Piecewise Cap ({cap})"
                    rul_eval = np.minimum(rul_raw, cap)
                    affected_count = int((rul_raw > cap).sum())
                    affected_pct = round((affected_count / total_obs) * 100, 2)
                    cap_val = cap

                records.append({
                    "subset": subset,
                    "split": split,
                    "strategy": strategy_name,
                    "cap_threshold": cap_val if not np.isnan(cap_val) else "None",
                    "total_engines": total_engines,
                    "total_observations": total_obs,
                    "affected_observations": affected_count,
                    "affected_percentage": affected_pct,
                    "min_rul": int(rul_eval.min()),
                    "max_rul": int(rul_eval.max()),
                    "mean_rul": round(float(rul_eval.mean()), 2),
                    "median_rul": round(float(rul_eval.median()), 2),
                    "std_rul": round(float(rul_eval.std()), 2),
                    "unique_rul_values": int(rul_eval.nunique()),
                })

    df_results = pd.DataFrame(records)

    # Save to reports/eda/rul_capping_comparison.csv
    csv_out = os.path.join(eda_dir, "rul_capping_comparison.csv")
    df_results.to_csv(csv_out, index=False)
    print(f"Saved capping comparison CSV to: {csv_out}")

    return df_results


def main():
    print("=" * 80)
    print("PHASE 3 - STEP 3.4: RUL CAPPING STRATEGY EVALUATION")
    print("=" * 80)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df_results = analyze_capping(base_dir)

    print("\n" + "=" * 80)
    print("RUL CAPPING DISTRIBUTION & IMPACT COMPARISON TABLE")
    print("=" * 80)
    cols_to_print = [
        "subset", "split", "strategy", "affected_percentage",
        "min_rul", "max_rul", "mean_rul", "median_rul", "std_rul", "unique_rul_values"
    ]
    print(df_results[cols_to_print].to_string(index=False))


if __name__ == "__main__":
    main()
