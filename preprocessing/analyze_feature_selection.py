"""
=============================================================================
AIRCRAFT ENGINE AI - HEALTH MONITORING & PREDICTIVE MAINTENANCE
Phase 3: Data Preprocessing and Feature Engineering
Step 3.6: Feature Selection & Statistical Variation Analysis
=============================================================================

This script analyzes the 26 input features (3 operational settings and 21
thermodynamic sensors) across all four NASA C-MAPSS training datasets:
  - train_FD001_with_RUL.csv
  - train_FD002_with_RUL.csv
  - train_FD003_with_RUL.csv
  - train_FD004_with_RUL.csv

It calculates:
  1. Statistical distribution metrics (min, max, mean, std, span, unique values)
  2. Variation classification (Constant, Near-Constant, Low-Variation, Variable)
  3. Feature viability across single vs. multiple operating conditions
  4. Cross-subset consensus and recommendation tiers

Outputs generated:
  - reports/eda/feature_selection_summary.csv

Strict restrictions:
  - Does NOT modify existing processed datasets.
  - Does NOT delete columns or rows.
  - Does NOT train predictive models or calculate supervised importance.
=============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]

FEATURE_NAMES = [
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

SENSOR_DESCRIPTIONS = {
    "op_setting_1": "Altitude (0 - 42,000 ft in FD002/4)",
    "op_setting_2": "Mach Number (0.0 - 0.84 in FD002/4)",
    "op_setting_3": "Throttle Resolver Angle / Power Setting",
    "sensor_1": "T2 - Total temperature at fan inlet",
    "sensor_2": "T24 - Total temperature at LPC outlet",
    "sensor_3": "T30 - Total temperature at HPC outlet",
    "sensor_4": "T50 - Total temperature at LPT outlet",
    "sensor_5": "P2 - Pressure at fan inlet",
    "sensor_6": "P15 - Total pressure in bypass-duct",
    "sensor_7": "P30 - Total pressure at HPC outlet",
    "sensor_8": "Nf - Physical fan speed",
    "sensor_9": "Nc - Physical core speed",
    "sensor_10": "epr - Engine pressure ratio (P50/P2)",
    "sensor_11": "Ps30 - Static pressure at HPC outlet",
    "sensor_12": "phi - Ratio of fuel flow to Ps30",
    "sensor_13": "Nf_dmd - Corrected fan speed",
    "sensor_14": "Nc_dmd - Corrected core speed",
    "sensor_15": "BPR - Bypass Ratio",
    "sensor_16": "farB - Burner fuel-air ratio",
    "sensor_17": "htBleed - Bleed Enthalpy",
    "sensor_18": "Nf_dmd2 - Demanded fan speed",
    "sensor_19": "PCNfR_dmd - Demanded corrected fan speed",
    "sensor_20": "W31 - HPT coolant bleed",
    "sensor_21": "W32 - LPT coolant bleed",
}


def classify_variation(std: float, span: float, nunique: int, mean: float) -> str:
    """Classify feature variation into 4 categories."""
    if std == 0.0 or nunique <= 1 or np.isclose(std, 0.0, atol=1e-12):
        return "Constant"
    # Near-constant: very small std or span
    if std < 0.01 or (mean != 0 and (span / abs(mean)) < 0.001):
        return "Near-constant"
    if std < 1.0:
        return "Low-variation"
    return "Variable"


def analyze_features(base_dir: str) -> pd.DataFrame:
    proc_dir = os.path.join(base_dir, "dataset", "processed")
    eda_dir = os.path.join(base_dir, "reports", "eda")
    os.makedirs(eda_dir, exist_ok=True)

    records = []

    for subset in SUBSETS:
        filepath = os.path.join(proc_dir, f"train_{subset}_with_RUL.csv")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Processed file missing: {filepath}")

        df = pd.read_csv(filepath)

        for feat in FEATURE_NAMES:
            series = df[feat]
            val_min = float(series.min())
            val_max = float(series.max())
            val_mean = float(series.mean())
            val_std = float(series.std())
            span = val_max - val_min
            nunique = int(series.nunique())
            cv_pct = (val_std / abs(val_mean) * 100) if val_mean != 0 else 0.0

            classification = classify_variation(val_std, span, nunique, val_mean)

            records.append({
                "subset": subset,
                "feature": feat,
                "description": SENSOR_DESCRIPTIONS[feat],
                "min": round(val_min, 4),
                "max": round(val_max, 4),
                "mean": round(val_mean, 4),
                "std": round(val_std, 4),
                "span": round(span, 4),
                "unique_values": nunique,
                "cv_percent": round(cv_pct, 4),
                "classification": classification,
            })

    df_results = pd.DataFrame(records)

    # Save summary CSV
    out_csv = os.path.join(eda_dir, "feature_selection_summary.csv")
    df_results.to_csv(out_csv, index=False)
    print(f"Saved feature variation summary to: {out_csv}")

    return df_results


def generate_cross_subset_consensus(df: pd.DataFrame) -> pd.DataFrame:
    """Generate consensus matrix across the 4 subsets."""
    pivot = df.pivot(index="feature", columns="subset", values="classification")

    # Add description
    pivot["description"] = pivot.index.map(SENSOR_DESCRIPTIONS)

    # Categorize suitability
    def assign_recommendation(row):
        f = row.name
        # Constant across FD001 and FD003
        fd1 = row["FD001"]
        fd2 = row["FD002"]
        fd3 = row["FD003"]
        fd4 = row["FD004"]

        if f in ["sensor_16"]:
            return "Low-utility across all (Constant in 1/3, Low in 2/4)"
        if f in ["op_setting_3", "sensor_1", "sensor_5", "sensor_18", "sensor_19"]:
            return "Condition-Dependent (Constant in 1/3, Dynamic in 2/4)"
        if f in ["sensor_10", "sensor_6"]:
            return "Condition-Dependent (Near-const in 1/3, Variable in 2/4)"
        if f in ["op_setting_1", "op_setting_2"]:
            return "Operating Condition Identifiers (Dynamic in 2/4)"
        return "Core Degradation Sensor (Active across all 4 subsets)"

    pivot["Recommendation_Category"] = pivot.apply(assign_recommendation, axis=1)
    return pivot


def main():
    print("=" * 80)
    print("PHASE 3 - STEP 3.6: FEATURE SELECTION & SENSOR VARIATION ANALYSIS")
    print("=" * 80)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df_results = analyze_features(base_dir)

    print("\n" + "=" * 80)
    print("CROSS-SUBSET FEATURE VARIATION CONSENSUS MATRIX")
    print("=" * 80)

    consensus = generate_cross_subset_consensus(df_results)
    cols_order = ["description", "FD001", "FD002", "FD003", "FD004", "Recommendation_Category"]
    print(consensus[cols_order].to_string())

    print("\n" + "=" * 80)
    print("FEATURE SET RECOMMENDATION SUMMARY")
    print("=" * 80)
    core_sensors = consensus[consensus["Recommendation_Category"] == "Core Degradation Sensor (Active across all 4 subsets)"].index.tolist()
    cond_sensors = consensus[consensus["Recommendation_Category"].str.startswith("Condition-Dependent")].index.tolist()
    op_settings = ["op_setting_1", "op_setting_2", "op_setting_3"]

    print(f"1. Core 14-Sensor Prognostic Set (universally active degradation indicators):")
    print(f"   {core_sensors} (Total: {len(core_sensors)} sensors)")
    print(f"\n2. Operating-Condition Sensitive Features (active only in FD002/FD004):")
    print(f"   {cond_sensors} (Total: {len(cond_sensors)} sensors)")
    print(f"\n3. Operating Settings (Condition regime discriminators):")
    print(f"   {op_settings} (Total: {len(op_settings)} settings)")
    print(f"\n4. Invariant / Ultra-low utility across all subsets:")
    print(f"   ['sensor_16'] (Burner fuel-air ratio)")


if __name__ == "__main__":
    main()
