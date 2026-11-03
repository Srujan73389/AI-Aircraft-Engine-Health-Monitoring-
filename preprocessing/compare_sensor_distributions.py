"""
=============================================================================
AIRCRAFT ENGINE AI - HEALTH MONITORING & PREDICTIVE MAINTENANCE
Phase 2: Dataset Understanding & Exploratory Data Analysis (EDA)
Step 2.10: Compare Sensor Distributions Across FD001-FD004
=============================================================================
This script statistically compares all 21 sensor measurements across the four
C-MAPSS subsets (FD001, FD002, FD003, FD004) for both training and testing sets.

Key outputs:
  1. Statistical comparison CSV files (8 metrics per sensor per subset):
     - reports/eda/sensor_distributions/train_sensor_distribution_comparison.csv
     - reports/eda/sensor_distributions/test_sensor_distribution_comparison.csv
  2. 42 comparative boxplots (21 sensors x 2 splits: train and test):
     - reports/eda/sensor_distributions/{split}_{sensor}_distribution.png
=============================================================================
"""

import os
import tempfile
# Set writable cache directory for matplotlib before importing
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib_cache")

import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Directory paths
DATA_DIR = os.path.join("dataset", "raw", "C-MAPSS")
OUTPUT_DIR = os.path.join("reports", "eda", "sensor_distributions")
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

# Physical metadata documented in PHM 2008 / C-MAPSS
SENSOR_METADATA = {
    "sensor_1":  {"name": "T2",        "desc": "Total Temperature at Fan Inlet",        "unit": "°R"},
    "sensor_2":  {"name": "T24",       "desc": "Total Temperature at LPC Outlet",       "unit": "°R"},
    "sensor_3":  {"name": "T30",       "desc": "Total Temperature at HPC Outlet",       "unit": "°R"},
    "sensor_4":  {"name": "T50",       "desc": "Total Temperature at LPT Outlet",       "unit": "°R"},
    "sensor_5":  {"name": "P2",        "desc": "Pressure at Fan Inlet",                 "unit": "psia"},
    "sensor_6":  {"name": "P15",       "desc": "Total Pressure in Bypass-Duct",         "unit": "psia"},
    "sensor_7":  {"name": "P30",       "desc": "Total Pressure at HPC Outlet",          "unit": "psia"},
    "sensor_8":  {"name": "Nf",        "desc": "Physical Fan Speed",                    "unit": "rpm"},
    "sensor_9":  {"name": "Nc",        "desc": "Physical Core Speed",                   "unit": "rpm"},
    "sensor_10": {"name": "epr",       "desc": "Engine Pressure Ratio (P50/P2)",        "unit": "-"},
    "sensor_11": {"name": "Ps30",      "desc": "Static Pressure at HPC Outlet",         "unit": "psia"},
    "sensor_12": {"name": "phi",       "desc": "Ratio of Fuel Flow to Ps30",            "unit": "pps/psia"},
    "sensor_13": {"name": "Nf_dmd",    "desc": "Corrected Fan Speed",                   "unit": "rpm"},
    "sensor_14": {"name": "Nc_dmd",    "desc": "Corrected Core Speed",                  "unit": "rpm"},
    "sensor_15": {"name": "BPR",       "desc": "Bypass Ratio",                          "unit": "-"},
    "sensor_16": {"name": "farB",      "desc": "Burner Fuel-Air Ratio",                 "unit": "-"},
    "sensor_17": {"name": "htBleed",   "desc": "Bleed Enthalpy",                        "unit": "-"},
    "sensor_18": {"name": "Nf_dmd2",   "desc": "Demanded Fan Speed",                    "unit": "rpm"},
    "sensor_19": {"name": "PCNfR_dmd", "desc": "Demanded Corrected Fan Speed",          "unit": "rpm"},
    "sensor_20": {"name": "W31",       "desc": "HPT Coolant Bleed",                     "unit": "lbm/s"},
    "sensor_21": {"name": "W32",       "desc": "LPT Coolant Bleed",                     "unit": "lbm/s"}
}

SUBSET_PALETTE = {
    "FD001": "#264653",
    "FD002": "#2A9D8F",
    "FD003": "#E76F51",
    "FD004": "#E63946"
}


def load_split_data(split="train"):
    """Loads all four subset files for a given split ('train' or 'test')."""
    dfs = {}
    for subset in SUBSETS:
        filename = f"{split}_{subset}.txt"
        filepath = os.path.join(DATA_DIR, filename)
        dfs[subset] = pd.read_csv(filepath, sep=r"\s+", header=None, names=COLUMN_NAMES)
    return dfs


def compute_distribution_statistics(dfs):
    """Computes count, mean, std, min, 25%, median, 75%, max for all 21 sensors across subsets."""
    records = []
    for sensor in SENSOR_COLS:
        meta = SENSOR_METADATA[sensor]
        for subset in SUBSETS:
            series = dfs[subset][sensor]
            stats = {
                "sensor": sensor,
                "sensor_name": meta["name"],
                "description": meta["desc"],
                "unit": meta["unit"],
                "subset": subset,
                "count": int(series.count()),
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": float(series.min()),
                "q25": float(series.quantile(0.25)),
                "median": float(series.median()),
                "q75": float(series.quantile(0.75)),
                "max": float(series.max())
            }
            records.append(stats)
    return pd.DataFrame(records)


def generate_boxplots(dfs, split="train"):
    """Generates comparative boxplots showing FD001-FD004 side by side for each of the 21 sensors."""
    plot_paths = []
    
    for sensor in SENSOR_COLS:
        meta = SENSOR_METADATA[sensor]
        unit_str = f" [{meta['unit']}]" if meta['unit'] != "-" else ""
        
        # Extract data series for all 4 subsets
        data_to_plot = [dfs[subset][sensor].values for subset in SUBSETS]
        
        fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=300)
        
        # Create boxplot with styled elements
        bp = ax.boxplot(
            data_to_plot,
            tick_labels=SUBSETS,
            patch_artist=True,
            showmeans=True,
            meanprops={
                "marker": "D",
                "markerfacecolor": "yellow",
                "markeredgecolor": "black",
                "markersize": 5
            },
            medianprops={"color": "white", "linewidth": 2.0},
            whiskerprops={"linewidth": 1.2, "color": "#333333"},
            capprops={"linewidth": 1.2, "color": "#333333"},
            flierprops={
                "marker": "o",
                "markersize": 2.5,
                "alpha": 0.35,
                "markerfacecolor": "#555555",
                "markeredgecolor": "none"
            }
        )
        
        # Color each box according to palette
        for patch, subset in zip(bp["boxes"], SUBSETS):
            color = SUBSET_PALETTE[subset]
            patch.set_facecolor(color)
            patch.set_alpha(0.85)
            patch.set_edgecolor("#222222")
            patch.set_linewidth(1.2)
            
        ax.set_xlabel("C-MAPSS Subset", fontsize=11, fontweight="bold")
        ax.set_ylabel(f"{sensor.upper()} ({meta['name']}){unit_str}", fontsize=11, fontweight="bold")
        
        split_title = "Training Set" if split == "train" else "Testing Set"
        ax.set_title(
            f"C-MAPSS {split_title}: {sensor.upper()} - {meta['desc']}\n"
            f"Comparative Distribution Across Subsets (FD001 - FD004)",
            fontsize=12,
            fontweight="bold",
            pad=12
        )
        ax.grid(True, linestyle="--", alpha=0.5, axis="y")
        
        # Annotate legend for mean and median markers
        ax.plot([], [], marker="D", color="yellow", markeredgecolor="black", linestyle="None", label="Mean")
        ax.plot([], [], color="white", linewidth=2.0, label="Median")
        ax.legend(loc="upper right", frameon=True, framealpha=0.9, facecolor="white", edgecolor="#CCCCCC", fontsize=9)
        
        plt.tight_layout()
        
        filename = f"{split}_{sensor}_distribution.png"
        filepath = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(filepath)
        plt.close(fig)
        
        plot_paths.append(filepath)
        
    return plot_paths


def print_sensor_distribution_insights(train_stats_df):
    """Analyzes and prints structured findings on constant, near-constant, and variable sensors."""
    print("\n" + "=" * 90)
    print("KEY DISTRIBUTION FINDINGS ACROSS FD001 - FD004 (TRAINING SET)")
    print("=" * 90)
    
    # 1. Sensors with exactly zero standard deviation in both FD001 and FD003
    exact_zero_sensors = ["sensor_1", "sensor_5", "sensor_16", "sensor_18", "sensor_19"]
    print("\n1. SENSORS WITH EXACTLY ZERO VARIATION IN FD001 AND FD003:")
    print("-" * 90)
    print(f"{'Sensor':<12}{'Description':<38}{'FD001 std':<12}{'FD002 std':<12}{'FD003 std':<12}{'FD004 std':<12}")
    print("-" * 90)
    for sensor in exact_zero_sensors:
        s_data = train_stats_df[train_stats_df["sensor"] == sensor].set_index("subset")
        desc = SENSOR_METADATA[sensor]["desc"][:36]
        std_1 = s_data.loc["FD001", "std"]
        std_2 = s_data.loc["FD002", "std"]
        std_3 = s_data.loc["FD003", "std"]
        std_4 = s_data.loc["FD004", "std"]
        print(f"{sensor:<12}{desc:<38}{std_1:<12.4f}{std_2:<12.4f}{std_3:<12.4f}{std_4:<12.4f}")
    print("-" * 90)
    print("Observation: These 5 sensors have exactly zero standard deviation in FD001 and FD003")
    print("             (single operating condition), but exhibit variation in FD002 and FD004")
    print("             (six operating conditions).")

    # 2. Sensor 10: Constant in FD001, Near-Constant in FD003
    print("\n2. SENSOR 10 (ENGINE PRESSURE RATIO - epr):")
    print("-" * 90)
    s10_data = train_stats_df[train_stats_df["sensor"] == "sensor_10"].set_index("subset")
    print(f"{'Sensor':<12}{'Description':<38}{'FD001 std':<12}{'FD002 std':<12}{'FD003 std':<12}{'FD004 std':<12}")
    print("-" * 90)
    print(f"{'sensor_10':<12}{'Engine Pressure Ratio (P50/P2)':<38}"
          f"{s10_data.loc['FD001', 'std']:<12.4f}{s10_data.loc['FD002', 'std']:<12.4f}"
          f"{s10_data.loc['FD003', 'std']:<12.4f}{s10_data.loc['FD004', 'std']:<12.4f}")
    print("-" * 90)
    print("Observation: sensor_10 has exactly zero standard deviation in FD001 (1.3000),")
    print("             near-constant values in FD003 (std = 0.0035, min = 1.29, max = 1.32),")
    print("             and wider variation in FD002 and FD004 (std ~ 0.127).")

    # 3. Sensor 6: Very Low Variation in FD001 and FD003
    print("\n3. SENSOR 6 (TOTAL PRESSURE IN BYPASS-DUCT - P15):")
    print("-" * 90)
    s6_data = train_stats_df[train_stats_df["sensor"] == "sensor_6"].set_index("subset")
    print(f"{'Sensor':<12}{'Description':<38}{'FD001 std':<12}{'FD002 std':<12}{'FD003 std':<12}{'FD004 std':<12}")
    print("-" * 90)
    print(f"{'sensor_6':<12}{'Total Pressure in Bypass-Duct':<38}"
          f"{s6_data.loc['FD001', 'std']:<12.4f}{s6_data.loc['FD002', 'std']:<12.4f}"
          f"{s6_data.loc['FD003', 'std']:<12.4f}{s6_data.loc['FD004', 'std']:<12.4f}")
    print("-" * 90)
    print("Observation: sensor_6 has very low standard deviation in FD001 (0.0014) and FD003 (0.0181),")
    print("             distinguishable from exactly zero, but substantially narrower than FD002 and FD004 (std ~ 5.44).")

    # 4. Comparison of Mean and Spread for Selected Dynamic Sensors
    print("\n4. COMPARISON OF MEAN AND SPREAD FOR SELECTED ACTIVE SENSORS:")
    print("-" * 90)
    print(f"{'Sensor':<10}{'Metric':<8}{'FD001 (1 cond)':<18}{'FD002 (6 cond)':<18}{'FD003 (1 cond)':<18}{'FD004 (6 cond)':<18}")
    print("-" * 90)
    
    sample_sensors = ["sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_9", "sensor_11", "sensor_14", "sensor_15"]
    for sensor in sample_sensors:
        s_data = train_stats_df[train_stats_df["sensor"] == sensor].set_index("subset")
        mean_str = [f"{s_data.loc[sub, 'mean']:10.2f}" for sub in SUBSETS]
        std_str = [f"{s_data.loc[sub, 'std']:10.2f}" for sub in SUBSETS]
        print(f"{sensor:<10}{'Mean':<8}{mean_str[0]:<18}{mean_str[1]:<18}{mean_str[2]:<18}{mean_str[3]:<18}")
        print(f"{'':<10}{'Std':<8}{std_str[0]:<18}{std_str[1]:<18}{std_str[2]:<18}{std_str[3]:<18}")
        print("-" * 90)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 90)
    print("PHASE 2 - STEP 2.10: SENSOR DISTRIBUTION COMPARISON ACROSS SUBSETS")
    print("=" * 90)
    print(f"Output directory: {OUTPUT_DIR}\n")
    
    # Process Training Data
    print("Loading Training Datasets (train_FD001 to train_FD004)...")
    train_dfs = load_split_data("train")
    for sub, df in train_dfs.items():
        print(f"  - {sub}: {df.shape[0]:,} rows x {df.shape[1]} columns")
        
    print("\nComputing statistical metrics for all 21 sensors across training subsets...")
    train_stats = compute_distribution_statistics(train_dfs)
    train_csv_path = os.path.join(OUTPUT_DIR, "train_sensor_distribution_comparison.csv")
    train_stats.to_csv(train_csv_path, index=False)
    print(f"  -> Saved training comparison CSV: {train_csv_path}")
    
    print("\nGenerating 21 comparative boxplots for Training Data...")
    train_plots = generate_boxplots(train_dfs, split="train")
    print(f"  -> Generated {len(train_plots)} training boxplots.")
    
    # Process Testing Data
    print("\nLoading Testing Datasets (test_FD001 to test_FD004)...")
    test_dfs = load_split_data("test")
    for sub, df in test_dfs.items():
        print(f"  - {sub}: {df.shape[0]:,} rows x {df.shape[1]} columns")
        
    print("\nComputing statistical metrics for all 21 sensors across testing subsets...")
    test_stats = compute_distribution_statistics(test_dfs)
    test_csv_path = os.path.join(OUTPUT_DIR, "test_sensor_distribution_comparison.csv")
    test_stats.to_csv(test_csv_path, index=False)
    print(f"  -> Saved testing comparison CSV: {test_csv_path}")
    
    print("\nGenerating 21 comparative boxplots for Testing Data...")
    test_plots = generate_boxplots(test_dfs, split="test")
    print(f"  -> Generated {len(test_plots)} testing boxplots.")
    
    # Print structured insights
    print_sensor_distribution_insights(train_stats)
    
    print("\n" + "=" * 90)
    print(f"COMPLETE: Total plots generated = {len(train_plots) + len(test_plots)} (21 train + 21 test)")
    print(f"Total CSV reports saved = 2 (train and test)")
    print("=" * 90)


if __name__ == "__main__":
    main()
