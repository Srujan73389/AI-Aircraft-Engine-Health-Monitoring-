"""
=============================================================================
AIRCRAFT ENGINE AI - HEALTH MONITORING & PREDICTIVE MAINTENANCE
Phase 2: Dataset Understanding & Exploratory Data Analysis (EDA)
Step 2.9: Plot Selected Sensor Readings Against Engine Cycles
=============================================================================
This script visualizes degradation trajectories for 11 selected sensors
across representative engines (Short, Medium, Long lifespan) for each of the
four NASA C-MAPSS training subsets (FD001, FD002, FD003, FD004).

Selected sensors:
  - sensor_2  (T24 - LPC outlet temperature)
  - sensor_3  (T30 - HPC outlet temperature)
  - sensor_4  (T50 - LPT outlet temperature)
  - sensor_7  (P30 - HPC outlet total pressure)
  - sensor_9  (Nc  - Physical core speed)
  - sensor_11 (Ps30 - HPC outlet static pressure)
  - sensor_14 (Nc_dmd - Corrected core speed)
  - sensor_15 (BPR - Bypass ratio)
  - sensor_17 (htBleed - Bleed enthalpy)
  - sensor_20 (W31 - HPT coolant bleed)
  - sensor_21 (W32 - LPT coolant bleed)

Output:
  44 high-resolution plots saved in reports/eda/sensor_trends/
=============================================================================
"""

import os
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import pandas as pd

# Paths and configurations
DATA_DIR = os.path.join("dataset", "raw", "C-MAPSS")
OUTPUT_DIR = os.path.join("reports", "eda", "sensor_trends")
SUBSETS = ["FD001", "FD002", "FD003", "FD004"]

COLUMN_NAMES = [
    "unit_number", "time_cycles",
    "op_setting_1", "op_setting_2", "op_setting_3",
    "sensor_1", "sensor_2", "sensor_3", "sensor_4", "sensor_5",
    "sensor_6", "sensor_7", "sensor_8", "sensor_9", "sensor_10",
    "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
    "sensor_16", "sensor_17", "sensor_18", "sensor_19", "sensor_20", "sensor_21"
]

# Physical sensor descriptions from PHM 2008 / C-MAPSS documentation
SENSOR_METADATA = {
    "sensor_2":  {"name": "T24",      "desc": "Total Temperature at LPC Outlet",    "unit": "°R"},
    "sensor_3":  {"name": "T30",      "desc": "Total Temperature at HPC Outlet",    "unit": "°R"},
    "sensor_4":  {"name": "T50",      "desc": "Total Temperature at LPT Outlet",    "unit": "°R"},
    "sensor_7":  {"name": "P30",      "desc": "Total Pressure at HPC Outlet",       "unit": "psia"},
    "sensor_9":  {"name": "Nc",       "desc": "Physical Core Speed",                "unit": "rpm"},
    "sensor_11": {"name": "Ps30",     "desc": "Static Pressure at HPC Outlet",      "unit": "psia"},
    "sensor_14": {"name": "Nc_dmd",   "desc": "Corrected Core Speed",               "unit": "rpm"},
    "sensor_15": {"name": "BPR",      "desc": "Bypass Ratio",                       "unit": "-"},
    "sensor_17": {"name": "htBleed",  "desc": "Bleed Enthalpy",                     "unit": "-"},
    "sensor_20": {"name": "W31",      "desc": "HPT Coolant Bleed",                  "unit": "lbm/s"},
    "sensor_21": {"name": "W32",      "desc": "LPT Coolant Bleed",                  "unit": "lbm/s"}
}

# Color styling for trajectories
ENGINE_STYLES = {
    "short":  {"color": "#D90429", "linestyle": "-",  "linewidth": 1.6, "label_prefix": "Short Lifespan"},
    "medium": {"color": "#2A9D8F", "linestyle": "-",  "linewidth": 1.6, "label_prefix": "Medium Lifespan"},
    "long":   {"color": "#1D3557", "linestyle": "-",  "linewidth": 1.6, "label_prefix": "Long Lifespan"}
}


def select_representative_engines(df):
    """
    Selects 3 representative engines based on operating lifespan:
      - Short: Engine with the minimum total cycles
      - Long: Engine with the maximum total cycles
      - Medium: Engine whose total cycles are closest to the median
    """
    lifespans = df.groupby("unit_number")["time_cycles"].max()
    
    short_unit = int(lifespans.idxmin())
    short_cycles = int(lifespans.min())
    
    long_unit = int(lifespans.idxmax())
    long_cycles = int(lifespans.max())
    
    median_val = lifespans.median()
    # Filter out short and long units to ensure 3 distinct units
    remaining_units = lifespans.drop(index=[short_unit, long_unit])
    medium_unit = int((remaining_units - median_val).abs().idxmin())
    medium_cycles = int(lifespans[medium_unit])
    
    return {
        "short":  {"unit_id": short_unit,  "cycles": short_cycles},
        "medium": {"unit_id": medium_unit, "cycles": medium_cycles},
        "long":   {"unit_id": long_unit,   "cycles": long_cycles}
    }


def plot_sensor_trends():
    """Generates and saves sensor trend plots for all subsets and selected sensors."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 80)
    print("PHASE 2 - STEP 2.9: SENSOR TREND PLOTTING (RUN-TO-FAILURE TRAJECTORIES)")
    print("=" * 80)
    print(f"Output directory: {OUTPUT_DIR}\n")
    
    selected_engines_summary = {}

    for subset in SUBSETS:
        filename = f"train_{subset}.txt"
        filepath = os.path.join(DATA_DIR, filename)
        
        print(f"Loading {filename}...")
        df = pd.read_csv(filepath, sep=r"\s+", header=None, names=COLUMN_NAMES)
        
        rep_engines = select_representative_engines(df)
        selected_engines_summary[subset] = rep_engines
        
        print(f"  Representative engines for {subset}:")
        for cat in ["short", "medium", "long"]:
            info = rep_engines[cat]
            print(f"    - {cat.capitalize():6s}: Unit {info['unit_id']:3d} ({info['cycles']} cycles)")
            
        # Plot each of the 11 sensors
        for sensor_id, meta in SENSOR_METADATA.items():
            fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
            
            for cat in ["short", "medium", "long"]:
                unit_id = rep_engines[cat]["unit_id"]
                cycles = rep_engines[cat]["cycles"]
                style = ENGINE_STYLES[cat]
                
                engine_data = df[df["unit_number"] == unit_id]
                
                label = f"{style['label_prefix']}: Unit {unit_id} ({cycles} cycles)"
                ax.plot(
                    engine_data["time_cycles"],
                    engine_data[sensor_id],
                    label=label,
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=style["linewidth"],
                    alpha=0.85
                )
                
            unit_str = f" [{meta['unit']}]" if meta['unit'] != "-" else ""
            ax.set_xlabel("Operating Time (Cycles)", fontsize=11, fontweight="bold")
            ax.set_ylabel(f"{sensor_id.upper()} - {meta['desc']}{unit_str}", fontsize=11, fontweight="bold")
            ax.set_title(
                f"C-MAPSS {subset} - {sensor_id.upper()} ({meta['desc']}) Trend\n"
                f"Run-to-Failure Trajectories for Representative Engines",
                fontsize=12,
                fontweight="bold",
                pad=12
            )
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(loc="best", frameon=True, framealpha=0.9, facecolor="white", edgecolor="#CCCCCC", fontsize=9.5)
            
            plt.tight_layout()
            
            output_filename = f"{subset}_{sensor_id}_trends.png"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            plt.savefig(output_path)
            plt.close(fig)
            
        print(f"  -> Generated 11 plots for {subset}.\n")
        
    print("=" * 80)
    print(f"Successfully generated all {len(SUBSETS) * len(SENSOR_METADATA)} sensor trend plots.")
    print("=" * 80)
    
    # Print summary table of chosen engines
    print("\nSUMMARY OF REPRESENTATIVE ENGINES SELECTED PER SUBSET:")
    print("-" * 80)
    print(f"{'Subset':<10}{'Short Unit (Cycles)':<25}{'Medium Unit (Cycles)':<25}{'Long Unit (Cycles)':<25}")
    print("-" * 80)
    for subset, engs in selected_engines_summary.items():
        s_str = f"Unit {engs['short']['unit_id']} ({engs['short']['cycles']} c)"
        m_str = f"Unit {engs['medium']['unit_id']} ({engs['medium']['cycles']} c)"
        l_str = f"Unit {engs['long']['unit_id']} ({engs['long']['cycles']} c)"
        print(f"{subset:<10}{s_str:<25}{m_str:<25}{l_str:<25}")
    print("-" * 80)


if __name__ == "__main__":
    plot_sensor_trends()
