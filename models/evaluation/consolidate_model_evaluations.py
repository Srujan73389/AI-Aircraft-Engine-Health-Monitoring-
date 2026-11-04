"""
Phase 4 - Step 4.4: Consolidated Model Evaluation & Cross-Architecture Synthesis
================================================================================
AI-Based Aircraft Engine Health Monitoring, Fault Diagnosis and Predictive Maintenance

This script loads, validates, reconciles, and synthesizes results across all 108
completed model configurations from Phase 4:
  - Step 4.1 & 4.2: Baselines (ConstantMean, Ridge, RandomForest) -> 52 runs
  - Step 4.2: XGBoost (FinalObs, FlattenedWindow, SummaryStats, Tuned) -> 40 runs
  - Step 4.3: Deep Sequential (Stacked_LSTM, Single_LSTM) -> 16 runs

Outputs:
  - reports/models/master_model_comparison.csv (108 configurations)
  - reports/models/model_selection_summary.csv (Curated selection decision matrix)
  - reports/figures/model_comparison_val_test_rmse.png
  - reports/figures/nasa_score_tradeoff_comparison.png
  - reports/figures/per_engine_rmse_distribution.png

IMPORTANT: This script performs synthesis, aggregation, and visualization only.
It does NOT retrain any models or alter historical experiment outputs.
"""

import os
import sys
import numpy as np
import pandas as pd

# Set Matplotlib config directory to avoid permission issues in sandboxed environment
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.environ["MPLCONFIGDIR"] = os.path.join(PROJECT_ROOT, "reports", "figures", ".mplconfig")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_and_merge_experiments() -> pd.DataFrame:
    """Load and merge all 108 configurations from Steps 4.1, 4.2, and 4.3."""
    reports_dir = os.path.join(PROJECT_ROOT, "reports", "models")
    
    f_baseline = os.path.join(reports_dir, "baseline_model_performance.csv")
    f_xgboost = os.path.join(reports_dir, "xgboost_model_performance.csv")
    f_lstm = os.path.join(reports_dir, "lstm_model_performance.csv")
    
    assert os.path.exists(f_baseline), f"Missing {f_baseline}"
    assert os.path.exists(f_xgboost), f"Missing {f_xgboost}"
    assert os.path.exists(f_lstm), f"Missing {f_lstm}"
    
    df_b = pd.read_csv(f_baseline)
    df_x = pd.read_csv(f_xgboost)
    df_l = pd.read_csv(f_lstm)
    
    print(f"[LOAD] Baseline configurations loaded: {len(df_b)} runs")
    print(f"[LOAD] XGBoost configurations loaded:  {len(df_x)} runs")
    print(f"[LOAD] LSTM configurations loaded:     {len(df_l)} runs")
    
    assert len(df_b) == 52, f"Expected 52 baseline runs, got {len(df_b)}"
    assert len(df_x) == 40, f"Expected 40 XGBoost runs, got {len(df_x)}"
    assert len(df_l) == 16, f"Expected 16 LSTM runs, got {len(df_l)}"
    
    # Standardize optional epoch columns
    if "best_epoch" not in df_b.columns:
        df_b["best_epoch"] = "N/A"
        df_b["total_epochs"] = "N/A"
    if "best_epoch" not in df_x.columns:
        df_x["best_epoch"] = "N/A"
        df_x["total_epochs"] = "N/A"
        
    df_master = pd.concat([df_b, df_x, df_l], ignore_index=True)
    assert len(df_master) == 108, f"Expected exactly 108 total runs, got {len(df_master)}"
    
    # Add comparative percentage improvement columns relative to naive mean baseline
    df_master["val_capped_rmse_gain_pct"] = 0.0
    df_master["test_rmse_gain_pct"] = 0.0
    
    for sub in ["FD001", "FD002", "FD003", "FD004"]:
        mean_mask = (df_master["subset"] == sub) & (df_master["model_family"].isin(["ConstantMean", "TrainingMean"]))
        mean_row = df_master[mean_mask].iloc[0]
        mean_val_rmse = mean_row["val_capped_rmse"]
        mean_test_rmse = mean_row["test_uncapped_rmse"]
        
        sub_mask = df_master["subset"] == sub
        df_master.loc[sub_mask, "val_capped_rmse_gain_pct"] = round(
            ((mean_val_rmse - df_master.loc[sub_mask, "val_capped_rmse"]) / mean_val_rmse) * 100.0, 2
        )
        df_master.loc[sub_mask, "test_rmse_gain_pct"] = round(
            ((mean_test_rmse - df_master.loc[sub_mask, "test_uncapped_rmse"]) / mean_test_rmse) * 100.0, 2
        )
        
    return df_master


def build_selection_summary(df_master: pd.DataFrame) -> pd.DataFrame:
    """Build the curated model selection summary comparing top models per family."""
    subsets = ["FD001", "FD002", "FD003", "FD004"]
    summary_rows = []
    
    for sub in subsets:
        sub_df = df_master[df_master["subset"] == sub]
        
        # Absolute champion by Validation Capped RMSE
        champ_row = sub_df.sort_values("val_capped_rmse").iloc[0]
        
        # Best per family
        families = ["ConstantMean", "Ridge", "RandomForest", "XGBoost", "LSTM"]
        for fam in families:
            fam_df = sub_df[sub_df["model_family"].isin([fam, "TrainingMean"] if fam == "ConstantMean" else [fam])]
            if len(fam_df) == 0:
                continue
            best_fam = fam_df.sort_values("val_capped_rmse").iloc[0]
            
            is_champ = (best_fam["model_variant"] == champ_row["model_variant"]) and (best_fam["feature_tier"] == champ_row["feature_tier"])
            
            # Academic recommendation role
            if is_champ and fam == "XGBoost":
                rec_role = "PRIMARY BENCHMARK (Overall Champion)"
            elif is_champ and fam == "LSTM":
                rec_role = "SPECIALIZED SEQUENCE CHAMPION (Validation Winner)"
            elif fam == "XGBoost" and not is_champ:
                rec_role = "PRIMARY TABULAR BENCHMARK (Test Generalization)"
            elif fam == "LSTM" and not is_champ:
                rec_role = "EXPERIMENTAL SEQUENCE MODEL"
            elif fam == "RandomForest":
                rec_role = "NON-LINEAR ENSEMBLE BASELINE"
            elif fam == "Ridge":
                rec_role = "LINEAR REGRESSION BASELINE"
            else:
                rec_role = "NAIVE STATISTICAL BENCHMARK"
                
            summary_rows.append({
                "subset": sub,
                "model_family": fam,
                "best_variant": best_fam["model_variant"],
                "feature_tier": best_fam["feature_tier"],
                "feature_dim": best_fam["feature_dim"],
                "val_capped_rmse": best_fam["val_capped_rmse"],
                "val_capped_r2": best_fam["val_capped_r2"],
                "val_mean_engine_rmse": best_fam["val_mean_engine_rmse"],
                "val_final_cycle_rmse": best_fam["val_final_cycle_rmse"],
                "test_uncapped_rmse": best_fam["test_uncapped_rmse"],
                "test_uncapped_mae": best_fam["test_uncapped_mae"],
                "test_uncapped_score": best_fam["test_uncapped_score"],
                "val_gain_vs_mean_pct": best_fam["val_capped_rmse_gain_pct"],
                "test_gain_vs_mean_pct": best_fam["test_rmse_gain_pct"],
                "is_subset_champion": is_champ,
                "academic_role": rec_role,
            })
            
    return pd.DataFrame(summary_rows)


def generate_comparison_plots(df_summary: pd.DataFrame, df_engine: pd.DataFrame):
    """Generate high-resolution comparative figures under reports/figures/."""
    figures_dir = os.path.join(PROJECT_ROOT, "reports", "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    subsets = ["FD001", "FD002", "FD003", "FD004"]
    families = ["Ridge", "RandomForest", "XGBoost", "LSTM"]
    family_colors = {
        "Ridge": "#4A90E2",         # Steel Blue
        "RandomForest": "#50E3C2",  # Teal
        "XGBoost": "#F5A623",       # Orange
        "LSTM": "#D0021B",          # Crimson
    }
    
    # -------------------------------------------------------------------------
    # Plot 1: Validation Capped RMSE vs. Official Test RMSE
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
    
    x = np.arange(len(subsets))
    width = 0.18
    
    # 1a. Validation Capped RMSE
    ax1 = axes[0]
    for i, fam in enumerate(families):
        vals = []
        for sub in subsets:
            row = df_summary[(df_summary["subset"] == sub) & (df_summary["model_family"] == fam)]
            vals.append(row["val_capped_rmse"].values[0])
        ax1.bar(x + (i - 1.5) * width, vals, width, label=fam, color=family_colors[fam], alpha=0.9, edgecolor="black", linewidth=0.8)
        
    ax1.set_title("Validation Capped RMSE (Selection Metric)", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(subsets, fontsize=11, fontweight="bold")
    ax1.set_ylabel("Capped RMSE (cycles, RUL <= 125)", fontsize=11)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", frameon=True, fontsize=10)
    
    # 1b. Official Test RMSE
    ax2 = axes[1]
    for i, fam in enumerate(families):
        vals = []
        for sub in subsets:
            row = df_summary[(df_summary["subset"] == sub) & (df_summary["model_family"] == fam)]
            vals.append(row["test_uncapped_rmse"].values[0])
        ax2.bar(x + (i - 1.5) * width, vals, width, label=fam, color=family_colors[fam], alpha=0.9, edgecolor="black", linewidth=0.8)
        
    ax2.set_title("Official Test RMSE (Uncapped Benchmark)", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(subsets, fontsize=11, fontweight="bold")
    ax2.set_ylabel("Uncapped RMSE (cycles)", fontsize=11)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left", frameon=True, fontsize=10)
    
    fig.suptitle("NASA C-MAPSS Cross-Architecture Accuracy: Validation vs. Official Test", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    p1_path = os.path.join(figures_dir, "model_comparison_val_test_rmse.png")
    fig.savefig(p1_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved: {p1_path}")
    
    # -------------------------------------------------------------------------
    # Plot 2: NASA Asymmetric Scoring Function Trade-Off (Log Scale)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, fam in enumerate(families):
        scores = []
        for sub in subsets:
            row = df_summary[(df_summary["subset"] == sub) & (df_summary["model_family"] == fam)]
            scores.append(row["test_uncapped_score"].values[0])
        ax.bar(x + (i - 1.5) * width, scores, width, label=fam, color=family_colors[fam], alpha=0.9, edgecolor="black", linewidth=0.8)
        
    ax.set_yscale("log")
    ax.set_title("Official NASA Asymmetric Test Score Comparison (Log Scale)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(subsets, fontsize=12, fontweight="bold")
    ax.set_ylabel("NASA Asymmetric Penalty Score (Lower is Better, Log Scale)", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.6, which="both")
    ax.legend(loc="upper left", frameon=True, fontsize=11)
    
    # Annotation explaining NASA score asymmetry
    ax.text(
        0.98, 0.04,
        "Penalty: exp(d/10)-1 for d>0 (late prediction) vs exp(-d/13)-1 for d<0 (early prediction)\nExponentially penalizes over-estimation (in-flight safety hazard).",
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFF9C4", edgecolor="#FBC02D", alpha=0.8),
    )
    
    plt.tight_layout()
    p2_path = os.path.join(figures_dir, "nasa_score_tradeoff_comparison.png")
    fig.savefig(p2_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved: {p2_path}")
    
    # -------------------------------------------------------------------------
    # Plot 3: Per-Engine Error Distribution Across Validation Engines
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, sub in enumerate(subsets):
        ax = axes[idx]
        box_data = []
        box_labels = []
        box_colors = []
        
        for fam in families:
            sum_row = df_summary[(df_summary["subset"] == sub) & (df_summary["model_family"] == fam)].iloc[0]
            m_variant = sum_row["best_variant"]
            f_tier = sum_row["feature_tier"]
            
            eng_data = df_engine[
                (df_engine["subset"] == sub) &
                (df_engine["model_variant"] == m_variant) &
                (df_engine["feature_tier"] == f_tier)
            ]["per_engine_rmse_capped"].values
            
            if len(eng_data) > 0:
                box_data.append(eng_data)
                box_labels.append(fam)
                box_colors.append(family_colors[fam])
                
        bplot = ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True, medianprops=dict(color="black", linewidth=1.5))
        for patch, col in zip(bplot["boxes"], box_colors):
            patch.set_facecolor(col)
            patch.set_alpha(0.8)
            
        ax.set_title(f"{sub} Engine-Level RMSE Distribution (142 Engines)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Engine Capped RMSE (cycles)", fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        
    fig.suptitle("Validation Error Consistency Across Individual Engine Units", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    p3_path = os.path.join(figures_dir, "per_engine_rmse_distribution.png")
    fig.savefig(p3_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved: {p3_path}")


def main():
    print("=" * 80)
    print("PHASE 4 - STEP 4.4: MODEL EVALUATION PROTOCOL CONSOLIDATION")
    print("=" * 80)
    
    # 1. Load and merge all 108 configurations
    df_master = load_and_merge_experiments()
    
    reports_dir = os.path.join(PROJECT_ROOT, "reports", "models")
    master_csv_path = os.path.join(reports_dir, "master_model_comparison.csv")
    df_master.to_csv(master_csv_path, index=False)
    print(f"\n[SAVE] Master model comparison table ({len(df_master)} rows): {master_csv_path}")
    
    # 2. Build model selection decision summary
    df_summary = build_selection_summary(df_master)
    summary_csv_path = os.path.join(reports_dir, "model_selection_summary.csv")
    df_summary.to_csv(summary_csv_path, index=False)
    print(f"[SAVE] Curated model selection summary ({len(df_summary)} rows): {summary_csv_path}")
    
    # 3. Load per-engine validation records for distribution plots
    f_be = os.path.join(reports_dir, "baseline_per_engine_validation.csv")
    f_xe = os.path.join(reports_dir, "xgboost_per_engine_validation.csv")
    f_le = os.path.join(reports_dir, "lstm_per_engine_validation.csv")
    
    df_be = pd.read_csv(f_be)
    df_xe = pd.read_csv(f_xe)
    df_le = pd.read_csv(f_le)
    df_engine = pd.concat([df_be, df_xe, df_le], ignore_index=True)
    print(f"[LOAD] Per-engine validation records combined: {len(df_engine)} engine-model evaluations")
    
    # 4. Generate comparative figures
    print("\n[PLOTTING] Generating publication-quality comparative figures...")
    generate_comparison_plots(df_summary, df_engine)
    
    # 5. Display curated findings
    print("\n" + "=" * 80)
    print("PHASE 4 CONSOLIDATED MODEL SELECTION MATRIX (ACADEMIC C-MAPSS RECOMMENDATIONS)")
    print("=" * 80)
    for sub in ["FD001", "FD002", "FD003", "FD004"]:
        print(f"\n[{sub}]")
        sub_sum = df_summary[df_summary["subset"] == sub]
        for _, r in sub_sum.iterrows():
            flag = " [CHAMPION]" if r["is_subset_champion"] else ""
            print(f"  - {r['model_family']:13s} | {r['best_variant']:25s} | {r['feature_tier']:19s} | Val RMSE={r['val_capped_rmse']:5.2f} | Test RMSE={r['test_uncapped_rmse']:5.2f} | Test Score={r['test_uncapped_score']:9.1f}{flag}")
            
    print("\n" + "=" * 80)
    print("STEP 4.4 SYNTHESIS AND VISUALIZATION COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
