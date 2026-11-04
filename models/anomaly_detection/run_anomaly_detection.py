"""
Phase 4 - Step 4.4: Anomaly Detection for Unusual Sensor Patterns
================================================================
AI-Based Aircraft Engine Health Monitoring, Fault Diagnosis and Predictive Maintenance
Dataset: NASA C-MAPSS (FD001, FD002, FD003, FD004)

This script implements and executes the approved Step 4.4 anomaly detection pipeline:
  - M1: Regime-Aware Isolation Forest (Primary Model)
  - M2: PCA-SPE (Principal Component Analysis - Squared Prediction Error Benchmark)
  - M3: Unconditioned Isolation Forest (Ablation for FD002 & FD004)

Mandatory Methodological Protocols:
  1. Engine-Level 3-Cohort Separation:
     - Cohort A (80% of Train Engines): Detector fitting strictly on healthy observations (t <= 30 or RUL > 125).
     - Cohort B (20% of Train Engines): Decision threshold (tau) calibration on healthy observations (95th percentile).
     - Cohort C (Phase 3 Validation Engines): Untouched evaluation of full lifecycles.
     - ZERO row-level shuffling across cohorts!
  2. Operating-Regime Awareness:
     - K-Means (K=6) regime clustering on operational settings for multi-condition subsets (FD002, FD004).
     - Regime-specific standardization against healthy centroids to decouple flight maneuvers from degradation.
  3. Strict Persistence & Lead Time:
     - Requires 3 consecutive anomaly scores above tau.
     - Alert confirmation cycle is the THIRD consecutive cycle (t_alert = t + 2).
     - Lead time = failure cycle - t_alert = RUL(t_alert).
  4. Explicit Missed Detections:
     - Engines with no persistent alert: TTFD=None, Lead Time=None, Persistent Detected=False, Missed Detection=True.
     - Never substitutes zero for missing lead time.
  5. Manifest & Deliverables:
     - Produces performance tables, per-engine CSVs, high-res figures, and manifest with verified software versions.
"""

import os
import sys
import json
import time
import platform
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Set matplotlib backend for headless execution
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.environ["MPLCONFIGDIR"] = os.path.join(PROJECT_ROOT, "reports", "figures", ".mplconfig")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from preprocessing.scaling_pipeline import (
    TIER1_CORE_SENSORS,
    TIER3_EXTENDED_ALL,
)

# Constants & Configuration
RANDOM_STATE = 42
SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
HEALTHY_EARLY_CYCLE_THRESHOLD = 30
HEALTHY_RUL_PLATEAU_THRESHOLD = 125.0
CALIBRATION_PERCENTILE = 95.0
PERSISTENCE_WINDOW = 3  # 3 consecutive cycles above tau
LATE_LIFE_RUL_THRESHOLD = 20

OP_SETTINGS = ["op_setting_1", "op_setting_2", "op_setting_3"]


def get_software_versions() -> dict:
    """Record runtime software environment metadata."""
    import sklearn
    import scipy
    return {
        "python_version": sys.version.split()[0],
        "python_build": sys.version,
        "platform": platform.platform(),
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "recorded_for": "NASA C-MAPSS Phase 4 Step 4.4 Anomaly Detection",
    }


def split_engine_cohorts(train_df: pd.DataFrame, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, list[int], list[int]]:
    """
    Split training engines at the engine level into:
      - Cohort A (80%): Detector Fitting
      - Cohort B (20%): Healthy Threshold Calibration
    Guarantees zero row-level leakage.
    """
    unique_engines = sorted(train_df["unit_number"].unique())
    cohort_a_engines, cohort_b_engines = train_test_split(
        unique_engines, test_size=0.20, random_state=random_state, shuffle=True
    )
    cohort_a_engines = sorted(cohort_a_engines)
    cohort_b_engines = sorted(cohort_b_engines)

    df_cohort_a = train_df[train_df["unit_number"].isin(cohort_a_engines)].copy()
    df_cohort_b = train_df[train_df["unit_number"].isin(cohort_b_engines)].copy()

    return df_cohort_a, df_cohort_b, cohort_a_engines, cohort_b_engines


def filter_healthy_observations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract healthy baseline observations:
    Cycles with time_cycles <= 30 OR raw RUL > 125.
    """
    mask = (df["time_cycles"] <= HEALTHY_EARLY_CYCLE_THRESHOLD) | (df["RUL"] > HEALTHY_RUL_PLATEAU_THRESHOLD)
    return df[mask].copy()


class RegimeNormalizer:
    """
    Operating-regime normalizer:
    For single-regime subsets (FD001, FD003), standardizes sensors against global healthy baseline.
    For multi-regime subsets (FD002, FD004), clusters operating settings into K=6 regimes using KMeans,
    and standardizes sensors against regime-specific healthy centroids.
    """
    def __init__(self, is_multi_regime: bool = False, n_clusters: int = 6, random_state: int = 42):
        self.is_multi_regime = is_multi_regime
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None
        self.regime_stats = {}  # cluster_id -> (mean_vector, std_vector)
        self.global_mean = None
        self.global_std = None
        self.feature_cols = None

    def fit(self, healthy_df: pd.DataFrame, feature_cols: list[str]):
        self.feature_cols = feature_cols
        X_feats = healthy_df[feature_cols].values

        if not self.is_multi_regime:
            self.global_mean = np.mean(X_feats, axis=0)
            self.global_std = np.std(X_feats, axis=0)
            self.global_std[self.global_std < 1e-5] = 1.0
        else:
            X_ops = healthy_df[OP_SETTINGS].values
            self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
            cluster_labels = self.kmeans.fit_predict(X_ops)

            # Global fallback stats
            self.global_mean = np.mean(X_feats, axis=0)
            self.global_std = np.std(X_feats, axis=0)
            self.global_std[self.global_std < 1e-5] = 1.0

            for k in range(self.n_clusters):
                mask_k = (cluster_labels == k)
                if np.sum(mask_k) > 5:
                    m_k = np.mean(X_feats[mask_k], axis=0)
                    s_k = np.std(X_feats[mask_k], axis=0)
                    s_k[s_k < 1e-5] = 1.0
                else:
                    m_k = self.global_mean.copy()
                    s_k = self.global_std.copy()
                self.regime_stats[k] = (m_k, s_k)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        X_feats = df[self.feature_cols].values
        N = len(df)
        residuals = np.zeros_like(X_feats, dtype=np.float32)

        if not self.is_multi_regime:
            residuals = (X_feats - self.global_mean) / self.global_std
        else:
            X_ops = df[OP_SETTINGS].values
            cluster_labels = self.kmeans.predict(X_ops)
            for k in range(self.n_clusters):
                mask_k = (cluster_labels == k)
                if np.any(mask_k):
                    m_k, s_k = self.regime_stats.get(k, (self.global_mean, self.global_std))
                    residuals[mask_k] = (X_feats[mask_k] - m_k) / s_k

        return residuals


class PCASPEAnomalyDetector:
    """
    PCA-SPE (Squared Prediction Error) Anomaly Detector:
    Fits PCA on healthy residual vectors (capturing 95% variance).
    Computes Q-residual: SPE(z) = ||z - P P^T z||^2 as the anomaly score.
    """
    def __init__(self, variance_ratio: float = 0.95, random_state: int = 42):
        self.variance_ratio = variance_ratio
        self.random_state = random_state
        self.pca = None

    def fit(self, Z_healthy: np.ndarray):
        self.pca = PCA(n_components=self.variance_ratio, random_state=self.random_state)
        self.pca.fit(Z_healthy)

    def score(self, Z: np.ndarray) -> np.ndarray:
        Z_proj = self.pca.transform(Z)
        Z_reconstructed = self.pca.inverse_transform(Z_proj)
        spe = np.sum((Z - Z_reconstructed) ** 2, axis=1)
        return spe.astype(np.float32)


def evaluate_engine_trajectories(
    df_val: pd.DataFrame,
    anomaly_scores: np.ndarray,
    threshold: float,
    persistence_k: int = PERSISTENCE_WINDOW
) -> tuple[dict, pd.DataFrame]:
    """
    Evaluate anomaly detection metrics across validation engines:
      - Mandatory Alert Confirmation: Requires k consecutive cycles above threshold.
      - Confirmation cycle is the k-th cycle (t_alert = t + k - 1).
      - Lead Time = T_i - t_alert = RUL(t_alert).
      - If no k consecutive cycles: TTFD=None, Lead Time=None, Missed Detection=True.
    """
    df_eval = df_val[["unit_number", "time_cycles", "RUL"]].copy()
    df_eval["score"] = anomaly_scores
    df_eval["is_anomaly_raw"] = (anomaly_scores > threshold).astype(int)

    engine_summary_records = []
    engine_groups = df_eval.groupby("unit_number")

    total_early_cycles = 0
    total_early_alarms = 0
    total_late_cycles = 0
    total_late_hits = 0
    spearman_correlations = []

    for u, g in engine_groups:
        g = g.sort_values("time_cycles").reset_index(drop=True)
        max_cycle = g["time_cycles"].max()
        raw_flags = g["is_anomaly_raw"].values
        scores = g["score"].values
        cycles = g["time_cycles"].values
        ruls = g["RUL"].values

        # 1. Early-life FAR tracking (t <= 30)
        early_mask = (cycles <= HEALTHY_EARLY_CYCLE_THRESHOLD)
        if np.any(early_mask):
            total_early_cycles += np.sum(early_mask)
            total_early_alarms += np.sum(raw_flags[early_mask])

        # 2. Late-life Sensitivity tracking (RUL <= 20)
        late_mask = (ruls <= LATE_LIFE_RUL_THRESHOLD)
        if np.any(late_mask):
            total_late_cycles += np.sum(late_mask)
            total_late_hits += np.sum(raw_flags[late_mask])

        # 3. Monotonicity: Spearman rank correlation between score and operational cycle
        if len(scores) > 5 and np.std(scores) > 1e-6:
            r_spearman, _ = spearmanr(cycles, scores)
            if not np.isnan(r_spearman):
                spearman_correlations.append(r_spearman)

        # 4. Persistent detection tracking (k consecutive cycles above threshold)
        alert_cycle = None
        lead_time = None
        persistent_detected = False
        missed_detection = True

        for idx in range(len(raw_flags) - persistence_k + 1):
            if np.all(raw_flags[idx : idx + persistence_k] == 1):
                # Alert confirmed at the k-th cycle in the sequence
                confirmation_idx = idx + persistence_k - 1
                alert_cycle = int(cycles[confirmation_idx])
                lead_time = int(ruls[confirmation_idx])
                persistent_detected = True
                missed_detection = False
                break

        engine_summary_records.append({
            "unit_number": int(u),
            "total_cycles": int(max_cycle),
            "persistent_anomaly_detected": persistent_detected,
            "missed_detection": missed_detection,
            "time_to_first_detection": alert_cycle,
            "maintenance_lead_time": lead_time,
            "mean_anomaly_score": round(float(np.mean(scores)), 4),
            "max_anomaly_score": round(float(np.max(scores)), 4),
        })

    per_engine_df = pd.DataFrame(engine_summary_records)

    # Compute aggregate metrics
    far_val = (total_early_alarms / total_early_cycles) if total_early_cycles > 0 else 0.0
    sensitivity_val = (total_late_hits / total_late_cycles) if total_late_cycles > 0 else 0.0
    mean_spearman = float(np.mean(spearman_correlations)) if spearman_correlations else 0.0

    detected_subset = per_engine_df[per_engine_df["persistent_anomaly_detected"] == True]
    num_total_engines = len(per_engine_df)
    num_missed = int(per_engine_df["missed_detection"].sum())
    missed_rate = num_missed / num_total_engines

    mean_ttfd = round(float(detected_subset["time_to_first_detection"].mean()), 2) if len(detected_subset) > 0 else None
    median_ttfd = round(float(detected_subset["time_to_first_detection"].median()), 2) if len(detected_subset) > 0 else None
    mean_lead_time = round(float(detected_subset["maintenance_lead_time"].mean()), 2) if len(detected_subset) > 0 else None
    median_lead_time = round(float(detected_subset["maintenance_lead_time"].median()), 2) if len(detected_subset) > 0 else None

    summary_metrics = {
        "early_life_far": round(float(far_val), 4),
        "late_life_sensitivity": round(float(sensitivity_val), 4),
        "mean_spearman_correlation": round(mean_spearman, 4),
        "total_validation_engines": num_total_engines,
        "persistent_detected_count": len(detected_subset),
        "missed_detection_count": num_missed,
        "missed_detection_rate": round(float(missed_rate), 4),
        "mean_ttfd": mean_ttfd,
        "median_ttfd": median_ttfd,
        "mean_lead_time": mean_lead_time,
        "median_lead_time": median_lead_time,
    }

    return summary_metrics, per_engine_df


def run_all_anomaly_detection_experiments():
    print("=" * 80)
    print("PHASE 4 - STEP 4.4: DEVELOP ANOMALY DETECTION FOR UNUSUAL SENSOR PATTERNS")
    print(f"Random State: {RANDOM_STATE} | Persistence Window: {PERSISTENCE_WINDOW} cycles")
    print("=" * 80)

    data_dir = os.path.join(PROJECT_ROOT, "dataset", "processed", "scaled")
    reports_dir = os.path.join(PROJECT_ROOT, "reports", "models")
    figures_dir = os.path.join(PROJECT_ROOT, "reports", "figures")
    models_dir = os.path.join(PROJECT_ROOT, "models", "anomaly_detection")
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    software_versions = get_software_versions()

    all_performance_records = []
    all_engine_records = []
    manifest_records = {
        "_environment_metadata": software_versions,
        "methodology": {
            "primary_model": "M1_RegimeAware_IsolationForest",
            "benchmark_model": "M2_PCA_SPE",
            "ablation_model": "M3_Unconditioned_IsolationForest (FD002/FD004 only)",
            "split_hierarchy": "Engine-level 3-cohort (Cohort A: Fitting, Cohort B: Threshold Calibration, Cohort C: Validation)",
            "threshold_calibration_percentile": CALIBRATION_PERCENTILE,
            "persistence_condition": f"{PERSISTENCE_WINDOW} consecutive cycles above tau (alert confirmed at 3rd cycle)",
            "leakage_safeguard": "Cohort C untouched; calibration strictly on Cohort B healthy data",
        },
        "experiments": {},
    }

    sample_trajectories_data = {}  # for plotting
    sample_rul_scatter_data = {}

    for subset in SUBSETS:
        print(f"\n{'#' * 40} SUBSET: {subset} {'#' * 40}")
        is_multi = subset in ["FD002", "FD004"]
        feature_cols = TIER1_CORE_SENSORS  # 14 degradation sensors

        train_path = os.path.join(data_dir, f"train_{subset}_scaled.csv")
        val_path = os.path.join(data_dir, f"validation_{subset}_scaled.csv")
        assert os.path.exists(train_path), f"Missing {train_path}"
        assert os.path.exists(val_path), f"Missing {val_path}"

        df_train_raw = pd.read_csv(train_path)
        df_val_raw = pd.read_csv(val_path)

        # 1. Engine-Level 3-Cohort Separation
        df_cohort_a, df_cohort_b, engines_a, engines_b = split_engine_cohorts(df_train_raw, random_state=RANDOM_STATE)
        engines_val = sorted(df_val_raw["unit_number"].unique())

        print(f"[{subset}] Engine Partitioning: Cohort A (Fitting)={len(engines_a)} eng | "
              f"Cohort B (Calibration)={len(engines_b)} eng | Cohort C (Validation)={len(engines_val)} eng")

        # Extract healthy observations from Cohorts A and B
        df_healthy_a = filter_healthy_observations(df_cohort_a)
        df_healthy_b = filter_healthy_observations(df_cohort_b)
        print(f"  - Cohort A Healthy Rows: {len(df_healthy_a)} / {len(df_cohort_a)}")
        print(f"  - Cohort B Healthy Rows: {len(df_healthy_b)} / {len(df_cohort_b)}")

        # 2. Fit Regime Normalizer strictly on Cohort A Healthy Data
        normalizer = RegimeNormalizer(is_multi_regime=is_multi, n_clusters=6, random_state=RANDOM_STATE)
        normalizer.fit(df_healthy_a, feature_cols)

        # Compute Residuals
        Z_healthy_a = normalizer.transform(df_healthy_a)
        Z_healthy_b = normalizer.transform(df_healthy_b)
        Z_val = normalizer.transform(df_val_raw)

        # Define Models to run on this subset
        models_to_run = [
            ("M1_RegimeAware_iForest", "Regime-Aware Isolation Forest", "residual"),
            ("M2_PCA_SPE", "PCA Squared Prediction Error", "residual"),
        ]
        if is_multi:
            models_to_run.append(("M3_Unconditioned_iForest", "Unconditioned Isolation Forest (Ablation)", "raw_sensor"))

        for m_id, m_name, input_type in models_to_run:
            t0 = time.time()
            print(f"  -> Running {m_name} ({m_id})...", end=" ", flush=True)

            if m_id == "M1_RegimeAware_iForest":
                detector = IsolationForest(
                    n_estimators=100,
                    max_samples=0.8,
                    contamination="auto",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                )
                detector.fit(Z_healthy_a)
                # Anomaly score is negated decision function (higher = more anomalous)
                calib_scores = -detector.decision_function(Z_healthy_b)
                val_scores = -detector.decision_function(Z_val)

            elif m_id == "M2_PCA_SPE":
                detector = PCASPEAnomalyDetector(variance_ratio=0.95, random_state=RANDOM_STATE)
                detector.fit(Z_healthy_a)
                calib_scores = detector.score(Z_healthy_b)
                val_scores = detector.score(Z_val)

            elif m_id == "M3_Unconditioned_iForest":
                # Ablation model: trained directly on raw scaled sensors without regime conditioning
                raw_healthy_a = df_healthy_a[feature_cols].values
                raw_healthy_b = df_healthy_b[feature_cols].values
                raw_val = df_val_raw[feature_cols].values

                detector = IsolationForest(
                    n_estimators=100,
                    max_samples=0.8,
                    contamination="auto",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                )
                detector.fit(raw_healthy_a)
                calib_scores = -detector.decision_function(raw_healthy_b)
                val_scores = -detector.decision_function(raw_val)

            # 3. Independent Decision Threshold Calibration on Cohort B
            tau = float(np.percentile(calib_scores, CALIBRATION_PERCENTILE))

            # 4. Evaluation on Untouched Cohort C (Validation)
            metrics, per_eng_df = evaluate_engine_trajectories(
                df_val=df_val_raw,
                anomaly_scores=val_scores,
                threshold=tau,
                persistence_k=PERSISTENCE_WINDOW,
            )
            elapsed = time.time() - t0
            print(f"Done! ({elapsed:.1f}s) | Threshold(tau)={tau:.4f} | Val FAR={metrics['early_life_far'] * 100:.2f}% | "
                  f"Sensitivity={metrics['late_life_sensitivity'] * 100:.2f}% | Spearman={metrics['mean_spearman_correlation']:.4f} | "
                  f"Missed={metrics['missed_detection_count']}/{metrics['total_validation_engines']}")

            # Store records
            perf_record = {
                "subset": subset,
                "model_id": m_id,
                "model_name": m_name,
                "operating_regime_mode": "Regime-Normalized" if input_type == "residual" else "Unconditioned (Raw Sensors)",
                "feature_count": len(feature_cols),
                "threshold_tau": round(tau, 6),
                "calibration_nominal_far": 0.05,
                "early_life_far": metrics["early_life_far"],
                "late_life_sensitivity": metrics["late_life_sensitivity"],
                "mean_spearman_correlation": metrics["mean_spearman_correlation"],
                "total_validation_engines": metrics["total_validation_engines"],
                "persistent_detected_count": metrics["persistent_detected_count"],
                "missed_detection_count": metrics["missed_detection_count"],
                "missed_detection_rate": metrics["missed_detection_rate"],
                "mean_ttfd": metrics["mean_ttfd"],
                "median_ttfd": metrics["median_ttfd"],
                "mean_lead_time": metrics["mean_lead_time"],
                "median_lead_time": metrics["median_lead_time"],
                "training_time_sec": round(elapsed, 2),
            }
            all_performance_records.append(perf_record)

            # Add model_id and subset to per_engine_df
            per_eng_df["subset"] = subset
            per_eng_df["model_id"] = m_id
            all_engine_records.append(per_eng_df)

            # Manifest entry
            exp_key = f"{subset}_{m_id}"
            manifest_records["experiments"][exp_key] = {
                "subset": subset,
                "model_id": m_id,
                "model_name": m_name,
                "operating_regime_mode": perf_record["operating_regime_mode"],
                "feature_columns": feature_cols,
                "cohort_a_engines_count": len(engines_a),
                "cohort_b_engines_count": len(engines_b),
                "cohort_c_validation_engines_count": len(engines_val),
                "threshold_calibration": {
                    "cohort": "Cohort B (Training Engines Group 2)",
                    "percentile": CALIBRATION_PERCENTILE,
                    "threshold_value": tau,
                },
                "metrics": metrics,
            }

            # Cache primary model trajectories for sample engine visualization
            if m_id == "M1_RegimeAware_iForest":
                first_unit = engines_val[0]
                unit_mask = (df_val_raw["unit_number"] == first_unit)
                sample_trajectories_data[subset] = {
                    "unit": first_unit,
                    "cycles": df_val_raw.loc[unit_mask, "time_cycles"].values,
                    "ruls": df_val_raw.loc[unit_mask, "RUL"].values,
                    "scores": val_scores[unit_mask],
                    "threshold": tau,
                    "ttfd": per_eng_df.loc[per_eng_df["unit_number"] == first_unit, "time_to_first_detection"].values[0],
                }
                sample_rul_scatter_data[subset] = {
                    "ruls": df_val_raw["RUL"].values,
                    "scores": val_scores,
                    "threshold": tau,
                }

    # Save outputs
    # 1. Performance CSV
    df_perf_all = pd.DataFrame(all_performance_records)
    perf_csv_path = os.path.join(reports_dir, "anomaly_detection_performance.csv")
    df_perf_all.to_csv(perf_csv_path, index=False)
    print(f"\n[SAVE] Anomaly detection performance comparison table: {perf_csv_path}")

    # 2. Per-Engine CSV
    df_engine_all = pd.concat(all_engine_records, ignore_index=True)
    engine_csv_path = os.path.join(reports_dir, "anomaly_detection_per_engine.csv")
    df_engine_all.to_csv(engine_csv_path, index=False)
    print(f"[SAVE] Per-engine anomaly detection metrics: {engine_csv_path}")

    # 3. Anomaly Manifest JSON
    manifest_path = os.path.join(models_dir, "anomaly_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_records, f, indent=2)
    print(f"[SAVE] Anomaly detection manifest: {manifest_path}")

    # 4. Generate Figures
    generate_anomaly_figures(sample_trajectories_data, sample_rul_scatter_data, df_perf_all, figures_dir)


def generate_anomaly_figures(trajectories_dict: dict, scatter_dict: dict, df_perf: pd.DataFrame, figures_dir: str):
    print("\n" + "=" * 80)
    print("GENERATING ANOMALY DETECTION VISUALIZATION FIGURES")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Figure 1: Anomaly Score Trajectories across Engine Lifecycles (4 Subsets)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    axes = axes.flatten()

    for idx, sub in enumerate(SUBSETS):
        ax = axes[idx]
        data = trajectories_dict[sub]
        unit = data["unit"]
        cycles = data["cycles"]
        scores = data["scores"]
        tau = data["threshold"]
        ttfd = data["ttfd"]

        ax.plot(cycles, scores, color="#1F77B4", linewidth=1.8, label="M1 Anomaly Score S(x)")
        ax.axhline(tau, color="#D62728", linestyle="--", linewidth=1.6, label=f"Calibrated Threshold (tau = {tau:.3f})")

        if ttfd is not None and not pd.isna(ttfd):
            ax.axvline(ttfd, color="#2CA02C", linestyle="-.", linewidth=1.8, label=f"Persistent Alert (TTFD = {ttfd})")
            ax.scatter([ttfd], [scores[list(cycles).index(ttfd)]], color="#2CA02C", s=80, zorder=5)

        ax.set_title(f"{sub} Engine Unit #{unit} Health Degradation Profile", fontsize=12, fontweight="bold")
        ax.set_xlabel("Operational Time Cycles", fontsize=11)
        ax.set_ylabel("Anomaly Severity Score", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper left", frameon=True, fontsize=9)

    fig.suptitle("Phase 4 - Step 4.4: Primary Anomaly Score Degradation Trajectories (Validation Cohort)",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    f1_path = os.path.join(figures_dir, "anomaly_score_trajectories.png")
    fig.savefig(f1_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved: {f1_path}")

    # -------------------------------------------------------------------------
    # Figure 2: Anomaly Score vs. Remaining Useful Life (RUL) Scatter / Trend
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    axes = axes.flatten()

    for idx, sub in enumerate(SUBSETS):
        ax = axes[idx]
        data = scatter_dict[sub]
        ruls = data["ruls"]
        scores = data["scores"]
        tau = data["threshold"]

        # Subsample for smooth rendering if dataset is large
        if len(ruls) > 5000:
            sample_idx = np.random.choice(len(ruls), 5000, replace=False)
            r_plot, s_plot = ruls[sample_idx], scores[sample_idx]
        else:
            r_plot, s_plot = ruls, scores

        ax.scatter(r_plot, s_plot, alpha=0.25, color="#34495E", s=12, edgecolors="none")
        ax.axhline(tau, color="#E74C3C", linestyle="--", linewidth=1.6, label=f"Decision Threshold (tau = {tau:.3f})")

        # Bin mean trendline
        bins = np.linspace(0, 200, 25)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_means = []
        for b_low, b_high in zip(bins[:-1], bins[1:]):
            mask = (ruls >= b_low) & (ruls < b_high)
            bin_means.append(np.mean(scores[mask]) if np.any(mask) else np.nan)
        ax.plot(bin_centers, bin_means, color="#E67E22", linewidth=2.5, label="Mean Anomaly Profile")

        ax.set_title(f"{sub} Anomaly Score vs. True RUL", fontsize=12, fontweight="bold")
        ax.set_xlabel("True Remaining Useful Life (cycles)", fontsize=11)
        ax.set_ylabel("Anomaly Severity Score", fontsize=11)
        ax.invert_xaxis()  # RUL decreases towards failure on right
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", frameon=True, fontsize=9)

    fig.suptitle("Phase 4 - Step 4.4: Sensor Anomaly Score Monotonicity vs. Engine Degradation (RUL)",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    f2_path = os.path.join(figures_dir, "anomaly_score_vs_rul.png")
    fig.savefig(f2_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved: {f2_path}")


if __name__ == "__main__":
    run_all_anomaly_detection_experiments()
