# Phase 4 – Roadmap Alignment Audit (PDF Steps 4.6–4.12)

**Project:** AI-Based Aircraft Engine Health Monitoring, Fault Diagnosis and Predictive Maintenance  
**Reference Document:** Master Project Roadmap PDF (Phase 4: Machine Learning and Model Evaluation)  
**Audit Target:** Verification of completed project artifacts against PDF Steps 4.6 through 4.12  
**Auditor:** Antigravity AI (Pair Programming Assistant)  
**Date & Timestamp:** 2026-11-04T13:22:00+05:30  
**Status:** **AUDIT COMPLETED**  

---

## 1. Executive Summary & Audit Overview

Following the completion of baseline experiments (Step 4.1 & 4.2), XGBoost regression models (Step 4.2), LSTM sequential architectures (Step 4.3), and consolidated model evaluation synthesis (Step 4.4), this audit benchmarks all existing codebase artifacts against the authoritative **Master Project Roadmap PDF** requirements for **Phase 4 – Steps 4.6 through 4.12**.

In strict accordance with audit guidelines:
- No models were retrained.
- No historical experiment results were altered.
- Manifests are **not** assumed to prove that binary model checkpoints were persisted.
- Comparative summary charts are **not** assumed to satisfy direct predicted-versus-actual trajectory plotting requirements.

### Scorecard Summary

| Roadmap Step | Roadmap Requirement Title | Audit Status | Key Evidence / Findings | Missing Work / Gaps |
| :---: | :--- | :---: | :--- | :--- |
| **Step 4.6** | Record model inputs, target, training configuration, random seed, software versions | **PARTIALLY COMPLETED** | Inputs, targets, hyperparameters, seeds recorded in manifests and markdown reports. | Software versions documented in markdown headers and `pyproject.toml`, but omitted from JSON manifests. |
| **Step 4.7** | Evaluate RUL predictions using MAE, RMSE; calculate NASA scoring function | **FULLY COMPLETED** | All 108 runs evaluated with RMSE, MAE, $R^2$, and NASA asymmetric score across partitions. | None. Mathematical definitions and decoupled reporting fully verified. |
| **Step 4.8** | Plot predicted RUL against actual RUL; inspect errors by engine and subset | **PARTIALLY COMPLETED** | Errors thoroughly inspected by engine (3,834 records) and subset (box plots, bar charts). | Direct predicted RUL vs. actual RUL trajectory curves or $\hat{y}$ vs $y$ scatter plots were not generated. |
| **Step 4.9** | Compare models using the same validation/test protocol | **FULLY COMPLETED** | Exactly identical 80/20 engine split, $y \le 125$ capping, and unclipped test scoring across all 108 runs. | None. Unified cross-architecture comparison table archived. |
| **Step 4.10** | Check for data leakage, overfitting, and performance differences across subsets | **FULLY COMPLETED** | Strict engine isolation, independent scalers, train/val/test tracking, early stopping, and multi-condition analysis. | None. Audited in multiple verified audit reports. |
| **Step 4.11** | Save trained models, preprocessing objects, and evaluation reports | **PARTIALLY COMPLETED** | Preprocessing scalers saved (`models/scalers/*.pkl`); evaluation reports saved in `docs/` and `reports/`. | Trained model binary weights/checkpoints (`.pkl`, `.pt`, `.json`) were deliberately not saved to disk. |
| **Step 4.12** | Document model limitations and avoid unsupported fault-identification claims | **FULLY COMPLETED** | Comprehensive limitations documented; strictly framed as RUL regression without unsupported damage claims. | None. Explicit boundaries between academic simulation and airworthy flight certification documented. |

---

## 2. Detailed Roadmap Verification by Step

### Step 4.6: Record Configuration Metadata & Software Versions
* **PDF Requirement:** Record model inputs, target, training configuration, random seed and software versions.
* **Existing Implementations & File Paths:**
  - `models/baselines/baseline_manifest.json`: Records subset, model family, variant, feature tier, input dimension, target definitions, and timing.
  - `models/xgboost/xgboost_manifest.json`: Records subset, variant, tier, feature dim, tree hyperparameters (`n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `tree_method`).
  - `models/lstm/lstm_manifest.json`: Records subset, variant, tier, feature dim, network architecture (`hidden_dim`, `dropout`, `batch_size`, `lr`, `weight_decay`), per-epoch loss curves, and `best_epoch`.
  - Global random seed `RANDOM_STATE = 42` is explicitly hardcoded and logged across all scripts.
  - Software versions are documented in report markdown headers and managed via `pyproject.toml` (`python==3.15.0`, `torch==2.15.0+cpu`, `xgboost==3.4.1`, `scikit-learn==1.9.1`, `pandas==3.1.0`, `numpy==2.5.4`, `matplotlib==3.10.8`).
* **Verification Evidence:**
  - `baseline_manifest.json` (52 configurations logged).
  - `xgboost_manifest.json` (40 configurations logged).
  - `lstm_manifest.json` (16 configurations logged).
* **Missing Work & Gap Analysis:**
  - While software versions are recorded in documentation files and virtual environment metadata, they are **not programmatically embedded** inside the JSON manifest files themselves (e.g., `"software_versions": {"torch": "2.15.0", "xgboost": "3.4.1"}`).
* **Audit Status:** **PARTIALLY COMPLETED**
* **Recommended Corrective Action:** Update manifest-generation utilities in future phases to automatically capture `sys.version` and installed package versions in JSON metadata.

---

### Step 4.7: Evaluate RUL Predictions Using MAE, RMSE, and NASA Scoring Function
* **PDF Requirement:** Evaluate RUL predictions using MAE and RMSE; calculate the NASA scoring function.
* **Existing Implementations & File Paths:**
  - Core scoring implementations:
    - [`models/baselines/run_baseline_models.py`](file:///c:/Users/LENOVO/Documents/Aircraft_Engine_AI/models/baselines/run_baseline_models.py#L135-L160)
    - [`models/xgboost/run_xgboost_models.py`](file:///c:/Users/LENOVO/Documents/Aircraft_Engine_AI/models/xgboost/run_xgboost_models.py#L135-L160)
    - [`models/lstm/run_lstm_models.py`](file:///c:/Users/LENOVO/Documents/Aircraft_Engine_AI/models/lstm/run_lstm_models.py#L160-L180)
    - [`models/evaluation/consolidate_model_evaluations.py`](file:///c:/Users/LENOVO/Documents/Aircraft_Engine_AI/models/evaluation/consolidate_model_evaluations.py)
  - Data artifacts:
    - [`reports/models/master_model_comparison.csv`](file:///c:/Users/LENOVO/Documents/Aircraft_Engine_AI/reports/models/master_model_comparison.csv) (108 rows)
    - [`reports/models/model_selection_summary.csv`](file:///c:/Users/LENOVO/Documents/Aircraft_Engine_AI/reports/models/model_selection_summary.csv) (20 rows)
* **Verification Evidence:**
  - Every single experiment record contains:
    - `val_capped_rmse`, `val_capped_mae`, `val_capped_r2`, `val_capped_score`
    - `val_uncapped_rmse`, `val_uncapped_mae`, `val_uncapped_r2`, `val_uncapped_score`
    - `test_uncapped_rmse`, `test_uncapped_mae`, `test_uncapped_score`
    - `test_unpadded_rmse`, `test_unpadded_score`, `test_padded_rmse`, `test_padded_score`
* **Missing Work & Gap Analysis:** None. All required metrics are implemented, calculated, verified, and decoupled.
* **Audit Status:** **FULLY COMPLETED**

---

### Step 4.8: Plot Predicted RUL Against Actual RUL & Error Inspection
* **PDF Requirement:** Plot predicted RUL against actual RUL and inspect errors by engine and subset.
* **Existing Implementations & File Paths:**
  - *Error Inspection by Engine:*
    - `reports/models/baseline_per_engine_validation.csv` (1,846 records)
    - `reports/models/xgboost_per_engine_validation.csv` (1,420 records)
    - `reports/models/lstm_per_engine_validation.csv` (568 records)
    - Total per-engine evaluations: **3,834 records** across all 142 validation engines.
  - *Error Inspection by Subset & Visualizations:*
    - `reports/figures/model_comparison_val_test_rmse.png` (Grouped bar chart across subsets)
    - `reports/figures/nasa_score_tradeoff_comparison.png` (NASA penalty comparison across subsets)
    - `reports/figures/per_engine_rmse_distribution.png` (Box plots showing engine-level RMSE variance)
* **Missing Work & Gap Analysis:**
  - While error inspection by engine and subset is exhaustively completed, **the specific requirement to plot predicted RUL against actual RUL is missing**:
    - **No cycle-by-cycle degradation trajectory plots** ($\hat{y}(t)$ vs. $y(t)$ over engine cycles for selected validation/test units) were generated.
    - **No predicted vs. actual scatter plots** ($\hat{y}$ vs. $y$ on the test cohort with reference line $y=x$) were saved.
* **Audit Status:** **PARTIALLY COMPLETED**
* **Recommended Corrective Action:** If trajectory plotting is required before proceeding, create a non-retraining script that loads the winning model specifications, evaluates a small sample of test engines, and renders actual vs. predicted degradation trajectories.

---

### Step 4.9: Standardized Model Comparison Protocol
* **PDF Requirement:** Compare models using the same validation/test protocol.
* **Existing Implementations & File Paths:**
  - Master comparison registry: [`reports/models/master_model_comparison.csv`](file:///c:/Users/LENOVO/Documents/Aircraft_Engine_AI/reports/models/master_model_comparison.csv)
  - Curated decision scorecard: [`reports/models/model_selection_summary.csv`](file:///c:/Users/LENOVO/Documents/Aircraft_Engine_AI/reports/models/model_selection_summary.csv)
  - Formal protocol report: [`docs/phase4_step4_4_model_evaluation_protocol.md`](file:///c:/Users/LENOVO/Documents/Aircraft_Engine_AI/docs/phase4_step4_4_model_evaluation_protocol.md)
* **Verification Evidence:**
  - All 108 model configurations were evaluated using:
    - Identical engine-level split assignments (80/20 for FD001/FD003; 208/52 for FD002; 199/50 for FD004).
    - Identical target capping threshold ($y \le 125$) during training and validation model selection.
    - Identical unclipped continuous evaluation on the official test set.
    - Uniform model selection criterion: exclusively minimum `val_capped_rmse`.
* **Missing Work & Gap Analysis:** None.
* **Audit Status:** **FULLY COMPLETED**

---

### Step 4.10: Data Leakage, Overfitting, and Cross-Subset Audits
* **PDF Requirement:** Check for data leakage, overfitting and performance differences across subsets.
* **Existing Implementations & File Paths:**
  - Dedicated audit documents:
    - [`docs/phase4_step4_2_methodological_audit.md`](file:///c:/Users/LENOVO/Documents/Aircraft_Engine_AI/docs/phase4_step4_2_methodological_audit.md)
    - [`docs/phase4_step4_2_xgboost_methodological_audit.md`](file:///c:/Users/LENOVO/Documents/Aircraft_Engine_AI/docs/phase4_step4_2_xgboost_methodological_audit.md)
    - [`docs/phase4_step4_3_lstm_methodological_audit.md`](file:///c:/Users/LENOVO/Documents/Aircraft_Engine_AI/docs/phase4_step4_3_lstm_methodological_audit.md)
* **Verification Evidence:**
  - *Data Leakage Prevention:* Preprocessing scalers fit strictly on training engines; zero engine overlap across partitions; test data strictly isolated from training, tuning, and early stopping.
  - *Overfitting Mitigation:* Early stopping implemented with patience (10 epochs for XGBoost, 12 epochs for LSTM) monitoring validation capped RMSE. Decoupled reporting of train loss vs. validation RMSE vs. official test RMSE.
  - *Subset Dynamics:* Documented fundamental behavioral divergence between single-condition (FD001/FD003, where XGBoost dominates) and multi-condition (FD002/FD004, where LSTM achieves lower validation RMSE while XGBoost achieves lower test NASA scores).
* **Missing Work & Gap Analysis:** None.
* **Audit Status:** **FULLY COMPLETED**

---

### Step 4.11: Persistence of Models, Preprocessors, and Reports
* **PDF Requirement:** Save trained models, preprocessing objects and evaluation reports.
* **Existing Implementations & File Paths:**
  - *Preprocessing Objects (Saved):*
    - [`models/scalers/scaler_FD001_tier3.pkl`](file:///c:/Users/LENOVO/Documents/Aircraft_Engine_AI/models/scalers/scaler_FD001_tier3.pkl)
    - [`models/scalers/scaler_FD002_tier3.pkl`](file:///c:/Users/LENOVO/Documents/Aircraft_Engine_AI/models/scalers/scaler_FD002_tier3.pkl)
    - [`models/scalers/scaler_FD003_tier3.pkl`](file:///c:/Users/LENOVO/Documents/Aircraft_Engine_AI/models/scalers/scaler_FD003_tier3.pkl)
    - [`models/scalers/scaler_FD004_tier3.pkl`](file:///c:/Users/LENOVO/Documents/Aircraft_Engine_AI/models/scalers/scaler_FD004_tier3.pkl)
    - [`models/scalers/scaler_manifest.json`](file:///c:/Users/LENOVO/Documents/Aircraft_Engine_AI/models/scalers/scaler_manifest.json)
  - *Evaluation Reports (Saved):*
    - All evaluation reports and execution summaries preserved in `docs/` (`phase4_step4_2_*.md`, `phase4_step4_3_*.md`, `phase4_step4_4_*.md`).
    - Tabular performance records preserved in `reports/models/` (`master_model_comparison.csv`, `model_selection_summary.csv`, per-engine CSVs).
* **Missing Work & Gap Analysis:**
  - **Trained Model Checkpoint Binaries Were NOT Saved to Disk:**
    - The directories `models/baselines/`, `models/xgboost/`, and `models/lstm/` contain code scripts and JSON manifests, but **zero model binary files** (`.pkl`, `.json` booster, `.pt`, `.pth`).
    - Models were evaluated in memory and discarded, relying on manifest parameter logging and deterministic seeds for reproducibility.
* **Audit Status:** **PARTIALLY COMPLETED**
* **Recommended Corrective Action:** If trained model binary files are required for Phase 5 deployment, serialize the winning model checkpoints (4 XGBoost winners and 4 LSTM winners) to disk under a dedicated `models/checkpoints/` directory.

---

### Step 4.12: Model Limitations & Avoid Unsupported Claims
* **PDF Requirement:** Document model limitations and avoid unsupported fault-identification claims.
* **Existing Implementations & File Paths:**
  - Formally documented in:
    - [`docs/phase4_step4_4_model_evaluation_protocol.md`](file:///c:/Users/LENOVO/Documents/Aircraft_Engine_AI/docs/phase4_step4_4_model_evaluation_protocol.md#L6-formal-methodological-audits--limitations)
    - [`docs/phase4_step4_3_lstm_methodological_audit.md`](file:///c:/Users/LENOVO/Documents/Aircraft_Engine_AI/docs/phase4_step4_3_lstm_methodological_audit.md)
* **Verification Evidence:**
  - *Documented Methodological Limitations:*
    1. **Target Capping Asymmetry:** Models trained on $y \le 125$ vs. uncapped test evaluations.
    2. **Sequence vs. Point Evaluation:** Continuous sliding window validation vs. single final-cycle test evaluation.
    3. **NASA Score Asymmetric Sensitivity:** Exponential penalties on late over-estimations skewing aggregate scores.
    4. **NASA Score Error Clipping:** Documented the defensive clipping difference between Baseline/XGBoost and LSTM, noting that prediction extrema were not persisted.
    5. **Lifecycle Regime Feasibility:** Documented that early-life vs. near-failure sub-metrics cannot be computed from CSV summary tables without prediction persistence.
    6. **Academic Simulation vs. Flight Readiness:** Explicitly clarified that model selections are academic recommendations for the C-MAPSS simulation benchmark, not real-aircraft flight-certified operational systems.
  - *Avoided Unsupported Fault-Identification Claims:*
    - The project models Remaining Useful Life (RUL regression).
    - No unvalidated claims are made regarding individual component fault isolation (e.g., claiming to pinpoint specific fan blade cracks or turbine seal tears from scalar RUL outputs), maintaining strict scientific integrity.
* **Missing Work & Gap Analysis:** None.
* **Audit Status:** **FULLY COMPLETED**

---

## 3. Comprehensive Roadmap Alignment Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│             Phase 4 Master Roadmap Alignment Scorecard (Steps 4.6–4.12)     │
├──────┬─────────────────────────────────────────────────┬────────────────────┤
│ Step │ Requirement Summary                             │ Compliance Status  │
├──────┼─────────────────────────────────────────────────┼────────────────────┤
│ 4.6  │ Configuration, inputs, targets, software logged │ PARTIALLY COMPLETED│
│ 4.7  │ MAE, RMSE, NASA score evaluation                │ FULLY COMPLETED    │
│ 4.8  │ Predicted vs actual plots & error inspection    │ PARTIALLY COMPLETED│
│ 4.9  │ Standardized cross-model validation protocol    │ FULLY COMPLETED    │
│ 4.10 │ Leakage, overfitting, and subset audits         │ FULLY COMPLETED    │
│ 4.11 │ Persistence of models, preprocessors, reports   │ PARTIALLY COMPLETED│
│ 4.12 │ Model limitations & no unsupported claims       │ FULLY COMPLETED    │
└──────┴─────────────────────────────────────────────────┴────────────────────┘
```

### Identification of Uncompleted / Missing Work:
1. **Missing Predicted vs. Actual Plots (Step 4.8):** Direct cycle-by-cycle degradation trajectory curves ($\hat{y}(t)$ vs $y(t)$) and test scatter plots with $y=x$ reference lines were not rendered.
2. **Missing Model Binary Checkpoints (Step 4.11):** Trained model weights (`.pkl`, `.pt`, `.json`) were not serialized to disk; only manifests and preprocessors exist.
3. **Missing Software Versions in JSON Manifests (Step 4.6):** Software versions are in markdown headers and `pyproject.toml`, but not embedded in the JSON manifests.

---

## 4. Recommended Next Steps

Before initiating downstream tasks (such as Anomaly Detection, Kalman Filtering, or Phase 5):
1. **Address Step 4.8 Gap:** Render direct predicted-versus-actual trajectory plots for selected engines across the winning models.
2. **Address Step 4.11 Gap:** Serialize the approved champion model binaries to a dedicated checkpoint directory if required for inference in Phase 5.
3. **Awaiting User Review:** Await user directives on whether to resolve these partial gaps or proceed to the next roadmap milestone.
