# Phase 4 – Step 4.4: Model Evaluation Protocol Proposal

**Project:** AI-Based Aircraft Engine Health Monitoring, Fault Diagnosis and Predictive Maintenance  
**Phase:** 4 (Model Development and Benchmarking) – Step 4.4 (Model Evaluation Protocol)  
**Author:** Antigravity AI (Pair Programming Assistant)  
**Date:** 2026-11-04  
**Status:** **PRE-IMPLEMENTATION PROPOSAL (INSPECTION ONLY – NO RETRAINING)**  

---

## 1. Executive Summary & Context

The project has successfully executed and audited three consecutive modeling phases across all four NASA C-MAPSS subsets (`FD001`, `FD002`, `FD003`, `FD004`):
1. **Phase 4 – Step 4.1 & 4.2 Baseline Models:** 52 configurations evaluated across Mean Predictor, Ridge Regression, and Random Forest.
2. **Phase 4 – Step 4.2 XGBoost Regression:** 40 configurations evaluated across feature tiers, windowing paradigms, and hyperparameter tuning. Approved as the **primary tabular benchmark**.
3. **Phase 4 – Step 4.3 LSTM Sequence Models:** 16 configurations evaluated across 2-layer `Stacked_LSTM` and 1-layer `Single_LSTM`. Retained as a **specialized sequential architecture** for multi-condition degradation tracking.

Total completed experiments to date: **108 configurations** ($52 + 40 + 16$).

In strict accordance with the master project roadmap, **Phase 4 – Step 4.4 establishes the unified Model Evaluation Protocol**. This step does **not** train new models, retrain existing checkpoints, alter datasets, or re-run prior experiments. Instead, it consolidates, standardizes, synthesizes, and rigorously analyzes the complete body of empirical evidence generated across all 108 model runs to establish an **academic model-selection recommendation for the NASA C-MAPSS simulation benchmark**.

---

## 2. Metric Formula Inspection & Consistency Audit

### 2.1 Formula Implementations
Inspection of `models/baselines/run_baseline_models.py`, `models/xgboost/run_xgboost_models.py`, and `models/lstm/run_lstm_models.py` verifies the following mathematical formulations:

1. **Root Mean Squared Error (RMSE):**
   $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i)^2}$$
   - *Implementation:* Exactly identical across Baseline, XGBoost, and LSTM scripts via `np.sqrt(np.mean((y_pred - y_true)**2))`.
2. **Mean Absolute Error (MAE):**
   $$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |\hat{y}_i - y_i|$$
   - *Implementation:* Exactly identical across all scripts via `np.mean(np.abs(y_pred - y_true))`.
3. **Coefficient of Determination ($R^2$):**
   $$R^2 = 1 - \frac{\sum_{i=1}^N (\hat{y}_i - y_i)^2}{\sum_{i=1}^N (y_i - \bar{y})^2}$$
   - *Implementation:* Exactly identical across all scripts with explicit zero-variance protection (`ss_tot > 0`).
4. **NASA Asymmetric Scoring Function:**
   $$d_i = \hat{y}_i - y_i, \quad s_i = \begin{cases} e^{-d_i / 13} - 1, & d_i < 0 \quad (\text{Early prediction / Conservative}) \\ e^{d_i / 10} - 1, & d_i \ge 0 \quad (\text{Late prediction / High Risk}) \end{cases}, \quad S = \sum_{i=1}^N s_i$$

### 2.2 Numerical Overflow Protection vs. Mathematical NASA Formula
- **Code Inspection:**
  - In `run_baseline_models.py` and `run_xgboost_models.py`, a defensive guard `d_clipped = np.clip(d, -500.0, 500.0)` was included prior to exponentiation to prevent NumPy `FloatingPointError: overflow in exp`.
  - In `run_lstm_models.py`, `np.clip` was omitted, evaluating $d$ directly in the exponential function.
- **Artifact Extrema & Verification Limitation:**
  - Training targets capped at 125 cycles do not mathematically guarantee that unbounded continuous regressors (e.g., Ridge regression or unconstrained neural network layers) never produce out-of-distribution predictions exceeding 125 or 135 cycles.
  - Furthermore, a maximum ground-truth cycle label of 464 cycles does not, on its own, mathematically bound the prediction error $d = \hat{y} - y$ within $[-500, 500]$.
  - Inspection of the saved execution manifests (`baseline_manifest.json`, `xgboost_manifest.json`, `lstm_manifest.json`) and summary CSV tables reveals that **raw prediction vectors, error vectors, and prediction extrema ($\min \hat{y}, \max \hat{y}, \min d, \max d$) were not persisted to disk**.
- **Audit Conclusion & Identified Limitation:**
  - Because individual prediction arrays were not serialized to disk, **it cannot be conclusively proven from persisted artifacts alone that the $[-500, 500]$ clipping was never triggered in any baseline or XGBoost run**.
  - **Methodological Status: LIMITATION.** Exact numerical equivalence between the clipped formulation (`np.clip(d, -500, 500)`) and the unclipped formulation cannot be independently established from the saved artifacts without re-inferring all predictions.
  - **Metric Preservation Policy:** In accordance with the non-retraining directive, all existing approved aggregate metrics and historical NASA scores are strictly preserved without modification. Step 4.4 documents this implementation distinction transparently.

---

## 3. Standardized Evaluation Framework & Feasibility Audit

### 3.1 Multi-Dimensional Evaluation Structure
Step 4.4 defines a standardized evaluation framework across available empirical dimensions:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                Standardized Model Evaluation Framework (Step 4.4)           │
├───────────────────────────────────┬─────────────────────────────────────────┤
│ Dimension                         │ Evaluation Scope & Metrics              │
├───────────────────────────────────┼─────────────────────────────────────────┤
│ 1. Model Selection Partition      │ Validation Capped RMSE (RUL <= 125)     │
│ 2. Trajectory Diagnostic          │ Validation Uncapped RMSE, MAE, R²       │
│ 3. Engine-Level Aggregation       │ Per-Engine Mean & Std RMSE (142 engines)│
│ 4. Official Test Benchmark        │ Official Test Uncapped RMSE, MAE, Score │
│ 5. Test Padding Ablation          │ Unpadded (L >= 30) vs Padded (L < 30)   │
│ 6. Lifecycle Regime Accuracy      │ Early-Life (>75) vs Late-Life (<=30)    │
│                                   │ [FEASIBILITY AUDIT: DATA LIMITED]       │
└───────────────────────────────────┴─────────────────────────────────────────┘
```

### 3.2 Feasibility Audit of Lifecycle Regime Analysis (Early-Life vs. Near-Failure)
- **Proposed Scope:** Evaluating performance split by lifecycle phase:
  - *Early-Life:* Raw true $RUL > 75$ cycles.
  - *Near-Failure / Late-Life:* Raw true $RUL \le 30$ cycles.
- **Artifact Availability Audit:**
  - *Summary Tables:* `baseline_model_performance.csv`, `xgboost_model_performance.csv`, and `lstm_model_performance.csv` contain aggregate RMSE, MAE, $R^2$, and NASA scores for the whole partition.
  - *Per-Engine Tables:* `baseline_per_engine_validation.csv`, `xgboost_per_engine_validation.csv`, and `lstm_per_engine_validation.csv` contain single summary RMSE/MAE values aggregated per engine unit.
  - *Prediction Persistence:* Neither window-by-window continuous prediction arrays ($\hat{y}_t$) nor trained model binary checkpoints (`.pkl`, `.pt`) were saved to disk (excluded to avoid Git repository bloat).
- **Feasibility Finding:**
  - Computing quantitative early-life ($RUL > 75$) vs. near-failure ($RUL \le 30$) sub-metrics requires window-level predictions aligned with true cycle labels.
  - Because window-level predictions and model weights are not persisted on disk, and retraining is strictly prohibited, **lifecycle regime partitioning cannot be calculated from existing CSV tables**.
  - **Methodological Rule:** To maintain research integrity, these values will **not** be fabricated or approximated. Step 4.4 will formally document this limitation and provide a qualitative trajectory analysis based on the verified final-cycle validation RMSE metric (`val_final_cycle_rmse`), which reflects near-truncation accuracy.

---

## 4. Cross-Model Benchmark Comparison (Approved Results)

The approved winners from all five model families across all four subsets are summarized below:

### 4.1 Comprehensive Performance Table

| Subset | Model Family | Approved Winning Configuration | Feature Tier | Validation Capped RMSE (Selection) | Validation Capped $R^2$ | Validation Mean Engine RMSE | Official Test RMSE | Official Test MAE | Official Test NASA Score | Role / Status |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **FD001** | **Constant Mean** | `TrainingMean` | `Tier3_Extended24` | 41.9071 | -0.0034 | 41.7229 | 41.8103 | 36.2073 | 17,968.15 | Naive Benchmark |
| | **Ridge Regression** | `Ridge_FlattenedWindow` | `Tier1_Core14` | 16.5460 | 0.8436 | 16.3263 | 17.2208 | 13.0640 | 461.38 | Linear Baseline |
| | **Random Forest** | `RandomForest_SummaryStats` | `Tier1_Core14` | 13.6599 | 0.8934 | 13.3146 | 14.2898 | 10.8251 | 294.40 | Non-linear Baseline |
| | **XGBoost** | `XGBoost_Tuned_SummaryStats` | `Tier1_Core14` | **12.7506** | **0.9071** | **12.5614** | **13.4770** | **10.1317** | **268.06** | **PRIMARY BENCHMARK** |
| | **LSTM** | `Stacked_LSTM` | `Tier1_Core14` | 13.1011 | 0.9019 | 13.0086 | 14.8409 | 11.0324 | 365.94 | Experimental Sequence |
| **FD002** | **Constant Mean** | `TrainingMean` | `Tier3_Extended24` | 41.8779 | -0.0003 | 41.7058 | 53.7822 | 45.2798 | 124,050.19 | Naive Benchmark |
| | **Ridge Regression** | `Ridge_FlattenedWindow` | `Tier3_Extended24` | 18.5945 | 0.8028 | 18.3970 | 29.6229 | 20.3547 | 14,862.81 | Linear Baseline |
| | **Random Forest** | `RandomForest_SummaryStats` | `Tier2_Operational17` | 21.3865 | 0.7392 | 20.7368 | 32.3185 | 22.8687 | 28,182.99 | Non-linear Baseline |
| | **XGBoost** | `XGBoost_FlattenedWindow` | `Tier3_Extended24` | 19.3303 | 0.7869 | 19.2139 | 30.0483 | 20.7569 | **14,159.01** | **PRIMARY TABULAR** |
| | **LSTM** | `Stacked_LSTM` | `Tier3_Extended24` | **17.7875** | **0.8195** | **17.9829** | **29.0728** | **19.4573** | 17,346.86 | **SPECIALIZED SEQUENCE** |
| **FD003** | **Constant Mean** | `TrainingMean` | `Tier3_Extended24` | 42.0191 | -0.0115 | 42.1481 | 43.8301 | 37.1997 | 41,457.02 | Naive Benchmark |
| | **Ridge Regression** | `Ridge_FlattenedWindow` | `Tier3_Extended24` | 16.2547 | 0.8486 | 15.9392 | 18.4969 | 13.9189 | 599.89 | Linear Baseline |
| | **Random Forest** | `RandomForest_SummaryStats` | `Tier1_Core14` | 12.8673 | 0.9052 | 12.4497 | 14.9610 | 11.0028 | 431.96 | Non-linear Baseline |
| | **XGBoost** | `XGBoost_Tuned_SummaryStats` | `Tier3_Extended24` | **12.7926** | **0.9063** | **12.6834** | **14.2476** | **10.6958** | **312.21** | **PRIMARY BENCHMARK** |
| | **LSTM** | `Single_LSTM` | `Tier3_Extended24` | 14.1079 | 0.8860 | 14.0299 | 17.4228 | 13.0206 | 1,242.62 | Experimental Sequence |
| **FD004** | **Constant Mean** | `TrainingMean` | `Tier3_Extended24` | 41.2877 | -0.0020 | 41.7766 | 54.5533 | 46.4673 | 139,976.97 | Naive Benchmark |
| | **Ridge Regression** | `Ridge_FlattenedWindow` | `Tier3_Extended24` | 17.9086 | 0.8115 | 17.7554 | 31.9923 | 24.3168 | 9,222.09 | Linear Baseline |
| | **Random Forest** | `RandomForest_SummaryStats` | `Tier3_Extended24` | 17.6606 | 0.8167 | 17.2023 | 32.5761 | 24.8710 | 10,955.40 | Non-linear Baseline |
| | **XGBoost** | `XGBoost_FlattenedWindow` | `Tier3_Extended24` | 15.8682 | 0.8520 | 15.7873 | **29.7334** | **22.0147** | **6,939.94** | **PRIMARY TABULAR** |
| | **LSTM** | `Stacked_LSTM` | `Tier3_Extended24` | **15.2748** | **0.8629** | **15.4419** | 30.3794 | 22.6096 | 9,274.51 | **SPECIALIZED SEQUENCE** |

### 4.2 Cross-Architecture Insights & Model Selection Recommendations
The proposed decision matrix represents an **academic model-selection recommendation tailored strictly to the C-MAPSS simulation benchmark environment**, rather than a real-aircraft flight-certified deployment or automated operational routing engine:
1. **Single-Condition Regimes (`FD001`, `FD003`):**
   - **XGBoost is the unambiguous primary model.**
   - Summary statistics features ($5 \times K$) condense the temporal trajectory effectively, allowing gradient boosted trees to achieve the lowest validation RMSE, lowest test RMSE, lowest test MAE, and lowest NASA score.
   - Deep sequential recurrence adds significant latency without empirical accuracy gains on single-condition profiles.
2. **Multi-Condition Regimes (`FD002`, `FD004`):**
   - **Nuanced Tradeoff Between Validation Tracking and Test Score Generalization:**
     - `Stacked_LSTM` achieves the lowest **validation capped RMSE** on both FD002 (17.7875 vs. XGBoost 19.3303) and FD004 (15.2748 vs. XGBoost 15.8682), demonstrating that recurrent gating models operational shifts effectively across continuous validation trajectories.
     - On the official test partition, `XGBoost_FlattenedWindow` achieves lower official test NASA scores on both subsets (FD002: 14,159.01 vs 17,346.86; FD004: 6,939.94 vs 9,274.51) and lower test RMSE on FD004 (29.7334 vs 30.3794).
   - *Selection Recommendation:* **XGBoost remains the primary general-purpose benchmark**; **LSTM is retained as a specialized deep sequential model** for continuous multi-condition trajectory research.

---

## 5. Evaluation Reliability & Methodological Limitations

Step 4.4 explicitly documents the following methodological constraints:

1. **Target Capping Asymmetry:**
   - Models were trained and validated on targets capped at 125 cycles ($y \le 125$), but official test sets evaluate against uncapped ground truth. Because models are trained with a ceiling of 125, test predictions rarely exceed $\sim 130$ cycles, bounding predictions for engines tested very early in life ($RUL > 125$). This is an inherent convention of the NASA C-MAPSS benchmark.
2. **Validation (Continuous Sequence) vs. Test (Single Final Cycle):**
   - Validation metrics evaluate all sliding windows across the full operating life of 142 engines.
   - Official test metrics evaluate exactly **one window per engine** at an arbitrary truncation point.
   - Consequently, validation NASA scores aggregate thousands of windows, whereas test NASA scores aggregate exactly $N_{\text{test}}$ predictions ($100, 259, 100, 248$). Numerical magnitudes are therefore not directly comparable between validation and test.
3. **NASA Score Asymmetric Sensitivity:**
   - Because late predictions ($d > 0$) incur an exponential penalty ($\exp(d/10)-1$), a few late predictions can dominate the total penalty score. RMSE is robust to this asymmetry, while NASA score highlights safety risk.
4. **Checkpoint and Prediction Persistence:**
   - Model binary files (`.pkl` and `.pt`) and window-level prediction arrays were not saved to disk. All evaluation metrics are consolidated directly from the verified execution manifests and CSV tables.

---

## 6. Proposed Deliverables for Step 4.4

Step 4.4 will produce the following non-retraining synthesis artifacts:

### 6.1 Code & Pipeline Script
- **`models/evaluation/consolidate_model_evaluations.py`**
  - Loads and merges the 108 existing experiment records from `reports/models/baseline_model_performance.csv`, `reports/models/xgboost_model_performance.csv`, and `reports/models/lstm_model_performance.csv`.
  - Performs cross-model ranking, percentage improvement calculations over naive baselines, and safety metric reconciliations.
  - Reuses existing approved data; **does not execute training loops**.

### 6.2 Data Artifacts
- **`reports/models/master_model_comparison.csv`**
  - Complete, unified tabular registry containing all 108 experimental configurations with standardized column headers, model family tags, tier classifications, and decoupled metrics.
- **`reports/models/model_selection_summary.csv`**
  - Curated summary table detailing the selected champion and runner-up for each of the four C-MAPSS subsets, highlighting primary validation selection metrics and secondary test benchmarks.

### 6.3 Visualization Artifacts (Publication Quality)
- **`reports/figures/model_comparison_val_test_rmse.png`**
  - Grouped bar chart comparing Validation Capped RMSE vs. Official Test RMSE across Mean, Ridge, Random Forest, XGBoost, and LSTM for each subset.
- **`reports/figures/nasa_score_tradeoff_comparison.png`**
  - Comparative log-scale chart illustrating the NASA asymmetric penalty tradeoff across models.
- **`reports/figures/per_engine_rmse_distribution.png`**
  - Box plot showing per-engine error spread ($\mu \pm \sigma$) across validation engines.

### 6.4 Documentation Deliverable
- **`docs/phase4_step4_4_model_evaluation_protocol.md`**
  - Definitive Phase 4 evaluation report establishing the formal model comparison, C-MAPSS simulation recommendations, operational risk profiles, and preparation for Phase 5.

---

## 7. Direct Answers to Specification Questions

### Q1: Which evaluation metrics are already implemented?
- **RMSE:** Implemented across all partitions (Validation Capped, Validation Uncapped, Per-Engine, Test Full, Test Unpadded, Test Padded).
- **MAE:** Implemented across validation and test partitions.
- **$R^2$:** Implemented with zero-variance protection.
- **NASA Asymmetric Score:** Implemented using official piecewise exponential formulation.
- **Per-Engine Aggregates:** Mean and standard deviation of validation engine RMSE across 142 engines.

### Q2: Are all models evaluated consistently?
- **Yes, with one documented numerical implementation limitation:**
  - All 108 models across Steps 4.1, 4.2, and 4.3 were evaluated using identical engine-level split assignments (80/20, 208/52, 80/20, 199/50).
  - Identical target capping threshold ($RUL \le 125$) during training and validation model selection.
  - Identical unclipped continuous evaluation on the official test set.
  - Identical mathematical formulations for RMSE, MAE, $R^2$, and NASA asymmetric score.
  - *Documented Limitation:* Baseline and XGBoost scripts included defensive overflow clipping of error to $[-500, 500]$ before exponentiation, whereas LSTM omitted clipping. Because prediction extrema were not serialized to disk, exact numerical equivalence cannot be independently proven from persisted artifacts alone; historical aggregate scores are preserved without alteration.

### Q3: What evaluation gaps remain?
- **Consolidated Master Registry:** Results currently reside in three separate CSV files; there is no unified master ranking table.
- **Visual Synthesis:** No comparative multi-model visualizations (bar charts, distributions, trade-off curves) have been generated.
- **Simulation Decision Matrix:** A formal scorecard synthesizing when XGBoost vs. LSTM is recommended in the C-MAPSS simulation context has not been compiled.

### Q4: What should Step 4.4 add without duplicating earlier work?
- **Zero Retraining:** Step 4.4 will **not** train any new models or rerun existing configurations.
- **Consolidation & Synthesis:** Unifies the 108 existing runs into a single relational table.
- **Visual Analytics:** Generates multi-model performance charts (`reports/figures/`).
- **Academic Recommendation Matrix:** Formalizes the definitive architectural findings for C-MAPSS benchmark research.

### Q5: How will the final model comparison be presented?
- **Multi-Level Presentation:**
  1. *Executive Scorecard:* Best model per subset with percentage gain over naive mean and linear baselines.
  2. *Dual-Benchmark Matrix:* Primary Validation Capped RMSE (selection) alongside Official Test RMSE and NASA score (generalization).
  3. *Regime Analysis:* Separation between single-condition (FD001/FD003) and multi-condition (FD002/FD004) dynamics.
  4. *Visual Dashboards:* High-resolution comparison figures illustrating error distributions and asymmetric penalty behaviors.

---

## 8. Summary of Execution Plan

Upon user review and approval of this proposal:
1. Create `models/evaluation/consolidate_model_evaluations.py` to synthesize existing CSVs and generate master comparisons.
2. Produce `reports/models/master_model_comparison.csv` and `reports/models/model_selection_summary.csv`.
3. Generate comparison plots in `reports/figures/`.
4. Compile the comprehensive evaluation report `docs/phase4_step4_4_model_evaluation_protocol.md`.
5. Perform an independent numerical and methodological audit of Step 4.4 before concluding Phase 4.
