# Phase 4 – Step 4.3: Methodological and Numerical Audit (LSTM)

**Project:** AI-Based Aircraft Engine Health Monitoring, Fault Diagnosis and Predictive Maintenance  
**Audit Target:** Phase 4 – Step 4.3 LSTM Sequence Model Implementation & Evaluation  
**Auditor:** Antigravity AI (Pair Programming Assistant)  
**Date & Timestamp:** 2026-11-04T13:10:00+05:30  
**Status:** **AUDIT COMPLETED**  

---

## 1. Audit Framework & Objectives

This audit provides an independent, objective evaluation of the implementation, data protection, metrics, benchmark comparisons, validation aggregations, and reproducibility of the Phase 4 – Step 4.3 LSTM experiments.

The audit evaluates six specific operational domains:
1. **Experiment Completeness:** All 16 configurations across FD001–FD004 completed with verified winning selections.
2. **Data & Leakage Protection:** Separation of splits, target capping rules, isolation of official test data, and unclipped predictions.
3. **Metric Verification:** Verification of mathematical formulas and alignment with benchmark tables.
4. **Head-to-Head Comparison:** Strict evaluation against the approved Step 4.2 XGBoost winners using approved criteria.
5. **Validation Aggregation Reconciliation:** Numerical reconciliation of the 568 per-engine validation records with the master performance table.
6. **Execution Reproducibility:** Completeness of manifests, hyperparameters, training loss curves, and persistence limitations.

Each audit item is explicitly assigned a status of **PASS**, **FAIL**, or **LIMITATION**.

---

## 2. Audit Findings by Category

### Audit 1: Experiment Completeness
* **Scope Requirement:** Exactly 16 configurations (4 subsets $\times$ [3 `Stacked_LSTM` tiers + 1 `Single_LSTM` tier]).
* **Verification Results:**
  - `reports/models/lstm_model_performance.csv` contains exactly **16 rows** with zero missing fields.
  - `models/lstm/lstm_manifest.json` contains exactly **16 completed execution blocks**.
  - All four subsets (FD001, FD002, FD003, FD004) are fully represented with 4 runs each:
    - `Stacked_LSTM` | `Tier1_Core14`
    - `Stacked_LSTM` | `Tier2_Operational17`
    - `Stacked_LSTM` | `Tier3_Extended24`
    - `Single_LSTM`  | `Tier3_Extended24`
  - Zero crashes, zero unhandled exceptions, and zero interrupted runs occurred during execution (total CPU runtime: 7,405.4 seconds).
  - Selected winning configuration for each subset corresponds strictly to the configuration with the **lowest validation capped RMSE**:
    - **FD001:** `Stacked_LSTM` (`Tier1_Core14`) $\rightarrow$ Val Capped RMSE: **13.1011** (Lowest of 13.1011, 13.9182, 41.9235, 13.1490).
    - **FD002:** `Stacked_LSTM` (`Tier3_Extended24`) $\rightarrow$ Val Capped RMSE: **17.7875** (Lowest of 41.8750, 41.8775, 17.7875, 21.1948).
    - **FD003:** `Single_LSTM` (`Tier3_Extended24`) $\rightarrow$ Val Capped RMSE: **14.1079** (Lowest of 41.8157, 41.8625, 41.9032, 14.1079).
    - **FD004:** `Stacked_LSTM` (`Tier3_Extended24`) $\rightarrow$ Val Capped RMSE: **15.2748** (Lowest of 41.2965, 16.1822, 15.2748, 18.7436).
* **Audit Status:** **PASS**

---

### Audit 2: Data & Leakage Protection
* **Engine-Level Separation:**
  - Training and validation sets are partitioned strictly at the engine level according to `dataset/processed/train_val_split_manifest.json`.
  - Engine counts verified:
    - FD001: 80 Train / 20 Validation
    - FD002: 208 Train / 52 Validation
    - FD003: 80 Train / 20 Validation
    - FD004: 199 Train / 50 Validation
  - Total validation engines across all subsets: exactly **142 engines**. Zero engine overlap exists between train and validation.
* **Target Capping Protocol:**
  - Training loss and early-stopping validation target arrays were explicitly capped at $RUL \le 125$ (`y_train_capped = np.minimum(y_train_raw, 125)` and `y_val_capped = np.minimum(y_val_raw, 125)`).
* **Official Test Data Isolation:**
  - Official test tensors (`X_test_final_{subset}.npy`) were never passed to `train_lstm_model()`, optimizer step, or early-stopping routines.
  - Early stopping was governed exclusively by `val_capped_rmse`.
* **Zero Clipping Rule:**
  - Model predictions on test data were continuous outputs; no artificial clipping or bounding to $[0, 125]$ was applied during evaluation.
* **Test Evaluation Execution Scope (Methodological Finding):**
  - In `models/lstm/run_lstm_models.py`, `test_preds` was computed and logged for all 16 trained models in the batch runner, rather than isolating test evaluation solely to the 4 post-hoc locked winners.
  - *Risk Assessment:* Because test metrics were purely passive outputs logged after early stopping restored the best checkpoint, test data never influenced model parameters, hyperparameter adjustments, or model selection.
* **Audit Status:** **PASS (with noted LIMITATION regarding all-config test logging)**

---

### Audit 3: Metric Verification
* **Mathematical Formula Verification:**
  - **RMSE:** $\sqrt{\frac{1}{N}\sum_{i=1}^N (\hat{y}_i - y_i)^2}$ — correctly implemented via `np.sqrt(np.mean((y_pred - y_true)**2))`.
  - **MAE:** $\frac{1}{N}\sum_{i=1}^N |\hat{y}_i - y_i|$ — correctly implemented via `np.mean(np.abs(y_pred - y_true))`.
  - **$R^2$ Score:** $1 - \frac{\sum (\hat{y}_i - y_i)^2}{\sum (y_i - \bar{y})^2}$ — correctly implemented with zero-variance protection.
  - **Official NASA Asymmetric Scoring Function:**
    $$d_i = \hat{y}_i - y_i, \quad s = \sum_{i=1}^N \begin{cases} e^{-d_i/13} - 1, & d_i < 0 \\ e^{d_i/10} - 1, & d_i \ge 0 \end{cases}$$
    Correctly implemented in `compute_nasa_score()` using vectorized NumPy operations.
* **Decoupled Metric Reporting:**
  - In `reports/models/lstm_model_performance.csv`, validation capped metrics, validation uncapped diagnostics, and official test metrics are maintained in separate dedicated columns.
* **Audit Status:** **PASS**

---

### Audit 4: Head-to-Head Comparison with Approved Step 4.2 XGBoost Benchmarks

All comparisons use the approved Step 4.2 XGBoost reference results from `reports/models/xgboost_model_performance.csv` and `docs/phase4_step4_2_xgboost_evaluation_report.md`:

| Subset | Approved XGBoost Winner | XGBoost Val Capped RMSE | Selected LSTM Winner | LSTM Val Capped RMSE | $\Delta$ Val RMSE (LSTM vs XGB) | Validation Winner | XGBoost Test RMSE | LSTM Test RMSE | XGBoost Test Score | LSTM Test Score | Comprehensive Subset Winner |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FD001** | `XGBoost_Tuned_SummaryStats` (Tier 1) | **12.7506** | `Stacked_LSTM` (Tier 1) | 13.1011 | +0.3505 | **XGBoost** | **13.4770** | 14.8409 | **268.06** | 365.94 | **XGBoost** (wins validation & all test metrics) |
| **FD002** | `XGBoost_FlattenedWindow` (Tier 3) | 19.3303 | `Stacked_LSTM` (Tier 3) | **17.7875** | **-1.5428** | **LSTM** | 30.0483 | **29.0728** | **14,159.01** | 17,346.86 | **Split / Contextual** (LSTM wins Val & Test RMSE; XGBoost wins NASA Score) |
| **FD003** | `XGBoost_Tuned_SummaryStats` (Tier 3) | **12.7926** | `Single_LSTM` (Tier 3) | 14.1079 | +1.3153 | **XGBoost** | **14.2476** | 17.4228 | **312.21** | 1,242.62 | **XGBoost** (wins validation & all test metrics) |
| **FD004** | `XGBoost_FlattenedWindow` (Tier 3) | 15.8682 | `Stacked_LSTM` (Tier 3) | **15.2748** | **-0.5934** | **LSTM** | **29.7334** | 30.3794 | **6,939.94** | 9,274.51 | **Split / Contextual** (LSTM wins Val RMSE; XGBoost wins Test RMSE & NASA Score) |

#### Explicit Methodological Findings by Subset:
1. **FD001:** **XGBoost wins validation RMSE and all reported test metrics.**  
   XGBoost achieves 12.7506 validation RMSE vs. LSTM's 13.1011, and 13.4770 test RMSE vs. LSTM's 14.8409. The tabular gradient boosted model is strictly superior on this single-condition regime.
2. **FD002:** **LSTM wins validation RMSE and official test RMSE, but XGBoost wins official test NASA score.**  
   LSTM achieves a substantial validation improvement (17.7875 vs. 19.3303, a $-1.5428$ point / $+8.0\%$ reduction in error) and lowers test RMSE from 30.0483 to 29.0728. However, because LSTM produced a small number of late predictions ($d > 0$) that were heavily penalized by the asymmetric exponential penalty, XGBoost maintains a lower NASA score (14,159.01 vs. 17,346.86).
3. **FD003:** **XGBoost wins validation RMSE and all reported test metrics.**  
   XGBoost achieves 12.7926 validation RMSE vs. LSTM's 14.1079, and 14.2476 test RMSE vs. LSTM's 17.4228. `Stacked_LSTM` suffered from optimization stagnation on FD003, and while `Single_LSTM` converged, it did not match XGBoost's performance.
4. **FD004:** **LSTM wins validation RMSE, but XGBoost wins official test RMSE and official test NASA score.**  
   LSTM improves validation capped RMSE from 15.8682 to 15.2748 (a $-0.5934$ point / $+3.7\%$ error reduction). However, on the official test set, XGBoost generalizes slightly better (test RMSE 29.7334 vs. 30.3794; test score 6,939.94 vs. 9,274.51).
5. **Roadmap Realignment & Retention Directive:**  
   **LSTM is NOT universally superior across multi-condition subsets.** While LSTM delivers meaningful validation gains on multi-condition operational regimes (FD002 and FD004), its test score dynamics are mixed due to exponential penalties on over-estimations. Therefore:
   - **XGBoost remains the primary production benchmark.**
   - **LSTM is retained as a specialized deep sequential model for multi-condition degradation analysis.**

* **Audit Status:** **PASS**

---

### Audit 5: Validation Aggregation Reconciliation
* **Unit Record Counts:**
  - FD001: 20 engines $\times$ 4 configs = 80 records.
  - FD002: 52 engines $\times$ 4 configs = 208 records.
  - FD003: 20 engines $\times$ 4 configs = 80 records.
  - FD004: 50 engines $\times$ 4 configs = 200 records.
  - Total records in `reports/models/lstm_per_engine_validation.csv`: **568 records**.
* **Aggregate Reconciliation Check:**
  - For every configuration, the mean of `per_engine_rmse_capped` across validation engines matches `val_mean_engine_rmse` in `reports/models/lstm_model_performance.csv` with a maximum deviation of $< 0.0001$.
  - The population standard deviation of `per_engine_rmse_capped` matches `val_std_engine_rmse` with a maximum deviation of $< 0.0001$.
  - Discrepancies across all 16 configurations: **0 discrepancies**.
* **Audit Status:** **PASS**

---

### Audit 6: Execution Reproducibility & Model Artifacts
* **Manifest Inspection (`models/lstm/lstm_manifest.json`):**
  - 16 JSON configuration objects corresponding to the 16 runs.
  - Contains exact architecture specifications (`model_class`, `input_dim`, `hidden_dim`, `dropout`).
  - Contains exact training hyperparameters (`batch_size=128`, `max_epochs=60`, `patience=12`, `lr=0.001`, `weight_decay=0.0001`, `ReduceLROnPlateau`).
  - Contains full per-epoch loss history (`train_loss`, `val_capped_rmse`, `lr` schedule) and `best_epoch` tracking.
  - Fixed global random seed: `42`.
* **Identified Reproducibility Limitations:**
  - **Trained Model Weights:** PyTorch model binary checkpoints (`.pt` or `.pth`) were not serialized to disk to prevent repository bloat, consistent with the baselines and XGBoost implementations. Models must be retrained to reproduce exact weights.
  - **Inference Predictions:** Raw continuous prediction vectors ($\hat{y}_{\text{val}}$ and $\hat{y}_{\text{test}}$) were evaluated in memory and aggregated into CSV reports, but raw 1D NumPy prediction arrays were not persisted to disk.
  - **CPU Multi-threading Determinism:** While seeds were fixed (`torch.manual_seed(42)` and `np.random.seed(42)`), slight cross-CPU floating-point accumulation variations may occur across different CPU microarchitectures unless `torch.use_deterministic_algorithms(True)` is enforced.
* **Audit Status:** **PASS (with noted LIMITATIONS)**

---

## 3. Comprehensive Audit Summary Table

| Audit Domain | Scope / Criterion | Verified Values | Discrepancies / Anomalies | Audit Status |
| :--- | :--- | :--- | :--- | :---: |
| **1. Completeness** | Exactly 16 runs (4 subsets $\times$ 4 configs) | 16/16 completed, 0 failed runs | None | **PASS** |
| **2. Leakage Protection** | Split separation, target capping, test isolation | $y \le 125$ applied, 142 val engines isolated | None (Test logged across all configs) | **PASS** |
| **3. Metric Calculations** | RMSE, MAE, $R^2$, NASA score formulas | Decoupled metrics verified | None | **PASS** |
| **4. Benchmark Comparison** | Head-to-head comparison vs. Step 4.2 XGBoost | Exact reference metrics reconciled | Nuanced multi-condition findings documented | **PASS** |
| **5. Aggregation Reconcile** | 568 per-engine records match master table | 16/16 configs reconcile ($\Delta < 10^{-4}$) | None | **PASS** |
| **6. Reproducibility** | Hyperparameters, seeds, training curves in JSON | Complete JSON manifest present | Weights and raw prediction arrays not saved | **PASS (w/ Limitations)** |

---

## 4. Required Report Adjustments

Based on this audit, [`docs/phase4_step4_3_lstm_execution_summary.md`](file:///c:/Users/LENOVO/Documents/Aircraft_Engine_AI/docs/phase4_step4_3_lstm_execution_summary.md) is largely methodologically sound, but should incorporate the following precise clarifications:
1. **Clarify Multi-Condition Verdict:** Avoid characterizing LSTM as globally superior on multi-condition subsets without qualification. Explicitly note that while LSTM wins validation RMSE on both FD002 and FD004 (and test RMSE on FD002), XGBoost achieves lower test NASA scores on both FD002 and FD004 and a lower test RMSE on FD004.
2. **Document Persistence Limitations:** Explicitly note in the execution summary that PyTorch model binaries and per-window raw prediction arrays are not persisted on disk.
