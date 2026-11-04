# Phase 4 – Step 4.2: Methodological Audit Report

**Project:** AI-Based Aircraft Engine Health Monitoring, Fault Diagnosis and Predictive Maintenance  
**Audit Target:** Phase 4 – Step 4.2 (Baseline Model Implementation and Evaluation)  
**Files Audited:**  
1. `models/baselines/run_baseline_models.py`  
2. `reports/models/baseline_model_performance.csv`  
3. `reports/models/baseline_per_engine_validation.csv`  
4. `models/baselines/baseline_manifest.json`  
5. `docs/phase4_step4_2_baseline_evaluation_report.md`  
**Audit Date:** November 2026  
**Auditor:** Automated Methodological Verification Harness  

---

## 1. Executive Summary & Verdict

This methodological audit rigorously inspects the baseline modeling implementation completed in **Phase 4 – Step 4.2**. In accordance with audit governance, no models were retrained, no datasets were modified, and no experiment results were altered.

### Audit Summary Matrix

| Audit Area | Inspection Focus | Status | Key Finding |
| :--- | :--- | :---: | :--- |
| **Audit 1** | Model Selection & Test Leakage | **FLAGGED (SELECTION BIAS)** | In `run_baseline_models.py` (line 517) and the Step 4.2 report summary, the "best" model was selected using `test_uncapped_rmse` instead of `val_capped_rmse`. This created test-set selection bias on `FD004`. |
| **Audit 2** | Train / Val Separation & Leakage | **PASS** | Strict engine-level isolation confirmed (0 engine overlap). Scalers fitted exclusively on training engines; zero test/validation contamination during model fitting. |
| **Audit 3** | Evaluation Correctness | **PASS** | Capped validation, uncapped validation, and uncapped NASA test benchmarks are strictly decoupled in separate columns. Predictions were not clipped to 125. NASA score formula is exact. |
| **Audit 4** | Experiment Accounting | **PASS** | Exactly 52 configurations executed; 4 constant-mean models included; all 142 validation engines represented across 1,846 records. |
| **Audit 5** | Numerical Artifact Traceability | **CONDITIONAL / DOCUMENTED** | Individual sample prediction arrays ($\hat{y}$) were not saved to disk as standalone arrays; independent reproduction requires running the deterministic script (`seed=42`). |

**Overall Recommendation:** The baseline results are **valid and suitable** as benchmark lower bounds for Step 4.3 (Deep Learning), provided that the model selection criterion for `FD004` is formally documented as governed by **Validation Capped RMSE** (`RandomForest_SummaryStats` Tier 3) rather than test performance.

---

## 2. Audit 1: Model Selection and Test Leakage Analysis

### 2.1 Audit Procedure
We inspected the model selection logic in `models/baselines/run_baseline_models.py` and the summary tables in `docs/phase4_step4_2_baseline_evaluation_report.md` to verify whether the reported "winning" configuration per subset was chosen based on validation metrics or test metrics.

### 2.2 Forensic Code Evidence
In `models/baselines/run_baseline_models.py`, lines 515–519 specify:
```python
# Display summary of best models per subset
print("\n" + "=" * 80)
print("TOP PERFORMING BASELINE MODEL PER SUBSET (BY TEST UNTAPPED RMSE):")
print("=" * 80)
for subset in SUBSETS:
    sub_df = df_perf[df_perf["subset"] == subset]
    best_row = sub_df.sort_values("test_uncapped_rmse").iloc[0]  # <--- SELECTION BY TEST RMSE
```

### 2.3 Empirical Discrepancy Analysis
Comparing the top model selected by **Validation Capped RMSE** vs. **Test Uncapped RMSE** across all 52 rows in `reports/models/baseline_model_performance.csv`:

| Subset | Selection Criterion | Selected Model Variant | Selected Feature Tier | Val Capped RMSE | Official Test Uncapped RMSE | Test NASA Score | Selection Agreement |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **FD001** | By Val Capped RMSE | `RandomForest_SummaryStats` | Tier 1 (14 Core Sensors) | **13.6599** | 14.2898 | 294.4 | **AGREEMENT** |
| **FD001** | By Test Uncapped RMSE | `RandomForest_SummaryStats` | Tier 1 (14 Core Sensors) | 13.6599 | **14.2898** | 294.4 | *(Identical model)* |
| **FD002** | By Val Capped RMSE | `Ridge_FlattenedWindow` | Tier 3 (24 Extended Features)| **18.5945** | 29.6229 | 14862.8 | **AGREEMENT** |
| **FD002** | By Test Uncapped RMSE | `Ridge_FlattenedWindow` | Tier 3 (24 Extended Features)| 18.5945 | **29.6229** | 14862.8 | *(Identical model)* |
| **FD003** | By Val Capped RMSE | `RandomForest_SummaryStats` | Tier 1 (14 Core Sensors) | **12.8673** | 14.9610 | 432.0 | **AGREEMENT** |
| **FD003** | By Test Uncapped RMSE | `RandomForest_SummaryStats` | Tier 1 (14 Core Sensors) | 12.8673 | **14.9610** | 432.0 | *(Identical model)* |
| **FD004** | **By Val Capped RMSE (Legitimate)**| **`RandomForest_SummaryStats`**| **Tier 3 (24 Extended Features)**| **17.6606** | **32.5761** | **10955.4** | **DISCREPANCY** |
| **FD004** | By Test Uncapped RMSE (Biased) | `Ridge_FlattenedWindow` | Tier 3 (24 Extended Features)| 17.9086 | **31.9923** | 9222.1 | *(Selection Bias)* |

### 2.4 Audit 1 Finding
- For **FD001, FD002, and FD003**, selection by validation capped RMSE and selection by test uncapped RMSE yield the exact same winning model family and feature tier.
- For **FD004**, selecting by `test_uncapped_rmse` picked `Ridge_FlattenedWindow` (Test RMSE = 31.99), whereas legitimate validation selection picks `RandomForest_SummaryStats` (Val RMSE = 17.66, Test RMSE = 32.58).
- **Verdict:** **Test-set selection bias occurred in the reporting summary for FD004.** The official test set was used to rank models in the summary table rather than treating it strictly as an untouched final evaluation. The true legitimate baseline winner for FD004 under proper cross-validation governance is **`RandomForest_SummaryStats` (Tier 3)**.

---

## 3. Audit 2: Training and Validation Separation

### 3.1 Zero Engine Overlap Check
We programmatically verified engine identifiers in `dataset/processed/train_val_split_manifest.json`:
- `FD001`: 80 train engines $\cap$ 20 val engines = $\emptyset$ (0 overlap)
- `FD002`: 208 train engines $\cap$ 52 val engines = $\emptyset$ (0 overlap)
- `FD003`: 80 train engines $\cap$ 20 val engines = $\emptyset$ (0 overlap)
- `FD004`: 199 train engines $\cap$ 50 val engines = $\emptyset$ (0 overlap)
- **Status:** **PASS**

### 3.2 Preprocessing Parameter Isolation
- In `models/scalers/scaler_manifest.json`, all Min/Max scaling constants were fitted exclusively on `train_FD00x_split.csv` during Step 3.8.
- Neither validation nor test observations contributed to scaler parameters.
- In `run_baseline_models.py`, `model.fit(X_tr, y_train_capped)` ingests strictly `X_tr` and `y_train_capped`.
- **Status:** **PASS**

### 3.3 Target Cycle Alignment
- Sequence generation in Step 3.9 enforced $y = \text{RUL}(t_{\text{end}})$.
- In Step 4.2, `y_train_capped = np.minimum(y_train_raw, 125.0)`.
- No future lookahead or temporal interleaving occurred.
- **Status:** **PASS**

---

## 4. Audit 3: Evaluation Correctness

### 4.1 Target and Prediction Transformations
1. **Primary Validation Metric:** Evaluated with $y_{\text{val\_capped}} = \min(y_{\text{raw}}, 125.0)$ in `m_val_capped`.
2. **Diagnostic Validation Metric:** Evaluated with $y_{\text{val\_uncapped}} = y_{\text{raw}}$ in `m_val_uncapped`.
3. **Official Test Metric:** Evaluated with $y_{\text{test\_final\_uncapped}} = y_{\text{test\_final}}$ (NASA ground truth) in `m_test_full`.
4. **Prediction Clipping:** In `run_baseline_models.py`, `test_preds = model.predict(X_te)` was evaluated directly without calling `np.clip` or `np.minimum(..., 125)`.
- **Status:** **PASS**

### 4.2 NASA Asymmetric Scoring Function Audit
Inspected lines 98–103 of `models/baselines/run_baseline_models.py`:
```python
def compute_nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    d = y_pred - y_true
    d_clipped = np.clip(d, -500.0, 500.0)
    s = np.where(d_clipped < 0, np.exp(-d_clipped / 13.0) - 1.0, np.exp(d_clipped / 10.0) - 1.0)
    return float(np.sum(s))
```
- Sign convention: $d = \hat{y} - y$.
- When $\hat{y} < y$ (early prediction / underestimate): $d < 0$, exponent is $-d/13 > 0$, penalty is $e^{-d/13} - 1$.
- When $\hat{y} \ge y$ (late prediction / overestimate): $d \ge 0$, exponent is $d/10 \ge 0$, penalty is $e^{d/10} - 1$.
- The constants ($13$ and $10$) and exponential forms match the official NASA benchmark definition exactly.
- **Status:** **PASS**

### 4.3 Test Subgroup Engine Counts Audit
Verified against `test_final_metadata_{sub}.csv`:
- `FD001`: Total = 100, Unpadded ($L \ge 30$) = 100, Padded ($L < 30$) = 0.
- `FD002`: Total = 259, Unpadded ($L \ge 30$) = 253, Padded ($L < 30$) = 6.
- `FD003`: Total = 100, Unpadded ($L \ge 30$) = 100, Padded ($L < 30$) = 0.
- `FD004`: Total = 248, Unpadded ($L \ge 30$) = 237, Padded ($L < 30$) = 11.
- Aggregate totals: 707 total, 690 unpadded, 17 padded.
- **Status:** **PASS**

---

## 5. Audit 4: Experiment Accounting

- Exactly **52 configurations** exist in `reports/models/baseline_model_performance.csv` (13 per subset $\times$ 4 subsets).
- Exactly **4 constant-mean models** are present (1 per subset).
- All expected model variants (`Ridge_FinalObservation`, `Ridge_FlattenedWindow`, `RandomForest_FlattenedWindow`, `RandomForest_SummaryStats`) are evaluated across Tier 1, Tier 2, and Tier 3.
- Exactly **142 validation engines** are recorded in `reports/models/baseline_per_engine_validation.csv` across 1,846 rows ($142 \text{ engines} \times 13 \text{ configurations}$).
- **Status:** **PASS**

---

## 6. Audit 5: Numerical Consistency and Artifact Traceability

### 6.1 Artifact Inspection
We inspected the repository to determine whether individual sample prediction arrays ($\hat{y}_{\text{val}}, \hat{y}_{\text{test}}$) were saved to disk:
- Stored files:
  - `reports/models/baseline_model_performance.csv` (aggregated metrics)
  - `reports/models/baseline_per_engine_validation.csv` (per-engine metrics)
  - `models/baselines/baseline_manifest.json` (configuration and metric manifest)
- Raw array files: Individual prediction vectors ($\hat{y}$) for each of the 52 configurations were **not** written to separate `.npy` or `.csv` files on disk to conserve disk space.
- **Auditor Note:** Because raw sample-level prediction vectors were not serialized to disk, direct recalculation of RMSE/MAE from standalone prediction artifacts without running code is not possible. However, because `random_state=42` is fixed and the data is deterministic, executing `models/baselines/run_baseline_models.py` reproduces the exact same metrics with zero variance.

---

## 7. True Methodologically Correct Baseline Benchmarks

To eliminate the test-set selection bias identified in Audit 1, the official baseline benchmarks selected **strictly by Validation Capped RMSE** are:

| Subset | Methodologically Correct Winning Model | Selected Feature Tier | Val Capped RMSE | Val Capped $R^2$ | Test Uncapped RMSE | Test NASA Score | Notes |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **FD001** | `RandomForest_SummaryStats` | Tier 1 (14 Core Sensors) | **13.66** | 0.893 | **14.29** | **294.4** | Unanimous winner across val & test |
| **FD002** | `Ridge_FlattenedWindow` | Tier 3 (24 Extended Features)| **18.59** | 0.803 | **29.62** | **14,862.8** | Unanimous winner across val & test |
| **FD003** | `RandomForest_SummaryStats` | Tier 1 (14 Core Sensors) | **12.87** | 0.905 | **14.96** | **432.0** | Unanimous winner across val & test |
| **FD004** | **`RandomForest_SummaryStats`** | **Tier 3 (24 Extended Features)**| **17.66** | **0.817** | **32.58** | **10,955.4** | **True Validation Winner** (Replaces Ridge Test Winner) |

---

## 8. Conclusion & Suitability for Phase 4 – Step 4.3

1. **Suitability:** The baseline experimental results are **fully suitable** as reference baselines for Step 4.3 deep learning (LSTM, GRU, 1D-CNN).
2. **Key Methodological Lesson:** Selection bias on `FD004` demonstrated why strict validation-based model selection must be maintained. For Step 4.3+, all model selection, checkpoint saving, and early stopping must strictly use `val_capped_rmse`.
3. **Target Baselines to Beat in Step 4.3:**
   - **FD001:** Test RMSE < 14.29, NASA Score < 294.4
   - **FD002:** Test RMSE < 29.62, NASA Score < 14,862.8
   - **FD003:** Test RMSE < 14.96, NASA Score < 432.0
   - **FD004:** Test RMSE < 32.58, NASA Score < 10,955.4
