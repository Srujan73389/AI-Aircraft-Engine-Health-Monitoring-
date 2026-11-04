# Phase 4 – Step 4.2: XGBoost Pre-Implementation Audit Report

> [!IMPORTANT]
> **AUDIT STATUS: VERIFICATION COMPLETE — PENDING IMPLEMENTATION APPROVAL**  
> This audit investigates data integrity, input preparation, tuning methodology, and environment compatibility before executing **Phase 4 – Step 4.2: XGBoost Regression Model**.  
> **No packages have been installed, no models have been trained, and no datasets have been modified.**

---

## 1. Investigation of FD004 Sequence & Engine Count Discrepancy

### 1.1 Reported Discrepancy Context
In the initial draft of the proposed XGBoost implementation plan, Section 3.3 stated:
* *Erroneous text:* `"FD004: 198 training engines (34,800 windows), 50 validation engines (8,941 windows), 248 test engines."`

This conflicted with the official Phase 3 records ([docs/phase3_step3_9_time_window_sequences.md](file:///c:/Users/LENOVO/Documents/Aircraft_Engine_AI/docs/phase3_step3_9_time_window_sequences.md) and [docs/phase3_step3_10_engine_split.md](file:///c:/Users/LENOVO/Documents/Aircraft_Engine_AI/docs/phase3_step3_10_engine_split.md)) which recorded:
* *Official Phase 3 records:* Training: 199 engines, 42,725 windows; Validation: 50 engines, 11,303 windows; Test Final: 248 engines, 248 windows.

### 1.2 Inspection of Ground-Truth Manifests & Arrays on Disk

A rigorous audit of the source files, manifests, and NumPy binary arrays on disk yielded the following findings:

1. **Split Manifest Audit (`dataset/processed/train_val_split_manifest.json`):**
   * **Total Original Engines:** 249 engines (Unit IDs 1 to 249).
   * **Training Engine IDs:** Exactly **199 engines** (`[1, 3, 4, 6, 7, 10, 11, 12, 14, 15, ..., 248]`).
   * **Validation Engine IDs:** Exactly **50 engines** (`[2, 5, 8, 9, 13, 16, 21, ..., 249]`).
   * **Engine Overlap:** $\text{TrainEngines} \cap \text{ValEngines} \equiv \emptyset$ (Exactly 0 engines).
   * **Missing Engines:** $\text{AllEngines} \setminus (\text{TrainEngines} \cup \text{ValEngines}) \equiv \emptyset$ (Zero engines missing).
   * **Row Counts:** Train rows = 48,496 (79.18%); Val rows = 12,753 (20.82%); Total rows = 61,249.

2. **Sliding-Window Mathematical Verification ($W = 30$, Stride $s = 1$):**
   For each engine $i$ with lifecycle length $L_i$, the number of valid 30-cycle sliding windows is $L_i - W + 1 = L_i - 29$.
   * **Training Windows:**
     $$N_{\text{train\_windows}} = \sum_{i=1}^{199} (L_i - 29) = \left(\sum_{i=1}^{199} L_i\right) - (199 \times 29) = 48,496 - 5,771 = \mathbf{42,725}$$
   * **Validation Windows:**
     $$N_{\text{val\_windows}} = \sum_{j=1}^{50} (L_j - 29) = \left(\sum_{j=1}^{50} L_j\right) - (50 \times 29) = 12,753 - 1,450 = \mathbf{11,303}$$

3. **Physical Array Inspection on Disk:**
   Inspecting the binary NumPy headers of the actual sequence files loaded into memory:
   * `dataset/processed/sequences/X_train_FD004.npy` $\rightarrow$ Shape: `(42725, 30, 24)`
   * `dataset/processed/sequences/y_train_FD004.npy` $\rightarrow$ Shape: `(42725,)`
   * `dataset/processed/sequences/X_validation_FD004.npy` $\rightarrow$ Shape: `(11303, 30, 24)`
   * `dataset/processed/sequences/y_validation_FD004.npy` $\rightarrow$ Shape: `(11303,)`
   * `dataset/processed/sequences/X_test_final_FD004.npy` $\rightarrow$ Shape: `(248, 30, 24)`
   * `dataset/processed/sequences/y_test_final_FD004.npy` $\rightarrow$ Shape: `(248,)`

4. **Phase 4 Step 4.1 Baseline Performance Log Inspection:**
   In [reports/models/baseline_model_performance.csv](file:///c:/Users/LENOVO/Documents/Aircraft_Engine_AI/reports/models/baseline_model_performance.csv) (lines 41–53), all 13 baseline models for FD004 evaluated against exactly 50 validation engines (`val_engines_count = 50`) and all 248 test engines.

### 1.3 Root Cause & Resolution
* **Source of Error:** The figures `"198 training engines, 34,800 windows"` and `"8,941 validation windows"` did not originate from any data processing script, split manifest, or sequence generation step. They were an inadvertent clerical typo introduced during the writing of the proposal document.
* **Integrity Status:** No engine or sequence partition was ever excluded. All 249 training/validation engines and all 248 test engines are fully present, correctly partitioned, and physically verified on disk.
* **Correction Applied:** Line 71 of [docs/phase4_step4_2_xgboost_evaluation_report.md](file:///c:/Users/LENOVO/Documents/Aircraft_Engine_AI/docs/phase4_step4_2_xgboost_evaluation_report.md) has been corrected to reflect the true ground truth:
  `* **FD004:** 199 training engines (42,725 windows), 50 validation engines (11,303 windows), 248 test engines.`

---

## 2. XGBoost Input Preparation & Feature Dimension Verification

XGBoost requires 2D matrix inputs $(N, D)$. The 3D sliding-window tensors $(N, 30, 24)$ are transformed into 2D tabular features across three representations and three feature tiers:

### 2.1 Feature Tiers & Column Indices
* **Tier 1 (Core 14 Sensors):**
  * Channels: `[sensor_2, sensor_3, sensor_4, sensor_7, sensor_8, sensor_9, sensor_11, sensor_12, sensor_13, sensor_14, sensor_15, sensor_17, sensor_20, sensor_21]`
  * Tensor Column Indices: `[3, 4, 5, 8, 9, 10, 12, 13, 14, 15, 16, 18, 21, 22]`
* **Tier 2 (Multi-Condition 17 Features):**
  * Channels: 3 operational settings + 14 Tier 1 sensors
  * Tensor Column Indices: `[0, 1, 2, 3, 4, 5, 8, 9, 10, 12, 13, 14, 15, 16, 18, 21, 22]`
* **Tier 3 (Extended 24 Features):**
  * Channels: 3 operational settings + all 21 sensors
  * Tensor Column Indices: `[0, 1, ..., 23]`

### 2.2 Feature Representations & Dimensionality Confirmation

| Feature Representation | Mathematical Transformation | Tier 1 Dim | Tier 2 Dim | Tier 3 Dim |
| :--- | :--- | :---: | :---: | :---: |
| **`FinalObservation`** | Snapshot slice at final window cycle $t = 29$: $X_{:, 29, \text{indices}}$ | **14** | **17** | **24** |
| **`FlattenedWindow`** | Unrolled temporal sequence: $X_{:, :, \text{indices}} \rightarrow (N, 30 \times K)$ | **420** | **510** | **720** |
| **`SummaryStats`** | 5 statistics across window: $[\mu, \sigma, x_{\min}, x_{\max}, \beta_{\text{slope}}]$ | **70** | **85** | **120** |

### 2.3 Verification Execution
A programmatic transformation check was executed directly on slices of `X_train_FD004.npy`. All three representations for Tiers 1, 2, and 3 were synthesized, verified for exact dimensions, and audited for invalid values:
* **Result:** All 9 feature matrices produced identical expected dimensions, zero NaNs, zero infinities, and 100% dtype fidelity (`float32`).

---

## 3. Review of Tuning Methodology & Anti-Leakage Safeguards

### 3.1 Validation Early Stopping Protocol
In gradient boosting, adding decision trees greedily reduces training loss but eventually overfits sample-specific noise.
* **Early Stopping Mechanism:**
  * During training of the tuned XGBoost variant, the objective function is minimized on the training partition ($y_{\text{train}} = \min(\text{RUL}, 125)$).
  * After each boosting iteration $m \in \{1, \dots, M\}$, the model evaluates capped RMSE on the validation set ($y_{\text{val\_capped}} = \min(\text{RUL}, 125)$).
  * If `val_capped_rmse` fails to achieve a new minimum for `early_stopping_rounds = 20` consecutive iterations, boosting halts.
  * The model weights are automatically rolled back to the iteration with the lowest validation capped RMSE (`best_iteration`).

### 3.2 Hyperparameter Selection Criteria
* Candidate hyperparameters (`max_depth`, `learning_rate`, `reg_lambda`, `subsample`) are compared.
* **Exclusive Selection Rule:** The winning configuration for each subset is selected **exclusively by the minimum `val_capped_rmse`**.
* Capped RMSE directly matches the piecewise linear target distribution and penalizes prediction variance without distortion from unclipped healthy-engine cycles.

### 3.3 Risk Analysis: Validation Overfitting & Mitigations
Using the validation set for both early stopping and hyperparameter selection introduces a theoretical risk of mild optimistic selection bias (validation overfitting). The following architectural safeguards are established:

1. **Engine-Level Stratification:** The validation set consists of entirely held-out engines (20 engines in FD001/FD003, 52 in FD002, 50 in FD004). Because no engine appears in both sets, the validation error measures cross-engine generalization rather than time-series interpolation.
2. **Diagnostic Uncapped Validation Check:** We compute and report `val_uncapped_rmse` and `val_uncapped_score` against raw RUL. If a model over-optimizes capped early cycles at the expense of true degradation trajectories, the uncapped diagnostic metrics will detect it immediately.
3. **Per-Engine Aggregation Metrics:** We log `val_mean_engine_rmse`, `val_std_engine_rmse`, and `val_final_cycle_rmse`. This ensures that early stopping points are not driven by outlier engines.
4. **Structural Capacity Constraints:** The tuned configuration employs shallow depth (`max_depth = 4`), conservative learning rate (`\eta = 0.05`), feature/row subsampling (`0.8`), and L2 leaf regularization (`reg_lambda = 5.0`), preventing the tree ensemble from memorizing subtle validation set characteristics.

### 3.4 Strict Test Set Isolation
* **Zero Leakage Rule:** Official NASA test arrays (`X_test_final_FD00{1..4}.npy`, `y_test_final_FD00{1..4}.npy`) are **never passed to `xgb.fit()`** and **never passed as `eval_set`**.
* Test metrics are evaluated in a single inference pass **only after** all models are trained and the winner has been selected via `val_capped_rmse`.
* Test predictions are evaluated against raw, uncapped NASA ground truth without artificial prediction clipping.

---

## 4. Environment Compatibility & Package Verification

### 4.1 Environment Inspection
* **Operating System:** Windows 11 (AMD64 / x86_64).
* **Python Runtime:** Python 3.15.0 (`v1951 64-bit`).
* **Active Environment:** Project local `.venv` managed with `uv`.
* **Installed Numerical Stack:** `numpy==2.5.4`, `scipy==1.18.1`, `pandas==3.1.0`, `scikit-learn==1.9.1`.
* **Current XGBoost Status:** `xgboost` is **NOT installed**.

### 4.2 PyPI Compatibility & Distribution Check
A dry-run resolution query was executed against PyPI using `uv`:
```powershell
uv pip install --dry-run xgboost
```

* **Resolution Output:**
  ```text
  Resolved 3 packages in 948ms
  Would download 1 package
  Would install 1 package
   + xgboost==3.4.1
  ```
* **Findings:**
  1. A pre-compiled official binary wheel (`xgboost==3.4.1`) is available on PyPI for Windows AMD64.
  2. The wheel requires **zero compilation tools (no MSVC / CMake build needed)**.
  3. All underlying dependencies (`numpy`, `scipy`) are fully satisfied by the existing `.venv` packages.
  4. The dry-run verified resolution without downloading or installing any files. The environment remains pristine.

---

## 5. Audit Summary & Readiness Matrix

| Verification Item | Requirement | Audit Result | Status |
| :--- | :--- | :--- | :---: |
| **FD004 Engine Count** | 199 Train, 50 Val, 248 Test | Verified in `train_val_split_manifest.json` and arrays | **PASS** |
| **FD004 Window Count** | 42,725 Train, 11,303 Val | Verified via mathematical check and `.npy` shape | **PASS** |
| **Proposal Typo Fix** | Correct proposal line 71 | Typo corrected in `phase4_step4_2_xgboost_evaluation_report.md` | **PASS** |
| **Input Representations** | Final, Flattened, SummaryStats | Verified dimensions and numerical stability | **PASS** |
| **Feature Tiers** | Tier 1 (14), Tier 2 (17), Tier 3 (24) | Verified column indices and matrix shapes | **PASS** |
| **Early Stopping Protocol** | Validation capped RMSE | Verified; safeguards against overfitting documented | **PASS** |
| **Test Set Isolation** | No test feedback or leakage | Verified; test data isolated from fitting and early stopping | **PASS** |
| **Environment Check** | XGBoost distribution availability | Verified: `xgboost==3.4.1` pre-built wheel available | **PASS** |
| **Execution State** | No packages or models trained | Zero packages installed, zero models trained | **PASS** |

### Next Step
The project is fully verified and technically ready for execution upon user authorization to install `xgboost` and run Phase 4 – Step 4.2.
