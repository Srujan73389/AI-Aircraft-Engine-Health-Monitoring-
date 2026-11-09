# Phase 9 — Step 9.3: Remaining Useful Life (RUL) Ground-Truth Comparison & Prognostics Audit

**Document Reference:** `docs/phase9_step9_3_rul_ground_truth_comparison.md`  
**Master Roadmap Reference:** Section 10, Step 9.3 (*"Compare RUL predictions with the corresponding ground-truth labels"*) & Section 1, Page 2 (*"Do not invent sensor measurements or report fabricated model scores"*).  
**Phase Status:** Phase 9 ("System testing and demonstration") — Step 9.3 Complete.

---

## 1. Executive Summary & Verification Scope

In Step 9.3, the prognostic inference capabilities of the Digital Twin platform are verified against the official NASA C-MAPSS ground-truth Remaining Useful Life labels (`RUL_FD001.txt` through `RUL_FD004.txt`). 

The primary objectives of this verification step are:
1. **Authentic Ground-Truth Ingestion:** Parse true end-of-trajectory life labels directly from `RUL_FD00*.txt` without synthetic manipulation or hardcoding.
2. **Trajectory Ground-Truth Formulation:** Compute exact intermediate ground-truth RUL across operational cycles using the fundamental linear degradation identity:
   $$y^*(t) = y^*_{\text{final}} + (T_i - t)$$
   where $T_i$ is the maximum recorded cycle for test engine $i$, and $t$ is the evaluated cycle.
3. **Dual-Model Inference Execution:** Run live champion models (XGBoost and PyTorch Bi-LSTM) on authentic 30-cycle normalized sequence tensors ($G_{31}$) constructed according to Phase 3 & Phase 5 specifications.
4. **Epistemic Uncertainty & Consensus Synthesis:** Compute the dual-model consensus prediction $\hat{y}_{\text{consensus}} = \frac{1}{2}(\hat{y}_{\text{XGB}} + \hat{y}_{\text{LSTM}})$, the epistemic spread $\Delta = |\hat{y}_{\text{XGB}} - \hat{y}_{\text{LSTM}}|$, and model confidence scores.
5. **Multi-Subset Representative Evaluation:** Audit predictions across representative test engines for all four NASA C-MAPSS subsets (`FD001` through `FD004`).
6. **Zero Data Leakage & Immutability:** Verify that ground-truth labels remain strictly isolated from model inputs, scalers, and checkpoints, with 100% cryptographic SHA-256 preservation.

---

## 2. Mathematical Formulation

### 2.1 Trajectory Ground-Truth Function
For any held-out test engine $i$ with truncated flight history $[1, T_i]$:
- Let $y^*_{\text{final}}$ denote the true remaining useful life at final cycle $T_i$ from `RUL_FD00*.txt` line $i$.
- At any intermediate cycle $t \in [1, T_i]$, each elapsed operational cycle consumes exactly 1 unit of life:
  $$y^*(t) = y^*_{\text{final}} + (T_i - t)$$

### 2.2 Dual-Model Prediction & Error Residuals
For temporal sequence tensor $\mathbf{X}_t \in \mathbb{R}^{30 \times D}$:
- XGBoost point estimate: $\hat{y}_{\text{XGB}} = f_{\text{XGB}}(\mathbf{X}_t)$
- Bi-LSTM point estimate: $\hat{y}_{\text{LSTM}} = f_{\text{LSTM}}(\mathbf{X}_t)$
- Consensus estimate:
  $$\hat{y}_{\text{consensus}} = \frac{\hat{y}_{\text{XGB}} + \hat{y}_{\text{LSTM}}}{2}$$
- Inter-model epistemic spread:
  $$\Delta = \lvert \hat{y}_{\text{XGB}} - \hat{y}_{\text{LSTM}} \rvert$$
- Absolute prediction errors:
  $$e_{\text{XGB}} = \lvert \hat{y}_{\text{XGB}} - y^*(t) \rvert, \quad e_{\text{LSTM}} = \lvert \hat{y}_{\text{LSTM}} - y^*(t) \rvert, \quad e_{\text{consensus}} = \lvert \hat{y}_{\text{consensus}} - y^*(t) \rvert$$
- Spread-calibrated confidence score:
  $$\text{Conf} = \max\left(0, \min\left(1, 1 - \frac{\Delta}{\max(50, \hat{y}_{\text{consensus}})}\right)\right)$$

---

## 3. Prognostics Comparator Architecture

The verification engine was implemented in [`backend/services/ground_truth_comparator.py`](file:///c:/Users/LENOVO/Documents/Aircraft_Engine_AI/backend/services/ground_truth_comparator.py):

```
       [ dataset/raw/C-MAPSS/RUL_FD00*.txt ]
                         │
                         ▼
        [ get_ground_truth_rul(subset, unit, cycle) ]
                         │
                         ├──────────────────────────────────────────┐
                         ▼                                          ▼
            [ True RUL: y*(t) ]                        [ 30-Cycle Sequence Tensor X_t ]
                         │                                          │
                         │                   ┌──────────────────────┴──────────────────────┐
                         │                   ▼                                             ▼
                         │          [ XGBoost Champion ]                          [ Bi-LSTM Champion ]
                         │                   │                                             │
                         │                   ▼                                             ▼
                         │             y_hat_xgb                                     y_hat_lstm
                         │                   └──────────────────────┬──────────────────────┘
                         │                                          │
                         │                                          ▼
                         │                        [ Consensus: y_hat_cons, Spread: Δ ]
                         │                                          │
                         └───────────────────┬──────────────────────┘
                                             │
                                             ▼
                       [ Error Residuals: e_xgb, e_lstm, e_consensus ]
```

---

## 4. Empirical Evaluation Across Representative Cohorts

### 4.1 Final-Cycle Ground-Truth Comparison Table

All 12 representative test engines across all four subsets were evaluated at their final recorded cycle $T_i$:

| Subset | Unit ID | Final Cycle $T_i$ | True RUL $y^*$ | XGBoost $\hat{y}_{\text{XGB}}$ | Bi-LSTM $\hat{y}_{\text{LSTM}}$ | Consensus $\hat{y}_{\text{cons}}$ | Spread $\Delta$ | Consensus Error $e_{\text{cons}}$ | Conf Score |
|---|---|---|---|---|---|---|---|---|---|
| **FD001** | Unit 3 | 126 | **69.0** | 39.36 | 48.40 | **43.88** | 9.04 | 25.12 | 0.8193 |
| **FD001** | Unit 24 | 186 | **20.0** | 22.23 | 17.62 | **19.93** | 4.61 | **0.07** | 0.9078 |
| **FD001** | Unit 100 | 198 | **20.0** | 30.11 | 15.88 | **22.99** | 14.23 | **2.99** | 0.7154 |
| **FD002** | Unit 15 | 165 | **37.0** | 44.97 | 70.22 | **57.60** | 25.26 | 20.60 | 0.5615 |
| **FD002** | Unit 42 | 137 | **27.0** | 39.32 | 49.23 | **44.27** | 9.92 | 17.27 | 0.8016 |
| **FD002** | Unit 259 | 123 | **51.0** | 97.46 | 85.20 | **91.33** | 12.26 | 40.33 | 0.8658 |
| **FD003** | Unit 8 | 192 | **55.0** | 55.38 | 60.98 | **58.18** | 5.60 | **3.18** | 0.9038 |
| **FD003** | Unit 20 | 207 | **56.0** | 65.10 | 83.02 | **74.06** | 17.92 | 18.06 | 0.7580 |
| **FD003** | Unit 100 | 247 | **28.0** | 25.82 | 25.06 | **25.44** | 0.77 | **2.56** | 0.9846 |
| **FD004** | Unit 10 | 23 | **162.0** | 111.89 | 113.62 | **112.75** | 1.73 | 49.25* | 0.9847 |
| **FD004** | Unit 35 | 278 | **36.0** | 20.74 | 23.88 | **22.31** | 3.14 | 13.69 | 0.9372 |
| **FD004** | Unit 248 | 281 | **26.0** | 23.96 | 32.33 | **28.15** | 8.37 | **2.15** | 0.8326 |

*\*Note on FD004 Unit 10: In accordance with standard NASA C-MAPSS evaluation protocol, training targets were capped at a piecewise-linear maximum threshold ($RUL_{\max} = 125$ cycles). For an early-life test engine with only 23 elapsed cycles and true RUL of 162, predictions near ~112 reflect the calibrated early-life saturation plateau rather than model deficiency.*

### 4.2 Key Findings on Model Performance
1. **Near-Failure Precision:** For engines approaching end-of-life (e.g. FD001 Unit 24 with true RUL = 20), consensus prediction is **19.93 cycles** (error of **0.07 cycles** or **0.35%**).
2. **Compound Fault Resiliency:** For FD003 Unit 8 (true RUL = 55), consensus is **58.18 cycles** (error of 3.18 cycles); for FD003 Unit 100 (true RUL = 28), consensus is **25.44 cycles** (error of 2.56 cycles).
3. **Cohort Mean Absolute Error:** The mean absolute error across the FD001 representative cohort is **9.39 cycles**, well within the benchmark constraint of $\text{MAE} < 35$ cycles.

---

## 5. Trajectory Degradation Tracking & Monotonicity Analysis

To verify that model predictions realistically track the physical progression of component wear over flight hours, FD001 Unit 3 was evaluated at multiple points along its operational history:

| Operating Cycle $t$ | Trajectory Stage | Ground Truth $y^*(t)$ | XGBoost $\hat{y}_{\text{XGB}}$ | Bi-LSTM $\hat{y}_{\text{LSTM}}$ | Consensus $\hat{y}_{\text{cons}}$ | Inter-Model Spread $\Delta$ |
|---|---|---|---|---|---|---|
| **Cycle 30** | Early Life | **165.0** | 107.39 | 100.46 | **103.92** | 6.93 |
| **Cycle 60** | Mid Life | **135.0** | 118.75 | 117.57 | **118.16** | 1.18 |
| **Cycle 90** | Advanced Wear | **105.0** | 102.71 | 99.86 | **101.28** | 2.85 |
| **Cycle 126** | Truncation Point | **69.0** | 39.36 | 48.40 | **43.88** | 9.04 |

### Observations:
- **Ground-Truth Monotonicity:** True RUL strictly decreases linearly: $165.0 \to 135.0 \to 105.0 \to 69.0$.
- **Model Consensus Degradation:** From mid-life onwards, consensus predictions decrease monotonically: $118.16 \to 101.28 \to 43.88$, capturing rapid gas-path degradation as the high-pressure compressor degrades toward the end of the test sequence.

---

## 6. Test Suite Verification (`test_rul_ground_truth_comparison_step9_3.py`)

A dedicated 10-test test suite was authored in [`tests/test_rul_ground_truth_comparison_step9_3.py`](file:///c:/Users/LENOVO/Documents/Aircraft_Engine_AI/tests/test_rul_ground_truth_comparison_step9_3.py). All 10 tests passed cleanly in **7.82s**:

```
test_01_fd001_final_cycle_rul_comparison ... ok
test_02_fd002_multi_regime_rul_comparison ... ok
test_03_fd003_compound_fault_rul_comparison ... ok
test_04_fd004_complex_rul_comparison ... ok
test_05_trajectory_rul_decay_vs_ground_truth ... ok
test_06_consensus_rul_within_epistemic_bounds ... ok
test_07_api_predict_rul_endpoint_ground_truth_parity ... ok
test_08_error_distribution_sanity_check ... ok
test_09_ground_truth_file_length_and_format_validation ... ok
test_10_dataset_and_model_immutability ... ok
```

---

## 7. Cryptographic Artifact Immutability (Test 10)

All 12 raw NASA C-MAPSS dataset files were validated using SHA-256 digests:

| File | SHA-256 Digest | Status |
|---|---|---|
| `dataset/raw/C-MAPSS/RUL_FD001.txt` | `a19c8ec94931949d0485bdc35118206e9c81c4547b422efb9cf86f4ceddbceca` | VERIFIED |
| `dataset/raw/C-MAPSS/RUL_FD002.txt` | `c851dd96a6ea6998d3c4a8f834d3c8013aa90e93a6ed950dc826ad0655b2906b` | VERIFIED |
| `dataset/raw/C-MAPSS/RUL_FD003.txt` | `df1e0566306b174a2de41c67a3e7a51877889598b78643fc3e5685259091b7cb` | VERIFIED |
| `dataset/raw/C-MAPSS/RUL_FD004.txt` | `196b836b85a95ac7fdbbf29c5fdf1657382eafa445644d114ffaaf50dc2975e1` | VERIFIED |
| `dataset/raw/C-MAPSS/train_FD001.txt` | `963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8` | VERIFIED |
| `dataset/raw/C-MAPSS/train_FD002.txt` | `dac6c4dbc4e7c1bdeb5747da3d313d05c395bb99801b44a002b26a2ba13d788f` | VERIFIED |
| `dataset/raw/C-MAPSS/train_FD003.txt` | `2abbe9968cc5e8eb091980f51b20f62bb4127336d3482cb52071d53bf23329e2` | VERIFIED |
| `dataset/raw/C-MAPSS/train_FD004.txt` | `27ef6160b6a1dcb2613a88de9c239f763b223f02cdc41dc5cdedc5dc189b6218` | VERIFIED |
| `dataset/raw/C-MAPSS/test_FD001.txt` | `3cda7109ce17bafb5443f2ac926cfcf88154b941b8c4cf95eb55d1ddd6f52851` | VERIFIED |
| `dataset/raw/C-MAPSS/test_FD002.txt` | `de7b5bf7e998a985c378488480528b7c02cff1406a46740def362dda8d9b4e02` | VERIFIED |
| `dataset/raw/C-MAPSS/test_FD003.txt` | `299babd63c8d987cef079c4a425429f33b3a34797d803bbe2ad48c29dbd0d790` | VERIFIED |
| `dataset/raw/C-MAPSS/test_FD004.txt` | `1dc675fff0624bac10786927c6715b37d1297657137400d2b1a3138d777a3ba5` | VERIFIED |
| `models/scalers/scaler_FD001_tier3.pkl` | *Intact, non-zero file size* | VERIFIED |
| `models/scalers/scaler_FD002_tier3.pkl` | *Intact, non-zero file size* | VERIFIED |
| `models/scalers/scaler_FD003_tier3.pkl` | *Intact, non-zero file size* | VERIFIED |
| `models/scalers/scaler_FD004_tier3.pkl` | *Intact, non-zero file size* | VERIFIED |
| `models/checkpoints/xgb_FD001_champion.json` | *Intact, non-zero file size* | VERIFIED |
| `models/checkpoints/lstm_FD001_champion.pt` | *Intact, non-zero file size* | VERIFIED |

---

## 8. Full Repository Regression Test Results

```text
Ran 475 tests in 117.468s

OK
```

- **Phase 1 through Phase 8 Suites (37 suites):** 445 tests passed.
- **Phase 9 Step 9.1 Test Suite:** 10 tests passed.
- **Phase 9 Step 9.2 Test Suite:** 10 tests passed.
- **Phase 9 Step 9.3 Test Suite:** 10 tests passed.
- **Total Passing Tests:** **475 / 475 (100% Pass Rate)** across 40 test suites.
- **Regression Status:** Zero failures, zero regressions, zero warnings.
