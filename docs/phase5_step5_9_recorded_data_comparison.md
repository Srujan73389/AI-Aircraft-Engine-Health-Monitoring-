# Phase 5 — Step 5.9: Compare Digital Twin Outputs with Recorded Data

**Project:** AI-Based Aircraft Engine Health Monitoring, Fault Diagnosis and Predictive Maintenance  
**Dataset:** NASA C-MAPSS Turbofan Engine Degradation Dataset (FD001, FD002, FD003, FD004)  
**Milestone:** Phase 5 — Step 5.9: Recorded Data Comparison  
**Roadmap Reference:** Master Project Roadmap PDF, Section 6, Page 4–5: *"Compare simulated outputs with recorded data where valid."*  
**Status:** **Completed, Validated, and Formally Verified**

---

## Executive Summary

Phase 5 Step 5.9 establishes a rigorous, scientifically valid comparison framework between the Digital Twin's simulated virtual engine outputs and the original recorded NASA C-MAPSS engine data.

In complex industrial Digital Twins, outputs span multiple epistemic tiers—ranging from direct sensor pass-through values and algebraic thermodynamic indicators to multi-layer neural network prognostic forecasts and analytical what-if perturbations. Comparing simulated outputs with recorded data requires a strict taxonomy that distinguishes between variables that can be verified against physical measurements, variables that can be independently recomputed, variables that are statistical forecasts compared against ground-truth failure targets, and variables that have no physical ground truth.

This milestone delivers:
1. **Core Comparison Engine (`digital_twin/recorded_data_comparison.py`):** A robust comparison module (`RecordedDataComparator`) implementing strict alignment by `(unit_number, time_cycles)`, exact floating-point difference tracking, independent formula recomputation, sequence array indexing, dual-target RUL evaluation, and non-finite value rejection.
2. **Multi-Subset Execution Script (`scripts/run_step5_9_comparison.py` and `digital_twin/run_step5_9_comparison.py`):** An end-to-end evaluation pipeline comparing training and validation engine trajectories across all four C-MAPSS subsets (FD001, FD002, FD003, FD004).
3. **Ten Verified Comparison Artifacts (`reports/digital_twin/step5_9/`):** Four tabular CSV datasets, structured JSON summary, Markdown executive report, and four publication-quality diagnostic plots.
4. **Comprehensive Test Suite (`tests/test_digital_twin_step5_9.py`):** 16 dedicated unit and integration tests verifying all required aspects of Step 5.9, passing 100% cleanly in 14.24s, bringing total passing tests to 119 across the project.
5. **Zero Artifact Corruption & Immutability:** Pre- and post-session SHA-256 hash verification confirming zero modification to the 12 raw C-MAPSS dataset files, 4 subset scalers, and 8 Phase 4 champion model checkpoints.

---

## 1. Recorded Data Comparison Scope and Objectives

The objective of Step 5.9, as defined in Section 6 (Page 4–5) of the Master Project Roadmap, is to:
$$\text{"Compare simulated outputs with recorded data where valid."}$$

To satisfy this mandate with scientific integrity, the comparison framework addresses four fundamental research questions:
1. **Direct Telemetry Fidelity (Category A):** Does the Digital Twin accurately ingest, track, and output the 26 physical operational and sensor variables (G1–G26) without numerical drift, truncation, or corruption?
2. **Thermodynamic Consistency (Category B):** Do the derived consistency indicators (G27–G29) computed within the Digital Twin exactly match independent, first-principles algebraic recomputations directly derived from the raw telemetry?
3. **Preprocessing & Temporal Representation Alignment (Category C):** Does the Digital Twin's feature normalization (G30) and 30-cycle temporal tensor buffer (G31) exactly align with the Phase 3 training and sequence feature arrays?
4. **Prognostic Predictive Fidelity (Category D):** How closely do the Digital Twin's online ML prognostic forecasts (G35 XGBoost and G36 LSTM) match the recorded validation ground-truth RUL targets and the champion model predictions logged during Phase 4?
5. **Epistemic Boundary Demarcation (Category E):** Which variables (G32–G34, G37–G40) possess no physical ground truth in recorded telemetry, and why must artificial error metrics never be fabricated for them?

---

## 2. Dataset Sources and Data Integrity

The comparison evaluates data across three verified project data tiers:

### 2.1 Data Sources
1. **Raw C-MAPSS Telemetry (`dataset/raw/C-MAPSS/`):**
   - Training files: `train_FD001.txt`, `train_FD002.txt`, `train_FD003.txt`, `train_FD004.txt`
   - Test files: `test_FD001.txt`, `test_FD002.txt`, `test_FD003.txt`, `test_FD004.txt`
   - Ground truth RUL files: `RUL_FD001.txt`, `RUL_FD002.txt`, `RUL_FD003.txt`, `RUL_FD004.txt`
   - Used for Category A (G1–G26 pass-through) and Category B (G27–G29 recomputations).
2. **Phase 3 Preprocessed Artifacts (`dataset/processed/`):**
   - Scaled feature matrices: `dataset/processed/scaled/train_FD00*_scaled.csv`
   - Sequence arrays and metadata: `dataset/processed/sequences/X_train_FD00*.npy` and `train_metadata_FD00*.csv`
   - Used for Category C (G30 normalization and G31 rolling sequence buffers).
3. **Phase 4 Validation Predictions & Checkpoints (`reports/models/` and `models/`):**
   - Validation trajectory logs: `reports/models/predicted_trajectories_validation.csv`
   - Fitted scalers: `models/scalers/scaler_FD00*_tier3.pkl`
   - Champion models: `models/checkpoints/xgb_FD00*_champion.json` and `lstm_FD00*_champion.pt`
   - Used for Category D (G35 XGBoost and G36 LSTM prognostics).

### 2.2 SHA-256 Immutability Verification
All dataset files, scalers, and model checkpoints are monitored via pre- and post-session cryptographic hashing. As verified in `test_15` and `test_16`, zero bits were altered during test execution:
- 12 Raw C-MAPSS data files: **100% Bit-Identical (Verified)**
- 4 Phase 3 Scaler files: **100% Bit-Identical (Verified)**
- 8 Phase 4 Champion Model checkpoints: **100% Bit-Identical (Verified)**

---

## 3. Trajectory Alignment and Window Processing Methodology

A critical requirement of Step 5.9 is ensuring strict temporal and identity alignment across all comparison categories:

```
+-----------------------------------------------------------------------------------+
|                           Trajectory Synchronization                               |
+-----------------------------------------------------------------------------------+
|  Cycle t = 1, 2, ..., 29 (Initial Buffer Filling)                                 |
|  - Digital Twin: Replication-padded temporal tensor (G31) to initialize inference |
|  - Phase 3 Sequences: Excluded (Phase 3 stores unpadded sequences for t >= 30)   |
|  - Category A & B: Fully compared (Cycles 1..29 matched exactly)                  |
|  ------------------------------------------------------------------------------- |
|  Cycle t >= 30 (Full Window Ingestion)                                            |
|  - Digital Twin: 30-cycle causal sliding window S(t-29 : t)                       |
|  - Phase 3 Sequences: Exact 1-to-1 index matching via metadata window_idx         |
|  - Phase 4 Validation: Exact matching on (unit_number, time_cycles)               |
+-----------------------------------------------------------------------------------+
```

### 3.1 Strict Composite Key Alignment
Every comparison record is indexed by the primary tuple `(unit_number, time_cycles)`. If any cycle index is missing, duplicated, non-contiguous, or misaligned between the Digital Twin state and recorded telemetry, a `ValueError` or `RuntimeError` is immediately raised.

### 3.2 Temporal Window Alignment and Padding Handling
- In Phase 3 and Phase 4, temporal sequence arrays (`X_train_FD00X.npy`) and validation prediction records were constructed strictly from full 30-cycle unpadded windows ($t \ge 30$). Consequently, unpadded sequence records begin at cycle 30.
- In the Digital Twin runtime simulator (`DigitalTwinSimulator`), the engine must operate from cycle $t=1$. For cycles $t \in [1, 29]$, the simulator utilizes left-edge replication padding to prime the internal 30-cycle history buffer without lookahead.
- **Scientific Treatment:** The comparison framework explicitly marks cycles $1 \le t \le 29$ as `EXCLUDED_REPLICATION_PADDED` when evaluating Category C (G31 sequence arrays) and Category D (validation prognostics). For all cycles $t \ge 30$, records are marked `VERIFIED_EXACT_ALIGNMENT` and evaluated against Phase 3 sequence arrays and Phase 4 validation targets.

### 3.3 Dual RUL Target Alignment
For Category D prognostic evaluations on validation engines, the framework maintains strict separation between:
1. **Piecewise Linear Capped Target ($RUL_{capped}$):** Saturated at $125$ cycles ($RUL_{capped} = \min(RUL_{true}, 125)$), representing the exact training target of the Phase 4 ML champions.
2. **Linear Uncapped True RUL ($RUL_{uncapped}$):** The continuous linear flight cycles remaining until engine retirement.
Both targets are recorded in `rul_comparison.csv` and never conflated.

---

## 4. Taxonomy of State Variables (Categories A–E)

The 40 virtual engine state variables (G1–G40) are categorized into five distinct epistemic classes:

| Category | Group Name | Variables | Description | Comparison Basis |
| :---: | :--- | :---: | :--- | :--- |
| **A** | Direct Measured Telemetry | G1–G26 | Operational metadata (unit, cycle), operating context (alt, Mach, TRA), and 21 thermodynamic sensors | Raw C-MAPSS text telemetry (`train_FD00*.txt`, `test_FD00*.txt`) |
| **B** | Derived Consistency Indicators | G27–G29 | High-pressure compressor temp rise ($\Delta T_{HPC}$), overall pressure ratio ($OPR$), spool speed ratio ($N_{ratio}$) | First-principles formulas recomputed independently from raw telemetry |
| **C** | Preprocessing & Sequences | G30–G31 | Normalized feature vector (14-dim or 24-dim) and 30-cycle temporal tensor ($30 \times 14$ or $30 \times 24$) | Phase 3 MinMax scaled CSVs and `.npy` sequence tensors |
| **D** | Machine Learning Prognostics | G35–G36 | Online XGBoost and LSTM Remaining Useful Life predictions | Validation ground-truth RUL targets and Phase 4 champion logs |
| **E** | Unobservable / Simulation Variables | G32–G34, G37–G40 | Anomaly scores, alert flag, prognostic spread, virtual health index, thermal stress offset, sensor bias | **None (Explicitly Classified as Unobservable / Simulation-Only)** |

---

## 5. Quantitative Verification Results Across All Four Subsets

Evaluations were performed across representative engines from all four C-MAPSS subsets:
- **Category A, B, and C:** Evaluated on verified training Unit 1 across FD001 (192 cycles), FD002 (149 cycles), FD003 (259 cycles), and FD004 (321 cycles), totaling 921 operational cycles.
- **Category D:** Evaluated on verified validation Unit 2 across FD001 (258 unpadded windows), FD002 (240 unpadded windows), FD003 (224 unpadded windows), and FD004 (270 unpadded windows), totaling 992 validation prognostic windows.

### 5.1 Category A: Direct Measured Telemetry Pass-Through (G1–G26)

Every state variable G1–G26 was compared against recorded C-MAPSS values:

| Subset | Engine ID | Cycles Compared | Variables Evaluated | Total Values | Exact Matches ($|\Delta| = 0.0$) | Mismatches | Max Abs Diff | Verification Result |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FD001** | Unit 1 | 192 | 26 | 4,992 | 4,992 | 0 | $0.00 \times 10^0$ | **100% Exact Value Equality** |
| **FD002** | Unit 1 | 149 | 26 | 3,874 | 3,874 | 0 | $0.00 \times 10^0$ | **100% Exact Value Equality** |
| **FD003** | Unit 1 | 259 | 26 | 6,734 | 6,734 | 0 | $0.00 \times 10^0$ | **100% Exact Value Equality** |
| **FD004** | Unit 1 | 321 | 26 | 8,346 | 8,346 | 0 | $0.00 \times 10^0$ | **100% Exact Value Equality** |
| **TOTAL** | — | **921** | **26** | **23,946** | **23,946** | **0** | **0.00** | **Zero Telemetry Drift** |

### 5.2 Category B: Independently Recomputed Derived Indicators (G27–G29)

Thermodynamic consistency formulas were independently recomputed directly from the raw telemetry:
- $G27: \Delta T_{HPC} = T_{30} - T_{24}$ ($sensor\_3 - sensor\_2$)
- $G28: OPR = P_{30} / P_2$ ($sensor\_7 / sensor\_5$)
- $G29: N_{ratio} = N_c / N_f$ ($sensor\_9 / sensor\_8$)

| Subset | Engine ID | Cycles | G27 Max Diff | G28 Max Diff | G29 Max Diff | Exact Matches (G27 / G28 / G29) | Finiteness Check | Recomputation Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FD001** | Unit 1 | 192 | $0.00$ | $0.00$ | $0.00$ | 192 / 192 / 192 | 100% Finite | **Exact Recomputation** |
| **FD002** | Unit 1 | 149 | $0.00$ | $0.00$ | $0.00$ | 149 / 149 / 149 | 100% Finite | **Exact Recomputation** |
| **FD003** | Unit 1 | 259 | $0.00$ | $0.00$ | $0.00$ | 259 / 259 / 259 | 100% Finite | **Exact Recomputation** |
| **FD004** | Unit 1 | 321 | $0.00$ | $0.00$ | $0.00$ | 321 / 321 / 321 | 100% Finite | **Exact Recomputation** |

### 5.3 Category C: Preprocessing and Sequence Representations (G30–G31)

Normalized feature vectors (G30) and 30-cycle temporal tensors (G31) were compared against Phase 3 artifacts:

| Subset | Engine ID | G30 Feature Dim | G30 Max Diff vs Scaled CSV | G31 Unpadded Windows ($t \ge 30$) | G31 Excluded Padded Cycles ($t < 30$) | G31 Max Tensor Diff vs Phase 3 Array | Precision Bound |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FD001** | Unit 1 | 14 (Core Tier 1) | $0.00 \times 10^0$ | 163 | 29 | $2.98 \times 10^{-8}$ | Float32 storage limit ($\le 10^{-7}$) |
| **FD002** | Unit 1 | 24 (Extended Tier 3) | $0.00 \times 10^0$ | 120 | 29 | $2.98 \times 10^{-8}$ | Float32 storage limit ($\le 10^{-7}$) |
| **FD003** | Unit 1 | 24 (Extended Tier 3) | $0.00 \times 10^0$ | 230 | 29 | $2.98 \times 10^{-8}$ | Float32 storage limit ($\le 10^{-7}$) |
| **FD004** | Unit 1 | 24 (Extended Tier 3) | $0.00 \times 10^0$ | 292 | 29 | $2.98 \times 10^{-8}$ | Float32 storage limit ($\le 10^{-7}$) |

### 5.4 Category D: ML Prognostic RUL Outputs (G35–G36)

Digital Twin prognostic predictions were benchmarked against validation ground-truth RUL targets and against the predictions recorded by Phase 4 champions in `predicted_trajectories_validation.csv`:

| Subset | Partition | Unit ID | Windows ($t \ge 30$) | XGB MAE (Capped) | XGB RMSE (Capped) | LSTM MAE (Capped) | LSTM RMSE (Capped) | XGB MAE vs Phase 4 Champion | LSTM MAE vs Phase 4 Champion |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FD001** | Validation | Unit 2 | 258 | 5.01 cycles | 6.45 cycles | 5.57 cycles | 7.77 cycles | **0.0194 cycles** | **0.0023 cycles** |
| **FD002** | Validation | Unit 2 | 240 | 23.37 cycles | 26.86 cycles | 17.82 cycles | 21.22 cycles | **0.0024 cycles** | **0.0027 cycles** |
| **FD003** | Validation | Unit 2 | 224 | 6.59 cycles | 9.86 cycles | 8.36 cycles | 13.83 cycles | **0.0074 cycles** | **0.0027 cycles** |
| **FD004** | Validation | Unit 2 | 270 | 8.44 cycles | 10.64 cycles | 6.87 cycles | 8.84 cycles | **0.0025 cycles** | **0.0025 cycles** |

**Key Prognostic Insights:**
1. **Model Accuracy vs Operating Complexity:** On single-condition datasets (FD001 and FD003), both models exhibit excellent accuracy with MAE $\le 6.6$ cycles (XGBoost) and $\le 8.4$ cycles (LSTM). On multi-condition datasets (FD002 and FD004), LSTM outperforms XGBoost on complex operating regimes (FD002: LSTM MAE 17.82 vs XGB MAE 23.37; FD004: LSTM MAE 6.87 vs XGB MAE 8.44).
2. **Inference Consistency with Phase 4:** The Digital Twin inference engine reproduces genuine Phase 4 champion predictions with an average absolute discrepancy $< 0.02$ cycles for XGBoost and $< 0.003$ cycles for LSTM, attributable to minor batch-versus-streaming runtime floating-point evaluation order.

### 5.5 Category E: Variables Without Recorded Physical Ground Truth

Variables G32–G34 and G37–G40 have no recorded physical counterparts in the NASA C-MAPSS dataset:

```
Category E State Variables (Strictly No Artificial Ground Truth Assigned):
-------------------------------------------------------------------------
- G32 (isolation_forest_anomaly_score): Unsupervised score from Isolation Forest ensemble.
- G33 (pca_spe_residual_energy): Reconstruction error (Q-statistic) from PCA model.
- G34 (persistence_alert_flag): Discrete rule-based multi-cycle alarm condition.
- G37 (prognostic_spread): Epistemic uncertainty metric |G35 - G36| between models.
- G38 (virtual_health_index): Continuous synthetic degradation indicator H(t) in [0, 1].
- G39 (thermal_stress_offset): Hypothetical what-if simulation parameter (delta T).
- G40 (sensor_bias_delta): Hypothetical what-if sensor bias scenario parameter.
```

**Diagnostic Integrity Policy:** In accordance with rigorous scientific practice, no artificial ground truth or synthetic error calculations are fabricated for Category E variables. They are documented strictly as model-derived diagnostics, uncertainty indicators, or simulation controls.

---

## 6. High-Resolution Diagnostic Visualizations

Four publication-quality multi-panel diagnostic figures were generated and saved to `reports/digital_twin/step5_9/`:

1. **`telemetry_comparison_plot.png` (Category A):**
   - Displays four critical thermodynamic sensors ($T_{30}$, $P_{30}$, $N_f$, $N_c$) across the complete 192-cycle trajectory of FD001 Unit 1.
   - Overlays recorded C-MAPSS measurements and Digital Twin state variables, with a lower residual subplot demonstrating exact zero residuals ($|\Delta| = 0.0$) across all cycles.
2. **`derived_indicator_comparison_plot.png` (Category B):**
   - Displays G27 ($\Delta T_{HPC}$), G28 ($OPR$), and G29 ($N_{ratio}$) for FD001 Unit 1.
   - Plots the Digital Twin derived state vs the independent algebraic recomputations, with absolute error panels showing zero error ($0.00$).
3. **`preprocessing_comparison_plot.png` (Category C):**
   - Panel 1: Bar chart comparing normalized feature vectors (G30) between Phase 3 scaled features and Digital Twin states across 14 core sensors.
   - Panel 2: Heatmap of the $30 \times 14$ temporal sequence tensor (G31) at cycle 100, verifying exact sliding-window temporal structure.
4. **`rul_prognostic_comparison_plot.png` (Category D):**
   - 4-panel multi-subset prognostic comparison across validation Unit 2 for FD001, FD002, FD003, and FD004.
   - Simultaneously plots uncapped true RUL, capped ground-truth target ($\le 125$), Digital Twin XGBoost RUL, and Digital Twin LSTM RUL over engine operating life.

---

## 7. Artifact Manifest

All comparison artifacts are stored in `reports/digital_twin/step5_9/`:

| Artifact Path | Format | Size | Description |
| :--- | :---: | :---: | :--- |
| `reports/digital_twin/step5_9/telemetry_comparison.csv` | CSV | 1,454,920 bytes | Record-by-record comparison of G1–G26 vs raw C-MAPSS telemetry across 921 cycles |
| `reports/digital_twin/step5_9/derived_indicator_comparison.csv` | CSV | 122,774 bytes | Cycle-by-cycle recomputed G27–G29 consistency indicators vs Digital Twin |
| `reports/digital_twin/step5_9/preprocessing_comparison.csv` | CSV | 93,123 bytes | G30 feature vector and G31 sequence tensor alignment and padding audits |
| `reports/digital_twin/step5_9/rul_comparison.csv` | CSV | 199,859 bytes | Validation RUL targets (capped & uncapped) vs DT XGBoost and LSTM predictions |
| `reports/digital_twin/step5_9/comparison_summary.json` | JSON | 9,441 bytes | Machine-readable quantitative comparison metrics and unobservable classifications |
| `reports/digital_twin/step5_9/comparison_summary.md` | Markdown | 5,354 bytes | Executive markdown summary report with multi-subset performance tables |
| `reports/digital_twin/step5_9/telemetry_comparison_plot.png` | PNG | 1,311,078 bytes | High-resolution Category A telemetry pass-through validation plot |
| `reports/digital_twin/step5_9/derived_indicator_comparison_plot.png` | PNG | 1,315,436 bytes | High-resolution Category B derived indicator recomputation plot |
| `reports/digital_twin/step5_9/preprocessing_comparison_plot.png` | PNG | 263,241 bytes | High-resolution Category C normalized feature vector and tensor heatmap plot |
| `reports/digital_twin/step5_9/rul_prognostic_comparison_plot.png` | PNG | 618,398 bytes | High-resolution Category D 4-subset validation RUL prognostic trajectory plot |

---

## 8. Verification and Test Suite Coverage

A dedicated test suite was implemented in `tests/test_digital_twin_step5_9.py`, containing 16 rigorous verification tests covering every technical aspect of Step 5.9:

| Test ID | Test Method Name | Verification Scope | Status |
| :---: | :--- | :--- | :---: |
| **01** | `test_01_correct_engine_and_cycle_alignment` | Validates monotonic, contiguous `(unit, cycle)` trajectory alignment | **PASS** |
| **02** | `test_02_direct_telemetry_exact_equality` | Verifies 100% exact equality for pass-through variables G1–G26 ($|\Delta| = 0.0$) | **PASS** |
| **03** | `test_03_detection_of_deliberately_mismatched_telemetry` | Confirms altered/tampered sensor values are detected and flagged as mismatches | **PASS** |
| **04** | `test_04_independent_g27_calculation` | Recomputes $\Delta T_{HPC} = T_{30} - T_{24}$ from raw measurements and verifies equality | **PASS** |
| **05** | `test_05_independent_g28_calculation` | Recomputes $OPR = P_{30} / P_2$ from raw measurements and verifies equality | **PASS** |
| **06** | `test_06_independent_g29_calculation` | Recomputes $N_{ratio} = N_c / N_f$ from raw measurements and verifies equality | **PASS** |
| **07** | `test_07_correct_handling_all_four_subsets` | Confirms clean execution across all four subsets (FD001–FD004) | **PASS** |
| **08** | `test_08_correct_scaler_and_feature_tier_identification` | Verifies FD001 uses 14 core features; FD002–FD004 use 24 extended features | **PASS** |
| **09** | `test_09_correct_sequence_alignment_and_window_validation` | Verifies $t < 30$ padding exclusion and $t \ge 30$ Phase 3 tensor alignment | **PASS** |
| **10** | `test_10_validation_rul_target_alignment` | Verifies capped target ($\le 125$) vs uncapped true RUL target preservation | **PASS** |
| **11** | `test_11_mae_rmse_max_error_calculation_fidelity` | Checks MAE, RMSE, and Max Error calculations against explicit numpy formulas | **PASS** |
| **12** | `test_12_nan_and_infinity_rejection` | Rejects non-finite values (NaN, +inf, -inf) in telemetry with ValueError | **PASS** |
| **13** | `test_13_duplicate_and_missing_alignment_key_detection` | Detects duplicate or misaligned cycle indices and raises ValueError | **PASS** |
| **14** | `test_14_separation_of_measured_derived_predicted_simulated_values` | Verifies Category E classification and absence of fabricated error metrics | **PASS** |
| **15** | `test_15_raw_dataset_before_and_after_immutability` | Validates SHA-256 hashes of all 12 raw C-MAPSS dataset files | **PASS** |
| **16** | `test_16_model_scalers_and_checkpoints_immutability` | Validates SHA-256 hashes of 4 scalers and 8 champion model checkpoints | **PASS** |

### 8.1 Test Execution Output
Execution of the Step 5.9 test module:
```powershell
.\.venv\Scripts\python.exe -m unittest tests/test_digital_twin_step5_9.py
................
----------------------------------------------------------------------
Ran 16 tests in 14.238s

OK
```

Full project-wide unit test discovery:
```powershell
.\.venv\Scripts\python.exe -m unittest discover tests
.......................................................................................................................
----------------------------------------------------------------------
Ran 119 tests in 30.436s

OK
```

---

## 9. Scientific Limitations and Operational Boundaries

To maintain strict scientific and engineering rigor, the following boundaries and limitations are formally stated:

1. **Academic Simulation Scope:** The Digital Twin prototype is developed solely as an analytical research demonstration using NASA C-MAPSS synthetic turbofan degradation simulations. It is not an FAA-certified, DO-178C software component or real-time Full Authority Digital Engine Control (FADEC) system.
2. **Telemetry Invariance vs Physical Gas Turbine Validation:** The exact mathematical pass-through equality ($|\Delta| = 0.0$) observed in Category A proves pipeline integrity, memory safety, and lossless ingestion. It does not imply that the software solves full 3D aerothermal Navier-Stokes equations inside an operating gas turbine.
3. **Statistical Prognostic Accuracy vs Ground Truth:** The Category D evaluation quantifies the statistical generalization accuracy of Phase 4 machine learning models on unseen validation engines. RUL predictions are estimates subject to epistemic uncertainty, sensor noise, and multi-regime variability.
4. **Epistemic Integrity of Unobservable Quantities:** Variables G32–G34 and G37–G40 are explicitly demarcated as unobservable or simulation-only. Refusing to invent synthetic ground truth for them ensures the Digital Twin adheres to standard scientific principles.

---

## 10. Conclusion and Next Steps

Phase 5 Step 5.9 has successfully verified the Digital Twin against recorded C-MAPSS data across all five epistemic categories and all four dataset subsets. Every deliverable specified in the Master Roadmap has been implemented, validated, and documented.

In strict adherence to project guidelines, **Step 5.10 (Numerical Stability & Repeatability)** and **Step 5.11 (Digital Twin Assumptions & Limitations Document)** remain untouched pending user instruction.
