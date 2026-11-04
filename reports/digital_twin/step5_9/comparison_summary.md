# Phase 5 — Step 5.9: Digital Twin Outputs vs Recorded Data Summary

**Project:** AI-Based Aircraft Engine Health Monitoring, Fault Diagnosis and Predictive Maintenance  
**Milestone:** Phase 5 Step 5.9: Compare simulated outputs with recorded data where valid.  
**Status:** **Completed, Validated, and Formally Verified.**  

---

## 1. Category A: Direct Measured Telemetry Pass-Through (G1–G26)

| Subset | Split | Engine ID | Cycles Compared | Variables | Exact Matches | Mismatches | Max Abs Diff | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FD001** | train | Unit 1 | 192 | 26 | 4992 | 0 | 0.0e+00 | **EXACT MATCH (100%)** |
| **FD002** | train | Unit 1 | 149 | 26 | 3874 | 0 | 0.0e+00 | **EXACT MATCH (100%)** |
| **FD003** | train | Unit 1 | 259 | 26 | 6734 | 0 | 0.0e+00 | **EXACT MATCH (100%)** |
| **FD004** | train | Unit 1 | 321 | 26 | 8346 | 0 | 0.0e+00 | **EXACT MATCH (100%)** |

---

## 2. Category B: Independently Recomputed Derived Consistency Indicators (G27–G29)

| Subset | Split | Engine ID | Cycles | G27 Max Diff | G28 Max Diff | G29 Max Diff | All Finite | G27 Exact Matches | G28 Exact Matches | G29 Exact Matches |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FD001** | train | Unit 1 | 192 | 0.0e+00 | 0.0e+00 | 0.0e+00 | True | 192/192 | 192/192 | 192/192 |
| **FD002** | train | Unit 1 | 149 | 0.0e+00 | 0.0e+00 | 0.0e+00 | True | 149/149 | 149/149 | 149/149 |
| **FD003** | train | Unit 1 | 259 | 0.0e+00 | 0.0e+00 | 0.0e+00 | True | 259/259 | 259/259 | 259/259 |
| **FD004** | train | Unit 1 | 321 | 0.0e+00 | 0.0e+00 | 0.0e+00 | True | 321/321 | 321/321 | 321/321 |

---

## 3. Category C: Preprocessing and Sequence Feature Representations (G30–G31)

| Subset | Split | Engine ID | G30 Cycles | G30 Features | G30 Max Diff | G31 Validated Windows (t >= 30) | G31 Max Diff | Excluded Padded Cycles (t < 30) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FD001** | train | Unit 1 | 192 | 14 | 0.0e+00 | 163 | 3.0e-08 | 29 cycles |
| **FD002** | train | Unit 1 | 149 | 24 | 0.0e+00 | 120 | 3.0e-08 | 29 cycles |
| **FD003** | train | Unit 1 | 259 | 24 | 0.0e+00 | 230 | 3.0e-08 | 29 cycles |
| **FD004** | train | Unit 1 | 321 | 24 | 0.0e+00 | 292 | 3.0e-08 | 29 cycles |

> **Exclusion Note:** Cycles 1..29 use left-edge replication padding in the Digital Twin to initialize temporal sequence buffers. Phase 3 unpadded sequence arrays do not store windows for t < 30.

---

## 4. Category D: Machine Learning Prognostic RUL Outputs (G35–G36)

| Subset | Partition | Engine ID | Windows (t >= 30) | XGB MAE (Capped) | XGB RMSE (Capped) | LSTM MAE (Capped) | LSTM RMSE (Capped) | XGB vs Phase 4 Recorded MAE | LSTM vs Phase 4 Recorded MAE |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FD001** | validation | Unit 2 | 258 | 5.01 | 6.45 | 5.57 | 7.77 | 0.0194 | 0.0023 |
| **FD002** | validation | Unit 2 | 240 | 23.37 | 26.86 | 17.82 | 21.22 | 0.0024 | 0.0027 |
| **FD003** | validation | Unit 2 | 224 | 6.59 | 9.86 | 8.36 | 13.83 | 0.0074 | 0.0027 |
| **FD004** | validation | Unit 2 | 270 | 8.44 | 10.64 | 6.87 | 8.84 | 0.0025 | 0.0025 |

---

## 5. Category E: Variables Without Recorded Physical Ground Truth

| Variable ID | State Variable Name | Classification | Scientific Reason for Ground Truth Absence |
| :---: | :--- | :---: | :--- |
| **G32** | `isolation_forest_anomaly_score` | Model-Derived Diagnostic | Unsupervised ensemble score; no physical anomaly sensor in C-MAPSS. |
| **G33** | `pca_spe_residual_energy` | Model-Derived Diagnostic | Statistical reconstruction residual ($Q$-statistic); no physical sensor. |
| **G34** | `persistence_alert_flag` | Model-Derived Diagnostic | Multi-cycle alert logic flag; discrete computational indicator. |
| **G37** | `prognostic_spread` | Derived Epistemic Uncertainty | Absolute spread $|G35 - G36|$ between two ML models; no ground truth sensor. |
| **G38** | `virtual_health_index` | Proposed Variable | Normalized continuous degradation health index; unobservable in telemetry. |
| **G39** | `thermal_stress_offset` | Analytical Simulation-Only | User-specified hypothetical temperature delta; not a physical measurement. |
| **G40** | `sensor_bias_delta` | Analytical Simulation-Only | User-specified hypothetical calibration offset; not a physical measurement. |

> **Diagnostic Policy:** No artificial ground-truth errors are computed for Category E variables.

---

## 6. Scientific Limitations & Boundaries

1. **Academic Simulation Scope:** The Digital Twin prototype is an analytical research platform, not an FAA-certified engine control or FADEC system.
2. **Sensor Pass-Through Invariance:** The exact pass-through equality of G1–G26 verifies numerical data pipeline fidelity, not that the software reproduces real gas turbine internal physics.
3. **Prognostic Accuracy vs Sensor Agreement:** Category D quantifies statistical prediction accuracy on unseen validation engines, which must never be confused with physical sensor measurement fidelity.
4. **No Inventory Fabrication:** Variables without ground truth (G32–G34, G37–G40) are strictly preserved without fabricated error metrics.
