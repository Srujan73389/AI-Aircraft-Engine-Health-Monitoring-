# Phase 3 – Step 3.8: Feature Scaling Execution & Verification Report

**Project:** AI-Based Aircraft Engine Health Monitoring, Fault Diagnosis and Predictive Maintenance  
**Dataset:** NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS)  
**Execution Script:** `preprocessing/apply_feature_scaling.py`  
**Pipeline Module:** `preprocessing/scaling_pipeline.py`  
**Scaler Manifest:** `models/scalers/scaler_manifest.json`  
**Document Status:** Complete (Step 3.8 Verification & Delivery)  
**Date:** November 2026  

---

## 1. Executive Summary & Objective

Step 3.8 executes the feature scaling strategy formulated in Step 3.7 using the strict engine-level training partitions established in Step 3.10.

To eliminate data leakage:
- **Fitting:** Each scaler is fitted **strictly on the training partition engines** (`train_FD00x_split.csv`).
- **Transformation:** The exact same fitted scaler parameters are applied without modification to transform the matching validation partition (`validation_FD00x_split.csv`) and test dataset (`test_FD00x_with_RUL.csv`).
- **No Data Clipping:** Values in validation or test partitions that fall outside the training min/max bounds are **not** artificially clipped to $[0, 1]$, preserving true physical distribution variances.
- **Target & Metadata Isolation:** `unit_number`, `time_cycles`, and `RUL` are completely preserved in their original unscaled values.

---

## 2. Scaling Methodology & Mathematical Formulation

### 2.1 Formula
The approved scaling method is `MinMaxScaler(feature_range=(0, 1))`. For each feature $j \in \{1, \dots, 24\}$:

$$x'_{ij} = \frac{x_{ij} - x_{\min, j}^{\text{train}}}{x_{\max, j}^{\text{train}} - x_{\min, j}^{\text{train}}}$$

Where:
- $x_{\min, j}^{\text{train}} = \min_{i \in \text{TrainEngines}} \{ x_{ij} \}$ is the minimum value observed in the training partition.
- $x_{\max, j}^{\text{train}} = \max_{i \in \text{TrainEngines}} \{ x_{ij} \}$ is the maximum value observed in the training partition.
- For invariant/constant features where $x_{\max, j}^{\text{train}} - x_{\min, j}^{\text{train}} \equiv 0$, the denominator is safely protected and $x'_{ij} \equiv 0.0$, preventing division-by-zero errors.

### 2.2 Reusable Pipeline Architecture
The execution reuses the modular class `CMAPSSScalingPipeline` implemented in `preprocessing/scaling_pipeline.py`. No redundant scaling logic was created.

---

## 3. Feature Configuration Tiers

While the pipeline supports all three feature tiers defined in Step 3.6, **Tier 3 (All 24 Predictors)** was selected as the comprehensive common baseline for Step 3.8:
- **Tier 1 (14 Core Sensors):** `sensor_2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21`.
- **Tier 2 (17 Multi-Condition Features):** Tier 1 features + `op_setting_1, op_setting_2, op_setting_3`.
- **Tier 3 (24 Extended Predictors – Deployed):** All 3 operating settings + 21 sensors (`sensor_1` through `sensor_21`).

*Note:* Generating Tier 1 and Tier 2 subsets downstream requires no re-scaling; they can be extracted directly via feature column masks or independently fitted using the existing pipeline class.

---

## 4. Subset Partitioning and Scaler Scope

Scalers are fitted **independently per subset** because aerodynamic flight conditions and degradation kinetics differ across the four C-MAPSS subsets:

| Subset | Operating Conditions | Fault Modes | Training Partition | Scaler Object Saved | Scaler Type | Features Scaled |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FD001** | Single (Sea Level) | HPC Degradation | `train_FD001_split.csv` | `models/scalers/scaler_FD001_tier3.pkl` | `SafeMinMaxScaler` | 24 |
| **FD002** | Six Regimes | HPC Degradation | `train_FD002_split.csv` | `models/scalers/scaler_FD002_tier3.pkl` | `SafeMinMaxScaler` | 24 |
| **FD003** | Single (Sea Level) | HPC & Fan Degradation | `train_FD003_split.csv` | `models/scalers/scaler_FD003_tier3.pkl` | `SafeMinMaxScaler` | 24 |
| **FD004** | Six Regimes | HPC & Fan Degradation | `train_FD004_split.csv` | `models/scalers/scaler_FD004_tier3.pkl` | `SafeMinMaxScaler` | 24 |

---

## 5. Output Datasets & Artifacts Generated

### 5.1 Scaled Dataset CSV Files (`dataset/processed/scaled/`)
All files contain 27 columns: `unit_number`, `time_cycles`, 24 scaled features, and `RUL`.

| Subset | Partition | Output File | Rows | Engines | File Size (Bytes) |
| :---: | :---: | :--- | :---: | :---: | :---: |
| **FD001** | Train | `dataset/processed/scaled/train_FD001_scaled.csv` | 16,340 | 80 | 5,559,643 |
| **FD001** | Validation | `dataset/processed/scaled/validation_FD001_scaled.csv` | 4,291 | 20 | 1,460,443 |
| **FD001** | Test | `dataset/processed/scaled/test_FD001_scaled.csv` | 13,096 | 100 | 4,464,522 |
| **FD002** | Train | `dataset/processed/scaled/train_FD002_scaled.csv` | 42,889 | 208 | 17,230,678 |
| **FD002** | Validation | `dataset/processed/scaled/validation_FD002_scaled.csv` | 10,870 | 52 | 4,366,244 |
| **FD002** | Test | `dataset/processed/scaled/test_FD002_scaled.csv` | 33,991 | 259 | 13,655,706 |
| **FD003** | Train | `dataset/processed/scaled/train_FD003_scaled.csv` | 20,177 | 80 | 7,313,591 |
| **FD003** | Validation | `dataset/processed/scaled/validation_FD003_scaled.csv` | 4,543 | 20 | 1,646,436 |
| **FD003** | Test | `dataset/processed/scaled/test_FD003_scaled.csv` | 16,596 | 100 | 6,033,023 |
| **FD004** | Train | `dataset/processed/scaled/train_FD004_scaled.csv` | 48,496 | 199 | 19,794,125 |
| **FD004** | Validation | `dataset/processed/scaled/validation_FD004_scaled.csv` | 12,753 | 50 | 5,209,416 |
| **FD004** | Test | `dataset/processed/scaled/test_FD004_scaled.csv` | 41,214 | 248 | 16,854,375 |

### 5.2 Fitted Scaler Objects & Audit Manifest (`models/scalers/`)
- `models/scalers/scaler_FD001_tier3.pkl` (Fitted pipeline object)
- `models/scalers/scaler_FD002_tier3.pkl` (Fitted pipeline object)
- `models/scalers/scaler_FD003_tier3.pkl` (Fitted pipeline object)
- `models/scalers/scaler_FD004_tier3.pkl` (Fitted pipeline object)
- [`models/scalers/scaler_manifest.json`](file:///c:/Users/LENOVO/Documents/Aircraft_Engine_AI/models/scalers/scaler_manifest.json): Contains complete per-feature training minimums, maximums, ranges, and constant feature indicators for every subset.

---

## 6. Mandatory Verification Checklist & Audit Results

All 12 mandatory verification checks passed across all four subsets:

| Check ID | Verification Item | FD001 | FD002 | FD003 | FD004 | Criteria |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **1** | Fitted on Train Only | PASS | PASS | PASS | PASS | Scaler fitted strictly on training partition |
| **2** | Train Bounds in $[0, 1]$ | PASS | PASS | PASS | PASS | Min $\ge 0.0$ and Max $\le 1.0$ within float tolerance |
| **3** | Scaler Reproducibility | PASS | PASS | PASS | PASS | Reloading `.pkl` reproduces transforms identically |
| **4** | Unclipped Val / Test | PASS | PASS | PASS | PASS | No artificial clipping to $[0, 1]$ applied |
| **5** | Zero NaNs / Infs | PASS | PASS | PASS | PASS | 0 nulls, 0 NaNs, 0 Infs across all scaled sets |
| **6** | Constant Features Safe | PASS | PASS | PASS | PASS | Constant features mapped strictly to $0.0$ |
| **7** | Metadata Preserved | PASS | PASS | PASS | PASS | `unit_number` and `time_cycles` match unscaled sets |
| **8** | RUL Target Preserved | PASS | PASS | PASS | PASS | `RUL` values match unscaled sets identically |
| **9** | Row & Engine Counts | PASS | PASS | PASS | PASS | Zero row loss and zero engine drop across all sets |
| **10** | Schema & Column Order | PASS | PASS | PASS | PASS | Exact 27 columns in identical order across splits |
| **11** | Inputs Immutability | PASS | PASS | PASS | PASS | Input CSVs accessed read-only; untouched |
| **12** | Reload Verification | PASS | PASS | PASS | PASS | Deserialized scalers match runtime transforms |

---

## 7. Out-of-Bounds Analysis for Validation and Test Partitions

Because the scalers were fitted strictly on training partition engines, unseen validation or test engines with operational points outside the training min/max bounds naturally transform slightly outside $[0, 1]$. 

Per project requirements, these values were **not forcibly clipped**, preserving physical anomalies:

### 7.1 Out-of-Bounds Summary Table

| Subset | Split | Transformed Range | Out-of-Bounds Cells | % of Total Cells | Affected Features |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **FD001** | Validation | $[-0.0151, 1.0446]$ | 6 / 102,984 | **0.006%** | `op_setting_1` (2), `sensor_2` (1), `sensor_4` (1), `sensor_14` (2) |
| **FD001** | Test | $[-0.0436, 1.0833]$ | 18 / 314,304 | **0.006%** | `op_setting_2` (5), `sensor_2` (2), `sensor_3` (3), `sensor_8` (1), `sensor_11` (2), `sensor_12` (3), `sensor_21` (2) |
| **FD002** | Validation | $[-0.0003, 1.0055]$ | 8 / 260,880 | **0.003%** | `sensor_3` (2), `sensor_12` (2), `sensor_20` (4) |
| **FD002** | Test | $[-0.0041, 1.0017]$ | 10 / 815,784 | **0.001%** | `sensor_2` (1), `sensor_7` (1), `sensor_11` (1), `sensor_15` (1), `sensor_20` (5), `sensor_21` (1) |
| **FD003** | Validation | $[0.0000, 1.0833]$ | 12 / 109,032 | **0.011%** | `op_setting_1` (1), `op_setting_2` (10), `sensor_7` (1) |
| **FD003** | Test | $[-0.0909, 1.0181]$ | 7 / 398,304 | **0.002%** | `op_setting_1` (2), `sensor_7` (1), `sensor_11` (2), `sensor_12` (1), `sensor_17` (1) |
| **FD004** | Validation | $[-0.0008, 1.0104]$ | 4 / 306,072 | **0.001%** | `sensor_3` (1), `sensor_8` (1), `sensor_17` (1), `sensor_20` (1) |
| **FD004** | Test | $[-0.0055, 1.0000]$ | 15 / 989,136 | **0.002%** | `sensor_2` (1), `sensor_3` (2), `sensor_6` (9), `sensor_7` (1), `sensor_11` (1), `sensor_12` (1) |

*Interpretation:* In all cases, fewer than **0.011%** of cells exceed the training $[0, 1]$ interval, and the maximum deviation is less than $0.09$ units. This demonstrates that the Step 3.10 engine-level split preserves excellent distribution alignment while strictly preventing data leakage.

---

## 8. Final Execution Summary Across All Subsets

| Subset | Training Rows | Validation Rows | Test Rows | Train Engines | Val Engines | Test Engines | Features Scaled | Scaler Object Saved | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| **FD001** | 16,340 | 4,291 | 13,096 | 80 | 20 | 100 | 24 | `scaler_FD001_tier3.pkl` | **PASS** |
| **FD002** | 42,889 | 10,870 | 33,991 | 208 | 52 | 259 | 24 | `scaler_FD002_tier3.pkl` | **PASS** |
| **FD003** | 20,177 | 4,543 | 16,596 | 80 | 20 | 100 | 24 | `scaler_FD003_tier3.pkl` | **PASS** |
| **FD004** | 48,496 | 12,753 | 41,214 | 199 | 50 | 248 | 24 | `scaler_FD004_tier3.pkl` | **PASS** |
| **Total** | **127,902** | **32,457** | **104,897** | **567** | **142** | **707** | **24** | **4 Scalers Fitted** | **ALL PASS** |

---

## 9. Status and Next Steps

- **Completed:** Step 3.8 scaling execution, 12-point verification checklist, artifact persistence, and documentation.
- **Leakage Integrity Confirmed:** Scalers fitted strictly on training partitions; validation and test sets transformed without refitting.
- **Next Task:** Awaiting approval before proceeding to **Phase 3 – Step 3.9: Create Time-Window Sequences**.
