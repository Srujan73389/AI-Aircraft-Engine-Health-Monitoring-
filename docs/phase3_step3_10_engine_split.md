# Phase 3 – Step 3.10: Engine-Level Training & Validation Partitioning

**Project:** AI-Based Aircraft Engine Health Monitoring, Fault Diagnosis and Predictive Maintenance  
**Dataset:** NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS)  
**Execution Script:** `preprocessing/split_engines.py`  
**Partition Manifest:** `dataset/processed/train_val_split_manifest.json`  
**Document Status:** Complete (Step 3.10 Specification & Verification)  
**Date:** November 2026  

---

## 1. Objective and Overview

Supervised model training and hyperparameter optimization require an internal validation holdout set to evaluate generalization error without touching the final test sets (`test_FD00x_with_RUL.csv`). 

In prognostic health management (PHM), standard row-level random splitting creates catastrophic data leakage. Step 3.10 implements a strictly **engine-level training and validation split** across all four C-MAPSS training datasets:
- `train_FD001_with_RUL.csv` $\rightarrow$ `train_FD001_split.csv` and `validation_FD001_split.csv`
- `train_FD002_with_RUL.csv` $\rightarrow$ `train_FD002_split.csv` and `validation_FD002_split.csv`
- `train_FD003_with_RUL.csv` $\rightarrow$ `train_FD003_split.csv` and `validation_FD003_split.csv`
- `train_FD004_with_RUL.csv` $\rightarrow$ `train_FD004_split.csv` and `validation_FD004_split.csv`

---

## 2. Why Engine-Level Splitting is Mandatory (Preventing Trajectory Leakage)

### 2.1 The Mechanism of Row-Level Data Leakage
In standard cross-sectional machine learning (e.g., tabular classification on independent customer records), random sampling of rows is standard practice. However, C-MAPSS data comprises **correlated multivariate time series** where each row represents a discrete flight cycle $t$ within an engine's continuous physical degradation lifespan $t \in [1, t_{\text{max}}]$.

If rows are sampled randomly:
- Cycle $t = 150$ of Engine #1 would be placed in the training set.
- Cycle $t = 151$ (or cycle $t = 149$) of Engine #1 would be placed in the validation set.
- Because physical sensor measurements (temperatures, pressures, fan speeds) change incrementally across consecutive flight cycles, the feature vector at cycle $151$ is nearly identical to cycle $150$ plus random sensor noise.
- The machine learning model merely memorizes the specific baseline offset and trajectory profile of Engine #1, achieving deceptively near-zero validation error.
- When subsequently deployed on unseen test engines (`test_FD00x`), the model fails completely because it never learned generalizable degradation dynamics across different engines.

```mermaid
flowchart TD
    subgraph Flawed Row-Level Split (Catastrophic Leakage)
        E1_c1["Engine 1, Cycle 100\n(Placed in Train)"] -. "Correlated Sensor Snapshot" .-> E1_c2["Engine 1, Cycle 101\n(Placed in Validation)"]
    end
    subgraph Correct Engine-Level Split (Leakage-Free)
        TrainEngines["Training Partition (80% Engines)\nAll Cycles of Engine 1, 4, 5, 6, 7, ..."]
        ValEngines["Validation Partition (20% Engines)\nAll Cycles of Engine 2, 3, 15, 21, ..."]
        TrainEngines -. "Zero Engine Overlap" .-> ValEngines
    end
```

### 2.2 Methodological Mandates Enforced
1. **Engine Grouping:** Splitting is performed strictly on unique `unit_number` identifiers.
2. **Whole-Trajectory Allocation:** An engine's entire operational trajectory ($t = 1$ to $t_{\text{max}}$) is assigned exclusively to either the training set or the validation set.
3. **Temporal Ordering:** Chronological cycle ordering within each engine is preserved without shuffling ($t_{k+1} - t_k \equiv 1$).
4. **Target Alignment:** Raw linear RUL labels calculated in Step 3.2 are retained without modification.

---

## 3. Split Configuration and Reproducibility

- **Target Split Ratio:** **80% Training Engines / 20% Validation Engines**.
- **Random Seed:** **`42`** (ensuring exact mathematical reproducibility).
- **Execution Mechanism:** For each subset, the sorted array of engine IDs is shuffled using a dedicated random number generator (`np.random.RandomState(42)`) and partitioned at $\text{round}(0.80 \times N_{\text{engines}})$.
- **Audit Manifest:** All engine assignments, row counts, and percentages are saved in machine-readable JSON format at [`dataset/processed/train_val_split_manifest.json`](file:///c:/Users/LENOVO/Documents/Aircraft_Engine_AI/dataset/processed/train_val_split_manifest.json).

---

## 4. Partitioning Summary & Dimensions

Across all four subsets, a total of **567 engines (127,902 rows)** were assigned to training, and **142 engines (32,457 rows)** were assigned to validation:

| Subset | Operating Conditions | Fault Modes | Total Engines | Train Engines (80%) | Val Engines (20%) | Total Rows | Train Rows | Val Rows | Actual Row Split (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FD001** | Single (Sea Level) | HPC Degradation | 100 | **80** | **20** | 20,631 | 16,340 | 4,291 | 79.20% / 20.80% |
| **FD002** | Six Regimes | HPC Degradation | 260 | **208** | **52** | 53,759 | 42,889 | 10,870 | 79.78% / 20.22% |
| **FD003** | Single (Sea Level) | HPC & Fan Degradation | 100 | **80** | **20** | 24,720 | 20,177 | 4,543 | 81.62% / 18.38% |
| **FD004** | Six Regimes | HPC & Fan Degradation | 249 | **199** | **50** | 61,249 | 48,496 | 12,753 | 79.18% / 20.82% |
| **Total** | — | — | **709** | **567** | **142** | **160,359** | **127,902** | **32,457** | **79.76% / 20.24%** |

*Note: Minor deviations of row percentages from 80.00% occur naturally because individual engine lifespans vary (e.g., engines in validation happen to have slightly longer or shorter mean run-to-failure cycles).*

---

## 5. Verification Checklist & Audit Results

Every subset underwent seven automated verification checks in `preprocessing/split_engines.py`:

| Check ID | Verification Item | FD001 | FD002 | FD003 | FD004 | Criteria |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **1** | Zero Engine Overlap | PASS | PASS | PASS | PASS | $\text{TrainEngines} \cap \text{ValEngines} \equiv \emptyset$ |
| **2** | All Engines Accounted | PASS | PASS | PASS | PASS | $\text{TrainEngines} \cup \text{ValEngines} \equiv \text{OrigEngines}$ |
| **3** | Exact Row Count Sum | PASS | PASS | PASS | PASS | $N_{\text{train\_rows}} + N_{\text{val\_rows}} == N_{\text{orig\_rows}}$ |
| **4** | Zero Duplicate Rows | PASS | PASS | PASS | PASS | 0 duplicate records in train or val partitions |
| **5** | Column Schema Preserved | PASS | PASS | PASS | PASS | All 27 columns match original names and order |
| **6** | Monotonic Cycle Ordering | PASS | PASS | PASS | PASS | $t_{k+1} - t_k \equiv 1$ for every engine trajectory |
| **7** | RUL Target Validity | PASS | PASS | PASS | PASS | Terminal RUL is strictly 0; all RUL $\ge 0$ |

---

## 6. Generated Split Files

The partitioned datasets are saved under `dataset/processed/`:

| Subset | Partition | Output File | Rows | Engines | File Size (Bytes) |
| :---: | :---: | :--- | :---: | :---: | :---: |
| **FD001** | Train | `dataset/processed/train_FD001_split.csv` | 16,340 | 80 | 2,762,746 |
| **FD001** | Validation | `dataset/processed/validation_FD001_split.csv` | 4,291 | 20 | 725,804 |
| **FD002** | Train | `dataset/processed/train_FD002_split.csv` | 42,889 | 208 | 7,174,858 |
| **FD002** | Validation | `dataset/processed/validation_FD002_split.csv` | 10,870 | 52 | 1,819,902 |
| **FD003** | Train | `dataset/processed/train_FD003_split.csv` | 20,177 | 80 | 3,415,535 |
| **FD003** | Validation | `dataset/processed/validation_FD003_split.csv` | 4,543 | 20 | 768,968 |
| **FD004** | Train | `dataset/processed/train_FD004_split.csv` | 48,496 | 199 | 8,118,789 |
| **FD004** | Validation | `dataset/processed/validation_FD004_split.csv` | 12,753 | 50 | 2,136,783 |
| **All** | Manifest | `dataset/processed/train_val_split_manifest.json` | — | 709 | 10,119 |

---

## 7. Downstream Implications for Scaler Fitting and Modeling

Now that the engine-level split is established:
1. **Scaler Fitting (Step 3.8 / Modeling Pipelines):** Feature scalers (`MinMaxScaler`, `StandardScaler`) can now be fitted **strictly on `train_FD00x_split.csv`**. The fitted scalers will then transform `validation_FD00x_split.csv` and `test_FD00x_with_RUL.csv` without leaking validation statistics.
2. **Early Stopping & Hyperparameter Tuning:** Validation partitions (`validation_FD00x_split.csv`) will be used during Phase 4 to monitor validation loss, trigger early stopping, and optimize hyperparameters without touching final test benchmarks.

---

## 8. Limitations of Single Holdout Partitioning

1. **Holdout Variance:** Although 80/20 is a standard benchmark partition, performance metrics on a single validation set are subject to the specific distribution of engines sampled (e.g. if the validation set happens to contain engines with slightly higher baseline wear).
2. **Cross-Validation Alternative:** For sensitivity testing, Group K-Fold cross-validation (grouping by `unit_number`) can be evaluated during Phase 4 to compute mean and variance across multiple folds if necessary.

---

## 9. Status and Next Steps

- **Completed:** Step 3.10 execution, automated verification, manifest generation, and documentation.
- **Dataset Immutability Confirmed:** Original processed datasets (`train_FD00x_with_RUL.csv`, `test_FD00x_with_RUL.csv`) and raw files in `dataset/raw/C-MAPSS/` remain untouched.
- **Next Task:** Awaiting approval before proceeding to the next scheduled roadmap step.
