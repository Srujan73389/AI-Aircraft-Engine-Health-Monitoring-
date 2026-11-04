# Phase 3 – Step 3.5: Data Quality Validation and Missing/Invalid Value Analysis

**Project:** AI-Based Aircraft Engine Health Monitoring, Fault Diagnosis and Predictive Maintenance  
**Dataset:** NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS)  
**Execution Script:** `preprocessing/validate_processed_data.py`  
**Document Status:** Complete (Step 3.5 Quality Report)  
**Date:** November 2026  

---

## 1. Objective

Phase 2 Exploratory Data Analysis (EDA) established that the raw NASA C-MAPSS data files contain no missing values or formatting anomalies. The objective of Step 3.5 is to rigorously audit the eight processed datasets (with appended RUL targets) generated in Steps 3.2 and 3.3 to determine whether any missing, corrupted, infinite, or invalid values exist that would require data cleaning, row removal, or imputation.

Per project guidelines:
- We do not assume cleaning is necessary without direct empirical evidence.
- We do not remove rows unless an actual invalid or corrupted record is identified.
- We do not apply arbitrary zero-filling, mean, or median imputation.
- We preserve the exact row counts, engine order, and target alignments established in prior steps.

---

## 2. Files Inspected

All eight processed CSV files in `dataset/processed/` were audited independently:

### Training Datasets:
1. `dataset/processed/train_FD001_with_RUL.csv`
2. `dataset/processed/train_FD002_with_RUL.csv`
3. `dataset/processed/train_FD003_with_RUL.csv`
4. `dataset/processed/train_FD004_with_RUL.csv`

### Testing Datasets:
5. `dataset/processed/test_FD001_with_RUL.csv`
6. `dataset/processed/test_FD002_with_RUL.csv`
7. `dataset/processed/test_FD003_with_RUL.csv`
8. `dataset/processed/test_FD004_with_RUL.csv`

---

## 3. Auditing Methodology & Checks Performed

The audit was executed via `preprocessing/validate_processed_data.py`, evaluating ten quality and structural dimensions across all 265,256 records:

1. **Dimensional Integrity:** Verified that row counts match the approved benchmark counts from Phase 2, and that each dataset possesses exactly 27 columns.
2. **Schema & Ordering:** Verified that all 26 original features (`unit_number`, `time_cycles`, `op_setting_1`–`3`, `sensor_1`–`21`) are preserved in their exact order, followed by the target column `RUL`.
3. **Missing Value Audit:** Computed cell-level null/NaN counts across all 27 columns (`isna().sum()`).
4. **Infinite Value Audit:** Evaluated whether any numeric cell contains positive infinity (`+inf`) or negative infinity (`-inf`).
5. **Data Type Validity:** Confirmed that key variables (`unit_number`, `time_cycles`, `RUL`) are integer types (`int64`), and that all remaining 24 features are valid numeric types without string/object corruption.
6. **Engine ID Integrity:** Verified that engine units are strictly positive, sequentially numbered $1, 2, \dots, N$ without missing units or numbering gaps.
7. **Cycle Sequence Monotonicity:** Verified for every engine trajectory that cycles begin at $t = 1$ and increment strictly by $+1$ with zero cycle omissions ($t_{k+1} - t_k \equiv 1$).
8. **Physical Feature Finite Bounds:** Checked that all 3 operating settings and 21 sensors contain bounded, finite values.
9. **RUL Target Integrity:**
   - Evaluated that all RUL values are non-negative integers ($\ge 0$).
   - Verified that RUL decrements strictly by $-1$ across consecutive operational cycles.
   - Training: Verified that the terminal cycle of every training engine terminates at $RUL = 0$.
   - Testing: Verified that the terminal recorded cycle of every test engine equals its NASA ground-truth value from `RUL_FD00x.txt`.
10. **Duplicate Record Detection:** Evaluated complete duplicate row detection (`duplicated().sum() == 0`).

---

## 4. Dataset-by-Dataset Audit Results

### 4.1 Testing Subsets

| Dataset | Shape | Nulls | Infs | Duplicates | Engine IDs | Cycle Continuity | Terminal RUL Check | Overall Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **test_FD001_with_RUL.csv** | (13,096, 27) | 0 | 0 | 0 | 100 units (1..100) | 1 to $t_{\text{cutoff}}$ (step +1) | Matches `RUL_FD001.txt` | **PASS** |
| **test_FD002_with_RUL.csv** | (33,991, 27) | 0 | 0 | 0 | 259 units (1..259) | 1 to $t_{\text{cutoff}}$ (step +1) | Matches `RUL_FD002.txt` | **PASS** |
| **test_FD003_with_RUL.csv** | (16,596, 27) | 0 | 0 | 0 | 100 units (1..100) | 1 to $t_{\text{cutoff}}$ (step +1) | Matches `RUL_FD003.txt` | **PASS** |
| **test_FD004_with_RUL.csv** | (41,214, 27) | 0 | 0 | 0 | 248 units (1..248) | 1 to $t_{\text{cutoff}}$ (step +1) | Matches `RUL_FD004.txt` | **PASS** |

### 4.2 Training Subsets

| Dataset | Shape | Nulls | Infs | Duplicates | Engine IDs | Cycle Continuity | Terminal RUL Check | Overall Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **train_FD001_with_RUL.csv** | (20,631, 27) | 0 | 0 | 0 | 100 units (1..100) | 1 to $t_{\text{max}}$ (step +1) | All $RUL(t_{\text{max}}) = 0$ | **PASS** |
| **train_FD002_with_RUL.csv** | (53,759, 27) | 0 | 0 | 0 | 260 units (1..260) | 1 to $t_{\text{max}}$ (step +1) | All $RUL(t_{\text{max}}) = 0$ | **PASS** |
| **train_FD003_with_RUL.csv** | (24,720, 27) | 0 | 0 | 0 | 100 units (1..100) | 1 to $t_{\text{max}}$ (step +1) | All $RUL(t_{\text{max}}) = 0$ | **PASS** |
| **train_FD004_with_RUL.csv** | (61,249, 27) | 0 | 0 | 0 | 249 units (1..249) | 1 to $t_{\text{max}}$ (step +1) | All $RUL(t_{\text{max}}) = 0$ | **PASS** |

---

## 5. Missing and Invalid Value Summary

Across all **265,256 rows** and **27 columns** (totaling **7,161,912 data cells** across the 8 files):

- **Total Missing / NaN Values:** **0**
- **Total Positive / Negative Infinites:** **0**
- **Non-Numeric Corruptions:** **0**
- **Engine ID Discontinuities:** **0**
- **Cycle Count Skips / Gaps:** **0**
- **Negative RUL Targets:** **0**
- **Duplicate Records:** **0**

---

## 6. Cleaning Actions Taken

### Decision: No Cleaning Required
Because all 7,161,912 data cells across all eight datasets passed every quality and integrity check without exception:
1. **Zero rows were removed.**
2. **Zero columns were modified or imputed.**
3. **No zero-filling, mean imputation, or median interpolation was applied.**
4. **All raw sensor values and operating settings were preserved exactly as simulated.**

---

## 7. Row and Column Count Verification (Before vs. After)

Because no rows or columns required cleaning, the dataset dimensions remain identical to the Phase 2 and Phase 3 baselines:

| Dataset File | Approved Prior Row Count | Step 3.5 Audited Row Count | Delta | Column Count |
| :--- | :---: | :---: | :---: | :---: |
| `train_FD001_with_RUL.csv` | 20,631 | 20,631 | 0 | 27 |
| `train_FD002_with_RUL.csv` | 53,759 | 53,759 | 0 | 27 |
| `train_FD003_with_RUL.csv` | 24,720 | 24,720 | 0 | 27 |
| `train_FD004_with_RUL.csv` | 61,249 | 61,249 | 0 | 27 |
| `test_FD001_with_RUL.csv` | 13,096 | 13,096 | 0 | 27 |
| `test_FD002_with_RUL.csv` | 33,991 | 33,991 | 0 | 27 |
| `test_FD003_with_RUL.csv` | 16,596 | 16,596 | 0 | 27 |
| `test_FD004_with_RUL.csv` | 41,214 | 41,214 | 0 | 27 |
| **Total Processed Records** | **265,256** | **265,256** | **0** | **27** |

---

## 8. Final Decision and Technical Boundaries

1. **Clean Baseline Established:** The processed datasets under `dataset/processed/` are confirmed to be structurally sound, completely populated, and fully aligned with ground truth.
2. **Preservation of Raw NASA Files:** Raw files under `dataset/raw/C-MAPSS/` remain untouched.
3. **No Feature Scaling or Selection in Step 3.5:** Feature selection (handling constant / non-informative sensors identified in Phase 2) and normalization/scaling are strictly reserved for downstream steps (Step 3.6+).

---

## 9. Status and Next Steps

- **Completed:** Quality validation script executed and documented; zero defects detected.
- **Next Task:** Awaiting approval before proceeding to **Phase 3 – Step 3.6: Feature Selection & Removal of Non-Informative Sensors**.
