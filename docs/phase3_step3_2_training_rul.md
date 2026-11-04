# Phase 3 – Step 3.2: Training Remaining Useful Life (RUL) Calculation & Appending

**Project:** AI-Based Aircraft Engine Health Monitoring, Fault Diagnosis and Predictive Maintenance  
**Dataset:** NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS)  
**Execution Script:** `preprocessing/calculate_training_rul.py`  
**Document Status:** Complete (Step 3.2 Report)  
**Date:** November 2026  

---

## 1. Objective and Overview

Step 3.2 implements the calculation and appending of raw linear Remaining Useful Life (RUL) target labels for all four NASA C-MAPSS training subsets:
- `train_FD001.txt`
- `train_FD002.txt`
- `train_FD003.txt`
- `train_FD004.txt`

The resulting datasets are stored under `dataset/processed/` in standard CSV format, preserving all original 26 operational variables and original row ordering, with exactly one new target column: `RUL`.

---

## 2. Mathematical Formulation & Implementation Approach

### 2.1 Formula
Because the training trajectories represent complete run-to-failure simulations, failure occurs at each engine's maximum observed operational cycle.

For each engine unit $i \in \{1, \dots, N\}$ at cycle $t \in \{1, \dots, t_{\text{max}, i}\}$:

$$RUL(i, t) = t_{\text{max}, i} - t$$

Where:
- $t_{\text{max}, i} = \max_{t} \{ \text{time\_cycles} \mid \text{unit\_number} = i \}$
- $t$ is the current recorded operational cycle.

### 2.2 Implementation Details
The processing was executed via `preprocessing/calculate_training_rul.py`:
1. **Raw Ingestion:** Each raw training file was loaded using whitespace separation without altering any original raw file in `dataset/raw/C-MAPSS/`.
2. **Column Retention:** Standardized column names were assigned to the 26 variables (`unit_number`, `time_cycles`, `op_setting_1` to `op_setting_3`, and `sensor_1` to `sensor_21`).
3. **Target Generation:** The maximum cycle for each `unit_number` group was evaluated via vectorised group transformation (`df.groupby('unit_number')['time_cycles'].transform('max')`), and `RUL` was computed as `(max_cycles - df['time_cycles']).astype(int)`.
4. **Target Preservation:** No RUL clipping or capping was applied; values reflect raw linear RUL. No normalization, scaling, or feature removal was performed.
5. **Output Writing:** Generated datasets were written to `dataset/processed/` with `index=False`.

---

## 3. Generated Files

The generated processed training datasets are located in `dataset/processed/`:

| Subset | Raw Source File | Generated Processed File | File Size (Bytes) |
| :---: | :--- | :--- | :---: |
| **FD001** | `dataset/raw/C-MAPSS/train_FD001.txt` | `dataset/processed/train_FD001_with_RUL.csv` | 3,488,281 |
| **FD002** | `dataset/raw/C-MAPSS/train_FD002.txt` | `dataset/processed/train_FD002_with_RUL.csv` | 8,994,491 |
| **FD003** | `dataset/raw/C-MAPSS/train_FD003.txt` | `dataset/processed/train_FD003_with_RUL.csv` | 4,184,234 |
| **FD004** | `dataset/raw/C-MAPSS/train_FD004.txt` | `dataset/processed/train_FD004_with_RUL.csv` | 10,255,303 |

---

## 4. Subset Statistics & Target Distributions

Across the four training subsets, a total of **160,359 records** across **709 engine units** were processed.

| Subset | Operating Conditions | Fault Modes | Engine Count | Total Rows | Min RUL | Max RUL | Mean RUL | Median RUL | Std RUL |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FD001** | Single (Sea Level) | HPC Degradation | 100 | 20,631 | 0 | 361 | 107.81 | 103.0 | 68.88 |
| **FD002** | Six Conditions | HPC Degradation | 260 | 53,759 | 0 | 377 | 108.15 | 103.0 | 69.18 |
| **FD003** | Single (Sea Level) | HPC & Fan Degradation | 100 | 24,720 | 0 | 524 | 138.08 | 123.0 | 98.85 |
| **FD004** | Six Conditions | HPC & Fan Degradation | 249 | 61,249 | 0 | 542 | 133.31 | 122.0 | 89.78 |
| **Total** | — | — | **709** | **160,359** | — | — | — | — | — |

---

## 5. Verification Protocols and Results

To guarantee dataset integrity, nine automated checks were executed for each subset within `preprocessing/calculate_training_rul.py`. All checks passed.

### 5.1 Verification Checklist by Subset

| Check ID | Verification Item | FD001 | FD002 | FD003 | FD004 | Criteria / Description |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **1** | Row Count Equality | PASS | PASS | PASS | PASS | Processed row count equals raw row count exactly |
| **2** | Column Preservation | PASS | PASS | PASS | PASS | First 26 columns match original names and order |
| **3** | Single New Column | PASS | PASS | PASS | PASS | Exactly 27 columns present; final column is `RUL` |
| **4** | Terminal Cycle $RUL = 0$ | PASS | PASS | PASS | PASS | Every engine's last recorded cycle has $RUL = 0$ |
| **5** | Non-Negative Integers | PASS | PASS | PASS | PASS | All RUL values are $\ge 0$ and of integer type (`int64`) |
| **6** | Monotonic Decrement | PASS | PASS | PASS | PASS | Consecutive step diffs within each unit are strictly $-1$ |
| **7** | No Missing or Infinite | PASS | PASS | PASS | PASS | 0 null/NaN values and 0 infinite values in `RUL` |
| **8** | Initial Cycle $RUL$ Match | PASS | PASS | PASS | PASS | Cycle 1 RUL equals $t_{\text{max}, i} - 1$ for every engine |
| **9** | Engine Count Preserved | PASS | PASS | PASS | PASS | Number of unique `unit_number` matches raw subset |

### 5.2 Verification Details
- **FD001:** 100 engines verified; all 100 engines terminate at $RUL=0$; 20,531 consecutive step diffs evaluated, each exactly $-1$.
- **FD002:** 260 engines verified; all 260 engines terminate at $RUL=0$; 53,499 consecutive step diffs evaluated, each exactly $-1$.
- **FD003:** 100 engines verified; all 100 engines terminate at $RUL=0$; 24,620 consecutive step diffs evaluated, each exactly $-1$.
- **FD004:** 249 engines verified; all 249 engines terminate at $RUL=0$; 61,000 consecutive step diffs evaluated, each exactly $-1$.

---

## 6. Assumptions and Limitations

1. **Target Formulation Convention:**  
   Raw linear RUL assigns a continuously decreasing integer target from cycle 1 to failure ($t_{\text{max}}$). This is a supervised target-labeling convention and does not imply that physical degradation initiates at cycle 1.
2. **Uncapped Values:**  
   In this step, no upper bound or clipping is applied (e.g., maximum RUL values reach 361 in FD001, 377 in FD002, 524 in FD003, and 542 in FD004). Downstream modeling may evaluate piecewise linear capping (e.g., at 125–130 cycles) during feature engineering.
3. **Training Data Exclusivity:**  
   This step pertains exclusively to training subsets (`train_FD00x.txt`). Test subsets (`test_FD00x.txt`) are censored before failure and will be labeled in Step 3.3 by combining observed cycle lengths with NASA's external ground-truth vectors (`RUL_FD00x.txt`).

---

## 7. Status and Next Steps

- **Completed:** Step 3.2 execution, automated verification, and documentation.
- **Raw Files Status:** Untouched in `dataset/raw/C-MAPSS/`.
- **Next Task:** **Phase 3 – Step 3.3: Calculate and Append Test RUL Labels**.
