# Phase 3 – Step 3.3: Test Remaining Useful Life (RUL) Calculation & Appending

**Project:** AI-Based Aircraft Engine Health Monitoring, Fault Diagnosis and Predictive Maintenance  
**Dataset:** NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS)  
**Execution Script:** `preprocessing/calculate_test_rul.py`  
**Document Status:** Complete (Step 3.3 Report)  
**Date:** November 2026  

---

## 1. Objective and Overview

Step 3.3 implements the calculation and appending of raw linear Remaining Useful Life (RUL) target labels for all four NASA C-MAPSS testing subsets:
- `test_FD001.txt` with `RUL_FD001.txt`
- `test_FD002.txt` with `RUL_FD002.txt`
- `test_FD003.txt` with `RUL_FD003.txt`
- `test_FD004.txt` with `RUL_FD004.txt`

Unlike training trajectories which run to failure, test trajectories are right-censored (truncated prior to failure). Ground-truth remaining cycles at the truncation cutoff cycle are provided by NASA in separate vector files (`RUL_FD00x.txt`).

The resulting processed datasets are saved in `dataset/processed/` in standard CSV format, preserving all original 26 operational variables and original row ordering, with exactly one new target column: `RUL`.

---

## 2. Mathematical Formulation & Implementation Approach

### 2.1 Formulation
For each test engine $i \in \{1, \dots, N_{\text{test}}\}$ and operational cycle $t \in \{1, \dots, t_{\text{cutoff}, i}\}$:

$$RUL_{\text{test}}(i, t) = RUL_{\text{GT}}(i) + (t_{\text{cutoff}, i} - t)$$

Where:
- $RUL_{\text{GT}}(i)$ is the ground-truth remaining useful life after the final recorded cycle $t_{\text{cutoff}, i}$, loaded from line $i$ of `RUL_FD00x.txt`.
- $t_{\text{cutoff}, i} = \max_{t} \{ \text{time\_cycles} \mid \text{unit\_number} = i \}$ is the terminal recorded cycle of test engine $i$.
- $t$ is the current recorded operational cycle.

### 2.2 Boundary Conditions & Properties
1. **Terminal Recorded Observation ($t = t_{\text{cutoff}, i}$):**
   $$RUL_{\text{test}}(i, t_{\text{cutoff}, i}) = RUL_{\text{GT}}(i) + (t_{\text{cutoff}, i} - t_{\text{cutoff}, i}) = RUL_{\text{GT}}(i)$$
   *At the cutoff cycle, the calculated RUL equals NASA's ground-truth scalar exactly.*
2. **Initial Recorded Observation ($t = 1$):**
   $$RUL_{\text{test}}(i, 1) = RUL_{\text{GT}}(i) + t_{\text{cutoff}, i} - 1$$
3. **Monotonic Stepwise Decrement:**
   $$RUL_{\text{test}}(i, t) - RUL_{\text{test}}(i, t + 1) = 1$$
   *RUL strictly decrements by exactly 1 cycle per elapsed operational cycle.*

### 2.3 Implementation Details
The processing was executed via `preprocessing/calculate_test_rul.py`:
1. **Raw Test Ingestion:** Loaded raw test files using whitespace separation without altering files in `dataset/raw/C-MAPSS/`.
2. **Ground-Truth Ingestion & Alignment:** Read NASA ground-truth RUL vectors. Re-indexed 1-to-$N$ to establish strict 1:1 alignment with 1-based test engine IDs (`unit_number`).
3. **Target Generation:** Evaluated $t_{\text{cutoff}, i}$ per unit via vectorised group transformation (`df.groupby('unit_number')['time_cycles'].transform('max')`), mapped ground truth values, and evaluated $RUL(i, t) = (RUL_{\text{GT}}(i) + (t_{\text{cutoff}, i} - t))$.
4. **Target Integrity:** Enforced `int64` integer type. No clipping, scaling, normalization, or feature selection was applied.
5. **Output Writing:** Generated datasets were written to `dataset/processed/` with `index=False`.

---

## 3. Generated Files

The generated processed test datasets are located in `dataset/processed/`:

| Subset | Raw Input Trajectory | Ground Truth Vector | Generated Processed File | File Size (Bytes) |
| :---: | :--- | :--- | :--- | :---: |
| **FD001** | `dataset/raw/C-MAPSS/test_FD001.txt` | `RUL_FD001.txt` | `dataset/processed/test_FD001_with_RUL.csv` | 2,215,596 |
| **FD002** | `dataset/raw/C-MAPSS/test_FD002.txt` | `RUL_FD002.txt` | `dataset/processed/test_FD002_with_RUL.csv` | 5,689,298 |
| **FD003** | `dataset/raw/C-MAPSS/test_FD003.txt` | `RUL_FD003.txt` | `dataset/processed/test_FD003_with_RUL.csv` | 2,810,002 |
| **FD004** | `dataset/raw/C-MAPSS/test_FD004.txt` | `RUL_FD004.txt` | `dataset/processed/test_FD004_with_RUL.csv` | 6,904,505 |

---

## 4. Subset Statistics & Target Distributions

Across the four test subsets, a total of **104,897 records** across **707 engine units** were processed.

| Subset | Operating Conditions | Fault Modes | Test Engines | GT Values | Total Rows | Min RUL | Max RUL | Mean RUL | Median RUL | Std RUL |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FD001** | Single (Sea Level) | HPC Degradation | 100 | 100 | 13,096 | 7 | 340 | 141.24 | 140.0 | 58.98 |
| **FD002** | Six Conditions | HPC Degradation | 259 | 259 | 33,991 | 6 | 377 | 143.96 | 143.0 | 63.87 |
| **FD003** | Single (Sea Level) | HPC & Fan Degradation | 100 | 100 | 16,596 | 6 | 483 | 169.88 | 159.0 | 84.23 |
| **FD004** | Six Conditions | HPC & Fan Degradation | 248 | 248 | 41,214 | 6 | 553 | 181.90 | 171.0 | 92.17 |
| **Total** | — | — | **707** | **707** | **104,897** | — | — | — | — | — |

*Note: In contrast to the training subsets where minimum RUL is strictly 0 (failure point), the minimum RUL across all test subsets is strictly positive ($\ge 6$), confirming that test trajectories are censored prior to failure.*

---

## 5. Verification Protocols and Results

Ten mandatory verification checks were executed for each test subset within `preprocessing/calculate_test_rul.py`. All checks passed.

### 5.1 Verification Checklist by Subset

| Check ID | Verification Item | FD001 | FD002 | FD003 | FD004 | Criteria / Description |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **1** | Row Count Equality | PASS | PASS | PASS | PASS | Processed row count equals raw row count exactly |
| **2** | Column Preservation | PASS | PASS | PASS | PASS | First 26 columns match original names and order |
| **3** | Single New Column | PASS | PASS | PASS | PASS | Exactly 27 columns present; final column is `RUL` |
| **4** | Engine Count Matches GT | PASS | PASS | PASS | PASS | Test engines == ground-truth RUL lines |
| **5** | Engine ID Alignment | PASS | PASS | PASS | PASS | Engine IDs strictly 1..$N$, matching GT line indices |
| **6** | Terminal Cycle Matches GT | PASS | PASS | PASS | PASS | Calculated RUL at $t_{\text{cutoff}}$ equals NASA GT |
| **7** | Non-Negative Integers | PASS | PASS | PASS | PASS | All RUL values $\ge 0$ and integer type (`int64`) |
| **8** | Monotonic Decrement | PASS | PASS | PASS | PASS | Consecutive step diffs within each unit are strictly $-1$ |
| **9** | No Missing or Infinite | PASS | PASS | PASS | PASS | 0 null/NaN values and 0 infinite values in `RUL` |
| **10** | Initial Cycle $RUL$ Match | PASS | PASS | PASS | PASS | Cycle 1 RUL equals $RUL_{\text{GT}} + t_{\text{cutoff}} - 1$ |

### 5.2 Verification Details
- **FD001:** 100 test engines verified; all 100 match ground-truth at cutoff cycle; 12,996 consecutive step diffs evaluated, each exactly $-1$.
- **FD002:** 259 test engines verified; all 259 match ground-truth at cutoff cycle; 33,732 consecutive step diffs evaluated, each exactly $-1$.
- **FD003:** 100 test engines verified; all 100 match ground-truth at cutoff cycle; 16,496 consecutive step diffs evaluated, each exactly $-1$.
- **FD004:** 248 test engines verified; all 248 match ground-truth at cutoff cycle; 40,966 consecutive step diffs evaluated, each exactly $-1$.

---

## 6. Sample Rows: Engine 1 Across All Test Subsets

### FD001 – Unit 1 ($t_{\text{cutoff}} = 31$, $RUL_{\text{GT}} = 112$)
```
First 3 cycles:
 unit_number  time_cycles  op_setting_1  sensor_1  sensor_2  sensor_21  RUL
           1            1        0.0023    518.67    643.02    23.3735  142
           1            2       -0.0027    518.67    641.71    23.3916  141
           1            3        0.0003    518.67    642.46    23.4166  140
Final 3 cycles (approaching cutoff):
 unit_number  time_cycles  op_setting_1  sensor_1  sensor_2  sensor_21  RUL
           1           29        0.0014    518.67    641.95    23.3621  114
           1           30       -0.0025    518.67    642.79    23.4069  113
           1           31       -0.0006    518.67    642.58    23.3552  112
```

### FD002 – Unit 1 ($t_{\text{cutoff}} = 258$, $RUL_{\text{GT}} = 18$)
```
First 3 cycles:
 unit_number  time_cycles  op_setting_1  sensor_1  sensor_2  sensor_21  RUL
           1            1        9.9987    489.05    605.03    17.1551  275
           1            2       20.0026    491.19    607.82    14.8039  274
           1            3       35.0045    449.44    556.00     8.9125  273
Final 3 cycles (approaching cutoff):
 unit_number  time_cycles  op_setting_1  sensor_1  sensor_2  sensor_21  RUL
           1          256       25.0054    462.54    537.52     8.5564   20
           1          257       42.0048    445.00    550.26     6.2911   19
           1          258       10.0076    489.05    605.42    17.0934   18
```

### FD003 – Unit 1 ($t_{\text{cutoff}} = 233$, $RUL_{\text{GT}} = 44$)
```
First 3 cycles:
 unit_number  time_cycles  op_setting_1  sensor_1  sensor_2  sensor_21  RUL
           1            1       -0.0017    518.67    641.94    23.4468  276
           1            2        0.0006    518.67    642.02    23.4807  275
           1            3        0.0014    518.67    641.68    23.4244  274
Final 3 cycles (approaching cutoff):
 unit_number  time_cycles  op_setting_1  sensor_1  sensor_2  sensor_21  RUL
           1          231       -0.0002    518.67    642.74    23.6160   46
           1          232        0.0038    518.67    642.13    23.5743   45
           1          233        0.0021    518.67    642.59    23.5679   44
```

### FD004 – Unit 1 ($t_{\text{cutoff}} = 230$, $RUL_{\text{GT}} = 22$)
```
First 3 cycles:
 unit_number  time_cycles  op_setting_1  sensor_1  sensor_2  sensor_21  RUL
           1            1       20.0072    491.19    606.67    14.7007  251
           1            2       24.9984    462.54    536.22     8.5748  250
           1            3       42.0000    445.00    549.23     6.4365  249
Final 3 cycles (approaching cutoff):
 unit_number  time_cycles  op_setting_1  sensor_1  sensor_2  sensor_21  RUL
           1          228       42.0010    445.00    549.45     6.5274   24
           1          229       42.0055    445.00    549.45     6.3622   23
           1          230       25.0070    462.54    537.66     8.6329   22
```

---

## 7. Assumptions and Technical Boundaries

1. **Target Formulation Convention:**  
   Raw linear RUL assigns a linearly decreasing target throughout each engine's observed life. This is a target-labeling convention and does not imply that physical degradation initiates at cycle 1.
2. **Ground-Truth Source of Truth:**  
   The NASA competition ground-truth vectors (`RUL_FD00x.txt`) are taken as the authoritative physical baseline for remaining flight cycles at cutoff.
3. **No Target Capping or Scaling:**  
   No piecewise capping, min-max scaling, standardization, or feature selection was applied. These decisions belong to downstream feature preprocessing (Step 3.4+).

---

## 8. Status and Next Steps

- **Completed:** Step 3.3 execution, automated verification, and documentation.
- **Processed Datasets Ready:** Both training and test subsets for FD001–FD004 now possess verified linear RUL labels.
- **Next Task:** Awaiting approval before proceeding to **Phase 3 – Step 3.4**.
