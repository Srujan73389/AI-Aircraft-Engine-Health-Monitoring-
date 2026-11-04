# Phase 3 – Step 3.9: Create Time-Window Sequences

## 1. Executive Summary

In **Phase 3 – Step 3.9**, the normalized and scaled NASA C-MAPSS datasets produced in Step 3.8 were converted into structured 3D sliding-window time-series tensors (`[samples, window_size, features]`). These tensors serve as the direct input representations for temporal deep learning architectures, including **Long Short-Term Memory (LSTM)**, **Gated Recurrent Units (GRU)**, and **1D Convolutional Neural Networks (1D-CNN)**.

All sequence generation was executed strictly within engine boundaries to guarantee zero cross-engine leakage and complete chronological fidelity.

### Key Specifications & Summary Metrics
- **Window Size ($W$):** 30 operating cycles.
- **Stride ($S$):** 1 cycle (dense sliding window).
- **Feature Set:** Tier 3 Extended (24 features: 3 operational settings + 21 sensor measurements).
- **Target ($y$):** Remaining Useful Life (RUL) at the final observation cycle ($t_{\text{end}}$) of each sliding window.
- **Total Generated Windows Across All 4 Subsets:**
  - **Training Windows:** 111,459 sliding windows (across 567 training engines)
  - **Validation Windows:** 28,339 sliding windows (across 142 validation engines)
  - **Test Full Sliding Windows:** 84,495 sliding windows (across 707 test engines)
  - **Test Final Evaluation Windows:** 707 single-instance evaluation windows (exactly 1 per test engine)
- **Verification Result:** 100% PASS rate across all 12 mandatory quality and data-integrity checks.

---

## 2. Temporal Representation & Deep Learning Rationale

Standard feed-forward regression algorithms (e.g., Ridge Regression, Random Forest, XGBoost) evaluate individual operational cycles as independent, static feature vectors. In contrast, turbofan engine degradation is an accumulative, dynamic physical process:
1. **Dynamic Degradation Trajectories:** Wear and tear manifest as progressive drifts in thermodynamic efficiency, vibration, and thermal margins over tens of cycles rather than instantaneous point anomalies.
2. **Operational Setting Transitions:** In multi-condition regimes (FD002 and FD004), a single sensor value is ambiguous without the historical context of altitude, Mach number, and throttle angle transitions over preceding operating cycles.
3. **Temporal Deep Learning Primitives:** Recurrent networks (LSTM, GRU) maintain internal hidden states that capture rates of change ($\frac{ds}{dt}$) and historical trend trajectories across consecutive cycles.

Generating 3D tensors of shape $(N, 30, 24)$ equips temporal models with 30 consecutive operational cycles of 24 synchronized physical parameters to predict the precise remaining engine life.

---

## 3. Trajectory Length Analysis & Short-Engine Padding Strategy

### 3.1 Empirical Trajectory Length Distribution

Prior to sequence creation, trajectory lengths across all partitions were analyzed to evaluate feasibility against candidate window sizes:

| Subset | Partition | Engine Count | Min Length | Max Length | Mean Length | Engines with $L < 30$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **FD001** | Train | 80 | 128 | 362 | 206.3 | 0 (0.0%) |
| **FD001** | Validation | 20 | 141 | 303 | 206.5 | 0 (0.0%) |
| **FD001** | Test | 100 | 31 | 303 | 130.9 | 0 (0.0%) |
| **FD002** | Train | 208 | 128 | 378 | 206.0 | 0 (0.0%) |
| **FD002** | Validation | 52 | 141 | 367 | 207.2 | 0 (0.0%) |
| **FD002** | Test | 259 | **21** | 367 | 131.1 | **6 (2.3%)** |
| **FD003** | Train | 80 | 145 | 525 | 247.2 | 0 (0.0%) |
| **FD003** | Validation | 20 | 153 | 439 | 247.3 | 0 (0.0%) |
| **FD003** | Test | 100 | 38 | 473 | 166.0 | 0 (0.0%) |
| **FD004** | Train | 199 | 128 | 543 | 244.8 | 0 (0.0%) |
| **FD004** | Validation | 50 | 149 | 486 | 245.8 | 0 (0.0%) |
| **FD004** | Test | 248 | **19** | 486 | 160.0 | **11 (4.4%)** |

### 3.2 Analysis of Short Test Engines ($L < 30$)
All 709 training and validation engines have at least 128 cycles, far exceeding $W=30$. However, in the test sets:
- **FD002 Test:** 6 engines (Unit IDs: 28, 59, 83, 122, 201, 223) have trajectory lengths of 23, 22, 21, 29, 25, and 29 cycles.
- **FD004 Test:** 11 engines (Unit IDs: 10, 19, 28, 125, 141, 156, 164, 204, 229, 239, 246) have trajectory lengths of 23, 24, 20, 21, 19, 28, 29, 19, 20, 28, and 29 cycles.

### 3.3 Strategy Formulation & Comparison

Three options were evaluated for handling engines with $L < 30$:

| Option | Implementation | Advantage | Disadvantage | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **Option A: Truncate / Discard** | Exclude test engines with $L < 30$ | Simple implementation | Fails benchmark compliance; discards 6 engines in FD002 and 11 engines in FD004, corrupting official NASA test evaluation | **REJECTED** |
| **Option B: Zero Padding** | Prepend zeros to reach length 30 | Fixed dimension | Introduces unnatural artificial sensor values (e.g. 0K temperature, 0 psia pressure) outside scaled operational ranges | **REJECTED** |
| **Option C: Left-Edge (Replication) Padding** | Repeat the earliest observed healthy state ($t=1$) backwards for $30 - L$ cycles | Physically realistic (engine was operating in a healthy state prior to $t=1$); preserves exact benchmark target and maintains 100% test engine coverage | Minor artificial state persistence during early life | **APPROVED & IMPLEMENTED** |

**Implementation Rule:** For engines where $L < 30$, the feature vector at initial observed cycle $t=1$ is replicated $30 - L$ times at the beginning of the sequence. The target RUL is strictly assigned to the actual ground-truth RUL at cycle $L$. Every padded window is explicitly flagged in the metadata (`is_padded=True`, `padded_cycles_count=30-L`).

---

## 4. Window Size ($W=30$) and Stride ($S=1$) Technical Rationale

1. **Window Size ($W=30$):**
   - **Temporal History:** 30 cycles is sufficient to observe trends in thermodynamic deterioration, thermal creep, and compressor degradation across single and multi-operating regimes.
   - **Coverage Preservation:** With $W=30$, 98.6% of test engines have full native sequences without padding (only 17 out of 707 test engines across all 4 datasets required edge-padding).
   - **Architectural Benchmark Alignment:** 30 cycles is the standard benchmark sequence length in peer-reviewed C-MAPSS literature (Heimes, 2008; Zheng et al., 2017; Li et al., 2018), facilitating direct performance benchmarking.
2. **Stride ($S=1$):**
   - **Sample Efficiency:** Dense sliding with stride 1 generates the maximal number of training sequences from available run-to-failure cycles (111,459 training samples across all subsets).
   - **Continuous Surveillance:** In operational deployment, health monitoring models evaluate telemetry after every completed flight cycle ($S=1$).

---

## 5. Sequence Generation Mechanics & Target Alignment

### 5.1 Per-Engine Boundary Isolation
To strictly avoid data contamination across different engines, sliding windows are constructed per engine group (`unit_number`):
```python
for unit_id, group in df.groupby("unit_number"):
    grp_sorted = group.sort_values("time_cycles")
    feats = grp_sorted[feature_cols].values.astype(np.float32)
    ruls = grp_sorted["RUL"].values.astype(np.float32)
    # Windows extracted exclusively from grp_sorted
```
Under no circumstances does a window combine telemetry from two distinct physical units.

### 5.2 Target Alignment Definition
For a window spanning cycles $[t, t+1, \dots, t+W-1]$, the target RUL is defined as the RUL at the **final observation cycle** of that window:
$$\text{Target}(X_{t:t+W-1}) = \text{RUL}(t + W - 1)$$

This formulation enforces causal prediction: given telemetry up through cycle $t+W-1$, the model predicts how many cycles remain from that current moment.

### 5.3 Test Set Dual Representation
For the test datasets (`test_FD00x`), two sets of sequence tensors were generated:
1. **Full Sliding Windows (`X_test`, `y_test`):** Dense stride-1 sequences across the entire observed test trajectories (used for trajectory monitoring and multi-step evaluation).
2. **Final Evaluation Windows (`X_test_final`, `y_test_final`):** Exactly one window per test engine representing the **last $W=30$ observed cycles** prior to sensor cutoff. This directly matches the NASA ground-truth RUL benchmark evaluation format.

---

## 6. Sequence Dataset Inventory & Shape Summary

All sequences and metadata were generated and saved under `dataset/processed/sequences/`.

### 6.1 Complete Sequence Inventory Table

| Subset | Partition | Engine Count | Window Count | Window Shape $(N, W, K)$ | Target Shape $(N,)$ | Target RUL Min | Target RUL Max | Target RUL Mean | X File | y File |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **FD001** | Train | 80 | 14,020 | $(14020, 30, 24)$ | $(14020,)$ | 0.0 | 332.0 | 93.3 | `X_train_FD001.npy` | `y_train_FD001.npy` |
| **FD001** | Validation | 20 | 3,711 | $(3711, 30, 24)$ | $(3711,)$ | 0.0 | 273.0 | 93.3 | `X_validation_FD001.npy` | `y_validation_FD001.npy` |
| **FD001** | Test | 100 | 10,196 | $(10196, 30, 24)$ | $(10196,)$ | 7.0 | 373.0 | 129.5 | `X_test_FD001.npy` | `y_test_FD001.npy` |
| **FD001** | Test Final | **100** | **100** | $(100, 30, 24)$ | $(100,)$ | 7.0 | 145.0 | 78.4 | `X_test_final_FD001.npy` | `y_test_final_FD001.npy` |
| **FD002** | Train | 208 | 36,857 | $(36857, 30, 24)$ | $(36857,)$ | 0.0 | 348.0 | 93.2 | `X_train_FD002.npy` | `y_train_FD002.npy` |
| **FD002** | Validation | 52 | 9,362 | $(9362, 30, 24)$ | $(9362,)$ | 0.0 | 337.0 | 94.4 | `X_validation_FD002.npy` | `y_validation_FD002.npy` |
| **FD002** | Test | 259 | 26,511 | $(26511, 30, 24)$ | $(26511,)$ | 8.0 | 382.0 | 126.7 | `X_test_FD002.npy` | `y_test_FD002.npy` |
| **FD002** | Test Final | **259** | **259** | $(259, 30, 24)$ | $(259,)$ | 8.0 | 148.0 | 79.8 | `X_test_final_FD002.npy` | `y_test_final_FD002.npy` |
| **FD003** | Train | 80 | 17,857 | $(17857, 30, 24)$ | $(17857,)$ | 0.0 | 495.0 | 114.7 | `X_train_FD003.npy` | `y_train_FD003.npy` |
| **FD003** | Validation | 20 | 3,963 | $(3963, 30, 24)$ | $(3963,)$ | 0.0 | 409.0 | 113.6 | `X_validation_FD003.npy` | `y_validation_FD003.npy` |
| **FD003** | Test | 100 | 13,696 | $(13696, 30, 24)$ | $(13696,)$ | 11.0 | 509.0 | 164.7 | `X_test_FD003.npy` | `y_test_FD003.npy` |
| **FD003** | Test Final | **100** | **100** | $(100, 30, 24)$ | $(100,)$ | 11.0 | 148.0 | 76.5 | `X_test_final_FD003.npy` | `y_test_final_FD003.npy` |
| **FD004** | Train | 199 | 42,725 | $(42725, 30, 24)$ | $(42725,)$ | 0.0 | 513.0 | 112.5 | `X_train_FD004.npy` | `y_train_FD004.npy` |
| **FD004** | Validation | 50 | 11,303 | $(11303, 30, 24)$ | $(11303,)$ | 0.0 | 456.0 | 114.2 | `X_validation_FD004.npy` | `y_validation_FD004.npy` |
| **FD004** | Test | 248 | 34,092 | $(34092, 30, 24)$ | $(34092,)$ | 8.0 | 502.0 | 154.5 | `X_test_FD004.npy` | `y_test_FD004.npy` |
| **FD004** | Test Final | **248** | **248** | $(248, 30, 24)$ | $(248,)$ | 8.0 | 147.0 | 78.4 | `X_test_final_FD004.npy` | `y_test_final_FD004.npy` |

---

## 7. Metadata Traceability Architecture

Every sequence array is paired with an exact row-aligned metadata CSV file containing tracing information for debugging, error analysis, and evaluation:

### 7.1 Sliding Window Metadata Schema (`train_metadata_*.csv`, `validation_metadata_*.csv`, `test_metadata_*.csv`)
- `window_idx`: Sequential integer index corresponding to slice index `X[window_idx]`.
- `unit_number`: Unique identifier of the engine.
- `start_cycle`: Operational cycle corresponding to the first time step in the window ($t_{\text{start}}$).
- `end_cycle`: Operational cycle corresponding to the final time step in the window ($t_{\text{end}}$).
- `is_padded`: Boolean flag indicating whether left-edge replication padding was applied.
- `padded_cycles_count`: Number of replicated initial cycles prepended to reach $W=30$ (0 for unpadded windows).
- `target_rul`: Exact target RUL value corresponding to `y[window_idx]`.

### 7.2 Final Test Window Metadata Schema (`test_final_metadata_*.csv`)
- `unit_number`: Unique engine unit ID.
- `trajectory_length`: Total observed operating cycles for this engine prior to cutoff.
- `start_cycle`: Start cycle of the final window ($\max(1, L - 29)$).
- `end_cycle`: Final cycle ($L$).
- `is_padded`: Boolean flag indicating whether left-edge replication was applied.
- `padded_cycles_count`: Count of replicated cycles ($30 - L$ if $L < 30$, else 0).
- `target_rul`: Ground-truth benchmark RUL at cutoff.

---

## 8. Verification Results (12-Point Checklist)

All 12 automated verification checks were executed programmatically during generation across all four subsets and splits:

| Check # | Verification Item | Specification / Requirement | Audit Result | Status |
| :---: | :--- | :--- | :--- | :---: |
| **1** | **X Tensor Dimensions** | Exact shape $(N, 30, 24)$ for all arrays | Matched exactly across all splits | **PASS** |
| **2** | **y Target Dimensions** | Exact 1D vector $(N,)$ aligned with window count | Matched exactly ($N$ targets) | **PASS** |
| **3** | **Target Value Alignment** | $y[i] == \text{RUL}(t_{\text{end}})$ | Exact mathematical match ($0.0$ residual) | **PASS** |
| **4** | **Engine Unit Purity** | Each window contains telemetry from exactly 1 engine | Zero multi-engine windows | **PASS** |
| **5** | **Chronological Continuity** | Window cycles are strictly contiguous $[t, t+1, \dots, t+W-1]$ | Contiguity verified on sorted cycle indices | **PASS** |
| **6** | **Boundary Isolation** | Windows never overlap across distinct engine boundaries | Groupby generator isolates each engine | **PASS** |
| **7** | **Data Cleanliness** | 0 NaN, 0 Infinite, 0 null values in $X$ and $y$ | Total NaNs = 0, Infs = 0 | **PASS** |
| **8** | **Feature Order Integrity** | Exact consistent ordering of 24 Tier 3 features | Verified matching `scaling_pipeline.py` | **PASS** |
| **9** | **Metadata Alignment** | Metadata row count strictly equals tensor samples $N$ | Lengths match 100% | **PASS** |
| **10** | **Engine Completeness** | All engines in original partition represented | 100% engine representation | **PASS** |
| **11** | **Final Test Coverage** | Exactly one final window per test engine ($N_{\text{final}} == N_{\text{engines}}$) | FD001: 100, FD002: 259, FD003: 100, FD004: 248 | **PASS** |
| **12** | **Source Immutability** | Scaled CSVs and raw data left unmodified | Read-only mode confirmed | **PASS** |

---

## 9. File Deliverables Summary

The following 49 files are created and stored under `dataset/processed/sequences/`:

### Configuration Manifest
- [`sequence_configuration.json`](file:///c:/Users/LENOVO/Documents/Aircraft_Engine_AI/dataset/processed/sequences/sequence_configuration.json)

### NumPy Tensor Arrays
- **Training Arrays:**
  - `X_train_FD001.npy`, `y_train_FD001.npy`
  - `X_train_FD002.npy`, `y_train_FD002.npy`
  - `X_train_FD003.npy`, `y_train_FD003.npy`
  - `X_train_FD004.npy`, `y_train_FD004.npy`
- **Validation Arrays:**
  - `X_validation_FD001.npy`, `y_validation_FD001.npy`
  - `X_validation_FD002.npy`, `y_validation_FD002.npy`
  - `X_validation_FD003.npy`, `y_validation_FD003.npy`
  - `X_validation_FD004.npy`, `y_validation_FD004.npy`
- **Test Full Sliding Arrays:**
  - `X_test_FD001.npy`, `y_test_FD001.npy`
  - `X_test_FD002.npy`, `y_test_FD002.npy`
  - `X_test_FD003.npy`, `y_test_FD003.npy`
  - `X_test_FD004.npy`, `y_test_FD004.npy`
- **Test Final Evaluation Arrays:**
  - `X_test_final_FD001.npy`, `y_test_final_FD001.npy`
  - `X_test_final_FD002.npy`, `y_test_final_FD002.npy`
  - `X_test_final_FD003.npy`, `y_test_final_FD003.npy`
  - `X_test_final_FD004.npy`, `y_test_final_FD004.npy`

### Traceability Metadata CSV Files
- **Training:** `train_metadata_FD001.csv`, `train_metadata_FD002.csv`, `train_metadata_FD003.csv`, `train_metadata_FD004.csv`
- **Validation:** `validation_metadata_FD001.csv`, `validation_metadata_FD002.csv`, `validation_metadata_FD003.csv`, `validation_metadata_FD004.csv`
- **Test Full:** `test_metadata_FD001.csv`, `test_metadata_FD002.csv`, `test_metadata_FD003.csv`, `test_metadata_FD004.csv`
- **Test Final:** `test_final_metadata_FD001.csv`, `test_final_metadata_FD002.csv`, `test_final_metadata_FD003.csv`, `test_final_metadata_FD004.csv`

---

## 10. Conclusion & Next Phase Readiness

Phase 3 is now complete across all steps:
- **Step 3.1:** RUL target definition and degradation models documented.
- **Step 3.2:** Training RUL labels computed and validated.
- **Step 3.3:** Test RUL labels computed and aligned with NASA ground-truth values.
- **Step 3.4:** RUL capping evaluated (125 cycles approved, raw baseline retained).
- **Step 3.5:** Data quality validated (0 missing values, 0 duplicate timestamps, monotonic cycles verified).
- **Step 3.6:** Feature selection completed across 3 tiers (14, 17, and 24 features).
- **Step 3.7:** Scaling strategy formulated (MinMaxScaler, no artificial clipping).
- **Step 3.10:** Engine-level train/validation split executed (80/20, seed 42).
- **Step 3.8:** Feature scaling fitted on training engines only and applied to all partitions without data leakage.
- **Step 3.9:** Sliding-window sequences created and saved as 3D NumPy arrays for deep learning.

The pipeline is now primed for **Phase 4: Model Development and Evaluation**.
