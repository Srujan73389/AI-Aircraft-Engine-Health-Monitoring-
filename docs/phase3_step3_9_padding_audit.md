# Phase 3 – Step 3.9: Padding Audit Report

## 1. Executive Overview

This audit reviews the temporal sequence preprocessing implemented in **Phase 3 – Step 3.9**, focusing specifically on the **left-edge replication padding** applied to test engines with fewer than 30 observations ($L < 30$).

The audit was performed on the existing generated arrays, metadata files, and source code without modifying any files, scalers, models, or datasets.

### Key Audit Findings
1. **Affected Engines:** Exactly **17 out of 707 test engines (2.40%)** have fewer than 30 observations (6 in `FD002`, 11 in `FD004`). All 709 training and validation engines have $\ge 128$ cycles and required zero padding.
2. **Artificial Observations Added:** Between 1 and 11 cycles are prepended per affected engine ($1 \le 30 - L \le 11$).
3. **Data Integrity:** Chronological order of original observations is strictly preserved.
4. **Target Alignment:** The final target RUL for every padded window exactly equals the official NASA ground-truth RUL at the cutoff cycle ($0.0$ residual).
5. **Coverage:** 100% of test engines (707 of 707) remain represented in the evaluation set.

---

## 2. Inventory of Affected Engines and Trajectory Lengths

The 17 test engines with $L < 30$ are detailed below:

| Subset | Unit ID | Trajectory Length ($L$) | Window Size ($W$) | Artificial Cycles Added ($30 - L$) | Fraction Natural Data | Ground-Truth Target RUL | Status in Metadata |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FD002** | 28 | 23 | 30 | 7 | 76.7% | 138.0 | `is_padded=True` |
| **FD002** | 59 | 22 | 30 | 8 | 73.3% | 178.0 | `is_padded=True` |
| **FD002** | 83 | 21 | 30 | 9 | 70.0% | 146.0 | `is_padded=True` |
| **FD002** | 122 | 29 | 30 | 1 | 96.7% | 145.0 | `is_padded=True` |
| **FD002** | 201 | 25 | 30 | 5 | 83.3% | 111.0 | `is_padded=True` |
| **FD002** | 223 | 29 | 30 | 1 | 96.7% | 151.0 | `is_padded=True` |
| **FD004** | 10 | 23 | 30 | 7 | 76.7% | 162.0 | `is_padded=True` |
| **FD004** | 19 | 24 | 30 | 6 | 80.0% | 194.0 | `is_padded=True` |
| **FD004** | 28 | 20 | 30 | 10 | 66.7% | 124.0 | `is_padded=True` |
| **FD004** | 125 | 21 | 30 | 9 | 70.0% | 184.0 | `is_padded=True` |
| **FD004** | 141 | 19 | 30 | 11 | 63.3% | 130.0 | `is_padded=True` |
| **FD004** | 156 | 28 | 30 | 2 | 93.3% | 133.0 | `is_padded=True` |
| **FD004** | 164 | 29 | 30 | 1 | 96.7% | 174.0 | `is_padded=True` |
| **FD004** | 204 | 19 | 30 | 11 | 63.3% | 151.0 | `is_padded=True` |
| **FD004** | 229 | 20 | 30 | 10 | 66.7% | 120.0 | `is_padded=True` |
| **FD004** | 239 | 28 | 30 | 2 | 93.3% | 166.0 | `is_padded=True` |
| **FD004** | 246 | 29 | 30 | 1 | 96.7% | 194.0 | `is_padded=True` |

---

## 3. Code Implementation Inspection

The padding logic is implemented in [`preprocessing/create_sequences.py`](file:///c:/Users/LENOVO/Documents/Aircraft_Engine_AI/preprocessing/create_sequences.py#L98-L151):

```python
pad_len = window_size - L
pad_x = np.repeat(feats[:1], pad_len, axis=0)
full_x = np.vstack([pad_x, feats])
w_y = ruls[-1]
```

### Technical Audit of Code Steps:
1. `feats[:1]` selects the 24-feature vector of the earliest recorded observation cycle ($t=1$).
2. `np.repeat(..., pad_len, axis=0)` replicates this initial observation vector exactly $30 - L$ times.
3. `np.vstack([pad_x, feats])` places the replicated observations at the beginning (left edge) of the window, followed immediately by the recorded observations from $t=1$ to $t=L$.
4. `w_y = ruls[-1]` extracts the final observed cycle's RUL value, ensuring that the target is identical to the ground-truth benchmark target at cutoff.
5. In `test_final_metadata_{sub}.csv`, each padded window is explicitly flagged with:
   - `is_padded = True`
   - `padded_cycles_count = pad_len`
   - `start_cycle = 1`
   - `end_cycle = L`
   - `target_rul = ruls[-1]`

---

## 4. Verification of Audit Questions

### 4.1 Chronological Order Verification
- Slices $0 \le t < \text{pad\_len}$ contain repeated initial feature vectors ($\mathbf{x}_1$).
- Slices $\text{pad\_len} \le t < 30$ contain natural recorded cycle vectors $[\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_L]$.
- Temporal continuity and sorting on `time_cycles` are strictly preserved without reversal, permutation, or interleaving.

### 4.2 Target RUL and Ground-Truth Invariance
- Verification confirmed that for all 17 engines:
  $$\max |\mathbf{y}_{\text{test\_final}}[i] - \text{NASA\_GT}[i]| = 0.0$$
- In both `test_FD002_with_RUL.csv` and `test_FD004_with_RUL.csv`, the underlying row-level RUL labels remain completely unchanged ($0.0$ residual).

### 4.3 Test Engine Coverage
- All 707 test engines are present in the final evaluation arrays:
  - `FD001`: 100 / 100 engines
  - `FD002`: 259 / 259 engines
  - `FD003`: 100 / 100 engines
  - `FD004`: 248 / 248 engines
  - **Total:** 707 / 707 engines (100% representation).

---

## 5. Computational and Modeling Implications

### 5.1 Nature and Mechanics of Left-Edge Replication
Left-edge replication is strictly a **computational padding technique** used to construct a fixed-length array of dimension $(W, K)$ when a trajectory contains fewer than $W$ observations. It repeats the earliest available feature vector ($\mathbf{x}_1$) backwards for $30 - L$ cycles. 

It **does not represent a confirmed healthy engine state or actual historical operating conditions** prior to logging. The true operational history of the engine prior to cycle 1 is unrecorded and unknown in the C-MAPSS dataset.

### 5.2 Impact on Model Predictions and Hidden States
- **Static Input Prefix:** During the first $30 - L$ cycles, recurrent models (LSTM / GRU) process identical feature vectors where time-step differences are zero ($\Delta \mathbf{x} = 0$). This holds hidden state activations stationary across the padded prefix before dynamic transitions begin at cycle $t=1$.
- **Operating Conditions in Multi-Regime Datasets (`FD002` / `FD004`):** Operating settings (`op_setting_1`, `op_setting_2`, `op_setting_3`) are repeated identically for 1 to 11 cycles. This creates an artificial period of static operating conditions in the prefix that does not reflect actual flight profile variations, nor does it prove sustained cruise conditions before recording began.
- **Natural Cycle Majority:** For all 17 engines, the recorded trajectory length is substantial ($19 \le L \le 29$). Consequently, between 63.3% and 96.7% of each sequence consists of genuine recorded observations leading up to the final evaluation cycle.

### 5.3 Comparison Against Alternative Padding Strategies

| Strategy | Mechanism | Technical Characteristics | Limitations | Verdict |
| :--- | :--- | :--- | :--- | :---: |
| **Discard Engines ($L < 30$)** | Drop 17 test engines from benchmark | Avoids artificial padding | Violates NASA benchmark protocol; prevents direct comparison with published literature | **Not Recommended** |
| **Zero-Padding** | Prepend zeros (`0.0`) | Fixed dimension | Because features are MinMax-scaled, `0.0` represents the minimum observed value in the training set for each feature. Forcing all 24 normalized features to `0.0` simultaneously creates an artificial edge-case state that may disrupt recurrent state transitions | **Not Recommended** |
| **Variable-Length with Masking** | Dynamic length with padding mask | Mathematically isolates recorded cycles from padded cycles | Incompatible with standard 1D-CNN architectures; increases inference and batch processing complexity | **Viable Alternative** |
| **Left-Edge Replication (Current)** | Repeat $t=1$ observation backwards by $30 - L$ cycles | Maintains values within the observed operational range of that engine; fixed 3D tensor shape $(N, 30, 24)$; universal architecture compatibility | Introduces repeated synthetic observations in the prefix; holds operating settings artificially constant | **Approved Primary Strategy** |

---

## 6. Architectural Recommendation and Limitations

### 6.1 Recommendation to Retain Left-Edge Replication for Primary Experiments
Left-edge replication should be retained as the primary padding strategy for fixed-length sequence experiments across LSTM, GRU, and 1D-CNN models. It enables uniform tensor batching (`[batch_size, 30, 24]`) across all model families and preserves complete evaluation coverage across all 707 test engines.

### 6.2 Explicitly Documented Limitations
1. **Synthetic Repetition:** The prepended observations are computational artifacts, not real physical measurements.
2. **Artificial Operating Setting Invariance:** In multi-condition subsets (`FD002`, `FD004`), operating settings are held artificially constant during the padded prefix.
3. **Approximation:** Left-edge replication is an engineering accommodation to fulfill fixed-window dimensional requirements, not a reconstruction of pre-observation physics.

### 6.3 Mandatory Phase 4 Evaluation Protocol
To maintain full scientific transparency and eliminate ambiguity regarding padding effects, Phase 4 evaluation will report metrics across three distinct scopes:
1. **Full Benchmark Evaluation:** Overall RMSE and NASA Scoring Function across all **707 test engines** (ensuring direct comparability with standard C-MAPSS literature).
2. **Unpadded Subgroup Evaluation:** Metrics calculated exclusively on the **690 unpadded test engines** ($L \ge 30$).
3. **Padded Subgroup Evaluation:** Metrics calculated separately on the **17 padded test engines** ($L < 30$).

This three-tier reporting protocol will reveal the exact empirical impact of the padding strategy on predictive accuracy.

---

## 7. Audit Conclusion

The left-edge replication padding in `preprocessing/create_sequences.py` is verified to be:
- Strictly aligned with NASA ground-truth RUL targets ($0.0$ residual).
- Non-destructive to existing feature values and RUL labels.
- Completely traceable via metadata flags (`is_padded=True`, `padded_cycles_count`).
- A practical and standard computational approach for fixed-window sequence modeling when accompanied by transparent subgroup reporting.
