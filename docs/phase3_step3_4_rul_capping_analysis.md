# Phase 3 – Step 3.4: RUL Capping Strategy Evaluation and Decision

**Project:** AI-Based Aircraft Engine Health Monitoring, Fault Diagnosis and Predictive Maintenance  
**Dataset:** NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS)  
**Execution Script:** `preprocessing/analyze_rul_capping.py`  
**Generated Report:** `reports/eda/rul_capping_comparison.csv`  
**Document Status:** Complete (Step 3.4 Specification & Decision)  
**Date:** November 2026  

---

## 1. Executive Summary & Purpose

Remaining Useful Life (RUL) target formulation substantially impacts the training convergence, generalization, and benchmark comparability of predictive maintenance models. 

This document evaluates three target formulation strategies across all eight processed NASA C-MAPSS datasets (`train_FD001` through `train_FD004` and `test_FD001` through `test_FD004`):
1. **Raw Linear RUL (No Cap):** Direct countdown from initial lifetime to failure ($RUL(t) = t_{\text{max}} - t$).
2. **Piecewise Capped RUL ($RUL_{\text{cap}} = 125$ cycles):** $RUL(t) = \min(RUL_{\text{raw}}(t), 125)$.
3. **Piecewise Capped RUL ($RUL_{\text{cap}} = 130$ cycles):** $RUL(t) = \min(RUL_{\text{raw}}(t), 130)$.

Per Phase 3 specifications, **Step 3.4 is strictly an evaluation and strategic decision milestone**. Existing processed CSV files (`dataset/processed/*_with_RUL.csv`) remain unchanged with their raw linear RUL labels intact. No feature scaling, feature selection, or machine learning model training is conducted within this step.

---

## 2. Mathematical Formulation of Evaluated Strategies

### 2.1 Strategy 1: Raw Linear RUL (No Cap)
$$RUL_{\text{raw}}(i, t) = T_{\text{total}, i} - t$$
- **Characteristics:** Unbounded upper limit determined solely by the engine's total operational lifespan.
- **Underlying Premise:** A continuous linear countdown from cycle 1 to failure. As noted in Step 3.1, this is a supervised target-labeling convention and does not imply that detectable physical degradation starts at cycle 1.

### 2.2 Strategies 2 & 3: Piecewise Linear Capped RUL
$$RUL_{\text{capped}}(i, t) = \min\left( RUL_{\text{raw}}(i, t), RUL_{\text{cap}} \right) \quad \text{where } RUL_{\text{cap}} \in \{125, 130\}$$
- **Characteristics:** During initial engine cycles where $RUL_{\text{raw}}(i, t) \ge RUL_{\text{cap}}$, the target is fixed at the constant saturation value $RUL_{\text{cap}}$. Once the remaining lifespan drops below the threshold ($RUL_{\text{raw}} < RUL_{\text{cap}}$), the target decreases linearly by 1 cycle per operational cycle until failure ($RUL = 0$ in training) or cutoff ($RUL = RUL_{\text{GT}}$ in testing).
- **Underlying Premise:** Reflects the physical reality that gas turbine components maintain healthy baseline performance for substantial operating periods before measurable degradation initiates.

```mermaid
flowchart LR
    subgraph Raw Linear RUL
        R1["Cycle 1\nRUL = 300"] --> R2["..."] --> R3["Cycle 175\nRUL = 125"] --> R4["..."] --> R5["Cycle 300\nRUL = 0"]
    end
    subgraph Piecewise Capped RUL (Cap = 125)
        C1["Cycle 1\nRUL = 125 (Saturated)"] --> C2["..."] --> C3["Cycle 175\nRUL = 125 (Onset)"] --> C4["..."] --> C5["Cycle 300\nRUL = 0"]
    end
```

---

## 3. Empirical Statistical Analysis Across All Eight Datasets

The numerical analysis was computed directly on the eight processed datasets generated in Steps 3.2 and 3.3. The full comparative dataset is saved in [`reports/eda/rul_capping_comparison.csv`](file:///c:/Users/LENOVO/Documents/Aircraft_Engine_AI/reports/eda/rul_capping_comparison.csv).

### 3.1 Complete Capping Comparison Table

| Subset | Split | Strategy | Cap | Total Obs | Affected Obs | Affected % | Min | Max | Mean | Median | Std Dev | Unique Values |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FD001** | Train | Raw Linear | None | 20,631 | 0 | 0.00% | 0 | 361 | 107.81 | 103.0 | 68.88 | 362 |
| **FD001** | Train | Piecewise Cap | 125 | 20,631 | 8,031 | 38.93% | 0 | 125 | 86.83 | 103.0 | 41.67 | 126 |
| **FD001** | Train | Piecewise Cap | 130 | 20,631 | 7,534 | 36.52% | 0 | 130 | 88.73 | 103.0 | 43.44 | 131 |
| **FD001** | Test | Raw Linear | None | 13,096 | 0 | 0.00% | 7 | 340 | 141.24 | 140.0 | 58.98 | 334 |
| **FD001** | Test | Piecewise Cap | 125 | 13,096 | 7,952 | 60.72% | 7 | 125 | 108.92 | 125.0 | 27.58 | 119 |
| **FD001** | Test | Piecewise Cap | 130 | 13,096 | 7,496 | 57.24% | 7 | 130 | 111.89 | 130.0 | 29.36 | 124 |
| **FD002** | Train | Raw Linear | None | 53,759 | 0 | 0.00% | 0 | 377 | 108.15 | 103.0 | 69.18 | 378 |
| **FD002** | Train | Piecewise Cap | 125 | 53,759 | 20,999 | 39.06% | 0 | 125 | 86.91 | 103.0 | 41.67 | 126 |
| **FD002** | Train | Piecewise Cap | 130 | 53,759 | 19,704 | 36.65% | 0 | 130 | 88.82 | 103.0 | 43.44 | 131 |
| **FD002** | Test | Raw Linear | None | 33,991 | 0 | 0.00% | 6 | 377 | 143.96 | 143.0 | 63.87 | 372 |
| **FD002** | Test | Piecewise Cap | 125 | 33,991 | 20,500 | 60.31% | 6 | 125 | 107.75 | 125.0 | 28.83 | 120 |
| **FD002** | Test | Piecewise Cap | 130 | 33,991 | 19,486 | 57.33% | 6 | 130 | 110.71 | 130.0 | 30.64 | 125 |
| **FD003** | Train | Raw Linear | None | 24,720 | 0 | 0.00% | 0 | 524 | 138.08 | 123.0 | 98.85 | 525 |
| **FD003** | Train | Piecewise Cap | 125 | 24,720 | 12,120 | 49.03% | 0 | 125 | 93.14 | 123.0 | 40.63 | 126 |
| **FD003** | Train | Piecewise Cap | 130 | 24,720 | 11,620 | 47.01% | 0 | 130 | 95.55 | 123.0 | 42.55 | 131 |
| **FD003** | Test | Raw Linear | None | 16,596 | 0 | 0.00% | 6 | 483 | 169.88 | 159.0 | 84.23 | 478 |
| **FD003** | Test | Piecewise Cap | 125 | 16,596 | 11,387 | 68.61% | 6 | 125 | 112.31 | 125.0 | 24.84 | 120 |
| **FD003** | Test | Piecewise Cap | 130 | 16,596 | 10,950 | 65.98% | 6 | 130 | 115.69 | 130.0 | 26.61 | 125 |
| **FD004** | Train | Raw Linear | None | 61,249 | 0 | 0.00% | 0 | 542 | 133.31 | 122.0 | 89.78 | 543 |
| **FD004** | Train | Piecewise Cap | 125 | 61,249 | 29,875 | 48.78% | 0 | 125 | 92.99 | 122.0 | 40.67 | 126 |
| **FD004** | Train | Piecewise Cap | 130 | 61,249 | 28,633 | 46.75% | 0 | 130 | 95.38 | 122.0 | 42.58 | 131 |
| **FD004** | Test | Raw Linear | None | 41,214 | 0 | 0.00% | 6 | 553 | 181.90 | 171.0 | 92.17 | 548 |
| **FD004** | Test | Piecewise Cap | 125 | 41,214 | 29,342 | 71.19% | 6 | 125 | 112.61 | 125.0 | 25.46 | 120 |
| **FD004** | Test | Piecewise Cap | 130 | 41,214 | 28,414 | 68.94% | 6 | 130 | 116.13 | 130.0 | 27.21 | 125 |

---

## 4. Key Empirical Observations

### 4.1 Training Set Observations
1. **Single Fault Subsets (FD001, FD002):**
   - High-Pressure Compressor (HPC) degradation only. Maximum raw RUL reaches 361 cycles (FD001) and 377 cycles (FD002).
   - A cap of **125** affects **38.93%** (FD001) and **39.06%** (FD002) of records.
   - A cap of **130** affects **36.52%** (FD001) and **36.65%** (FD002) of records.
   - Standard deviation drops from ~69 cycles to ~41.7 cycles (cap 125) and ~43.4 cycles (cap 130).
2. **Two Fault Subsets (FD003, FD004):**
   - Combined HPC and Fan degradation modes. Engines exhibit substantially longer lifespans, reaching maximum raw RUL of 524 cycles (FD003) and 542 cycles (FD004).
   - A cap of **125** affects **49.03%** (FD003) and **48.78%** (FD004) of records.
   - A cap of **130** affects **47.01%** (FD003) and **46.75%** (FD004) of records.
   - Standard deviation drops sharply from ~90–99 cycles to ~40.6 cycles (cap 125) and ~42.5 cycles (cap 130).

### 4.2 Test Set Observations
- Because test engine trajectories are right-censored prior to failure, they contain a disproportionately higher share of earlier operational cycles compared to failure-stage cycles.
- In FD001 and FD002 test sets, **60.3%–60.7%** of observations are capped at 125 cycles (**57.2%–57.3%** at 130 cycles).
- In FD003 and FD004 test sets, **68.6%–71.2%** of observations are capped at 125 cycles (**66.0%–68.9%** at 130 cycles).
- Test set standard deviation is heavily compressed, dropping from ~59–92 cycles down to ~25–30 cycles.

### 4.3 Target Range Compression
- Capping at 125 collapses the target domain from 362–543 unique discrete values down to exactly **126 unique values** in training ($0, 1, \dots, 125$) and **119–120 unique values** in testing.
- Capping at 130 produces exactly **131 unique values** in training and **124–125 unique values** in testing.

---

## 5. How Capping Changes the Learning Target

### 5.1 Elimination of Early-Stage Noise Fitting
Under raw linear RUL, an engine at cycle 5 with 350 cycles remaining has target $RUL = 345$, whereas at cycle 55 it has target $RUL = 295$. However, exploratory data analysis in Phase 2 demonstrated that sensor measurements during early operational cycles show no detectable degradation trend; they fluctuate randomly around a healthy baseline. 

Forcing a machine learning or deep learning model to regress distinct targets (e.g., 345 vs. 295) on statistically indistinguishable feature vectors introduces significant label noise. The model is forced to fit sensor noise rather than genuine degradation features. Capping assigns both states the identical target (125 or 130), removing spurious variance.

### 5.2 Two-Phase Operational Representation
Piecewise capping converts the regression problem into two distinct physical regimes:
1. **Healthy Saturated Regime ($RUL = RUL_{\text{cap}}$):** The model learns that sensor readings correspond to normal healthy operation without attempting to pinpoint exact historical time.
2. **Active Degradation Regime ($RUL < RUL_{\text{cap}}$):** The model learns the continuous monotonic mapping between sensor drift and decreasing remaining operational life.

---

## 6. Advantages and Limitations of RUL Capping

### 6.1 Advantages
- **Physical Realism:** Aligns with standard mechanical degradation dynamics (fatigue crack initiation and propagation, where initiation accounts for a long latency period without measurable macro-level impact).
- **Substantially Improved Model Performance:** Published literature consistently reports lower Root Mean Squared Error (RMSE) and significantly lower asymmetric PHM scoring metric values when models are trained on piecewise capped targets.
- **Gradient Stability in Deep Networks:** Reduces target variance, preventing extreme early-life gradients that destabilize training in Multi-Layer Perceptrons (MLPs), Convolutional Neural Networks (CNNs), and Long Short-Term Memory (LSTM) networks.
- **Benchmark Comparability:** Enables direct comparison with major C-MAPSS published benchmarks.

### 6.2 Limitations and Trade-offs
- **Loss of Early-Life Lifespan Visibility:** When an engine is healthy, the system outputs $RUL = RUL_{\text{cap}}$ and cannot predict whether the engine will ultimately survive for 200 cycles or 400 cycles.
- **Artificial Knee-Point Derivative Discontinuity:** The piecewise function introduces an abrupt slope change at $RUL = RUL_{\text{cap}}$ that is not physically continuous.
- **Fixed-Threshold Assumption:** Applying a single uniform threshold (e.g., 125) assumes that degradation begins at the same remaining cycle count across all engines, regardless of operating regime variations or fault modes.

---

## 7. Cross-Subset Evaluation: Is One Common Cap Appropriate?

An important finding from our statistical analysis is the divergence between single-fault subsets (FD001/FD002, where ~37%–39% of train rows are capped) and two-fault subsets (FD003/FD004, where ~47%–49% of train rows are capped).

### Theoretical vs. Practical Evaluation:
- **Theoretical Argument for Different Caps:** Because FD003 and FD004 engines exhibit longer overall life (mean lifetime ~247 cycles vs ~206 cycles in FD001), one might consider a larger cap (e.g., 140–150 cycles) for FD003/FD004.
- **Practical and Literature Consensus:** Peer-reviewed literature overwhelmingly adopts a **uniform cap (125 or 130 cycles)** across all four subsets. The rationale is that the onset of accelerated, observable sensor drift typically occurs within 100–130 cycles of failure regardless of whether the initial healthy phase lasted 100 cycles or 300 cycles.

---

## 8. Academic Literature Review on C-MAPSS RUL Capping

The table below summarizes peer-reviewed prognostic literature evaluating piecewise linear RUL targets on NASA C-MAPSS:

| Citation | Study Title | Cap Evaluated | Key Findings / Rationale | DOI / Reference Link |
| :--- | :--- | :---: | :--- | :--- |
| **Heimes, F. O. (2008)** | *Recurrent neural networks for remaining useful life estimation* | **130 cycles** | Winner of the 2008 PHM Data Challenge. First introduced the piecewise linear RUL model, arguing that engines exhibit negligible observable wear until ~130 cycles before failure. | [DOI: 10.1109/PHM.2008.4711422](https://doi.org/10.1109/PHM.2008.4711422) |
| **Babu, G. S., Zhao, P., & Li, X. L. (2016)** | *Deep Convolutional Neural Network Based Regression Approach for Estimation of Remaining Useful Life* | **125 cycles** | Demonstrated that a 125-cycle cap on FD001 prevents the CNN from learning spurious patterns from early healthy data, establishing a widely cited C-MAPSS benchmark. | [DOI: 10.1007/978-3-319-32025-0_14](https://doi.org/10.1007/978-3-319-32025-0_14) |
| **Zheng, S., Ristovski, K., Farahat, A., & Gupta, C. (2017)** | *Long Short-Term Memory Network for Remaining Useful Life estimation* | **125 cycles** | Evaluated LSTM models across all four subsets (FD001–FD004) using $RUL_{\text{cap}} = 125$, demonstrating consistent performance and establishing standard baseline scores. | [DOI: 10.1109/ICPHM.2017.7998311](https://doi.org/10.1109/ICPHM.2017.7998311) |
| **Li, X., Ding, Q., & Sun, J. Q. (2018)** | *Remaining useful life estimation in prognostics using deep convolution neural networks* | **125 cycles** | Showed that 125-cycle piecewise linear RUL significantly out-performs linear RUL across all C-MAPSS sub-datasets, confirming degradation onset occurs in late stages. | [DOI: 10.1016/j.ress.2017.11.021](https://doi.org/10.1016/j.ress.2017.11.021) |
| **Ramasso, E., & Saxena, A. (2014)** | *Performance Benchmarking and Analysis of Prognostic Methods for CMAPSS Datasets* | Review / Synthesis | Systematic benchmark study reviewing scoring metrics and target definitions, noting that piecewise clipping is essential for fair comparison across algorithmic approaches. | [DOI: 10.36001/ijphm.2014.v5i2.2244](https://doi.org/10.36001/ijphm.2014.v5i2.2244) |

> [!NOTE]
> None of the cited literature claims that 125 or 130 is universally optimal for every mechanical system. Rather, both values serve as well-justified heuristic thresholds that effectively balance early-stage noise reduction with degradation sensitivity on C-MAPSS.

---

## 9. Distinguishing Empirical Observations from Modeling Assumptions

To maintain academic rigor, the distinction between empirical facts and modeling assumptions is clarified below:

| Aspect | Empirical Observation (Fact) | Modeling Assumption / Literature Convention |
| :--- | :--- | :--- |
| **Early Cycles** | Sensor measurements exhibit near-stationary mean and random variance during initial cycles (Phase 2 EDA). | The engine is assumed to be in an entirely healthy state, so RUL can be treated as constant ($RUL = RUL_{\text{cap}}$). |
| **Capping Percentage** | Capping at 125 affects 38.9%–49.0% of train rows and 60.3%–71.2% of test rows. | The affected proportion is assumed to represent non-informative early life rather than missed progressive wear. |
| **Cap Threshold** | 125 and 130 cycles produce nearly identical statistical profiles (within 2%–3% affected rows). | 125 is assumed to be a representative inflection point for degradation onset across all engines and fault modes. |
| **Target Linearity** | Sensor signals show non-linear drift in late stages for selected engines (Phase 2 EDA). | Linear decrement after the cap is adopted as a pragmatic regression target rather than an exact physical wear law. |

---

## 10. Strategic Recommendation and Project Decision

### 10.1 Decision
Based on empirical data distribution, physical degradation logic, and literature consensus, the project adopts the following multi-tiered strategy:

1. **Primary Target Strategy for Model Training: Piecewise Linear Capped RUL with $RUL_{\text{cap}} = 125$ cycles.**
   - **Why 125 cycles?** It is the single most widely adopted threshold in modern deep learning C-MAPSS literature (Babu et al., 2016; Zheng et al., 2017; Li et al., 2018), ensuring direct comparability with published benchmark results.
   - It effectively mitigates early-cycle sensor noise fitting across all four subsets.
2. **Preservation of Raw Linear RUL as Unconstrained Baseline:**
   - The processed files generated in Steps 3.2 and 3.3 (`train_FD00x_with_RUL.csv` and `test_FD00x_with_RUL.csv`) **will retain raw linear RUL as their base ground truth column**.
   - Capping will be applied during feature preprocessing / pipeline configuration or as an explicit target transformation. This preserves maximum architectural flexibility without corrupting or overwriting base data.
3. **Sensitivity Validation During Modeling:**
   - In Phase 4/5 (Model Training and Evaluation), models will be evaluated against:
     - Baseline: Raw Linear RUL
     - Primary: Capped RUL ($RUL_{\text{cap}} = 125$)
     - Sensitivity Variant: Capped RUL ($RUL_{\text{cap}} = 130$)
   - This empirically quantifies the exact performance delta attributable to the capping strategy on our models.

---

## 11. Compliance and Verification Summary

- [x] Evaluated all eight processed datasets (train and test for FD001–FD004).
- [x] Analyzed Raw, Cap 125, and Cap 130 distributions and impact metrics.
- [x] Generated [`reports/eda/rul_capping_comparison.csv`](file:///c:/Users/LENOVO/Documents/Aircraft_Engine_AI/reports/eda/rul_capping_comparison.csv).
- [x] Conducted comprehensive academic literature review with valid citations and DOIs.
- [x] Maintained strict dataset immutability: zero modifications to raw data or processed CSVs.
- [x] Formulated clear, reasoned strategic recommendation.

*Awaiting your approval before proceeding to **Phase 3 – Step 3.5**.*
