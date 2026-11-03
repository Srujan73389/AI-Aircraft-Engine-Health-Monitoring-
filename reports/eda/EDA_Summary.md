# NASA C-MAPSS Exploratory Data Analysis (EDA) Summary

**Project Title:** AI-Based Aircraft Engine Health Monitoring, Fault Diagnosis and Predictive Maintenance  
**Phase Reference:** Phase 2 – Dataset Understanding & Exploratory Data Analysis (Steps 2.1–2.12)  
**Date:** November 2026  
**Primary Reference Document:** [Master EDA Report (`CMAPSS_EDA_Report.md`)](CMAPSS_EDA_Report.md)  
**Dataset Source:** NASA Ames Research Center / Prognostics Center of Excellence (PCoE)  

---

## 1. Overview and Objectives
This document provides an executive summary of the Exploratory Data Analysis (EDA) conducted across the four subsets of the NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) turbofan dataset (`FD001`, `FD002`, `FD003`, `FD004`). The purpose of the EDA was to establish dataset integrity, map physical sensor properties, characterize operating condition dynamics, and evaluate sensor dispersion before entering Phase 3 (Data Preprocessing & Feature Engineering).

---

## 2. Dataset Architecture & Dimensions
The benchmark comprises 12 raw data files covering 709 training trajectories (run-to-failure) and 707 testing trajectories (right-censored observations), totaling **265,256 multivariate operational records**:

| Subset | Operating Conditions | Fault Modes | Training Rows (Engines) | Testing Rows (Engines) | Ground-Truth RULs | Trajectory Length (Train Min–Max) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **FD001** | 1 (Single Condition) | 1 (HPC Wear) | 20,631 (100) | 13,096 (100) | 100 | 128 – 362 cycles |
| **FD002** | 6 (Multi-Condition) | 1 (HPC Wear) | 53,759 (260) | 33,991 (259) | 259 | 128 – 378 cycles |
| **FD003** | 1 (Single Condition) | 2 (HPC + Fan) | 24,720 (100) | 16,596 (100) | 100 | 145 – 525 cycles |
| **FD004** | 6 (Multi-Condition) | 2 (HPC + Fan) | 61,249 (249) | 41,214 (248) | 248 | 128 – 543 cycles |
| **Total** | — | — | **160,359 (709)** | **104,897 (707)** | **707** | — |

Each observation contains 26 numerical variables: engine unit ID, operational cycle counter, 3 operational settings (`op_setting_1` to `op_setting_3`), and 21 thermodynamic and mechanical sensors (`sensor_1` to `sensor_21`).

---

## 3. Data Quality & Integrity Assessment (Steps 2.5–2.7)
Automated verification scripts confirmed that the raw datasets possess flawless structural integrity:
- **Missing / Null Values:** Exactly **0** missing or empty values across all 6,896,656 data cells ([`preprocessing/check_data_quality.py`](../../preprocessing/check_data_quality.py)).
- **Non-Numeric Tokens:** Exactly **0** non-numeric tokens; all fields parse into standard IEEE 64-bit numerical types.
- **Infinite Values:** Exactly **0** positive or negative infinite values.
- **Duplicate Rows:** Exactly **0** full-row duplicates across all 26 columns ([`preprocessing/check_duplicates.py`](../../preprocessing/check_duplicates.py)).
- **Key Collisions:** Exactly **0** duplicate `(unit_number, time_cycles)` primary keys.
- **Chronological Continuity:** Consecutive cycles increment strictly by $+1$ (`curr_cycle - prev_cycle == 1`) with zero missing cycle numbers or order inversions ([`preprocessing/check_cycle_order.py`](../../preprocessing/check_cycle_order.py)).
- **Engine Grouping:** All engines are organized as contiguous blocks from unit $1$ to $N$.

---

## 4. Key EDA Findings & Sensor Behavior

### 4.1 Single vs. Multiple Operating Conditions
- **FD001 and FD003 (Single Operating Condition):** Operational settings remain constant across cycles (`op_setting_1` $\approx 0$, `op_setting_2` $\approx 0$, `op_setting_3` = 100). Sensors 1, 5, 16, 18, and 19 (and sensor 10 in FD001) exhibit **identically zero standard deviation**. Active thermodynamic channels display monotonic intra-engine wear trends.
- **FD002 and FD004 (Multiple Operating Conditions):** Engines operate across six distinct flight regimes spanning altitude (0 to 42,000 ft), Mach number (0.00 to 0.84), and throttle resolver angle (60 to 100). All 21 sensors exhibit non-zero empirical variation across records. Operational regime shifts produce discrete banding in sensor distributions that heavily obscure underlying degradation in unnormalized plots.

### 4.2 Sensor Dispersion Classification (Step 2.11)
All 21 sensors across all four subsets were classified into four analytical categories ([`sensor_variation_classification.csv`](sensor_distributions/sensor_variation_classification.csv)):
1. **Constant (zero variance):** Sensors 1, 5, 10, 16, 18, 19 in FD001; Sensors 1, 5, 16, 18, 19 in FD003. (Zero sensors are constant in FD002 and FD004).
2. **Near-Constant (negligible variance):** Sensors 8 and 13 in FD001/FD003 (governed fan speeds, $\text{CV} < 0.01\%$); Sensor 6 in FD001 ($\text{std} = 0.0014\,\text{psia}$); Sensor 10 in FD003 ($\text{std} = 0.0035$).
3. **Low-Variation ($\text{std} < 1.0$):** `sensor_15` (Bypass Ratio) maintains $\text{std} < 1.0$ across **all four subsets** ($0.04$ in FD001, $0.75$ in FD002, $0.06$ in FD003, $0.75$ in FD004). Sensors 2, 11, 20, 21 have low variation in FD001/3, but higher dispersion in FD002/4.
4. **Variable ($\text{std} \ge 1.0$):** High-amplitude core variables (`sensor_3` $T_{30}$, `sensor_4` $T_{50}$, `sensor_9` $N_c$, `sensor_14` $Nc_{\text{dmd}}$, `sensor_17` $ht\text{Bleed}$) show substantial variation in all subsets.

### 4.3 Training vs. Testing Numerical Consistency
- Independent evaluation of the classification criteria across training and testing data yielded **84 out of 84 (100%) exact classification matches**.
- Mean differences between train and test across active channels remain under $0.20\%$ in 74 of the 84 comparisons (max difference $0.96\%$ for `sensor_16` in FD004).
- Lower sample variances observed in test datasets are consistent with right-censored testing trajectories cut off before failure.

---

## 5. Phase 3 Readiness & Recommendations
The dataset is in pristine structural condition and fully ready for Phase 3 (Data Preprocessing & Feature Engineering):
1. **Subset-Specific Feature Review:** Invariant features in FD001 and FD003 may be candidates for removal depending on model architecture (e.g. to streamline linear/distance-based models), whereas in FD002 and FD004 all 21 sensors should be retained to provide operating regime information.
2. **Operating Regime Normalization:** In multi-condition subsets (FD002 and FD004), regime clustering (e.g. k-means) and regime-specific standardization should be evaluated to decouple flight condition shifts from degradation tracking.
3. **RUL Target Construction:** Prepare both linear run-to-failure targets ($RUL_t = t_{\text{max}} - t$) and piecewise linear clipped targets for training sets, matching against ground-truth RUL vectors for test evaluation.
4. **Zero Data Leakage:** Ensure all scaler transformations, clustering models, and statistical baselines are fit strictly on training data before transforming test data.

---
*For the complete detailed analysis, statistical tables, and embedded visualizations, refer to the [Master EDA Report (`CMAPSS_EDA_Report.md`)](CMAPSS_EDA_Report.md).*
