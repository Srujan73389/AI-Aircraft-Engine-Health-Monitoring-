# Phase 2 – Step 2.11: Sensor Variation Analysis Across NASA C-MAPSS Subsets

## 1. Objective
This report classifies and analyzes the empirical dispersion of all 21 sensor channels across all four subsets (FD001, FD002, FD003, FD004) of the NASA C-MAPSS dataset. The objective is to identify constant, near-constant, low-variation, and variable sensors separately for each subset using documented analytical thresholds.

---

## 2. Classification Methodology and Analytical Choices

The 21 sensors are evaluated separately for each subset and categorized into four analytical classes based on their empirical dispersion. These criteria represent documented analytical choices tailored to the C-MAPSS dataset rather than universal engineering standards:

1. **Constant:**
   - **Criterion:** Standard deviation identically zero ($\text{std} = 0.0$ or within floating-point zero $< 10^{-12}$).
   - **Definition:** The sensor outputs an identical numerical value across all recorded rows in that subset (zero within-subset variation).
2. **Near-Constant:**
   - **Criterion:** Non-zero standard deviation with negligible dispersion relative to observed scale:
     - $\text{std} \le 0.005$ with an absolute span $(\text{max} - \text{min}) \le 0.05$ and relative span $\le 5.0\%$, **OR**
     - Coefficient of variation $\text{CV} = \frac{\text{std}}{|\text{mean}|} \times 100\% < 0.01\%$.
   - **Definition:** The sensor exhibits minute numerical fluctuations (such as fractional-rpm flutter on governed fan speeds or single-digit digitizer steps).
3. **Low-Variation:**
   - **Criterion:** Standard deviation $\text{std} < 1.0$ (with $\text{std} > 0.005$ and span $> 0.05$).
   - **Definition:** The sensor displays limited dispersion compared with major high-amplitude variables in that subset.
4. **Variable:**
   - **Criterion:** Standard deviation $\text{std} \ge 1.0$.
   - **Definition:** The sensor shows comparatively substantial statistical variation across records in that subset.

---

## 3. Comprehensive Sensor Classification Table

The complete 84-record dataset is saved at [`reports/eda/sensor_distributions/sensor_variation_classification.csv`](file:///c:/Users/LENOVO/Documents/Aircraft_Engine_AI/reports/eda/sensor_distributions/sensor_variation_classification.csv). The summary matrix below compares classifications and empirical standard deviations across all four subsets:

| Sensor | Symbol | Description | Unit | FD001 Class (std) | FD002 Class (std) | FD003 Class (std) | FD004 Class (std) | Analytical Explanation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **sensor_1** | $T_2$ | Fan Inlet Total Temp | $^\circ\text{R}$ | **Constant** (0.00) | **Variable** (26.39) | **Constant** (0.00) | **Variable** (26.44) | Constant in single operating condition; varies across multiple operating conditions. |
| **sensor_2** | $T_{24}$ | LPC Outlet Total Temp | $^\circ\text{R}$ | **Low-variation** (0.50) | **Variable** (37.29) | **Low-variation** (0.52) | **Variable** (37.34) | Limited standard deviation in FD001/FD003 (< 1.0); substantial dispersion in FD002/FD004. |
| **sensor_3** | $T_{30}$ | HPC Outlet Total Temp | $^\circ\text{R}$ | **Variable** (6.13) | **Variable** (105.95) | **Variable** (6.81) | **Variable** (106.17) | Substantial standard deviation and numerical range across records in all subsets. |
| **sensor_4** | $T_{50}$ | LPT Outlet Total Temp | $^\circ\text{R}$ | **Variable** (9.00) | **Variable** (119.12) | **Variable** (9.77) | **Variable** (119.33) | Wide dispersion in recorded values across all subsets. |
| **sensor_5** | $P_2$ | Fan Inlet Pressure | $\text{psia}$ | **Constant** (0.00) | **Variable** (3.61) | **Constant** (0.00) | **Variable** (3.62) | Pressure invariant in single operating condition; varies across multiple operating conditions. |
| **sensor_6** | $P_{15}$ | Bypass-Duct Total Pressure | $\text{psia}$ | **Near-constant** (0.0014) | **Variable** (5.43) | **Low-variation** (0.0181) | **Variable** (5.44) | Very narrow range in single operating condition ($[21.45, 21.61]$); wider spread in multi-condition. |
| **sensor_7** | $P_{30}$ | HPC Outlet Total Pressure | $\text{psia}$ | **Low-variation** (0.89) | **Variable** (146.01) | **Variable** (3.44) | **Variable** (146.88) | Low std in FD001 (0.89); larger spread in FD003 (3.44) and multi-condition subsets. |
| **sensor_8** | $N_f$ | Physical Fan Speed | $\text{rpm}$ | **Near-constant** (0.07) | **Variable** (145.21) | **Near-constant** (0.16) | **Variable** (145.35) | Governed speed with minute dispersion in single operating condition (CV < 0.01%); varies in 2/4. |
| **sensor_9** | $N_c$ | Physical Core Speed | $\text{rpm}$ | **Variable** (22.08) | **Variable** (335.81) | **Variable** (19.98) | **Variable** (336.93) | High variation reflecting core spool dynamics across all subsets. |
| **sensor_10** | $\text{epr}$ | Engine Pressure Ratio | — | **Constant** (0.00) | **Low-variation** (0.13) | **Near-constant** (0.0035) | **Low-variation** (0.13) | Constant in FD001 (1.30); span 0.03 in FD003; varies across operating regimes in 2/4. |
| **sensor_11** | $Ps_{30}$ | HPC Outlet Static Pressure | $\text{psia}$ | **Low-variation** (0.27) | **Variable** (3.23) | **Low-variation** (0.30) | **Variable** (3.24) | Measurable empirical variation across cycles within a narrow absolute range. |
| **sensor_12** | $\phi$ | Fuel Flow to $Ps_{30}$ Ratio | $\text{pps/psia}$ | **Low-variation** (0.74) | **Variable** (137.66) | **Variable** (3.26) | **Variable** (138.48) | Measurable variation across cycles in FD001; wider spread in FD003 and multi-condition subsets. |
| **sensor_13** | $Nf_{\text{dmd}}$ | Corrected Fan Speed | $\text{rpm}$ | **Near-constant** (0.07) | **Variable** (128.07) | **Near-constant** (0.16) | **Variable** (128.20) | Follows demanded speed; negligible spread in single operating condition. |
| **sensor_14** | $Nc_{\text{dmd}}$ | Corrected Core Speed | $\text{rpm}$ | **Variable** (19.08) | **Variable** (84.84) | **Variable** (16.50) | **Variable** (85.67) | Substantial standard deviation in all subsets (std 16.5 to 85.7). |
| **sensor_15** | $\text{BPR}$ | Bypass Ratio | — | **Low-variation** (0.04) | **Low-variation** (0.75) | **Low-variation** (0.06) | **Low-variation** (0.75) | Low standard deviation (< 1.0) in all subsets despite measurable cyclic progression. |
| **sensor_16** | $\text{farB}$ | Burner Fuel-Air Ratio | — | **Constant** (0.00) | **Low-variation** (0.0047) | **Constant** (0.00) | **Low-variation** (0.0047) | Invariant in single condition (0.03); low absolute std (0.0047) across multiple regimes. |
| **sensor_17** | $ht\text{Bleed}$ | Bleed Enthalpy | — | **Variable** (1.55) | **Variable** (27.75) | **Variable** (1.76) | **Variable** (27.81) | Substantial standard deviation across records in all four subsets. |
| **sensor_18** | $Nf_{\text{dmd2}}$ | Demanded Fan Speed | $\text{rpm}$ | **Constant** (0.00) | **Variable** (145.33) | **Constant** (0.00) | **Variable** (145.47) | Invariant demand (2388 rpm) in single condition; variable in multiple operating conditions. |
| **sensor_19** | $PCNfR_{\text{dmd}}$ | Demanded Corr. Fan Speed | $\text{rpm}$ | **Constant** (0.00) | **Variable** (5.36) | **Constant** (0.00) | **Variable** (5.37) | Invariant demand (100 rpm) in single condition; variable in multiple operating conditions. |
| **sensor_20** | $W_{31}$ | HPT Coolant Bleed | $\text{lbm/s}$ | **Low-variation** (0.18) | **Variable** (9.87) | **Low-variation** (0.25) | **Variable** (9.94) | Limited standard deviation in FD001/FD003 (< 0.25); substantial variation in FD002/FD004. |
| **sensor_21** | $W_{32}$ | LPT Coolant Bleed | $\text{lbm/s}$ | **Low-variation** (0.11) | **Variable** (5.92) | **Low-variation** (0.15) | **Variable** (5.96) | Limited standard deviation in FD001/FD003 (< 0.25); substantial variation in FD002/FD004. |

---

## 4. Key Findings Across FD001–FD004

### A. Sensors Constant Across ALL Four Subsets
- **Finding:** **Zero sensors** are constant across all four subsets.
- Every sensor that is invariant in FD001 and FD003 exhibits measurable variation in FD002 and FD004.

### B. Sensors Constant Only in Particular Subsets
- **FD001 (6 sensors):** `sensor_1`, `sensor_5`, `sensor_10`, `sensor_16`, `sensor_18`, `sensor_19` (identically zero variance).
- **FD003 (5 sensors):** `sensor_1`, `sensor_5`, `sensor_16`, `sensor_18`, `sensor_19` (identically zero variance).
- **FD002 and FD004 (0 sensors):** All sensors exhibit non-zero variance.

### C. Low-Variation in FD001/FD003 but Substantial Variation in FD002/FD004
- Sensors `sensor_1`, `sensor_5`, `sensor_6`, `sensor_8`, `sensor_10`, `sensor_13`, `sensor_16`, `sensor_18`, and `sensor_19` display zero, near-zero, or very narrow dispersion under single operating condition subsets, but show substantial dispersion in FD002 and FD004.
- In FD002 and FD004, the standard deviations of these sensors are orders of magnitude higher (e.g. `sensor_18` std is $145.4\,\text{rpm}$ vs. $0.0$; `sensor_1` std is $26.4\,^\circ\text{R}$ vs. $0.0$). This wider spread is consistent with operation across multiple operating conditions.

### D. Sensors Showing Variation Across Multiple Subsets
Sensors displaying non-zero variation across multiple subsets fall into distinct statistical dispersion profiles:

1. **Substantial Variation Across All Subsets (Variable in FD001–FD004):**
   - **Turbine temperatures:** `sensor_3` ($T_{30}$) and `sensor_4` ($T_{50}$) maintain $\text{std} \ge 6.1$ in all four subsets.
   - **Core speeds:** `sensor_9` ($N_c$) and `sensor_14` ($Nc_{\text{dmd}}$) exhibit $\text{std} \ge 16.5$ in all four subsets.
   - **Bleed enthalpy:** `sensor_17` ($ht\text{Bleed}$) maintains $\text{std} \ge 1.55$ in all four subsets.
2. **Substantial Variation in Most Subsets:**
   - `sensor_7` ($P_{30}$) and `sensor_12` ($\phi$) are classified as **Low-variation** in FD001 ($\text{std} < 1.0$), but **Variable** in FD002, FD003, and FD004 ($\text{std} \ge 3.2$).
3. **Low-Variation in Single-Condition Subsets, Variable in Multi-Condition Subsets:**
   - `sensor_2` ($T_{24}$), `sensor_11` ($Ps_{30}$), `sensor_20` ($W_{31}$), and `sensor_21` ($W_{32}$) exhibit limited dispersion in single operating condition subsets ($\text{std} < 0.55$ in FD001 and FD003), but substantial dispersion in multiple operating condition subsets ($\text{std} \ge 3.2$ in FD002 and FD004).
4. **Low-Variation Across All Four Subsets:**
   - **Bypass Ratio (`sensor_15`):** Classified as **Low-variation in all four subsets** ($\text{std} = 0.04$ in FD001, $0.75$ in FD002, $0.06$ in FD003, and $0.75$ in FD004). It does not exhibit $\text{std} \ge 1.0$ in any subset.

---

## 5. Training versus Testing Numerical Comparison

To verify that classification agreement is reproducible and not an artifact of reusing training outputs, the classification procedure was executed completely independently on both datasets:

### Methodology
1. **Independent Evaluation:** The exact analytical classification logic `classify_sensor(std, mean, min, max)` was executed separately on the training statistics ([`train_sensor_distribution_comparison.csv`](file:///c:/Users/LENOVO/Documents/Aircraft_Engine_AI/reports/eda/sensor_distributions/train_sensor_distribution_comparison.csv)) and the testing statistics ([`test_sensor_distribution_comparison.csv`](file:///c:/Users/LENOVO/Documents/Aircraft_Engine_AI/reports/eda/sensor_distributions/test_sensor_distribution_comparison.csv)).
2. **Automated Cross-Verification:** The two resulting independent classification sets were merged using an inner join on `[sensor, subset]`, comparing `classification_train` against `classification_test` across all 84 pairs (21 sensors × 4 subsets).
3. **Metric Drift Verification:** Two distinct metrics were computed for every sensor-subset pair:
   - **Relative Mean Difference:** $|\mu_{\text{train}} - \mu_{\text{test}}| / |\mu_{\text{train}}| \times 100\%$.
   - **Absolute Standard Deviation Difference:** $|\sigma_{\text{train}} - \sigma_{\text{test}}|$ (expressed in sensor measurement units).
   - **Relative Standard Deviation Difference:** $|\sigma_{\text{train}} - \sigma_{\text{test}}| / \sigma_{\text{train}} \times 100\%$ (where $\sigma_{\text{train}} > 0$).

### Comparison Results
1. **Classification Agreement:** Exactly **84 out of 84 (100%)** sensor-subset classifications are identical between training and testing sets.
2. **Mean Difference:**
   - The maximum relative mean difference observed across all 84 comparisons is **0.9607%** (`sensor_16` in FD004: train mean = 0.02325, test mean = 0.02303).
   - In 74 of the 84 comparisons, the relative mean difference is below 0.20%.
3. **Standard Deviation Difference:**
   - **Absolute differences:** Across active sensors, the largest absolute standard deviation difference observed is **10.65 rpm** (`sensor_9` in FD001: train $\text{std} = 22.08$, test $\text{std} = 11.44$).
   - **Relative differences:** Across active sensors, the largest relative standard deviation difference is **48.21%** (`sensor_9` in FD001), followed by **46.59%** (`sensor_14` in FD001).
   - **Observed Pattern:** For active degradation-sensitive sensors in FD001 and FD003, test-set standard deviations are empirically lower than training-set standard deviations. A possible explanation is that testing engine trajectories are cut off prior to failure (censored observations), but establishing this conclusively requires trajectory-level analysis.

---

## 6. Implications for Phase 3 Feature Selection

1. **Features with Zero Variation (FD001 & FD003):**
   - In single operating condition subsets, features with identically zero standard deviation (`sensor_1`, `sensor_5`, `sensor_16`, `sensor_18`, `sensor_19`, and `sensor_10` in FD001) carry no within-subset information.
   - These constant features are candidates for review during feature selection. However, feature removal is not universally required for all machine learning models (e.g. tree-based algorithms naturally ignore zero-variance features without penalty, whereas certain linear or distance-based algorithms can encounter numerical instability).
   - Usefulness and retention should be evaluated separately for each subset and proposed model type.
2. **Multiple Operating Condition Subsets (FD002 & FD004):**
   - Features that are invariant in FD001/FD003 vary substantially across records in FD002/FD004.
   - Removing these features globally based solely on FD001/FD003 observations would discard variables that distinguish operating conditions in FD002 and FD004. Feature selection must therefore be tailored per subset or account for operating regime normalization.
3. **No Features Removed in Phase 2:**
   - In accordance with Phase 2 requirements, **no features have been dropped, imputed, scaled, or excluded** in this step. All 21 original sensors remain available for Phase 3 evaluation.

---

## 7. Limitations and Analytical Assumptions

1. **Analytical vs. Engineering Thresholds:** The boundaries chosen (e.g. $\text{std} < 1.0$ for low-variation, $\text{CV} < 0.01\%$) are analytical conventions designed to structure the exploratory analysis. They do not represent regulatory, physical, or manufacturer sensor thresholds.
2. **Measurement Scale Sensitivity:** Absolute standard deviation depends on measurement units ($^\circ\text{R}$, $\text{psia}$, $\text{rpm}$, or dimensionless ratios). A small absolute standard deviation in a ratio like BPR ($0.04$) reflects significant physical movement, whereas a standard deviation of $0.07\,\text{rpm}$ in fan speed represents minute noise.
3. **Variability vs. Degradation:** Standard deviation and range establish empirical statistical variation, not physical degradation causation. Step 2.9 provided exploratory plots for selected engines and sensors, while establishing physical degradation causation requires further comprehensive trajectory analysis and modeling.
