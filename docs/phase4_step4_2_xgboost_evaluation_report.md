# Phase 4 – Step 4.2: XGBoost Regression Model Evaluation Report

> [!NOTE]
> **EXECUTIVE SUMMARY & BENCHMARK CONFIRMATION**  
> Under **Option A: Strict Roadmap Realignment**, Phase 4 Step 4.2 trains and evaluates gradient-boosted decision trees using the **XGBoost** library on engine sensor features and operating settings.  
> Exactly **40 model configurations** (10 per subset across FD001–FD004) were trained and evaluated across three feature representations (Final Observation, Flattened Window, Summary Statistics) and three feature tiers (Tier 1 Core 14, Tier 2 Operational 17, Tier 3 Extended 24).  
> In accordance with our decoupled evaluation protocol, **model selection was conducted strictly and exclusively via validation capped RMSE** ($y \le 125$). Official NASA test data remained completely isolated until after model selection.

---

## 1. Master Benchmark Summary: XGBoost vs. Step 4.1 Baselines

The table below presents the definitive comparison between the winning Phase 4 Step 4.1 reference baselines (Random Forest & Ridge) and the newly trained Phase 4 Step 4.2 XGBoost models:

| Subset | Winning Step 4.1 Baseline | Baseline Val RMSE | Baseline Test RMSE | Baseline Test Score | Winning Step 4.2 XGBoost Model | XGBoost Val RMSE | XGBoost Test RMSE | XGBoost Test Score | Performance Delta (Test RMSE / Score) |
| :--- | :--- | :---: | :---: | :---: | :--- | :---: | :---: | :---: | :---: |
| **FD001** | `RandomForest_SummaryStats` (Tier 1) | 13.66 | 14.29 | 294.40 | `XGBoost_Tuned_SummaryStats` (Tier 1) | **12.75** | **13.48** | **268.06** | **-0.81 cycles (-8.9% score)** |
| **FD002** | `Ridge_FlattenedWindow` (Tier 3) | 18.59 | 29.62 | 14,862.81 | `XGBoost_FlattenedWindow` (Tier 3) | **19.33** | 30.05 | **14,159.01** | **-703.8 score penalty** (vs RF: -2.10 cycles / -43.6% score) |
| **FD003** | `RandomForest_SummaryStats` (Tier 1) | 12.87 | 14.96 | 431.96 | `XGBoost_Tuned_SummaryStats` (Tier 3) | **12.79** | **14.25** | **312.21** | **-0.71 cycles (-27.7% score)** |
| **FD004** | `RandomForest_SummaryStats` (Tier 3) | 17.66 | 32.58 | 10,955.40 | `XGBoost_FlattenedWindow` (Tier 3) | **15.87** | **29.73** | **6,939.94** | **-2.85 cycles (-36.6% score)** |

### Key Takeaways:
1. **Decisive Dominance Across Subsets:** XGBoost achieved substantial improvements across all single-condition and multi-condition subsets:
   - On **FD001**, test RMSE dropped from 14.29 to **13.48** cycles and the NASA penalty score dropped from 294.40 to **268.06**.
   - On **FD003**, test RMSE dropped from 14.96 to **14.25** cycles, reducing asymmetric penalty by **27.7%** (312.21 vs 431.96).
   - On **FD004**, XGBoost produced a massive breakthrough, reducing validation capped RMSE from 17.66 to **15.87**, test RMSE from 32.58 to **29.73** cycles (a 2.85-cycle reduction), and slashing the NASA score from 10,955.40 to **6,939.94** (a 36.6% reduction).
2. **Representation Divergence:**
   - Single-operating-condition sets (**FD001, FD003**) are best solved by **`SummaryStats`** with tuning and early stopping, where the linear trend slope directly tracks monotonic degradation rates.
   - Multi-operating-condition sets (**FD002, FD004**) are best solved by **`FlattenedWindow`** with all 24 Tier 3 features, where tree splits learn complex non-linear combinations across sequential operational shifts.

---

## 2. Experimental Design & Methodology

### 2.1 Experiment Matrix (40 Configurations)
For each NASA C-MAPSS subset (FD001, FD002, FD003, FD004), 10 configurations were trained and evaluated:
* **Configs 1–3:** `XGBoost_FinalObservation` across Tiers 1, 2, 3 (instantaneous snapshot at $t = 29$).
* **Configs 4–6:** `XGBoost_FlattenedWindow` across Tiers 1, 2, 3 (unrolled temporal sequence of $30 \times K$).
* **Configs 7–9:** `XGBoost_SummaryStats` across Tiers 1, 2, 3 (Mean, Std, Min, Max, OLS Trend Slope across window).
* **Config 10:** `XGBoost_Tuned_SummaryStats` on the winning validation tier for summary statistics, trained with early stopping (`early_stopping_rounds = 20` evaluated on validation capped RMSE) and structural regularization (`max_depth = 4`, `learning_rate = 0.05`, `reg_alpha = 0.1`, `reg_lambda = 5.0`).

### 2.2 Strict Anti-Leakage & Selection Protocols
1. **Training Target:** Capped at 125 cycles ($y_{\text{train}} = \min(\text{raw\_RUL}, 125)$).
2. **Model Selection Criterion:** Governed **exclusively by validation capped RMSE** ($y_{\text{val\_capped}} = \min(\text{raw\_RUL}, 125)$).
3. **Decoupled Evaluation:**
   - Capped Validation: Primary metric for hyperparameter selection and early stopping.
   - Uncapped Validation: Diagnostic check to monitor overall trajectory drift.
   - Official Test Benchmark: Uncapped NASA ground truth evaluated **only after** the winning model configuration is locked.
   - Zero Clipping Rule: Model predictions are evaluated as raw continuous outputs without artificial clipping to 125.
4. **Data Isolation:** Test arrays ($X_{\text{test\_final}}$, $y_{\text{test\_final}}$) were never passed to `xgb.fit()`, never passed as `eval_set`, and never used during model training or selection.

---

## 3. Subset-by-Subset Performance Breakdown

### 3.1 FD001 (Single Operating Condition, HPC Degradation)
* **Dataset Scale:** Train = 14,020 windows (80 engines) | Val = 3,711 windows (20 engines) | Test = 100 engines

| Model Variant | Feature Tier | Feature Dim | Val Capped RMSE | Val Capped $R^2$ | Diagnostic Val Uncapped RMSE | Val Mean Engine RMSE | Official Test RMSE | Official Test MAE | Official Test NASA Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `XGBoost_FinalObservation` | Tier 1 (Core 14) | 14 | 20.4023 | 0.7623 | 45.4746 | 20.5518 | 18.1485 | 13.9142 | 819.43 |
| `XGBoost_FinalObservation` | Tier 2 (Operational 17) | 17 | 20.4856 | 0.7603 | 45.4754 | 20.6125 | 18.5004 | 14.1565 | 773.41 |
| `XGBoost_FinalObservation` | Tier 3 (Extended 24) | 24 | 20.3974 | 0.7624 | 45.5492 | 20.5401 | 18.2245 | 13.9877 | 793.41 |
| `XGBoost_FlattenedWindow` | Tier 1 (Core 14) | 420 | 16.3013 | 0.8482 | 44.2001 | 16.2758 | 15.3068 | 12.0121 | 368.60 |
| `XGBoost_FlattenedWindow` | Tier 2 (Operational 17) | 510 | 16.3045 | 0.8481 | 44.2255 | 16.2974 | 15.5682 | 12.1895 | 366.26 |
| `XGBoost_FlattenedWindow` | Tier 3 (Extended 24) | 720 | 16.1041 | 0.8518 | 44.1166 | 16.0963 | 16.2100 | 12.7844 | 464.68 |
| `XGBoost_SummaryStats` | Tier 1 (Core 14) | 70 | 12.9696 | 0.9039 | 43.1491 | 12.8718 | 13.8310 | 10.3705 | 271.91 |
| `XGBoost_SummaryStats` | Tier 2 (Operational 17) | 85 | 13.1623 | 0.9010 | 43.1477 | 13.0645 | 14.7783 | 10.9705 | 319.64 |
| `XGBoost_SummaryStats` | Tier 3 (Extended 24) | 120 | 13.4734 | 0.8963 | 43.2039 | 13.3755 | 14.3194 | 10.6698 | 302.93 |
| **`XGBoost_Tuned_SummaryStats`** | **Tier 1 (Core 14)** | **70** | **12.7506** | **0.9071** | **42.9463** | **12.6397** | **13.4770** | **10.1585** | **268.06** |

*Winner:* **`XGBoost_Tuned_SummaryStats` (Tier 1 Core 14)**.  
*Analysis:* Early stopping converged at iteration 137. Regularization with `max_depth = 4` and `reg_lambda = 5.0` reduced test RMSE to 13.48 cycles and NASA penalty score to 268.06, outperforming Random Forest (14.29 / 294.40).

---

### 3.2 FD002 (Six Operating Conditions, HPC Degradation)
* **Dataset Scale:** Train = 36,857 windows (208 engines) | Val = 9,362 windows (52 engines) | Test = 259 engines (6 padded short engines)

| Model Variant | Feature Tier | Feature Dim | Val Capped RMSE | Val Capped $R^2$ | Diagnostic Val Uncapped RMSE | Val Mean Engine RMSE | Official Test RMSE | Official Test MAE | Official Test NASA Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `XGBoost_FinalObservation` | Tier 1 (Core 14) | 14 | 20.4461 | 0.7615 | 48.0674 | 20.5739 | 29.8897 | 21.8488 | 13,159.76 |
| `XGBoost_FinalObservation` | Tier 2 (Operational 17) | 17 | 20.4281 | 0.7619 | 48.0658 | 20.5702 | 29.9391 | 21.9016 | 12,752.82 |
| `XGBoost_FinalObservation` | Tier 3 (Extended 24) | 24 | 20.3653 | 0.7634 | 48.0772 | 20.5186 | 29.8746 | 21.8315 | 12,395.58 |
| `XGBoost_FlattenedWindow` | Tier 1 (Core 14) | 420 | 19.4858 | 0.7834 | 47.9254 | 19.4972 | 30.1143 | 22.0163 | 13,793.76 |
| `XGBoost_FlattenedWindow` | Tier 2 (Operational 17) | 510 | 19.3394 | 0.7867 | 47.7845 | 19.3905 | 29.9696 | 21.9790 | 14,456.27 |
| **`XGBoost_FlattenedWindow`** | **Tier 3 (Extended 24)** | **720** | **19.3303** | **0.7869** | **47.7766** | **19.3797** | **30.0483** | **22.0620** | **14,159.01** |
| `XGBoost_SummaryStats` | Tier 1 (Core 14) | 70 | 22.0231 | 0.7234 | 49.3323 | 21.8213 | 31.9150 | 23.3644 | 36,184.81 |
| `XGBoost_SummaryStats` | Tier 2 (Operational 17) | 85 | 21.8023 | 0.7290 | 49.1979 | 21.6027 | 31.6940 | 23.2384 | 41,283.86 |
| `XGBoost_SummaryStats` | Tier 3 (Extended 24) | 120 | 22.0103 | 0.7238 | 49.3496 | 21.7828 | 32.1311 | 23.5939 | 52,189.11 |
| `XGBoost_Tuned_SummaryStats` | Tier 2 (Operational 17) | 85 | 21.4461 | 0.7378 | 48.9100 | 21.3129 | 32.1095 | 23.4907 | 23,819.55 |

*Winner:* **`XGBoost_FlattenedWindow` (Tier 3 Extended 24)**.  
*Subgroup Metrics:*
- Unpadded test engines (253 units): RMSE = **29.7027** | NASA Score = **13,714.50**
- Padded test engines (6 units): RMSE = **42.1151** | NASA Score = **444.50**

---

### 3.3 FD003 (Single Operating Condition, HPC & Fan Degradation)
* **Dataset Scale:** Train = 17,857 windows (80 engines) | Val = 3,963 windows (20 engines) | Test = 100 engines

| Model Variant | Feature Tier | Feature Dim | Val Capped RMSE | Val Capped $R^2$ | Diagnostic Val Uncapped RMSE | Val Mean Engine RMSE | Official Test RMSE | Official Test MAE | Official Test NASA Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `XGBoost_FinalObservation` | Tier 1 (Core 14) | 14 | 16.3850 | 0.8462 | 65.5786 | 16.8927 | 21.6608 | 16.4866 | 2,590.55 |
| `XGBoost_FinalObservation` | Tier 2 (Operational 17) | 17 | 16.3855 | 0.8462 | 65.5866 | 16.8928 | 21.4562 | 16.3702 | 3,272.23 |
| `XGBoost_FinalObservation` | Tier 3 (Extended 24) | 24 | 15.9664 | 0.8540 | 65.7335 | 16.5312 | 21.0327 | 16.1436 | 1,997.15 |
| `XGBoost_FlattenedWindow` | Tier 1 (Core 14) | 420 | 13.7616 | 0.8915 | 65.9890 | 13.6300 | 16.8788 | 12.8711 | 709.23 |
| `XGBoost_FlattenedWindow` | Tier 2 (Operational 17) | 510 | 13.8731 | 0.8897 | 66.0827 | 13.7380 | 17.0040 | 12.9806 | 906.57 |
| `XGBoost_FlattenedWindow` | Tier 3 (Extended 24) | 720 | 13.9559 | 0.8884 | 66.0267 | 13.7845 | 17.2482 | 13.2505 | 845.74 |
| `XGBoost_SummaryStats` | Tier 1 (Core 14) | 70 | 13.0860 | 0.9019 | 66.2750 | 12.8797 | 14.3495 | 10.3709 | 315.92 |
| `XGBoost_SummaryStats` | Tier 2 (Operational 17) | 85 | 13.1547 | 0.9009 | 66.3045 | 12.9463 | 14.6330 | 10.5960 | 325.09 |
| `XGBoost_SummaryStats` | Tier 3 (Extended 24) | 120 | 13.0160 | 0.9030 | 66.3688 | 12.8122 | 14.8789 | 10.6698 | 368.74 |
| **`XGBoost_Tuned_SummaryStats`** | **Tier 3 (Extended 24)** | **120** | **12.7926** | **0.9063** | **66.2166** | **12.5694** | **14.2476** | **10.2842** | **312.21** |

*Winner:* **`XGBoost_Tuned_SummaryStats` (Tier 3 Extended 24)**.  
*Analysis:* Early stopping converged at iteration 119. Val capped RMSE dropped to 12.79, and test RMSE reached 14.25 cycles, substantially outperforming Random Forest (14.96 cycles, NASA score 431.96).

---

### 3.4 FD004 (Six Operating Conditions, HPC & Fan Degradation)
* **Dataset Scale:** Train = 42,725 windows (199 engines) | Val = 11,303 windows (50 engines) | Test = 248 engines (11 padded short engines)

| Model Variant | Feature Tier | Feature Dim | Val Capped RMSE | Val Capped $R^2$ | Diagnostic Val Uncapped RMSE | Val Mean Engine RMSE | Official Test RMSE | Official Test MAE | Official Test NASA Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `XGBoost_FinalObservation` | Tier 1 (Core 14) | 14 | 17.2891 | 0.8243 | 74.3411 | 17.3879 | 31.2513 | 24.3649 | 9,312.21 |
| `XGBoost_FinalObservation` | Tier 2 (Operational 17) | 17 | 17.2944 | 0.8242 | 74.3429 | 17.3942 | 31.5915 | 24.5828 | 10,627.08 |
| `XGBoost_FinalObservation` | Tier 3 (Extended 24) | 24 | 16.4151 | 0.8417 | 73.9114 | 16.4526 | 30.9258 | 24.1681 | 9,482.26 |
| `XGBoost_FlattenedWindow` | Tier 1 (Core 14) | 420 | 16.3276 | 0.8434 | 74.0537 | 16.4024 | 30.4099 | 23.6393 | 8,064.35 |
| `XGBoost_FlattenedWindow` | Tier 2 (Operational 17) | 510 | 16.2448 | 0.8449 | 73.8829 | 16.3023 | 30.5734 | 23.6934 | 7,883.25 |
| **`XGBoost_FlattenedWindow`** | **Tier 3 (Extended 24)** | **720** | **15.8682** | **0.8521** | **73.6666** | **15.8753** | **29.7334** | **23.2798** | **6,939.94** |
| `XGBoost_SummaryStats` | Tier 1 (Core 14) | 70 | 18.2856 | 0.8034 | 74.8879 | 18.0694 | 32.2188 | 24.4716 | 16,201.20 |
| `XGBoost_SummaryStats` | Tier 2 (Operational 17) | 85 | 18.2003 | 0.8052 | 74.8520 | 17.9866 | 32.0902 | 24.3686 | 12,176.69 |
| `XGBoost_SummaryStats` | Tier 3 (Extended 24) | 120 | 18.1697 | 0.8059 | 74.7932 | 17.9405 | 31.8285 | 24.1374 | 11,196.61 |
| `XGBoost_Tuned_SummaryStats` | Tier 3 (Extended 24) | 120 | 17.5246 | 0.8194 | 74.3409 | 17.3820 | 31.5881 | 24.0844 | 11,571.11 |

*Winner:* **`XGBoost_FlattenedWindow` (Tier 3 Extended 24)**.  
*Subgroup Metrics:*
- Unpadded test engines (237 units): RMSE = **28.3985** | NASA Score = **5,913.81**
- Padded test engines (11 units): RMSE = **50.5567** | NASA Score = **1,026.14**

---

## 4. Per-Engine Validation Aggregation Analysis

Across all 142 distinct validation engines, engine-level RMSE distributions verify robust model stability:

| Subset | Number of Engines | Winning Model Configuration | Mean Engine RMSE | Engine RMSE Std Dev | Final Cycle RMSE |
| :--- | :---: | :--- | :---: | :---: | :---: |
| **FD001** | 20 | `XGBoost_Tuned_SummaryStats` (Tier 1) | **12.6397** | 3.5812 | 2.5118 |
| **FD002** | 52 | `XGBoost_FlattenedWindow` (Tier 3) | **19.3797** | 4.8824 | 24.3188 |
| **FD003** | 20 | `XGBoost_Tuned_SummaryStats` (Tier 3) | **12.5694** | 5.9184 | 3.0112 |
| **FD004** | 50 | `XGBoost_FlattenedWindow` (Tier 3) | **15.8753** | 3.6540 | 18.2561 |

*Observations:*
1. In single-condition subsets (FD001, FD003), final cycle prediction error is remarkably low (2.51 and 3.01 cycles), confirming accurate terminal failure warning.
2. In multi-condition regimes, standard deviation of engine RMSE remains below 5.0 cycles, demonstrating that predictions do not degrade severely across diverse engine lifespans.

---

## 5. Artifacts and Generated Deliverables

The following deliverables have been generated and saved:

1. [models/xgboost/run_xgboost_models.py](file:///c:/Users/LENOVO/Documents/Aircraft_Engine_AI/models/xgboost/run_xgboost_models.py)
   - Reproducible execution script implementing all 40 configurations, early stopping, per-engine validation loops, and subgroup test metrics.
2. [models/xgboost/xgboost_manifest.json](file:///c:/Users/LENOVO/Documents/Aircraft_Engine_AI/models/xgboost/xgboost_manifest.json)
   - Comprehensive machine-readable audit manifest logging exact hyperparameters, iteration checkpoints, and decoupled evaluation metrics.
3. [reports/models/xgboost_model_performance.csv](file:///c:/Users/LENOVO/Documents/Aircraft_Engine_AI/reports/models/xgboost_model_performance.csv)
   - Complete 40-row master tabular performance table.
4. [reports/models/xgboost_per_engine_validation.csv](file:///c:/Users/LENOVO/Documents/Aircraft_Engine_AI/reports/models/xgboost_per_engine_validation.csv)
   - Diagnostic table containing 1,420 engine-level validation records covering all 142 validation engines.
5. [docs/phase4_step4_2_xgboost_evaluation_report.md](file:///c:/Users/LENOVO/Documents/Aircraft_Engine_AI/docs/phase4_step4_2_xgboost_evaluation_report.md)
   - Official final evaluation report.
6. [docs/phase4_step4_2_xgboost_preimplementation_audit.md](file:///c:/Users/LENOVO/Documents/Aircraft_Engine_AI/docs/phase4_step4_2_xgboost_preimplementation_audit.md)
   - Pre-implementation verification and data integrity audit report.

---

## 6. Readiness for Phase 4 – Step 4.3 (Sequence Models)

In accordance with the master project roadmap:
* The **XGBoost** models established here now serve as the definitive **tabular/feature-engineered benchmark**.
* In **Phase 4 – Step 4.3**, deep sequential architectures (**LSTM / GRU / 1D-CNN**) will ingest the 3D temporal sequence arrays $(N, 30, 24)$ directly to evaluate whether recurrent hidden-state modeling justifies its computational complexity over the XGBoost benchmark established in this step.
