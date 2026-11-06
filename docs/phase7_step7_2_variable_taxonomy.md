# Phase 7 — Step 7.2: Variable Taxonomy & Provenance Classification

## 1. Executive Summary

In aerospace digital twin engineering, maintaining unambiguous provenance for every variable is vital for flight-critical safety, airworthiness verification, and regulatory auditability. 

Step 7.2 establishes a formal four-tier provenance categorization system for the virtual engine state vector ($G1 \dots G40$) and associated configuration metadata. By explicitly partitioning variables into **`MEASURED`**, **`PREDICTED`**, **`SIMULATED`**, and **`MANUALLY_CONFIGURED`** categories, the Digital Twin enforces strict safety firewalls: no machine learning model or simulated perturbation can overwrite or corrupt measured physical telemetry.

All definitions and verification logic reside in [`safety/taxonomy.py`](file:///c:/Users/LENOVO/Documents/Aircraft_Engine_AI/safety/taxonomy.py).

---

## 2. Four-Tier Provenance Taxonomy

| Provenance Category | Definition & Scope | Telemetry Immutability | Mathematical / Physical Origin |
| :--- | :--- | :--- | :--- |
| **`MEASURED`** | Direct physical measurements acquired by physical engine instrumentation during operation. | **`True` (Strictly Immutable)** | Thermocouples, pyrometers, strain gauges, pressure transducers, and phonic wheel speed pickups ($G6 \dots G26$). |
| **`PREDICTED`** | Synthetic estimations output by machine learning algorithms, statistical models, or degradation curves. | `False` (Modifiable by Agents) | XGBoost regressor ($G35$), Bi-LSTM sequential neural network ($G36$), Isolation Forest ($G32$), PCA-SPE residual energy ($G33$), and Health Index $H(t)$ ($G38$). |
| **`SIMULATED`** | Synthetically computed state representations derived from physical equations, state space models, or what-if perturbation injections. | `False` (Modifiable by Twin) | Macroscopic derived thermodynamic indicators ($G27 \dots G29$), sliding window tensors ($G30 \dots G31$), and what-if stress offsets ($G39 \dots G40$). |
| **`MANUALLY_CONFIGURED`** | Fixed operational flight conditions, mission profile settings, scenario flags, and algorithmic hyperparameters. | `False` (Configurable by Operator) | Engine ID ($G1$), flight cycle clock ($G2$), flight altitude/Mach/TRA command ($G3 \dots G5$), sampling rates, and alert threshold bands. |

---

## 3. Canonical Variable Classification Matrix ($G1 \dots G40$ + Auxiliaries)

| Slot | Variable Identifier | Canonical Name | Units | Provenance Category | Immutable in Twin | Source Model / Equation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **G1** | `unit_number` | Engine Unit Serial | dimensionless | `MANUALLY_CONFIGURED` | `False` | Fleet registration tracking |
| **G2** | `time_cycles` | Flight Cycle Index | cycles | `MANUALLY_CONFIGURED` | `False` | Operational cycle clock |
| **G3** | `altitude` | Pressure Altitude | $\text{kft}$ | `MANUALLY_CONFIGURED` | `False` | Mission flight profile schedule |
| **G4** | `mach_number` | Flight Mach Number | dimensionless | `MANUALLY_CONFIGURED` | `False` | FMS target flight speed |
| **G5** | `throttle_resolver_angle` | Throttle Resolver Angle | $\text{TRA \%}$ | `MANUALLY_CONFIGURED` | `False` | Cockpit power lever command |
| **G6** | `T2` | Fan Inlet Total Temp | $^\circ\text{R}$ | `MEASURED` | **`True`** | Fan inlet total temperature probe |
| **G7** | `T24` | LPC Outlet Total Temp | $^\circ\text{R}$ | `MEASURED` | **`True`** | Low-pressure compressor exit rake |
| **G8** | `T30` | HPC Outlet Total Temp | $^\circ\text{R}$ | `MEASURED` | **`True`** | High-pressure compressor exit thermocouple |
| **G9** | `T50` | LPT Outlet Total Temp | $^\circ\text{R}$ | `MEASURED` | **`True`** | Exhaust gas temperature (EGT) pyrometer |
| **G10** | `P2` | Fan Inlet Total Pressure | $\text{psia}$ | `MEASURED` | **`True`** | Inlet pitot-static transducer |
| **G11** | `P15` | Bypass Duct Total Pressure | $\text{psia}$ | `MEASURED` | **`True`** | Bypass duct stagnation pressure sensor |
| **G12** | `P30` | HPC Outlet Total Pressure | $\text{psia}$ | `MEASURED` | **`True`** | Combustor casing total pressure sensor |
| **G13** | `Nf` | Fan Spool Physical Speed | $\text{rpm}$ | `MEASURED` | **`True`** | Low-pressure shaft eddy current pickup |
| **G14** | `Nc` | Core Spool Physical Speed | $\text{rpm}$ | `MEASURED` | **`True`** | High-pressure shaft phonic wheel pickup |
| **G15** | `epr` | Engine Pressure Ratio | dimensionless | `MEASURED` | **`True`** | Differential pressure ratio transmitter |
| **G16** | `Ps30` | HPC Outlet Static Pressure | $\text{psia}$ | `MEASURED` | **`True`** | HPC exit wall static pressure tap |
| **G17** | `phi` | Fuel Flow Ratio $(W_f / P_{s30})$ | $\text{pps/psia}$ | `MEASURED` | **`True`** | Fuel meter scaled by static pressure |
| **G18** | `NRf` | Corrected Fan Speed | $\text{rpm}$ | `MEASURED` | **`True`** | $N_f / \sqrt{\theta_2}$ |
| **G19** | `NRc` | Corrected Core Speed | $\text{rpm}$ | `MEASURED` | **`True`** | $N_c / \sqrt{\theta_{25}}$ |
| **G20** | `BPR` | Bypass Ratio | dimensionless | `MEASURED` | **`True`** | Bypass-to-core airflow mass split |
| **G21** | `farB` | Burner Fuel-Air Ratio | dimensionless | `MEASURED` | **`True`** | Combustor fuel-to-air mass ratio |
| **G22** | `htBleed` | Bleed Enthalpy | dimensionless ratio | `MEASURED` | **`True`** | Thermal management bleed enthalpy |
| **G23** | `Nf_dmd` | Demanded Fan Speed | $\text{rpm}$ | `MEASURED` | **`True`** | FADEC demanded fan speed setpoint |
| **G24** | `PCNfR_dmd` | Demanded Corrected Fan Speed | $\%$ | `MEASURED` | **`True`** | FADEC closed-loop reference |
| **G25** | `W31` | HPT Coolant Bleed Flow | $\text{lbm/s}$ | `MEASURED` | **`True`** | Turbine cooling valve flow telemetry |
| **G26** | `W32` | LPT Coolant Bleed Flow | $\text{lbm/s}$ | `MEASURED` | **`True`** | Low-pressure cooling bleed telemetry |
| **G27** | `hpc_temp_rise` | HPC Temperature Rise | $^\circ\text{R}$ | `SIMULATED` | `False` | $\Delta T = T_{30} - T_{24}$ |
| **G28** | `overall_pressure_ratio` | Overall Pressure Ratio | dimensionless | `SIMULATED` | `False` | $\text{OPR} = P_{30} / P_2$ |
| **G29** | `spool_speed_ratio` | Spool Speed Ratio | dimensionless | `SIMULATED` | `False` | $\text{Spool Ratio} = N_c / N_f$ |
| **G30** | `normalized_feature_vec` | Scaled Feature Array | $[0, 1]$ | `SIMULATED` | `False` | `SafeMinMaxScaler(G3-G26)` |
| **G31** | `temporal_seq_tensor` | Temporal Sequence Tensor | $[0, 1]$ | `SIMULATED` | `False` | 30-cycle temporal sliding window $[30 \times D]$ |
| **G32** | `anomaly_score_m1` | iForest Anomaly Score | anomaly score | `PREDICTED` | `False` | `IsolationForest.score_samples()` |
| **G33** | `anomaly_score_m2` | PCA-SPE Residual Energy | residual ($Q$-stat) | `PREDICTED` | `False` | $\text{SPE} = \|x - \hat{x}\|^2$ |
| **G34** | `anomaly_alert_flag` | 3-Cycle Persistence Alarm | binary $\{0, 1\}$ | `PREDICTED` | `False` | Persistence filter: $\sum_{t-2}^t \text{alarm} \ge 3$ |
| **G35** | `predicted_rul_xgboost` | XGBoost RUL Prediction | cycles | `PREDICTED` | `False` | Phase 4 XGBoost Champion Regressor |
| **G36** | `predicted_rul_lstm` | Bi-LSTM RUL Prediction | cycles | `PREDICTED` | `False` | Phase 4 PyTorch Bi-LSTM Champion Regressor |
| **G37** | `prognostic_spread` | Epistemic Uncertainty Spread | cycles | `PREDICTED` | `False` | $|\text{RUL}_{\text{xgb}} - \text{RUL}_{\text{lstm}}|$ |
| **G38** | `virtual_health_index` | Virtual Health Index $H(t)$ | index $[0, 1]$ | `PREDICTED` | `False` | Continuous health index degradation curve |
| **G39** | `thermal_stress_offset` | Hypothetical Thermal Stress | $^\circ\text{R}$ | `SIMULATED` | `False` | Synthetic perturbation $\Delta T$ injection |
| **G40** | `sensor_bias_delta` | Hypothetical Sensor Bias | sensor units | `SIMULATED` | `False` | Synthetic sensor drift injection dictionary |
| **CFG** | `sampling_frequency_hz` | Sampling Frequency | $\text{Hz}$ | `MANUALLY_CONFIGURED` | `False` | FADEC bus rate configuration |
| **CFG** | `baseline_subset` | Operational Subset | string | `MANUALLY_CONFIGURED` | `False` | Fleet deployment profile (FD001–FD004) |
| **CFG** | `caution_threshold_t30` | T30 Caution Limit | $^\circ\text{R}$ | `MANUALLY_CONFIGURED` | `False` | Airline maintenance policy limit ($1605^\circ\text{R}$) |
| **CFG** | `warning_threshold_t30` | T30 Warning Limit | $^\circ\text{R}$ | `MANUALLY_CONFIGURED` | `False` | OEM operating manual hard ceiling ($1625^\circ\text{R}$) |

---

## 4. Verification and Enforcement Mechanics

1. **State Partitioning (`classify_state_variables`)**:
   Inspects an `EngineState` or telemetry dictionary and automatically partitions its variables into `measured`, `predicted`, `simulated`, `manually_configured`, and `unclassified` sub-dictionaries.

2. **Safety Firewall Verification (`verify_provenance_immutability`)**:
   Compares an original state against an updated state. If any measured sensor ($G6 \dots G26$) differs by more than $10^{-9}$ in relative or absolute tolerance, a [`ProvenanceViolationError`](file:///c:/Users/LENOVO/Documents/Aircraft_Engine_AI/safety/taxonomy.py) is raised immediately, halting downstream execution and preserving flight-data integrity.

---

## 5. Verification Test Suite Coverage

The test suite [`tests/test_variable_taxonomy_step7_2.py`](file:///c:/Users/LENOVO/Documents/Aircraft_Engine_AI/tests/test_variable_taxonomy_step7_2.py) confirms:
- `test_01_taxonomy_registry_completeness`: 100% of G1–G40 and auxiliary variables registered.
- `test_02_category_enumeration_coverage`: All 4 categories (`MEASURED`, `PREDICTED`, `SIMULATED`, `MANUALLY_CONFIGURED`) verified.
- `test_03_measured_immutability_flags`: Strictly all 21 measured sensors ($G6 \dots G26$) enforce `is_immutable_in_twin = True`.
- `test_04_predicted_variable_classification`: RUL models and anomaly scores verified as `PREDICTED`.
- `test_05_simulated_variable_classification`: Derived aerodynamic indicators and scenario offsets verified as `SIMULATED`.
- `test_06_manually_configured_classification`: Flight context and operating thresholds verified as `MANUALLY_CONFIGURED`.
- `test_07_classify_state_variables_execution`: State vector partitioning verified on realistic engine states.
- `test_08_verify_provenance_immutability_pass`: Safe enrichment (adding RUL and health index) passes without violation.
- `test_09_verify_provenance_immutability_fail`: Direct tampering with measured sensor raises `ProvenanceViolationError`.
- `test_10_dataset_and_model_immutability`: Bit-for-bit cryptographic SHA-256 invariance confirmed for 12 raw files, 4 scalers, and 8 champion checkpoints.
