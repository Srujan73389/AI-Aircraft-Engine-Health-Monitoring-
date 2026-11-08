# Phase 8 — Step 8.8 Deliverable Report: Digital Twin State and Scenario Outputs Display

**Milestone Reference**: Master Project Roadmap PDF, Section 9 (Phase 8: Human-in-the-Loop Web Interface & API Integration), Step 8.8.  
**Subsystem Reference**: Phase 5 (Digital Twin State Vector Construction & Scenario Perturbation, Steps 5.1–5.11) and Phase 7 (Safety Verification & Closed-Loop Governor, Steps 7.1–7.9).  
**Status**: Certified & Passed (405/405 Automated Regression Tests Passing).

---

## 1. Executive Summary

Phase 8 Step 8.8 implements the **Digital Twin State & Scenario Display Subsystem** for the mission control interface. The system provides an interactive twin-spool turbofan engine cutaway schematic, a full 40-dimensional state vector breakdown with rigorous 4-tier visual provenance classification, an interactive What-If scenario sandbox (strictly bounded to Phase 5 supported perturbation scenarios), and a closed-loop safety-governed baseline vs. simulated candidate comparison matrix.

All simulated scenario candidate states are strictly isolated from recorded NASA C-MAPSS dataset baselines, with explicit visual badges and mandatory academic simulation notices ensuring zero ambiguity between real physical telemetry and hypothetical perturbations.

---

## 2. Key Architecture & Deliverables

### 2.1 Interactive Turbofan Engine Cutaway Schematic
- **Component File**: [`frontend/src/components/dashboard/DigitalTwinSimulationPanel.jsx`](file:///c:/Users/LENOVO/Documents/Aircraft_Engine_AI/frontend/src/components/dashboard/DigitalTwinSimulationPanel.jsx)
- **High-Fidelity SVG Aerothermal Stations**:
  - **Station 2 (Fan Inlet)**: Inlet total temperature ($T_2$, `sensor_1`) and total pressure ($P_2$, `sensor_5`).
  - **Station 1.5 (Bypass Duct)**: Fan bypass airflow channel, pressure ($P_{15}$, `sensor_6`), and bypass ratio ($\text{BPR}$, `sensor_15`).
  - **Station 2.4 (LPC Outlet)**: Intermediate compressor discharge total temperature ($T_{24}$, `sensor_2`).
  - **Station 3 (HPC Outlet)**: High-pressure compressor exit total temperature ($T_{30}$, `sensor_3`), total pressure ($P_{30}$, `sensor_7`), static wall pressure ($P_{s30}$, `sensor_11`), HPC temperature rise ($\Delta T_{\text{HPC}}$, $G_{27}$), and Overall Pressure Ratio ($\text{OPR}$, $G_{28}$).
  - **Station 4 (Combustor / Burner)**: Fuel-air ratio ($\text{farB}$, `sensor_16`) and fuel flow ratio ($\phi$, `sensor_12`).
  - **Station 4.5 (HPT Core Spool)**: High-pressure turbine core rotational speed ($N_c$, `sensor_9`), corrected core speed ($NR_c$, `sensor_14`), and coolant bleed flow ($W_{31}$, `sensor_20`).
  - **Station 5 (LPT Outlet / EGT)**: Low-pressure turbine discharge total temperature ($T_{50}$, `sensor_4`), fan spool speeds ($N_f$, `sensor_8`, $NR_f$, `sensor_13`), coolant bleed ($W_{32}$, `sensor_21`), and spool speed ratio ($N_c / N_f$, $G_{29}$).
- **Interactive Station Hotspots**:
  - Pulsing active station halo markers.
  - Quick telemetry readout card displaying station name, thermodynamic description, sensor values with engineering units, and dataset provenance tags.

---

### 2.2 4-Tier State Vector Visual Provenance Classification
The 40-variable state vector ($G_1$ to $G_{40}$) is systematically categorized and rendered with color-coded badges to prevent confusion:
1. **`MEASURED / DATASET` (Blue Badge)**:
   - Real NASA C-MAPSS recorded telemetry ($G_1$ to $G_{26}$): flight operational settings, ambient conditions, and 21 gas-path sensor channels.
2. **`PHYSICAL / DERIVED` (Emerald Badge)**:
   - First-principles physical indicators ($G_{27}$ to $G_{31}$, $G_{37}$, $G_{38}$): $\Delta T_{\text{HPC}}$, $\text{OPR}$, spool speed ratio, prognostic spread, and Virtual Health Index (VHI).
3. **`MODEL / PREDICTED` (Purple Badge)**:
   - Machine learning diagnostic and prognostic inferences ($G_{32}$ to $G_{36}$): Isolation Forest score, PCA-SPE residual, 3-cycle persistence flag, XGBoost RUL, and Bi-LSTM RUL.
4. **`SIMULATED / SCENARIO` (Amber Badge)**:
   - Hypothetical synthetic perturbations ($G_{39}$ to $G_{40}$): `thermal_stress_offset` and `sensor_bias_delta`.

---

### 2.3 Interactive What-If Scenario Sandbox
- **Supported Perturbation Modes (Strictly Phase 5 Certified)**:
  1. `THERMAL_PERTURBATION` ($G_{39}$): Hypothetical gas-path thermal stress offset ($\Delta T \in [-5.0, +30.0]^\circ\text{R}$) to evaluate aerothermal safety margin.
  2. `SENSOR_BIAS_INJECTION` ($G_{40}$): Hypothetical sensor instrumentation calibration drift ($\Delta \in [-20.0, +20.0]$) targeting $T_{30}$, $T_{24}$, $T_{50}$, $P_{30}$, or $NR_c$.
  *(Note: Accelerated Aging is strictly excluded in adherence to Phase 5 scope).*
- **Closed-Loop Safety Gating**:
  - Operator signoff authorization checkbox for high-impact candidate states.
  - Integration with backend `POST /api/v1/simulation/what-if` via [`ClosedLoopSafetyGovernor`](file:///c:/Users/LENOVO/Documents/Aircraft_Engine_AI/safety/closed_loop.py).

---

### 2.4 Baseline vs. Simulated Comparison Matrix
- **Governor Decision Banner**:
  - Displays decision status: `ACCEPTED` (emerald), `REQUIRES_REVIEW` (amber), or `REJECTED` (rose).
  - State update indicator: `YES (APPLIED)` or `NO (BLOCKED)`.
  - Governor reason summary and unique audit action ID (`act_whatif_...`).
- **Telemetry Delta Comparison Table**:
  - Direct side-by-side comparison of baseline measured values vs. simulated candidate values.
  - Explicit delta calculation ($\Delta$) with directionality indicators.
  - Safe tolerance verification envelope status.
  - Recovery advisory plan display when safety intervention is triggered.

---

### 2.5 Academic Non-Flight Operations Notice
- Clear, persistent banner:
  > *"All Digital Twin What-If scenario perturbations (thermal stress, sensor bias) are synthetic candidate states evaluated inside a simulated virtual sandbox. Simulated outputs are strictly isolated from recorded NASA C-MAPSS dataset baselines and must never be interpreted as certified hardware telemetry or applied to live aircraft control systems."*

---

## 3. Test Verification & Zero-Tampering Audit

The independent test suite [`tests/test_digital_twin_display_step8_8.py`](file:///c:/Users/LENOVO/Documents/Aircraft_Engine_AI/tests/test_digital_twin_display_step8_8.py) executes 10 comprehensive tests:

| Test Case | Description | Status |
| :--- | :--- | :--- |
| `test_01_digital_twin_panel_component_exists` | Asserts component file existence and clean exports | **PASS** |
| `test_02_turbofan_schematic_stations_rendered` | Validates all 7 aerothermal stations rendered in SVG | **PASS** |
| `test_03_provenance_badges_distinction` | Verifies 4-tier visual provenance badge distinctions | **PASS** |
| `test_04_what_if_controls_supported_scenarios` | Ensures only Phase 5 certified scenarios are available | **PASS** |
| `test_05_baseline_vs_simulated_comparison_matrix` | Validates side-by-side baseline vs. simulated table and deltas | **PASS** |
| `test_06_simulated_values_labeled_separately` | Confirms explicit synthetic labeling and non-flight notice | **PASS** |
| `test_07_backend_simulation_what_if_integration` | Verifies backend `POST /api/v1/simulation/what-if` integration | **PASS** |
| `test_08_safety_status_badge_in_comparison` | Asserts safety governor decision badges and recovery plan | **PASS** |
| `test_09_app_integration_digital_twin_panel` | Validates mounting in root `App.jsx` dashboard | **PASS** |
| `test_10_dataset_and_model_immutability` | Cryptographic SHA-256 integrity of all raw files, scalers, and models | **PASS** |

### Full Test Suite Regression
- Total Tests: **405 / 405 passing** (100% green across all 8 phases).
- Cryptographic SHA-256 integrity: Zero tampering verified across 12 raw dataset files, 4 scalers, and 8 champion checkpoints.
