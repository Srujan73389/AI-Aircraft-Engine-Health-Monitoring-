# Phase 5 — Step 5.7: Controlled What-If Scenario Interface

**Project:** AI-Based Aircraft Engine Health Monitoring, Fault Diagnosis and Predictive Maintenance  
**Roadmap Reference:** Phase 5 (Aircraft Engine Digital Twin) — Step 5.7: Add a scenario interface for controlled what-if changes.  
**Dataset:** NASA C-MAPSS Turbofan Engine Degradation Simulation Dataset (`FD001`, `FD002`, `FD003`, `FD004`)  
**Status:** **Implemented, Validated, and Formally Verified.**

---

## 1. Executive Summary & Purpose

Step 5.7 provides a controlled, reproducible **What-If Scenario Interface** for the Aircraft Engine Digital Twin prototype. In aerospace condition monitoring and predictive fleet maintenance, fleet engineers analyze operational what-if hypotheses:
- *"What is the impact on predicted Remaining Useful Life (RUL) if the high-pressure compressor exit temperature ($T_{30}$) operates $+35^\circ\text{R}$ above nominal?"*
- *"How does a $+15\text{ psia}$ transducer bias on $P_{30}$ affect multi-model prognostic spread between XGBoost and LSTM?"*
- *"How should an accelerated operational usage scenario be represented without corrupting recorded temporal sequences or inventing ungrounded physics?"*

The Step 5.7 interface provides an academic simulation framework to address these questions **without mutating raw C-MAPSS data, corrupting measured telemetry, modifying Phase 4 models, or introducing temporal gaps into ML prognostic inference**.

> [!IMPORTANT]
> **Academic Simulation Scope & Boundaries**
> - The what-if scenario interface is an **analytical and exploratory simulation tool**, not an aircraft flight control or FADEC system.
> - **Simulation-Only Requirement:** All scenario configurations require `is_simulation_only=True`. Any attempt to disable this flag is rejected.
> - **Zero In-Place Telemetry Mutation:** Original recorded C-MAPSS sensor readings ($G6–G26$) are never overwritten in `EngineState` or `raw_history`.
> - **Accelerated Aging Deferral:** Physical microscopic component degradation laws (blade erosion, crack propagation, HPC flow capacity modifiers) are unobservable in C-MAPSS. Furthermore, cycle skipping via temporal subsampling is mathematically invalid for Phase 4 sequence-based models (LSTM/XGBoost). Accelerated aging execution in the simulator is **formally deferred and disabled with an explicit error**.

---

## 2. Scenario Architecture & Simulation Pathway

The Digital Twin maintains an isolated, simulation-only pathway for hypothetical scenarios to guarantee strict separation between **ground-truth measured telemetry** and **hypothetical simulation perturbations**.

```
                C-MAPSS Telemetry Record: G3 - G26
                                │
                                ▼
                 [Raw Measured Telemetry: x_raw(t)]
                                │
       ┌────────────────────────┴────────────────────────┐
       ▼                                                 ▼
[Preserved Ground Truth]                        [Active ScenarioConfig?]
  - EngineState.G6..G26                           - Category: Thermal Stress,
  - raw_history buffer                              Sensor Bias, or Aging
  - Zero In-Place Mutation                        - Strict Input Validation
                                                  - Canonical Bounds Enforced
                                                         │
                                                         ▼
                                            [ScenarioModifier.apply_scenario()]
                                              - sim_raw_24 = copy(raw_24)
                                              - Add bounded perturbation delta
                                              - Thermodynamic sanity check (> 0)
                                              - Generate complete audit record
                                                         │
                                                         ▼
                                            [Phase 3 SafeMinMaxScaler]
                                              - sim_norm_24 = scaler.transform(...)
                                                         │
                                                         ▼
                                            [Tier-Aware Windowing Buffer]
                                              - G30: Normalized Feature Vector
                                              - G31: Consecutive Temporal Sequence Tensor (30 x C)
                                                         │
                                                         ▼
                                            [Phase 4 Prognostics Model Manager]
                                              - G35: XGBoost Predicted RUL
                                              - G36: LSTM Predicted RUL
                                              - G37: Prognostic Spread |xgb - lstm|
                                              - G39: Thermal Stress Offset (ΔT_stress)
                                              - G40: Sensor Bias Delta (Δs_bias)
                                              - EngineState.scenario_audit retained
```

### 2.1 Architectural Invariants
1. **Measured Telemetry Invariance:** Fields `state.T2`, `state.T24`, `state.T30`, `state.T50`, `state.P30`, etc. ($G6–G26$) always reflect the **unaltered recorded C-MAPSS sensor values**.
2. **Dedicated State Entries:**
   - **G39 (`thermal_stress_offset`):** Captures $\Delta T_{stress}$ in $^\circ\text{R}$ (scalar float).
   - **G40 (`sensor_bias_delta`):** Captures $\Delta s_{bias}$ as a dictionary `{channel: delta}` in native engineering units.
3. **ML Inference Coupling:** When a what-if scenario is active, `temporal_seq_tensor` ($G31$) and `normalized_feature_vec` ($G30$) are transformed using the simulated telemetry $\mathbf{x}_{sim}(t)$, allowing Phase 4 ML models to predict RUL under the scenario without retuning or retraining.
4. **Baseline Invariance:** When `scenario=None` (or after invoking `clear_scenario()`), the simulator executes the unperturbed baseline trajectory. In regression verification tests (`test_20`), an independent unperturbed simulator was compared cycle-by-cycle against a simulator initialized with `scenario=None` across 35 recorded flight cycles of engine 1 (spanning both left-edge replication-padded and full unpadded rolling history windows). The exact tested coverage comprises $G1–G31$ and $G35–G37$: tracking metadata ($G1–G2$), operating settings ($G3–G5$), gas-path telemetry ($G6–G26$), macroscopic consistency indicators ($G27–G29$), normalized feature vectors ($G30$), temporal sequence tensors ($G31$), and Phase 4 ML prognostics ($G35–G37$: XGBoost RUL, LSTM RUL, prognostic spread), which were verified to match within strict numerical tolerances ($10^{-7}$ for arrays, $10^{-5}$ for predictions). Note that anomaly detection variables $G32–G34$ (Isolation Forest score, PCA-SPE score, and persistence alert flag) belong to unsupervised anomaly detection models and are not compared by `test_20`. Scenario-specific fields (`thermal_stress_offset` $G39$, `sensor_bias_delta` $G40$, and `scenario_audit`) remained strictly `None` across all cycles.
5. **Scenario Audit Retention:** Every scenario-generated `EngineState` and simulator instance retains a complete audit dictionary documenting the baseline value, applied modification, resulting simulated value, units, and simulation-only designation.

### 2.2 Telemetry vs. Representation Invariants
- **Gas-Path Measured Telemetry ($G6–G26$):** Sensor fields `T2`, `T24`, `T30`, `T50`, `P2`, `P15`, `P30`, `Nf`, `Nc`, etc., preserve the **unmodified physical telemetry** recorded in the NASA C-MAPSS dataset.
- **Macroscopic Consistency Indicators ($G27–G29$):** HPC temperature rise ($G27 = T_{30} - T_{24}$), overall compression ratio ($G28 = P_{30}/P_2$), and spool speed ratio ($G29 = N_c/N_f$) **remain calculated from the original measured telemetry** from the recorded flight cycle snapshot. They represent ground-truth physical relationships and are not altered by what-if scenario perturbations.
- **Composite Computational Representations ($G30, G31$):** The normalized feature vector ($G30$) and 30-cycle temporal sequence tensor ($G31$) **use scenario-modified normalized telemetry** ($\mathbf{x}_{sim}(t)$ transformed via the fitted Phase 3 `SafeMinMaxScaler`) when a scenario is active. This allows Phase 4 data-driven models (XGBoost $\widehat{RUL}_{xgb}$ and LSTM $\widehat{RUL}_{lstm}$) to evaluate prognostic sensitivity under hypothetical operational conditions while keeping measured sensor readings and macroscopic indicators uncorrupted.

---

## 3. Supported Scenario Categories & Canonical Validation Bounds

Step 5.7 implements three structured scenario categories defined in `digital_twin/scenario_interface.py`:

```python
class ScenarioCategory(str, Enum):
    THERMAL_STRESS = "thermal_stress"
    SENSOR_BIAS = "sensor_bias"
    ACCELERATED_AGING = "accelerated_aging"
```

### 3.1 Category A: Thermal Stress Scenario
- **Physical Motivation:** Models thermal excursions, elevated turbine cooling degradation, hot-day takeoff stress, or combustor pattern factor degradation.
- **Affected Channels:** High-pressure compressor exit ($T_{30}$ / `sensor_3`), low-pressure turbine exit ($T_{50}$ / `sensor_4`), LPC exit ($T_{24}$ / `sensor_2`), and fan inlet ($T_2$ / `sensor_1`).
- **Engineering Unit:** Rankine ($^\circ\text{R}$).
- **Canonical Validation Bounds:** Defined in `THERMAL_STRESS_TARGETS` as exactly $[-100.0^\circ\text{R}, +200.0^\circ\text{R}]$.
- **Canonical Bounds Enforcement:** The validator **strictly enforces** this canonical range. Any attempt to broaden or customize the allowed range via `validation_range` is rejected with `ValueError`.
- **Physical Constraint:** $T_{sim} = T_{measured} + \Delta T_{stress} > 0^\circ\text{R}$.
- **State Registration:** Recorded directly in **G39** (`thermal_stress_offset`).

### 3.2 Category B: Sensor Bias Scenario
- **Physical Motivation:** Simulates transducer calibration drift, pneumatic sensing line leaks, thermocouple junction degradation, or signal conditioning offset errors.
- **Affected Channels:** Any of the 24 C-MAPSS operational settings or gas-path sensor channels (or their physical aliases: `altitude`, `mach_number`, `TRA`, `T2`, `T24`, `T30`, `T50`, `P2`, `P15`, `P30`, `Nf`, `Nc`, `epr`, `Ps30`, `phi`, `NRf`, `NRc`, `BPR`, `farB`, `htBleed`, `Nf_dmd`, `PCNfR_dmd`, `W31`, `W32`).
- **Canonical Validation Limits (from `SENSOR_BIAS_LIMITS`):**
  - **Pressures ($P_2, P_{15}$):** Unit `psia`, canonical bounds $[-20.0, +50.0]\text{ psia}$. Strictly requires $P_{sim} > 0$.
  - **Pressures ($P_{30}, P_{s30}$):** Unit `psia`, canonical bounds $[-50.0, +100.0]\text{ psia}$. Strictly requires $P_{sim} > 0$.
  - **Spool Speeds ($N_f, N_c, NR_f, NR_c, Nf_{dmd}$):** Unit `rpm`, canonical bounds $[-500.0, +500.0]\text{ rpm}$. Strictly requires $N_{sim} > 0$.
  - **Flight Settings:**
    - Altitude: Unit `kft`, canonical bounds $[-10.0, +10.0]\text{ kft}$.
    - Mach number: Unit `Mach` / `dimensionless`, canonical bounds $[-0.2, +0.2]$.
    - Throttle Resolver Angle (TRA): Unit `%`, canonical bounds $[-20.0, +20.0]\%$.
  - **Gas-Path Ratios & Flows:**
    - Engine Pressure Ratio (`epr`): Unit `dimensionless`, canonical bounds $[-0.5, +0.5]$ ($epr_{sim} > 0$).
    - Fuel flow ratio (`phi`): Unit `pps/psia`, canonical bounds $[-100.0, +100.0]$ ($\phi_{sim} > 0$).
    - Bypass ratio (`BPR`): Unit `dimensionless`, canonical bounds $[-2.0, +2.0]$ ($BPR_{sim} > 0$).
    - Bleed enthalpy (`htBleed`): Unit `dimensionless`, canonical bounds $[-50.0, +50.0]$.
    - Coolant bleed flows (`W31`, `W32`): Unit `lbm/s`, canonical bounds $[-10.0, +10.0]\text{ lbm/s}$ ($W_{sim} > 0$).
- **Canonical Bounds Enforcement:** Any custom `validation_range` that deviates from `SENSOR_BIAS_LIMITS` is rejected with `ValueError`.
- **State Registration:** Recorded directly in **G40** (`sensor_bias_delta = {channel: bias}`).

### 3.3 Category C: Accelerated Aging Scenario & Physics Limitations
- **Investigation & Formal Deferral:**
  - In the original NASA C-MAPSS simulation engine (Saxena et al., 2008), degradation was injected internally via 13 hidden component health parameters (e.g. `fan_eff_mod`, `fan_flow_mod`, `HPC_eff_mod`, `HPC_flow_mod`, `HPT_eff_mod`, `HPT_flow_mod`). These parameters were **internal Fortran states** never published in the benchmark datasets.
  - The C-MAPSS dataset provides only discrete flight-cycle records of measurable station telemetry.
  - Formulating a synthetic microscopic damage law (e.g., Paris' law crack growth, blade metal loss rates, or empirical erosion equations) without internal calibration parameters would constitute invented physics, violating Roadmap boundaries.
- **Removal of Invalid Temporal Subsampling:**
  - An earlier conceptual attempt to simulate accelerated aging by subsampling recorded cycles ($\text{step\_inc} = \lfloor \alpha_{aging} \rceil$) was reviewed and **formally removed**.
  - Skipping cycles creates temporally discontinuous sequences ($\Delta t > 1\text{ cycle}$), which invalidates the assumptions of the Phase 4 LSTM sequence model ($30\times C$) and summary-statistic trend slopes.
- **Simulator Execution Disabled:**
  - The `ScenarioConfig` for accelerated aging is retained as a validated serializable configuration (proposed capability) with bounds $[1.0, 3.0]$.
  - Attempting to execute an accelerated aging scenario in `DigitalTwinSimulator` via `__init__`, `set_scenario()`, or `step()` raises an explicit `NotImplementedError`:
    > *"Accelerated aging execution is formally deferred. Temporal subsampling of recorded C-MAPSS cycles produces invalid discontinuous sequences for Phase 4 ML models (LSTM/XGBoost), and physical component wear laws are unobservable in C-MAPSS telemetry."*
  - Standalone audit records generated via `ScenarioModifier.apply_scenario` explicitly document `internal_wear_laws_deferred: True` and this limitation note.

---

## 4. Configuration Schema & Validation

Scenarios are represented by the validated, serializable `ScenarioConfig` dataclass:

```python
@dataclass
class ScenarioConfig:
    scenario_id: str
    category: Union[ScenarioCategory, str]
    name: str
    target_parameter: str
    modification_value: float
    unit: str
    validation_range: Tuple[float, float]
    description: str
    is_simulation_only: bool = True
    baseline_value: Optional[float] = None
    simulated_value: Optional[float] = None
```

### 4.1 Strict Safety Safeguards
1. **Simulation-Only Enforcement:** `is_simulation_only` must be `True`. Setting `is_simulation_only=False` raises `ValueError`.
2. **Canonical Range Match:** The validator enforces that `config.validation_range` matches the canonical range for the target parameter. Custom/widened ranges are strictly rejected.
3. **No Silent Clipping:** Values outside the canonical range raise `ValueError`. Values are **never** clipped silently.
4. **Strict Numerical Finiteness:** Inputs with `NaN`, `+inf`, or `-inf` (including optional `baseline_value` and `simulated_value`) are rejected with `ValueError`.
5. **Positive Physical Boundaries:** Simulated gas-path temperatures, pressures, and speeds must remain strictly positive ($> 0$).
6. **Reproducibility:** Supported via `to_dict() -> Dict[str, Any]` and `ScenarioConfig.from_dict(data)`.

---

## 5. Scenario Audit Persistence

Step 5.7 implements comprehensive audit retention across the Digital Twin pipeline:

1. **`ScenarioModifier.apply_scenario()`:** Generates an audit dictionary containing:
   - `scenario_id`: Unique scenario string ID.
   - `category`: Scenario category.
   - `target_parameter`: Target parameter name or alias.
   - `feature_column`: Canonical C-MAPSS 24-channel column name.
   - `channel_index`: Index in the 24-feature telemetry array.
   - `baseline_value` / `original_baseline_value`: Unmodified recorded telemetry value.
   - `applied_modification` / `applied_delta`: Applied numerical delta.
   - `simulated_value` / `resulting_simulated_value`: Perturbed simulation input value.
   - `unit`: Validated engineering unit.
   - `is_simulation_only`: Flag set to `True`.
2. **`EngineState.scenario_audit`:** Every state generated under an active scenario retains this complete audit dictionary. Baseline cycles have `scenario_audit = None`.
3. **`DigitalTwinSimulator` State:** The simulator maintains `current_scenario_audit` and `scenario_audit_history`.
4. **Serialization:** `EngineState.to_dict()` serializes `scenario_audit` alongside $G1–G40$.
5. **Backward Compatibility:** Preserves all existing $G1–G40$ field definitions and behavior.

---

## 6. Artifact Immutability Verification

To guarantee that no scenario execution modifies underlying project assets, tests capture and compare cryptographic SHA-256 hashes **before and after the test session**:

1. **Raw C-MAPSS Dataset Files (`dataset/raw/C-MAPSS/*.txt`):**
   - 12 verified text files (all `train_FD00*.txt`, `test_FD00*.txt`, `RUL_FD00*.txt`).
   - Verified 100% byte-for-byte identical before and after all scenario tests.
2. **Phase 3 Model Scalers (`models/scalers/*.pkl`):**
   - Exactly 4 verified scaler files (`scaler_FD001_tier3.pkl` through `scaler_FD004_tier3.pkl`).
   - Verified 100% byte-for-byte identical before and after all scenario tests.
3. **Phase 4 Champion Checkpoints (`models/checkpoints/`):**
   - Exactly 8 verified champion checkpoint files (4 XGBoost `.json` and 4 LSTM `.pt`).
   - Verified 100% byte-for-byte identical before and after all scenario tests.

> [!NOTE]
> **Immutability Scope:** This check confirms **before-and-after test-session immutability**, verifying that running what-if scenarios never alters training data, fitted scalers, or trained model weights.

---

## 7. Verification & Test Suite Summary

The verification suite in `tests/test_digital_twin_step5_7.py` contains 26 comprehensive unit and integration tests:

| Test Identifier | Category | Verified Condition | Status |
| :--- | :--- | :--- | :---: |
| `test_01_valid_thermal_stress_config` | Thermal Stress | Valid configuration across $T_{30}, T_{50}, T_{24}, T_2$ | **PASS** |
| `test_02_invalid_thermal_offset_out_of_bounds` | Thermal Stress | Rejection of offsets outside $[-100, +200]^\circ\text{R}$ | **PASS** |
| `test_03_invalid_thermal_stress_non_temperature_channel` | Thermal Stress | Rejection of non-thermal channels under thermal stress | **PASS** |
| `test_04_thermal_stress_rejects_custom_validation_range_bypass` | Thermal Stress | Rejection of custom widened validation ranges | **PASS** |
| `test_05_valid_sensor_bias_configs` | Sensor Bias | Valid bias on $P_{30}, N_c, \text{altitude}, \text{TRA}$ | **PASS** |
| `test_06_invalid_sensor_selection_bias` | Sensor Bias | Rejection of unrecognized sensor channels | **PASS** |
| `test_07_invalid_sensor_bias_range` | Sensor Bias | Rejection of out-of-canonical-range biases | **PASS** |
| `test_08_invalid_sensor_bias_unit` | Sensor Bias | Rejection of incompatible engineering units | **PASS** |
| `test_09_sensor_bias_rejects_custom_validation_range_bypass` | Sensor Bias | Rejection of custom widened validation ranges | **PASS** |
| `test_10_nan_and_infinity_rejection` | Numerical Safety | Strict rejection of `NaN`, `+inf`, and `-inf` parameters | **PASS** |
| `test_11_optional_values_finiteness_validation` | Numerical Safety | Strict rejection of non-finite optional values | **PASS** |
| `test_12_is_simulation_only_enforcement` | Safety Policy | Rejection of `is_simulation_only=False` and non-booleans | **PASS** |
| `test_12b_strict_boolean_from_dict_validation` | Deserialization Safety | Rejection of strings ("false", "true", "0", "yes") and integers (0, 1) in `from_dict` | **PASS** |
| `test_13_unsupported_scenario_category_rejection` | Schema Safety | Rejection of unsupported scenario categories | **PASS** |
| `test_14_accelerated_aging_rejects_custom_validation_range_bypass` | Accelerated Aging | Rejection of custom widened aging ranges | **PASS** |
| `test_15_accelerated_aging_deferred_execution_in_simulator` | Sequence Safety | `NotImplementedError` raised; no discontinuous sequences | **PASS** |
| `test_16_accelerated_aging_audit_documents_deferral` | Physics Limits | Standalone modifier audit records formal deferral | **PASS** |
| `test_17_scenario_reproducibility_from_dict` | Reproducibility | Complete round-trip serialization via `to_dict` / `from_dict` | **PASS** |
| `test_18_measured_sensor_values_remain_unchanged` | Data Integrity | Measured fields ($T_{30}$) remain uncorrupted under scenario | **PASS** |
| `test_19_scenario_audit_persistence_and_serialization` | Audit Retention | Full audit persistence in `EngineState` and simulator | **PASS** |
| `test_20_baseline_simulation_remains_identical_without_scenario` | Baseline Integrity | 35-cycle comparison of G1–G29, G30, G31, and G35–G37 between unperturbed baseline and scenario=None | **PASS** |
| `test_21_clear_scenario_restores_baseline` | State Isolation | Verifies clearing of scenario-specific fields (G39, G40) and audit metadata after `clear_scenario()` | **PASS** |
| `test_22_ml_inference_under_what_if_scenario` | ML Integration | XGBoost & LSTM prognostics execute cleanly under scenario inputs | **PASS** |
| `test_23_raw_dataset_files_before_and_after_immutability` | Immutability | SHA-256 hashes of all 12+ raw files identical | **PASS** |
| `test_24_model_scalers_before_and_after_immutability` | Immutability | SHA-256 hashes of all 4 Phase 3 scalers identical | **PASS** |
| `test_25_model_checkpoints_before_and_after_immutability` | Immutability | SHA-256 hashes of all 8 champion checkpoints identical | **PASS** |

### 7.1 Full Project Test Discovery
```bash
.\.venv\Scripts\python.exe -m unittest discover tests
```

**Results:**
- **Ran 84 tests.**
- **Failures: 0**
- **Errors: 0**
- **Status: OK**

Breakdown:
- Step 5.4 (Discrete State Advancement): 11 tests passed.
- Step 5.5 (Physics-Based Component Feasibility): 21 tests passed.
- Step 5.6 (ML Model Prognostic Integration): 26 tests passed.
- Step 5.7 (What-If Scenario Interface): 26 tests passed.

---

## 8. Scope Compliance & Boundary Confirmations

- **Step 5.8 (Scenario Comparisons):** Not implemented. Deferred to Step 5.8.
- **Step 5.9 (Recorded Data Comparisons):** Not implemented. Deferred to Step 5.9.
- **Step 5.10 (Stability & Repeatability):** Not implemented. Deferred to Step 5.10.
- **Step 5.11 (Final Assumptions Documentation):** Not implemented. Deferred to Step 5.11.
- **Git State:** Clean working tree relative to tracked files. Zero commits or pushes performed.
