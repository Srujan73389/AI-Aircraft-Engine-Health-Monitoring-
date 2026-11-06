# Phase 7 — Step 7.3: Safety Verification Engine & Simulation Action Validation

## 1. Executive Summary

In aerospace digital twin runtime environments, simulation actions, synthetic stress injections, and agent enrichments must never be committed without formal prior verification. 

Step 7.3 implements the **Safety Verification Engine** ([`safety/verification.py`](file:///c:/Users/LENOVO/Documents/Aircraft_Engine_AI/safety/verification.py)), which serves as a pre-execution safety gatekeeper. Before any simulation action or state alteration is applied to the Virtual Engine State Vector ($G1 \dots G40$), the verifier:
1. Enforces the **Telemetry Immutability Firewall** by immediately rejecting any action targeting direct mutation of physical sensors ($G6 \dots G26$).
2. Projects the candidate virtual engine state vector ($S_{\text{projected}}$) resulting from the proposed action.
3. Evaluates $S_{\text{projected}}$ against all 20 registered aerothermal operating constraints and physical invariants from Step 7.1.
4. Returns an immutable [`SafetyCheckResult`](file:///c:/Users/LENOVO/Documents/Aircraft_Engine_AI/safety/verification.py) authorizing or rejecting the proposed action.

---

## 2. Action Classification Taxonomy

Proposed simulation operations are classified into 5 discrete action categories:

| Action Type | Operational Purpose | Allowable Target Variables | Projection Behavior |
| :--- | :--- | :--- | :--- |
| **`THERMAL_PERTURBATION`** | Inject hypothetical thermal degradation or combustor hot-spots. | `thermal_stress_offset` ($G39$) | Projects $T_{30} = T_{30}^{\text{meas}} + \Delta T_{\text{offset}}$, updates $\Delta T_{\text{HPC}} = T_{30} - T_{24}$. |
| **`SENSOR_BIAS_INJECTION`** | Simulate sensor calibration drift or instrumentation bias. | `sensor_bias_delta` ($G40$) | Injects bias map into simulated sensor readings; recomputes macroscopic indicators (OPR, spool ratio). |
| **`OPERATIONAL_RECONFIGURATION`** | Adjust flight mission context and operating conditions. | `altitude` ($G3$), `mach_number` ($G4$), `throttle_resolver_angle` ($G5$) | Updates operating context parameters within aerodynamic flight boundaries. |
| **`STATE_ENRICHMENT`** | Ingest ML prognostics, RUL predictions, and health indexes. | `predicted_rul_xgboost` ($G35$), `predicted_rul_lstm` ($G36$), `virtual_health_index` ($G38$), `anomaly_alert_flag` ($G34$) | Enriches state vector with model outputs while verifying non-negativity and unit-interval bounds. |
| **`RECOVERY_ACTION`** | Roll back perturbations or simulate post-maintenance resets. | `thermal_stress_offset`, `sensor_bias_delta`, baseline states | Resets simulation offsets to zero; returns state towards nominal baseline. |

---

## 3. Strict Safety Firewall Enforcement

The verifier strictly enforces the physical telemetry immutability boundary:

```
[Proposed Action] ---> [Is Target in G6..G26?]
                              |
                     +--------+--------+
                     |                 |
                   (YES)              (NO)
                     |                 |
         [FIREWALL REJECTION]    [Project State]
         is_safe = False               |
         provenance_allowed = False    v
                             [Evaluate Constraints]
                                       |
                              +--------+--------+
                              |                 |
                         (Violations)      (Zero Violations)
                              |                 |
                           is_safe = False   is_safe = True
```

Any action directly targeting $G6 \dots G26$ (e.g., `T30`, `P30`, `Nf`, `Ps30`) with direct value overrides is rejected outright with:
- `is_safe = False`
- `provenance_allowed = False`
- `violations = [ConstraintViolation("FIREWALL_MEASURED_IMMUTABILITY", severity=HARD_LIMIT)]`

Perturbations must legitimately target simulated offset slots ($G39$ `thermal_stress_offset` or $G40$ `sensor_bias_delta`).

---

## 4. State Projection & Invariant Evaluation

When an authorized action is evaluated, [`SafetyVerifier.project_state()`](file:///c:/Users/LENOVO/Documents/Aircraft_Engine_AI/safety/verification.py) computes candidate telemetry:

1. **Thermal Stress Projection**:
   $$\begin{aligned}
   T_{30}^{\text{projected}} &= T_{30}^{\text{base}} + \Delta T_{\text{thermal\_stress\_offset}} \\
   \Delta T_{\text{hpc}}^{\text{projected}} &= T_{30}^{\text{projected}} - T_{24}^{\text{base}}
   \end{aligned}$$

2. **Sensor Bias Projection**:
   $$S_i^{\text{projected}} = S_i^{\text{base}} + \delta_{\text{bias}, i}$$
   $$\text{OPR}^{\text{projected}} = \frac{P_{30}^{\text{projected}}}{P_2^{\text{projected}}}$$

3. **Invariants & Constraint Verification**:
   $S_{\text{projected}}$ is passed to `evaluate_constraints()`:
   - Validates maximum warning and caution boundaries ($T_{30} \le 1625^\circ\text{R}$, $P_{30} \le 700\text{ psia}$, $\text{Altitude} \le 45\text{ kft}$, $\text{Mach} \le 0.90$).
   - Validates aerothermal stage ordering ($T_{30} > T_{24} > T_2$, $P_{30} > P_{15} > P_2$, $N_c > N_f$).
   - Validates mathematical validity ($\text{RUL} \ge 0$, $H(t) \in [0, 1]$).

---

## 5. Verification Test Suite Coverage

The test suite [`tests/test_safety_verification_step7_3.py`](file:///c:/Users/LENOVO/Documents/Aircraft_Engine_AI/tests/test_safety_verification_step7_3.py) validates all 10 unit test cases:

```
test_01_verifier_initialization: Confirms verifier initializes with default or custom constraints. -> PASS
test_02_accept_nominal_simulation_action: Verifies valid what-if perturbation within safe thermal limits. -> PASS
test_03_reject_immutable_telemetry_action: Asserts rejection of any action targeting MEASURED sensors directly. -> PASS
test_04_reject_overtemp_thermal_action: Asserts rejection of thermal actions breaching T30 warning limits. -> PASS
test_05_reject_overpressure_action: Asserts rejection of actions driving P30 beyond casing limits. -> PASS
test_06_reject_flight_envelope_exceedance: Asserts rejection of actions setting altitude > 45 kft or Mach > 0.90. -> PASS
test_07_reject_aerothermal_ordering_inversion: Asserts rejection of actions inverting temperature (T24 >= T30). -> PASS
test_08_reject_negative_rul_action: Asserts rejection of actions attempting to set negative RUL. -> PASS
test_09_prognostics_state_enrichment_action: Verifies safe state enrichment with valid RUL and health index. -> PASS
test_10_dataset_and_model_immutability: Cryptographic SHA-256 verification of 12 raw files, 4 scalers, 8 checkpoints. -> PASS
```
