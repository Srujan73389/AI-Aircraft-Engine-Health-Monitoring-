# Phase 7 — Step 7.6: Simulated Recovery Strategies & Remediation Architecture

## 1. Executive Summary

When an autonomous digital twin or predictive health monitoring system detects an anomalous condition, caution exceedance, or rejected operational action, simply halting execution is insufficient. A production-grade aerospace digital twin must synthesize **actionable remediation recommendations** tailored to the specific failure mode.

Step 7.6 implements the **Simulated Recovery Strategy Subsystem** ([`safety/recovery.py`](file:///c:/Users/LENOVO/Documents/Aircraft_Engine_AI/safety/recovery.py)), which translates safety decisions ([`ActionSafetyDecision`](file:///c:/Users/LENOVO/Documents/Aircraft_Engine_AI/safety/decision.py)) into concrete simulated and operational remediation plans ([`SimulatedRecoveryPlan`](file:///c:/Users/LENOVO/Documents/Aircraft_Engine_AI/safety/recovery.py)).

---

## 2. Recovery Strategy Taxonomy & Priority Levels

### 2.1 Strategy Classification

| Strategy Type | Operational Definition | Primary Applications |
| :--- | :--- | :--- |
| **`CONTINUE_MONITORING`** | Steady-state operational continuation without intervention. | `ACCEPTED` actions; nominal flight trajectories with zero threshold breaches. |
| **`MAINTENANCE_REVIEW`** | Flag incipient degradation for scheduled turnaround inspection. | `REQUIRES_REVIEW` status; HPC discharge temperature creep ($1605^\circ\text{R} < T_{30} \le 1625^\circ\text{R}$). |
| **`PHYSICAL_INSPECTION`** | Mandatory hardware inspection (optical borescope, compressor stall/surge audit). | `REJECTED` structural overtemp ($T_{30} > 1625^\circ\text{R}$), casing overpressure ($P_{30} > 700\text{ psia}$), or aerodynamic stage ordering inversion ($T_{30} \le T_{24}$, $P_{30} \le P_{15}$). |
| **`PARAMETER_CLAMP`** | Mathematical bound enforcement maintaining physical causality. | `REJECTED` unphysical prognostics ($\text{RUL} < 0\text{ cycles}$ clamped to $0.0$; $H(t) \notin [0, 1]$ clamped to unit interval). |
| **`SCENARIO_ROLLBACK`** | Transactional reversion of virtual perturbations to pre-action baseline. | `REJECTED` telemetry firewall violations ($G6 \dots G26$ mutation attempts) or aerodynamic service ceiling exceedance ($> 45\text{ kft}$). |

### 2.2 Priority Levels

| Priority Level | Response Requirement | Typical Aircraft Action |
| :--- | :--- | :--- |
| **`LOW`** | Routine observation | Standard in-flight continuous health tracking. |
| **`MEDIUM`** | Non-urgent maintenance scheduling | Line maintenance visual check upon routine turnaround. |
| **`HIGH`** | Urgent simulation hold / prioritized check | Immediate algorithmic clamping or flight profile rollback prior to next test. |
| **`CRITICAL`** | Immediate simulated abort | Ground hold; mandatory optical borescope or casing structural non-destructive testing (NDT). |

---

## 3. Decision-to-Remediation Mapping Engine

The engine [`generate_recovery_recommendation()`](file:///c:/Users/LENOVO/Documents/Aircraft_Engine_AI/safety/recovery.py) evaluates the decision context and yields deterministic mitigation plans:

```
                  [ActionSafetyDecision]
                             |
         +-------------------+-------------------+
         |                   |                   |
    [ACCEPTED]       [REQUIRES_REVIEW]       [REJECTED]
         |                   |                   |
         v                   v                   +--------------+--------------+--------------+
    [CONTINUE_          [MAINTENANCE_            |              |              |              |
    MONITORING]           REVIEW]           (Firewall /     (Hardware     (Prognostics   (Inversion)
     (LOW)               (MEDIUM)            Envelope)     Overtemp)       Causality)         |
                                                 |              |              |              v
                                                 v              v              v         [PHYSICAL_
                                             [SCENARIO_     [PHYSICAL_    [PARAMETER_    INSPECTION]
                                             ROLLBACK]     INSPECTION]      CLAMP]       (CRITICAL)
                                               (HIGH)       (CRITICAL)      (HIGH)
```

### 3.1 Mitigation Parameters Schema Examples

1. **Parameter Clamping (Negative RUL)**:
   ```json
   {
     "clamp_applied": true,
     "original_value": -15.0,
     "clamped_rul": 0.0
   }
   ```

2. **Optical Borescope Inspection (Severe Over-temp)**:
   ```json
   {
     "rollback_performed": true,
     "inspection_type": "OPTICAL_BORESCOPE_HPC_COMBUSTOR",
     "structural_hard_limit_variable": "thermal_stress_offset",
     "safe_threshold": 1625.0,
     "reset_offsets": {
       "thermal_stress_offset": 0.0
     }
   }
   ```

3. **Telemetry Firewall Rollback**:
   ```json
   {
     "rollback_performed": true,
     "rejected_target": "T30",
     "firewall_enforced": true,
     "allowed_alternatives": [
       "thermal_stress_offset (G39)",
       "sensor_bias_delta (G40)"
     ]
   }
   ```

---

## 4. Integration with Digital Twin Verifier

Recovery plan generation is seamlessly integrated into [`SafetyVerifier`](file:///c:/Users/LENOVO/Documents/Aircraft_Engine_AI/safety/verification.py):
- Direct invocation via `verifier.generate_recovery_plan(decision, action, current_state)`.
- Automated attachment during decision classification via `verifier.evaluate_action_decision(state, action, attach_recovery_plan=True)`.
- Global helper `generate_simulation_recovery_plan(decision, action)`.

---

## 5. Verification Test Suite Coverage

The test suite [`tests/test_safety_recovery_step7_6.py`](file:///c:/Users/LENOVO/Documents/Aircraft_Engine_AI/tests/test_safety_recovery_step7_6.py) validates all 10 unit test cases:

```
test_01_strategy_type_enumeration: Confirms 5 strategy types and 4 priority levels exist. -> PASS
test_02_plan_nominal_continue_monitoring: Verifies CONTINUE_MONITORING for an ACCEPTED action. -> PASS
test_03_plan_firewall_rollback: Verifies SCENARIO_ROLLBACK for telemetry firewall breaches. -> PASS
test_04_plan_thermal_creep_maintenance_review: Verifies MAINTENANCE_REVIEW for thermal creep. -> PASS
test_05_plan_overtemp_physical_inspection: Verifies PHYSICAL_INSPECTION + CRITICAL for overtemp. -> PASS
test_06_plan_inversion_physical_inspection: Verifies PHYSICAL_INSPECTION for aerodynamic stage inversions. -> PASS
test_07_plan_prognostic_parameter_clamp: Verifies PARAMETER_CLAMP for negative RUL or invalid health index. -> PASS
test_08_plan_mitigation_parameters_content: Verifies mitigation_parameters provides exact rollback target / clamp values. -> PASS
test_09_verifier_recovery_integration: Verifies seamless invocation through SafetyVerifier.generate_recovery_plan. -> PASS
test_10_dataset_and_model_immutability: Cryptographic SHA-256 verification of 12 raw files, 4 scalers, 8 checkpoints. -> PASS
```
