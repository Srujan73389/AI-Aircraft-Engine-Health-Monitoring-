# Phase 7 — Step 7.4: Tri-State Safety Decision Classification

## 1. Executive Summary

In high-consequence aerospace flight monitoring systems, automated verification must return clear, deterministic, and unambiguous operational verdicts. Binary accept/reject paradigms fail to differentiate between hazardous structural breaches and incipient operational drift that warrants engineering review without grounding an aircraft.

Step 7.4 establishes a formal **Tri-State Safety Decision Engine** ([`safety/decision.py`](file:///c:/Users/LENOVO/Documents/Aircraft_Engine_AI/safety/decision.py)) that classifies proposed simulation actions, what-if stress perturbations, and model enrichments into three distinct statuses:
1. **`ACCEPTED`**: Zero constraint breaches or physical anomalies.
2. **`REQUIRES_REVIEW`**: Advisory caution threshold exceedance (e.g. HPC discharge temperature creep between $1605^\circ\text{R}$ and $1625^\circ\text{R}$).
3. **`REJECTED`**: Safety firewall violation, structural hard limit breach, aerodynamic flight ceiling exceedance, or thermodynamic ordering inversion.

---

## 2. Tri-State Decision Model & Severity Mapping

```
                                [SafetyCheckResult]
                                         |
                       +-----------------+-----------------+
                       |                                   |
           (Firewall Breach OR                             |
        Critical/Hard Limit Violations)                    |
                       |                                   |
                       v                                   v
                 [REJECTED]                     (Zero Severe Breaches)
                                                           |
                                            +--------------+--------------+
                                            |                             |
                                    (Caution Violations)          (Zero Violations)
                                            |                             |
                                            v                             v
                                    [REQUIRES_REVIEW]                [ACCEPTED]
```

### 2.1 Decision Status Definitions

| Status | Operational Criteria | Engine Implication | Enforcement Action |
| :--- | :--- | :--- | :--- |
| **`ACCEPTED`** | Zero violations ($\text{total\_violations} = 0$). All 20 operating constraints and invariants satisfied. | Nominal, safe flight operation. | Action committed to digital twin state. |
| **`REQUIRES_REVIEW`** | $\ge 1$ `CAUTION` violations; $0$ `CRITICAL` or `HARD_LIMIT` violations. | Incipient thermal/mechanical creep beyond nominal fleet envelope. | Advisory flagged; action logged for engineering review before irreversible execution. |
| **`REJECTED`** | $\ge 1$ `HARD_LIMIT` or `CRITICAL` violations, OR direct tampering with `MEASURED` telemetry ($G6 \dots G26$). | Threat of compressor surge, casing burst, disk plastic failure, or corrupted data integrity. | Action blocked immediately; candidate state rejected; incident logged. |

---

## 3. Decision Matrix Details

| Evaluated Condition | Applicable Variable / Invariant | Threshold Boundary | Assigned Status |
| :--- | :--- | :--- | :--- |
| **Telemetry Firewall Mutation** | Direct override of $G6 \dots G26$ | Direct assignment attempt | **`REJECTED`** |
| **HPC Discharge Over-temp (Hard)** | $T_{30}$ ($G8$) | $> 1625.0^\circ\text{R}$ | **`REJECTED`** |
| **HPC Discharge Creep (Caution)** | $T_{30}$ ($G8$) | $1605.0^\circ\text{R} < T_{30} \le 1625.0^\circ\text{R}$ | **`REQUIRES_REVIEW`** |
| **Combustor Casing Overpressure** | $P_{30}$ ($G12$) | $> 700.0\text{ psia}$ | **`REJECTED`** |
| **Flight Ceiling Exceedance** | `altitude` ($G3$) | $> 45,000\text{ ft}$ ($45.0\text{ kft}$) | **`REJECTED`** |
| **Transonic Mach Exceedance** | `mach_number` ($G4$) | $> 0.90$ | **`REJECTED`** |
| **Negative Flight Hours Remaining** | `predicted_rul` ($G35, G36$) | $< 0.0\text{ cycles}$ | **`REJECTED`** |
| **Health Index Bound Exceedance** | `virtual_health_index` ($G38$) | $\notin [0.0, 1.0]$ | **`REJECTED`** |
| **Thermodynamic Ordering Inversion** | Temperature / Pressure / Spool | $T_{30} \le T_{24}$ or $P_{30} \le P_{15}$ or $N_c \le N_f$ | **`REJECTED`** |
| **Nominal Stress / Enrichment** | All state variables | Within baseline envelope | **`ACCEPTED`** |

---

## 4. Architecture & Interface Integration

The decision logic is exposed via two high-level interfaces:
1. **Method on `SafetyVerifier`**:
   ```python
   decision: ActionSafetyDecision = verifier.evaluate_action_decision(current_state, action)
   ```
2. **Functional Entrypoint**:
   ```python
   decision: ActionSafetyDecision = evaluate_simulation_decision(current_state, action)
   ```

Each [`ActionSafetyDecision`](file:///c:/Users/LENOVO/Documents/Aircraft_Engine_AI/safety/decision.py) contains:
- `action_id` (str)
- `status` (`ACCEPTED`, `REJECTED`, or `REQUIRES_REVIEW`)
- `summary_message` (human-readable justification citing primary breached rule)
- `caution_count`, `critical_count`, `hard_limit_count`
- `violations` (complete list of [`ConstraintViolation`](file:///c:/Users/LENOVO/Documents/Aircraft_Engine_AI/safety/constraints.py) objects)
- `projected_state` (resulting candidate state dictionary, or `None` if blocked)
- `decision_timestamp` (ISO-8601 UTC string)

---

## 5. Verification Test Suite Coverage

The test suite [`tests/test_safety_decision_step7_4.py`](file:///c:/Users/LENOVO/Documents/Aircraft_Engine_AI/tests/test_safety_decision_step7_4.py) confirms:
- `test_01_decision_status_enumeration`: Validates all 3 statuses (`ACCEPTED`, `REJECTED`, `REQUIRES_REVIEW`).
- `test_02_classify_nominal_action_accepted`: Validates nominal perturbation receives `ACCEPTED`.
- `test_03_classify_caution_breach_requires_review`: Validates thermal creep in $[1605, 1625]^\circ\text{R}$ receives `REQUIRES_REVIEW`.
- `test_04_classify_firewall_breach_rejected`: Direct physical sensor mutation triggers `REJECTED`.
- `test_05_classify_hard_limit_rejected`: Overtemp $T_{30} > 1625^\circ\text{R}$ triggers `REJECTED`.
- `test_06_classify_ordering_inversion_rejected`: Thermodynamic inversion triggers `REJECTED`.
- `test_07_classify_flight_envelope_exceedance_rejected`: Altitude $> 45\text{ kft}$ or Mach $> 0.90$ triggers `REJECTED`.
- `test_08_classify_negative_rul_rejected`: Negative RUL prediction triggers `REJECTED`.
- `test_09_verifier_evaluate_action_decision_integration`: End-to-end verifier integration verified.
- `test_10_dataset_and_model_immutability`: Bit-for-bit SHA-256 cryptographic verification of 12 raw files, 4 scalers, and 8 champion checkpoints.
