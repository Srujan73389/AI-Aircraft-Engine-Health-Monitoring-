# Phase 7 — Step 7.5: Safety Decision Audit Logging & Reason Recording

## 1. Executive Summary

In airworthiness verification and autonomous digital twin operations, every safety decision—whether accepting an operational reconfiguration, advisory-flagging incipient degradation, or rejecting an unphysical simulation action—must be permanently recorded with an explicit, standardized root cause.

Step 7.5 establishes the **Safety Decision Audit Logging Subsystem** ([`safety/audit.py`](file:///c:/Users/LENOVO/Documents/Aircraft_Engine_AI/safety/audit.py)), providing:
1. **Domain-Specific Reason Codes**: A formal taxonomy of 9 explicit aerospace reason codes ([`SafetyReasonCode`](file:///c:/Users/LENOVO/Documents/Aircraft_Engine_AI/safety/audit.py)).
2. **Structured Audit Record**: A strongly typed dataclass ([`SafetyAuditRecord`](file:///c:/Users/LENOVO/Documents/Aircraft_Engine_AI/safety/audit.py)) capturing timestamp, action ID, decision status, target variable, summary reason, detailed violated constraints, initiator, flight cycle, and full metadata.
3. **Thread-Safe Append-Only Persistence**: The [`SafetyAuditLogger`](file:///c:/Users/LENOVO/Documents/Aircraft_Engine_AI/safety/audit.py) writes records directly to disk in JSON Lines (`.jsonl`) format while providing thread locking, query filtering, and graceful exception containment.
4. **Seamless Verifier Integration**: The [`SafetyVerifier`](file:///c:/Users/LENOVO/Documents/Aircraft_Engine_AI/safety/verification.py) automatically records decisions through an attached audit logger during `evaluate_action_decision()`.

---

## 2. Standardized Safety Reason Codes

| Reason Code | Applicable Decision Status | Operational Domain Meaning | Example Triggering Condition |
| :--- | :--- | :--- | :--- |
| **`NOMINAL_EXECUTION`** | `ACCEPTED` | All 20 operating constraints, aerothermal ordering invariants, and provenance boundaries fully satisfied. | Mild thermal offset ($+4^\circ\text{R}$) or safe prognostics state enrichment. |
| **`TELEMETRY_FIREWALL_BREACH`** | `REJECTED` | Attempted direct mutation of physical sensors ($G6 \dots G26$) blocked by Digital Twin safety firewall. | Direct overwrite of measured `T30`, `P30`, or `Nf`. |
| **`THERMAL_CREEP_ADVISORY`** | `REQUIRES_REVIEW` | Operational caution limit breached without structural damage hazard; operational review recommended. | High-Pressure Compressor discharge temperature $1605^\circ\text{R} < T_{30} \le 1625^\circ\text{R}$. |
| **`METALLURGICAL_HARD_LIMIT_BREACH`** | `REJECTED` | Gas-path temperature exceeds structural disk distortion and blade creep rupture boundaries. | HPC exit total temperature $T_{30} > 1625^\circ\text{R}$. |
| **`PNEUMATIC_CASING_OVERPRESSURE`** | `REJECTED` | Compressor discharge pressure breaches casing pneumatic burst containment limits. | HPC exit total pressure $P_{30} > 700\text{ psia}$. |
| **`STAGE_ORDERING_INVERSION`** | `REJECTED` | Proposed state violates fundamental thermodynamic stage compression laws. | Compression temperature inversion ($T_{30} \le T_{24}$) or pressure drop ($P_{30} \le P_{15}$). |
| **`AERODYNAMIC_ENVELOPE_EXCEEDANCE`** | `REJECTED` | Proposed flight condition exceeds certified subsonic aircraft service envelope. | Pressure altitude $> 45,000\text{ ft}$ (FL450) or flight Mach $> 0.90$. |
| **`PROGNOSTIC_CAUSALITY_VIOLATION`** | `REJECTED` | Machine learning or prognostics output violates physical causality or unit-interval bounds. | Negative Remaining Useful Life ($\text{RUL} < 0$) or Health Index $H(t) \notin [0, 1]$. |
| **`GENERAL_CONSTRAINT_VIOLATION`** | `REJECTED` | Fallback category for other specific operational constraint breaches. | Generic lower/upper bound limit breaches. |

---

## 3. Structured Audit Record Schema

Each logged record conforms to the following schema:

```json
{
  "record_id": "SAR_a8e4b7c109d2",
  "action_id": "act_caution_creep",
  "decision_status": "REQUIRES_REVIEW",
  "target_variable": "thermal_stress_offset",
  "reason_code": "THERMAL_CREEP_ADVISORY",
  "summary_reason": "Action 'act_caution_creep' REQUIRES_REVIEW: 1 operational caution limit(s) breached without exceeding structural hard limits. Operational review advised. Primary: [CAUTION] CONST_TEMP_T30_CAUTION: Upper limit breach: T30 = 1611.70 °R > max allowed 1605.00 °R.",
  "detailed_reasons": [
    "Upper limit breach: T30 = 1611.70 °R > max allowed 1605.00 °R."
  ],
  "violated_constraints": [
    "CONST_TEMP_T30_CAUTION"
  ],
  "initiator_id": "simulation_harness",
  "flight_cycle": 1,
  "timestamp": "2026-11-06T14:18:22.105432+00:00",
  "metadata": {
    "action_type": "THERMAL_PERTURBATION",
    "proposed_value": "22.0"
  }
}
```

---

## 4. Append-Only Persistence Architecture

```
[Simulation Action] ---> [SafetyVerifier] ---> [ActionSafetyDecision]
                                                    |
                                                    v
                                          [SafetyAuditLogger]
                                          (infer_reason_code)
                                                    |
                                    +---------------+---------------+
                                    |                               |
                                    v                               v
                        [Thread-Safe In-Memory]            [Append-Only JSONL]
                        (get_recent_records,              reports/safety_logs/
                         filter_by_status)                safety_audit_trail.jsonl
```

### Safety & Reliability Guarantees:
1. **Thread Safety**: Uses reentrant/threading locks (`threading.Lock()`) around in-memory arrays and disk append operations.
2. **Crash Resilience**: All disk write operations are wrapped in graceful exception containment blocks; an I/O or filesystem failure will never disrupt or crash the live engine simulation harness.
3. **Repository Cleanliness**: Ephemeral `reports/safety_logs/*.jsonl` audit trails are tracked and ignored by `.gitignore` to preserve repository lightweightness.

---

## 5. Verification Test Suite Coverage

The test suite [`tests/test_safety_audit_step7_5.py`](file:///c:/Users/LENOVO/Documents/Aircraft_Engine_AI/tests/test_safety_audit_step7_5.py) validates all 10 unit test cases:

```
test_01_reason_code_enumeration: Confirms all 9 domain reason codes exist. -> PASS
test_02_record_creation_nominal: Verifies record structure for an ACCEPTED decision. -> PASS
test_03_record_creation_firewall_breach: Verifies TELEMETRY_FIREWALL_BREACH for direct sensor mutation. -> PASS
test_04_record_creation_thermal_creep: Verifies THERMAL_CREEP_ADVISORY for REQUIRES_REVIEW status. -> PASS
test_05_record_creation_hard_limit: Verifies METALLURGICAL_HARD_LIMIT_BREACH for severe overtemp. -> PASS
test_06_record_creation_inversion: Verifies STAGE_ORDERING_INVERSION for thermodynamic inversions. -> PASS
test_07_append_only_jsonl_logging: Verifies writing records to disk and parsing valid JSON lines back. -> PASS
test_08_logger_query_and_filtering: Tests filtering audit trail records by status and retrieving recent entries. -> PASS
test_09_verifier_automatic_audit_integration: Verifies that verifier with logger records decisions automatically. -> PASS
test_10_dataset_and_model_immutability: Cryptographic SHA-256 verification of 12 raw files, 4 scalers, 8 checkpoints. -> PASS
```
