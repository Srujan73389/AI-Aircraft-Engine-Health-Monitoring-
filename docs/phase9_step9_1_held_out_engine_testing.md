# Phase 9 — Step 9.1: Held-Out Test Engine Evaluation & Trajectory Verification

**Document Reference:** `docs/phase9_step9_1_held_out_engine_testing.md`  
**Master Roadmap Reference:** Section 10, Step 9.1 (*"Test the application using several engine IDs from the held-out test data"*).  
**Phase Status:** Phase 9 ("System testing and demonstration") — Step 9.1 Complete.

---

## 1. Executive Summary & Verification Scope

In Step 9.1, the application is rigorously tested against the official NASA C-MAPSS held-out test split (`test_FD001.txt` through `test_FD004.txt`). Unlike training trajectories that proceed continuously to mechanical failure (0 remaining cycles), held-out test trajectories are truncated at unknown operational flight cycles prior to catastrophic failure.

The primary engineering objectives of this evaluation were:
1. **Partition Support & Discovery:** Validate that backend discovery endpoints resolve the exact held-out test engine inventories across all four subsets.
2. **Representative Test Cohort Evaluation:** Thoroughly test multi-cycle telemetry loading and full 40-variable state vector ($S(t)$) generation across an 8-engine representative cohort spanning all four operational subsets:
   - **FD001 (Single SLS regime, HPC fault):** Unit 3, Unit 24
   - **FD002 (6 flight regimes, HPC fault):** Unit 15, Unit 42
   - **FD003 (Single SLS regime, compound HPC & Fan faults):** Unit 8, Unit 20
   - **FD004 (6 flight regimes, compound HPC & Fan faults):** Unit 10, Unit 35
3. **Lifespan Truncation & Boundary Defense:** Confirm that requesting cycles beyond a test engine's truncated operational lifespan raises strict HTTP 400 boundary defenses without off-by-one errors or silent extrapolations.
4. **Endpoint `split` Parameter Hardening:** Add and certify optional `split` query parameters (`?split=test`) to fleet discovery, cycle state retrieval, sensor trend tracking, and prognostics prediction endpoints.
5. **Zero Data Leakage & Artifact Immutability:** Verify that raw C-MAPSS dataset files, Phase 3 scalers, and Phase 4 champion checkpoints remain completely immutable via cryptographic SHA-256 verification.

---

## 2. Held-Out Test Inventory & Representative Cohort

### 2.1 Complete Test Partition Inventory
| Subset | Flight Regimes | Failure Modes | Test Engines Count | Engine ID Range |
|---|---|---|---|---|
| **FD001** | 1 (Sea Level) | HPC Degradation | **100** | Units 1 – 100 |
| **FD002** | 6 Altitude/Mach Envelopes | HPC Degradation | **259** | Units 1 – 259 |
| **FD003** | 1 (Sea Level) | HPC + Fan Degradation | **100** | Units 1 – 100 |
| **FD004** | 6 Altitude/Mach Envelopes | HPC + Fan Degradation | **248** | Units 1 – 248 |

### 2.2 Representative Test Cohort Characteristics
| Subset | Unit ID | Truncated Cycle Span | Trajectory Type | Aerothermal Characteristics |
|---|---|---|---|---|
| **FD001** | Unit 3 | Cycles 1 – 126 | Mid-life Truncation | Steady single-regime gas-path degradation; $T_{30}$ rising monotonically. |
| **FD001** | Unit 24 | Cycles 1 – 186 | Near-failure Truncation | High-cycle engine with significant baseline drift across all 21 channels. |
| **FD002** | Unit 15 | Cycles 1 – 165 | Multi-Regime Complex | Alternates across 6 operating condition clusters; altitude from 0 to 42 kft. |
| **FD002** | Unit 42 | Cycles 1 – 137 | Multi-Regime Moderate | Highly dynamic mach and altitude variations with pronounced core speed shifts. |
| **FD003** | Unit 8 | Cycles 1 – 192 | Compound Fault Moderate | Coupled degradation across high-pressure compressor and low-pressure fan spool. |
| **FD003** | Unit 20 | Cycles 1 – 207 | Compound Fault Extended | Long operational test history showing simultaneous fan flow and HPC thermal drift. |
| **FD004** | Unit 10 | Cycles 1 – 23 | Early-Life Truncation | Minimal elapsed cycles; serves as a critical test for left-edge padding and early bounds. |
| **FD004** | Unit 35 | Cycles 1 – 278 | Highly Stressed Mission | Longest held-out test trajectory tested, combining 6 envelopes with compound fault modes. |

---

## 3. Backend Trajectory Loader & Split Support Hardening

### 3.1 Endpoint Modifications

1. **`GET /api/v1/fleet/{subset}/engines`** ([`backend/api/fleet.py`](file:///c:/Users/LENOVO/Documents/Aircraft_Engine_AI/backend/api/fleet.py))
   - Added `split: str = Query("train", pattern="^(train|test)$")`.
   - When `split="test"` is specified for FD002, returns exactly 259 engines (Unit 260 exists only in train).
   - Defaults to `"train"` for backwards compatibility.

2. **`GET /api/v1/engine/{subset}/{unit}/cycle/{cycle}`** ([`backend/api/fleet.py`](file:///c:/Users/LENOVO/Documents/Aircraft_Engine_AI/backend/api/fleet.py))
   - Added `split: str = Query("train", pattern="^(train|test)$")`.
   - Enforces truncated boundary defense: e.g. for FD001 Unit 3, cycle 126 succeeds (`200 OK`), while cycle 127 returns HTTP 400:
     ```json
     {
       "detail": "Requested cycle 127 exceeds engine 3 (test) maximum cycle 126."
     }
     ```

3. **`GET /api/v1/engine/{subset}/{unit}/sensors`** ([`backend/api/monitoring.py`](file:///c:/Users/LENOVO/Documents/Aircraft_Engine_AI/backend/api/monitoring.py))
   - Added `split: str = Query("train", pattern="^(train|test)$")`.
   - Returns 21-channel telemetry curves matching the exact truncated cycle length (e.g. 23 cycles for FD004 Unit 10, 126 cycles for FD001 Unit 3).

4. **`POST /api/v1/predict/rul`** ([`backend/api/prognostics.py`](file:///c:/Users/LENOVO/Documents/Aircraft_Engine_AI/backend/api/prognostics.py))
   - Added optional `split: str = "train"` to `RULPredictionRequest`.
   - Correctly loads held-out test engine history to compute 30-cycle temporal window tensor and infer champion XGBoost and Bi-LSTM RUL predictions.

---

## 4. Test Suite Verification (`test_held_out_engines_step9_1.py`)

A dedicated 10-test test suite was developed in [`tests/test_held_out_engines_step9_1.py`](file:///c:/Users/LENOVO/Documents/Aircraft_Engine_AI/tests/test_held_out_engines_step9_1.py):

| Test Case | Description | Result |
|---|---|---|
| `test_01_held_out_engine_inventory_discovery` | Validates test engine discovery counts (100, 259, 100, 248) across all 4 subsets | **PASS** |
| `test_02_held_out_fd001_trajectory_loading` | Tests trajectory integrity for FD001 Units 3 (126 cyc) and 24 (186 cyc) | **PASS** |
| `test_03_held_out_fd002_multi_regime_loading` | Tests 6-regime test engines FD002 Units 15 (165 cyc) and 42 (137 cyc) | **PASS** |
| `test_04_held_out_fd003_compound_fault_loading` | Tests compound fault test engines FD003 Units 8 (192 cyc) and 20 (207 cyc) | **PASS** |
| `test_05_held_out_fd004_complex_compound_loading` | Tests FD004 early truncation (Unit 10, 23 cyc) and long mission (Unit 35, 278 cyc) | **PASS** |
| `test_06_held_out_cycle_state_vector_construction` | Verifies full 40-variable state vector $S(t)$ generation without NaN or empty channels | **PASS** |
| `test_07_held_out_cycle_exceedance_boundary_defense` | Validates HTTP 400 rejection when requesting cycles beyond truncated test spans | **PASS** |
| `test_08_held_out_sensor_trends_parity` | Asserts exact length parity between 21 sensor trends and test cycle counts | **PASS** |
| `test_09_held_out_split_parameter_in_fleet_endpoints` | Confirms `split` query parameter isolates test engines and handles invalid input | **PASS** |
| `test_10_dataset_and_model_immutability` | Validates SHA-256 integrity of all 12 raw files, 4 scalers, and 8 checkpoints | **PASS** |

---

## 5. Full Regression Suite Results

Executing the complete unit and integration test suite across all 9 phases:
```powershell
.\.venv\Scripts\python.exe -m unittest discover tests
```
**Output:**
```text
Ran 455 tests in 85.039s
OK
```

All 455 unit and integration tests passed cleanly with zero failures and zero errors.

---

## 6. Step 9.1 Completion Sign-Off

Step 9.1 is **100% complete**:
- Held-out test split inventories verified across all 4 C-MAPSS subsets.
- Representative test cohort (FD001 U3/U24, FD002 U15/U42, FD003 U8/U20, FD004 U10/U35) verified.
- Fleet and monitoring endpoints support `split="test"` query parameter.
- Truncated cycle boundary defenses certified.
- Full regression suite (455/455 tests) passing cleanly.
- Stopping immediately as required; Step 9.2 not started; zero git commits or pushes performed.
