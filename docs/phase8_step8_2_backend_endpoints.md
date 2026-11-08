# Phase 8 — Step 8.2: RESTful API Endpoints Specification & Verification

## 1. Overview & Roadmap Alignment
- **Milestone:** Phase 8 — Interactive Application & Deployment (Step 8.2: *"Create endpoints for dataset/engine selection, monitoring, RUL prediction, simulation and safety checks"*).
- **Master Project Roadmap Reference:** Section 9, Step 8.2.
- **Architectural Scope:** Exposes high-performance, validated RESTful endpoints via FastAPI serving the entire engine health monitoring, diagnostics, prognostic RUL inference, digital twin simulation, and closed-loop safety verification subsystems.
- **Academic Boundary Constraint:** All endpoints enforce the fundamental rule that this software is an academic simulation and diagnostic research prototype, not an airborne flight-control authority or certified maintenance system. Every response contains standard academic simulation disclaimers.

---

## 2. API Architecture & Routing Layer

### 2.1 Router Topology
All routers are mounted under `/api/v1` (primary production prefix) and `/api` (convenience prefix) on the main FastAPI application (`backend/main.py`):

| Router Prefix | Tag | Description | Handler File |
| :--- | :--- | :--- | :--- |
| `/api/v1/fleet` & `/api/v1/engine` | `Fleet & Engine Selection` | Dataset subsets, engine catalogs, 40-variable state vector inspection | `backend/api/fleet.py` |
| `/api/v1/monitoring` | `Health Monitoring` | Timeseries sensor telemetry, trajectory trending, statistical anomaly checks | `backend/api/monitoring.py` |
| `/api/v1/predict` | `Prognostics & RUL` | Real-time multi-model RUL inference (XGBoost, LSTM), consensus ensembling, bounds | `backend/api/prognostics.py` |
| `/api/v1/simulation` | `Digital Twin Simulation` | Gated What-If scenario injection, perturbation execution, safety rollback | `backend/api/simulation.py` |
| `/api/v1/safety` | `Safety Verification & Audit` | Standalone action verification, tri-state decisions, gated commit, audit trail | `backend/api/safety.py` |

---

## 3. Comprehensive Endpoint Specifications

### 3.1 Fleet & Engine Selection (`/api/v1/fleet`, `/api/v1/engine`)
1. **`GET /api/v1/fleet/subsets`**
   - **Description:** Returns the 4 standard NASA C-MAPSS subsets (FD001–FD004) with operating conditions, fault modes, engine counts, and feature dimensions.
   - **Response Schema:** `List[FleetSubsetInfo]`
2. **`GET /api/v1/fleet/{subset}/engines`**
   - **Description:** Returns engine units for a specified subset (e.g., FD001: units 1–100) with trajectory flight cycle counts.
   - **Response Schema:** `List[EngineSummary]`
3. **`GET /api/v1/engine/{subset}/{unit}/cycle/{cycle}`**
   - **Description:** Advances digital twin simulation to the specified operational cycle and returns the complete 40-variable state vector ($G_1$ to $G_{40}$), supporting both canonical names and taxonomy group prefixes.
   - **Response Schema:** `EngineCycleState`

### 3.2 Health Monitoring & Telemetry (`/api/v1/monitoring`)
1. **`GET /api/v1/engine/{subset}/{unit}/sensors`**
   - **Description:** Retrieves historical timeseries trajectories for all 21 raw sensor channels and operating settings across flight cycles for graphing.
   - **Query Parameters:** `sensors: Optional[List[str]]`, `start_cycle: int = 1`, `end_cycle: Optional[int] = None`.
   - **Response Schema:** `SensorTrendsResponse`
2. **`POST /api/v1/monitoring/anomaly-check`**
   - **Description:** Evaluates current telemetry or a targeted engine cycle against baseline statistical thresholds (e.g., $3\sigma$ Z-score deviations, sensor limits) to return anomaly alerts.
   - **Request Schema:** `AnomalyCheckRequest`
   - **Response Schema:** `AnomalyCheckResponse`

### 3.3 Prognostics & Remaining Useful Life (`/api/v1/predict`)
1. **`POST /api/v1/predict/rul`**
   - **Description:** Executes high-speed prognostics inference across loaded champion models (XGBoost, LSTM), computes consensus ensemble predictions, prognostics spread, confidence intervals, and virtual health indices ($VHI$).
   - **Request Schema:** `RULPredictionRequest`
   - **Response Schema:** `RULPredictionResponse`

### 3.4 What-If Scenario Simulation (`/api/v1/simulation`)
1. **`POST /api/v1/simulation/what-if`**
   - **Description:** Evaluates and applies a proposed simulation scenario action (e.g., thermal perturbation, power derate, sensor drift) through the `ClosedLoopSafetyGovernor`. If safe (`ACCEPTED`), state mutations commit; if unsafe (`REJECTED`), automatic rollback is triggered and a simulated recovery plan is attached.
   - **Request Schema:** `SimulationWhatIfRequest`
   - **Response Schema:** `SimulationWhatIfResponse`

### 3.5 Safety Verification & Audit Trail (`/api/v1/safety`)
1. **`POST /api/v1/safety/verify`**
   - **Description:** Evaluates candidate simulation actions against aerospace physical boundaries, thermodynamic operational envelopes, and telemetry firewall immutability rules without modifying simulator state. Returns tri-state verdicts: `ACCEPTED`, `REJECTED`, or `REQUIRES_REVIEW`.
   - **Request Schema:** `SafetyVerificationRequest`
   - **Response Schema:** `SafetyVerificationResponse`
2. **`POST /api/v1/safety/execute-gated`**
   - **Description:** Direct governor-gated execution endpoint that enforces operator approval requirements on review-flagged actions.
   - **Request Schema:** `SimulationWhatIfRequest`
   - **Response Schema:** `SimulationWhatIfResponse`
3. **`GET /api/v1/safety/audit-trail`**
   - **Description:** Queries the append-only cryptographic safety audit log for recorded decisions, violations, and operator approvals.
   - **Query Parameters:** `limit: int = 50`, `decision_status: Optional[str] = None`, `target_variable: Optional[str] = None`.
   - **Response Schema:** `SafetyAuditTrailResponse`

---

## 4. Verification & Regression Testing

### 4.1 Test Suite Structure (`tests/test_backend_endpoints_step8_2.py`)
A dedicated 10-test suite was implemented with a dual-mode `FastAPITestClientAdapter` that provides:
1. Native `fastapi.testclient.TestClient` execution when unconstrained network/loopback is available.
2. High-performance direct in-process FastAPI router dispatch when running inside sandboxed environments where Windows socketpair loopback creation is prohibited.

```python
TestBackendEndpointsStep8_2:
├── test_01_health_and_model_registry_loaded   [PASS]
├── test_02_fleet_catalog_endpoints           [PASS]
├── test_03_engine_cycle_state_endpoint        [PASS]
├── test_04_sensor_trends_endpoint             [PASS]
├── test_05_monitoring_anomaly_check_endpoint  [PASS]
├── test_06_predict_rul_endpoint               [PASS]
├── test_07_simulation_what_if_endpoint        [PASS]
├── test_08_safety_verify_endpoint             [PASS]
├── test_09_safety_execute_gated_endpoint      [PASS]
└── test_10_safety_audit_trail_endpoint        [PASS]
```

### 4.2 Full System Regression Results
All 345 unit and integration tests across Phases 1 through 8 pass cleanly:
- Phases 1–7 test suite: 325 tests passed.
- Phase 8 Step 8.1 (Model Loader & Service Layer): 10 tests passed.
- Phase 8 Step 8.2 (RESTful API Endpoints): 10 tests passed.
- **Total passing tests: 345 / 345 (100% green).**
