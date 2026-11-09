# Phase 9 — Step 9.9: Backend Errors, Frontend Errors, and Model-Loading Failures Audit

**Roadmap Reference:** Master Project Roadmap PDF, Section 10, Step 9.9 ("Test backend errors, frontend errors and model-loading failures")  
**Design Principle:** Section 1, Page 2 ("Do not invent sensor measurements or report fabricated model scores")  
**Audit Date:** November 9, 2026  
**Auditor:** Lead Fault Injection & Site Reliability Engineer  
**Status:** Certified & Approved (10/10 Tests Passed)  

---

## 1. Executive Summary & Verification Scope

In mission-critical aerospace prognostics and health management (PHM), high reliability demands that defensive validation reject out-of-bounds queries, malformed payloads, and corrupt model artifacts gracefully without crashing the server process, corrupting internal state, or returning fabricated telemetry.

Step 9.9 executes rigorous fault-injection and boundary-stress testing across three system tiers:
1. **Backend Defensive Routing & Input Validation:** Ensures all REST API endpoints intercept invalid subsets, nonexistent engine IDs, out-of-envelope flight cycles, and malformed request bodies with correct HTTP status codes (`400`, `404`, `422`).
2. **Model-Loading Failure Resilience:** Validates that `ModelRegistry` and the inference engine handle missing or corrupted champion model checkpoints (`.json`, `.pt`) and feature scalers (`.pkl`) by raising informative exceptions and returning structured HTTP `503 Service Unavailable` (`MODEL_UNAVAILABLE`) responses without crashing the server process.
3. **Frontend Defensive Contract Alignment:** Verifies that all error payloads adhere strictly to the standard FastAPI `{ "detail": ... }` contract, directly consumed and classified into actionable alert banners (`INVALID_ENGINE_ID`, `MISSING_INPUT`, `MODEL_UNAVAILABLE`) by `ValidationAlertBanner.jsx` and `EngineContext.jsx`.

---

## 2. Defensive Routing & HTTP Status Code Contract

| Error Condition | Trigger Vector | HTTP Status | Detail Message Contract | Frontend Classification |
| :--- | :--- | :---: | :--- | :--- |
| **Invalid Dataset Subset** | `GET /fleet/FD999/engines`, `GET /engine/UNKNOWN/1/cycle/1` | `404 Not Found` | `Subset '{subset}' not found. Supported subsets: ['FD001', 'FD002', 'FD003', 'FD004']` | `INVALID_ENGINE_ID` |
| **Nonexistent Engine ID** | `GET /engine/FD001/999/cycle/1` | `404 Not Found` | `Engine unit 999 not found in subset 'FD001' (train). Valid engine ID range is 1 to 100.` | `INVALID_ENGINE_ID` |
| **Cycle Exceedance** | `GET /engine/FD001/1/cycle/999` | `400 Bad Request` | `Requested cycle 999 exceeds engine 1 (train) maximum cycle 192.` | `INVALID_ENGINE_ID` |
| **Cycle Non-Positive** | `GET /engine/FD001/1/cycle/0`, `GET /engine/FD001/1/cycle/-10` | `400 Bad Request` | `Requested cycle {cycle} must be a positive integer >= 1.` | `INVALID_ENGINE_ID` |
| **Missing Required Payload** | `POST /simulation/what-if` with `{}` | `422 Unprocessable` | Standard Pydantic field validation error list | `MISSING_INPUT` |
| **Schema Field Type Mismatch**| `POST /monitoring/anomaly-check` with `{"unit_number": "not_an_int"}` | `422 Unprocessable` | `{"loc": ["body", "unit_number"], "msg": "Input should be a valid integer"}` | `MISSING_INPUT` |
| **Improper Temporal Window** | `POST /predict/rul` with sequence length $T \ne 30$ | `400 Bad Request` | `Malformed sequence tensor: required window length is 30 cycles, got {T} cycles.` | `MISSING_INPUT` |
| **Missing Model Checkpoint** | Checkpoint unreadable or removed from disk | `503 Service Unavailable` | `MODEL_UNAVAILABLE: Required champion checkpoint or scaler file is missing or inaccessible.` | `MODEL_UNAVAILABLE` |
| **Corrupted Model Checkpoint**| Checkpoint payload corrupted with invalid binary/JSON | `503 Service Unavailable` | `MODEL_UNAVAILABLE: Checkpoint file unreadable or corrupted.` | `MODEL_UNAVAILABLE` |

---

## 3. Model-Loading & Checkpoint Failure Resilience

The backend prognostics and health monitoring architecture relies on 4 champion XGBoost regression models (`.json`), 4 PyTorch Bi-LSTM models (`.pt`), 4 feature scalers (`.pkl`), and 8 anomaly detectors. Step 9.9 tests system behavior under severe artifact degradation using isolated temporary directory sandboxes:

### 3.1 Missing Checkpoint Simulation
- **Mechanism:** `FaultInjectionAuditor.simulate_missing_checkpoint_error(subset, model_type)` configures `ModelRegistry` pointing to an isolated empty directory.
- **Verification:**
  - `load_xgboost_models()` detects missing `xgb_{sub}_champion.json` and cleanly raises `FileNotFoundError`.
  - `load_lstm_models()` detects missing `lstm_{sub}_champion.pt` and cleanly raises `FileNotFoundError`.
  - The API router maps `FileNotFoundError` to HTTP `503 Service Unavailable` with `MODEL_UNAVAILABLE` error signature.
  - Zero unhandled exceptions; the running process remains stable.

### 3.2 Corrupted Checkpoint Simulation
- **Mechanism:** `FaultInjectionAuditor.simulate_corrupted_checkpoint_error(subset, model_type)` generates corrupted checkpoint payloads containing non-model binary garbage and malformed JSON.
- **Verification:**
  - XGBoost parser rejects invalid JSON/binary without segfaulting or terminating Python.
  - PyTorch `torch.load()` rejects unpicklable corrupt headers cleanly.
  - Failures are intercepted, reported as `MODEL_UNAVAILABLE` with status `503`, and prevented from corrupting active twin states.

---

## 4. Frontend Defensive Alignment & Contract Testing

Frontend mission control displays high-visibility warnings via `ValidationAlertBanner.jsx` backed by state management in `EngineContext.jsx`. The defensive contract guarantees that:
1. Every error payload returns a top-level `"detail"` key containing either an actionable string or Pydantic validation items.
2. `EngineContext` accurately maps HTTP error responses:
   - `400` / `404` $\to$ `INVALID_ENGINE_ID` (renders high-visibility yellow/red bounds alerts).
   - `422` $\to$ `MISSING_INPUT` (informs operator of missing simulation actions or parameters).
   - `503` $\to$ `MODEL_UNAVAILABLE` (prompts operator to verify model registry without crashing the UI).
3. The UI error boundary (`ErrorBoundary.jsx`) wraps the component tree, preventing application crashes even if network disruptions occur.

---

## 5. Verification Test Suite (`tests/test_error_handling_step9_9.py`)

All 10 required tests pass cleanly with zero warnings:

| Test ID | Method Name | Verification Target | HTTP Status | Result |
| :--- | :--- | :--- | :---: | :---: |
| **TEST-01** | `test_01_invalid_subset_returns_404` | Rejection of non-existent subset identifiers (`FD999`, `UNKNOWN`) | `404 Not Found` | **PASSED** |
| **TEST-02** | `test_02_invalid_engine_unit_returns_404` | Rejection of out-of-range engine units with valid range bounds | `404 Not Found` | **PASSED** |
| **TEST-03** | `test_03_cycle_exceedance_returns_400` | Rejection of cycle exceeding engine's operational envelope | `400 Bad Request` | **PASSED** |
| **TEST-04** | `test_04_cycle_zero_or_negative_returns_400` | Rejection of zero and negative cycle index queries | `400 Bad Request` | **PASSED** |
| **TEST-05** | `test_05_malformed_json_body_returns_422` | Rejection of missing mandatory schema fields and type mismatches | `422 Unprocessable` | **PASSED** |
| **TEST-06** | `test_06_missing_checkpoint_returns_503` | Graceful failure when model checkpoint files are missing | `503 Unavailable` | **PASSED** |
| **TEST-07** | `test_07_corrupted_checkpoint_graceful_handling` | Graceful handling of corrupted/truncated checkpoint files | `503 Unavailable` | **PASSED** |
| **TEST-08** | `test_08_error_payload_schema_compliance` | FastAPI detail schema compliance and frontend category mapping | `400/404/422/503` | **PASSED** |
| **TEST-09** | `test_09_prognostics_missing_temporal_window_handling`| Rejection of sequence tensors not adhering to 30-cycle length | `400 Bad Request` | **PASSED** |
| **TEST-10** | `test_10_dataset_and_model_immutability` | Cryptographic SHA-256 verification of datasets, scalers, checkpoints | N/A | **PASSED** |

---

## 6. Cryptographic Artifact Immutability Audit

Full cryptographic SHA-256 preservation confirmed across all 24 protected system artifacts:
- **Raw C-MAPSS Datasets (12 files in `dataset/raw/C-MAPSS/`):** 100% matched, zero modifications.
- **Phase 3 Preprocessing Scalers (4 files in `models/scalers/`):** 100% matched, zero modifications.
- **Phase 4 Champion Checkpoints (8 files in `models/checkpoints/`):** 100% matched, zero modifications.

```
FD001 Champion XGBoost:   xgb_FD001_champion.json  -> SHA-256 verified untouched
FD001 Champion Bi-LSTM:   lstm_FD001_champion.pt   -> SHA-256 verified untouched
FD002 Champion XGBoost:   xgb_FD002_champion.json  -> SHA-256 verified untouched
FD002 Champion Bi-LSTM:   lstm_FD002_champion.pt   -> SHA-256 verified untouched
FD003 Champion XGBoost:   xgb_FD003_champion.json  -> SHA-256 verified untouched
FD003 Champion Bi-LSTM:   lstm_FD003_champion.pt   -> SHA-256 verified untouched
FD004 Champion XGBoost:   xgb_FD004_champion.json  -> SHA-256 verified untouched
FD004 Champion Bi-LSTM:   lstm_FD004_champion.pt   -> SHA-256 verified untouched
```

---

## 7. Sign-off & Audit Certification

Step 9.9 is fully implemented, verified, and certified. The API backend, Digital Twin simulator, and ML model loader exhibit defensive resilience under fault injection and error conditions.
