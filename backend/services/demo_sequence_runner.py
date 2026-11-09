"""
Demonstration Sequence Runner Service (Phase 9 Step 9.12)
========================================================

Executes an automated, reproducible end-to-end demonstration sequence spanning
all three core Mission Control pillars:
  - Pillar 1: Current State Tracking (Fleet discovery, 21-channel telemetry grid, aerothermal ratios)
  - Pillar 2: Prognostics & Surveillance (Timeline scrubbing, dual-model RUL consensus, anomaly detection)
  - Pillar 3: What-If Sandbox & Safety Governance (Safe acceptance, redline rejection & rollback, firewall protection)

Strict Engineering & Academic Integrity Constraints:
  1. Follows PDF Master Project Roadmap Section 10, Step 9.12 and Section 1, Page 2.
  2. Measures latency per step with millisecond precision for 5-minute / 10-minute presentation budgets.
  3. Validates actual ground truth comparisons, safety decisions, and defensive rollbacks deterministically.
"""

import time
from typing import Any, Dict, List, Optional
from backend.main import app
from tests.test_backend_endpoints_step8_2 import FastAPITestClientAdapter


def get_demo_client() -> FastAPITestClientAdapter:
    """Returns a client adapter attached to the FastAPI application."""
    return FastAPITestClientAdapter(app)


def execute_demo_sequence(client: Optional[Any] = None) -> Dict[str, Any]:
    """
    Executes the 10-step demonstration sequence across the three pillars.

    Args:
        client: Optional HTTP client (defaults to FastAPITestClientAdapter).

    Returns:
        Structured dictionary containing:
          - status: 'PASSED' or 'FAILED'
          - total_latency_ms: Total sequence execution time in milliseconds
          - steps_count: Total steps executed (10)
          - steps_passed: Number of steps that satisfied all assertions
          - pillars: Dict keyed by pillar name with detailed telemetry & validation
          - step_details: Ordered list of step execution records
          - summary: Concise human-readable execution summary
    """
    if client is None:
        client = get_demo_client()

    # Pre-warm client/endpoints with a lightweight fleet query
    client.get("/api/v1/fleet/subsets")

    start_total = time.perf_counter()
    step_records: List[Dict[str, Any]] = []

    def record_step(
        step_id: str,
        name: str,
        pillar: str,
        method: str,
        endpoint: str,
        status_code: int,
        latency_ms: float,
        passed: bool,
        details: Dict[str, Any]
    ) -> Dict[str, Any]:
        rec = {
            "step_id": step_id,
            "name": name,
            "pillar": pillar,
            "method": method,
            "endpoint": endpoint,
            "status_code": status_code,
            "latency_ms": round(latency_ms, 2),
            "passed": passed,
            "details": details
        }
        step_records.append(rec)
        return rec

    # =========================================================================
    # PILLAR 1: Current State Tracking
    # =========================================================================
    # Step 1.1: Fleet Subsets Discovery
    t0 = time.perf_counter()
    r1 = client.get("/api/v1/fleet/subsets")
    t1 = time.perf_counter()
    r1_json = r1.json()
    subsets_list = [s.get("subset") for s in r1_json if isinstance(s, dict)]
    r1_passed = (
        r1.status_code == 200
        and "FD001" in subsets_list
        and len(subsets_list) == 4
    )
    record_step(
        step_id="STEP-1.1",
        name="Fleet Subsets Discovery",
        pillar="Pillar 1: Current State Tracking",
        method="GET",
        endpoint="/api/v1/fleet/subsets",
        status_code=r1.status_code,
        latency_ms=(t1 - t0) * 1000.0,
        passed=r1_passed,
        details={"available_subsets": subsets_list, "total_subsets": len(subsets_list)}
    )

    # Step 1.2: FD001 Test Fleet Selection (Unit 3)
    t0 = time.perf_counter()
    r2 = client.get("/api/v1/fleet/FD001/engines?split=test")
    t1 = time.perf_counter()
    r2_json = r2.json()
    r2_passed = (
        r2.status_code == 200
        and isinstance(r2_json, list)
        and len(r2_json) == 100
        and 3 in r2_json
    )
    record_step(
        step_id="STEP-1.2",
        name="FD001 Test Fleet Selection",
        pillar="Pillar 1: Current State Tracking",
        method="GET",
        endpoint="/api/v1/fleet/FD001/engines?split=test",
        status_code=r2.status_code,
        latency_ms=(t1 - t0) * 1000.0,
        passed=r2_passed,
        details={
            "total_test_engines": len(r2_json) if isinstance(r2_json, list) else 0,
            "selected_unit": 3,
            "unit_in_fleet": 3 in r2_json if isinstance(r2_json, list) else False
        }
    )

    # Step 1.3: FD001 Unit 3 Cycle 30 State & Aerothermal Ratios
    t0 = time.perf_counter()
    r3 = client.get("/api/v1/engine/FD001/3/cycle/30?split=test")
    t1 = time.perf_counter()
    r3_json = r3.json()
    state_vector = r3_json.get("state_vector", {})
    t30 = state_vector.get("T30") or state_vector.get("G13_T30")
    p30 = state_vector.get("P30") or state_vector.get("G15_P30")
    # Ambient / inlet pressure P2 in sea-level FD001 is standard 14.696 psia
    p2 = state_vector.get("sensor_1", 14.696)
    opr = (p30 / p2) if (p30 is not None and p2) else None
    r3_passed = (
        r3.status_code == 200
        and r3_json.get("cycle") == 30
        and t30 is not None and 1500.0 <= t30 <= 1650.0
        and p30 is not None and 500.0 <= p30 <= 600.0
        and opr is not None and 35.0 <= opr <= 42.0
    )
    record_step(
        step_id="STEP-1.3",
        name="Cycle 30 Telemetry & Aerothermal Ratios",
        pillar="Pillar 1: Current State Tracking",
        method="GET",
        endpoint="/api/v1/engine/FD001/3/cycle/30?split=test",
        status_code=r3.status_code,
        latency_ms=(t1 - t0) * 1000.0,
        passed=r3_passed,
        details={
            "cycle": 30,
            "T30": round(float(t30), 2) if t30 else None,
            "P30": round(float(p30), 2) if p30 else None,
            "OPR": round(float(opr), 2) if opr else None,
            "state_variables_count": len(state_vector)
        }
    )

    # Step 1.4: 21-Sensor Telemetry Trends
    t0 = time.perf_counter()
    r4 = client.get("/api/v1/engine/FD001/3/sensors?split=test")
    t1 = time.perf_counter()
    r4_json = r4.json()
    trends = r4_json.get("trends", {})
    r4_passed = (
        r4.status_code == 200
        and len(trends) == 21
        and all(f"sensor_{i}" in trends for i in range(1, 22))
        and r4_json.get("total_cycles") == 126
    )
    record_step(
        step_id="STEP-1.4",
        name="21-Channel Sensor Telemetry History",
        pillar="Pillar 1: Current State Tracking",
        method="GET",
        endpoint="/api/v1/engine/FD001/3/sensors?split=test",
        status_code=r4.status_code,
        latency_ms=(t1 - t0) * 1000.0,
        passed=r4_passed,
        details={
            "channel_count": len(trends),
            "total_cycles": r4_json.get("total_cycles"),
            "sensor_2_T24_readings": len(trends.get("sensor_2", []))
        }
    )

    # =========================================================================
    # PILLAR 2: Prognostics & Surveillance
    # =========================================================================
    # Step 2.1: Healthy Baseline RUL Inference at Cycle 30
    t0 = time.perf_counter()
    r5 = client.post("/api/v1/predict/rul", json={
        "subset": "FD001",
        "unit_number": 3,
        "cycle": 30,
        "split": "test"
    })
    t1 = time.perf_counter()
    r5_json = r5.json()
    consensus_c30 = r5_json.get("consensus_rul")
    xgb_c30 = r5_json.get("predicted_rul_xgboost")
    lstm_c30 = r5_json.get("predicted_rul_lstm")
    spread_c30 = r5_json.get("prognostic_spread")
    r5_passed = (
        r5.status_code == 200
        and consensus_c30 is not None
        and consensus_c30 >= 95.0
        and xgb_c30 is not None
        and lstm_c30 is not None
    )
    record_step(
        step_id="STEP-2.1",
        name="Healthy Baseline RUL (Cycle 30)",
        pillar="Pillar 2: Prognostics & Surveillance",
        method="POST",
        endpoint="/api/v1/predict/rul",
        status_code=r5.status_code,
        latency_ms=(t1 - t0) * 1000.0,
        passed=r5_passed,
        details={
            "cycle": 30,
            "consensus_rul": round(float(consensus_c30), 2) if consensus_c30 else None,
            "xgb_rul": round(float(xgb_c30), 2) if xgb_c30 else None,
            "bilstm_rul": round(float(lstm_c30), 2) if lstm_c30 else None,
            "prognostic_spread": round(float(spread_c30), 2) if spread_c30 else None,
            "confidence_score": round(float(r5_json.get("confidence_score", 0)), 4)
        }
    )

    # Step 2.2: Aged Trajectory RUL Inference at Final Cycle 126
    t0 = time.perf_counter()
    r6 = client.post("/api/v1/predict/rul", json={
        "subset": "FD001",
        "unit_number": 3,
        "cycle": 126,
        "split": "test"
    })
    t1 = time.perf_counter()
    r6_json = r6.json()
    consensus_c126 = r6_json.get("consensus_rul")
    xgb_c126 = r6_json.get("predicted_rul_xgboost")
    lstm_c126 = r6_json.get("predicted_rul_lstm")
    spread_c126 = r6_json.get("prognostic_spread")
    true_rul_c126 = 69.0  # Raw C-MAPSS RUL_FD001.txt line 3
    r6_passed = (
        r6.status_code == 200
        and consensus_c126 is not None
        and consensus_c30 is not None
        and consensus_c30 > consensus_c126  # Demonstrates monotonic prognostic degradation
        and abs(consensus_c126 - true_rul_c126) <= 30.0  # Conservative early prediction
    )
    record_step(
        step_id="STEP-2.2",
        name="Aged Trajectory RUL (Cycle 126 vs Ground Truth)",
        pillar="Pillar 2: Prognostics & Surveillance",
        method="POST",
        endpoint="/api/v1/predict/rul",
        status_code=r6.status_code,
        latency_ms=(t1 - t0) * 1000.0,
        passed=r6_passed,
        details={
            "cycle": 126,
            "consensus_rul": round(float(consensus_c126), 2) if consensus_c126 else None,
            "ground_truth_rul": true_rul_c126,
            "rul_residual": round(float(consensus_c126 - true_rul_c126), 2) if consensus_c126 else None,
            "prognostic_decay_cycles": round(float(consensus_c30 - consensus_c126), 2) if (consensus_c30 and consensus_c126) else None,
            "xgb_rul": round(float(xgb_c126), 2) if xgb_c126 else None,
            "bilstm_rul": round(float(lstm_c126), 2) if lstm_c126 else None,
            "prognostic_spread": round(float(spread_c126), 2) if spread_c126 else None
        }
    )

    # Step 2.3: Surveillance Anomaly & Health Check at Cycle 126
    t0 = time.perf_counter()
    r7 = client.post("/api/v1/monitoring/anomaly-check", json={
        "subset": "FD001",
        "unit_number": 3,
        "cycle": 126,
        "split": "test"
    })
    t1 = time.perf_counter()
    r7_json = r7.json()
    r7_passed = (
        r7.status_code == 200
        and "is_anomaly" in r7_json
        and "severity" in r7_json
        and "isolation_forest_score" in r7_json
        and "pca_spe_score" in r7_json
    )
    record_step(
        step_id="STEP-2.3",
        name="Surveillance Anomaly & Health Check",
        pillar="Pillar 2: Prognostics & Surveillance",
        method="POST",
        endpoint="/api/v1/monitoring/anomaly-check",
        status_code=r7.status_code,
        latency_ms=(t1 - t0) * 1000.0,
        passed=r7_passed,
        details={
            "cycle": 126,
            "is_anomaly": r7_json.get("is_anomaly"),
            "severity": r7_json.get("severity"),
            "iforest_score": round(float(r7_json.get("isolation_forest_score", 0)), 4),
            "pca_spe_score": round(float(r7_json.get("pca_spe_score", 0)), 4),
            "persistence_alarm": r7_json.get("persistence_alarm")
        }
    )

    # =========================================================================
    # PILLAR 3: What-If Sandbox & Safety Governance
    # =========================================================================
    # Step 3.1: Safe Thermal Perturbation (+3.0 °R) -> ACCEPTED
    t0 = time.perf_counter()
    r8 = client.post("/api/v1/simulation/what-if", json={
        "subset": "FD001",
        "unit_number": 3,
        "cycle": 126,
        "action_type": "THERMAL_PERTURBATION",
        "target_variable": "thermal_stress_offset",
        "proposed_value": 3.0
    })
    t1 = time.perf_counter()
    r8_json = r8.json()
    r8_passed = (
        r8.status_code == 200
        and r8_json.get("decision_status") == "ACCEPTED"
        and r8_json.get("update_applied") is True
        and r8_json.get("state_after") is not None
    )
    record_step(
        step_id="STEP-3.1",
        name="What-If Safe Thermal Action (+3.0 °R Acceptance)",
        pillar="Pillar 3: What-If Sandbox & Safety Governance",
        method="POST",
        endpoint="/api/v1/simulation/what-if",
        status_code=r8.status_code,
        latency_ms=(t1 - t0) * 1000.0,
        passed=r8_passed,
        details={
            "action_type": "THERMAL_PERTURBATION",
            "proposed_value": 3.0,
            "decision_status": r8_json.get("decision_status"),
            "update_applied": r8_json.get("update_applied"),
            "reason_summary": r8_json.get("reason_summary")
        }
    )

    # Step 3.2: Redline Temperature Breach (+50.0 °R) -> REJECTED & Rolled Back
    t0 = time.perf_counter()
    r9 = client.post("/api/v1/simulation/what-if", json={
        "subset": "FD001",
        "unit_number": 3,
        "cycle": 126,
        "action_type": "THERMAL_PERTURBATION",
        "target_variable": "thermal_stress_offset",
        "proposed_value": 50.0
    })
    t1 = time.perf_counter()
    r9_json = r9.json()
    rec_plan_9 = r9_json.get("recovery_plan") or {}
    mitig_9 = rec_plan_9.get("mitigation_parameters") or {}
    rollback_performed = mitig_9.get("rollback_performed", False)
    r9_passed = (
        r9.status_code == 200
        and r9_json.get("decision_status") == "REJECTED"
        and r9_json.get("update_applied") is False
        and "CONST_TEMP_T30_MAX_WARNING" in r9_json.get("reason_summary", "")
        and rollback_performed is True
    )
    record_step(
        step_id="STEP-3.2",
        name="What-If Redline Breach (+50.0 °R Rejection & Rollback)",
        pillar="Pillar 3: What-If Sandbox & Safety Governance",
        method="POST",
        endpoint="/api/v1/simulation/what-if",
        status_code=r9.status_code,
        latency_ms=(t1 - t0) * 1000.0,
        passed=r9_passed,
        details={
            "action_type": "THERMAL_PERTURBATION",
            "proposed_value": 50.0,
            "decision_status": r9_json.get("decision_status"),
            "update_applied": r9_json.get("update_applied"),
            "reason_summary": r9_json.get("reason_summary"),
            "rollback_confirmed": rollback_performed
        }
    )

    # Step 3.3: Direct Firewall Measured Immutability Breach (Direct T30 Write) -> REJECTED
    t0 = time.perf_counter()
    r10 = client.post("/api/v1/simulation/what-if", json={
        "subset": "FD001",
        "unit_number": 3,
        "cycle": 126,
        "action_type": "THERMAL_PERTURBATION",
        "target_variable": "T30",
        "proposed_value": 1650.0
    })
    t1 = time.perf_counter()
    r10_json = r10.json()
    r10_passed = (
        r10.status_code == 200
        and r10_json.get("decision_status") == "REJECTED"
        and r10_json.get("update_applied") is False
        and "FIREWALL" in str(r10_json.get("reason_summary", "")).upper()
    )
    record_step(
        step_id="STEP-3.3",
        name="What-If Firewall Defense (Direct Telemetry Immutability)",
        pillar="Pillar 3: What-If Sandbox & Safety Governance",
        method="POST",
        endpoint="/api/v1/simulation/what-if",
        status_code=r10.status_code,
        latency_ms=(t1 - t0) * 1000.0,
        passed=r10_passed,
        details={
            "target_variable": "T30",
            "decision_status": r10_json.get("decision_status"),
            "update_applied": r10_json.get("update_applied"),
            "reason_summary": r10_json.get("reason_summary")
        }
    )

    end_total = time.perf_counter()
    total_latency_ms = (end_total - start_total) * 1000.0
    steps_passed = sum(1 for s in step_records if s["passed"])
    overall_status = "PASSED" if steps_passed == len(step_records) else "FAILED"

    pillars_summary = {
        "pillar_1_current_state": {
            "name": "Current State Tracking",
            "steps": [s for s in step_records if s["pillar"].startswith("Pillar 1")],
            "latency_ms": round(sum(s["latency_ms"] for s in step_records if s["pillar"].startswith("Pillar 1")), 2),
            "passed": all(s["passed"] for s in step_records if s["pillar"].startswith("Pillar 1"))
        },
        "pillar_2_prognostics_surveillance": {
            "name": "Prognostics & Surveillance",
            "steps": [s for s in step_records if s["pillar"].startswith("Pillar 2")],
            "latency_ms": round(sum(s["latency_ms"] for s in step_records if s["pillar"].startswith("Pillar 2")), 2),
            "passed": all(s["passed"] for s in step_records if s["pillar"].startswith("Pillar 2"))
        },
        "pillar_3_whatif_sandbox": {
            "name": "What-If Sandbox & Safety Governance",
            "steps": [s for s in step_records if s["pillar"].startswith("Pillar 3")],
            "latency_ms": round(sum(s["latency_ms"] for s in step_records if s["pillar"].startswith("Pillar 3")), 2),
            "passed": all(s["passed"] for s in step_records if s["pillar"].startswith("Pillar 3"))
        }
    }

    summary = (
        f"Demonstration Sequence completed with status {overall_status} "
        f"({steps_passed}/{len(step_records)} steps passed in {total_latency_ms:.2f} ms)."
    )

    return {
        "status": overall_status,
        "total_latency_ms": round(total_latency_ms, 2),
        "steps_count": len(step_records),
        "steps_passed": steps_passed,
        "pillars": pillars_summary,
        "step_details": step_records,
        "summary": summary
    }
