"""
Standalone Endpoint Smoke Test Diagnostic Script
Phase 8 - Step 8.3: Independent Endpoint Verification

Systematically exercises all 10 core backend API endpoints against the FastAPI application,
recording HTTP status codes, latency in milliseconds, and schema/behavioral verdicts.

Roadmap Reference: Section 9, Step 8.3:
"Test each backend endpoint independently before connecting the interface"
"""

import os
import sys
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Dict, Any, List, Tuple, Callable

from backend.main import app
from tests.test_backend_independent_endpoints_step8_3 import FastAPITestClientAdapter


def run_smoke_tests() -> int:
    client = FastAPITestClientAdapter(app)

    scenarios: List[Dict[str, Any]] = [
        {
            "endpoint": "/api/v1/fleet/subsets",
            "method": "GET",
            "scenario": "Catalog metadata check (4 subsets)",
            "call": lambda: client.get("/api/v1/fleet/subsets"),
            "expected_status": 200,
            "validate": lambda data: len(data) == 4 and all(s["subset"] in ["FD001", "FD002", "FD003", "FD004"] for s in data),
        },
        {
            "endpoint": "/api/v1/fleet/FD001/engines",
            "method": "GET",
            "scenario": "Trajectory unit discovery (FD001)",
            "call": lambda: client.get("/api/v1/fleet/FD001/engines"),
            "expected_status": 200,
            "validate": lambda data: len(data) == 100 and data[0] == 1,
        },
        {
            "endpoint": "/api/v1/fleet/FD999/engines",
            "method": "GET",
            "scenario": "Error boundary check on invalid subset",
            "call": lambda: client.get("/api/v1/fleet/FD999/engines"),
            "expected_status": 404,
            "validate": lambda data: "not found" in data.get("detail", "").lower(),
        },
        {
            "endpoint": "/api/v1/engine/FD001/1/cycle/1",
            "method": "GET",
            "scenario": "Full 40-variable state vector inquiry",
            "call": lambda: client.get("/api/v1/engine/FD001/1/cycle/1"),
            "expected_status": 200,
            "validate": lambda data: "T30" in data.get("state_vector", {}) and data.get("cycle") == 1,
        },
        {
            "endpoint": "/api/v1/engine/FD001/1/sensors",
            "method": "GET",
            "scenario": "Multi-channel sensor trends history",
            "call": lambda: client.get("/api/v1/engine/FD001/1/sensors"),
            "expected_status": 200,
            "validate": lambda data: data.get("total_cycles", 0) > 0 and len(data.get("trends", {})) >= 10,
        },
        {
            "endpoint": "/api/v1/monitoring/anomaly-check",
            "method": "POST",
            "scenario": "Statistical baseline anomaly scoring",
            "call": lambda: client.post(
                "/api/v1/monitoring/anomaly-check",
                json={
                    "subset": "FD001",
                    "anomaly_scores": {"isolation_forest": -0.15, "pca_spe": 0.5},
                },
            ),
            "expected_status": 200,
            "validate": lambda data: data.get("is_anomaly") is False and data.get("severity") == "NOMINAL",
        },
        {
            "endpoint": "/api/v1/predict/rul",
            "method": "POST",
            "scenario": "XGBoost & Bi-LSTM consensus RUL",
            "call": lambda: client.post(
                "/api/v1/predict/rul",
                json={"subset": "FD001", "unit_number": 1, "cycle": 20},
            ),
            "expected_status": 200,
            "validate": lambda data: data.get("consensus_rul", 0.0) > 0.0 and 0.0 <= data.get("confidence_score", 0.0) <= 1.0,
        },
        {
            "endpoint": "/api/v1/simulation/what-if",
            "method": "POST",
            "scenario": "Closed-loop safe thermal perturbation",
            "call": lambda: client.post(
                "/api/v1/simulation/what-if",
                json={
                    "subset": "FD001",
                    "unit_number": 1,
                    "cycle": 1,
                    "action_type": "THERMAL_PERTURBATION",
                    "target_variable": "thermal_stress_offset",
                    "proposed_value": 2.0,
                    "operator_approval": False,
                },
            ),
            "expected_status": 200,
            "validate": lambda data: data.get("decision_status") == "ACCEPTED" and data.get("update_applied") is True,
        },
        {
            "endpoint": "/api/v1/safety/verify",
            "method": "POST",
            "scenario": "Firewall telemetry immutability check",
            "call": lambda: client.post(
                "/api/v1/safety/verify",
                json={
                    "action_id": "act_smoke_fw_check",
                    "action_type": "SET_SENSOR_BIAS",
                    "target_variable": "T30",
                    "proposed_value": 1650.0,
                    "subset": "FD001",
                    "unit_number": 1,
                    "cycle": 1,
                },
            ),
            "expected_status": 200,
            "validate": lambda data: data.get("status") == "REJECTED" and data.get("is_safe") is False,
        },
        {
            "endpoint": "/api/v1/safety/audit-trail",
            "method": "GET",
            "scenario": "Cryptographic audit trail query",
            "call": lambda: client.get("/api/v1/safety/audit-trail?limit=5"),
            "expected_status": 200,
            "validate": lambda data: data.get("returned_records", 0) <= 5 and "disclaimer" in data,
        },
    ]

    header_fmt = "| {0:<35} | {1:<6} | {2:<38} | {3:<6} | {4:<9} | {5:<7} |"
    divider = "+" + "-" * 37 + "+" + "-" * 8 + "+" + "-" * 40 + "+" + "-" * 8 + "+" + "-" * 11 + "+" + "-" * 9 + "+"

    print("\n" + "=" * 120)
    print(" C-MAPSS AIRCRAFT ENGINE AI DIGITAL TWIN - INDEPENDENT BACKEND ENDPOINT SMOKE TEST")
    print("=" * 120)
    print(divider)
    print(header_fmt.format("Endpoint", "Method", "Test Scenario", "Status", "Latency", "Verdict"))
    print(divider)

    all_passed = True
    total_latency = 0.0

    for item in scenarios:
        t0 = time.perf_counter()
        try:
            resp = item["call"]()
            latency = (time.perf_counter() - t0) * 1000
            total_latency += latency
            status_code = resp.status_code
            json_data = resp.json()

            status_ok = status_code == item["expected_status"]
            valid_ok = False
            if status_ok:
                try:
                    valid_ok = item["validate"](json_data)
                except Exception:
                    valid_ok = False

            verdict = "PASS" if (status_ok and valid_ok) else "FAIL"
            if verdict == "FAIL":
                all_passed = False

            lat_str = f"{latency:6.1f} ms"
            print(header_fmt.format(item["endpoint"], item["method"], item["scenario"], status_code, lat_str, verdict))
        except Exception as exc:
            all_passed = False
            latency = (time.perf_counter() - t0) * 1000
            lat_str = f"{latency:6.1f} ms"
            print(header_fmt.format(item["endpoint"], item["method"], item["scenario"], "ERR", lat_str, "FAIL"))
            print(f"   [!] Exception: {exc}")

    print(divider)
    avg_latency = total_latency / len(scenarios) if scenarios else 0.0
    print(f"Summary: {len(scenarios)} endpoints evaluated. Average Latency: {avg_latency:.1f} ms.")

    if all_passed:
        print("Verdict: ALL 10 INDEPENDENT ENDPOINT TESTS PASSED CLEANLY (100% SUCCESS)\n")
        return 0
    else:
        print("Verdict: ONE OR MORE ENDPOINT TESTS FAILED\n")
        return 1


if __name__ == "__main__":
    sys.exit(run_smoke_tests())
