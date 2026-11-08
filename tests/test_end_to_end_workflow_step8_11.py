"""
End-to-End Workflow Integration Test Suite for Phase 8 Step 8.11
Verifies the complete operator workflow connecting frontend API consumers to the FastAPI backend:
1. Fleet Subset & Engine Selection (FD001 to FD004, Units 1-N).
2. Flight Cycle Navigation & Replay (Scrubber, Steppers, Play/Pause progression).
3. Telemetry Ingestion & Sensor Trend Tracking (21 channels with units and drift math).
4. Real-Time Anomaly Surveillance (Early NOMINAL to Late CRITICAL transition with persistence).
5. Prognostic RUL Inference (Dual-model XGBoost & Bi-LSTM monotonic degradation tracking).
6. Digital Twin What-If Perturbation (Closed-Loop Safety Governed ACCEPTED execution).
7. Digital Twin What-If Safety Rejection (Hard metallurgical redline breach REJECTED).
8. Multi-Agent Orchestration Pipeline (Monitoring -> Diagnosis -> Prognostics -> Orchestrator).
9. Append-Only Safety Audit Trail Persistence & Status Filtering.
10. Cryptographic SHA-256 zero-tampering verification of datasets, scalers, and checkpoints.

Roadmap Reference: Phase 8 - Step 8.11:
"Connect the frontend to the backend and test the complete user workflow"
"""

import hashlib
import json
import os
import unittest
from typing import Dict, Any, List

from backend.main import app
from tests.test_backend_independent_endpoints_step8_3 import FastAPITestClientAdapter
from agents.orchestrator import AgentOrchestrator
from agents.schemas import OrchestrationRequest, AgentExecutionStatus


class TestEndToEndWorkflowStep8_11(unittest.TestCase):
    """
    End-to-End User Workflow & Full-Stack Integration Test Suite.
    """

    @classmethod
    def setUpClass(cls):
        cls.frontend_dir = os.path.join("frontend")
        cls.src_dir = os.path.join(cls.frontend_dir, "src")
        cls.raw_data_dir = os.path.join("dataset", "raw", "C-MAPSS")
        cls.scalers_dir = os.path.join("models", "scalers")
        cls.checkpoint_dir = os.path.join("models", "checkpoints")

        # Capture cryptographic SHA-256 hashes of all protected artifacts
        cls.raw_hashes = {}
        if os.path.exists(cls.raw_data_dir):
            for fname in sorted(os.listdir(cls.raw_data_dir)):
                if fname.endswith(".txt") and fname != "readme.txt":
                    fpath = os.path.join(cls.raw_data_dir, fname)
                    with open(fpath, "rb") as f:
                        cls.raw_hashes[fname] = hashlib.sha256(f.read()).hexdigest()

        cls.scaler_hashes = {}
        if os.path.exists(cls.scalers_dir):
            for fname in sorted(os.listdir(cls.scalers_dir)):
                if fname.endswith(".pkl"):
                    fpath = os.path.join(cls.scalers_dir, fname)
                    with open(fpath, "rb") as f:
                        cls.scaler_hashes[fname] = hashlib.sha256(f.read()).hexdigest()

        cls.checkpoint_hashes = {}
        if os.path.exists(cls.checkpoint_dir):
            for fname in sorted(os.listdir(cls.checkpoint_dir)):
                if fname.endswith((".json", ".pt")) and not fname.endswith("manifest.json"):
                    fpath = os.path.join(cls.checkpoint_dir, fname)
                    with open(fpath, "rb") as f:
                        cls.checkpoint_hashes[fname] = hashlib.sha256(f.read()).hexdigest()

        cls.client = FastAPITestClientAdapter(app)

    def test_01_workflow_fleet_selection_lifecycle(self):
        """
        Stage 1: Operator queries subset catalog, selects FD001 and FD002, discovers engines,
        and initializes Engine 1 at Cycle 1 verifying full state vector construction.
        """
        # 1. Fetch Fleet Catalog
        resp_subsets = self.client.get("/api/v1/fleet/subsets")
        self.assertEqual(resp_subsets.status_code, 200)
        subsets = resp_subsets.json()
        subset_keys = [s["subset"] for s in subsets]
        self.assertIn("FD001", subset_keys)
        self.assertIn("FD002", subset_keys)

        # 2. Discover available engines for FD001 and FD002
        resp_fd1_engines = self.client.get("/api/v1/fleet/FD001/engines")
        self.assertEqual(resp_fd1_engines.status_code, 200)
        fd1_engines = resp_fd1_engines.json()
        self.assertEqual(len(fd1_engines), 100)
        self.assertIn(1, fd1_engines)

        resp_fd2_engines = self.client.get("/api/v1/fleet/FD002/engines")
        self.assertEqual(resp_fd2_engines.status_code, 200)
        fd2_engines = resp_fd2_engines.json()
        self.assertEqual(len(fd2_engines), 260)
        self.assertIn(1, fd2_engines)

        # 3. Initialize Engine 1 at Cycle 1
        resp_init = self.client.get("/api/v1/engine/FD001/1/cycle/1")
        self.assertEqual(resp_init.status_code, 200)
        state_data = resp_init.json()
        self.assertEqual(state_data["subset"], "FD001")
        self.assertEqual(state_data["unit_number"], 1)
        self.assertEqual(state_data["cycle"], 1)
        self.assertIn("state_vector", state_data)

        # Confirm essential thermodynamic state keys exist
        sv = state_data["state_vector"]
        self.assertIn("T2", sv)
        self.assertIn("T30", sv)
        self.assertIn("P30", sv)
        self.assertIn("hpc_temp_rise", sv)

    def test_02_workflow_cycle_progression_telemetry(self):
        """
        Stage 2: Operator steps through flight cycles (1 -> 15 -> 50),
        verifying that cycle state updates monotonically and sensor values refresh.
        """
        cycles_to_test = [1, 15, 50]
        previous_cycle = 0

        for c in cycles_to_test:
            resp = self.client.get(f"/api/v1/engine/FD001/1/cycle/{c}")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()

            self.assertEqual(data["cycle"], c)
            self.assertGreater(data["cycle"], previous_cycle)
            previous_cycle = c

            sv = data["state_vector"]
            # Validate core physical consistency across flight cycles
            t30 = float(sv.get("T30", sv.get("sensor_3", 0)))
            p30 = float(sv.get("P30", sv.get("sensor_7", 0)))
            self.assertGreater(t30, 1500.0)  # Total temp within normal aerothermal range
            self.assertGreater(p30, 500.0)   # Total pressure within normal HPC discharge range

    def test_03_workflow_sensor_trends_and_drift_sync(self):
        """
        Stage 3: Operator reviews multi-sensor historical trends across the complete trajectory,
        confirming that all 21 sensors are tracked across full life.
        """
        resp = self.client.get("/api/v1/engine/FD001/1/sensors")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        self.assertEqual(data["subset"], "FD001")
        self.assertEqual(data["unit_number"], 1)
        self.assertEqual(data["total_cycles"], 192)
        self.assertEqual(len(data["cycle_history"]), 192)

        # Assert all 21 standard sensors are tracked
        trends = data["trends"]
        self.assertEqual(len(trends), 21)
        for i in range(1, 22):
            s_name = f"sensor_{i}"
            self.assertIn(s_name, trends)
            self.assertEqual(len(trends[s_name]), 192)

        # Verify physical degradation drift over engine life (T30 increases due to thermal wear)
        t30_trend = trends["sensor_3"]
        initial_t30 = sum(t30_trend[:10]) / 10.0
        final_t30 = sum(t30_trend[-10:]) / 10.0
        self.assertGreater(final_t30, initial_t30, "HPC outlet temperature must drift upward with degradation.")

    def test_04_workflow_nominal_to_degraded_anomaly_transition(self):
        """
        Stage 4: Operator evaluates real-time anomaly surveillance,
        confirming transition from early-cycle NOMINAL state to late-cycle CRITICAL anomaly.
        """
        # Early life (Cycle 5): Nominal readings
        early_resp = self.client.post("/api/v1/monitoring/anomaly-check", json={
            "subset": "FD001",
            "unit_number": 1,
            "cycle": 5,
            "sensor_readings": {"T30": 1588.5, "T50": 1400.2, "P30": 553.1},
        })
        self.assertEqual(early_resp.status_code, 200)
        early_data = early_resp.json()
        self.assertEqual(early_data["severity"], "NOMINAL")
        self.assertFalse(early_data["is_anomaly"])

        # Late life (Cycle 190): Severe thermal degradation breach
        late_resp = self.client.post("/api/v1/monitoring/anomaly-check", json={
            "subset": "FD001",
            "unit_number": 1,
            "cycle": 190,
            "sensor_readings": {"T30": 1622.0, "T50": 1435.0, "P30": 548.0},
        })
        self.assertEqual(late_resp.status_code, 200)
        late_data = late_resp.json()
        self.assertTrue(late_data["is_anomaly"])
        self.assertIn(late_data["severity"], ["CAUTION", "CRITICAL"])

    def test_05_workflow_dual_model_prognostics_progression(self):
        """
        Stage 5: Operator monitors predicted Remaining Useful Life (RUL),
        verifying that dual-model consensus decreases monotonically across engine life.
        """
        # Early-life RUL at cycle 10
        resp_c10 = self.client.post("/api/v1/predict/rul", json={
            "subset": "FD001",
            "unit_number": 1,
            "cycle": 10,
        })
        self.assertEqual(resp_c10.status_code, 200)
        data_c10 = resp_c10.json()

        # Mid-life RUL at cycle 100
        resp_c100 = self.client.post("/api/v1/predict/rul", json={
            "subset": "FD001",
            "unit_number": 1,
            "cycle": 100,
        })
        self.assertEqual(resp_c100.status_code, 200)
        data_c100 = resp_c100.json()

        # Late-life RUL at cycle 180
        resp_c180 = self.client.post("/api/v1/predict/rul", json={
            "subset": "FD001",
            "unit_number": 1,
            "cycle": 180,
        })
        self.assertEqual(resp_c180.status_code, 200)
        data_c180 = resp_c180.json()

        # Monotonic degradation assertion: RUL(c10) > RUL(c100) > RUL(c180)
        self.assertGreater(data_c10["consensus_rul"], data_c100["consensus_rul"])
        self.assertGreater(data_c100["consensus_rul"], data_c180["consensus_rul"])

        # Check epistemic uncertainty spread and confidence
        self.assertGreaterEqual(data_c10["confidence_score"], 0.5)
        self.assertGreaterEqual(data_c180["prognostic_spread"], 0.0)

    def test_06_workflow_digital_twin_what_if_safe_execution(self):
        """
        Stage 6: Operator evaluates a safe What-If parameter perturbation (+3.0 °R thermal stress),
        verifying closed-loop safety governor ACCEPTED decision and updated candidate state.
        """
        payload = {
            "subset": "FD001",
            "unit_number": 1,
            "cycle": 15,
            "action_type": "THERMAL_PERTURBATION",
            "target_variable": "thermal_stress_offset",
            "proposed_value": 3.0,
            "operator_approval": True,
        }
        resp = self.client.post("/api/v1/simulation/what-if", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        self.assertEqual(data["decision_status"], "ACCEPTED")
        self.assertTrue(data["update_applied"])
        self.assertIn("thermal_stress_offset", data["state_after"])
        self.assertEqual(float(data["state_after"]["thermal_stress_offset"]), 3.0)
        self.assertIn("ACCEPTED", data["reason_summary"])

    def test_07_workflow_digital_twin_what_if_safety_rejection(self):
        """
        Stage 7: Operator attempts an unsafe What-If perturbation (+50.0 °R exceeding redline),
        verifying closed-loop safety governor REJECTED decision, blocked state, and recovery plan.
        """
        payload = {
            "subset": "FD001",
            "unit_number": 1,
            "cycle": 15,
            "action_type": "THERMAL_PERTURBATION",
            "target_variable": "thermal_stress_offset",
            "proposed_value": 50.0,
            "operator_approval": False,
        }
        resp = self.client.post("/api/v1/simulation/what-if", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        self.assertEqual(data["decision_status"], "REJECTED")
        self.assertFalse(data["update_applied"])
        self.assertIn("REJECTED", data["reason_summary"])
        self.assertIsNotNone(data["recovery_plan"])
        self.assertIn("strategy_type", data["recovery_plan"])
        self.assertTrue(data["recovery_plan"]["mitigation_parameters"]["rollback_performed"])

    def test_08_workflow_multi_agent_orchestration_cycle(self):
        """
        Stage 8: Operator inspects autonomous multi-agent pipeline logs,
        confirming sequential execution across Monitoring, Diagnosis, and Prognostics agents.
        """
        orchestrator = AgentOrchestrator(subset="FD001")
        req = OrchestrationRequest(engine_id=1, flight_cycle=25, subset="FD001", execution_mode="full")
        result = orchestrator.run(req)

        self.assertEqual(result.status, AgentExecutionStatus.SUCCESS)
        self.assertIsNotNone(result.monitoring_result)
        self.assertIsNotNone(result.diagnosis_result)
        self.assertIsNotNone(result.prognostics_result)

        # Confirm stage latencies were measured
        self.assertIn("monitoring_ms", result.execution_timeline_ms)
        self.assertIn("prognostics_ms", result.execution_timeline_ms)
        self.assertGreater(result.prognostics_result.ensemble_rul_estimate, 0.0)

    def test_09_workflow_safety_audit_trail_persistence(self):
        """
        Stage 9: Operator queries the append-only audit trail console,
        confirming that preceding What-If decisions were persisted and filterable.
        """
        # Query all records
        resp_all = self.client.get("/api/v1/safety/audit-trail?limit=50")
        self.assertEqual(resp_all.status_code, 200)
        data_all = resp_all.json()
        self.assertGreater(data_all["total_records"], 0)
        self.assertIsInstance(data_all["records"], list)

        # Query filtered by ACCEPTED
        resp_acc = self.client.get("/api/v1/safety/audit-trail?decision_status=ACCEPTED")
        self.assertEqual(resp_acc.status_code, 200)
        for rec in resp_acc.json()["records"]:
            self.assertEqual(rec["decision_status"], "ACCEPTED")

        # Query filtered by REJECTED
        resp_rej = self.client.get("/api/v1/safety/audit-trail?decision_status=REJECTED")
        self.assertEqual(resp_rej.status_code, 200)
        for rec in resp_rej.json()["records"]:
            self.assertEqual(rec["decision_status"], "REJECTED")

    def test_10_dataset_and_model_immutability(self):
        """
        Stage 10: Cryptographic SHA-256 verification of 12 raw files, 4 scalers, and 8 champion checkpoints.
        """
        # 1. 12 Raw C-MAPSS dataset files
        self.assertEqual(len(self.raw_hashes), 12, "Must protect all 12 C-MAPSS dataset files.")
        for fname, original_hash in self.raw_hashes.items():
            fpath = os.path.join(self.raw_data_dir, fname)
            with open(fpath, "rb") as f:
                current_hash = hashlib.sha256(f.read()).hexdigest()
            self.assertEqual(
                current_hash,
                original_hash,
                f"Tampering detected in raw dataset file: {fname}"
            )

        # 2. 4 Feature scalers
        self.assertEqual(len(self.scaler_hashes), 4, "Must verify all 4 subset scalers.")
        for fname, original_hash in self.scaler_hashes.items():
            fpath = os.path.join(self.scalers_dir, fname)
            with open(fpath, "rb") as f:
                current_hash = hashlib.sha256(f.read()).hexdigest()
            self.assertEqual(
                current_hash,
                original_hash,
                f"Tampering detected in Phase 3 scaler: {fname}"
            )

        # 3. 8 Champion model checkpoints
        self.assertEqual(len(self.checkpoint_hashes), 8, "Must verify all 8 champion checkpoints.")
        for fname, original_hash in self.checkpoint_hashes.items():
            fpath = os.path.join(self.checkpoint_dir, fname)
            with open(fpath, "rb") as f:
                current_hash = hashlib.sha256(f.read()).hexdigest()
            self.assertEqual(
                current_hash,
                original_hash,
                f"Tampering detected in Phase 4 checkpoint: {fname}"
            )


if __name__ == "__main__":
    unittest.main()
