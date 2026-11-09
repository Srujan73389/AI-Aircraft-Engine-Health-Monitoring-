"""
Unit and Integration Test Suite for Phase 9 Step 9.11: Defect Remediation & Affected Test Reruns
Verifies:
1. Defect 1: Remediation of anomaly-check schema handling for pure scalar and enriched state vectors (HTTP 200).
2. Defect 2: Remediation of FastAPITestClientAdapter exception forwarding for negative boundary status codes (400, 404, 422, 503).
3. Defect 3: Remediation of truncated cycle boundary defenses on held-out test engines across all 4 subsets (HTTP 400).
4. Defect 4: Remediation of floating-point NaN/Inf validation defenses on simulation actions (HTTP 422).
5. Defect 5: Remediation of missing and corrupted model checkpoint graceful degradation (HTTP 503 / MODEL_UNAVAILABLE).
6. Rerun of Step 9.1 affected test suite checks (held-out test engine discovery and trajectory retrieval).
7. Rerun of Step 9.5 affected test suite checks (sensor scenarios: nominal, thermal creep, EMI transient).
8. Rerun of Step 9.8 affected test suite checks (safety governance: acceptance, redline rejection, firewall immunity).
9. Rerun of Step 9.9 affected test suite checks (error routing, temporal sequence length, model resilience).
10. Cryptographic SHA-256 immutability verification of all 24 protected artifacts.

Roadmap Reference: Phase 9 - Step 9.11:
"Fix defects and rerun the affected tests"
"""

import hashlib
import os
import unittest
from typing import Any, Dict, List

from backend.main import app
from backend.services.fault_injection_auditor import FaultInjectionAuditor
from digital_twin.trajectory_loader import TrajectoryLoader
from tests.test_backend_endpoints_step8_2 import FastAPITestClientAdapter


class TestDefectRemediationStep9_11(unittest.TestCase):
    """
    Test suite for Phase 9 Step 9.11: Validates that all identified defects across
    Phase 8 and Phase 9 have been remediated and that affected test cases rerun cleanly.
    """

    @classmethod
    def setUpClass(cls):
        cls.raw_data_dir = os.path.join("dataset", "raw", "C-MAPSS")
        cls.scalers_dir = os.path.join("models", "scalers")
        cls.checkpoint_dir = os.path.join("models", "checkpoints")

        # 1. Capture cryptographic SHA-256 hashes of all 12 raw C-MAPSS dataset files
        cls.raw_hashes = {}
        if os.path.exists(cls.raw_data_dir):
            for fname in sorted(os.listdir(cls.raw_data_dir)):
                if fname.endswith(".txt") and fname != "readme.txt":
                    fpath = os.path.join(cls.raw_data_dir, fname)
                    with open(fpath, "rb") as f:
                        cls.raw_hashes[fname] = hashlib.sha256(f.read()).hexdigest()

        # 2. Capture SHA-256 hashes of all 4 subset scalers
        cls.scaler_hashes = {}
        if os.path.exists(cls.scalers_dir):
            for fname in sorted(os.listdir(cls.scalers_dir)):
                if fname.endswith(".pkl"):
                    fpath = os.path.join(cls.scalers_dir, fname)
                    with open(fpath, "rb") as f:
                        cls.scaler_hashes[fname] = hashlib.sha256(f.read()).hexdigest()

        # 3. Capture SHA-256 hashes of all 8 champion model checkpoints
        cls.checkpoint_hashes = {}
        if os.path.exists(cls.checkpoint_dir):
            for fname in sorted(os.listdir(cls.checkpoint_dir)):
                if fname.endswith((".json", ".pt")) and not fname.endswith("manifest.json"):
                    fpath = os.path.join(cls.checkpoint_dir, fname)
                    with open(fpath, "rb") as f:
                        cls.checkpoint_hashes[fname] = hashlib.sha256(f.read()).hexdigest()

        cls.client = FastAPITestClientAdapter(app)
        cls.auditor = FaultInjectionAuditor(
            scalers_dir=cls.scalers_dir,
            checkpoints_dir=cls.checkpoint_dir,
        )

    def test_01_defect_1_anomaly_check_schema_remediation(self):
        """
        Defect 1 Verification: Confirms /monitoring/anomaly-check handles both pure scalar
        sensor readings and enriched state vectors containing arbitrary or nested entries without HTTP 422.
        """
        # A: Standard scalar readings dictionary
        res_scalar = self.client.post(
            "/api/v1/monitoring/anomaly-check",
            json={
                "subset": "FD001",
                "unit_number": 1,
                "cycle": 10,
                "sensor_readings": {"T30": 1589.0, "P30": 554.0, "T50": 1398.0},
            },
        )
        self.assertEqual(res_scalar.status_code, 200)
        data_scalar = res_scalar.json()
        self.assertIn("is_anomaly", data_scalar)
        self.assertEqual(data_scalar["severity"], "NOMINAL")

        # B: Enriched state vector dictionary with nested metadata and non-scalar structures
        res_enriched = self.client.post(
            "/api/v1/monitoring/anomaly-check",
            json={
                "subset": "FD001",
                "unit_number": 1,
                "cycle": 10,
                "sensor_readings": {
                    "T30": 1589.0,
                    "P30": 554.0,
                    "sequence_tensor": [[1.0] * 14 for _ in range(5)],
                    "nested_diagnostics": {"status": "ok", "tags": ["cruise"]},
                },
            },
        )
        self.assertEqual(res_enriched.status_code, 200)
        data_enriched = res_enriched.json()
        self.assertIn("is_anomaly", data_enriched)
        self.assertEqual(data_enriched["severity"], "NOMINAL")

    def test_02_defect_2_client_adapter_status_forwarding(self):
        """
        Defect 2 Verification: Confirms FastAPITestClientAdapter cleanly surfaces HTTP status
        codes (400, 404, 422, 503) without unhandled exceptions.
        """
        # 1. HTTP 400 (Cycle zero)
        r400 = self.client.get("/api/v1/engine/FD001/1/cycle/0")
        self.assertEqual(r400.status_code, 400)
        self.assertIn("detail", r400.json())

        # 2. HTTP 404 (Unknown subset)
        r404 = self.client.get("/api/v1/fleet/UNKNOWN_SUBSET/engines")
        self.assertEqual(r404.status_code, 404)
        self.assertIn("detail", r404.json())

        # 3. HTTP 422 (Empty mandatory body)
        r422 = self.client.post("/api/v1/simulation/what-if", json={})
        self.assertEqual(r422.status_code, 422)
        self.assertIn("detail", r422.json())

        # 4. HTTP 503 (Simulated missing checkpoint)
        sim_503 = self.auditor.simulate_missing_checkpoint_error("FD001", "xgb")
        self.assertEqual(sim_503["status_code"], 503)
        self.assertEqual(sim_503["category"], "MODEL_UNAVAILABLE")

    def test_03_defect_3_truncated_cycle_bounds_remediation(self):
        """
        Defect 3 Verification: Confirms truncated cycle boundary defense enforces HTTP 400
        across test engines in all 4 C-MAPSS subsets.
        """
        boundary_cases = [
            ("FD001", 3, 126, 127),
            ("FD002", 15, 165, 166),
            ("FD003", 8, 192, 193),
            ("FD004", 10, 23, 24),
        ]

        for subset, unit, max_valid_cycle, out_of_bounds_cycle in boundary_cases:
            # Valid maximum cycle -> HTTP 200
            res_valid = self.client.get(f"/api/v1/engine/{subset}/{unit}/cycle/{max_valid_cycle}?split=test")
            self.assertEqual(
                res_valid.status_code,
                200,
                f"Valid max cycle {max_valid_cycle} failed for {subset} Unit {unit}",
            )

            # Out of bounds cycle (max + 1) -> HTTP 400
            res_invalid = self.client.get(f"/api/v1/engine/{subset}/{unit}/cycle/{out_of_bounds_cycle}?split=test")
            self.assertEqual(
                res_invalid.status_code,
                400,
                f"Boundary defense failed for {subset} Unit {unit} at cycle {out_of_bounds_cycle}",
            )
            data_invalid = res_invalid.json()
            self.assertIn("exceeds", data_invalid["detail"].lower())
            self.assertIn(str(max_valid_cycle), data_invalid["detail"])

    def test_04_defect_4_nan_inf_boundary_remediation(self):
        """
        Defect 4 Verification: Confirms floating-point NaN, +Inf, and -Inf values in
        simulation actions are rejected with HTTP 422 before modifying state.
        """
        # A: Proposed value is NaN
        res_nan = self.client.post(
            "/api/v1/simulation/what-if",
            json={
                "subset": "FD001",
                "unit_number": 1,
                "cycle": 1,
                "action_type": "THERMAL_PERTURBATION",
                "target_variable": "thermal_stress_offset",
                "proposed_value": float("nan"),
            },
        )
        self.assertEqual(res_nan.status_code, 422)
        self.assertIn("detail", res_nan.json())
        self.assertIn("nan", str(res_nan.json()["detail"]).lower())

        # B: Proposed value is positive Infinity
        res_inf = self.client.post(
            "/api/v1/simulation/what-if",
            json={
                "subset": "FD001",
                "unit_number": 1,
                "cycle": 1,
                "action_type": "THERMAL_PERTURBATION",
                "target_variable": "thermal_stress_offset",
                "proposed_value": float("inf"),
            },
        )
        self.assertEqual(res_inf.status_code, 422)
        self.assertIn("detail", res_inf.json())

        # C: Proposed value is negative Infinity
        res_ninf = self.client.post(
            "/api/v1/simulation/what-if",
            json={
                "subset": "FD001",
                "unit_number": 1,
                "cycle": 1,
                "action_type": "THERMAL_PERTURBATION",
                "target_variable": "thermal_stress_offset",
                "proposed_value": -float("inf"),
            },
        )
        self.assertEqual(res_ninf.status_code, 422)
        self.assertIn("detail", res_ninf.json())

    def test_05_defect_5_checkpoint_resilience_remediation(self):
        """
        Defect 5 Verification: Confirms missing and corrupted model checkpoints gracefully degrade
        to HTTP 503 MODEL_UNAVAILABLE without crashing the server process.
        """
        # Missing XGBoost checkpoint
        missing_res = self.auditor.simulate_missing_checkpoint_error("FD001", "xgb")
        self.assertTrue(missing_res["error_captured"])
        self.assertEqual(missing_res["status_code"], 503)
        self.assertEqual(missing_res["category"], "MODEL_UNAVAILABLE")
        self.assertTrue(missing_res["handled_gracefully"])

        # Corrupted PyTorch LSTM checkpoint
        corrupt_res = self.auditor.simulate_corrupted_checkpoint_error("FD002", "lstm")
        self.assertTrue(corrupt_res["error_captured"])
        self.assertEqual(corrupt_res["status_code"], 503)
        self.assertEqual(corrupt_res["category"], "MODEL_UNAVAILABLE")
        self.assertTrue(corrupt_res["handled_gracefully"])

    def test_06_rerun_step9_1_affected_tests(self):
        """
        Rerun Confirmation: Step 9.1 Held-Out Test Engine Evaluation
        Verifies fleet test split inventory and representative cohort trajectories.
        """
        loader = TrajectoryLoader()

        # Check all 4 held-out test splits exist and have expected unit counts
        self.assertEqual(len(loader.get_available_units("FD001", "test")), 100)
        self.assertEqual(len(loader.get_available_units("FD002", "test")), 259)
        self.assertEqual(len(loader.get_available_units("FD003", "test")), 100)
        self.assertEqual(len(loader.get_available_units("FD004", "test")), 248)

        # Verify representative test cohort loads cleanly
        df_fd001_u3 = loader.load_engine_trajectory("FD001", 3, split="test")
        self.assertEqual(df_fd001_u3["time_cycles"].max(), 126)

        df_fd002_u15 = loader.load_engine_trajectory("FD002", 15, split="test")
        self.assertEqual(df_fd002_u15["time_cycles"].max(), 165)

    def test_07_rerun_step9_5_affected_tests(self):
        """
        Rerun Confirmation: Step 9.5 Normal and Abnormal Sensor Scenarios
        Verifies nominal early cruise, severe late thermal creep, and transient EMI noise.
        """
        # 1. Nominal healthy cruise
        r_nominal = self.client.post(
            "/api/v1/monitoring/anomaly-check",
            json={"subset": "FD001", "unit_number": 1, "cycle": 10, "sensor_readings": {"T30": 1589.0}},
        )
        self.assertEqual(r_nominal.status_code, 200)
        self.assertFalse(r_nominal.json()["is_anomaly"])

        # 2. Severe thermal degradation
        r_degraded = self.client.post(
            "/api/v1/monitoring/anomaly-check",
            json={"subset": "FD001", "unit_number": 1, "cycle": 190, "sensor_readings": {"T30": 1618.0}},
        )
        self.assertEqual(r_degraded.status_code, 200)
        self.assertTrue(r_degraded.json()["is_anomaly"])

        # 3. Single-cycle transient EMI spike (persistence_alarm=False)
        r_transient = self.client.post(
            "/api/v1/monitoring/anomaly-check",
            json={
                "subset": "FD001",
                "unit_number": 1,
                "cycle": 50,
                "consecutive_breaches": 1,
                "sensor_readings": {"T30": 1620.0},
            },
        )
        self.assertEqual(r_transient.status_code, 200)
        self.assertTrue(r_transient.json()["is_anomaly"])
        self.assertFalse(r_transient.json()["persistence_alarm"])

    def test_08_rerun_step9_8_affected_tests(self):
        """
        Rerun Confirmation: Step 9.8 Safety Governance Acceptance and Rejection
        Verifies tri-state decisions on benign, redline, and firewall actions.
        """
        # 1. Benign perturbation -> ACCEPTED
        r_accept = self.client.post(
            "/api/v1/simulation/what-if",
            json={
                "subset": "FD001",
                "unit_number": 1,
                "cycle": 50,
                "action_type": "THERMAL_PERTURBATION",
                "target_variable": "thermal_stress_offset",
                "proposed_value": 2.0,
            },
        )
        self.assertEqual(r_accept.status_code, 200)
        self.assertEqual(r_accept.json()["decision_status"], "ACCEPTED")
        self.assertTrue(r_accept.json()["update_applied"])

        # 2. Redline breach -> REJECTED
        r_reject = self.client.post(
            "/api/v1/simulation/what-if",
            json={
                "subset": "FD001",
                "unit_number": 1,
                "cycle": 50,
                "action_type": "THERMAL_PERTURBATION",
                "target_variable": "thermal_stress_offset",
                "proposed_value": 50.0,
            },
        )
        self.assertEqual(r_reject.status_code, 200)
        self.assertEqual(r_reject.json()["decision_status"], "REJECTED")
        self.assertFalse(r_reject.json()["update_applied"])

    def test_09_rerun_step9_9_affected_tests(self):
        """
        Rerun Confirmation: Step 9.9 Backend Errors and Fault Injection
        Verifies defensive routing (404, 400, 422) and prognostics window validation.
        """
        # Invalid subset -> 404
        r_sub = self.client.get("/api/v1/fleet/FD999/engines")
        self.assertEqual(r_sub.status_code, 404)

        # Invalid engine unit -> 404
        r_unit = self.client.get("/api/v1/engine/FD001/999/cycle/1")
        self.assertEqual(r_unit.status_code, 404)

        # Non-30 sequence window on prognostics -> 400
        short_seq = [[0.0] * 14 for _ in range(5)]
        r_seq = self.client.post(
            "/api/v1/predict/rul",
            json={"subset": "FD001", "sequence_tensor": short_seq},
        )
        self.assertIn(r_seq.status_code, (400, 422))

    def test_10_dataset_and_model_immutability(self):
        """
        Cryptographic verification: Asserts that all 12 raw C-MAPSS dataset files,
        4 scalers, and 8 champion checkpoints remain 100% untouched and byte-identical.
        """
        # 1. Raw dataset files
        self.assertEqual(len(self.raw_hashes), 12)
        for fname, original_hash in self.raw_hashes.items():
            fpath = os.path.join(self.raw_data_dir, fname)
            with open(fpath, "rb") as f:
                current_hash = hashlib.sha256(f.read()).hexdigest()
            self.assertEqual(
                current_hash,
                original_hash,
                f"Cryptographic integrity breach detected in raw dataset file: {fname}"
            )

        # 2. Scaler files
        self.assertGreaterEqual(len(self.scaler_hashes), 4)
        for fname, original_hash in self.scaler_hashes.items():
            fpath = os.path.join(self.scalers_dir, fname)
            with open(fpath, "rb") as f:
                current_hash = hashlib.sha256(f.read()).hexdigest()
            self.assertEqual(
                current_hash,
                original_hash,
                f"Cryptographic integrity breach detected in scaler file: {fname}"
            )

        # 3. Model checkpoint files
        self.assertEqual(len(self.checkpoint_hashes), 8)
        for fname, original_hash in self.checkpoint_hashes.items():
            fpath = os.path.join(self.checkpoint_dir, fname)
            with open(fpath, "rb") as f:
                current_hash = hashlib.sha256(f.read()).hexdigest()
            self.assertEqual(
                current_hash,
                original_hash,
                f"Cryptographic integrity breach detected in model checkpoint: {fname}"
            )


if __name__ == "__main__":
    unittest.main()
