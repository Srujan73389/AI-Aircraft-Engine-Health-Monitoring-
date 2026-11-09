"""
Unit and Integration Test Suite for Phase 9 Step 9.9: Backend Errors, Frontend Error Contracts, and Model-Loading Failures
Verifies:
1. Backend defensive routing (HTTP 404 for invalid subsets/units, HTTP 400 for cycle bounds, HTTP 422 for malformed schemas).
2. Model-loading failure resilience (missing or corrupted checkpoints/scalers handled gracefully returning HTTP 503 / MODEL_UNAVAILABLE).
3. Frontend error defense alignment (standard detail schema and category mapping for ValidationAlertBanner).
4. Temporal window validation on prognostics inference.
5. Cryptographic SHA-256 immutability of raw datasets, scalers, and champion checkpoints.

Roadmap Reference: Phase 9 - Step 9.9:
"Test backend errors, frontend errors and model-loading failures"
"""

import hashlib
import os
import unittest
from typing import Any, Dict, List

from backend.main import app
from backend.services.fault_injection_auditor import FaultInjectionAuditor
from tests.test_backend_endpoints_step8_2 import FastAPITestClientAdapter


class TestErrorHandlingStep9_9(unittest.TestCase):
    """
    Test suite for Phase 9 Step 9.9: Backend error routing, model failure resilience,
    and frontend error contract verification.
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

    def test_01_invalid_subset_returns_404(self):
        """
        Tests querying invalid subsets (e.g. FD999, UNKNOWN) returns HTTP 404
        with descriptive list of supported subsets.
        """
        # A: /fleet/{subset}/engines
        res1 = self.client.get("/api/v1/fleet/FD999/engines")
        self.assertEqual(res1.status_code, 404)
        data1 = res1.json()
        self.assertIn("detail", data1)
        self.assertIn("FD999", data1["detail"])
        self.assertIn("FD001", data1["detail"])

        # B: /engine/{subset}/1/cycle/1
        res2 = self.client.get("/api/v1/engine/UNKNOWN/1/cycle/1")
        self.assertEqual(res2.status_code, 404)
        data2 = res2.json()
        self.assertIn("detail", data2)
        self.assertIn("UNKNOWN", data2["detail"])

    def test_02_invalid_engine_unit_returns_404(self):
        """
        Tests querying invalid engine unit IDs returns HTTP 404 with valid range bounds.
        """
        # In FD001 train partition, valid unit IDs are 1 to 100
        res = self.client.get("/api/v1/engine/FD001/999/cycle/1")
        self.assertEqual(res.status_code, 404)
        data = res.json()
        self.assertIn("detail", data)
        self.assertIn("999", data["detail"])
        self.assertIn("Valid engine ID range", data["detail"])

    def test_03_cycle_exceedance_returns_400(self):
        """
        Tests querying flight cycle exceeding recorded maximum trajectory returns HTTP 400.
        """
        # FD001 Unit 1 has 192 total cycles; cycle 999 exceeds this envelope
        res = self.client.get("/api/v1/engine/FD001/1/cycle/999")
        self.assertEqual(res.status_code, 400)
        data = res.json()
        self.assertIn("detail", data)
        self.assertIn("exceeds", data["detail"].lower())
        self.assertIn("192", data["detail"])

    def test_04_cycle_zero_or_negative_returns_400(self):
        """
        Tests querying cycle 0 or negative cycle indices returns HTTP 400 Bad Request.
        """
        # Cycle 0
        res_zero = self.client.get("/api/v1/engine/FD001/1/cycle/0")
        self.assertEqual(res_zero.status_code, 400)
        data_zero = res_zero.json()
        self.assertIn("detail", data_zero)
        self.assertTrue("positive" in data_zero["detail"].lower() or "0" in data_zero["detail"])

        # Negative cycle -10
        res_neg = self.client.get("/api/v1/engine/FD001/1/cycle/-10")
        self.assertEqual(res_neg.status_code, 400)
        data_neg = res_neg.json()
        self.assertIn("detail", data_neg)
        self.assertTrue("positive" in data_neg["detail"].lower() or "-10" in data_neg["detail"])

    def test_05_malformed_json_body_returns_422(self):
        """
        Tests sending malformed payload or missing required schema fields returns HTTP 422.
        """
        # A: POST /simulation/what-if with empty body missing action_type, target_variable, proposed_value
        res_sim = self.client.post("/api/v1/simulation/what-if", json={})
        self.assertEqual(res_sim.status_code, 422)
        data_sim = res_sim.json()
        self.assertIn("detail", data_sim)

        # B: POST /monitoring/anomaly-check with illegal field type
        res_anom = self.client.post("/api/v1/monitoring/anomaly-check", json={"unit_number": "not_an_int"})
        self.assertEqual(res_anom.status_code, 422)
        data_anom = res_anom.json()
        self.assertIn("detail", data_anom)

    def test_06_missing_checkpoint_returns_503(self):
        """
        Tests ModelRegistry handles missing checkpoints gracefully, returning HTTP 503 / MODEL_UNAVAILABLE.
        """
        result_xgb = self.auditor.simulate_missing_checkpoint_error("FD001", "xgb")
        self.assertTrue(result_xgb["error_captured"])
        self.assertEqual(result_xgb["status_code"], 503)
        self.assertEqual(result_xgb["category"], "MODEL_UNAVAILABLE")
        self.assertTrue(result_xgb["handled_gracefully"])

        result_lstm = self.auditor.simulate_missing_checkpoint_error("FD002", "lstm")
        self.assertTrue(result_lstm["error_captured"])
        self.assertEqual(result_lstm["status_code"], 503)
        self.assertEqual(result_lstm["category"], "MODEL_UNAVAILABLE")
        self.assertTrue(result_lstm["handled_gracefully"])

    def test_07_corrupted_checkpoint_graceful_handling(self):
        """
        Tests ModelRegistry handles corrupted or unreadable checkpoints without crashing the interpreter.
        """
        result_corrupt_xgb = self.auditor.simulate_corrupted_checkpoint_error("FD003", "xgb")
        self.assertTrue(result_corrupt_xgb["error_captured"])
        self.assertEqual(result_corrupt_xgb["status_code"], 503)
        self.assertEqual(result_corrupt_xgb["category"], "MODEL_UNAVAILABLE")
        self.assertTrue(result_corrupt_xgb["handled_gracefully"])

        result_corrupt_lstm = self.auditor.simulate_corrupted_checkpoint_error("FD004", "lstm")
        self.assertTrue(result_corrupt_lstm["error_captured"])
        self.assertEqual(result_corrupt_lstm["status_code"], 503)
        self.assertEqual(result_corrupt_lstm["category"], "MODEL_UNAVAILABLE")
        self.assertTrue(result_corrupt_lstm["handled_gracefully"])

    def test_08_error_payload_schema_compliance(self):
        """
        Verifies that error payloads adhere to standard FastAPI detail schema
        and map properly to frontend alert banner categories.
        """
        # 1. 400 Bad Request
        c1 = self.auditor.verify_frontend_error_contract(400, {"detail": "Cycle 999 exceeds maximum cycle 192."})
        self.assertTrue(c1["contract_valid"])
        self.assertEqual(c1["mapped_frontend_category"], "INVALID_ENGINE_ID")

        # 2. 404 Not Found
        c2 = self.auditor.verify_frontend_error_contract(404, {"detail": "Engine unit 999 not found."})
        self.assertTrue(c2["contract_valid"])
        self.assertEqual(c2["mapped_frontend_category"], "INVALID_ENGINE_ID")

        # 3. 422 Unprocessable Entity
        c3 = self.auditor.verify_frontend_error_contract(422, {"detail": [{"loc": ["body", "action_type"], "msg": "field required"}]})
        self.assertTrue(c3["contract_valid"])
        self.assertEqual(c3["mapped_frontend_category"], "MISSING_INPUT")

        # 4. 503 Service Unavailable / MODEL_UNAVAILABLE
        c4 = self.auditor.verify_frontend_error_contract(503, {"detail": "MODEL_UNAVAILABLE: Checkpoint missing."})
        self.assertTrue(c4["contract_valid"])
        self.assertEqual(c4["mapped_frontend_category"], "MODEL_UNAVAILABLE")

    def test_09_prognostics_missing_temporal_window_handling(self):
        """
        Tests prognostics endpoint handles improper sequence length or invalid split defensively.
        """
        # A: Truncated temporal window (5 cycles instead of required 30)
        short_seq = [[0.0] * 14 for _ in range(5)]
        res_short = self.client.post("/api/v1/predict/rul", json={"subset": "FD001", "sequence_tensor": short_seq})
        self.assertIn(res_short.status_code, (400, 422))
        data_short = res_short.json()
        self.assertIn("detail", data_short)
        self.assertIn("30", str(data_short["detail"]))

        # B: Invalid split parameter
        res_split = self.client.post("/api/v1/predict/rul", json={"subset": "FD001", "unit_number": 1, "split": "invalid_split"})
        self.assertIn(res_split.status_code, (400, 422))
        data_split = res_split.json()
        self.assertIn("detail", data_split)
        self.assertIn("invalid split", str(data_split["detail"]).lower())

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
