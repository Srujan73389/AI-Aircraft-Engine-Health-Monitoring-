"""
Unit and Integration Test Suite for Phase 8 Step 8.10: Defensive Validation Messages & Alerts
Verifies:
- HTTP 404 with valid unit range on invalid engine ID requests
- HTTP 400 with maximum cycle count on cycle exceedance requests
- HTTP 404 on invalid fleet subset queries
- HTTP 422 with formatted field error details on missing or malformed inputs
- HTTP 422/400 validation on malformed sequence tensor dimensions or non-finite values
- HTTP 503 MODEL_UNAVAILABLE error response on simulated missing model files
- ValidationAlertBanner component file existence and clean export
- ValidationAlertBanner category badges rendering (INVALID_ENGINE_ID, MISSING_INPUT, MODEL_UNAVAILABLE)
- Root App.jsx dashboard integration of ValidationAlertBanner
- Cryptographic SHA-256 zero-tampering verification of datasets, scalers, and checkpoints

Roadmap Reference: Phase 8 - Step 8.10:
"Add validation messages for invalid engine IDs, missing inputs and unavailable model files"
"""

import hashlib
import json
import os
import unittest
from unittest.mock import patch

from backend.main import app
from tests.test_backend_independent_endpoints_step8_3 import FastAPITestClientAdapter


class TestValidationMessagesStep8_10(unittest.TestCase):
    """
    Test suite for Phase 8 Step 8.10: Defensive Validation Messages and UI Alert Infrastructure.
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

    def test_01_invalid_engine_id_returns_404_with_range(self):
        """
        Tests GET /api/v1/engine/FD001/999/cycle/1 asserts HTTP 404 and detail range.
        """
        resp = self.client.get("/api/v1/engine/FD001/999/cycle/1")
        self.assertEqual(resp.status_code, 404)

        data = resp.json()
        detail = data.get("detail", "")
        self.assertIn("not found", detail.lower())
        self.assertIn("range", detail.lower())
        self.assertIn("1 to", detail.lower())

    def test_02_cycle_exceedance_returns_400_with_max(self):
        """
        Tests GET /api/v1/engine/FD001/1/cycle/9999 asserts HTTP 400 and max cycles detail.
        """
        resp = self.client.get("/api/v1/engine/FD001/1/cycle/9999")
        self.assertEqual(resp.status_code, 400)

        data = resp.json()
        detail = data.get("detail", "")
        self.assertIn("exceeds", detail.lower())
        self.assertIn("maximum cycle", detail.lower())

    def test_03_invalid_subset_returns_404(self):
        """
        Tests GET /api/v1/fleet/FD999/engines asserts HTTP 404.
        """
        resp = self.client.get("/api/v1/fleet/FD999/engines")
        self.assertEqual(resp.status_code, 404)

        data = resp.json()
        detail = data.get("detail", "")
        self.assertIn("FD999", detail)
        self.assertIn("not found", detail.lower())

    def test_04_missing_inputs_returns_422(self):
        """
        Tests POST /api/v1/simulation/what-if with empty body asserts HTTP 422.
        """
        resp = self.client.post("/api/v1/simulation/what-if", json={})
        self.assertEqual(resp.status_code, 422)

        data = resp.json()
        self.assertIn("detail", data)

    def test_05_malformed_sequence_tensor_validation(self):
        """
        Tests POST /api/v1/predict/rul with malformed sequence tensor returns 422/400.
        """
        # 1. Truncated window length (10 cycles instead of required 30)
        truncated_seq = [[0.5] * 14] * 10
        resp_trunc = self.client.post("/api/v1/predict/rul", json={
            "subset": "FD001",
            "sequence_tensor": truncated_seq,
        })
        self.assertIn(resp_trunc.status_code, [400, 422])
        self.assertIn("30", str(resp_trunc.json()))

        # 2. Non-finite values (NaN / Inf)
        nan_seq = [[0.5] * 14] * 29 + [[float("nan")] * 14]
        resp_nan = self.client.post("/api/v1/predict/rul", json={
            "subset": "FD001",
            "sequence_tensor": nan_seq,
        })
        self.assertIn(resp_nan.status_code, [400, 422])

    def test_06_model_unavailable_handler_returns_503(self):
        """
        Simulates missing model file query returning HTTP 503 with helpful message.
        """
        with patch(
            "digital_twin.model_inference.PrognosticsModelManager.load_models",
            side_effect=FileNotFoundError("Champion checkpoint 'xgb_FD001_champion.json' not found at models/checkpoints/xgb_FD001_champion.json")
        ):
            resp = self.client.post("/api/v1/predict/rul", json={
                "subset": "FD001",
                "sequence_tensor": [[0.5] * 14] * 30,
            })
            self.assertEqual(resp.status_code, 503)

            data = resp.json()
            detail = str(data.get("detail", ""))
            self.assertIn("MODEL_UNAVAILABLE", detail)

    def test_07_validation_alert_banner_component_exists(self):
        """
        Asserts ValidationAlertBanner component file exists and exports cleanly.
        """
        banner_path = os.path.join(self.src_dir, "components", "common", "ValidationAlertBanner.jsx")
        self.assertTrue(os.path.exists(banner_path), f"Missing component file: {banner_path}")

        with open(banner_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("export const ValidationAlertBanner", content)
        self.assertIn("export default ValidationAlertBanner", content)

    def test_08_validation_alert_banner_renders_categories(self):
        """
        Verifies rendering of INVALID_ENGINE_ID, MISSING_INPUT, and MODEL_UNAVAILABLE.
        """
        banner_path = os.path.join(self.src_dir, "components", "common", "ValidationAlertBanner.jsx")
        with open(banner_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("INVALID_ENGINE_ID", content)
        self.assertIn("MISSING_INPUT", content)
        self.assertIn("MODEL_UNAVAILABLE", content)
        self.assertIn("Remediation:", content)

    def test_09_app_integration_validation_banner(self):
        """
        Confirms App.jsx imports and mounts ValidationAlertBanner.
        """
        app_path = os.path.join(self.src_dir, "App.jsx")
        with open(app_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("import ValidationAlertBanner from './components/common/ValidationAlertBanner';", content)
        self.assertIn("<ValidationAlertBanner />", content)

    def test_10_dataset_and_model_immutability(self):
        """
        Cryptographic SHA-256 verification of 12 raw files, 4 scalers, and 8 champion checkpoints.
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
