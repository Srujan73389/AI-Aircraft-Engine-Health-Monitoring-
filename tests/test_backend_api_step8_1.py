"""
Unit and Integration Test Suite for Phase 8 Step 8.1: Backend API & Model Loader Service
Verifies:
- In-memory ModelRegistry initialization, artifact discovery, and caching.
- Zero-leakage loading of Phase 3 scalers and Phase 4 champion models (XGBoost, LSTM, iForest, PCA-SPE).
- FastAPI application creation, CORS middleware, and route dispatching (/health, /api/models/status).
- Inclusion of the mandatory academic simulation non-authority disclaimer.
- Cryptographic SHA-256 immutability of raw datasets and champion model checkpoints.

Roadmap Reference: Phase 8 - Step 8.1:
"Create a backend API that loads the saved preprocessing pipeline and trained models"
"""

import hashlib
import json
import os
import socket
import unittest
from typing import Dict, Any

import torch.nn as nn
import xgboost as xgb
from fastapi.testclient import TestClient

from backend.main import app, health, get_models_status, root
from backend.services.model_loader import (
    ModelRegistry,
    AnomalyDetector,
    CHAMPION_SPECS,
    SUPPORTED_SUBSETS,
    get_model_registry,
)
from safety.disclaimers import ACADEMIC_SIMULATION_DISCLAIMER


def _is_loopback_allowed() -> bool:
    """Helper to detect if localhost TCP socket loopback is permitted in the current environment."""
    s = None
    c = None
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.2)
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.listen(1)
        c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        c.settimeout(0.2)
        c.connect(("127.0.0.1", port))
        return True
    except Exception:
        return False
    finally:
        if c:
            try:
                c.close()
            except Exception:
                pass
        if s:
            try:
                s.close()
            except Exception:
                pass


class _MockResponse:
    """Lightweight response wrapper mimicking TestClient response API."""
    def __init__(self, status_code: int, json_data: Dict[str, Any]):
        self.status_code = status_code
        self._json_data = json_data

    def json(self) -> Dict[str, Any]:
        return self._json_data


class FastAPITestClientAdapter:
    """
    Dual-mode test client adapter:
    Uses standard TestClient(app) when loopback network access is allowed,
    or direct FastAPI handler execution in sandboxed loopback-restricted environments.
    """
    def __init__(self, fast_app):
        self.app = fast_app
        self._real_client = TestClient(fast_app) if _is_loopback_allowed() else None

    def get(self, url: str) -> Any:
        if self._real_client is not None:
            return self._real_client.get(url)
        reg = get_model_registry(auto_load=True)
        if url == "/health":
            return _MockResponse(200, health(reg))
        elif url == "/api/models/status":
            return _MockResponse(200, get_models_status(reg))
        elif url == "/":
            return _MockResponse(200, root())
        raise ValueError(f"Unsupported endpoint route in adapter: {url}")


class TestBackendApiStep8_1(unittest.TestCase):
    """
    Test suite for Phase 8 Step 8.1 Backend API and Model Loader Service.
    """

    @classmethod
    def setUpClass(cls):
        cls.raw_data_dir = os.path.join("dataset", "raw", "C-MAPSS")
        cls.scalers_dir = os.path.join("models", "scalers")
        cls.checkpoint_dir = os.path.join("models", "checkpoints")

        # 1. Capture SHA-256 hashes of all 12 raw C-MAPSS dataset files
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

        # Initialize registry singleton and test client adapter
        cls.registry = get_model_registry(auto_load=True)
        cls.client = FastAPITestClientAdapter(app)

    def test_01_model_registry_initialization(self):
        """
        Confirms registry initializes and successfully discovers all artifact paths on disk.
        """
        reg = ModelRegistry()
        inventory = reg.discover_artifacts()

        self.assertIn("scalers", inventory)
        self.assertIn("checkpoints", inventory)
        self.assertTrue(inventory["anomaly_manifest_present"], "Anomaly manifest must exist on disk.")

        for sub in SUPPORTED_SUBSETS:
            self.assertTrue(inventory["scalers"][sub]["exists"], f"Scaler for {sub} must be discovered.")
            self.assertTrue(inventory["checkpoints"][sub]["xgb"]["exists"], f"XGB checkpoint for {sub} must exist.")
            self.assertTrue(inventory["checkpoints"][sub]["lstm"]["exists"], f"LSTM checkpoint for {sub} must exist.")

    def test_02_load_all_four_scalers(self):
        """
        Verifies all 4 subset scalers (FD001 - FD004) load into memory and have 24 scaled features.
        """
        self.assertEqual(self.registry.scalers_count, 4, "Expected exactly 4 scalers loaded in registry.")

        for sub in SUPPORTED_SUBSETS:
            scaler = self.registry.get_scaler(sub)
            self.assertIsNotNone(scaler)
            self.assertTrue(scaler.is_fitted)
            self.assertEqual(scaler.scaler.__class__.__name__, "SafeMinMaxScaler")
            self.assertEqual(len(scaler.feature_columns), 24)

    def test_03_load_xgboost_models(self):
        """
        Verifies all 4 XGBoost champion checkpoints load into memory and match feature dimensions.
        """
        for sub in SUPPORTED_SUBSETS:
            model = self.registry.get_model(sub, "xgboost")
            self.assertIsInstance(model, xgb.XGBRegressor)
            expected_dim = CHAMPION_SPECS[sub]["xgb"]["feature_dim"]
            self.assertEqual(
                model.n_features_in_,
                expected_dim,
                f"XGBoost {sub} feature dimension mismatch: expected {expected_dim}, got {model.n_features_in_}",
            )

    def test_04_load_lstm_models(self):
        """
        Verifies all 4 PyTorch Bi-LSTM champion checkpoints instantiate correctly in eval mode.
        """
        for sub in SUPPORTED_SUBSETS:
            model = self.registry.get_model(sub, "lstm")
            self.assertIsInstance(model, nn.Module)
            self.assertFalse(model.training, f"LSTM model for {sub} must be loaded in evaluation mode.")

    def test_05_load_anomaly_models(self):
        """
        Verifies Isolation Forest and PCA-SPE anomaly detectors are loaded for all 4 subsets.
        """
        for sub in SUPPORTED_SUBSETS:
            # 1. Isolation Forest detector
            det_if = self.registry.get_model(sub, "isolation_forest")
            self.assertIsInstance(det_if, AnomalyDetector)
            self.assertEqual(det_if.model_type, "isolation_forest")
            self.assertIsInstance(det_if.threshold, float)
            self.assertEqual(len(det_if.feature_columns), 14)
            # Verify threshold breach classification logic
            self.assertTrue(det_if.is_anomaly(det_if.threshold + 0.1))
            self.assertFalse(det_if.is_anomaly(det_if.threshold - 0.1))

            # 2. PCA-SPE detector
            det_pca = self.registry.get_model(sub, "pca_spe")
            self.assertIsInstance(det_pca, AnomalyDetector)
            self.assertEqual(det_pca.model_type, "pca_spe")
            self.assertIsInstance(det_pca.threshold, float)
            self.assertEqual(len(det_pca.feature_columns), 14)

    def test_06_fastapi_app_creation(self):
        """
        Tests FastAPI application metadata configuration and CORS middleware enablement.
        """
        self.assertEqual(app.title, "Aircraft Engine AI Health Monitoring & Prognostics API")
        self.assertEqual(app.version, "1.0.0")

        # Verify CORS middleware is registered
        middleware_names = [m.cls.__name__ for m in app.user_middleware]
        self.assertIn("CORSMiddleware", middleware_names)

    def test_07_health_endpoint(self):
        """
        Tests GET /health endpoint returning 200 OK and expected registry counts.
        """
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["version"], "1.0.0")
        self.assertTrue(data["ready"])
        self.assertEqual(data["scalers_loaded"], 4)
        self.assertEqual(data["champion_checkpoints_loaded"], 8)
        self.assertGreaterEqual(data["models_loaded"], 8)

    def test_08_models_status_endpoint(self):
        """
        Tests GET /api/models/status verifying all 4 scalers and 8 champion models are reported ready.
        """
        response = self.client.get("/api/models/status")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["status"], "success")
        reg_info = data["data"]
        self.assertTrue(reg_info["ready"])
        self.assertEqual(reg_info["scalers_loaded_count"], 4)
        self.assertEqual(reg_info["champion_checkpoints_loaded"], 8)

        # Confirm all 4 subsets report loaded scalers and models
        for sub in SUPPORTED_SUBSETS:
            self.assertTrue(reg_info["scalers"][sub]["loaded"])
            self.assertTrue(reg_info["models"][sub]["xgboost"]["loaded"])
            self.assertTrue(reg_info["models"][sub]["lstm"]["loaded"])
            self.assertTrue(reg_info["models"][sub]["isolation_forest"]["loaded"])
            self.assertTrue(reg_info["models"][sub]["pca_spe"]["loaded"])

    def test_09_academic_disclaimer_in_response(self):
        """
        Asserts that API responses include the mandatory academic simulation non-authority notice.
        """
        for endpoint in ["/health", "/api/models/status"]:
            response = self.client.get(endpoint)
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn("disclaimer", data)
            self.assertEqual(data["disclaimer"], ACADEMIC_SIMULATION_DISCLAIMER)
            self.assertIn("RESEARCH & ACADEMIC SIMULATION ONLY", data["disclaimer"])

    def test_10_dataset_and_model_immutability(self):
        """
        Cryptographic SHA-256 verification ensuring that raw datasets, scalers,
        and champion model checkpoints remain bit-for-bit immutable.
        """
        # 1. Verify all 12 raw C-MAPSS dataset files
        self.assertEqual(len(self.raw_hashes), 12, "Expected 12 raw C-MAPSS text files.")
        for fname, expected_hash in self.raw_hashes.items():
            fpath = os.path.join(self.raw_data_dir, fname)
            with open(fpath, "rb") as f:
                current_hash = hashlib.sha256(f.read()).hexdigest()
            self.assertEqual(
                current_hash,
                expected_hash,
                f"Raw dataset file '{fname}' was modified during Step 8.1 execution!",
            )

        # 2. Verify all 4 subset scalers
        self.assertEqual(len(self.scaler_hashes), 4, "Expected 4 subset scalers.")
        for fname, expected_hash in self.scaler_hashes.items():
            fpath = os.path.join(self.scalers_dir, fname)
            with open(fpath, "rb") as f:
                current_hash = hashlib.sha256(f.read()).hexdigest()
            self.assertEqual(
                current_hash,
                expected_hash,
                f"Scaler file '{fname}' was modified during Step 8.1 execution!",
            )

        # 3. Verify all 8 champion model checkpoints
        self.assertEqual(len(self.checkpoint_hashes), 8, "Expected 8 champion checkpoints.")
        for fname, expected_hash in self.checkpoint_hashes.items():
            fpath = os.path.join(self.checkpoint_dir, fname)
            with open(fpath, "rb") as f:
                current_hash = hashlib.sha256(f.read()).hexdigest()
            self.assertEqual(
                current_hash,
                expected_hash,
                f"Champion checkpoint '{fname}' was modified during Step 8.1 execution!",
            )


if __name__ == "__main__":
    unittest.main()
