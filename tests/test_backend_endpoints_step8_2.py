"""
Unit and Integration Test Suite for Phase 8 Step 8.2: Backend API Endpoints
Verifies:
- Fleet discovery and engine trajectory state endpoints (/fleet/subsets, /fleet/{subset}/engines, /engine/.../cycle/...)
- Monitoring sensor history trends and statistical anomaly check (/monitoring/anomaly-check)
- Prognostics dual-model Remaining Useful Life prediction (/predict/rul)
- What-If simulation scenario perturbations (/simulation/what-if)
- Pre-execution safety verification and closed-loop state gating (/safety/verify, /safety/execute-gated)
- Cryptographic SHA-256 immutability of raw datasets, scalers, and champion checkpoints.

Roadmap Reference: Phase 8 - Step 8.2:
"Create endpoints for dataset/engine selection, monitoring, RUL prediction, simulation and safety checks"
"""

import hashlib
import json
import os
import socket
import unittest
from typing import Dict, Any, List

from fastapi.testclient import TestClient

from backend.main import app
from backend.schemas.api_schemas import (
    FleetSubsetInfo,
    AnomalyCheckRequest,
    RULPredictionRequest,
    SimulationWhatIfRequest,
    SafetyVerificationRequest,
)
from backend.services.model_loader import get_model_registry
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
    """Response wrapper mimicking TestClient response object."""
    def __init__(self, status_code: int, json_data: Any):
        self.status_code = status_code
        self._json_data = json_data

    def json(self) -> Any:
        return self._json_data


class FastAPITestClientAdapter:
    """
    Dual-mode test client adapter:
    Uses standard TestClient(app) when loopback network access is allowed,
    or direct FastAPI router execution in sandboxed loopback-restricted environments.
    """
    def __init__(self, fast_app):
        self.app = fast_app
        self._real_client = TestClient(fast_app) if _is_loopback_allowed() else None

    def get(self, url: str) -> Any:
        if self._real_client is not None:
            return self._real_client.get(url)

        # Direct route resolution for sandboxed mode
        from backend.api import fleet, monitoring, safety

        # 1. /api/v1/fleet/subsets
        if url.endswith("/fleet/subsets"):
            subsets_data = [s.model_dump() for s in fleet.get_subsets()]
            return _MockResponse(200, subsets_data)

        # 2. /api/v1/fleet/{subset}/engines
        if "/fleet/" in url and url.endswith("/engines"):
            parts = url.strip("/").split("/")
            subset = parts[parts.index("fleet") + 1]
            engines_data = fleet.get_subset_engines(subset=subset)
            return _MockResponse(200, engines_data)

        # 3. /api/v1/engine/{subset}/{unit}/cycle/{cycle}
        if "/engine/" in url and "/cycle/" in url:
            parts = url.strip("/").split("/")
            subset = parts[parts.index("engine") + 1]
            unit = int(parts[parts.index("engine") + 2])
            cycle = int(parts[parts.index("cycle") + 1])
            res = fleet.get_engine_cycle_state(subset=subset, unit=unit, cycle=cycle)
            return _MockResponse(200, res.model_dump())

        # 4. /api/v1/engine/{subset}/{unit}/sensors
        if "/engine/" in url and url.endswith("/sensors"):
            parts = url.strip("/").split("/")
            subset = parts[parts.index("engine") + 1]
            unit = int(parts[parts.index("engine") + 2])
            res = monitoring.get_engine_sensors(subset=subset, unit=unit)
            return _MockResponse(200, res.model_dump())

        # 5. /api/v1/safety/audit-trail
        if url.endswith("/safety/audit-trail"):
            res = safety.get_audit_trail()
            return _MockResponse(200, res.model_dump())

        raise ValueError(f"Unsupported GET route in adapter: {url}")

    def post(self, url: str, json: Dict[str, Any] = None) -> Any:
        if self._real_client is not None:
            return self._real_client.post(url, json=json)

        payload = json or {}
        from backend.api import monitoring, prognostics, simulation, safety

        if url.endswith("/monitoring/anomaly-check"):
            req = AnomalyCheckRequest(**payload)
            res = monitoring.check_anomaly(req)
            return _MockResponse(200, res.model_dump())

        elif url.endswith("/predict/rul"):
            req = RULPredictionRequest(**payload)
            res = prognostics.predict_rul(req)
            return _MockResponse(200, res.model_dump())

        elif url.endswith("/simulation/what-if"):
            req = SimulationWhatIfRequest(**payload)
            res = simulation.run_simulation_what_if(req)
            return _MockResponse(200, res.model_dump())

        elif url.endswith("/safety/verify"):
            req = SafetyVerificationRequest(**payload)
            res = safety.verify_safety_action(req)
            return _MockResponse(200, res.model_dump())

        elif url.endswith("/safety/execute-gated"):
            req = SimulationWhatIfRequest(**payload)
            res = safety.execute_gated_action(req)
            return _MockResponse(200, res.model_dump())

        raise ValueError(f"Unsupported POST route in adapter: {url}")


class TestBackendEndpointsStep8_2(unittest.TestCase):
    """
    Test suite for Phase 8 Step 8.2 Backend API endpoints.
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

        # Pre-warm model registry and test client adapter
        cls.registry = get_model_registry(auto_load=True)
        cls.client = FastAPITestClientAdapter(app)

    def test_01_fleet_subsets_endpoint(self):
        """
        Tests GET /api/v1/fleet/subsets confirming all 4 subsets (FD001 - FD004) are returned with metadata.
        """
        response = self.client.get("/api/v1/fleet/subsets")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(len(data), 4)
        subsets = [item["subset"] for item in data]
        self.assertListEqual(sorted(subsets), ["FD001", "FD002", "FD003", "FD004"])

        # Check metadata fields
        for item in data:
            self.assertIn("operating_conditions", item)
            self.assertIn("fault_modes", item)
            self.assertIn("training_engines_count", item)

    def test_02_fleet_engines_endpoint(self):
        """
        Tests GET /api/v1/fleet/FD001/engines confirming available engine unit numbers are returned.
        """
        response = self.client.get("/api/v1/fleet/FD001/engines")
        self.assertEqual(response.status_code, 200)

        units = response.json()
        self.assertIsInstance(units, list)
        self.assertEqual(len(units), 100)  # FD001 has 100 training engines
        self.assertEqual(units[0], 1)
        self.assertEqual(units[-1], 100)

    def test_03_engine_cycle_state_endpoint(self):
        """
        Tests GET /api/v1/engine/FD001/1/cycle/1 confirming full 40-variable state vector retrieval.
        """
        response = self.client.get("/api/v1/engine/FD001/1/cycle/1")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["subset"], "FD001")
        self.assertEqual(data["unit_number"], 1)
        self.assertEqual(data["cycle"], 1)

        state = data["state_vector"]
        # Confirm essential variables across G1 - G40
        self.assertIn("unit_number", state)
        self.assertIn("time_cycles", state)
        self.assertIn("T24", state)
        self.assertIn("T30", state)
        self.assertIn("T50", state)
        self.assertIn("P30", state)
        self.assertIn("disclaimer", data)

    def test_04_sensor_trends_endpoint(self):
        """
        Tests GET /api/v1/engine/FD001/1/sensors verifying timeseries history for trending.
        """
        response = self.client.get("/api/v1/engine/FD001/1/sensors")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["subset"], "FD001")
        self.assertEqual(data["unit_number"], 1)
        self.assertGreaterEqual(data["total_cycles"], 100)
        self.assertIn("sensor_3", data["trends"])  # T30
        self.assertIn("sensor_4", data["trends"])  # T50
        self.assertEqual(len(data["cycle_history"]), data["total_cycles"])

    def test_05_anomaly_check_endpoint(self):
        """
        Tests POST /api/v1/monitoring/anomaly-check with nominal vs anomalous sensor patterns.
        """
        # Case A: Nominal scores below 95th-percentile thresholds
        payload_nom = {
            "subset": "FD001",
            "anomaly_scores": {
                "isolation_forest": -0.05,  # Below -0.0311
                "pca_spe": 1.20,            # Below 1.6100
            }
        }
        res_nom = self.client.post("/api/v1/monitoring/anomaly-check", json=payload_nom)
        self.assertEqual(res_nom.status_code, 200)
        data_nom = res_nom.json()
        self.assertFalse(data_nom["is_anomaly"])
        self.assertEqual(data_nom["severity"], "NOMINAL")

        # Case B: Anomalous scores breaching thresholds
        payload_anom = {
            "subset": "FD001",
            "anomaly_scores": {
                "isolation_forest": 0.05,  # Breached
                "pca_spe": 2.50,           # Breached
            }
        }
        res_anom = self.client.post("/api/v1/monitoring/anomaly-check", json=payload_anom)
        self.assertEqual(res_anom.status_code, 200)
        data_anom = res_anom.json()
        self.assertTrue(data_anom["is_anomaly"])
        self.assertTrue(data_anom["isolation_forest_breached"])
        self.assertTrue(data_anom["pca_spe_breached"])
        self.assertEqual(data_anom["severity"], "CRITICAL")

    def test_06_predict_rul_endpoint(self):
        """
        Tests POST /api/v1/predict/rul validating dual-model predictions (XGBoost, LSTM) and consensus spread.
        """
        payload = {
            "subset": "FD001",
            "unit_number": 1,
            "cycle": 1,
        }
        response = self.client.post("/api/v1/predict/rul", json=payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["subset"], "FD001")
        self.assertGreater(data["predicted_rul_xgboost"], 0.0)
        self.assertGreater(data["predicted_rul_lstm"], 0.0)
        self.assertGreaterEqual(data["prognostic_spread"], 0.0)
        self.assertGreater(data["consensus_rul"], 0.0)
        self.assertGreaterEqual(data["confidence_score"], 0.0)
        self.assertLessEqual(data["confidence_score"], 1.0)
        self.assertIn("disclaimer", data)

    def test_07_simulation_what_if_endpoint(self):
        """
        Tests POST /api/v1/simulation/what-if executing a mild thermal perturbation.
        """
        payload = {
            "subset": "FD001",
            "unit_number": 1,
            "cycle": 1,
            "action_type": "THERMAL_PERTURBATION",
            "target_variable": "thermal_stress_offset",
            "proposed_value": 4.0,
            "operator_approval": False,
        }
        response = self.client.post("/api/v1/simulation/what-if", json=payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["decision_status"], "ACCEPTED")
        self.assertTrue(data["update_applied"])
        self.assertEqual(data["state_after"]["thermal_stress_offset"], 4.0)
        self.assertIn("disclaimer", data)

    def test_08_safety_verify_endpoint(self):
        """
        Tests POST /api/v1/safety/verify evaluating candidate actions and returning tri-state verdicts.
        """
        # Case A: Nominal mild offset (+3 °R) -> ACCEPTED
        payload_safe = {
            "subset": "FD001",
            "unit_number": 1,
            "cycle": 1,
            "action_type": "THERMAL_PERTURBATION",
            "target_variable": "thermal_stress_offset",
            "proposed_value": 3.0,
        }
        res_safe = self.client.post("/api/v1/safety/verify", json=payload_safe)
        self.assertEqual(res_safe.status_code, 200)
        data_safe = res_safe.json()
        self.assertEqual(data_safe["status"], "ACCEPTED")
        self.assertTrue(data_safe["is_safe"])
        self.assertEqual(len(data_safe["violations"]), 0)

        # Case B: Severe overtemp (+50 °R) -> REJECTED with inspection plan
        payload_unsafe = {
            "subset": "FD001",
            "unit_number": 1,
            "cycle": 1,
            "action_type": "THERMAL_PERTURBATION",
            "target_variable": "thermal_stress_offset",
            "proposed_value": 50.0,
        }
        res_unsafe = self.client.post("/api/v1/safety/verify", json=payload_unsafe)
        self.assertEqual(res_unsafe.status_code, 200)
        data_unsafe = res_unsafe.json()
        self.assertEqual(data_unsafe["status"], "REJECTED")
        self.assertFalse(data_unsafe["is_safe"])
        self.assertGreaterEqual(len(data_unsafe["violations"]), 1)
        self.assertIsNotNone(data_unsafe["recovery_plan"])
        self.assertEqual(data_unsafe["recovery_plan"]["strategy_type"], "PHYSICAL_INSPECTION")

    def test_09_safety_execute_gated_endpoint(self):
        """
        Tests POST /api/v1/safety/execute-gated verifying transactional gating and rollback.
        """
        # Attempt direct mutation of immutable telemetry T30 -> REJECTED
        payload_illegal = {
            "subset": "FD001",
            "unit_number": 1,
            "cycle": 1,
            "action_type": "STATE_ENRICHMENT",
            "target_variable": "T30",
            "proposed_value": 1650.0,
            "operator_approval": False,
        }
        response = self.client.post("/api/v1/safety/execute-gated", json=payload_illegal)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["decision_status"], "REJECTED")
        self.assertFalse(data["update_applied"])
        self.assertIsNotNone(data["recovery_plan"])
        self.assertEqual(data["recovery_plan"]["strategy_type"], "SCENARIO_ROLLBACK")

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
                f"Raw dataset file '{fname}' was modified during Step 8.2 execution!",
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
                f"Scaler file '{fname}' was modified during Step 8.2 execution!",
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
                f"Champion checkpoint '{fname}' was modified during Step 8.2 execution!",
            )


if __name__ == "__main__":
    unittest.main()
