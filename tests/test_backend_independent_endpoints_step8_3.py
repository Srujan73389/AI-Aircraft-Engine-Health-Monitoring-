"""
Independent Endpoint Unit and Integration Test Suite for Phase 8 Step 8.3
Verifies every backend API endpoint independently under nominal, boundary, and error conditions:
- test_01_fleet_subsets_independent_validation
- test_02_fleet_engines_boundary_and_invalid_subsets
- test_03_engine_cycle_state_bounds_and_exceedance
- test_04_sensor_trends_independent_consistency
- test_05_monitoring_anomaly_check_edge_cases
- test_06_prognostics_rul_sequence_and_trajectory_modes
- test_07_simulation_what_if_accepted_vs_review_gating
- test_08_safety_verify_rejection_and_firewall_breach
- test_09_safety_audit_trail_filtering_and_persistence
- test_10_dataset_and_model_immutability

Roadmap Reference: Phase 8 - Step 8.3:
"Test each backend endpoint independently before connecting the interface"
"""

import hashlib
import json
import os
import socket
import unittest
from typing import Dict, Any, List, Optional

from fastapi import HTTPException
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from fastapi.testclient import TestClient

from backend.main import app
from backend.schemas.api_schemas import (
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

    def _parse_url(self, url: str, params: Optional[Dict[str, Any]] = None):
        qp = dict(params or {})
        if "?" in url:
            path, qstr = url.split("?", 1)
            for pair in qstr.split("&"):
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    qp[k] = v
        else:
            path = url
        return path, qp

    def get(self, url: str, params: Optional[Dict[str, Any]] = None) -> Any:
        if self._real_client is not None:
            return self._real_client.get(url, params=params)

        path, qp = self._parse_url(url, params)
        from backend.api import fleet, monitoring, safety

        try:
            # 1. /api/v1/fleet/subsets
            if path.endswith("/fleet/subsets"):
                subsets_data = [s.model_dump() for s in fleet.get_subsets()]
                return _MockResponse(200, subsets_data)

            # 2. /api/v1/fleet/{subset}/engines
            if "/fleet/" in path and path.endswith("/engines"):
                parts = path.strip("/").split("/")
                subset = parts[parts.index("fleet") + 1]
                split_val = qp.get("split", "train")
                engines_data = fleet.get_subset_engines(subset=subset, split=split_val)
                return _MockResponse(200, engines_data)

            # 3. /api/v1/engine/{subset}/{unit}/cycle/{cycle}
            if "/engine/" in path and "/cycle/" in path:
                parts = path.strip("/").split("/")
                subset = parts[parts.index("engine") + 1]
                unit = int(parts[parts.index("engine") + 2])
                cycle = int(parts[parts.index("cycle") + 1])
                split_val = qp.get("split", "train")
                res = fleet.get_engine_cycle_state(subset=subset, unit=unit, cycle=cycle, split=split_val)
                return _MockResponse(200, res.model_dump())

            # 4. /api/v1/engine/{subset}/{unit}/sensors
            if "/engine/" in path and path.endswith("/sensors"):
                parts = path.strip("/").split("/")
                subset = parts[parts.index("engine") + 1]
                unit = int(parts[parts.index("engine") + 2])
                split_val = qp.get("split", "train")
                res = monitoring.get_engine_sensors(subset=subset, unit=unit, split=split_val)
                return _MockResponse(200, res.model_dump())

            # 5. /api/v1/safety/audit-trail
            if path.endswith("/safety/audit-trail"):
                limit = int(qp.get("limit", 50))
                decision_status = qp.get("decision_status")
                target_variable = qp.get("target_variable")
                res = safety.get_audit_trail(
                    limit=limit,
                    decision_status=decision_status,
                    target_variable=target_variable,
                )
                return _MockResponse(200, res.model_dump())

            raise ValueError(f"Unsupported GET route in adapter: {url}")
        except (ValidationError, RequestValidationError) as ve:
            return _MockResponse(422, {"detail": str(ve)})
        except HTTPException as he:
            return _MockResponse(he.status_code, {"detail": he.detail})
        except Exception as e:
            return _MockResponse(500, {"detail": str(e)})

    def post(self, url: str, json: Dict[str, Any] = None) -> Any:
        if self._real_client is not None:
            return self._real_client.post(url, json=json)

        payload = json or {}
        from backend.api import monitoring, prognostics, simulation, safety

        try:
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
        except (ValidationError, RequestValidationError) as ve:
            return _MockResponse(422, {"detail": str(ve)})
        except HTTPException as he:
            return _MockResponse(he.status_code, {"detail": he.detail})
        except Exception as e:
            return _MockResponse(500, {"detail": str(e)})


class TestBackendIndependentEndpointsStep8_3(unittest.TestCase):
    """
    Independent Endpoint Test Suite for Phase 8 Step 8.3.
    """

    @classmethod
    def setUpClass(cls):
        cls.raw_data_dir = os.path.join("dataset", "raw", "C-MAPSS")
        cls.scalers_dir = os.path.join("models", "scalers")
        cls.checkpoint_dir = os.path.join("models", "checkpoints")

        # 1. Capture cryptographic SHA-256 hashes of all 12 raw C-MAPSS files
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

        # 3. Capture SHA-256 hashes of all 8 champion checkpoints
        cls.checkpoint_hashes = {}
        if os.path.exists(cls.checkpoint_dir):
            for fname in sorted(os.listdir(cls.checkpoint_dir)):
                if fname.endswith((".json", ".pt")) and not fname.endswith("manifest.json"):
                    fpath = os.path.join(cls.checkpoint_dir, fname)
                    with open(fpath, "rb") as f:
                        cls.checkpoint_hashes[fname] = hashlib.sha256(f.read()).hexdigest()

        # Pre-warm model registry and initialize test client adapter
        cls.registry = get_model_registry(auto_load=True)
        cls.client = FastAPITestClientAdapter(app)

    def test_01_fleet_subsets_independent_validation(self):
        """
        Tests GET /api/v1/fleet/subsets confirming all 4 subsets, schema validity, and field types.
        """
        response = self.client.get("/api/v1/fleet/subsets")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 4)

        expected_engine_counts = {
            "FD001": 100,
            "FD002": 260,
            "FD003": 100,
            "FD004": 249,
        }

        subsets_seen = set()
        for s in data:
            self.assertIn("subset", s)
            self.assertIn("operating_conditions", s)
            self.assertIn("fault_modes", s)
            self.assertIn("training_engines_count", s)
            self.assertIn("test_engines_count", s)
            self.assertIn("description", s)

            sub_id = s["subset"]
            subsets_seen.add(sub_id)

            self.assertIsInstance(s["operating_conditions"], str)
            self.assertIsInstance(s["fault_modes"], list)
            self.assertIsInstance(s["training_engines_count"], int)
            self.assertIsInstance(s["test_engines_count"], int)
            self.assertIsInstance(s["description"], str)

            self.assertEqual(s["training_engines_count"], expected_engine_counts[sub_id])

        self.assertEqual(subsets_seen, {"FD001", "FD002", "FD003", "FD004"})

    def test_02_fleet_engines_boundary_and_invalid_subsets(self):
        """
        Tests GET /api/v1/fleet/{subset}/engines with valid subsets and verifies HTTP 404 for invalid subsets.
        """
        # 1. Valid subset FD001
        res_fd001 = self.client.get("/api/v1/fleet/FD001/engines")
        self.assertEqual(res_fd001.status_code, 200)
        engines_fd001 = res_fd001.json()
        self.assertEqual(len(engines_fd001), 100)
        self.assertEqual(engines_fd001[0], 1)
        self.assertEqual(engines_fd001[-1], 100)

        # 2. Valid subset FD002
        res_fd002 = self.client.get("/api/v1/fleet/FD002/engines")
        self.assertEqual(res_fd002.status_code, 200)
        engines_fd002 = res_fd002.json()
        self.assertEqual(len(engines_fd002), 260)
        self.assertEqual(engines_fd002[0], 1)
        self.assertEqual(engines_fd002[-1], 260)

        # 3. Invalid subset FD999 -> HTTP 404
        res_invalid = self.client.get("/api/v1/fleet/FD999/engines")
        self.assertEqual(res_invalid.status_code, 404)
        self.assertIn("not found", res_invalid.json().get("detail", "").lower())

        # 4. Non-existent subset string -> HTTP 404
        res_bogus = self.client.get("/api/v1/fleet/UNKNOWN_SUBSET/engines")
        self.assertEqual(res_bogus.status_code, 404)

    def test_03_engine_cycle_state_bounds_and_exceedance(self):
        """
        Tests GET /api/v1/engine/FD001/1/cycle/{cycle}; verifies nominal cycle 1 and asserts HTTP 400 when cycle exceeds max.
        """
        # 1. Nominal cycle 1
        res_nom = self.client.get("/api/v1/engine/FD001/1/cycle/1")
        self.assertEqual(res_nom.status_code, 200)
        data = res_nom.json()
        self.assertEqual(data["subset"], "FD001")
        self.assertEqual(data["unit_number"], 1)
        self.assertEqual(data["cycle"], 1)

        state = data["state_vector"]
        # Verify both canonical and G-prefixed key access
        self.assertIn("T24", state)
        self.assertIn("T30", state)
        self.assertIn("T50", state)
        self.assertIn("P30", state)
        self.assertIn("G8_T30", state)
        self.assertIn("G12_P30", state)

        # 2. Cycle exceedance beyond trajectory life (e.g. cycle 9999) -> HTTP 400
        res_exceed = self.client.get("/api/v1/engine/FD001/1/cycle/9999")
        self.assertEqual(res_exceed.status_code, 400)
        self.assertIn("exceeds", res_exceed.json().get("detail", "").lower())

        # 3. Invalid subset on cycle state -> HTTP 404
        res_sub_inv = self.client.get("/api/v1/engine/FD999/1/cycle/1")
        self.assertEqual(res_sub_inv.status_code, 404)

    def test_04_sensor_trends_independent_consistency(self):
        """
        Tests GET /api/v1/engine/FD001/1/sensors; validates sensor list, positive cycle counts, and length parity.
        """
        response = self.client.get("/api/v1/engine/FD001/1/sensors")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["subset"], "FD001")
        self.assertEqual(data["unit_number"], 1)

        total_cycles = data["total_cycles"]
        self.assertGreater(total_cycles, 0)

        cycle_history = data["cycle_history"]
        self.assertEqual(len(cycle_history), total_cycles)
        self.assertEqual(cycle_history[0], 1)
        self.assertEqual(cycle_history[-1], total_cycles)

        sensor_names = data["sensor_names"]
        self.assertIsInstance(sensor_names, list)
        self.assertGreaterEqual(len(sensor_names), 10)
        self.assertIn("sensor_2", sensor_names)
        self.assertIn("sensor_3", sensor_names)
        self.assertIn("sensor_4", sensor_names)

        # Strict length parity verification: all sensor series must match total_cycles
        trends = data["trends"]
        for sensor_id, series in trends.items():
            self.assertEqual(
                len(series),
                total_cycles,
                f"Length mismatch for {sensor_id}: expected {total_cycles}, got {len(series)}"
            )

    def test_05_monitoring_anomaly_check_edge_cases(self):
        """
        Tests POST /api/v1/monitoring/anomaly-check with nominal, single-breach caution, and dual-breach critical inputs.
        """
        # Case A: Nominal inputs (scores well below calibrated thresholds)
        payload_nom = {
            "subset": "FD001",
            "anomaly_scores": {
                "isolation_forest": -0.15,
                "pca_spe": 0.5,
            }
        }
        res_nom = self.client.post("/api/v1/monitoring/anomaly-check", json=payload_nom)
        self.assertEqual(res_nom.status_code, 200)
        data_nom = res_nom.json()
        self.assertFalse(data_nom["is_anomaly"])
        self.assertEqual(data_nom["severity"], "NOMINAL")
        self.assertFalse(data_nom["isolation_forest_breached"])
        self.assertFalse(data_nom["pca_spe_breached"])

        # Case B: Single-breach caution (iForest breaches, PCA-SPE within bounds)
        tau_if = data_nom["thresholds"]["isolation_forest"]
        tau_pca = data_nom["thresholds"]["pca_spe"]

        payload_caution = {
            "subset": "FD001",
            "anomaly_scores": {
                "isolation_forest": tau_if + 0.1,
                "pca_spe": tau_pca - 0.5,
            }
        }
        res_caution = self.client.post("/api/v1/monitoring/anomaly-check", json=payload_caution)
        self.assertEqual(res_caution.status_code, 200)
        data_caution = res_caution.json()
        self.assertTrue(data_caution["is_anomaly"])
        self.assertTrue(data_caution["isolation_forest_breached"])
        self.assertFalse(data_caution["pca_spe_breached"])
        self.assertEqual(data_caution["severity"], "CAUTION")

        # Case C: Dual-breach critical (both models breach thresholds)
        payload_crit = {
            "subset": "FD001",
            "anomaly_scores": {
                "isolation_forest": tau_if + 0.2,
                "pca_spe": tau_pca + 2.0,
            }
        }
        res_crit = self.client.post("/api/v1/monitoring/anomaly-check", json=payload_crit)
        self.assertEqual(res_crit.status_code, 200)
        data_crit = res_crit.json()
        self.assertTrue(data_crit["is_anomaly"])
        self.assertTrue(data_crit["isolation_forest_breached"])
        self.assertTrue(data_crit["pca_spe_breached"])
        self.assertEqual(data_crit["severity"], "CRITICAL")

    def test_06_prognostics_rul_sequence_and_trajectory_modes(self):
        """
        Tests POST /api/v1/predict/rul via trajectory lookup and raw sequence tensor;
        asserts confidence_score in [0, 1] and positive consensus RUL.
        """
        # 1. Mode A: Trajectory lookup mode
        payload_traj = {
            "subset": "FD001",
            "unit_number": 1,
            "cycle": 25,
        }
        res_traj = self.client.post("/api/v1/predict/rul", json=payload_traj)
        self.assertEqual(res_traj.status_code, 200)
        data_traj = res_traj.json()
        self.assertEqual(data_traj["subset"], "FD001")
        self.assertEqual(data_traj["unit_number"], 1)
        self.assertEqual(data_traj["cycle"], 25)
        self.assertGreater(data_traj["predicted_rul_xgboost"], 0.0)
        self.assertGreater(data_traj["predicted_rul_lstm"], 0.0)
        self.assertGreater(data_traj["consensus_rul"], 0.0)
        self.assertGreaterEqual(data_traj["prognostic_spread"], 0.0)
        self.assertGreaterEqual(data_traj["confidence_score"], 0.0)
        self.assertLessEqual(data_traj["confidence_score"], 1.0)

        # 2. Mode B: Direct sequence tensor mode (30 timesteps, 14 sensor features)
        sample_seq = [[0.0] * 14 for _ in range(30)]
        payload_seq = {
            "subset": "FD001",
            "sequence_tensor": sample_seq,
        }
        res_seq = self.client.post("/api/v1/predict/rul", json=payload_seq)
        self.assertEqual(res_seq.status_code, 200)
        data_seq = res_seq.json()
        self.assertGreater(data_seq["consensus_rul"], 0.0)
        self.assertGreaterEqual(data_seq["confidence_score"], 0.0)
        self.assertLessEqual(data_seq["confidence_score"], 1.0)
        self.assertIn("disclaimer", data_seq)

    def test_07_simulation_what_if_accepted_vs_review_gating(self):
        """
        Tests POST /api/v1/simulation/what-if under accepted mild offset (+2 °R) and reviewable offset (+22 °R)
        with and without operator approval.
        """
        # 1. Scenario A: Accepted mild thermal perturbation (+2 °R) -> ACCEPTED
        payload_acc = {
            "subset": "FD001",
            "unit_number": 1,
            "cycle": 1,
            "action_type": "THERMAL_PERTURBATION",
            "target_variable": "thermal_stress_offset",
            "proposed_value": 2.0,
            "operator_approval": False,
        }
        res_acc = self.client.post("/api/v1/simulation/what-if", json=payload_acc)
        self.assertEqual(res_acc.status_code, 200)
        data_acc = res_acc.json()
        self.assertEqual(data_acc["decision_status"], "ACCEPTED")
        self.assertTrue(data_acc["update_applied"])
        self.assertEqual(data_acc["state_after"]["thermal_stress_offset"], 2.0)

        # 2. Scenario B: Reviewable perturbation (+22 °R) WITHOUT operator approval -> REQUIRES_REVIEW, update rejected
        payload_rev_no_app = {
            "subset": "FD001",
            "unit_number": 1,
            "cycle": 1,
            "action_type": "THERMAL_PERTURBATION",
            "target_variable": "thermal_stress_offset",
            "proposed_value": 22.0,
            "operator_approval": False,
        }
        res_rev1 = self.client.post("/api/v1/simulation/what-if", json=payload_rev_no_app)
        self.assertEqual(res_rev1.status_code, 200)
        data_rev1 = res_rev1.json()
        self.assertEqual(data_rev1["decision_status"], "REQUIRES_REVIEW")
        self.assertFalse(data_rev1["update_applied"])
        self.assertFalse(data_rev1["operator_approved"])

        # 3. Scenario C: Reviewable perturbation (+22 °R) WITH operator approval -> REQUIRES_REVIEW, update applied
        payload_rev_app = {
            "subset": "FD001",
            "unit_number": 1,
            "cycle": 1,
            "action_type": "THERMAL_PERTURBATION",
            "target_variable": "thermal_stress_offset",
            "proposed_value": 22.0,
            "operator_approval": True,
        }
        res_rev2 = self.client.post("/api/v1/simulation/what-if", json=payload_rev_app)
        self.assertEqual(res_rev2.status_code, 200)
        data_rev2 = res_rev2.json()
        self.assertEqual(data_rev2["decision_status"], "REQUIRES_REVIEW")
        self.assertTrue(data_rev2["update_applied"])
        self.assertTrue(data_rev2["operator_approved"])

    def test_08_safety_verify_rejection_and_firewall_breach(self):
        """
        Tests POST /api/v1/safety/verify against direct telemetry mutation (T30) and severe overtemp (+50 °R);
        asserts REJECTED and valid recovery plans.
        """
        # 1. Case A: Firewall breach targeting immutable measured sensor (T30)
        payload_firewall = {
            "action_id": "act_test_telemetry_breach",
            "action_type": "SET_SENSOR_BIAS",
            "target_variable": "T30",
            "proposed_value": 1650.0,
            "subset": "FD001",
            "unit_number": 1,
            "cycle": 1,
        }
        res_fw = self.client.post("/api/v1/safety/verify", json=payload_firewall)
        self.assertEqual(res_fw.status_code, 200)
        data_fw = res_fw.json()
        self.assertEqual(data_fw["status"], "REJECTED")
        self.assertFalse(data_fw["is_safe"])
        self.assertGreater(len(data_fw["violations"]), 0)
        self.assertIsNotNone(data_fw["recovery_plan"])
        self.assertEqual(data_fw["recovery_plan"]["strategy_type"], "SCENARIO_ROLLBACK")
        self.assertTrue(data_fw["recovery_plan"]["mitigation_parameters"].get("rollback_performed", False))

        # 2. Case B: Severe overtemperature redline breach (+50 °R) -> HARD_LIMIT breach
        payload_overtemp = {
            "action_id": "act_test_severe_overtemp",
            "action_type": "THERMAL_PERTURBATION",
            "target_variable": "thermal_stress_offset",
            "proposed_value": 50.0,
            "subset": "FD001",
            "unit_number": 1,
            "cycle": 1,
        }
        res_ot = self.client.post("/api/v1/safety/verify", json=payload_overtemp)
        self.assertEqual(res_ot.status_code, 200)
        data_ot = res_ot.json()
        self.assertEqual(data_ot["status"], "REJECTED")
        self.assertFalse(data_ot["is_safe"])
        self.assertGreaterEqual(data_ot["hard_limit_count"] + data_ot["critical_count"], 1)
        self.assertIsNotNone(data_ot["recovery_plan"])
        self.assertIn("inspection", str(data_ot["recovery_plan"]["recommended_actions"]).lower())

        # 3. Case C: Prohibited command authority verb rejection
        payload_cmd = {
            "action_id": "EXECUTE_FADEC_OVERRIDE",
            "action_type": "THERMAL_PERTURBATION",
            "target_variable": "thermal_stress_offset",
            "proposed_value": 1.0,
            "subset": "FD001",
            "unit_number": 1,
            "cycle": 1,
        }
        res_cmd = self.client.post("/api/v1/safety/verify", json=payload_cmd)
        self.assertEqual(res_cmd.status_code, 200)
        data_cmd = res_cmd.json()
        self.assertEqual(data_cmd["status"], "REJECTED")
        self.assertFalse(data_cmd["is_safe"])

    def test_09_safety_audit_trail_filtering_and_persistence(self):
        """
        Tests GET /api/v1/safety/audit-trail with status and limit filters;
        validates audit record fields and academic disclaimer.
        """
        # Ensure at least one rejected action is registered in audit log
        self.test_08_safety_verify_rejection_and_firewall_breach()

        # 1. Query with limit filter
        res_lim = self.client.get("/api/v1/safety/audit-trail?limit=5")
        self.assertEqual(res_lim.status_code, 200)
        data_lim = res_lim.json()
        self.assertLessEqual(data_lim["returned_records"], 5)
        self.assertGreater(data_lim["total_records"], 0)
        self.assertIn("disclaimer", data_lim)

        # 2. Query with decision_status=REJECTED filter
        res_rej = self.client.get("/api/v1/safety/audit-trail?limit=10&decision_status=REJECTED")
        self.assertEqual(res_rej.status_code, 200)
        data_rej = res_rej.json()
        self.assertGreater(data_rej["returned_records"], 0)
        for rec in data_rej["records"]:
            self.assertEqual(rec.get("decision_status"), "REJECTED")
            self.assertIn("action_id", rec)
            self.assertIn("target_variable", rec)
            self.assertIn("timestamp", rec)

    def test_10_dataset_and_model_immutability(self):
        """
        Cryptographic SHA-256 verification of all 12 raw C-MAPSS files, 4 scalers, and 8 champion checkpoints.
        """
        # 1. Verify all 12 raw C-MAPSS dataset files
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

        # 2. Verify all 4 subset scalers
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

        # 3. Verify all 8 champion checkpoints (4 XGBoost, 4 LSTM)
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
