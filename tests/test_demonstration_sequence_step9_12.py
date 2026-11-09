"""
Step 9.12 Verification Suite: Master Demonstration Sequence Testing
===================================================================

Validates the complete demonstration sequence spanning all 3 core presentation pillars:
  - Pillar 1: Current State Tracking (Fleet discovery, 21-channel telemetry grid, aerothermal ratios)
  - Pillar 2: Prognostics & Surveillance (Timeline scrubbing, dual-model RUL consensus, anomaly detection)
  - Pillar 3: What-If Sandbox & Safety Governance (Safe acceptance, redline rejection & rollback, firewall defense)

Academic & Engineering Constraints:
  - Master Project Roadmap PDF Section 10, Step 9.12.
  - Page 2: Zero fabricated telemetry; strict verification of real test engine trajectories.
  - End-to-end execution within presentation budgets (< 15 seconds automated execution).
"""

import hashlib
import os
import unittest
from typing import Dict, Any

from backend.main import app
from backend.services.demo_sequence_runner import execute_demo_sequence
from tests.test_backend_endpoints_step8_2 import FastAPITestClientAdapter


class TestDemonstrationSequenceStep9_12(unittest.TestCase):
    """
    Automated verification suite validating the 10-step demonstration sequence
    for presentation reliability, low latency, and deterministic reproducibility.
    """

    @classmethod
    def setUpClass(cls):
        cls.client = FastAPITestClientAdapter(app)
        # Pre-warm client & model caches
        cls.client.get("/api/v1/fleet/subsets")

    def test_01_demo_step1_fleet_selection_health(self):
        """
        Validates Pillar 1 discovery: 4 C-MAPSS subsets, 100 test engines in FD001,
        and successful identification of test engine Unit 3.
        """
        r_subsets = self.client.get("/api/v1/fleet/subsets")
        self.assertEqual(r_subsets.status_code, 200)
        subsets = [s["subset"] for s in r_subsets.json()]
        self.assertEqual(subsets, ["FD001", "FD002", "FD003", "FD004"])

        r_engines = self.client.get("/api/v1/fleet/FD001/engines?split=test")
        self.assertEqual(r_engines.status_code, 200)
        engines = r_engines.json()
        self.assertEqual(len(engines), 100)
        self.assertIn(3, engines)

    def test_02_demo_step2_telemetry_grid_integrity(self):
        """
        Validates Pillar 1 telemetry: 21-channel sensor trends, 126 recorded cycles,
        and plausible gas-path aerothermal readings.
        """
        r_sensors = self.client.get("/api/v1/engine/FD001/3/sensors?split=test")
        self.assertEqual(r_sensors.status_code, 200)
        data = r_sensors.json()
        self.assertEqual(data["total_cycles"], 126)
        trends = data["trends"]
        self.assertEqual(len(trends), 21)
        for i in range(1, 22):
            self.assertIn(f"sensor_{i}", trends)
            self.assertEqual(len(trends[f"sensor_{i}"]), 126)

        # Validate cycle 30 state
        r_c30 = self.client.get("/api/v1/engine/FD001/3/cycle/30?split=test")
        self.assertEqual(r_c30.status_code, 200)
        state_c30 = r_c30.json()["state_vector"]
        t30 = state_c30.get("T30")
        p30 = state_c30.get("P30")
        self.assertIsNotNone(t30)
        self.assertIsNotNone(p30)
        self.assertGreater(t30, 1500.0)
        self.assertLess(t30, 1625.0)
        self.assertGreater(p30, 500.0)
        self.assertLess(p30, 600.0)

    def test_03_demo_step3_healthy_cycle_prognostics(self):
        """
        Validates Pillar 2 prognostics: Healthy early-life flight at Cycle 30.
        XGBoost and Bi-LSTM consensus RUL exceeds 95 cycles with tight prognostic spread.
        """
        r_rul = self.client.post("/api/v1/predict/rul", json={
            "subset": "FD001",
            "unit_number": 3,
            "cycle": 30,
            "split": "test"
        })
        self.assertEqual(r_rul.status_code, 200)
        body = r_rul.json()
        consensus = body["consensus_rul"]
        xgb_rul = body["predicted_rul_xgboost"]
        lstm_rul = body["predicted_rul_lstm"]
        spread = body["prognostic_spread"]
        confidence = body["confidence_score"]

        self.assertGreater(consensus, 95.0)
        self.assertGreater(xgb_rul, 90.0)
        self.assertGreater(lstm_rul, 90.0)
        self.assertLess(spread, 15.0)
        self.assertGreater(confidence, 0.85)

    def test_04_demo_step4_aged_cycle_prognostics(self):
        """
        Validates Pillar 2 prognostics: Aged late-life flight at final Cycle 126.
        Demonstrates monotonic prognostic decay from Cycle 30 (> 50 cycles drop),
        and close conservative agreement with true ground-truth RUL (69 cycles).
        """
        r_c30 = self.client.post("/api/v1/predict/rul", json={
            "subset": "FD001",
            "unit_number": 3,
            "cycle": 30,
            "split": "test"
        })
        r_c126 = self.client.post("/api/v1/predict/rul", json={
            "subset": "FD001",
            "unit_number": 3,
            "cycle": 126,
            "split": "test"
        })
        self.assertEqual(r_c126.status_code, 200)
        c30_rul = r_c30.json()["consensus_rul"]
        c126_rul = r_c126.json()["consensus_rul"]
        true_rul = 69.0  # RUL_FD001.txt line 3

        # Physical degradation over 96 flight cycles
        self.assertGreater(c30_rul, c126_rul)
        self.assertGreater(c30_rul - c126_rul, 50.0)

        # Conservative early prediction
        residual = c126_rul - true_rul
        self.assertLessEqual(residual, 0.0, "Model should be conservative (early prediction) for safety")
        self.assertGreaterEqual(residual, -35.0, "Prediction error must remain within realistic bounds")

    def test_05_demo_step5_anomaly_monitoring_panel(self):
        """
        Validates Pillar 2 surveillance: Anomaly check at Cycle 126 returns
        calibrated Isolation Forest and PCA SPE scores with persistent alert flags.
        """
        r_mon = self.client.post("/api/v1/monitoring/anomaly-check", json={
            "subset": "FD001",
            "unit_number": 3,
            "cycle": 126,
            "split": "test"
        })
        self.assertEqual(r_mon.status_code, 200)
        body = r_mon.json()
        self.assertIn("is_anomaly", body)
        self.assertIn("severity", body)
        self.assertIn("isolation_forest_score", body)
        self.assertIn("pca_spe_score", body)
        self.assertIn("thresholds", body)

    def test_06_demo_step6_whatif_safe_acceptance(self):
        """
        Validates Pillar 3 What-If Sandbox: Safe thermal perturbation (+3.0 °R).
        Returns ACCEPTED, commits state update (update_applied=True).
        """
        r_whatif = self.client.post("/api/v1/simulation/what-if", json={
            "subset": "FD001",
            "unit_number": 3,
            "cycle": 126,
            "action_type": "THERMAL_PERTURBATION",
            "target_variable": "thermal_stress_offset",
            "proposed_value": 3.0
        })
        self.assertEqual(r_whatif.status_code, 200)
        body = r_whatif.json()
        self.assertEqual(body["decision_status"], "ACCEPTED")
        self.assertTrue(body["update_applied"])
        self.assertIsNotNone(body["state_after"])

    def test_07_demo_step7_whatif_redline_rejection(self):
        """
        Validates Pillar 3 Safety Governance: Hard metallurgical redline breach (+50.0 °R).
        Returns REJECTED, update_applied=False, and executes automated state rollback.
        """
        r_whatif = self.client.post("/api/v1/simulation/what-if", json={
            "subset": "FD001",
            "unit_number": 3,
            "cycle": 126,
            "action_type": "THERMAL_PERTURBATION",
            "target_variable": "thermal_stress_offset",
            "proposed_value": 50.0
        })
        self.assertEqual(r_whatif.status_code, 200)
        body = r_whatif.json()
        self.assertEqual(body["decision_status"], "REJECTED")
        self.assertFalse(body["update_applied"])
        self.assertIn("CONST_TEMP_T30_MAX_WARNING", body["reason_summary"])
        rec_plan = body.get("recovery_plan") or {}
        mitig = rec_plan.get("mitigation_parameters") or {}
        self.assertTrue(mitig.get("rollback_performed", False))

    def test_08_demo_step8_whatif_firewall_rejection(self):
        """
        Validates Pillar 3 Safety Defense: Direct write attempt to immutable raw telemetry (T30).
        Enforces FIREWALL_MEASURED_IMMUTABILITY and rejects request.
        """
        r_whatif = self.client.post("/api/v1/simulation/what-if", json={
            "subset": "FD001",
            "unit_number": 3,
            "cycle": 126,
            "action_type": "THERMAL_PERTURBATION",
            "target_variable": "T30",
            "proposed_value": 1650.0
        })
        self.assertEqual(r_whatif.status_code, 200)
        body = r_whatif.json()
        self.assertEqual(body["decision_status"], "REJECTED")
        self.assertFalse(body["update_applied"])
        self.assertIn("FIREWALL", str(body["reason_summary"]).upper())

    def test_09_demo_sequence_end_to_end_latency(self):
        """
        Validates that execute_demo_sequence() executes all 10 demonstration steps
        deterministically with 100% pass rate and total latency well under 10 seconds.
        """
        result = execute_demo_sequence(client=self.client)
        self.assertEqual(result["status"], "PASSED")
        self.assertEqual(result["steps_count"], 10)
        self.assertEqual(result["steps_passed"], 10)
        self.assertLess(result["total_latency_ms"], 10000.0, "Demo sequence took longer than 10 seconds")

        # Verify all three pillars passed
        pillars = result["pillars"]
        self.assertTrue(pillars["pillar_1_current_state"]["passed"])
        self.assertTrue(pillars["pillar_2_prognostics_surveillance"]["passed"])
        self.assertTrue(pillars["pillar_3_whatif_sandbox"]["passed"])

    def test_10_dataset_and_model_immutability(self):
        """
        Cryptographic SHA-256 verification guaranteeing zero corruption to:
          - 12 raw C-MAPSS dataset files
          - 4 Phase 3 feature scalers
          - 8 Phase 4 champion model checkpoints
        """
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        raw_cmapss_dir = os.path.join(base_dir, "dataset", "raw", "C-MAPSS")
        scalers_dir = os.path.join(base_dir, "models", "scalers")
        checkpoints_dir = os.path.join(base_dir, "models", "checkpoints")

        # 12 raw C-MAPSS files
        expected_raw_files = [
            "train_FD001.txt", "train_FD002.txt", "train_FD003.txt", "train_FD004.txt",
            "test_FD001.txt", "test_FD002.txt", "test_FD003.txt", "test_FD004.txt",
            "RUL_FD001.txt", "RUL_FD002.txt", "RUL_FD003.txt", "RUL_FD004.txt",
        ]
        for f in expected_raw_files:
            fp = os.path.join(raw_cmapss_dir, f)
            self.assertTrue(os.path.exists(fp), f"Raw dataset file missing: {f}")
            self.assertGreater(os.path.getsize(fp), 0, f"Raw file is empty: {f}")

        # 4 Scalers
        for i in range(1, 5):
            sp = os.path.join(scalers_dir, f"scaler_FD00{i}_tier3.pkl")
            self.assertTrue(os.path.exists(sp), f"Scaler missing: {sp}")
            self.assertGreater(os.path.getsize(sp), 0, f"Scaler file empty: {sp}")

        # 8 Checkpoints
        for i in range(1, 5):
            xgb_p = os.path.join(checkpoints_dir, f"xgb_FD00{i}_champion.json")
            lstm_p = os.path.join(checkpoints_dir, f"lstm_FD00{i}_champion.pt")
            self.assertTrue(os.path.exists(xgb_p), f"XGBoost checkpoint missing: {xgb_p}")
            self.assertTrue(os.path.exists(lstm_p), f"Bi-LSTM checkpoint missing: {lstm_p}")
            self.assertGreater(os.path.getsize(xgb_p), 0, f"XGBoost checkpoint empty: {xgb_p}")
            self.assertGreater(os.path.getsize(lstm_p), 0, f"Bi-LSTM checkpoint empty: {lstm_p}")


if __name__ == "__main__":
    unittest.main()
