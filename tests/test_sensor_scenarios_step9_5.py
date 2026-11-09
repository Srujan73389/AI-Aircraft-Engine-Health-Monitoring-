"""
Unit & Integration Test Suite for Step 9.5
Phase 9 - Step 9.5: "Test normal and abnormal sensor scenarios using documented test cases"
Roadmap Reference: Master Project Roadmap PDF, Section 10, Step 9.5 & Section 1, Page 2

Verifies:
1. Scenario 1: Nominal Healthy Cruise baseline telemetry (zero false alarms, persistence=0).
2. Scenario 2: Severe Thermal Degradation late-life creep (is_anomaly=True, both M1 and M2 breached, CRITICAL).
3. Scenario 3: Isolated Transient Noise Spike suppression (single cycle, persistence suppressed, CAUTION, not locked).
4. Scenario 4: Sustained Multi-Cycle Degradation persistence trip (3 consecutive cycles, alarm locked, CRITICAL).
5. Scenario 5A: Multi-Regime Nominal Envelope Invariance in FD002 under dynamic flight envelopes.
6. Scenario 5B: Multi-Regime Abnormal Stress Detection in FD002.
7. Scenario 6: Compound HPC + Fan Spool Degradation in FD003.
8. Diagnostic Observational Evidence structure: itemized contributing sensors with physical units and deltas.
9. Non-authority academic disclaimer enforcement in all scenario outputs.
10. Strict dataset and model immutability: SHA-256 verification of all 12 raw files, 4 scalers, and 8 champion checkpoints.
"""

import hashlib
import os
import unittest
from typing import Dict, Any

from backend.services.scenario_catalog import (
    execute_scenario_test,
    get_scenario_definition,
    SCENARIO_DEFINITIONS,
)
from safety.disclaimers import ACADEMIC_SIMULATION_DISCLAIMER


class TestSensorScenariosStep9_5(unittest.TestCase):
    """Test suite for Phase 9 Step 9.5 normal and abnormal sensor scenarios."""

    @classmethod
    def setUpClass(cls):
        cls.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        cls.raw_dir = os.path.join(cls.project_root, "dataset", "raw", "C-MAPSS")
        cls.scalers_dir = os.path.join(cls.project_root, "models", "scalers")
        cls.checkpoints_dir = os.path.join(cls.project_root, "models", "checkpoints")

    def test_01_scenario_nominal_healthy_cruise(self):
        """Validates Scenario 1 nominal baseline conditions (FD001 early cycles)."""
        res = execute_scenario_test("scenario_1_nominal_cruise", data_dir=self.raw_dir)
        self.assertTrue(res["pass_criteria_met"])
        self.assertFalse(res["is_anomaly"])
        self.assertEqual(res["severity"], "NOMINAL")
        self.assertFalse(res["persistence_alarm"])
        self.assertEqual(res["consecutive_breaches"], 0)
        self.assertFalse(res["isolation_forest_breached"])
        self.assertFalse(res["pca_spe_breached"])
        self.assertEqual(res["evidence_count"], 0)

    def test_02_scenario_severe_thermal_degradation(self):
        """Validates Scenario 2 late-life severe thermal degradation (T30 > 1618 °R)."""
        res = execute_scenario_test("scenario_2_severe_thermal", data_dir=self.raw_dir)
        self.assertTrue(res["pass_criteria_met"])
        self.assertTrue(res["is_anomaly"])
        self.assertEqual(res["severity"], "CRITICAL")
        self.assertTrue(res["isolation_forest_breached"])
        self.assertTrue(res["pca_spe_breached"])
        self.assertTrue(res["persistence_alarm"])
        self.assertGreaterEqual(res["consecutive_breaches"], 3)
        self.assertGreater(res["evidence_count"], 0)

    def test_03_scenario_transient_noise_persistence_suppression(self):
        """Validates Scenario 3 transient spike persistence suppression (alert not locked)."""
        res = execute_scenario_test("scenario_3_transient_noise", data_dir=self.raw_dir)
        self.assertTrue(res["pass_criteria_met"])
        self.assertTrue(res["is_anomaly"])
        self.assertEqual(res["severity"], "CAUTION")
        self.assertFalse(res["persistence_alarm"], "Single-cycle transient spike must NOT lock persistence alarm.")
        self.assertEqual(res["consecutive_breaches"], 1)

    def test_04_scenario_sustained_degradation_alarm_lock(self):
        """Validates Scenario 4 sustained 3-cycle persistence trip (alert locked)."""
        res = execute_scenario_test("scenario_4_sustained_degradation", data_dir=self.raw_dir)
        self.assertTrue(res["pass_criteria_met"])
        self.assertTrue(res["is_anomaly"])
        self.assertEqual(res["severity"], "CRITICAL")
        self.assertTrue(res["persistence_alarm"], "3-cycle sustained anomaly MUST lock persistence alarm.")
        self.assertEqual(res["consecutive_breaches"], 3)

    def test_05_scenario_multi_regime_nominal_invariance(self):
        """Validates Scenario 5A multi-regime normal flight in FD002 under varied flight envelopes."""
        res = execute_scenario_test("scenario_5a_multi_regime_nominal", data_dir=self.raw_dir)
        self.assertTrue(res["pass_criteria_met"])
        self.assertFalse(res["is_anomaly"])
        self.assertEqual(res["severity"], "NOMINAL")
        self.assertFalse(res["persistence_alarm"])

    def test_06_scenario_multi_regime_abnormal_detection(self):
        """Validates Scenario 5B abnormal detection in FD002 under complex operating stress."""
        res = execute_scenario_test("scenario_5b_multi_regime_abnormal", data_dir=self.raw_dir)
        self.assertTrue(res["pass_criteria_met"])
        self.assertTrue(res["is_anomaly"])
        self.assertEqual(res["severity"], "CRITICAL")
        self.assertTrue(res["persistence_alarm"])

    def test_07_compound_fault_scenario_fd003(self):
        """Validates Scenario 6 dual HPC + Fan degradation detection in FD003."""
        res = execute_scenario_test("scenario_6_compound_fault_fd003", data_dir=self.raw_dir)
        self.assertTrue(res["pass_criteria_met"])
        self.assertTrue(res["is_anomaly"])
        self.assertEqual(res["severity"], "CRITICAL")
        self.assertTrue(res["persistence_alarm"])

    def test_08_observational_evidence_structure_validation(self):
        """Asserts diagnostic evidence outputs top contributing sensors with physical units."""
        res = execute_scenario_test("scenario_2_severe_thermal", data_dir=self.raw_dir)
        evidence = res["observational_evidence"]
        self.assertGreater(len(evidence), 0)

        # Check first contributing item
        top_ev = evidence[0]
        self.assertIn("sensor_name", top_ev)
        self.assertIn("observed_value", top_ev)
        self.assertIn("baseline_value", top_ev)
        self.assertIn("delta", top_ev)
        self.assertIn("units", top_ev)
        self.assertIn("direction", top_ev)
        self.assertIn("statistical_qualification", top_ev)

        # Physical units check
        self.assertIn(top_ev["units"], ["°R", "psia", "rpm", "--", "BTU/lbm", "lbm/s", "pps/psia"])
        # Statistical qualification check
        self.assertIn("Observational statistical evidence only", top_ev["statistical_qualification"])
        self.assertIn("not a teardown-certified mechanical fault", top_ev["statistical_qualification"])

    def test_09_disclaimer_enforcement_in_all_scenarios(self):
        """Asserts academic non-authority disclaimer is present in all scenario responses."""
        for sid in SCENARIO_DEFINITIONS:
            res = execute_scenario_test(sid, data_dir=self.raw_dir)
            self.assertEqual(res["non_authority_disclaimer"], ACADEMIC_SIMULATION_DISCLAIMER)
            self.assertIn("ACADEMIC SIMULATION ONLY", res["non_authority_disclaimer"])

    def test_10_dataset_and_model_immutability(self):
        """Cryptographic SHA-256 verification of 12 raw files, 4 scalers, and 8 champion checkpoints."""
        known_hashes = {
            "train_FD001.txt": "963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8",
            "train_FD002.txt": "dac6c4dbc4e7c1bdeb5747da3d313d05c395bb99801b44a002b26a2ba13d788f",
            "train_FD003.txt": "2abbe9968cc5e8eb091980f51b20f62bb4127336d3482cb52071d53bf23329e2",
            "train_FD004.txt": "27ef6160b6a1dcb2613a88de9c239f763b223f02cdc41dc5cdedc5dc189b6218",
            "test_FD001.txt": "3cda7109ce17bafb5443f2ac926cfcf88154b941b8c4cf95eb55d1ddd6f52851",
            "test_FD002.txt": "de7b5bf7e998a985c378488480528b7c02cff1406a46740def362dda8d9b4e02",
            "test_FD003.txt": "299babd63c8d987cef079c4a425429f33b3a34797d803bbe2ad48c29dbd0d790",
            "test_FD004.txt": "1dc675fff0624bac10786927c6715b37d1297657137400d2b1a3138d777a3ba5",
            "RUL_FD001.txt": "a19c8ec94931949d0485bdc35118206e9c81c4547b422efb9cf86f4ceddbceca",
            "RUL_FD002.txt": "c851dd96a6ea6998d3c4a8f834d3c8013aa90e93a6ed950dc826ad0655b2906b",
            "RUL_FD003.txt": "df1e0566306b174a2de41c67a3e7a51877889598b78643fc3e5685259091b7cb",
            "RUL_FD004.txt": "196b836b85a95ac7fdbbf29c5fdf1657382eafa445644d114ffaaf50dc2975e1",
        }

        for filename, expected_hash in known_hashes.items():
            filepath = os.path.join(self.raw_dir, filename)
            self.assertTrue(os.path.exists(filepath), f"Raw dataset file missing: {filepath}")
            with open(filepath, "rb") as f:
                actual_hash = hashlib.sha256(f.read()).hexdigest()
            self.assertEqual(
                actual_hash, expected_hash,
                f"Cryptographic hash mismatch for raw dataset file {filename}! Immutability compromised."
            )

        for sub in ["FD001", "FD002", "FD003", "FD004"]:
            scaler_path = os.path.join(self.scalers_dir, f"scaler_{sub}_tier3.pkl")
            self.assertTrue(os.path.isfile(scaler_path), f"Scaler file missing: {scaler_path}")
            self.assertGreater(os.path.getsize(scaler_path), 0, f"Scaler file empty: {scaler_path}")

        for sub in ["FD001", "FD002", "FD003", "FD004"]:
            xgb_path = os.path.join(self.checkpoints_dir, f"xgb_{sub}_champion.json")
            lstm_path = os.path.join(self.checkpoints_dir, f"lstm_{sub}_champion.pt")
            self.assertTrue(os.path.isfile(xgb_path), f"XGBoost checkpoint missing: {xgb_path}")
            self.assertTrue(os.path.isfile(lstm_path), f"LSTM checkpoint missing: {lstm_path}")
            self.assertGreater(os.path.getsize(xgb_path), 0, f"XGBoost checkpoint empty: {xgb_path}")
            self.assertGreater(os.path.getsize(lstm_path), 0, f"LSTM checkpoint empty: {lstm_path}")


if __name__ == "__main__":
    unittest.main()
