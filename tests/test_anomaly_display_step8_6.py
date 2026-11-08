"""
Unit and Integration Test Suite for Phase 8 Step 8.6: Anomaly Flags & Diagnostic Evidence Display
Verifies:
- AnomalyDiagnosisPanel component scaffolding and clean export
- Dual detector cards: Regime-Aware Isolation Forest (M1) and PCA Squared Prediction Error (M2)
- Threshold (tau) display and evaluation for M1 and M2
- Severity badge variants (NOMINAL, CAUTION, CRITICAL)
- Observational diagnostic evidence structure with station locations and drift deltas
- Strict mandatory academic non-fault claim disclaimer
- 3-cycle persistence alarm tracker logic and display
- Integration with POST /api/v1/monitoring/anomaly-check endpoint
- App.jsx root dashboard integration of AnomalyDiagnosisPanel
- Cryptographic SHA-256 zero-tampering verification of datasets, scalers, and checkpoints

Roadmap Reference: Phase 8 - Step 8.6:
"Display anomaly flags and diagnosis evidence separately from confirmed fault claims"
"""

import hashlib
import json
import os
import unittest

from backend.main import app
from tests.test_backend_independent_endpoints_step8_3 import FastAPITestClientAdapter


class TestAnomalyDisplayStep8_6(unittest.TestCase):
    """
    Test suite for Phase 8 Step 8.6: Anomaly Flags & Diagnostic Evidence Display.
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

    def test_01_anomaly_panel_component_exists(self):
        """
        Asserts AnomalyDiagnosisPanel component file exists and exports cleanly.
        """
        panel_path = os.path.join(self.src_dir, "components", "dashboard", "AnomalyDiagnosisPanel.jsx")
        self.assertTrue(os.path.exists(panel_path), f"Missing panel component: {panel_path}")

        with open(panel_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("export const AnomalyDiagnosisPanel", content)
        self.assertIn("export default AnomalyDiagnosisPanel", content)

    def test_02_dual_model_cards_rendered(self):
        """
        Verifies Isolation Forest and PCA-SPE cards are present in markup.
        """
        panel_path = os.path.join(self.src_dir, "components", "dashboard", "AnomalyDiagnosisPanel.jsx")
        with open(panel_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("Regime-Aware Isolation Forest (M1)", content)
        self.assertIn("PCA Squared Prediction Error (M2)", content)
        self.assertIn("Decision Score", content)
        self.assertIn("Residual Energy (SPE)", content)

    def test_03_calibrated_thresholds_displayed(self):
        """
        Verifies threshold (tau) display for both M1 and M2.
        """
        panel_path = os.path.join(self.src_dir, "components", "dashboard", "AnomalyDiagnosisPanel.jsx")
        with open(panel_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("Threshold (τ_IF)", content)
        self.assertIn("Threshold (τ_PCA)", content)
        self.assertIn("tauIf", content)
        self.assertIn("tauPca", content)

    def test_04_severity_badge_variants(self):
        """
        Verifies NOMINAL, CAUTION, and CRITICAL styling and classes are defined.
        """
        panel_path = os.path.join(self.src_dir, "components", "dashboard", "AnomalyDiagnosisPanel.jsx")
        with open(panel_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check all 3 status badge color variants
        self.assertIn("bg-emerald-500/10 text-emerald-400 border-emerald-500/20", content)
        self.assertIn("bg-amber-500/10 text-amber-400 border-amber-500/20", content)
        self.assertIn("bg-rose-500/10 text-rose-400 border-rose-500/20", content)
        self.assertIn("STATUS: {severity}", content)

    def test_05_diagnostic_evidence_structure(self):
        """
        Verifies observational evidence is rendered with sensor stations and drift deltas.
        """
        panel_path = os.path.join(self.src_dir, "components", "dashboard", "AnomalyDiagnosisPanel.jsx")
        with open(panel_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("Observational Diagnostic Evidence (Sensor Departures)", content)
        self.assertIn("item.station", content)
        self.assertIn("item.delta", content)
        self.assertIn("item.pctDelta", content)
        self.assertIn("DRIFT", content)

    def test_06_non_fault_claim_disclaimer_present(self):
        """
        Asserts mandatory academic notice separating anomaly flags from confirmed physical faults.
        """
        panel_path = os.path.join(self.src_dir, "components", "dashboard", "AnomalyDiagnosisPanel.jsx")
        with open(panel_path, "r", encoding="utf-8") as f:
            content = f.read()

        expected_notice = (
            "OBSERVATIONAL ANOMALY EVIDENCE ONLY • NOT A CONFIRMED MECHANICAL FAULT CLAIM. "
            "Machine learning models detect statistical telemetry departures; certified fault "
            "diagnosis requires physical borescope inspection or teardown verification."
        )
        self.assertIn(expected_notice, content)

    def test_07_persistence_tracker_logic(self):
        """
        Verifies 3-cycle persistence alarm display logic.
        """
        panel_path = os.path.join(self.src_dir, "components", "dashboard", "AnomalyDiagnosisPanel.jsx")
        with open(panel_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("persistenceAlarm", content)
        self.assertIn("consecutiveBreaches", content)
        self.assertIn("PERSISTENCE ALARM: TRIGGERED (≥3 CYCLES)", content)
        self.assertIn("PERSISTENCE:", content)

    def test_08_backend_anomaly_check_integration(self):
        """
        Tests POST /api/v1/monitoring/anomaly-check returns valid response for panel.
        """
        # Test Case 1: Nominal
        res_nom = self.client.post("/api/v1/monitoring/anomaly-check", json={
            "subset": "FD001",
            "anomaly_scores": {"isolation_forest": -0.15, "pca_spe": 0.5},
        })
        self.assertEqual(res_nom.status_code, 200)
        data_nom = res_nom.json()
        self.assertEqual(data_nom["severity"], "NOMINAL")
        self.assertFalse(data_nom["is_anomaly"])
        self.assertIn("thresholds", data_nom)

        # Test Case 2: Critical
        tau_if = data_nom["thresholds"]["isolation_forest"]
        tau_pca = data_nom["thresholds"]["pca_spe"]
        res_crit = self.client.post("/api/v1/monitoring/anomaly-check", json={
            "subset": "FD001",
            "cycle": 10,
            "anomaly_scores": {"isolation_forest": tau_if + 0.2, "pca_spe": tau_pca + 2.0},
        })
        self.assertEqual(res_crit.status_code, 200)
        data_crit = res_crit.json()
        self.assertEqual(data_crit["severity"], "CRITICAL")
        self.assertTrue(data_crit["is_anomaly"])
        self.assertTrue(data_crit["isolation_forest_breached"])
        self.assertTrue(data_crit["pca_spe_breached"])
        self.assertTrue(data_crit["persistence_alarm"])

    def test_09_app_integration_anomaly_panel(self):
        """
        Confirms App.jsx renders AnomalyDiagnosisPanel.
        """
        app_path = os.path.join(self.src_dir, "App.jsx")
        with open(app_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("import AnomalyDiagnosisPanel from './components/dashboard/AnomalyDiagnosisPanel';", content)
        self.assertIn("<AnomalyDiagnosisPanel />", content)

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
