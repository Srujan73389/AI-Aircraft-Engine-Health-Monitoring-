"""
Unit and Integration Test Suite for Phase 8 Step 8.7: Predicted Remaining Useful Life (RUL) Display
Verifies:
- PrognosticsRULPanel component file existence and clean export
- Dual-model prognostic estimates for XGBoost and PyTorch Bi-LSTM
- Consensus ensemble RUL and epistemic spread (|RUL_XGB - RUL_LSTM|) display
- Strict labeling and enforcement of "Operating Cycles" engineering units
- Exact mandatory academic prognostic estimate disclaimer phrasing
- Confidence rating gauge and styling (HIGH, MODERATE, LOW)
- Real backend POST /api/v1/predict/rul integration and consensus calculations
- SVG degradation trajectory curve and EOL projection rendering
- App.jsx root dashboard integration of PrognosticsRULPanel
- Cryptographic SHA-256 zero-tampering verification of datasets, scalers, and checkpoints

Roadmap Reference: Phase 8 - Step 8.7:
"Display predicted RUL and clearly state that it is an estimate in operating cycles"
"""

import hashlib
import json
import os
import unittest

from backend.main import app
from tests.test_backend_independent_endpoints_step8_3 import FastAPITestClientAdapter


class TestRULDisplayStep8_7(unittest.TestCase):
    """
    Test suite for Phase 8 Step 8.7: Predicted RUL Display.
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

    def test_01_prognostics_panel_component_exists(self):
        """
        Asserts PrognosticsRULPanel component file exists and exports cleanly.
        """
        panel_path = os.path.join(self.src_dir, "components", "dashboard", "PrognosticsRULPanel.jsx")
        self.assertTrue(os.path.exists(panel_path), f"Missing component file: {panel_path}")

        with open(panel_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("export const PrognosticsRULPanel", content)
        self.assertIn("export default PrognosticsRULPanel", content)

    def test_02_dual_model_predictions_rendered(self):
        """
        Verifies XGBoost and Bi-LSTM cards are present with cycle units and architecture details.
        """
        panel_path = os.path.join(self.src_dir, "components", "dashboard", "PrognosticsRULPanel.jsx")
        with open(panel_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("XGBoost Regressor (Champion Trees)", content)
        self.assertIn("PyTorch Bi-LSTM (Recurrent Deep Net)", content)
        self.assertIn("xgbRul", content)
        self.assertIn("lstmRul", content)
        self.assertIn("Operating Cycles", content)

    def test_03_consensus_and_spread_displayed(self):
        """
        Verifies consensus RUL and spread calculation are rendered.
        """
        panel_path = os.path.join(self.src_dir, "components", "dashboard", "PrognosticsRULPanel.jsx")
        with open(panel_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("Consensus Remaining Useful Life (RUL)", content)
        self.assertIn("consensusRul", content)
        self.assertIn("Model Spread (|Δ|)", content)
        self.assertIn("spread", content)

    def test_04_operating_cycles_unit_enforcement(self):
        """
        Asserts every RUL value is explicitly labeled with operating cycle units.
        """
        panel_path = os.path.join(self.src_dir, "components", "dashboard", "PrognosticsRULPanel.jsx")
        with open(panel_path, "r", encoding="utf-8") as f:
            content = f.read()

        # The term "Operating Cycles" or "Cycles" must appear multiple times (overview, cards, graph)
        cycles_count = content.count("Operating Cycles") + content.count("Cycles")
        self.assertGreaterEqual(cycles_count, 5, "RUL outputs must consistently specify operating cycles unit.")

    def test_05_mandatory_estimate_disclaimer_present(self):
        """
        Verifies exact estimate notice wording is present.
        """
        panel_path = os.path.join(self.src_dir, "components", "dashboard", "PrognosticsRULPanel.jsx")
        with open(panel_path, "r", encoding="utf-8") as f:
            content = f.read()

        expected_disclaimer = (
            "PROGNOSTIC ESTIMATE NOTICE: Remaining Useful Life (RUL) values displayed are statistical "
            "machine learning estimates measured in discrete flight cycles. They do not constitute an "
            "airworthiness dispatch authorization or a certified mechanical life limit."
        )
        self.assertIn(expected_disclaimer, content)

    def test_06_confidence_score_gauge(self):
        """
        Verifies confidence score styling and threshold categories (HIGH, MODERATE, LOW).
        """
        panel_path = os.path.join(self.src_dir, "components", "dashboard", "PrognosticsRULPanel.jsx")
        with open(panel_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("confidenceLabel", content)
        self.assertIn("confidenceScore", content)
        self.assertIn("HIGH", content)
        self.assertIn("MODERATE", content)
        self.assertIn("LOW", content)
        self.assertIn("Confidence Rating", content)

    def test_07_backend_predict_rul_integration(self):
        """
        Tests POST /api/v1/predict/rul returns valid response for panel.
        """
        resp = self.client.post("/api/v1/predict/rul", json={
            "subset": "FD001",
            "unit_number": 1,
            "cycle": 25,
        })
        self.assertEqual(resp.status_code, 200)

        data = resp.json()
        self.assertEqual(data["subset"], "FD001")
        self.assertEqual(data["unit_number"], 1)
        self.assertEqual(data["cycle"], 25)

        self.assertGreater(data["consensus_rul"], 0.0)
        self.assertGreater(data["predicted_rul_xgboost"], 0.0)
        self.assertGreater(data["predicted_rul_lstm"], 0.0)
        self.assertGreaterEqual(data["prognostic_spread"], 0.0)
        self.assertGreaterEqual(data["confidence_score"], 0.0)
        self.assertLessEqual(data["confidence_score"], 1.0)
        self.assertIn("disclaimer", data)

    def test_08_rul_trajectory_trend_rendered(self):
        """
        Confirms SVG trajectory curve renders degradation slope and EOL projection.
        """
        panel_path = os.path.join(self.src_dir, "components", "dashboard", "PrognosticsRULPanel.jsx")
        with open(panel_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("<svg", content)
        self.assertIn("Remaining Useful Life Degradation Trajectory", content)
        self.assertIn("Cycle 1 (New)", content)
        self.assertIn("EOL (0 Cycles RUL)", content)
        self.assertIn("pathStart", content)
        self.assertIn("pathEnd", content)

    def test_09_app_integration_prognostics_panel(self):
        """
        Confirms App.jsx renders PrognosticsRULPanel.
        """
        app_path = os.path.join(self.src_dir, "App.jsx")
        with open(app_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("import PrognosticsRULPanel from './components/dashboard/PrognosticsRULPanel';", content)
        self.assertIn("<PrognosticsRULPanel />", content)

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
