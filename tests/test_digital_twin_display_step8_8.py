"""
Unit and Integration Test Suite for Phase 8 Step 8.8: Digital Twin State and Scenario Output Display
Verifies:
- DigitalTwinSimulationPanel component file existence and clean export
- Turbofan cutaway schematic stations rendering (Station 2, 1.5, 2.4, 3, 4, 4.5, 5)
- Visual provenance badge distinction (MEASURED, DERIVED, PREDICTED, SIMULATED)
- Strict restriction to Phase 5 supported scenarios (Thermal Perturbation, Sensor Bias Injection)
- Baseline vs. simulated comparison matrix with separate labeling
- Simulated values explicitly tagged and isolated from measured telemetry
- Real backend POST /api/v1/simulation/what-if integration via FastAPI test adapter
- Closed-loop safety governor decision badges (ACCEPTED, REQUIRES_REVIEW, REJECTED)
- Root App.jsx integration of DigitalTwinSimulationPanel
- Cryptographic SHA-256 zero-tampering verification of datasets, scalers, and checkpoints

Roadmap Reference: Phase 8 - Step 8.8:
"Display Digital Twin state and scenario outputs, distinguishing simulated values from dataset values"
"""

import hashlib
import json
import os
import unittest

from backend.main import app
from tests.test_backend_independent_endpoints_step8_3 import FastAPITestClientAdapter


class TestDigitalTwinDisplayStep8_8(unittest.TestCase):
    """
    Test suite for Phase 8 Step 8.8: Digital Twin State & Scenario Display.
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

    def test_01_digital_twin_panel_component_exists(self):
        """
        Asserts DigitalTwinSimulationPanel component file exists and exports cleanly.
        """
        panel_path = os.path.join(self.src_dir, "components", "dashboard", "DigitalTwinSimulationPanel.jsx")
        self.assertTrue(os.path.exists(panel_path), f"Missing component file: {panel_path}")

        with open(panel_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("export const DigitalTwinSimulationPanel", content)
        self.assertIn("export default DigitalTwinSimulationPanel", content)

    def test_02_turbofan_schematic_stations_rendered(self):
        """
        Verifies all 7 gas-path aerodynamic stations are mapped and rendered in the schematic.
        """
        panel_path = os.path.join(self.src_dir, "components", "dashboard", "DigitalTwinSimulationPanel.jsx")
        with open(panel_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("Station 2 - Fan Inlet", content)
        self.assertIn("Station 1.5 - Bypass Duct", content)
        self.assertIn("Station 2.4 - LPC Outlet", content)
        self.assertIn("Station 3 - HPC Outlet", content)
        self.assertIn("Station 4 - Combustor / Burner", content)
        self.assertIn("Station 4.5 - HPT Core Spool", content)
        self.assertIn("Station 5 - LPT Outlet / EGT", content)
        self.assertIn("<svg", content)

    def test_03_provenance_badges_distinction(self):
        """
        Verifies 4-tier visual provenance badges (MEASURED, DERIVED, PREDICTED, SIMULATED).
        """
        panel_path = os.path.join(self.src_dir, "components", "dashboard", "DigitalTwinSimulationPanel.jsx")
        with open(panel_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("MEASURED / DATASET", content)
        self.assertIn("PHYSICAL / DERIVED", content)
        self.assertIn("MODEL / PREDICTED", content)
        self.assertIn("SIMULATED / SCENARIO", content)

    def test_04_what_if_controls_supported_scenarios(self):
        """
        Verifies only Phase 5 supported scenarios (Thermal Perturbation, Sensor Bias Injection)
        are present, and Accelerated Aging is strictly omitted.
        """
        panel_path = os.path.join(self.src_dir, "components", "dashboard", "DigitalTwinSimulationPanel.jsx")
        with open(panel_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("THERMAL_PERTURBATION", content)
        self.assertIn("Thermal Stress Perturbation", content)
        self.assertIn("SENSOR_BIAS_INJECTION", content)
        self.assertIn("Sensor Bias Injection", content)
        self.assertNotIn("ACCELERATED_AGING", content)
        self.assertNotIn("Accelerated Aging", content)

    def test_05_baseline_vs_simulated_comparison_matrix(self):
        """
        Verifies comparison matrix displaying baseline vs. simulated candidate values and deltas.
        """
        panel_path = os.path.join(self.src_dir, "components", "dashboard", "DigitalTwinSimulationPanel.jsx")
        with open(panel_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("Baseline Measured Value", content)
        self.assertIn("Simulated Candidate Value", content)
        self.assertIn("Delta (Δ)", content)
        self.assertIn("Safety Verification Envelope", content)

    def test_06_simulated_values_labeled_separately(self):
        """
        Confirms simulated scenario values are explicitly labeled as SIMULATED / SCENARIO
        and accompanied by the academic non-flight operations notice.
        """
        panel_path = os.path.join(self.src_dir, "components", "dashboard", "DigitalTwinSimulationPanel.jsx")
        with open(panel_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("Digital Twin Academic Non-Flight Operations Notice", content)
        self.assertIn("synthetic candidate states", content)
        self.assertIn("strictly isolated from recorded NASA C-MAPSS dataset baselines", content)

    def test_07_backend_simulation_what_if_integration(self):
        """
        Verifies backend POST /api/v1/simulation/what-if endpoint integration with closed-loop governor.
        """
        payload = {
            "subset": "FD001",
            "unit_number": 1,
            "cycle": 15,
            "action_type": "THERMAL_PERTURBATION",
            "target_variable": "thermal_stress_offset",
            "proposed_value": 4.0,
            "operator_approval": True,
        }
        resp = self.client.post("/api/v1/simulation/what-if", json=payload)
        self.assertEqual(resp.status_code, 200)

        data = resp.json()
        self.assertIn("action_id", data)
        self.assertIn("decision_status", data)
        self.assertIn(data["decision_status"], ["ACCEPTED", "REQUIRES_REVIEW", "REJECTED"])
        self.assertIn("state_before", data)
        self.assertIn("state_after", data)
        self.assertIn("reason_summary", data)
        self.assertIn("disclaimer", data)

    def test_08_safety_status_badge_in_comparison(self):
        """
        Verifies governor decision badges (ACCEPTED, REQUIRES_REVIEW, REJECTED) and recovery plan rendering.
        """
        panel_path = os.path.join(self.src_dir, "components", "dashboard", "DigitalTwinSimulationPanel.jsx")
        with open(panel_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("GOVERNOR DECISION:", content)
        self.assertIn("Safe Recovery Advisory Plan Generated:", content)
        self.assertIn("operatorApproval", content)

    def test_09_app_integration_digital_twin_panel(self):
        """
        Confirms App.jsx imports and mounts DigitalTwinSimulationPanel.
        """
        app_path = os.path.join(self.src_dir, "App.jsx")
        with open(app_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("import DigitalTwinSimulationPanel from './components/dashboard/DigitalTwinSimulationPanel';", content)
        self.assertIn("<DigitalTwinSimulationPanel />", content)

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
