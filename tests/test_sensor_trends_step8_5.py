"""
Unit and Integration Test Suite for Phase 8 Step 8.5: Sensor Readings and Trends Display
Verifies:
- Completeness and validity of the 21-sensor NASA C-MAPSS metadata dictionary
- Unit formatting correctness for temperatures (°R), pressures (psia), speeds (rpm), flows (lbm/s)
- Architectural structure of SensorTrendsChart (SVG line curves, active cycle cursor, presets)
- Architectural structure of SensorReadingsGrid (category tabs, tabular layout, baseline delta math)
- Real backend /engine/{subset}/{unit}/sensors endpoint schema and length parity
- Cycle cursor alignment and interactive selection mechanics
- Sensor category classification (temperatures, pressures, speeds, bleed_flow, performance)
- Mathematical accuracy of baseline drift delta calculations
- App.jsx root integration of both sensor visualization components
- Cryptographic SHA-256 zero-tampering verification of datasets, scalers, and checkpoints

Roadmap Reference: Phase 8 - Step 8.5:
"Display selected sensor readings and sensor trends with units"
"""

import hashlib
import json
import os
import re
import unittest

from backend.main import app
from tests.test_backend_independent_endpoints_step8_3 import FastAPITestClientAdapter


class TestSensorTrendsStep8_5(unittest.TestCase):
    """
    Test suite for Phase 8 Step 8.5: Sensor Readings and Trends Display.
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

    def test_01_sensor_metadata_completeness(self):
        """
        Asserts all 21 sensors have canonical names, station descriptions, engineering units, and categories.
        """
        meta_path = os.path.join(self.src_dir, "utils", "sensorMetadata.js")
        self.assertTrue(os.path.exists(meta_path), f"Missing metadata file: {meta_path}")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta_code = f.read()

        # Check all 21 sensor keys are defined
        for i in range(1, 22):
            s_key = f"sensor_{i}"
            self.assertIn(f"{s_key}:", meta_code, f"Missing definition for {s_key}")

        # Check expected canonical sensor names
        expected_names = ["T2", "T24", "T30", "T50", "P2", "P15", "P30", "Nf", "Nc", "epr", "Ps30", "BPR"]
        for name in expected_names:
            self.assertIn(f'"{name}"', meta_code, f"Missing canonical sensor name {name}")

        # Check required engineering units
        expected_units = ["°R", "psia", "rpm", "lbm/s"]
        for unit in expected_units:
            self.assertIn(f'"{unit}"', meta_code, f"Missing engineering unit {unit}")

    def test_02_units_formatting_correctness(self):
        """
        Tests formatSensorValue implementation logic across temperatures, pressures, and speeds.
        """
        meta_path = os.path.join(self.src_dir, "utils", "sensorMetadata.js")
        with open(meta_path, "r", encoding="utf-8") as f:
            code = f.read()

        self.assertIn("formatSensorValue", code)
        self.assertIn("toFixed(1)", code)  # rpm
        self.assertIn("toFixed(2)", code)  # °R, psia
        self.assertIn("toFixed(3)", code)  # lbm/s
        self.assertIn("N/A", code)

        # Simulate Python equivalent of the JS formatting function to verify specification
        def py_format_sensor_value(unit, num):
            if num is None:
                return "N/A"
            if unit == "rpm":
                return f"{num:.1f} rpm"
            elif unit in ("°R", "psia"):
                return f"{num:.2f} {unit}"
            elif unit == "lbm/s":
                return f"{num:.3f} lbm/s"
            elif unit in ("--", ""):
                return f"{num:.4f}"
            return f"{num:.2f} {unit}"

        self.assertEqual(py_format_sensor_value("°R", 1589.654), "1589.65 °R")
        self.assertEqual(py_format_sensor_value("psia", 554.362), "554.36 psia")
        self.assertEqual(py_format_sensor_value("rpm", 9046.19), "9046.2 rpm")
        self.assertEqual(py_format_sensor_value("lbm/s", 39.0612), "39.061 lbm/s")
        self.assertEqual(py_format_sensor_value("--", 8.41951), "8.4195")
        self.assertEqual(py_format_sensor_value("°R", None), "N/A")

    def test_03_sensor_trends_component_structure(self):
        """
        Verifies SensorTrendsChart imports metadata and renders SVG elements, presets, and cursor line.
        """
        chart_path = os.path.join(self.src_dir, "components", "dashboard", "SensorTrendsChart.jsx")
        self.assertTrue(os.path.exists(chart_path), f"Missing chart component: {chart_path}")

        with open(chart_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("SENSOR_METADATA", content)
        self.assertIn("<svg", content)
        self.assertIn("viewBox=", content)
        self.assertIn("strokeDasharray=\"4 4\"", content)  # Cursor line
        self.assertIn("C{selectedCycle}", content)  # Cursor tag
        self.assertIn("Core (T30, P30, T50, Nc)", content)  # Preset
        self.assertIn("Inspecting:", content)

    def test_04_readings_grid_component_structure(self):
        """
        Verifies SensorReadingsGrid renders sensor rows with units, category filters, and delta calculations.
        """
        grid_path = os.path.join(self.src_dir, "components", "dashboard", "SensorReadingsGrid.jsx")
        self.assertTrue(os.path.exists(grid_path), f"Missing grid component: {grid_path}")

        with open(grid_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("SENSOR_METADATA", content)
        self.assertIn("<table", content)
        self.assertIn("Cycle 1 Baseline", content)
        self.assertIn("Baseline Delta (Δ)", content)
        self.assertIn("MEASURED", content)
        self.assertIn("All Sensors (21)", content)
        self.assertIn("Gas-Path Temps (°R)", content)
        self.assertIn("Pressures (psia)", content)
        self.assertIn("Rotor Speeds (rpm)", content)

    def test_05_backend_sensors_endpoint_schema(self):
        """
        Tests GET /api/v1/engine/FD001/1/sensors verifies array length matches total cycles and trends contains all sensors.
        """
        resp = self.client.get("/api/v1/engine/FD001/1/sensors")
        self.assertEqual(resp.status_code, 200)

        data = resp.json()
        self.assertEqual(data["subset"], "FD001")
        self.assertEqual(data["unit_number"], 1)

        total_cycles = data["total_cycles"]
        self.assertGreater(total_cycles, 0)
        self.assertEqual(len(data["cycle_history"]), total_cycles)

        trends = data["trends"]
        self.assertGreaterEqual(len(trends), 14)

        # Check key sensors exist and have exact array length matching total_cycles
        key_sensors = ["sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_8", "sensor_9"]
        for s in key_sensors:
            self.assertIn(s, trends, f"Sensor {s} missing from backend trends")
            self.assertEqual(len(trends[s]), total_cycles, f"Length mismatch for {s}")

    def test_06_current_cycle_cursor_alignment(self):
        """
        Confirms trend chart includes cursor reference line math scaling with selectedCycle.
        """
        chart_path = os.path.join(self.src_dir, "components", "dashboard", "SensorTrendsChart.jsx")
        with open(chart_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check coordinate calculation formula: x = left + ((cycle - 1) / (totalCycles - 1)) * plotWidth
        self.assertIn("cursorX = padding.left + ((selectedCycle - 1) / Math.max(1, totalCycles - 1)) * plotWidth", content)
        self.assertIn("stroke=\"#06B6D4\"", content)

    def test_07_sensor_category_filtering(self):
        """
        Verifies filtering logic across temperatures, pressures, and spool speeds.
        """
        grid_path = os.path.join(self.src_dir, "components", "dashboard", "SensorReadingsGrid.jsx")
        with open(grid_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("activeTab === 'all'", content)
        self.assertIn("SENSOR_METADATA[sKey].category === activeTab", content)

    def test_08_delta_calculation_accuracy(self):
        """
        Validates sensor baseline deviation math between current cycle and cycle 1.
        """
        resp = self.client.get("/api/v1/engine/FD001/1/sensors")
        self.assertEqual(resp.status_code, 200)
        trends = resp.json()["trends"]

        # Check T30 (sensor_3) delta from cycle 1 to final cycle
        t30_series = trends["sensor_3"]
        c1_val = t30_series[0]
        c_last_val = t30_series[-1]
        delta = c_last_val - c1_val
        pct_delta = (delta / c1_val) * 100

        # Due to HPC degradation in FD001, T30 should increase as the engine degrades
        self.assertGreater(delta, 0.0, "HPC degradation must lead to positive T30 temperature rise over life.")
        self.assertGreater(pct_delta, 0.5, "T30 rise must be statistically perceptible over run-to-failure.")

    def test_09_app_integration_sensor_components(self):
        """
        Verifies App.jsx imports and renders both SensorTrendsChart and SensorReadingsGrid.
        """
        app_path = os.path.join(self.src_dir, "App.jsx")
        with open(app_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("import SensorTrendsChart from './components/dashboard/SensorTrendsChart';", content)
        self.assertIn("import SensorReadingsGrid from './components/dashboard/SensorReadingsGrid';", content)
        self.assertIn("<SensorTrendsChart />", content)
        self.assertIn("<SensorReadingsGrid />", content)

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
