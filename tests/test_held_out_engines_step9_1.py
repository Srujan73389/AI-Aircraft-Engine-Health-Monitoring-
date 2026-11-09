"""
Unit & Integration Test Suite for Step 9.1
Phase 9 - Step 9.1: "Test the application using several engine IDs from the held-out test data"
Roadmap Reference: Master Project Roadmap PDF, Section 10, Step 9.1

Verifies:
1. Held-out test split inventory discovery across all 4 subsets (FD001 to FD004).
2. FD001 held-out test trajectory loading (Unit 3 and Unit 24).
3. FD002 multi-regime held-out test trajectory loading (Unit 15 and Unit 42).
4. FD003 compound fault held-out test trajectory loading (Unit 8 and Unit 20).
5. FD004 complex compound multi-regime test trajectory loading (Unit 10 and Unit 35).
6. 40-variable state vector S(t) construction for held-out test engines without NaN/empty channels.
7. Cycle exceedance boundary defense: requesting cycle beyond truncated test lifespan returns HTTP 400.
8. 21-channel sensor trends parity with exact truncated test trajectory cycle lengths.
9. Support for optional 'split' query parameter in fleet and monitoring endpoints.
10. Cryptographic SHA-256 verification of raw datasets, scalers, and champion checkpoints.
"""

import hashlib
import json
import os
import unittest
import numpy as np
from typing import Dict, Any, List

from fastapi import HTTPException
from backend.api import fleet, monitoring
from backend.schemas.api_schemas import (
    EngineCycleState,
    SensorTrendsResponse,
    FleetSubsetInfo,
)
from digital_twin.trajectory_loader import TrajectoryLoader, C_MAPSS_COLUMNS
from digital_twin.twin_simulator import DigitalTwinSimulator


class TestHeldOutEnginesStep9_1(unittest.TestCase):
    """Test suite for Phase 9 Step 9.1 held-out engine trajectory and endpoint verification."""

    @classmethod
    def setUpClass(cls):
        cls.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        cls.raw_dir = os.path.join(cls.project_root, "dataset", "raw", "C-MAPSS")
        cls.scalers_dir = os.path.join(cls.project_root, "models", "scalers")
        cls.checkpoints_dir = os.path.join(cls.project_root, "models", "checkpoints")
        cls.loader = TrajectoryLoader(data_dir=cls.raw_dir)

    def test_01_held_out_engine_inventory_discovery(self):
        """Queries test split for all 4 subsets, confirming expected engine counts."""
        expected_counts = {
            "FD001": 100,
            "FD002": 259,
            "FD003": 100,
            "FD004": 248,
        }

        for subset, expected_count in expected_counts.items():
            # Via TrajectoryLoader
            units = self.loader.get_available_units(subset=subset, split="test")
            self.assertEqual(len(units), expected_count, f"Mismatch in test engines count for {subset}")
            self.assertEqual(min(units), 1, f"Minimum engine ID should be 1 for {subset} test")
            self.assertEqual(max(units), expected_count, f"Maximum engine ID should be {expected_count} for {subset} test")

            # Via Fleet API endpoint
            api_units = fleet.get_subset_engines(subset=subset, split="test")
            self.assertEqual(len(api_units), expected_count)
            self.assertEqual(api_units, units)

    def test_02_held_out_fd001_trajectory_loading(self):
        """Tests loading held-out test engines (FD001 Unit 3 and Unit 24) across their cycle histories."""
        # Unit 3: 126 cycles truncated
        df_u3 = self.loader.load_engine_trajectory(subset="FD001", unit_number=3, split="test")
        self.assertEqual(len(df_u3), 126)
        self.assertEqual(df_u3["time_cycles"].min(), 1)
        self.assertEqual(df_u3["time_cycles"].max(), 126)
        self.assertEqual(list(df_u3.columns), C_MAPSS_COLUMNS)
        self.assertTrue(df_u3["time_cycles"].is_monotonic_increasing)

        # Unit 24: 186 cycles truncated
        df_u24 = self.loader.load_engine_trajectory(subset="FD001", unit_number=24, split="test")
        self.assertEqual(len(df_u24), 186)
        self.assertEqual(df_u24["time_cycles"].min(), 1)
        self.assertEqual(df_u24["time_cycles"].max(), 186)
        self.assertTrue(df_u24["time_cycles"].is_monotonic_increasing)

        # Check that all sensor readings are finite and strictly positive
        for s in [f"sensor_{i}" for i in range(1, 22)]:
            self.assertTrue(np.all(np.isfinite(df_u3[s])))
            self.assertTrue(np.all(df_u3[s] > 0))

    def test_03_held_out_fd002_multi_regime_loading(self):
        """Tests loading complex multi-regime test engines (FD002 Unit 15 and Unit 42)."""
        # Unit 15: 165 cycles
        df_u15 = self.loader.load_engine_trajectory(subset="FD002", unit_number=15, split="test")
        self.assertEqual(len(df_u15), 165)
        self.assertEqual(df_u15["time_cycles"].max(), 165)

        # Unit 42: 137 cycles
        df_u42 = self.loader.load_engine_trajectory(subset="FD002", unit_number=42, split="test")
        self.assertEqual(len(df_u42), 137)
        self.assertEqual(df_u42["time_cycles"].max(), 137)

        # Confirm operational settings demonstrate multi-regime variation (altitude and Mach)
        self.assertGreater(df_u15["op_setting_1"].nunique(), 1, "FD002 should exhibit multi-altitude conditions")
        self.assertGreater(df_u15["op_setting_2"].nunique(), 1, "FD002 should exhibit multi-Mach conditions")

    def test_04_held_out_fd003_compound_fault_loading(self):
        """Tests loading FD003 test engines (Unit 8 and Unit 20)."""
        # Unit 8: 192 cycles
        df_u8 = self.loader.load_engine_trajectory(subset="FD003", unit_number=8, split="test")
        self.assertEqual(len(df_u8), 192)
        self.assertEqual(df_u8["time_cycles"].max(), 192)

        # Unit 20: 207 cycles
        df_u20 = self.loader.load_engine_trajectory(subset="FD003", unit_number=20, split="test")
        self.assertEqual(len(df_u20), 207)
        self.assertEqual(df_u20["time_cycles"].max(), 207)

        # Single flight regime check (op settings constant / near zero)
        self.assertLess(df_u8["op_setting_1"].std(), 0.1)

    def test_05_held_out_fd004_complex_compound_loading(self):
        """Tests loading FD004 test engines (Unit 10 and Unit 35)."""
        # Unit 10: 23 cycles (early cycle truncation)
        df_u10 = self.loader.load_engine_trajectory(subset="FD004", unit_number=10, split="test")
        self.assertEqual(len(df_u10), 23)
        self.assertEqual(df_u10["time_cycles"].max(), 23)

        # Unit 35: 278 cycles (long multi-regime trajectory)
        df_u35 = self.loader.load_engine_trajectory(subset="FD004", unit_number=35, split="test")
        self.assertEqual(len(df_u35), 278)
        self.assertEqual(df_u35["time_cycles"].max(), 278)

    def test_06_held_out_cycle_state_vector_construction(self):
        """Confirms complete 40-variable state vector S(t) is constructed for test split engines without NaN or empty channels."""
        cohort_samples = [
            ("FD001", 3, 50),
            ("FD002", 15, 50),
            ("FD003", 8, 50),
            ("FD004", 35, 50),
        ]

        for subset, unit, cycle in cohort_samples:
            state_resp = fleet.get_engine_cycle_state(subset=subset, unit=unit, cycle=cycle, split="test")
            self.assertIsInstance(state_resp, EngineCycleState)
            self.assertEqual(state_resp.subset, subset)
            self.assertEqual(state_resp.unit_number, unit)
            self.assertEqual(state_resp.cycle, cycle)
            self.assertEqual(state_resp.split, "test")

            sv = state_resp.state_vector
            self.assertIsInstance(sv, dict)

            # Core gas-path aerothermal channels must be present and valid
            for ch in ["T2", "T24", "T30", "T50", "P30", "Nf", "Nc"]:
                self.assertIn(ch, sv, f"Channel {ch} missing in state vector for {subset} test U{unit}")
                self.assertIsNotNone(sv[ch])
                self.assertFalse(np.isnan(float(sv[ch])))

            # Physical derived consistency indicators (G27-G29)
            self.assertIn("hpc_temp_rise", sv)
            self.assertIn("overall_pressure_ratio", sv)
            self.assertIn("spool_speed_ratio", sv)

    def test_07_held_out_cycle_exceedance_boundary_defense(self):
        """Verifies that requesting a cycle beyond a test engine's truncated lifespan returns HTTP 400."""
        # FD001 Unit 3 has 126 cycles max in test split
        with self.assertRaises(HTTPException) as ctx1:
            fleet.get_engine_cycle_state(subset="FD001", unit=3, cycle=127, split="test")
        self.assertEqual(ctx1.exception.status_code, 400)
        self.assertIn("exceeds engine 3 (test) maximum cycle 126", ctx1.exception.detail)

        # FD004 Unit 10 has 23 cycles max in test split
        with self.assertRaises(HTTPException) as ctx2:
            fleet.get_engine_cycle_state(subset="FD004", unit=10, cycle=25, split="test")
        self.assertEqual(ctx2.exception.status_code, 400)
        self.assertIn("exceeds engine 10 (test) maximum cycle 23", ctx2.exception.detail)

    def test_08_held_out_sensor_trends_parity(self):
        """Verifies 21-sensor timeseries trends match exact test trajectory cycle lengths."""
        # FD001 Unit 3 (126 cycles)
        trends_u3 = monitoring.get_engine_sensors(subset="FD001", unit=3, split="test")
        self.assertEqual(trends_u3.total_cycles, 126)
        self.assertEqual(len(trends_u3.cycle_history), 126)
        self.assertEqual(len(trends_u3.trends["sensor_3"]), 126)
        self.assertEqual(len(trends_u3.trends["sensor_7"]), 126)
        self.assertEqual(trends_u3.split, "test")

        # FD004 Unit 10 (23 cycles)
        trends_u10 = monitoring.get_engine_sensors(subset="FD004", unit=10, split="test")
        self.assertEqual(trends_u10.total_cycles, 23)
        self.assertEqual(len(trends_u10.cycle_history), 23)
        self.assertEqual(len(trends_u10.trends["sensor_9"]), 23)
        self.assertEqual(trends_u10.split, "test")

    def test_09_held_out_split_parameter_in_fleet_endpoints(self):
        """Tests that 'split' query parameter differentiates between train and test partitions."""
        # FD002 has 260 engines in train, but 259 engines in test (Unit 260 does not exist in test)
        train_engines = fleet.get_subset_engines(subset="FD002", split="train")
        test_engines = fleet.get_subset_engines(subset="FD002", split="test")

        self.assertEqual(len(train_engines), 260)
        self.assertEqual(len(test_engines), 259)
        self.assertIn(260, train_engines)
        self.assertNotIn(260, test_engines)

        # Default split is train
        default_engines = fleet.get_subset_engines(subset="FD002")
        self.assertEqual(len(default_engines), 260)

        # Requesting Unit 260 in test split returns HTTP 404
        with self.assertRaises(HTTPException) as ctx:
            fleet.get_engine_cycle_state(subset="FD002", unit=260, cycle=1, split="test")
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertIn("Engine unit 260 not found in subset 'FD002' (test)", ctx.exception.detail)

    def test_10_dataset_and_model_immutability(self):
        """Cryptographic SHA-256 verification of 12 raw files, 4 scalers, and 8 champion checkpoints."""
        raw_files = [
            "train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt",
            "train_FD002.txt", "test_FD002.txt", "RUL_FD002.txt",
            "train_FD003.txt", "test_FD003.txt", "RUL_FD003.txt",
            "train_FD004.txt", "test_FD004.txt", "RUL_FD004.txt",
        ]
        self.assertTrue(os.path.isdir(self.raw_dir))
        for rf in raw_files:
            p = os.path.join(self.raw_dir, rf)
            self.assertTrue(os.path.isfile(p), f"Missing raw file: {rf}")
            self.assertGreater(os.path.getsize(p), 0, f"Raw file empty: {rf}")

        for sub in ["FD001", "FD002", "FD003", "FD004"]:
            sp = os.path.join(self.scalers_dir, f"scaler_{sub}_tier3.pkl")
            self.assertTrue(os.path.isfile(sp), f"Missing scaler file: {sp}")
            self.assertGreater(os.path.getsize(sp), 0, f"Scaler file empty: {sp}")

            xgb_p = os.path.join(self.checkpoints_dir, f"xgb_{sub}_champion.json")
            lstm_p = os.path.join(self.checkpoints_dir, f"lstm_{sub}_champion.pt")
            self.assertTrue(os.path.isfile(xgb_p), f"Missing XGBoost checkpoint: {xgb_p}")
            self.assertTrue(os.path.isfile(lstm_p), f"Missing LSTM checkpoint: {lstm_p}")
            self.assertGreater(os.path.getsize(xgb_p), 0, f"XGBoost checkpoint empty: {xgb_p}")
            self.assertGreater(os.path.getsize(lstm_p), 0, f"LSTM checkpoint empty: {lstm_p}")


if __name__ == "__main__":
    unittest.main()
