"""
Unit & Integration Test Suite for Phase 5 Step 5.4 Digital Twin
Tests trajectory loading, time-step advancement, state representation,
derived indicators G27-G29, rolling window history, left-edge padding,
feature tier slicing, and dataset immutability.
"""

import hashlib
import os
import sys
import unittest
import numpy as np
import pandas as pd

# Add repository root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from digital_twin.engine_state import EngineState, StateClassification
from digital_twin.trajectory_loader import TrajectoryLoader
from digital_twin.twin_simulator import DigitalTwinSimulator, ORDERED_SCALER_FEATURES, TIER1_CORE_SENSORS


def compute_file_sha256(filepath: str) -> str:
    """Calculate SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class TestDigitalTwinStep5_4(unittest.TestCase):
    """Comprehensive test cases for Phase 5 Step 5.4 Digital Twin foundation."""

    @classmethod
    def setUpClass(cls):
        cls.raw_data_dir = os.path.join("dataset", "raw", "C-MAPSS")
        cls.fd001_train_path = os.path.join(cls.raw_data_dir, "train_FD001.txt")
        # Store pre-test hash of raw data to verify immutability
        cls.pre_test_hash = compute_file_sha256(cls.fd001_train_path)

    def test_01_trajectory_loader_valid_engine(self):
        """Test loading a valid engine trajectory from train_FD001."""
        loader = TrajectoryLoader(data_dir=self.raw_data_dir)
        df_engine = loader.load_engine_trajectory(subset="FD001", unit_number=1, split="train")

        self.assertIsInstance(df_engine, pd.DataFrame)
        self.assertEqual(df_engine.shape[1], 26)
        self.assertEqual(df_engine["unit_number"].iloc[0], 1)
        self.assertEqual(df_engine["time_cycles"].iloc[0], 1)
        self.assertEqual(len(df_engine), 192)  # FD001 engine 1 has exactly 192 cycles
        # Monotonicity check
        self.assertTrue((np.diff(df_engine["time_cycles"].values) == 1).all())

    def test_02_trajectory_loader_invalid_engine_id(self):
        """Test error handling when requesting an invalid or non-existent engine ID."""
        loader = TrajectoryLoader(data_dir=self.raw_data_dir)
        # Unit 999 does not exist in FD001 (which has units 1..100)
        with self.assertRaises(ValueError) as ctx:
            loader.load_engine_trajectory(subset="FD001", unit_number=999, split="train")
        self.assertIn("not found in FD001", str(ctx.exception))

    def test_03_simulator_initial_state(self):
        """Test simulator initialization at cycle 1."""
        sim = DigitalTwinSimulator(subset="FD001")
        initial_state = sim.load_engine(unit_number=1, split="train")

        self.assertIsInstance(initial_state, EngineState)
        self.assertEqual(initial_state.unit_number, 1)
        self.assertEqual(initial_state.time_cycles, 1)
        self.assertEqual(initial_state.subset, "FD001")
        self.assertEqual(initial_state.feature_tier, "Tier1_Core14")
        self.assertTrue(initial_state.is_padded)
        self.assertEqual(initial_state.padded_cycles_count, 29)  # 30 - 1 = 29
        self.assertEqual(initial_state.history_window_length, 1)

    def test_04_simulator_advance_cycle(self):
        """Test advancing simulator from cycle 1 to cycle 2."""
        sim = DigitalTwinSimulator(subset="FD001")
        s1 = sim.load_engine(unit_number=1, split="train")
        self.assertEqual(s1.time_cycles, 1)

        s2 = sim.step()
        self.assertIsNotNone(s2)
        self.assertEqual(s2.unit_number, 1)
        self.assertEqual(s2.time_cycles, 2)
        self.assertTrue(s2.is_padded)
        self.assertEqual(s2.padded_cycles_count, 28)  # 30 - 2 = 28
        self.assertEqual(s2.history_window_length, 2)

    def test_05_sensor_value_alignment_against_raw(self):
        """Verify that sensor values in EngineState match raw C-MAPSS text file values exactly."""
        sim = DigitalTwinSimulator(subset="FD001")
        state = sim.load_engine(unit_number=1, split="train")

        # Read line 1 of train_FD001.txt
        with open(self.fd001_train_path, "r") as f:
            first_line = f.readline().strip().split()

        # Check values
        self.assertAlmostEqual(state.altitude, float(first_line[2]), places=6)
        self.assertAlmostEqual(state.mach_number, float(first_line[3]), places=6)
        self.assertAlmostEqual(state.throttle_resolver_angle, float(first_line[4]), places=6)
        self.assertAlmostEqual(state.T2, float(first_line[5]), places=4)
        self.assertAlmostEqual(state.T24, float(first_line[6]), places=4)
        self.assertAlmostEqual(state.T30, float(first_line[7]), places=4)
        self.assertAlmostEqual(state.T50, float(first_line[8]), places=4)
        self.assertAlmostEqual(state.P2, float(first_line[9]), places=4)
        self.assertAlmostEqual(state.P15, float(first_line[10]), places=4)
        self.assertAlmostEqual(state.P30, float(first_line[11]), places=4)
        self.assertAlmostEqual(state.Nf, float(first_line[12]), places=4)
        self.assertAlmostEqual(state.Nc, float(first_line[13]), places=4)

    def test_06_derived_consistency_indicators_g27_g29(self):
        """Test calculation and validity of derived indicators G27, G28, G29."""
        sim = DigitalTwinSimulator(subset="FD001")
        state = sim.load_engine(unit_number=1, split="train")

        # G27: T30 - T24
        expected_g27 = state.T30 - state.T24
        self.assertAlmostEqual(state.hpc_temp_rise, expected_g27, places=6)
        self.assertGreater(state.hpc_temp_rise, 700.0)

        # G28: P30 / P2
        expected_g28 = state.P30 / state.P2
        self.assertAlmostEqual(state.overall_pressure_ratio, expected_g28, places=6)
        self.assertGreater(state.overall_pressure_ratio, 24.0)

        # G29: Nc / Nf
        expected_g29 = state.Nc / state.Nf
        self.assertAlmostEqual(state.spool_speed_ratio, expected_g29, places=6)
        self.assertGreater(state.spool_speed_ratio, 3.5)

        # Sanity check method
        sanity = state.check_thermodynamic_sanity()
        self.assertTrue(sanity["overall_sanity_pass"])

    def test_07_left_edge_replication_padding(self):
        """Test that short sequences (< 30 cycles) replicate cycle 1 backwards."""
        sim = DigitalTwinSimulator(subset="FD001")
        s1 = sim.load_engine(unit_number=1, split="train")

        # At cycle 1, the sequence tensor should repeat the cycle 1 feature vector across all 30 rows
        g31 = s1.temporal_seq_tensor
        self.assertEqual(g31.shape, (30, 14))
        for row_idx in range(30):
            np.testing.assert_array_almost_equal(g31[row_idx], s1.normalized_feature_vec)

        # Advance to cycle 5
        for _ in range(4):
            state = sim.step()

        self.assertEqual(state.time_cycles, 5)
        self.assertTrue(state.is_padded)
        self.assertEqual(state.padded_cycles_count, 25)
        g31_c5 = state.temporal_seq_tensor

        # Rows 0 to 25 should all equal cycle 1 normalized features
        c1_core = s1.normalized_feature_vec
        for r in range(26):  # rows 0 to 25
            np.testing.assert_array_almost_equal(g31_c5[r], c1_core)

    def test_08_full_window_unpadded_at_cycle_30(self):
        """Test that at cycle 30 and beyond, is_padded is False and padded_count is 0."""
        sim = DigitalTwinSimulator(subset="FD001")
        sim.load_engine(unit_number=1, split="train")

        # Advance to cycle 30
        for _ in range(29):
            state = sim.step()

        self.assertEqual(state.time_cycles, 30)
        self.assertFalse(state.is_padded)
        self.assertEqual(state.padded_cycles_count, 0)
        self.assertEqual(state.temporal_seq_tensor.shape, (30, 14))

    def test_09_feature_tier_dimensions_fd001_vs_fd002(self):
        """Test feature tier slicing: FD001 (14 Core) vs FD002 (24 Extended)."""
        # FD001
        sim_fd001 = DigitalTwinSimulator(subset="FD001")
        s_fd001 = sim_fd001.load_engine(unit_number=1, split="train")
        self.assertEqual(s_fd001.normalized_feature_vec.shape, (14,))
        self.assertEqual(s_fd001.temporal_seq_tensor.shape, (30, 14))
        self.assertEqual(s_fd001.feature_tier, "Tier1_Core14")

        # FD002
        sim_fd002 = DigitalTwinSimulator(subset="FD002")
        s_fd002 = sim_fd002.load_engine(unit_number=1, split="train")
        self.assertEqual(s_fd002.normalized_feature_vec.shape, (24,))
        self.assertEqual(s_fd002.temporal_seq_tensor.shape, (30, 24))
        self.assertEqual(s_fd002.feature_tier, "Tier3_Extended24")

    def test_10_reproducibility_of_state_updates(self):
        """Test that stepping through the same engine twice yields identical state vectors."""
        sim1 = DigitalTwinSimulator(subset="FD001")
        sim1.load_engine(unit_number=5, split="train")
        for _ in range(10):
            sim1.step()
        state1 = sim1.current_state

        sim2 = DigitalTwinSimulator(subset="FD001")
        sim2.load_engine(unit_number=5, split="train")
        for _ in range(10):
            sim2.step()
        state2 = sim2.current_state

        self.assertEqual(state1.time_cycles, state2.time_cycles)
        self.assertEqual(state1.hpc_temp_rise, state2.hpc_temp_rise)
        self.assertEqual(state1.overall_pressure_ratio, state2.overall_pressure_ratio)
        np.testing.assert_array_equal(state1.normalized_feature_vec, state2.normalized_feature_vec)
        np.testing.assert_array_equal(state1.temporal_seq_tensor, state2.temporal_seq_tensor)

    def test_11_dataset_immutability(self):
        """Verify that simulation does not modify raw NASA dataset files."""
        post_test_hash = compute_file_sha256(self.fd001_train_path)
        self.assertEqual(self.pre_test_hash, post_test_hash, "Raw dataset hash altered during testing!")


if __name__ == "__main__":
    unittest.main()
