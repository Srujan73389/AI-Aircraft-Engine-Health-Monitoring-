"""
Unit and Integration Tests for Phase 5 Step 5.10: Numerical Stability, Repeatability, and Edge Robustness
Verifies multi-run deterministic repeatability, ML inference repeatability, thermodynamic boundary compliance,
long-trajectory numerical stability (>300 cycles), FIFO buffer invariance, edge-case rejection, and artifact immutability.

Roadmap Reference: Phase 5 - Step 5.10:
"Validate numerical stability, repeatability, and edge robustness of the Digital Twin."
"""

import hashlib
import json
import math
import os
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

from digital_twin.engine_state import EngineState
from digital_twin.twin_simulator import (
    DigitalTwinSimulator,
    ORDERED_SCALER_FEATURES,
    TIER1_CORE_SENSORS,
    CORE_INDICES_IN_24,
)
from digital_twin.scenario_interface import (
    create_thermal_stress_scenario,
    create_sensor_bias_scenario,
)
from digital_twin.stability_validator import (
    TwinStabilityValidator,
    PhysicalRangeCheckResult,
    LongTrajectoryStabilityResult,
    RepeatabilityResult,
    EdgeCaseResilienceResult,
    THERMODYNAMIC_BOUNDS,
)


class TestDigitalTwinStep5_10(unittest.TestCase):
    """
    Verification suite for Phase 5 Step 5.10: Numerical Stability, Repeatability, and Edge Robustness.
    """

    @classmethod
    def setUpClass(cls):
        cls.raw_data_dir = os.path.join("dataset", "raw", "C-MAPSS")
        cls.scalers_dir = os.path.join("models", "scalers")
        cls.checkpoint_dir = os.path.join("models", "checkpoints")

        # 1. Capture SHA-256 hashes of all 12 raw C-MAPSS dataset files
        cls.raw_hashes = {}
        for fname in sorted(os.listdir(cls.raw_data_dir)):
            if fname.endswith(".txt") and fname != "readme.txt":
                fpath = os.path.join(cls.raw_data_dir, fname)
                with open(fpath, "rb") as f:
                    cls.raw_hashes[fname] = hashlib.sha256(f.read()).hexdigest()

        # 2. Capture SHA-256 hashes of all 4 subset scalers
        cls.scaler_hashes = {}
        for fname in sorted(os.listdir(cls.scalers_dir)):
            if fname.endswith(".pkl"):
                fpath = os.path.join(cls.scalers_dir, fname)
                with open(fpath, "rb") as f:
                    cls.scaler_hashes[fname] = hashlib.sha256(f.read()).hexdigest()

        # 3. Capture SHA-256 hashes of all 8 champion model checkpoints
        cls.checkpoint_hashes = {}
        for fname in sorted(os.listdir(cls.checkpoint_dir)):
            if fname.endswith((".json", ".pt")) and not fname.endswith("manifest.json"):
                fpath = os.path.join(cls.checkpoint_dir, fname)
                with open(fpath, "rb") as f:
                    cls.checkpoint_hashes[fname] = hashlib.sha256(f.read()).hexdigest()

        cls.validator = TwinStabilityValidator(
            raw_data_dir=cls.raw_data_dir,
            scalers_dir=cls.scalers_dir,
            checkpoint_dir=cls.checkpoint_dir,
        )

    def test_01_multi_run_deterministic_state_repeatability(self):
        """
        Verify that running two independent DigitalTwinSimulator instances on identical telemetry
        yields exactly identical state histories across G1-G29 (max difference = 0.0).
        """
        result = self.validator.validate_deterministic_repeatability(
            subset="FD001", unit_number=1, split="train", enable_ml=False
        )
        self.assertTrue(result.exact_match)
        self.assertEqual(result.max_diff_g1_g29, 0.0)
        self.assertEqual(result.max_diff_g30, 0.0)
        self.assertEqual(result.max_diff_g31, 0.0)
        self.assertEqual(result.cycles_compared, 192)

    def test_02_multi_run_ml_inference_repeatability(self):
        """
        Verify that running two independent DigitalTwinSimulator instances yields exactly
        identical ML prognostic predictions (G35 XGBoost and G36 LSTM max abs diff = 0.0).
        """
        result = self.validator.validate_deterministic_repeatability(
            subset="FD001", unit_number=1, split="train", enable_ml=True
        )
        self.assertTrue(result.exact_match)
        self.assertEqual(result.max_diff_g35_xgb, 0.0)
        self.assertEqual(result.max_diff_g36_lstm, 0.0)

    def test_03_scenario_repeatability_thermal_stress(self):
        """
        Verify that running two independent DigitalTwinSimulator instances under identical
        What-If thermal stress perturbations yields identical outputs (max abs diff = 0.0).
        """
        sc = create_thermal_stress_scenario(
            scenario_id="SC_TEST_REPEAT_THERMAL",
            target="T30",
            offset_degR=20.0,
            name="Test Thermal Stress Repeatability",
        )
        result = self.validator.validate_scenario_repeatability(
            subset="FD001", unit_number=1, scenario_config=sc, split="train", enable_ml=True
        )
        self.assertTrue(result.exact_match)
        self.assertEqual(result.max_diff_g1_g29, 0.0)
        self.assertEqual(result.max_diff_g30, 0.0)
        self.assertEqual(result.max_diff_g31, 0.0)
        self.assertEqual(result.max_diff_g35_xgb, 0.0)
        self.assertEqual(result.max_diff_g36_lstm, 0.0)

    def test_04_scenario_repeatability_sensor_bias(self):
        """
        Verify that running two independent DigitalTwinSimulator instances under identical
        What-If sensor bias perturbations yields identical outputs (max abs diff = 0.0).
        """
        sc = create_sensor_bias_scenario(
            scenario_id="SC_TEST_REPEAT_BIAS",
            target="P30",
            bias=15.0,
            unit="psia",
            name="Test Sensor Bias Repeatability",
        )
        result = self.validator.validate_scenario_repeatability(
            subset="FD002", unit_number=1, scenario_config=sc, split="train", enable_ml=True
        )
        self.assertTrue(result.exact_match)
        self.assertEqual(result.max_diff_g1_g29, 0.0)
        self.assertEqual(result.max_diff_g30, 0.0)
        self.assertEqual(result.max_diff_g31, 0.0)

    def test_05_numerical_boundary_enforcement_t24_t30_p30(self):
        """
        Assert that physical variables T24, T30, and P30 remain within defensible thermodynamic bounds.
        """
        sim = DigitalTwinSimulator(subset="FD001", data_dir=self.raw_data_dir)
        sim.load_engine(unit_number=1, split="train")
        sim.run_all()

        is_valid, checks = self.validator.validate_telemetry_physical_ranges(sim.state_history)
        self.assertTrue(is_valid)

        check_map = {c.variable_name: c for c in checks}
        # T24
        self.assertIn("T24", check_map)
        self.assertTrue(check_map["T24"].is_valid)
        self.assertGreaterEqual(check_map["T24"].observed_min, 450.0)
        self.assertLessEqual(check_map["T24"].observed_max, 750.0)

        # T30
        self.assertIn("T30", check_map)
        self.assertTrue(check_map["T30"].is_valid)
        self.assertGreaterEqual(check_map["T30"].observed_min, 1100.0)
        self.assertLessEqual(check_map["T30"].observed_max, 1850.0)

        # P30
        self.assertIn("P30", check_map)
        self.assertTrue(check_map["P30"].is_valid)
        self.assertGreaterEqual(check_map["P30"].observed_min, 100.0)
        self.assertLessEqual(check_map["P30"].observed_max, 700.0)

    def test_06_numerical_boundary_enforcement_opr_and_spool_ratio(self):
        """
        Assert that derived indicators OPR (P30 / P2) and spool speed ratio (Nc / Nf)
        remain within defensible physical bounds across multi-regime operations (FD002).
        """
        sim = DigitalTwinSimulator(subset="FD002", data_dir=self.raw_data_dir)
        sim.load_engine(unit_number=1, split="train")
        sim.run_all()

        is_valid, checks = self.validator.validate_telemetry_physical_ranges(sim.state_history)
        self.assertTrue(is_valid)

        check_map = {c.variable_name: c for c in checks}
        # OPR
        self.assertIn("overall_pressure_ratio", check_map)
        opr_c = check_map["overall_pressure_ratio"]
        self.assertTrue(opr_c.is_valid)
        self.assertGreaterEqual(opr_c.observed_min, 15.0)
        self.assertLessEqual(opr_c.observed_max, 55.0)

        # Spool speed ratio
        self.assertIn("spool_speed_ratio", check_map)
        spool_c = check_map["spool_speed_ratio"]
        self.assertTrue(spool_c.is_valid)
        self.assertGreaterEqual(spool_c.observed_min, 3.0)
        self.assertLessEqual(spool_c.observed_max, 5.5)

    def test_07_robustness_to_corrupted_input_nan(self):
        """
        Verify that non-finite NaN input raises a graceful ValueError.
        """
        bad_rec = {"sensor_1": 518.67, "sensor_3": float("nan"), "sensor_5": 14.62, "sensor_8": 2388.0}
        with self.assertRaises(ValueError) as ctx:
            TwinStabilityValidator.validate_telemetry_record(bad_rec)
        self.assertIn("Non-finite", str(ctx.exception))

    def test_08_robustness_to_corrupted_input_inf(self):
        """
        Verify that positive/negative infinity input raises a graceful ValueError.
        """
        bad_rec_pos = {"sensor_1": 518.67, "sensor_3": float("inf"), "sensor_5": 14.62, "sensor_8": 2388.0}
        with self.assertRaises(ValueError) as ctx:
            TwinStabilityValidator.validate_telemetry_record(bad_rec_pos)
        self.assertIn("Non-finite", str(ctx.exception))

        bad_rec_neg = {"sensor_1": 518.67, "sensor_3": float("-inf"), "sensor_5": 14.62, "sensor_8": 2388.0}
        with self.assertRaises(ValueError) as ctx:
            TwinStabilityValidator.validate_telemetry_record(bad_rec_neg)
        self.assertIn("Non-finite", str(ctx.exception))

    def test_09_robustness_to_negative_pressure(self):
        """
        Verify that unphysical negative or zero pressure (P2 <= 0, P30 <= 0) raises a graceful ValueError.
        """
        bad_rec_zero = {"sensor_5": 0.0, "sensor_8": 2388.0}
        with self.assertRaises(ValueError) as ctx:
            TwinStabilityValidator.validate_telemetry_record(bad_rec_zero)
        self.assertIn("non-positive pressure", str(ctx.exception).lower())

        bad_rec_neg = {"sensor_7": -10.0, "sensor_8": 2388.0}
        with self.assertRaises(ValueError) as ctx:
            TwinStabilityValidator.validate_telemetry_record(bad_rec_neg)
        self.assertIn("non-positive pressure", str(ctx.exception).lower())

    def test_10_robustness_to_zero_speed(self):
        """
        Verify that unphysical zero or negative rotational speed (Nf <= 0, Nc <= 0) raises a graceful ValueError.
        """
        bad_rec_zero = {"sensor_5": 14.62, "sensor_8": 0.0}
        with self.assertRaises(ValueError) as ctx:
            TwinStabilityValidator.validate_telemetry_record(bad_rec_zero)
        self.assertIn("non-positive rotational speed", str(ctx.exception).lower())

        bad_rec_neg = {"sensor_5": 14.62, "sensor_9": -50.0}
        with self.assertRaises(ValueError) as ctx:
            TwinStabilityValidator.validate_telemetry_record(bad_rec_neg)
        self.assertIn("non-positive rotational speed", str(ctx.exception).lower())

    def test_11_robustness_to_non_monotonic_cycles(self):
        """
        Verify that non-monotonic or discontinuous cycle sequences raise a graceful ValueError.
        """
        # Duplicate cycle 5
        bad_df_dup = pd.DataFrame({
            "unit_number": [1, 1, 1],
            "time_cycles": [4, 5, 5],
            "sensor_5": [14.62, 14.62, 14.62],
            "sensor_8": [2388.0, 2388.0, 2388.0],
        })
        with self.assertRaises(ValueError) as ctx:
            TwinStabilityValidator.validate_trajectory_sequence(bad_df_dup)
        self.assertIn("Non-monotonic", str(ctx.exception))

        # Backward cycle jump 6 -> 4
        bad_df_jump = pd.DataFrame({
            "unit_number": [1, 1, 1],
            "time_cycles": [5, 6, 4],
            "sensor_5": [14.62, 14.62, 14.62],
            "sensor_8": [2388.0, 2388.0, 2388.0],
        })
        with self.assertRaises(ValueError) as ctx:
            TwinStabilityValidator.validate_trajectory_sequence(bad_df_jump)
        self.assertIn("Non-monotonic", str(ctx.exception))

    def test_12_stability_of_internal_fifo_buffer_long_trajectory(self):
        """
        Verify the stability of the internal 30-cycle FIFO buffer over long execution runs (>300 cycles):
        - Tensor shape remains exactly (30, C).
        - All states remain finite throughout all cycles.
        - Memory consumption does not grow unboundedly.
        """
        # FD004 Unit 1 has 321 cycles
        res = self.validator.validate_long_trajectory_stability(
            subset="FD004", unit_number=1, split="train", enable_ml=True
        )
        self.assertTrue(res.is_stable)
        self.assertTrue(res.fifo_buffer_shape_invariant)
        self.assertEqual(res.expected_fifo_shape, (30, 24))
        self.assertTrue(res.all_states_finite)
        self.assertGreaterEqual(res.total_cycles, 300)

        # FD001 Unit 69 has 362 cycles
        res_fd001 = self.validator.validate_long_trajectory_stability(
            subset="FD001", unit_number=69, split="train", enable_ml=True
        )
        self.assertTrue(res_fd001.is_stable)
        self.assertTrue(res_fd001.fifo_buffer_shape_invariant)
        self.assertEqual(res_fd001.expected_fifo_shape, (30, 14))
        self.assertTrue(res_fd001.all_states_finite)
        self.assertGreaterEqual(res_fd001.total_cycles, 350)

    def test_13_edge_case_resilience_comprehensive_suite(self):
        """
        Verify that all 5 edge-case resilience tests pass in the integrated validator suite.
        """
        edge_res = self.validator.validate_edge_case_resilience()
        self.assertTrue(edge_res.all_passed)
        self.assertTrue(edge_res.nan_rejection_passed)
        self.assertTrue(edge_res.inf_rejection_passed)
        self.assertTrue(edge_res.negative_pressure_rejection_passed)
        self.assertTrue(edge_res.zero_speed_rejection_passed)
        self.assertTrue(edge_res.non_monotonic_cycles_rejection_passed)

    def test_14_step5_10_artifact_generation_and_integrity(self):
        """
        Verify that Step 5.10 artifacts exist and contain valid data in reports/digital_twin/step5_10/.
        """
        out_dir = os.path.join("reports", "digital_twin", "step5_10")
        json_file = os.path.join(out_dir, "stability_summary.json")
        md_file = os.path.join(out_dir, "stability_summary.md")
        plot_file = os.path.join(out_dir, "repeatability_and_stability_plot.png")

        self.assertTrue(os.path.exists(json_file))
        self.assertTrue(os.path.exists(md_file))
        self.assertTrue(os.path.exists(plot_file))

        with open(json_file, "r") as f:
            data = json.load(f)

        self.assertIn("long_trajectory_stability", data)
        self.assertIn("deterministic_repeatability", data)
        self.assertIn("scenario_repeatability", data)
        self.assertIn("physical_range_validation", data)
        self.assertIn("edge_case_resilience", data)

    def test_15_raw_dataset_before_and_after_immutability(self):
        """
        Verify that raw C-MAPSS dataset files were not modified during the test session.
        Compares SHA-256 hashes captured in setUpClass with current hashes.
        """
        self.assertGreaterEqual(len(self.raw_hashes), 12)
        for fname, initial_hash in self.raw_hashes.items():
            fpath = os.path.join(self.raw_data_dir, fname)
            with open(fpath, "rb") as f:
                current_hash = hashlib.sha256(f.read()).hexdigest()
            self.assertEqual(
                current_hash,
                initial_hash,
                f"Raw dataset file {fname} was modified during test execution!",
            )

    def test_16_model_scalers_and_checkpoints_immutability(self):
        """
        Verify that model scalers and champion checkpoints were not modified during the test session.
        Compares SHA-256 hashes captured in setUpClass with current hashes.
        """
        # Scalers
        self.assertEqual(len(self.scaler_hashes), 4)
        for fname, initial_hash in self.scaler_hashes.items():
            fpath = os.path.join(self.scalers_dir, fname)
            with open(fpath, "rb") as f:
                current_hash = hashlib.sha256(f.read()).hexdigest()
            self.assertEqual(
                current_hash,
                initial_hash,
                f"Scaler file {fname} was modified during test execution!",
            )

        # Checkpoints
        self.assertEqual(len(self.checkpoint_hashes), 8)
        for fname, initial_hash in self.checkpoint_hashes.items():
            fpath = os.path.join(self.checkpoint_dir, fname)
            with open(fpath, "rb") as f:
                current_hash = hashlib.sha256(f.read()).hexdigest()
            self.assertEqual(
                current_hash,
                initial_hash,
                f"Checkpoint file {fname} was modified during test execution!",
            )


if __name__ == "__main__":
    unittest.main()
