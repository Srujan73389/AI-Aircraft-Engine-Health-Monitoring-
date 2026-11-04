"""
Unit and Integration Tests for Phase 5 Step 5.8: Baseline vs Altered Scenario Execution
Verifies controlled comparison experiments, fair comparison conditions, and prognostic sensitivity.

Roadmap Reference: Phase 5 - Step 5.8
Evaluates prognostic response under identical starting conditions and recorded telemetry.
"""

import hashlib
import json
import math
import os
import unittest
import numpy as np
import pandas as pd

from digital_twin.engine_state import EngineState
from digital_twin.scenario_interface import (
    ScenarioCategory,
    ScenarioConfig,
    create_thermal_stress_scenario,
    create_sensor_bias_scenario,
    create_accelerated_aging_scenario,
)
from digital_twin.twin_simulator import DigitalTwinSimulator
from digital_twin.scenario_comparison import (
    ScenarioComparisonRunner,
    ScenarioComparisonResult,
)


class TestDigitalTwinStep5_8(unittest.TestCase):
    """
    Verification suite for Phase 5 Step 5.8: Baseline vs Altered Scenario Execution.
    Ensures identical initial conditions, non-corrupted telemetry, reproducible outputs,
    and before-and-after test-session artifact immutability.
    """

    @classmethod
    def setUpClass(cls):
        cls.data_dir = os.path.join("dataset", "raw", "C-MAPSS")
        cls.scalers_dir = os.path.join("models", "scalers")
        cls.ckpt_dir = os.path.join("models", "checkpoints")

        # 1. Capture SHA-256 hashes of all raw C-MAPSS dataset files
        cls.raw_file_hashes = {}
        for fname in sorted(os.listdir(cls.data_dir)):
            if fname.endswith(".txt"):
                fpath = os.path.join(cls.data_dir, fname)
                with open(fpath, "rb") as f:
                    cls.raw_file_hashes[fname] = hashlib.sha256(f.read()).hexdigest()

        # 2. Capture SHA-256 hashes of all 4 subset scalers
        cls.scaler_hashes = {}
        for fname in sorted(os.listdir(cls.scalers_dir)):
            if fname.endswith(".pkl"):
                fpath = os.path.join(cls.scalers_dir, fname)
                with open(fpath, "rb") as f:
                    cls.scaler_hashes[fname] = hashlib.sha256(f.read()).hexdigest()

        # 3. Capture SHA-256 hashes of all 8 Phase 4 champion checkpoints
        cls.ckpt_hashes = {}
        for fname in sorted(os.listdir(cls.ckpt_dir)):
            if fname.startswith(("xgb_", "lstm_")) and fname.endswith(("_champion.json", "_champion.pt")):
                fpath = os.path.join(cls.ckpt_dir, fname)
                with open(fpath, "rb") as f:
                    cls.ckpt_hashes[fname] = hashlib.sha256(f.read()).hexdigest()

        assert len(cls.raw_file_hashes) >= 12, f"Expected >= 12 raw files, found {len(cls.raw_file_hashes)}."
        assert len(cls.scaler_hashes) == 4, f"Expected 4 scaler files, found {len(cls.scaler_hashes)}."
        assert len(cls.ckpt_hashes) == 8, f"Expected 8 champion checkpoints, found {len(cls.ckpt_hashes)}."

        cls.runner = ScenarioComparisonRunner(
            data_dir=cls.data_dir,
            scalers_dir=cls.scalers_dir,
            checkpoint_dir=cls.ckpt_dir,
        )

    # -------------------------------------------------------------------------
    # 1. Fair Comparison & Initial Conditions Tests
    # -------------------------------------------------------------------------
    def test_01_identical_initial_conditions(self):
        """Verify that baseline and altered simulators start from identical initial states."""
        sc = create_thermal_stress_scenario(
            scenario_id="SCEN_INIT_TEST",
            target="T30",
            offset_degR=35.0,
        )
        sim_base = DigitalTwinSimulator(subset="FD001", data_dir=self.data_dir, enable_ml_inference=True)
        sim_alt = DigitalTwinSimulator(subset="FD001", data_dir=self.data_dir, enable_ml_inference=True, scenario=sc)

        s_b = sim_base.load_engine(unit_number=1, split="train")
        s_a = sim_alt.load_engine(unit_number=1, split="train")

        self.assertEqual(s_b.unit_number, s_a.unit_number)
        self.assertEqual(s_b.time_cycles, 1)
        self.assertEqual(s_a.time_cycles, 1)
        self.assertEqual(s_b.T30, s_a.T30)
        self.assertEqual(s_b.P30, s_a.P30)
        self.assertEqual(s_b.Nc, s_a.Nc)
        self.assertEqual(s_b.altitude, s_a.altitude)

    def test_02_baseline_execution_without_scenario(self):
        """Verify baseline simulator has no active scenario fields populated."""
        sim = DigitalTwinSimulator(subset="FD001", data_dir=self.data_dir, enable_ml_inference=True)
        s = sim.load_engine(unit_number=1, split="train")

        self.assertIsNone(sim.active_scenario)
        self.assertIsNone(s.thermal_stress_offset)
        self.assertIsNone(s.sensor_bias_delta)
        self.assertIsNone(s.scenario_id)
        self.assertIsNone(s.scenario_audit)
        self.assertIsNone(sim.current_scenario_audit)

    # -------------------------------------------------------------------------
    # 2. Experiment Execution Tests (Thermal Stress & Sensor Bias)
    # -------------------------------------------------------------------------
    def test_03_thermal_stress_execution(self):
        """Execute Thermal Stress experiment (+35 °R on T30) across 15 cycles and verify outputs."""
        sc = create_thermal_stress_scenario(
            scenario_id="SCEN_TS_TEST",
            target="T30",
            offset_degR=35.0,
        )
        res = self.runner.run_comparison(
            subset="FD001",
            unit_number=1,
            split="train",
            scenario=sc,
            experiment_name="Test Thermal Stress",
            max_cycles=15,
        )

        self.assertEqual(res.total_cycles, 15)
        self.assertTrue(res.fair_comparison_checks["identical_initial_conditions"])
        self.assertTrue(res.fair_comparison_checks["unmodified_measured_telemetry"])
        self.assertTrue(res.fair_comparison_checks["baseline_scenario_fields_unset"])
        self.assertTrue(res.fair_comparison_checks["altered_scenario_fields_populated"])
        self.assertTrue(res.fair_comparison_checks["all_outputs_finite"])

        # Check delta metrics exist and are finite
        self.assertTrue(math.isfinite(res.summary_metrics["delta_rul_xgb"]["mean"]))
        self.assertTrue(math.isfinite(res.summary_metrics["delta_rul_lstm"]["mean"]))

    def test_04_sensor_bias_execution(self):
        """Execute Sensor Bias experiment (+15 psia on P30) across 15 cycles and verify outputs."""
        sc = create_sensor_bias_scenario(
            scenario_id="SCEN_SB_TEST",
            target="P30",
            bias=15.0,
            unit="psia",
        )
        res = self.runner.run_comparison(
            subset="FD001",
            unit_number=1,
            split="train",
            scenario=sc,
            experiment_name="Test Sensor Bias",
            max_cycles=15,
        )

        self.assertEqual(res.total_cycles, 15)
        self.assertTrue(res.fair_comparison_checks["identical_initial_conditions"])
        self.assertTrue(res.fair_comparison_checks["unmodified_measured_telemetry"])
        self.assertTrue(res.fair_comparison_checks["all_outputs_finite"])
        self.assertTrue(math.isfinite(res.summary_metrics["delta_rul_xgb"]["mean"]))
        self.assertTrue(math.isfinite(res.summary_metrics["delta_rul_lstm"]["mean"]))

    # -------------------------------------------------------------------------
    # 3. Telemetry Preservation & State Segregation Tests
    # -------------------------------------------------------------------------
    def test_05_preservation_of_measured_telemetry(self):
        """Verify original measured telemetry G6–G26 is never overwritten by scenarios."""
        sc = create_thermal_stress_scenario(
            scenario_id="SCEN_TEL_PRESERVE",
            target="T30",
            offset_degR=35.0,
        )
        res = self.runner.run_comparison(
            subset="FD001", unit_number=1, split="train", scenario=sc, max_cycles=10
        )

        for b_state, a_state in zip(res.baseline_records, res.altered_records):
            # Ground truth measured values must be identical
            self.assertEqual(b_state.T30, a_state.T30)
            self.assertEqual(b_state.P30, a_state.P30)
            self.assertEqual(b_state.T24, a_state.T24)
            self.assertEqual(b_state.Nc, a_state.Nc)

    def test_06_correct_g39_g40_population(self):
        """Verify G39 is populated only for thermal stress, G40 only for sensor bias."""
        sc_ts = create_thermal_stress_scenario(scenario_id="TS_01", target="T30", offset_degR=25.0)
        sc_sb = create_sensor_bias_scenario(scenario_id="SB_01", target="P30", bias=10.0, unit="psia")

        res_ts = self.runner.run_comparison(subset="FD001", unit_number=1, split="train", scenario=sc_ts, max_cycles=5)
        res_sb = self.runner.run_comparison(subset="FD001", unit_number=1, split="train", scenario=sc_sb, max_cycles=5)

        # Thermal stress state
        s_ts = res_ts.altered_records[-1]
        self.assertEqual(s_ts.thermal_stress_offset, 25.0)
        self.assertIsNone(s_ts.sensor_bias_delta)
        self.assertEqual(s_ts.scenario_id, "TS_01")

        # Sensor bias state
        s_sb = res_sb.altered_records[-1]
        self.assertIsNone(s_sb.thermal_stress_offset)
        self.assertEqual(s_sb.sensor_bias_delta, {"P30": 10.0})
        self.assertEqual(s_sb.scenario_id, "SB_01")

    def test_07_correct_g30_g31_scenario_input_behavior(self):
        """
        Verify that G30 and G31 in the altered simulator differ from the baseline
        on the modified channel while remaining identical on unperturbed channels.
        """
        sc = create_thermal_stress_scenario(
            scenario_id="SCEN_FEAT_DIFF",
            target="T30",
            offset_degR=35.0,
        )
        res = self.runner.run_comparison(
            subset="FD001", unit_number=1, split="train", scenario=sc, max_cycles=5
        )

        b_state = res.baseline_records[-1]
        a_state = res.altered_records[-1]

        # G30 normalized vectors must NOT be identical
        self.assertFalse(np.array_equal(b_state.normalized_feature_vec, a_state.normalized_feature_vec))
        # G31 sequence tensors must NOT be identical
        self.assertFalse(np.array_equal(b_state.temporal_seq_tensor, a_state.temporal_seq_tensor))

    # -------------------------------------------------------------------------
    # 4. Prognostic ML Model Inference Tests
    # -------------------------------------------------------------------------
    def test_08_xgboost_and_lstm_inference(self):
        """Verify XGBoost and LSTM predictions exist and are valid numbers across all cycles."""
        sc = create_sensor_bias_scenario(
            scenario_id="SCEN_INF_CHECK",
            target="P30",
            bias=12.0,
            unit="psia",
        )
        res = self.runner.run_comparison(
            subset="FD001", unit_number=1, split="train", scenario=sc, max_cycles=10
        )
        df = res.comparison_df

        self.assertEqual(len(df), 10)
        self.assertTrue(df["baseline_rul_xgb"].notnull().all())
        self.assertTrue(df["altered_rul_xgb"].notnull().all())
        self.assertTrue(df["baseline_rul_lstm"].notnull().all())
        self.assertTrue(df["altered_rul_lstm"].notnull().all())

    def test_09_all_outputs_finite(self):
        """Verify no NaN, +inf, or -inf occurs in predictions, spreads, deltas, feature vectors, or sequence tensors."""
        sc = create_thermal_stress_scenario(
            scenario_id="SCEN_FINITE_CHECK",
            target="T30",
            offset_degR=35.0,
        )
        res = self.runner.run_comparison(
            subset="FD001", unit_number=1, split="train", scenario=sc, max_cycles=10
        )
        df = res.comparison_df

        # 1. Check predictions, spreads, and deltas in tabular output
        for col in ["baseline_rul_xgb", "altered_rul_xgb", "delta_rul_xgb",
                    "baseline_rul_lstm", "altered_rul_lstm", "delta_rul_lstm",
                    "baseline_spread", "altered_spread", "delta_spread"]:
            self.assertTrue(np.isfinite(df[col].to_numpy()).all(), f"Column {col} contains non-finite values.")

        # 2. Check normalized feature vectors (G30) and temporal sequence tensors (G31) in recorded states
        for b_state, a_state in zip(res.baseline_records, res.altered_records):
            self.assertIsNotNone(b_state.normalized_feature_vec)
            self.assertIsNotNone(a_state.normalized_feature_vec)
            self.assertTrue(np.all(np.isfinite(b_state.normalized_feature_vec)))
            self.assertTrue(np.all(np.isfinite(a_state.normalized_feature_vec)))

            self.assertIsNotNone(b_state.temporal_seq_tensor)
            self.assertIsNotNone(a_state.temporal_seq_tensor)
            self.assertTrue(np.all(np.isfinite(b_state.temporal_seq_tensor)))
            self.assertTrue(np.all(np.isfinite(a_state.temporal_seq_tensor)))

    # -------------------------------------------------------------------------
    # 5. Reproducibility & State Isolation Tests
    # -------------------------------------------------------------------------
    def test_10_experiment_reproducibility(self):
        """Verify running the comparison twice with identical parameters produces identical results."""
        sc = create_thermal_stress_scenario(
            scenario_id="SCEN_REPRO_TEST",
            target="T30",
            offset_degR=35.0,
        )
        res1 = self.runner.run_comparison(
            subset="FD001", unit_number=1, split="train", scenario=sc, max_cycles=10
        )
        res2 = self.runner.run_comparison(
            subset="FD001", unit_number=1, split="train", scenario=sc, max_cycles=10
        )

        np.testing.assert_allclose(
            res1.comparison_df["delta_rul_xgb"].to_numpy(),
            res2.comparison_df["delta_rul_xgb"].to_numpy(),
            rtol=1e-7, atol=1e-7,
        )
        np.testing.assert_allclose(
            res1.comparison_df["delta_rul_lstm"].to_numpy(),
            res2.comparison_df["delta_rul_lstm"].to_numpy(),
            rtol=1e-7, atol=1e-7,
        )

    def test_11_separation_of_baseline_and_altered_results(self):
        """Verify running an altered simulation does not leak state into a fresh baseline simulator."""
        sc = create_thermal_stress_scenario(scenario_id="SCEN_LEAK", target="T30", offset_degR=40.0)
        sim_alt = DigitalTwinSimulator(subset="FD001", data_dir=self.data_dir, enable_ml_inference=True, scenario=sc)
        sim_alt.load_engine(unit_number=1, split="train")
        sim_alt.step()

        # Create new baseline simulator
        sim_base = DigitalTwinSimulator(subset="FD001", data_dir=self.data_dir, enable_ml_inference=True)
        s_base = sim_base.load_engine(unit_number=1, split="train")

        self.assertIsNone(sim_base.active_scenario)
        self.assertIsNone(s_base.thermal_stress_offset)
        self.assertIsNone(s_base.scenario_id)

    def test_12_accelerated_aging_remains_deferred(self):
        """Verify attempting to run accelerated aging in comparison runner raises NotImplementedError."""
        sc_aging = create_accelerated_aging_scenario(
            scenario_id="SCEN_AA_REJECT",
            aging_multiplier=2.0,
        )
        with self.assertRaises(NotImplementedError) as ctx:
            self.runner.run_comparison(
                subset="FD001", unit_number=1, split="train", scenario=sc_aging
            )
        self.assertIn("Accelerated aging execution is formally deferred", str(ctx.exception))

    def test_13_artifact_export_and_files_exist(self):
        """Verify artifact exporter writes valid non-empty files."""
        sc = create_thermal_stress_scenario(scenario_id="SCEN_EXP_TEST", target="T30", offset_degR=20.0)
        res = self.runner.run_comparison(subset="FD001", unit_number=1, split="train", scenario=sc, max_cycles=5)

        test_out_dir = os.path.join("reports", "digital_twin", "step5_8", "test_scratch")
        artifacts = ScenarioComparisonRunner.export_artifacts(res, output_dir=test_out_dir, file_prefix="test_run")

        for key, path in artifacts.items():
            self.assertTrue(os.path.exists(path), f"Artifact {key} at {path} does not exist.")
            self.assertGreater(os.path.getsize(path), 0, f"Artifact {key} at {path} is empty.")

        # Cleanup scratch artifacts
        for path in artifacts.values():
            if os.path.exists(path):
                os.remove(path)
        if os.path.exists(test_out_dir):
            os.rmdir(test_out_dir)

    # -------------------------------------------------------------------------
    # 6. Before-and-After Test-Session Immutability Safeguards
    # -------------------------------------------------------------------------
    def test_14_raw_dataset_before_and_after_immutability(self):
        """Verify before-and-after test-session immutability of all raw C-MAPSS dataset files."""
        for fname, original_hash in self.raw_file_hashes.items():
            fpath = os.path.join(self.data_dir, fname)
            with open(fpath, "rb") as f:
                current_hash = hashlib.sha256(f.read()).hexdigest()
            self.assertEqual(
                current_hash,
                original_hash,
                f"Raw dataset file '{fname}' was modified! SHA-256 mismatch.",
            )

    def test_15_model_scalers_before_and_after_immutability(self):
        """Verify before-and-after test-session immutability of all 4 Phase 3 model scalers."""
        self.assertEqual(len(self.scaler_hashes), 4, "Expected exactly 4 scalers.")
        for fname, original_hash in self.scaler_hashes.items():
            fpath = os.path.join(self.scalers_dir, fname)
            with open(fpath, "rb") as f:
                current_hash = hashlib.sha256(f.read()).hexdigest()
            self.assertEqual(
                current_hash,
                original_hash,
                f"Model scaler file '{fname}' was modified! SHA-256 mismatch.",
            )

    def test_16_champion_checkpoints_before_and_after_immutability(self):
        """Verify before-and-after test-session immutability of all 8 Phase 4 champion checkpoints."""
        self.assertEqual(len(self.ckpt_hashes), 8, "Expected exactly 8 champion checkpoints.")
        for fname, original_hash in self.ckpt_hashes.items():
            fpath = os.path.join(self.ckpt_dir, fname)
            with open(fpath, "rb") as f:
                current_hash = hashlib.sha256(f.read()).hexdigest()
            self.assertEqual(
                current_hash,
                original_hash,
                f"Champion checkpoint file '{fname}' was modified! SHA-256 mismatch.",
            )

    # -------------------------------------------------------------------------
    # 7. Direct Feature Perturbation & Strengthened Fairness Checks
    # -------------------------------------------------------------------------
    def test_17_direct_p30_sensor_bias_perturbation(self):
        """
        Directly verify P30 sensor bias scenario application on a known 24-element raw feature vector.
        Confirms P30 maps to sensor_7 at channel_index=9, modifies only that channel by +15.0 psia,
        leaves sensor_8 (index 10) and all other 22 features unchanged, preserves original raw input,
        and accurately records baseline and simulated values in the audit record.
        """
        from digital_twin.scenario_interface import ScenarioModifier, create_sensor_bias_scenario

        # 1. Create a P30 sensor-bias scenario of +15.0 psia
        sc = create_sensor_bias_scenario(
            scenario_id="SCEN_DIRECT_P30_CHECK",
            target="P30",
            bias=15.0,
            unit="psia",
        )

        # 2. Known 24-element raw feature vector:
        # [op1, op2, op3, s1, s2, s3, s4, s5, s6, s7(P30), s8(Nf), s9, s10, s11, s12, s13, s14, s15, s16, s17, s18, s19, s20, s21]
        raw_vec = np.array([
            0.0, 0.0, 100.0,
            518.67, 642.0, 1580.0, 1400.0, 14.62, 21.61, 553.0, 2388.0,
            9050.0, 1.30, 47.5, 521.5, 2388.0, 8130.0, 8.4, 0.03, 392.0,
            2388.0, 100.0, 38.8, 23.3
        ], dtype=float)
        raw_vec_copy = np.copy(raw_vec)

        sim_vec, audit = ScenarioModifier.apply_scenario(raw_vec, sc)

        # 3. Confirms the audit reports feature_column="sensor_7"
        self.assertEqual(audit["feature_column"], "sensor_7")

        # 4. Confirms the audit reports channel_index=9
        self.assertEqual(audit["channel_index"], 9)

        # 5. Confirms the simulated sensor_7 value equals the original sensor_7 value + 15.0
        self.assertAlmostEqual(sim_vec[9], raw_vec[9] + 15.0, places=7)
        self.assertAlmostEqual(sim_vec[9], 568.0, places=7)

        # 6. Confirms sensor_8 (channel_index=10) and every other feature remain unchanged
        self.assertEqual(sim_vec[10], raw_vec[10])  # sensor_8 (Nf)
        for idx in range(24):
            if idx != 9:
                self.assertEqual(sim_vec[idx], raw_vec[idx], f"Feature at channel index {idx} was modified!")

        # 7. Confirms the original raw input array remains unchanged
        np.testing.assert_array_equal(raw_vec, raw_vec_copy)

        # 8. Confirms the audit records the original and simulated values correctly
        self.assertEqual(audit["target_parameter"], "P30")
        self.assertAlmostEqual(audit["original_baseline_value"], 553.0, places=7)
        self.assertAlmostEqual(audit["applied_delta"], 15.0, places=7)
        self.assertAlmostEqual(audit["resulting_simulated_value"], 568.0, places=7)
        self.assertAlmostEqual(audit["baseline_value"], 553.0, places=7)
        self.assertAlmostEqual(audit["applied_modification"], 15.0, places=7)
        self.assertAlmostEqual(audit["simulated_value"], 568.0, places=7)

    def test_18_fairness_checks_and_audit_verification(self):
        """
        Verify that ScenarioComparisonRunner performs all fairness checks, validates cycle IDs,
        verifies scenario audits across successful execution.
        """
        sc = create_thermal_stress_scenario(
            scenario_id="SCEN_FAIR_CHECK",
            target="T30",
            offset_degR=35.0,
        )
        res = self.runner.run_comparison(
            subset="FD001", unit_number=1, split="train", scenario=sc, max_cycles=10
        )

        # Ensure all required checks exist and evaluate to True
        required_keys = [
            "identical_initial_conditions",
            "identical_cycle_count",
            "identical_cycle_ids",
            "all_requested_cycles_evaluated",
            "unmodified_measured_telemetry",
            "unmodified_macroscopic_indicators",
            "baseline_scenario_fields_unset",
            "altered_scenario_fields_populated",
            "scenario_audit_verified",
            "all_outputs_finite",
        ]
        for k in required_keys:
            self.assertIn(k, res.fair_comparison_checks)
            self.assertTrue(res.fair_comparison_checks[k], f"Check {k} was not True!")

    def test_19_fairness_violation_raises_runtime_error(self):
        """
        Deliberately cause a fairness invariant (unmodified_measured_telemetry) to fail via controlled
        method interception, and verify that ScenarioComparisonRunner raises RuntimeError.
        Guarantees that fair comparison failures are never silently ignored.
        """
        from unittest.mock import patch

        sc = create_thermal_stress_scenario(
            scenario_id="SCEN_FAIL_PATH",
            target="T30",
            offset_degR=35.0,
        )

        orig_step = DigitalTwinSimulator.step

        def tampered_step(sim_self, *args, **kwargs):
            state = orig_step(sim_self, *args, **kwargs)
            # Deliberately violate telemetry invariance on cycle 2 of altered twin
            if state is not None and sim_self.active_scenario is not None and sim_self.current_step_idx == 2:
                state.T30 += 5.0
            return state

        with patch.object(DigitalTwinSimulator, "step", side_effect=tampered_step, autospec=True):
            with self.assertRaises(RuntimeError) as ctx:
                self.runner.run_comparison(
                    subset="FD001",
                    unit_number=1,
                    split="train",
                    scenario=sc,
                    max_cycles=5,
                )
            self.assertIn("Fair comparison failure", str(ctx.exception))
            self.assertIn("unmodified_measured_telemetry", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
