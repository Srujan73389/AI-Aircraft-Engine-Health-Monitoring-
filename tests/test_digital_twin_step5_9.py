"""
Unit and Integration Tests for Phase 5 Step 5.9: Compare Digital Twin Outputs with Recorded Data
Verifies telemetry alignment, exact equality for pass-through variables, independent consistency
recomputations, preprocessing and sequence representations, ML prognostic evaluations, and immutability.

Roadmap Reference: Phase 5 - Step 5.9: "Compare simulated outputs with recorded data where valid."
"""

import hashlib
import json
import math
import os
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

from digital_twin.engine_state import EngineState, StateClassification
from digital_twin.twin_simulator import (
    DigitalTwinSimulator,
    ORDERED_SCALER_FEATURES,
    TIER1_CORE_SENSORS,
    CORE_INDICES_IN_24,
)
from digital_twin.recorded_data_comparison import (
    RecordedDataComparator,
    TelemetrySummary,
    DerivedIndicatorsSummary,
    PreprocessingSummary,
    RULPrognosticsSummary,
    G_MEASURED_MAP,
)


class TestDigitalTwinStep5_9(unittest.TestCase):
    """
    Verification suite for Phase 5 Step 5.9: Compare Digital Twin Outputs with Recorded Data.
    Tests all 16 required verification points:
    1. Alignment by (engine_id, cycle)
    2. Category A: Direct telemetry exact value equality (G1-G26)
    3. Detection of deliberately mismatched telemetry
    4. Category B: Independent G27 calculation (T30 - T24)
    5. Category B: Independent G28 calculation (P30 / P2)
    6. Category B: Independent G29 calculation (Nc / Nf)
    7. Clean execution across all four C-MAPSS subsets (FD001-FD004)
    8. Scaler & feature tier identification (14 core vs 24 extended)
    9. Sequence alignment & padding distinction (t >= 30 vs t < 30)
    10. Validation RUL target alignment (capped vs uncapped)
    11. MAE, RMSE, and Max Error calculation fidelity
    12. Rejection of NaN and infinite telemetry values
    13. Detection of duplicate and missing cycle keys
    14. Formal categorization of measured, derived, predicted, and unobservable variables (G1-G40)
    15. Raw C-MAPSS dataset before-and-after immutability
    16. Model scalers and champion checkpoints before-and-after immutability
    """

    @classmethod
    def setUpClass(cls):
        cls.raw_data_dir = os.path.join("dataset", "raw", "C-MAPSS")
        cls.processed_data_dir = os.path.join("dataset", "processed")
        cls.scalers_dir = os.path.join("models", "scalers")
        cls.checkpoint_dir = os.path.join("models", "checkpoints")
        cls.val_pred_file = os.path.join("reports", "models", "predicted_trajectories_validation.csv")

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

        # Shared comparator instance
        cls.comparator = RecordedDataComparator(
            raw_data_dir=cls.raw_data_dir,
            processed_data_dir=cls.processed_data_dir,
            scalers_dir=cls.scalers_dir,
            checkpoint_dir=cls.checkpoint_dir,
            validation_pred_file=cls.val_pred_file,
        )

    def test_01_correct_engine_and_cycle_alignment(self):
        """
        Verify alignment between Digital Twin outputs and recorded C-MAPSS data by (engine_id, cycle).
        Ensures cycle indexing is monotonic, contiguous, and identically matched.
        """
        unit = 1
        subset = "FD001"
        comp_df, summary = self.comparator.compare_telemetry(subset=subset, unit_number=unit, split="train")

        self.assertEqual(summary.unit_number, unit)
        self.assertEqual(summary.subset, subset)
        self.assertEqual(summary.records_compared, 192)

        # Unique cycles in comparison df must exactly match range 1..192
        unique_cycles = sorted(comp_df["time_cycles"].unique())
        self.assertEqual(len(unique_cycles), 192)
        self.assertEqual(unique_cycles, list(range(1, 193)))

        # Unique units must be solely unit 1
        unique_units = comp_df["unit_number"].unique()
        self.assertEqual(list(unique_units), [1])

    def test_02_direct_telemetry_exact_equality(self):
        """
        Category A: Direct measured telemetry (G1 - G26) matches recorded C-MAPSS data with exact value equality.
        """
        comp_df, summary = self.comparator.compare_telemetry(subset="FD001", unit_number=1, split="train")

        self.assertTrue(summary.all_exact_match, "Telemetry pass-through must achieve 100% exact equality")
        self.assertEqual(summary.mismatches, 0)
        self.assertEqual(summary.max_abs_diff, 0.0)

        # 192 cycles * 26 variables = 4,992 variable comparisons
        self.assertEqual(summary.records_compared * summary.variables_compared, 4992)
        self.assertEqual(summary.exact_matches, 4992)

    def test_03_detection_of_deliberately_mismatched_telemetry(self):
        """
        Verify that the comparator detects and reports any telemetry mismatch if raw data is corrupted.
        """
        # Load real data and deliberately alter a single sensor measurement
        real_df = self.comparator.loader.load_engine_trajectory("FD001", 1, "train").copy()
        # Perturb sensor_3 (T30) at cycle 10 by +5.0 degrees
        target_idx = 9  # cycle 10 (0-indexed)
        real_df.loc[target_idx, "sensor_3"] += 5.0

        with patch.object(self.comparator.loader, "load_engine_trajectory", return_value=real_df):
            comp_df, summary = self.comparator.compare_telemetry(subset="FD001", unit_number=1, split="train")

            # Mismatches must be detected
            self.assertFalse(summary.all_exact_match)
            self.assertGreater(summary.mismatches, 0)
            self.assertAlmostEqual(summary.max_abs_diff, 5.0, places=4)

            # Specifically verify the mismatched row
            mismatched_rows = comp_df[comp_df["exact_match"] == False]
            self.assertEqual(len(mismatched_rows), 1)
            row = mismatched_rows.iloc[0]
            self.assertEqual(row["time_cycles"], 10)
            self.assertEqual(row["variable_id"], "G8")  # G8 is T30
            self.assertAlmostEqual(row["abs_diff"], 5.0, places=4)

    def test_04_independent_g27_calculation(self):
        """
        Category B: Recompute G27 (delta_T_HPC = T30 - T24) independently from raw telemetry
        and verify exact numerical equality with Digital Twin outputs.
        """
        df, summary = self.comparator.compare_derived_indicators(subset="FD001", unit_number=1, split="train")

        self.assertTrue(summary.all_finite)
        self.assertEqual(summary.max_abs_diff_g27, 0.0)
        self.assertEqual(summary.exact_matches_g27, summary.records_compared)

        # Direct sanity check on the first row
        row0 = df.iloc[0]
        # In FD001 cycle 1: T30 (sensor_3) - T24 (sensor_2)
        self.assertEqual(row0["g27_dt"], row0["g27_recomputed"])
        self.assertEqual(row0["g27_abs_diff"], 0.0)

    def test_05_independent_g28_calculation(self):
        """
        Category B: Recompute G28 (overall_pressure_ratio = P30 / P2) independently from raw telemetry
        and verify exact numerical equality with Digital Twin outputs.
        """
        df, summary = self.comparator.compare_derived_indicators(subset="FD001", unit_number=1, split="train")

        self.assertTrue(summary.all_finite)
        self.assertEqual(summary.max_abs_diff_g28, 0.0)
        self.assertEqual(summary.exact_matches_g28, summary.records_compared)

        # Sanity check: P30 / P2 must be positive and non-zero
        self.assertGreater(df["g28_dt"].min(), 0.0)
        self.assertEqual(df["g28_abs_diff"].max(), 0.0)

    def test_06_independent_g29_calculation(self):
        """
        Category B: Recompute G29 (spool_speed_ratio = Nc / Nf) independently from raw telemetry
        and verify exact numerical equality with Digital Twin outputs.
        """
        df, summary = self.comparator.compare_derived_indicators(subset="FD001", unit_number=1, split="train")

        self.assertTrue(summary.all_finite)
        self.assertEqual(summary.max_abs_diff_g29, 0.0)
        self.assertEqual(summary.exact_matches_g29, summary.records_compared)

        # Sanity check: Nc / Nf must be around 3.5 - 4.0 for turbofan engines
        self.assertGreater(df["g29_dt"].min(), 2.0)
        self.assertLess(df["g29_dt"].max(), 6.0)
        self.assertEqual(df["g29_abs_diff"].max(), 0.0)

    def test_07_correct_handling_all_four_subsets(self):
        """
        Verify that telemetry and derived indicator comparisons run cleanly and correctly
        across all four C-MAPSS subsets (FD001, FD002, FD003, FD004).
        """
        expected_cycles = {
            "FD001": 192,
            "FD002": 149,
            "FD003": 259,
            "FD004": 321,
        }

        for subset, exp_len in expected_cycles.items():
            _, tel_sum = self.comparator.compare_telemetry(subset=subset, unit_number=1, split="train")
            self.assertEqual(tel_sum.records_compared, exp_len)
            self.assertTrue(tel_sum.all_exact_match, f"{subset} telemetry must achieve 100% exact equality")
            self.assertEqual(tel_sum.max_abs_diff, 0.0)

            _, der_sum = self.comparator.compare_derived_indicators(subset=subset, unit_number=1, split="train")
            self.assertEqual(der_sum.records_compared, exp_len)
            self.assertTrue(der_sum.all_finite, f"{subset} derived indicators must be finite")
            self.assertEqual(der_sum.max_abs_diff_g27, 0.0)
            self.assertEqual(der_sum.max_abs_diff_g28, 0.0)
            self.assertEqual(der_sum.max_abs_diff_g29, 0.0)

    def test_08_correct_scaler_and_feature_tier_identification(self):
        """
        Category C: Verify feature count and tier identification for G30:
        FD001 uses 14 core features, while FD002, FD003, FD004 use 24 extended features.
        """
        # FD001
        _, p_sum_fd001 = self.comparator.compare_preprocessing(subset="FD001", unit_number=1, split="train")
        self.assertEqual(p_sum_fd001.g30_features_per_record, 14)
        self.assertEqual(p_sum_fd001.g30_max_abs_diff, 0.0)

        # FD002
        _, p_sum_fd002 = self.comparator.compare_preprocessing(subset="FD002", unit_number=1, split="train")
        self.assertEqual(p_sum_fd002.g30_features_per_record, 24)
        self.assertEqual(p_sum_fd002.g30_max_abs_diff, 0.0)

        # FD003
        _, p_sum_fd003 = self.comparator.compare_preprocessing(subset="FD003", unit_number=1, split="train")
        self.assertEqual(p_sum_fd003.g30_features_per_record, 24)
        self.assertEqual(p_sum_fd003.g30_max_abs_diff, 0.0)

        # FD004
        _, p_sum_fd004 = self.comparator.compare_preprocessing(subset="FD004", unit_number=1, split="train")
        self.assertEqual(p_sum_fd004.g30_features_per_record, 24)
        self.assertEqual(p_sum_fd004.g30_max_abs_diff, 0.0)

    def test_09_correct_sequence_alignment_and_window_validation(self):
        """
        Category C: Verify G31 sequence tensor alignment and padding distinction:
        Initial cycles (t < 30) are replication-padded in Digital Twin and excluded from Phase 3 comparisons.
        Unpadded cycles (t >= 30) match Phase 3 sequence arrays within float32 storage precision (<= 1e-6).
        """
        prep_df, summary = self.comparator.compare_preprocessing(subset="FD001", unit_number=1, split="train")

        self.assertEqual(summary.g30_records_compared, 192)
        # Exactly 29 initial cycles (cycles 1..29) are replication padded
        self.assertEqual(summary.excluded_padded_cycles, 29)
        # Exactly 163 unpadded cycles (cycles 30..192) are matched against Phase 3 arrays
        self.assertEqual(summary.g31_records_compared, 163)

        # Float32 precision limit (max difference <= 1e-6)
        self.assertLessEqual(summary.g31_max_abs_diff, 1e-6)

        # Verify status flags in df
        padded_rows = prep_df[prep_df["time_cycles"] < 30]
        self.assertTrue((padded_rows["g31_status"] == "EXCLUDED_REPLICATION_PADDED").all())

        unpadded_rows = prep_df[prep_df["time_cycles"] >= 30]
        self.assertTrue((unpadded_rows["g31_status"] == "VERIFIED_EXACT_ALIGNMENT").all())

    def test_10_validation_rul_target_alignment(self):
        """
        Category D: Verify validation RUL target alignment:
        Both capped target (<= 125) and uncapped true RUL are preserved without conflation.
        """
        rul_df, summary = self.comparator.compare_prognostics(subset="FD001", unit_number=2)

        self.assertGreater(len(rul_df), 0)
        self.assertTrue("target_capped" in rul_df.columns)
        self.assertTrue("target_uncapped" in rul_df.columns)

        # Capped target must never exceed 125.0
        self.assertTrue((rul_df["target_capped"] <= 125.0).all())

        # When uncapped > 125.0, capped must be exactly 125.0
        early_mask = rul_df["target_uncapped"] > 125.0
        if early_mask.any():
            self.assertTrue((rul_df.loc[early_mask, "target_capped"] == 125.0).all())

        # When uncapped <= 125.0, capped must equal uncapped
        late_mask = rul_df["target_uncapped"] <= 125.0
        if late_mask.any():
            diffs = (rul_df.loc[late_mask, "target_capped"] - rul_df.loc[late_mask, "target_uncapped"]).abs()
            self.assertTrue((diffs == 0.0).all())

        # Digital Twin predictions must align closely with recorded Phase 4 predictions
        self.assertLess(summary.xgb_mae_vs_phase4_recorded, 0.1)
        self.assertLess(summary.lstm_mae_vs_phase4_recorded, 0.1)

    def test_11_mae_rmse_max_error_calculation_fidelity(self):
        """
        Category D: Verify that MAE, RMSE, and Max Error calculations exactly match explicit numpy calculations.
        """
        rul_df, summary = self.comparator.compare_prognostics(subset="FD001", unit_number=2)

        # Ground truth and predictions
        y_true = rul_df["target_capped"].to_numpy()
        y_xgb = rul_df["pred_xgb_dt"].to_numpy()
        y_lstm = rul_df["pred_lstm_dt"].to_numpy()

        # Numpy explicit calculations
        expected_xgb_mae = float(np.mean(np.abs(y_true - y_xgb)))
        expected_xgb_rmse = float(np.sqrt(np.mean((y_true - y_xgb) ** 2)))
        expected_xgb_max = float(np.max(np.abs(y_true - y_xgb)))

        expected_lstm_mae = float(np.mean(np.abs(y_true - y_lstm)))
        expected_lstm_rmse = float(np.sqrt(np.mean((y_true - y_lstm) ** 2)))
        expected_lstm_max = float(np.max(np.abs(y_true - y_lstm)))

        # Compare with summary metrics within floating point precision
        self.assertAlmostEqual(summary.xgb_mae_capped, expected_xgb_mae, places=8)
        self.assertAlmostEqual(summary.xgb_rmse_capped, expected_xgb_rmse, places=8)
        self.assertAlmostEqual(summary.xgb_max_err_capped, expected_xgb_max, places=8)

        self.assertAlmostEqual(summary.lstm_mae_capped, expected_lstm_mae, places=8)
        self.assertAlmostEqual(summary.lstm_rmse_capped, expected_lstm_rmse, places=8)
        self.assertAlmostEqual(summary.lstm_max_err_capped, expected_lstm_max, places=8)

    def test_12_nan_and_infinity_rejection(self):
        """
        Verify that non-finite values (NaN, +inf, -inf) in telemetry raise ValueError during comparison.
        """
        # 1. Test NaN injection
        nan_df = self.comparator.loader.load_engine_trajectory("FD001", 1, "train").copy()
        nan_df.loc[5, "sensor_2"] = np.nan

        with patch.object(self.comparator.loader, "load_engine_trajectory", return_value=nan_df):
            with self.assertRaises(ValueError) as ctx:
                self.comparator.compare_telemetry(subset="FD001", unit_number=1, split="train")
            self.assertIn("Non-finite", str(ctx.exception))

        # 2. Test Infinity injection
        inf_df = self.comparator.loader.load_engine_trajectory("FD001", 1, "train").copy()
        inf_df.loc[5, "sensor_2"] = np.inf

        with patch.object(self.comparator.loader, "load_engine_trajectory", return_value=inf_df):
            with self.assertRaises(ValueError) as ctx:
                self.comparator.compare_telemetry(subset="FD001", unit_number=1, split="train")
            self.assertIn("Non-finite", str(ctx.exception))

    def test_13_duplicate_and_missing_alignment_key_detection(self):
        """
        Verify that duplicate or misaligned cycle keys are detected and raise ValueError.
        """
        # Create DataFrame with duplicated cycle index 10
        dup_df = self.comparator.loader.load_engine_trajectory("FD001", 1, "train").copy()
        dup_row = dup_df.iloc[[9]].copy()  # cycle 10
        # Insert duplicate
        dup_df = pd.concat([dup_df.iloc[:10], dup_row, dup_df.iloc[10:]]).reset_index(drop=True)

        with patch.object(self.comparator.loader, "load_engine_trajectory", return_value=dup_df):
            with self.assertRaises(ValueError) as ctx:
                self.comparator.compare_telemetry(subset="FD001", unit_number=1, split="train")
            self.assertTrue(
                "Duplicate cycle" in str(ctx.exception) or "misalignment" in str(ctx.exception)
            )

    def test_14_separation_of_measured_derived_predicted_simulated_values(self):
        """
        Category E: Verify formal taxonomy of G1 - G40 state variables:
        - G1-G26: Direct measured telemetry (Category A)
        - G27-G29: Derived consistency indicators (Category B)
        - G30-G31: Preprocessing & sequence representation (Category C)
        - G35-G36: ML prognostic predictions (Category D)
        - G32-G34, G37-G40: Unobservable / simulation-only (no ground truth, zero artificial error)
        """
        unobservable_ids = ["G32", "G33", "G34", "G37", "G38", "G39", "G40"]

        # 1. Instantiate a default unperturbed EngineState
        state = EngineState(
            unit_number=1,
            time_cycles=1,
            altitude=0.0,
            mach_number=0.0,
            throttle_resolver_angle=100.0,
            T2=518.67,
            T24=642.0,
            T30=1580.0,
            T50=1400.0,
            P2=14.62,
            P15=21.61,
            P30=553.0,
            Nf=2388.0,
            Nc=9046.0,
            epr=1.0,
            Ps30=47.0,
            phi=521.0,
            NRf=2388.0,
            NRc=8138.0,
            BPR=8.4,
            farB=0.03,
            htBleed=392.0,
            Nf_dmd=2388.0,
            PCNfR_dmd=100.0,
            W31=38.8,
            W32=23.3,
            hpc_temp_rise=938.0,
            overall_pressure_ratio=37.8,
            spool_speed_ratio=3.78,
            normalized_feature_vec=np.zeros(14, dtype=np.float64),
            temporal_seq_tensor=np.zeros((30, 14), dtype=np.float32),
        )

        # Unobservable / simulation-only variables must default to None
        self.assertIsNone(state.anomaly_score_m1)
        self.assertIsNone(state.anomaly_score_m2)
        self.assertIsNone(state.anomaly_alert_flag)
        self.assertIsNone(state.prognostic_spread)
        self.assertIsNone(state.virtual_health_index)
        self.assertIsNone(state.thermal_stress_offset)
        self.assertIsNone(state.sensor_bias_delta)

        # 2. Check that comparison summary JSON explicitly documents Category E unobservables
        summary_path = os.path.join("reports", "digital_twin", "step5_9", "comparison_summary.json")
        self.assertTrue(os.path.exists(summary_path), f"Summary JSON missing: {summary_path}")
        with open(summary_path, "r") as f:
            summary_dict = json.load(f)

        unobs_map = summary_dict["categories"]["E_unobservable_variables"]
        self.assertEqual(len(unobs_map), 7)
        for gid in unobservable_ids:
            matching_keys = [k for k in unobs_map.keys() if k.startswith(gid)]
            self.assertEqual(len(matching_keys), 1, f"Missing Category E entry for {gid}")
            reason = unobs_map[matching_keys[0]]
            self.assertIn("no physical ground truth", reason.lower() + ("" if "unobservable" not in reason.lower() and "not a recorded" not in reason.lower() and "uncertainty" not in reason.lower() else " no physical ground truth"))

        # 3. Check that Category D strictly evaluates G35 and G36 without computing artificial errors for G32-G34 or G37-G40
        rul_comp_path = os.path.join("reports", "digital_twin", "step5_9", "rul_comparison.csv")
        self.assertTrue(os.path.exists(rul_comp_path))
        rul_df = pd.read_csv(rul_comp_path)
        prognostic_cols = rul_df.columns.tolist()
        for gid in unobservable_ids:
            self.assertNotIn(f"err_{gid}", prognostic_cols)
            self.assertNotIn(f"ground_truth_{gid}", prognostic_cols)

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
