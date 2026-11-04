"""
Unit and Integration Tests for Phase 5 Step 5.6: Machine Learning Model Integration
Verifies coupling of Phase 4 champion models (XGBoost & LSTM) into the Digital Twin.

Roadmap Reference: Phase 5 - Step 5.6
Phase 4 Reference: Steps 4.2, 4.3, 4.6, 4.8, 4.11
"""

import hashlib
import json
import math
import os
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
import torch

from digital_twin.engine_state import EngineState
from digital_twin.model_inference import (
    CHAMPION_MODEL_REGISTRY,
    PrognosticsModelManager,
    compute_window_summary_statistics,
    extract_flattened_window,
)
from digital_twin.twin_simulator import DigitalTwinSimulator


class TestDigitalTwinStep5_6(unittest.TestCase):
    """
    Comprehensive verification suite for Digital Twin ML model integration.
    """

    @classmethod
    def setUpClass(cls):
        cls.subsets = ["FD001", "FD002", "FD003", "FD004"]
        cls.ckpt_dir = os.path.join("models", "checkpoints")
        cls.scalers_dir = os.path.join("models", "scalers")

    # -------------------------------------------------------------------------
    # 1. Model Loading & Configuration Tests
    # -------------------------------------------------------------------------
    def test_01_checkpoint_loading_all_subsets(self):
        """1. Verify that champion XGBoost and LSTM models load for all 4 subsets."""
        for subset in self.subsets:
            mgr = PrognosticsModelManager(subset=subset, checkpoint_dir=self.ckpt_dir)
            self.assertTrue(mgr.is_loaded, f"Manager for {subset} should be marked loaded.")
            self.assertIsNotNone(mgr.xgb_model, f"XGBoost model for {subset} should not be None.")
            self.assertIsNotNone(mgr.lstm_model, f"LSTM model for {subset} should not be None.")

    def test_02_scaler_selection_all_subsets(self):
        """2. Verify correct scaler file selection and properties for all subsets."""
        for subset in self.subsets:
            sim = DigitalTwinSimulator(subset=subset, scalers_dir=self.scalers_dir)
            self.assertEqual(sim.subset, subset)
            self.assertIsNotNone(sim.scaling_pipeline)
            self.assertEqual(len(sim.scaling_pipeline.feature_columns), 24)
            self.assertEqual(sim.scaling_pipeline.scaler.__class__.__name__, "SafeMinMaxScaler")

    # -------------------------------------------------------------------------
    # 2. Feature Slicing & Dimensionality Tests
    # -------------------------------------------------------------------------
    def test_03_fd001_tier1_feature_selection(self):
        """3. Verify FD001 Tier 1 Core 14 feature selection."""
        sim = DigitalTwinSimulator(subset="FD001")
        s1 = sim.load_engine(unit_number=1, split="train")

        self.assertEqual(s1.feature_tier, "Tier1_Core14")
        self.assertEqual(s1.normalized_feature_vec.shape, (14,))
        self.assertEqual(s1.temporal_seq_tensor.shape, (30, 14))

    def test_04_fd002_fd004_tier3_feature_selection(self):
        """4. Verify FD002-FD004 Tier 3 Extended 24 feature selection."""
        for subset in ["FD002", "FD003", "FD004"]:
            sim = DigitalTwinSimulator(subset=subset)
            s1 = sim.load_engine(unit_number=1, split="train")

            self.assertEqual(s1.feature_tier, "Tier3_Extended24")
            self.assertEqual(s1.normalized_feature_vec.shape, (24,))
            self.assertEqual(s1.temporal_seq_tensor.shape, (30, 24))

    def test_05_xgboost_input_dimensions_all_subsets(self):
        """5. Verify XGBoost feature engineering dimensions across all subsets."""
        expected_dims = {
            "FD001": 70,   # 14 sensors x 5 summary statistics
            "FD002": 720,  # 30 cycles x 24 features flattened
            "FD003": 120,  # 24 features x 5 summary statistics
            "FD004": 720,  # 30 cycles x 24 features flattened
        }
        for subset, exp_dim in expected_dims.items():
            mgr = PrognosticsModelManager(subset=subset, checkpoint_dir=self.ckpt_dir)
            channels = 14 if subset == "FD001" else 24
            dummy_seq = np.zeros((30, channels), dtype=np.float32)

            feats = mgr.prepare_xgboost_features(dummy_seq)
            self.assertEqual(feats.shape, (1, exp_dim), f"Mismatched XGBoost dim for {subset}.")

    def test_06_lstm_input_dimensions_all_subsets(self):
        """6. Verify LSTM input tensor shapes across all subsets."""
        expected_dims = {
            "FD001": (1, 30, 14),
            "FD002": (1, 30, 24),
            "FD003": (1, 30, 24),
            "FD004": (1, 30, 24),
        }
        for subset, exp_shape in expected_dims.items():
            mgr = PrognosticsModelManager(subset=subset, checkpoint_dir=self.ckpt_dir)
            channels = exp_shape[2]
            dummy_seq = np.zeros((30, channels), dtype=np.float32)

            tensor = mgr.prepare_lstm_tensor(dummy_seq)
            self.assertEqual(tensor.shape, exp_shape, f"Mismatched LSTM tensor shape for {subset}.")

    # -------------------------------------------------------------------------
    # 3. Rolling Window & Sequence Padding Tests
    # -------------------------------------------------------------------------
    def test_07_rolling_window_handling(self):
        """7. Verify rolling window sliding at cycles >= 30."""
        sim = DigitalTwinSimulator(subset="FD001")
        sim.load_engine(unit_number=1, split="train")

        for _ in range(35):
            state = sim.step()

        self.assertEqual(state.time_cycles, 36)
        self.assertFalse(state.is_padded)
        self.assertEqual(state.padded_cycles_count, 0)
        self.assertEqual(state.temporal_seq_tensor.shape, (30, 14))

    def test_08_short_history_padding_and_inference(self):
        """8. Verify left-edge replication padding at cycles < 30 and successful inference."""
        sim = DigitalTwinSimulator(subset="FD001", enable_ml_inference=True)
        s1 = sim.load_engine(unit_number=1, split="train")

        self.assertEqual(s1.time_cycles, 1)
        self.assertTrue(s1.is_padded)
        self.assertEqual(s1.padded_cycles_count, 29)
        # Predictions must be valid on padded cycles
        self.assertIsNotNone(s1.predicted_rul_xgboost)
        self.assertIsNotNone(s1.predicted_rul_lstm)
        self.assertTrue(math.isfinite(s1.predicted_rul_xgboost))
        self.assertTrue(math.isfinite(s1.predicted_rul_lstm))

    # -------------------------------------------------------------------------
    # 4. Inference Execution & Output Storage Tests
    # -------------------------------------------------------------------------
    def test_09_successful_xgboost_inference_all_subsets(self):
        """9. Verify successful XGBoost prediction across all 4 subsets."""
        for subset in self.subsets:
            sim = DigitalTwinSimulator(subset=subset)
            s1 = sim.load_engine(unit_number=1, split="train")
            mgr = PrognosticsModelManager(subset=subset, checkpoint_dir=self.ckpt_dir)

            pred_xgb = mgr.predict_xgboost(s1.temporal_seq_tensor)
            self.assertIsInstance(pred_xgb, float)
            self.assertTrue(math.isfinite(pred_xgb))
            self.assertGreater(pred_xgb, 0.0)

    def test_10_successful_lstm_inference_all_subsets(self):
        """10. Verify successful LSTM prediction across all 4 subsets."""
        for subset in self.subsets:
            sim = DigitalTwinSimulator(subset=subset)
            s1 = sim.load_engine(unit_number=1, split="train")
            mgr = PrognosticsModelManager(subset=subset, checkpoint_dir=self.ckpt_dir)

            pred_lstm = mgr.predict_lstm(s1.temporal_seq_tensor)
            self.assertIsInstance(pred_lstm, float)
            self.assertTrue(math.isfinite(pred_lstm))
            self.assertGreater(pred_lstm, 0.0)

    def test_11_correct_storage_g35_and_g36(self):
        """11. Verify correct population of G35 (XGBoost RUL) and G36 (LSTM RUL)."""
        sim = DigitalTwinSimulator(subset="FD001", enable_ml_inference=True)
        s1 = sim.load_engine(unit_number=1, split="train")

        self.assertIsNotNone(s1.predicted_rul_xgboost)
        self.assertIsNotNone(s1.predicted_rul_lstm)
        self.assertIsInstance(s1.predicted_rul_xgboost, float)
        self.assertIsInstance(s1.predicted_rul_lstm, float)
        # Advance one cycle and verify state updates
        s2 = sim.step()
        self.assertIsNotNone(s2.predicted_rul_xgboost)
        self.assertIsNotNone(s2.predicted_rul_lstm)

    def test_12_correct_calculation_g37_model_spread(self):
        """12. Verify G37 prognostic spread formula: |G35 - G36|."""
        sim = DigitalTwinSimulator(subset="FD001", enable_ml_inference=True)
        s1 = sim.load_engine(unit_number=1, split="train")

        expected_spread = abs(s1.predicted_rul_xgboost - s1.predicted_rul_lstm)
        self.assertAlmostEqual(s1.prognostic_spread, expected_spread, places=6)
        self.assertGreaterEqual(s1.prognostic_spread, 0.0)

    # -------------------------------------------------------------------------
    # 5. Robustness, Repeatability & Validation Tests
    # -------------------------------------------------------------------------
    def test_13_rejection_of_invalid_sequence_dimensions(self):
        """13. Verify error handling for invalid sequence length (not 30) and channel dimensions."""
        mgr = PrognosticsModelManager(subset="FD001", checkpoint_dir=self.ckpt_dir)

        # Invalid window length (20 instead of 30)
        invalid_seq = np.zeros((20, 14), dtype=np.float32)
        with self.assertRaises(ValueError):
            mgr.prepare_xgboost_features(invalid_seq)
        with self.assertRaises(ValueError):
            mgr.prepare_lstm_tensor(invalid_seq)

        # Invalid channel dimension (24 instead of 14 for FD001)
        wrong_channels = np.zeros((30, 24), dtype=np.float32)
        with self.assertRaises(ValueError):
            mgr.prepare_lstm_tensor(wrong_channels)

    def test_14_correct_subset_selection_validation(self):
        """14. Verify that unsupported subset names raise clear ValueError."""
        with self.assertRaises(ValueError):
            PrognosticsModelManager(subset="FD005")
        with self.assertRaises(ValueError):
            PrognosticsModelManager(subset="INVALID")

    def test_15_repeatability_of_predictions_for_identical_inputs(self):
        """15. Verify deterministic, bitwise repeatable predictions for identical inputs."""
        mgr = PrognosticsModelManager(subset="FD001", checkpoint_dir=self.ckpt_dir)
        sim = DigitalTwinSimulator(subset="FD001")
        s1 = sim.load_engine(unit_number=1, split="train")
        seq = s1.temporal_seq_tensor

        pxgb_1, plstm_1, sp_1 = mgr.predict_prognostics(seq)
        pxgb_2, plstm_2, sp_2 = mgr.predict_prognostics(seq)

        self.assertEqual(pxgb_1, pxgb_2)
        self.assertEqual(plstm_1, plstm_2)
        self.assertEqual(sp_1, sp_2)

    def test_16_checkpoint_manifest_file_sizes(self):
        """16. Verify checkpoint file existence and file sizes against checkpoint_manifest.json."""
        manifest_path = os.path.join(self.ckpt_dir, "checkpoint_manifest.json")
        with open(manifest_path, "r", encoding="utf-8") as f:
            ckpt_manifest = json.load(f)

        for key, entry in ckpt_manifest.items():
            if key.startswith("_"):
                continue
            ckpt_file = os.path.join(self.ckpt_dir, entry["checkpoint_file"])
            self.assertTrue(os.path.exists(ckpt_file), f"Checkpoint missing: {ckpt_file}")
            actual_size = os.path.getsize(ckpt_file)
            self.assertEqual(
                actual_size,
                entry["file_size_bytes"],
                f"File size changed for {entry['checkpoint_file']}.",
            )

    # -------------------------------------------------------------------------
    # 6. Comparison Against Phase 4 Reproduction Pipeline
    # -------------------------------------------------------------------------
    def test_17_phase4_reproduction_comparison(self):
        """
        17. Compare Digital Twin simulation predictions against recorded Phase 4 validation trajectories.
        Target: FD001 Unit 2 across unpadded cycles 30 to 34 from predicted_trajectories_validation.csv.
        """
        val_csv_path = os.path.join("reports", "models", "predicted_trajectories_validation.csv")
        self.assertTrue(os.path.exists(val_csv_path), "Validation trajectories CSV missing.")

        df_val = pd.read_csv(val_csv_path)
        val_u2 = df_val[(df_val["subset"] == "FD001") & (df_val["unit_number"] == 2)]
        val_u2_c30_34 = val_u2[val_u2["time_cycles"].between(30, 34)].sort_values("time_cycles")

        # Run Digital Twin on FD001 Unit 2 with ML inference enabled
        sim = DigitalTwinSimulator(subset="FD001", enable_ml_inference=True)
        sim.load_engine(unit_number=2, split="train")

        twin_preds = {}
        for c in range(1, 35):
            state = sim.current_state if c == 1 else sim.step()
            if state.time_cycles >= 30:
                twin_preds[state.time_cycles] = {
                    "xgb": state.predicted_rul_xgboost,
                    "lstm": state.predicted_rul_lstm,
                }

        # Verify numerical match against Phase 4 rounded reporting precision (0.01 cycles)
        for _, row in val_u2_c30_34.iterrows():
            cyc = int(row["time_cycles"])
            rec_xgb = float(row["pred_xgb_champion"])
            rec_lstm = float(row["pred_lstm_champion"])

            twin_xgb = twin_preds[cyc]["xgb"]
            twin_lstm = twin_preds[cyc]["lstm"]

            self.assertAlmostEqual(
                twin_xgb,
                rec_xgb,
                delta=0.01,
                msg=f"Cycle {cyc} XGBoost prediction mismatch: Twin={twin_xgb:.4f} vs Recorded={rec_xgb:.4f}",
            )
            self.assertAlmostEqual(
                twin_lstm,
                rec_lstm,
                delta=0.01,
                msg=f"Cycle {cyc} LSTM prediction mismatch: Twin={twin_lstm:.4f} vs Recorded={rec_lstm:.4f}",
            )

    # -------------------------------------------------------------------------
    # 7. Additional Input Safeguards & Rejection Tests (Step 5.6 Correction Pass)
    # -------------------------------------------------------------------------
    def test_18_nan_input_rejection(self):
        """18. Verify that NaN values in sequence tensor raise clear ValueError."""
        mgr = PrognosticsModelManager(subset="FD001", checkpoint_dir=self.ckpt_dir)
        nan_seq = np.zeros((30, 14), dtype=np.float32)
        nan_seq[15, 7] = float("nan")

        with self.assertRaises(ValueError) as ctx:
            mgr.validate_sequence_input(nan_seq)
        self.assertIn("NaN", str(ctx.exception))

        with self.assertRaises(ValueError):
            mgr.prepare_xgboost_features(nan_seq)
        with self.assertRaises(ValueError):
            mgr.prepare_lstm_tensor(nan_seq)
        with self.assertRaises(ValueError):
            mgr.predict_xgboost(nan_seq)
        with self.assertRaises(ValueError):
            mgr.predict_lstm(nan_seq)

    def test_19_positive_infinity_input_rejection(self):
        """19. Verify that +inf in sequence tensor raises clear ValueError."""
        mgr = PrognosticsModelManager(subset="FD001", checkpoint_dir=self.ckpt_dir)
        posinf_seq = np.zeros((30, 14), dtype=np.float32)
        posinf_seq[5, 2] = float("inf")

        with self.assertRaises(ValueError) as ctx:
            mgr.validate_sequence_input(posinf_seq)
        self.assertIn("+inf", str(ctx.exception))

        with self.assertRaises(ValueError):
            mgr.prepare_xgboost_features(posinf_seq)
        with self.assertRaises(ValueError):
            mgr.prepare_lstm_tensor(posinf_seq)

    def test_20_negative_infinity_input_rejection(self):
        """20. Verify that -inf in sequence tensor raises clear ValueError."""
        mgr = PrognosticsModelManager(subset="FD001", checkpoint_dir=self.ckpt_dir)
        neginf_seq = np.zeros((30, 14), dtype=np.float32)
        neginf_seq[10, 4] = float("-inf")

        with self.assertRaises(ValueError) as ctx:
            mgr.validate_sequence_input(neginf_seq)
        self.assertIn("-inf", str(ctx.exception))

        with self.assertRaises(ValueError):
            mgr.prepare_xgboost_features(neginf_seq)
        with self.assertRaises(ValueError):
            mgr.prepare_lstm_tensor(neginf_seq)

    def test_21_invalid_channel_dimensions(self):
        """21. Verify that mismatched channel counts raise clear ValueError."""
        mgr_001 = PrognosticsModelManager(subset="FD001", checkpoint_dir=self.ckpt_dir)
        mgr_002 = PrognosticsModelManager(subset="FD002", checkpoint_dir=self.ckpt_dir)

        # FD001 expects 14 channels, pass 24
        wrong_for_001 = np.zeros((30, 24), dtype=np.float32)
        with self.assertRaises(ValueError) as ctx:
            mgr_001.validate_sequence_input(wrong_for_001)
        self.assertIn("14 channels, got 24", str(ctx.exception))

        # FD002 expects 24 channels, pass 14
        wrong_for_002 = np.zeros((30, 14), dtype=np.float32)
        with self.assertRaises(ValueError) as ctx:
            mgr_002.validate_sequence_input(wrong_for_002)
        self.assertIn("24 channels, got 14", str(ctx.exception))

    def test_22_input_shape_2d_and_30_cycles_validation(self):
        """22. Verify that non-2D or non-30-cycle inputs raise clear ValueError/TypeError."""
        mgr = PrognosticsModelManager(subset="FD001", checkpoint_dir=self.ckpt_dir)

        # 1D array
        seq_1d = np.zeros(30, dtype=np.float32)
        with self.assertRaises(ValueError) as ctx:
            mgr.validate_sequence_input(seq_1d)
        self.assertIn("2-dimensional", str(ctx.exception))

        # 3D array
        seq_3d = np.zeros((1, 30, 14), dtype=np.float32)
        with self.assertRaises(ValueError) as ctx:
            mgr.validate_sequence_input(seq_3d)
        self.assertIn("2-dimensional", str(ctx.exception))

        # Wrong time steps (25 instead of 30)
        seq_25 = np.zeros((25, 14), dtype=np.float32)
        with self.assertRaises(ValueError) as ctx:
            mgr.validate_sequence_input(seq_25)
        self.assertIn("30 time steps", str(ctx.exception))

        # Non-numpy type
        with self.assertRaises(TypeError):
            mgr.validate_sequence_input([[0.0] * 14] * 30)

    def test_23_non_finite_xgboost_output_rejection(self):
        """23. Verify that non-finite XGBoost output raises ValueError (tested via mock)."""
        mgr = PrognosticsModelManager(subset="FD001", checkpoint_dir=self.ckpt_dir)
        valid_seq = np.zeros((30, 14), dtype=np.float32)

        # Mock XGBoost returning NaN
        with patch.object(mgr.xgb_model, "predict", return_value=np.array([float("nan")])):
            with self.assertRaises(ValueError) as ctx:
                mgr.predict_xgboost(valid_seq)
            self.assertIn("non-finite", str(ctx.exception))

        # Mock XGBoost returning +Inf
        with patch.object(mgr.xgb_model, "predict", return_value=np.array([float("inf")])):
            with self.assertRaises(ValueError) as ctx:
                mgr.predict_xgboost(valid_seq)
            self.assertIn("non-finite", str(ctx.exception))

    def test_24_non_finite_lstm_output_rejection(self):
        """24. Verify that non-finite LSTM output raises ValueError (tested via mock)."""
        mgr = PrognosticsModelManager(subset="FD001", checkpoint_dir=self.ckpt_dir)
        valid_seq = np.zeros((30, 14), dtype=np.float32)

        # Mock LSTM returning NaN
        with patch.object(mgr.lstm_model, "forward", return_value=torch.tensor([float("nan")])):
            with self.assertRaises(ValueError) as ctx:
                mgr.predict_lstm(valid_seq)
            self.assertIn("non-finite", str(ctx.exception))

        # Mock LSTM returning -Inf
        with patch.object(mgr.lstm_model, "forward", return_value=torch.tensor([float("-inf")])):
            with self.assertRaises(ValueError) as ctx:
                mgr.predict_lstm(valid_seq)
            self.assertIn("non-finite", str(ctx.exception))

    def test_25_finite_g37_spread_validation(self):
        """25. Verify that G37 prognostic spread is strictly finite and non-negative."""
        mgr = PrognosticsModelManager(subset="FD001", checkpoint_dir=self.ckpt_dir)
        valid_seq = np.zeros((30, 14), dtype=np.float32)

        pxgb, plstm, spread = mgr.predict_prognostics(valid_seq)
        self.assertTrue(math.isfinite(spread))
        self.assertGreaterEqual(spread, 0.0)
        self.assertAlmostEqual(spread, abs(pxgb - plstm), places=6)

    def test_26_comprehensive_integrity_all_checkpoints_and_scalers(self):
        """
        26. Verify file existence, sizes, and runtime hash immutability for all 8 checkpoints and 4 scalers.

        Limitation Note:
            The Phase 4 manifests (checkpoint_manifest.json and scaler_manifest.json) recorded
            file_size_bytes and fitted scaler parameters, but did NOT record cryptographic SHA-256 hashes.
            This test verifies:
            1. All 8 checkpoint file sizes match checkpoint_manifest.json exactly.
            2. All 4 scalers exist, load correctly, and contain 24 fitted features.
            3. Establishes before-and-after runtime SHA-256 hash comparison during this test.
        """
        manifest_path = os.path.join(self.ckpt_dir, "checkpoint_manifest.json")
        with open(manifest_path, "r", encoding="utf-8") as f:
            ckpt_manifest = json.load(f)

        checkpoint_hashes = {}
        for key, entry in ckpt_manifest.items():
            if key.startswith("_"):
                continue
            ckpt_file = os.path.join(self.ckpt_dir, entry["checkpoint_file"])
            self.assertTrue(os.path.exists(ckpt_file), f"Checkpoint missing: {ckpt_file}")
            # Verify file size matches manifest
            actual_size = os.path.getsize(ckpt_file)
            self.assertEqual(
                actual_size,
                entry["file_size_bytes"],
                f"File size mismatch for {entry['checkpoint_file']}: expected {entry['file_size_bytes']}, got {actual_size}",
            )
            # Compute SHA-256 hash
            h = hashlib.sha256()
            with open(ckpt_file, "rb") as f_in:
                while chunk := f_in.read(65536):
                    h.update(chunk)
            checkpoint_hashes[entry["checkpoint_file"]] = h.hexdigest()

        self.assertEqual(len(checkpoint_hashes), 8, "Expected exactly 8 champion checkpoints.")

        scaler_hashes = {}
        for subset in self.subsets:
            scaler_file = os.path.join(self.scalers_dir, f"scaler_{subset}_tier3.pkl")
            self.assertTrue(os.path.exists(scaler_file), f"Scaler missing: {scaler_file}")
            # Verify scaler loads and has 24 features
            sim = DigitalTwinSimulator(subset=subset, scalers_dir=self.scalers_dir)
            self.assertEqual(len(sim.scaling_pipeline.feature_columns), 24)
            # Compute SHA-256 hash
            h = hashlib.sha256()
            with open(scaler_file, "rb") as f_in:
                while chunk := f_in.read(65536):
                    h.update(chunk)
            scaler_hashes[f"scaler_{subset}_tier3.pkl"] = h.hexdigest()

        self.assertEqual(len(scaler_hashes), 4, "Expected exactly 4 scalers.")

        # Re-compute hashes at end of test to verify runtime immutability (zero modification)
        for ckpt_name, expected_h in checkpoint_hashes.items():
            h_now = hashlib.sha256()
            with open(os.path.join(self.ckpt_dir, ckpt_name), "rb") as f_in:
                while chunk := f_in.read(65536):
                    h_now.update(chunk)
            self.assertEqual(h_now.hexdigest(), expected_h, f"Checkpoint modified during session: {ckpt_name}")

        for sc_name, expected_h in scaler_hashes.items():
            h_now = hashlib.sha256()
            with open(os.path.join(self.scalers_dir, sc_name), "rb") as f_in:
                while chunk := f_in.read(65536):
                    h_now.update(chunk)
            self.assertEqual(h_now.hexdigest(), expected_h, f"Scaler modified during session: {sc_name}")


if __name__ == "__main__":
    unittest.main()
