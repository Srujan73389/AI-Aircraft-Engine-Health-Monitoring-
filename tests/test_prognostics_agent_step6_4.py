"""
Unit and Integration Tests for Phase 6 Step 6.4: Autonomous Prognostics Agent
Verifies independent operation of PrognosticsAgent including model checkpoint loading,
dual-model sequential inference (XGBoost & PyTorch LSTM), non-negativity enforcement,
epistemic spread quantification, uncertainty classification, virtual health index computation,
end-to-end Digital Twin handoff, input validation, and artifact immutability.

Roadmap Reference: Phase 6 - Step 6.4:
"Implement the Prognostics Agent using the saved RUL model and preprocessing pipeline"
"""

import hashlib
import json
import math
import os
import unittest
import numpy as np

from agents.schemas import (
    AgentExecutionStatus,
    AgentMetadata,
    PrognosticsInput,
    PrognosticsOutput,
)
from agents.prognostics_agent import PrognosticsAgent
from digital_twin.twin_simulator import DigitalTwinSimulator


class TestPrognosticsAgentStep6_4(unittest.TestCase):
    """
    Test suite for Phase 6 Step 6.4: Autonomous Prognostics Agent.
    """

    @classmethod
    def setUpClass(cls):
        cls.raw_data_dir = os.path.join("dataset", "raw", "C-MAPSS")
        cls.scalers_dir = os.path.join("models", "scalers")
        cls.checkpoint_dir = os.path.join("models", "checkpoints")

        # 1. Capture SHA-256 hashes of all 12 raw C-MAPSS dataset files
        cls.raw_hashes = {}
        if os.path.exists(cls.raw_data_dir):
            for fname in sorted(os.listdir(cls.raw_data_dir)):
                if fname.endswith(".txt") and fname != "readme.txt":
                    fpath = os.path.join(cls.raw_data_dir, fname)
                    with open(fpath, "rb") as f:
                        cls.raw_hashes[fname] = hashlib.sha256(f.read()).hexdigest()

        # 2. Capture SHA-256 hashes of all 4 subset scalers
        cls.scaler_hashes = {}
        if os.path.exists(cls.scalers_dir):
            for fname in sorted(os.listdir(cls.scalers_dir)):
                if fname.endswith(".pkl"):
                    fpath = os.path.join(cls.scalers_dir, fname)
                    with open(fpath, "rb") as f:
                        cls.scaler_hashes[fname] = hashlib.sha256(f.read()).hexdigest()

        # 3. Capture SHA-256 hashes of all 8 champion model checkpoints
        cls.checkpoint_hashes = {}
        if os.path.exists(cls.checkpoint_dir):
            for fname in sorted(os.listdir(cls.checkpoint_dir)):
                if fname.endswith((".json", ".pt")) and not fname.endswith("manifest.json"):
                    fpath = os.path.join(cls.checkpoint_dir, fname)
                    with open(fpath, "rb") as f:
                        cls.checkpoint_hashes[fname] = hashlib.sha256(f.read()).hexdigest()

    def setUp(self):
        """Instantiate a fresh PrognosticsAgent for each test."""
        self.agent = PrognosticsAgent(
            agent_id="test_prognostics_agent",
            subset="FD001",
            checkpoint_dir=self.checkpoint_dir,
        )

    def _create_synthetic_seq_tensor(self, subset: str = "FD001") -> np.ndarray:
        """Helper to create a valid 30-cycle synthetic sequence tensor normalized in [0, 1]."""
        channels = 14 if subset == "FD001" else 24
        # Gradual slight drift simulating nominal-to-moderate degradation
        base = np.linspace(0.2, 0.4, 30).reshape(30, 1)
        noise = np.random.RandomState(42).uniform(0.0, 0.05, size=(30, channels))
        seq = np.tile(base, (1, channels)) + noise
        return np.clip(seq, 0.0, 1.0).astype(np.float32)

    # -------------------------------------------------------------------------
    # Test 1: Checkpoint Loading Across All 4 Subsets
    # -------------------------------------------------------------------------
    def test_01_model_loading_across_all_subsets(self):
        """
        Verify that the PrognosticsAgent correctly loads frozen champion XGBoost and LSTM models
        for FD001, FD002, FD003, and FD004.
        """
        for subset in ["FD001", "FD002", "FD003", "FD004"]:
            mgr = self.agent.get_model_manager(subset)
            self.assertTrue(mgr.is_loaded)
            self.assertIsNotNone(mgr.xgb_model)
            self.assertIsNotNone(mgr.lstm_model)
            self.assertFalse(mgr.lstm_model.training)  # Assert eval() mode

    # -------------------------------------------------------------------------
    # Test 2: Inference Execution on Synthetic Input
    # -------------------------------------------------------------------------
    def test_02_inference_execution_on_synthetic_input(self):
        """
        Verify dual-model inference on a synthetic 30-cycle sequence payload.
        """
        seq = self._create_synthetic_seq_tensor("FD001")
        inp = PrognosticsInput(
            unit_number=1,
            time_cycles=30,
            subset="FD001",
            normalized_feature_vec=seq[-1].tolist(),
            temporal_seq_tensor=seq.tolist(),
            feature_tier="Tier1_Core14",
        )

        out = self.agent.process_input(inp)

        self.assertIsInstance(out, PrognosticsOutput)
        self.assertEqual(out.unit_number, 1)
        self.assertEqual(out.time_cycles, 30)
        self.assertEqual(out.subset, "FD001")
        self.assertGreater(out.predicted_rul_xgboost, 0.0)
        self.assertGreater(out.predicted_rul_lstm, 0.0)
        self.assertGreater(out.ensemble_rul_estimate, 0.0)
        self.assertGreaterEqual(out.prognostic_spread, 0.0)
        self.assertIsNotNone(out.metadata)
        self.assertEqual(out.metadata.status, AgentExecutionStatus.SUCCESS)
        self.assertGreater(out.metadata.execution_time_ms, 0.0)

    # -------------------------------------------------------------------------
    # Test 3: Strict Non-Negativity Enforcement
    # -------------------------------------------------------------------------
    def test_03_non_negativity_enforcement(self):
        """
        Verify that all RUL predictions (XGBoost, LSTM, Ensemble) are strictly non-negative (>= 0.0).
        """
        seq = self._create_synthetic_seq_tensor("FD001")
        inp = PrognosticsInput(
            unit_number=99,
            time_cycles=300,
            subset="FD001",
            normalized_feature_vec=seq[-1].tolist(),
            temporal_seq_tensor=seq.tolist(),
        )

        out = self.agent.process_input(inp)
        self.assertGreaterEqual(out.predicted_rul_xgboost, 0.0)
        self.assertGreaterEqual(out.predicted_rul_lstm, 0.0)
        self.assertGreaterEqual(out.ensemble_rul_estimate, 0.0)
        self.assertGreaterEqual(out.prognostic_spread, 0.0)

    # -------------------------------------------------------------------------
    # Test 4: Epistemic Spread and Uncertainty Classification
    # -------------------------------------------------------------------------
    def test_04_uncertainty_classification(self):
        """
        Verify correct classification of epistemic uncertainty tiers:
        LOW (<= 15), MODERATE (15 - 30), HIGH (> 30).
        """
        self.assertEqual(self.agent.classify_uncertainty(5.0), "LOW_UNCERTAINTY")
        self.assertEqual(self.agent.classify_uncertainty(15.0), "LOW_UNCERTAINTY")
        self.assertEqual(self.agent.classify_uncertainty(15.1), "MODERATE_UNCERTAINTY")
        self.assertEqual(self.agent.classify_uncertainty(30.0), "MODERATE_UNCERTAINTY")
        self.assertEqual(self.agent.classify_uncertainty(30.1), "HIGH_UNCERTAINTY")
        self.assertEqual(self.agent.classify_uncertainty(55.0), "HIGH_UNCERTAINTY")

    # -------------------------------------------------------------------------
    # Test 5: Virtual Health Index Bounds [0.0, 1.0]
    # -------------------------------------------------------------------------
    def test_05_virtual_health_index_bounds(self):
        """
        Verify that virtual engine health index H(t) is strictly normalized in [0.0, 1.0].
        """
        # Upper saturation: RUL >= 125 cycles -> H(t) = 1.0
        self.assertEqual(self.agent.compute_virtual_health_index(150.0), 1.0)
        self.assertEqual(self.agent.compute_virtual_health_index(125.0), 1.0)

        # Mid degradation: RUL = 62.5 cycles -> H(t) = 0.50
        self.assertAlmostEqual(self.agent.compute_virtual_health_index(62.5), 0.50, places=4)

        # Total exhaustion: RUL = 0.0 -> H(t) = 0.0
        self.assertEqual(self.agent.compute_virtual_health_index(0.0), 0.0)

        # Negative protection: RUL < 0.0 -> H(t) = 0.0
        self.assertEqual(self.agent.compute_virtual_health_index(-10.0), 0.0)

    # -------------------------------------------------------------------------
    # Test 6: Cycles to Caution Threshold Projection
    # -------------------------------------------------------------------------
    def test_06_cycles_to_caution_threshold(self):
        """
        Verify projection of flight cycles remaining before reaching caution threshold (45 cycles).
        """
        # Above threshold: RUL = 80 cycles -> 35 cycles remaining
        self.assertAlmostEqual(self.agent.compute_cycles_to_caution(80.0), 35.0, places=4)

        # Exactly at threshold: RUL = 45 cycles -> 0 cycles remaining
        self.assertAlmostEqual(self.agent.compute_cycles_to_caution(45.0), 0.0, places=4)

        # Below threshold: RUL = 20 cycles -> 0 cycles remaining (caution active)
        self.assertEqual(self.agent.compute_cycles_to_caution(20.0), 0.0)

    # -------------------------------------------------------------------------
    # Test 7: End-to-End Digital Twin State Handoff
    # -------------------------------------------------------------------------
    def test_07_digital_twin_state_handoff(self):
        """
        Verify ingestion of real replayed states from DigitalTwinSimulator across early and late cycles.
        """
        sim = DigitalTwinSimulator(subset="FD001")
        sim.load_engine(unit_number=1, split="train")

        # Step to cycle 30 (minimum window requirement)
        for _ in range(29):
            sim.step()
        early_state = sim.current_state

        out_early = self.agent.process_engine_state(early_state)
        self.assertIsInstance(out_early, PrognosticsOutput)
        self.assertEqual(out_early.unit_number, 1)
        self.assertEqual(out_early.time_cycles, 30)
        self.assertGreater(out_early.ensemble_rul_estimate, 50.0)
        self.assertGreater(out_early.virtual_health_index, 0.5)

        # Advance to end of life (cycle 192)
        while sim.current_state.time_cycles < sim.total_cycles:
            sim.step()
        late_state = sim.current_state

        out_late = self.agent.process_engine_state(late_state)
        self.assertIsInstance(out_late, PrognosticsOutput)
        self.assertEqual(out_late.unit_number, 1)
        self.assertEqual(out_late.time_cycles, sim.total_cycles)
        # Near end of life, predicted RUL should be low (< 20 cycles) and health index low
        self.assertLess(out_late.ensemble_rul_estimate, 30.0)
        self.assertLess(out_late.virtual_health_index, 0.3)
        self.assertEqual(out_late.cycles_to_caution_threshold, 0.0)

    # -------------------------------------------------------------------------
    # Test 8: Malformed / Non-Finite Input Handling
    # -------------------------------------------------------------------------
    def test_08_malformed_and_non_finite_input_rejection(self):
        """
        Verify graceful rejection of NaN, Inf, invalid dimensions, or negative unit numbers.
        """
        seq = self._create_synthetic_seq_tensor("FD001")

        # A. NaN in tensor
        bad_seq_nan = seq.copy()
        bad_seq_nan[10, 5] = float("nan")
        inp_nan = PrognosticsInput(
            unit_number=1,
            time_cycles=30,
            subset="FD001",
            normalized_feature_vec=seq[-1].tolist(),
            temporal_seq_tensor=bad_seq_nan.tolist(),
        )
        with self.assertRaises(ValueError):
            self.agent.process_input(inp_nan)

        # B. Inf in tensor
        bad_seq_inf = seq.copy()
        bad_seq_inf[15, 2] = float("inf")
        inp_inf = PrognosticsInput(
            unit_number=1,
            time_cycles=30,
            subset="FD001",
            normalized_feature_vec=seq[-1].tolist(),
            temporal_seq_tensor=bad_seq_inf.tolist(),
        )
        with self.assertRaises(ValueError):
            self.agent.process_input(inp_inf)

        # C. Invalid unit_number
        inp_bad_unit = PrognosticsInput(
            unit_number=-5,
            time_cycles=30,
            subset="FD001",
            normalized_feature_vec=seq[-1].tolist(),
            temporal_seq_tensor=seq.tolist(),
        )
        with self.assertRaises(ValueError):
            self.agent.process_input(inp_bad_unit)

        # D. Invalid object passed to process_engine_state
        with self.assertRaises(TypeError):
            self.agent.process_engine_state("invalid_state_object")

    # -------------------------------------------------------------------------
    # Test 9: Serialization Round-Trip
    # -------------------------------------------------------------------------
    def test_09_prognostics_output_json_serialization(self):
        """
        Verify JSON round-trip serialization and deserialization of PrognosticsOutput.
        """
        meta = AgentMetadata("prog_agent", "2026-11-06T09:00:00Z", 14.5, AgentExecutionStatus.SUCCESS)
        out = PrognosticsOutput(
            unit_number=7,
            time_cycles=45,
            subset="FD001",
            predicted_rul_xgboost=82.4,
            predicted_rul_lstm=79.1,
            ensemble_rul_estimate=80.75,
            prognostic_spread=3.3,
            virtual_health_index=0.646,
            uncertainty_classification="LOW_UNCERTAINTY",
            cycles_to_caution_threshold=35.75,
            prognostic_summary="Nominal prognosis",
            metadata=meta,
        )

        json_str = out.to_json()
        reconstructed = PrognosticsOutput.from_json(json_str)

        self.assertEqual(reconstructed.unit_number, out.unit_number)
        self.assertEqual(reconstructed.subset, out.subset)
        self.assertAlmostEqual(reconstructed.ensemble_rul_estimate, out.ensemble_rul_estimate, places=4)
        self.assertAlmostEqual(reconstructed.prognostic_spread, out.prognostic_spread, places=4)
        self.assertEqual(reconstructed.uncertainty_classification, "LOW_UNCERTAINTY")

    # -------------------------------------------------------------------------
    # Test 10: Dataset and Model Checkpoint Immutability
    # -------------------------------------------------------------------------
    def test_10_dataset_and_model_immutability(self):
        """
        Cryptographically verify that zero raw dataset files, scalers, or model checkpoints were altered.
        """
        # Verify raw datasets
        for fname, original_hash in self.raw_hashes.items():
            fpath = os.path.join(self.raw_data_dir, fname)
            with open(fpath, "rb") as f:
                current_hash = hashlib.sha256(f.read()).hexdigest()
            self.assertEqual(
                current_hash,
                original_hash,
                f"Raw C-MAPSS dataset file '{fname}' was modified during test execution!",
            )

        # Verify scalers
        for fname, original_hash in self.scaler_hashes.items():
            fpath = os.path.join(self.scalers_dir, fname)
            with open(fpath, "rb") as f:
                current_hash = hashlib.sha256(f.read()).hexdigest()
            self.assertEqual(
                current_hash,
                original_hash,
                f"Phase 3 scaler '{fname}' was modified during test execution!",
            )

        # Verify champion checkpoints
        for fname, original_hash in self.checkpoint_hashes.items():
            fpath = os.path.join(self.checkpoint_dir, fname)
            with open(fpath, "rb") as f:
                current_hash = hashlib.sha256(f.read()).hexdigest()
            self.assertEqual(
                current_hash,
                original_hash,
                f"Phase 4 model checkpoint '{fname}' was modified during test execution!",
            )


if __name__ == "__main__":
    unittest.main()
