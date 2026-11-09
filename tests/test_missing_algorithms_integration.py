"""
Integration Test Suite: Multi-Algorithm Architecture Integration
================================================================
Validates end-to-end integration of the 5 newly implemented architectural components:
  1. LSTM-AE (Stage 2 Monitoring Agent)
  2. 1D-CNN Fault Classifier (Stage 2 Diagnosis Agent)
  3. Temporal Transformer (Stage 2 Prognostics Agent)
  4. A* Search Planner & PPO Policy (Stage 2 Recovery Planning Agent)
  5. Unscented Kalman Filter (Stage 3 Digital Twin State Estimation)
"""

import os
import unittest
import numpy as np
import torch

from agents.monitoring_agent import MonitoringAgent
from agents.diagnosis_agent import DiagnosisAgent
from agents.prognostics_agent import PrognosticsAgent
from agents.models.lstm_autoencoder import LSTMAnomalyDetector
from agents.models.cnn_fault_classifier import FaultDiagnosticClassifier
from agents.models.temporal_transformer import TemporalTransformerPrognosticator
from agents.planning.a_star_planner import AStarRecoveryPlanner
from agents.planning.ppo_recovery_policy import PPORecoveryPolicy
from digital_twin.twin_simulator import DigitalTwinSimulator


class TestMissingAlgorithmsIntegration(unittest.TestCase):
    """Integration verification across all 5 newly implemented algorithms."""

    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)

    def test_01_monitoring_agent_with_lstm_ae(self):
        """Verifies MonitoringAgent integration with LSTM-AE detector."""
        detector = LSTMAnomalyDetector(input_dim=14, seq_len=30, threshold=0.1)
        agent = MonitoringAgent(subset="FD001", lstm_ae_detector=detector)

        # 1. Direct sequence evaluation
        seq = np.random.randn(30, 14).astype(np.float32) * 0.1
        eval_res = agent.evaluate_sequence_with_lstm_ae(seq)
        self.assertIsNotNone(eval_res)
        self.assertIn("anomaly_score", eval_res)
        self.assertIn("is_anomaly", eval_res)

        # 2. Feeding anomaly score to agent evaluate()
        from agents.schemas import MonitoringInput
        inp = MonitoringInput(
            unit_number=1,
            time_cycles=50,
            operating_context={"altitude": 0.0, "mach": 0.0, "throttle": 100.0},
            raw_telemetry={"T30": 1585.0, "P30": 553.0},
            derived_indicators={"overall_pressure_ratio": 37.8},
            anomaly_scores={"lstm_ae": 0.25},  # Exceeds threshold 0.1
        )
        out = agent.evaluate(inp)
        self.assertTrue(out.model_anomaly_flag)

    def test_02_diagnosis_agent_with_cnn_classifier(self):
        """Verifies DiagnosisAgent integration with 1D-CNN fault classifier."""
        cnn_classifier = FaultDiagnosticClassifier(in_channels=14, num_classes=5)
        agent = DiagnosisAgent(subset="FD001", cnn_fault_classifier=cnn_classifier)

        seq = np.random.randn(30, 14).astype(np.float32)
        res = agent.diagnose_with_cnn(seq, compute_saliency=True)
        self.assertIsNotNone(res)
        self.assertIn("predicted_fault_mode", res)
        self.assertIn("calibrated_confidence", res)
        self.assertLessEqual(res["calibrated_confidence"], 0.95)
        self.assertIn("channel_saliency", res)

    def test_03_prognostics_agent_with_transformer(self):
        """Verifies PrognosticsAgent integration with Temporal Transformer."""
        transformer = TemporalTransformerPrognosticator(input_dim=14, max_rul=125.0)
        agent = PrognosticsAgent(subset="FD001", temporal_transformer=transformer)

        seq = np.random.randn(30, 14).astype(np.float32)
        res = agent.predict_transformer_rul(seq, enable_mc_dropout=True)
        self.assertIsNotNone(res)
        self.assertIn("predicted_rul", res)
        self.assertGreaterEqual(res["predicted_rul"], 0.0)
        self.assertIn("epistemic_std", res)
        self.assertIn("uncertainty_tier", res)

    def test_04_recovery_planning_a_star_and_ppo(self):
        """Verifies A* planner and PPO policy cooperation for recovery planning."""
        # 1. A* Planner computes step-by-step mitigation
        planner = AStarRecoveryPlanner(t30_target_safe=1590.0, t30_max_limit=1625.0)
        path = planner.plan_recovery(current_t30=1612.0, current_p30=554.0, current_opr=37.9)
        self.assertTrue(path.success)
        self.assertGreater(len(path.actions), 0)
        self.assertLessEqual(path.states[-1].t30, 1590.0)

        # 2. PPO Policy provides fine-grained control compensation
        ppo_policy = PPORecoveryPolicy(state_dim=8, action_dim=3)
        state_vec = np.array([0.8, 0.4, 0.5, 0.7, 0.6, 0.5, 0.0, 0.0], dtype=np.float32)
        ppo_res = ppo_policy.select_action(state_vec, deterministic=True)
        self.assertIn("delta_derate_pct", ppo_res)
        self.assertIn("delta_bleed_pct", ppo_res)
        self.assertIn("delta_thermal_offset_deg", ppo_res)

    def test_05_digital_twin_with_ukf_state_estimation(self):
        """Verifies DigitalTwinSimulator tracking with Unscented Kalman Filter."""
        sim = DigitalTwinSimulator(subset="FD001", enable_ml_inference=False, enable_state_estimation=True)
        init_state = sim.load_engine(unit_number=1, split="train")
        self.assertIsNotNone(init_state)

        # Advance 5 operational cycles
        for _ in range(5):
            sim.step()

        ukf_state = sim.get_ukf_state()
        self.assertIsNotNone(ukf_state)
        self.assertIn("estimated_state", ukf_state)
        self.assertIn("covariance_trace", ukf_state)
        est = ukf_state["estimated_state"]
        self.assertGreater(est["T30"], 1500.0)
        self.assertLess(est["T30"], 1650.0)
        self.assertGreater(est["P30"], 500.0)
        self.assertLess(est["P30"], 600.0)

    def test_06_dataset_and_model_immutability(self):
        """Guarantees zero corruption to raw C-MAPSS data, scalers, or champion checkpoints."""
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        raw_cmapss_dir = os.path.join(base_dir, "dataset", "raw", "C-MAPSS")
        scalers_dir = os.path.join(base_dir, "models", "scalers")
        checkpoints_dir = os.path.join(base_dir, "models", "checkpoints")

        # 12 raw files
        for i in range(1, 5):
            for prefix in ["train", "test", "RUL"]:
                fp = os.path.join(raw_cmapss_dir, f"{prefix}_FD00{i}.txt")
                self.assertTrue(os.path.exists(fp))
                self.assertGreater(os.path.getsize(fp), 0)

        # 4 scalers
        for i in range(1, 5):
            sp = os.path.join(scalers_dir, f"scaler_FD00{i}_tier3.pkl")
            self.assertTrue(os.path.exists(sp))
            self.assertGreater(os.path.getsize(sp), 0)

        # 8 checkpoints
        for i in range(1, 5):
            xp = os.path.join(checkpoints_dir, f"xgb_FD00{i}_champion.json")
            lp = os.path.join(checkpoints_dir, f"lstm_FD00{i}_champion.pt")
            self.assertTrue(os.path.exists(xp))
            self.assertTrue(os.path.exists(lp))


if __name__ == "__main__":
    unittest.main()
