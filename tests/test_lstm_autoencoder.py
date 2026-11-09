"""
Unit Test Suite: LSTM Autoencoder (LSTM-AE) Anomaly Detector
============================================================
Validates recurrent sequence reconstruction, anomaly scoring, channel attribution,
and dimension validation for Stage 2 Monitoring Agent.
"""

import unittest
import numpy as np
import torch

from agents.models.lstm_autoencoder import (
    LSTMEncoder,
    LSTMDecoder,
    LSTMAutoencoder,
    LSTMAnomalyDetector,
)


class TestLSTMAutoencoder(unittest.TestCase):
    """Test suite for LSTM Autoencoder and Anomaly Detector."""

    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.input_dim = 14
        self.seq_len = 30
        self.batch_size = 4

    def test_01_encoder_decoder_shapes(self):
        """Verifies tensor shapes through LSTMEncoder and LSTMDecoder."""
        encoder = LSTMEncoder(input_dim=self.input_dim, hidden_dims=(32, 16), latent_dim=8)
        decoder = LSTMDecoder(latent_dim=8, seq_len=self.seq_len, hidden_dims=(16, 32), output_dim=self.input_dim)

        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        z = encoder(x)
        self.assertEqual(z.shape, (self.batch_size, 8))

        x_hat = decoder(z)
        self.assertEqual(x_hat.shape, (self.batch_size, self.seq_len, self.input_dim))

    def test_02_autoencoder_forward_and_loss(self):
        """Verifies end-to-end reconstruction forward pass and loss reductions."""
        ae = LSTMAutoencoder(input_dim=self.input_dim, seq_len=self.seq_len)
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)

        x_hat = ae(x)
        self.assertEqual(x_hat.shape, x.shape)

        loss_mean = ae.compute_reconstruction_loss(x, reduction="mean")
        self.assertEqual(loss_mean.dim(), 0)
        self.assertGreater(loss_mean.item(), 0.0)

        loss_batch = ae.compute_reconstruction_loss(x, reduction="batch")
        self.assertEqual(loss_batch.shape, (self.batch_size,))

        loss_channel = ae.compute_reconstruction_loss(x, reduction="channel")
        self.assertEqual(loss_channel.shape, (self.batch_size, self.input_dim))

    def test_03_detector_evaluation_nominal(self):
        """Verifies detector evaluation on nominal low-error sequence."""
        detector = LSTMAnomalyDetector(input_dim=self.input_dim, seq_len=self.seq_len, threshold=1.0)
        seq = np.zeros((self.seq_len, self.input_dim), dtype=np.float32)

        res = detector.evaluate_sequence(seq)
        self.assertEqual(res["model_type"], "LSTM_AUTOENCODER")
        self.assertIn("anomaly_score", res)
        self.assertIn("is_anomaly", res)
        self.assertFalse(res["is_anomaly"])
        self.assertEqual(len(res["channel_errors"]), self.input_dim)
        self.assertLessEqual(len(res["top_contributors"]), 5)

    def test_04_detector_evaluation_anomalous_spike(self):
        """Verifies detector flags large anomalous perturbations."""
        detector = LSTMAnomalyDetector(input_dim=self.input_dim, seq_len=self.seq_len, threshold=0.01)
        # Introduce massive anomalous spike
        seq = np.ones((self.seq_len, self.input_dim), dtype=np.float32) * 50.0

        res = detector.evaluate_sequence(seq)
        self.assertTrue(res["is_anomaly"])
        self.assertGreater(res["anomaly_score"], detector.threshold)
        self.assertGreater(res["score_margin"], 0.0)

    def test_05_reconstruction_retrieval(self):
        """Verifies return_reconstruction flag returns reconstructed array."""
        detector = LSTMAnomalyDetector(input_dim=self.input_dim, seq_len=self.seq_len)
        seq = np.random.randn(self.seq_len, self.input_dim).astype(np.float32)

        res = detector.evaluate_sequence(seq, return_reconstruction=True)
        self.assertIn("reconstruction", res)
        self.assertEqual(res["reconstruction"].shape, (self.seq_len, self.input_dim))

    def test_06_input_validation(self):
        """Verifies dimension and type error handling."""
        detector = LSTMAnomalyDetector(input_dim=self.input_dim, seq_len=self.seq_len)

        # Wrong channel dimension
        with self.assertRaises(ValueError):
            detector.evaluate_sequence(np.zeros((self.seq_len, 5), dtype=np.float32))

        # Wrong ndim (1D)
        with self.assertRaises(ValueError):
            detector.evaluate_sequence(np.zeros(10, dtype=np.float32))

        # Invalid type
        with self.assertRaises(TypeError):
            detector.evaluate_sequence("invalid_string_input")


if __name__ == "__main__":
    unittest.main()
