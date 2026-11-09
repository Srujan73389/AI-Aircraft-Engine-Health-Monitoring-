"""
Unit Test Suite: Temporal Transformer Prognostics Model
=======================================================
Validates multi-head self-attention sequence modeling, continuous non-negative RUL forecasting,
epistemic Monte Carlo Dropout uncertainty, and temporal attention pooling for Stage 2 Prognostics Agent.
"""

import unittest
import numpy as np
import torch

from agents.models.temporal_transformer import (
    PositionalEncoding,
    TemporalTransformer,
    TemporalTransformerPrognosticator,
)


class TestTemporalTransformer(unittest.TestCase):
    """Test suite for Temporal Transformer Prognostics Model."""

    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.input_dim = 14
        self.seq_len = 30
        self.d_model = 32
        self.batch_size = 4

    def test_01_positional_encoding(self):
        """Verifies positional encoding shape and non-zero values."""
        pe = PositionalEncoding(d_model=self.d_model, max_len=100, dropout=0.0)
        x = torch.zeros(self.batch_size, self.seq_len, self.d_model)
        x_pe = pe(x)
        self.assertEqual(x_pe.shape, (self.batch_size, self.seq_len, self.d_model))
        # Ensure values are non-zero after positional encoding
        self.assertNotEqual(float(torch.norm(x_pe)), 0.0)

    def test_02_transformer_forward_and_non_negativity(self):
        """Verifies forward pass and strict non-negativity constraint on RUL."""
        model = TemporalTransformer(
            input_dim=self.input_dim,
            d_model=self.d_model,
            nhead=2,
            num_layers=1,
            dim_feedforward=64,
            max_rul=125.0,
        )
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        rul = model(x)
        self.assertEqual(rul.shape, (self.batch_size, 1))
        # RUL must be strictly non-negative
        self.assertTrue(torch.all(rul >= 0.0))

    def test_03_attention_weights_extraction(self):
        """Verifies temporal attention weights sum to 1.0 across cycles."""
        model = TemporalTransformer(
            input_dim=self.input_dim,
            d_model=self.d_model,
            nhead=2,
            num_layers=1,
        )
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        attn = model.extract_attention_weights(x)
        self.assertEqual(attn.shape, (self.batch_size, self.seq_len))
        # Softmax sum should be approximately 1.0 per sample
        sums = torch.sum(attn, dim=-1)
        for s in sums:
            self.assertAlmostEqual(float(s.item()), 1.0, places=4)

    def test_04_prognosticator_point_estimate(self):
        """Verifies high-level prognosticator point prediction output schema."""
        prognosticator = TemporalTransformerPrognosticator(
            input_dim=self.input_dim,
            d_model=self.d_model,
            nhead=2,
            num_layers=1,
            max_rul=125.0,
        )
        seq = np.random.randn(self.seq_len, self.input_dim).astype(np.float32)

        res = prognosticator.predict_rul(seq, enable_mc_dropout=False)
        self.assertEqual(res["model_type"], "TEMPORAL_TRANSFORMER")
        self.assertGreaterEqual(res["predicted_rul"], 0.0)
        self.assertLessEqual(res["predicted_rul"], 125.0)
        self.assertEqual(len(res["temporal_attention"]), self.seq_len)
        self.assertIn("max_attention_cycle_offset", res)

    def test_05_mc_dropout_epistemic_uncertainty(self):
        """Verifies Monte Carlo Dropout uncertainty estimation."""
        prognosticator = TemporalTransformerPrognosticator(
            input_dim=self.input_dim,
            d_model=self.d_model,
            nhead=2,
            num_layers=1,
        )
        seq = np.random.randn(self.seq_len, self.input_dim).astype(np.float32)

        res = prognosticator.predict_rul(seq, enable_mc_dropout=True, mc_samples=10)
        self.assertIn("epistemic_std", res)
        self.assertIn("uncertainty_tier", res)
        self.assertIn(res["uncertainty_tier"], ["LOW", "MODERATE", "HIGH"])

    def test_06_input_validation(self):
        """Verifies dimension and type error handling."""
        prognosticator = TemporalTransformerPrognosticator(input_dim=self.input_dim)

        with self.assertRaises(ValueError):
            prognosticator.predict_rul(np.zeros((self.seq_len, 5), dtype=np.float32))

        with self.assertRaises(ValueError):
            prognosticator.predict_rul(np.zeros(10, dtype=np.float32))

        with self.assertRaises(TypeError):
            prognosticator.predict_rul("invalid_input")


if __name__ == "__main__":
    unittest.main()
