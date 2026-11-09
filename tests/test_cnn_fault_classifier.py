"""
Unit Test Suite: 1D-CNN Fault Classifier
========================================
Validates 1D-CNN architecture, class predictions, calibrated probabilities,
saliency attribution, and input validation for Stage 2 Diagnosis Agent.
"""

import unittest
import numpy as np
import torch

from agents.models.cnn_fault_classifier import (
    Conv1DFaultClassifier,
    FaultDiagnosticClassifier,
    FAULT_CLASS_LABELS,
)


class TestCNNFaultClassifier(unittest.TestCase):
    """Test suite for 1D-CNN Fault Classifier and Diagnostic Engine."""

    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.in_channels = 14
        self.seq_len = 30
        self.num_classes = 5
        self.batch_size = 4

    def test_01_model_forward_shapes(self):
        """Verifies forward pass output logits shape for (B, C, T) and (B, T, C)."""
        model = Conv1DFaultClassifier(in_channels=self.in_channels, num_classes=self.num_classes)

        # Shape (B, C, T)
        x_c_first = torch.randn(self.batch_size, self.in_channels, self.seq_len)
        logits1 = model(x_c_first)
        self.assertEqual(logits1.shape, (self.batch_size, self.num_classes))

        # Shape (B, T, C)
        x_t_first = torch.randn(self.batch_size, self.seq_len, self.in_channels)
        logits2 = model(x_t_first)
        self.assertEqual(logits2.shape, (self.batch_size, self.num_classes))

    def test_02_feature_extraction(self):
        """Verifies latent embedding extraction."""
        model = Conv1DFaultClassifier(in_channels=self.in_channels, num_classes=self.num_classes)
        x = torch.randn(self.batch_size, self.in_channels, self.seq_len)
        feats = model.extract_features(x)
        self.assertEqual(feats.shape, (self.batch_size, 128))

    def test_03_diagnostic_classifier_inference(self):
        """Verifies end-to-end diagnosis with class probabilities and bounds."""
        classifier = FaultDiagnosticClassifier(
            in_channels=self.in_channels,
            num_classes=self.num_classes,
            max_confidence=0.95,
        )
        seq = np.random.randn(self.seq_len, self.in_channels).astype(np.float32)

        res = classifier.diagnose_sequence(seq)
        self.assertEqual(res["model_type"], "1D_CNN_FAULT_CLASSIFIER")
        self.assertIn(res["predicted_fault_mode"], FAULT_CLASS_LABELS.values())
        self.assertGreaterEqual(res["calibrated_confidence"], 0.0)
        self.assertLessEqual(res["calibrated_confidence"], 0.95)
        self.assertEqual(len(res["class_probabilities"]), self.num_classes)

        # Sum of probabilities approximately 1.0
        prob_sum = sum(res["class_probabilities"].values())
        self.assertAlmostEqual(prob_sum, 1.0, places=2)

    def test_04_saliency_attribution(self):
        """Verifies gradient saliency map generation across input channels."""
        classifier = FaultDiagnosticClassifier(in_channels=self.in_channels, num_classes=self.num_classes)
        seq = np.random.randn(self.seq_len, self.in_channels).astype(np.float32)

        res = classifier.diagnose_sequence(seq, compute_saliency=True)
        self.assertIn("channel_saliency", res)
        saliency = res["channel_saliency"]
        self.assertEqual(len(saliency), self.in_channels)
        # All saliencies must be non-negative
        for ch, val in saliency.items():
            self.assertGreaterEqual(val, 0.0)

    def test_05_input_validation(self):
        """Verifies dimension and type error handling."""
        classifier = FaultDiagnosticClassifier(in_channels=self.in_channels, num_classes=self.num_classes)

        # Wrong channel count (e.g. 7 instead of 14)
        with self.assertRaises(ValueError):
            classifier.diagnose_sequence(np.zeros((self.seq_len, 7), dtype=np.float32))

        # Wrong ndim (1D)
        with self.assertRaises(ValueError):
            classifier.diagnose_sequence(np.zeros(20, dtype=np.float32))

        # Invalid type
        with self.assertRaises(TypeError):
            classifier.diagnose_sequence([1, 2, 3])


if __name__ == "__main__":
    unittest.main()
