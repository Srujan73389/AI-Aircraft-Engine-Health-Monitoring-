"""
Agents Deep Learning Models Package
===================================
Contains specialized neural network architectures for multi-agent health monitoring,
fault diagnostics, and prognostic life forecasting:
- LSTM-AE: Temporal sequence reconstruction for Stage 2 Monitoring Agent
- 1D-CNN: Multi-channel convolutional fault classifier for Stage 2 Diagnosis Agent
- Temporal Transformer: Self-attention RUL forecasting for Stage 2 Prognostics Agent
"""

from agents.models.lstm_autoencoder import LSTMAutoencoder, LSTMAnomalyDetector
from agents.models.cnn_fault_classifier import Conv1DFaultClassifier, FaultDiagnosticClassifier
from agents.models.temporal_transformer import TemporalTransformer, TemporalTransformerPrognosticator

__all__ = [
    "LSTMAutoencoder",
    "LSTMAnomalyDetector",
    "Conv1DFaultClassifier",
    "FaultDiagnosticClassifier",
    "TemporalTransformer",
    "TemporalTransformerPrognosticator",
]
