"""
1D Convolutional Neural Network (1D-CNN) Fault Classifier
=========================================================
Architectural component for Stage 2 Diagnosis Agent:
Performs deep temporal pattern recognition and multi-channel fault mode classification
over turbofan sensor sequences.

Supported Fault Modes:
- 0: NOMINAL (Healthy operational envelope)
- 1: HPC_DEGRADATION (High-Pressure Compressor efficiency loss / blade erosion)
- 2: FAN_DEGRADATION (Fan blade fouling / tip clearance enlargement)
- 3: SENSOR_BIAS (Sensor calibration drift or measurement bias)
- 4: THERMAL_DISTRESS (Turbine/combustor gas-path thermal overload)

Architecture:
- Hierarchical 1D temporal convolutions extracting multi-scale spectral features.
- Batch normalization and adaptive global pooling for length-invariance.
- Channel saliency / gradient attribution to pinpoint specific sensor anomalies.
- Calibrated softmax probabilities bounded to conservative aerospace confidence [0.0, 0.95].
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


FAULT_CLASS_LABELS: Dict[int, str] = {
    0: "NOMINAL",
    1: "HPC_DEGRADATION",
    2: "FAN_DEGRADATION",
    3: "SENSOR_BIAS",
    4: "THERMAL_DISTRESS",
}

FAULT_LABEL_TO_ID: Dict[str, int] = {v: k for k, v in FAULT_CLASS_LABELS.items()}


class Conv1DFaultClassifier(nn.Module):
    """
    1D-CNN model for multivariate time-series fault mode classification.
    """

    def __init__(
        self,
        in_channels: int = 14,
        num_classes: int = 5,
        hidden_dims: Tuple[int, ...] = (32, 64, 128),
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Feature Extractor Blocks
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dims[0], kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
        )

        self.block2 = nn.Sequential(
            nn.Conv1d(hidden_dims[0], hidden_dims[1], kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
        )

        self.block3 = nn.Sequential(
            nn.Conv1d(hidden_dims[1], hidden_dims[2], kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(hidden_dims[2]),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )

        # Classifier Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dims[2], 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Tensor of shape (Batch, Channels, Time) or (Batch, Time, Channels)
        Returns:
            logits: Tensor of shape (Batch, num_classes)
        """
        # If input is (Batch, Time, Channels), transpose to (Batch, Channels, Time)
        if x.size(1) != self.in_channels and x.size(2) == self.in_channels:
            x = x.transpose(1, 2)

        feat1 = self.block1(x)
        feat2 = self.block2(feat1)
        feat3 = self.block3(feat2)
        logits = self.classifier(feat3)
        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts latent feature embedding before final classification."""
        if x.size(1) != self.in_channels and x.size(2) == self.in_channels:
            x = x.transpose(1, 2)
        f = self.block3(self.block2(self.block1(x)))
        return f.squeeze(-1)


class FaultDiagnosticClassifier:
    """
    High-level diagnostic engine integrating the 1D-CNN classifier with
    DiagnosisAgent rules, class probability calibrations, and channel attribution.
    """

    def __init__(
        self,
        in_channels: int = 14,
        num_classes: int = 5,
        model: Optional[Conv1DFaultClassifier] = None,
        channel_names: Optional[List[str]] = None,
        max_confidence: float = 0.95,
        device: Optional[str] = None,
    ):
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.max_confidence = max_confidence
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.channel_names = channel_names or [f"sensor_{i+1}" for i in range(in_channels)]

        if model is None:
            self.model = Conv1DFaultClassifier(in_channels=in_channels, num_classes=num_classes)
            self._init_reproducible_weights()
        else:
            self.model = model

        self.model.to(self.device)
        self.model.eval()

    def _init_reproducible_weights(self) -> None:
        """Initializes weights with Xavier/Kaiming initialization."""
        torch.manual_seed(42)
        for m in self.model.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def diagnose_sequence(
        self,
        sequence: Union[np.ndarray, torch.Tensor],
        compute_saliency: bool = False,
    ) -> Dict[str, Any]:
        """
        Classifies sequence into fault mode candidates with calibrated probabilities.

        Args:
            sequence: Sequence of shape (T, C) or (C, T) or (1, C, T)
            compute_saliency: Whether to compute gradient-based channel saliency.

        Returns:
            Dict containing predicted fault mode, confidence, class probabilities, and evidence.
        """
        if isinstance(sequence, np.ndarray):
            seq_np = sequence
            if seq_np.ndim == 2:
                # If shape is (Time, Channels) with Channels == self.in_channels
                if seq_np.shape[1] == self.in_channels:
                    tensor = torch.from_numpy(seq_np).float().transpose(0, 1).unsqueeze(0)
                elif seq_np.shape[0] == self.in_channels:
                    tensor = torch.from_numpy(seq_np).float().unsqueeze(0)
                else:
                    raise ValueError(f"Neither dimension matches in_channels={self.in_channels}, shape={seq_np.shape}")
            elif seq_np.ndim == 3:
                tensor = torch.from_numpy(seq_np).float()
                if tensor.size(1) != self.in_channels and tensor.size(2) == self.in_channels:
                    tensor = tensor.transpose(1, 2)
            else:
                raise ValueError(f"Expected 2D or 3D sequence array, got ndim={seq_np.ndim}")
        elif isinstance(sequence, torch.Tensor):
            tensor = sequence.float()
            if tensor.dim() == 2:
                if tensor.size(1) == self.in_channels:
                    tensor = tensor.transpose(0, 1).unsqueeze(0)
                elif tensor.size(0) == self.in_channels:
                    tensor = tensor.unsqueeze(0)
                else:
                    raise ValueError(f"Dimensions do not match in_channels={self.in_channels}, shape={tensor.shape}")
            elif tensor.dim() == 3:
                if tensor.size(1) != self.in_channels and tensor.size(2) == self.in_channels:
                    tensor = tensor.transpose(1, 2)
        else:
            raise TypeError(f"Expected numpy ndarray or torch Tensor, got {type(sequence).__name__}")

        tensor = tensor.to(self.device)

        if compute_saliency:
            tensor.requires_grad_(True)
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=-1)
            pred_id = int(torch.argmax(probs, dim=-1).item())
            # Backward pass on top class logit for saliency
            self.model.zero_grad()
            logits[0, pred_id].backward(retain_graph=True)
            saliency = tensor.grad.data.abs().mean(dim=-1).squeeze(0).cpu().numpy()
            tensor.requires_grad_(False)
        else:
            with torch.no_grad():
                logits = self.model(tensor)
                probs = F.softmax(logits, dim=-1)
                pred_id = int(torch.argmax(probs, dim=-1).item())
                saliency = None

        probs_np = probs.detach().squeeze(0).cpu().numpy()
        raw_conf = float(probs_np[pred_id])
        # Bound confidence strictly in [0.0, max_confidence] per aerospace guidelines
        bounded_conf = min(raw_conf, self.max_confidence)

        prob_dict = {
            FAULT_CLASS_LABELS.get(i, f"CLASS_{i}"): round(float(probs_np[i]), 4)
            for i in range(len(probs_np))
        }

        # Build candidate ranked list
        sorted_candidates = sorted(prob_dict.items(), key=lambda kv: kv[1], reverse=True)

        res: Dict[str, Any] = {
            "model_type": "1D_CNN_FAULT_CLASSIFIER",
            "predicted_fault_mode": FAULT_CLASS_LABELS.get(pred_id, "UNKNOWN"),
            "fault_class_id": pred_id,
            "raw_confidence": round(raw_conf, 4),
            "calibrated_confidence": round(bounded_conf, 4),
            "class_probabilities": prob_dict,
            "ranked_candidates": sorted_candidates,
        }

        if saliency is not None:
            saliency_dict = {
                self.channel_names[i] if i < len(self.channel_names) else f"ch_{i}": float(saliency[i])
                for i in range(len(saliency))
            }
            res["channel_saliency"] = {k: round(v, 6) for k, v in saliency_dict.items()}

        return res
