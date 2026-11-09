"""
LSTM Autoencoder (LSTM-AE) Anomaly Detector
===========================================
Architectural component for Stage 2 Monitoring Agent:
Provides deep temporal reconstruction-based sequence anomaly detection
over multivariate gas-path sensor telemetry.

Theory & Math:
- Encodes a temporal window X in R^{T x D} into a low-dimensional bottleneck z in R^{d_latent}.
- Decodes z back into reconstructed sequence X_hat in R^{T x D}.
- Pointwise residual: e_t = x_t - x_hat_t.
- Mean Squared Reconstruction Error:
    L(X) = (1 / (T * D)) * sum_{t=1}^T sum_{d=1}^D (x_{t,d} - x_hat_{t,d})^2
- Calibrated threshold tau_AE: If L(X) > tau_AE, flags a temporal sequence anomaly.
- Channel-wise attribution: Identifies physical sensors contributing most to the reconstruction breach.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import math
import numpy as np
import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
    """
    Recurrent sequence encoder compressing temporal multivariate telemetry into a latent vector.
    """

    def __init__(
        self,
        input_dim: int = 14,
        hidden_dims: Tuple[int, ...] = (64, 32),
        latent_dim: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim

        # Multi-layer LSTM stack
        self.lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dims[0],
            batch_first=True,
        )
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lstm2 = nn.LSTM(
            input_size=hidden_dims[0],
            hidden_size=hidden_dims[1],
            batch_first=True,
        )
        self.fc_latent = nn.Linear(hidden_dims[1], latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (Batch, Seq_Len, Input_Dim)
        Returns:
            z: Latent vector of shape (Batch, Latent_Dim)
        """
        out1, _ = self.lstm1(x)
        out1 = self.drop(out1)
        out2, (h_n, _) = self.lstm2(out1)
        # Take the final recurrent hidden state
        z = self.fc_latent(h_n[-1])
        return z


class LSTMDecoder(nn.Module):
    """
    Recurrent sequence decoder expanding latent vector back to the original temporal sequence.
    """

    def __init__(
        self,
        latent_dim: int = 16,
        seq_len: int = 30,
        hidden_dims: Tuple[int, ...] = (32, 64),
        output_dim: int = 14,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        self.fc_expand = nn.Linear(latent_dim, hidden_dims[0])
        self.lstm1 = nn.LSTM(
            input_size=hidden_dims[0],
            hidden_size=hidden_dims[0],
            batch_first=True,
        )
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lstm2 = nn.LSTM(
            input_size=hidden_dims[0],
            hidden_size=hidden_dims[1],
            batch_first=True,
        )
        self.fc_out = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, z: torch.Tensor, target_seq_len: Optional[int] = None) -> torch.Tensor:
        """
        Args:
            z: Latent tensor of shape (Batch, Latent_Dim)
            target_seq_len: Optional override for sequence length
        Returns:
            x_hat: Reconstructed sequence of shape (Batch, Seq_Len, Output_Dim)
        """
        t_len = target_seq_len or self.seq_len
        batch_size = z.size(0)

        # Expand latent representation
        init_repr = self.fc_expand(z)  # (Batch, hidden_dims[0])
        # Repeat over time dimension
        seq_in = init_repr.unsqueeze(1).repeat(1, t_len, 1)  # (Batch, Seq_Len, hidden_dims[0])

        out1, _ = self.lstm1(seq_in)
        out1 = self.drop(out1)
        out2, _ = self.lstm2(out1)
        x_hat = self.fc_out(out2)  # (Batch, Seq_Len, Output_Dim)
        return x_hat


class LSTMAutoencoder(nn.Module):
    """
    Full LSTM Autoencoder combining Encoder and Decoder modules.
    """

    def __init__(
        self,
        input_dim: int = 14,
        seq_len: int = 30,
        hidden_dims: Tuple[int, ...] = (64, 32),
        latent_dim: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.latent_dim = latent_dim

        self.encoder = LSTMEncoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            dropout=dropout,
        )
        self.decoder = LSTMDecoder(
            latent_dim=latent_dim,
            seq_len=seq_len,
            hidden_dims=(hidden_dims[1], hidden_dims[0]),
            output_dim=input_dim,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: reconstruct input sequence."""
        z = self.encoder(x)
        x_hat = self.decoder(z, target_seq_len=x.size(1))
        return x_hat

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode sequence to latent representation."""
        return self.encoder(x)

    def compute_reconstruction_loss(
        self, x: torch.Tensor, reduction: str = "mean"
    ) -> torch.Tensor:
        """
        Computes reconstruction error between x and x_hat.
        """
        x_hat = self.forward(x)
        residuals = (x - x_hat) ** 2
        if reduction == "mean":
            return residuals.mean()
        elif reduction == "batch":
            # Return mean MSE per sample in batch: (Batch,)
            return residuals.mean(dim=(1, 2))
        elif reduction == "channel":
            # Return MSE per channel: (Batch, Channel)
            return residuals.mean(dim=1)
        elif reduction == "none":
            return residuals
        else:
            raise ValueError(f"Unknown reduction: {reduction}")


class LSTMAnomalyDetector:
    """
    High-level inference and surveillance integration engine for LSTM-AE anomaly detection.
    Provides type validation, channel attribution, threshold evaluation, and telemetry scoring.
    """

    def __init__(
        self,
        input_dim: int = 14,
        seq_len: int = 30,
        threshold: float = 0.05,
        model: Optional[LSTMAutoencoder] = None,
        channel_names: Optional[List[str]] = None,
        device: Optional[str] = None,
    ):
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.threshold = threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.channel_names = channel_names or [f"channel_{i+1}" for i in range(input_dim)]

        if model is None:
            self.model = LSTMAutoencoder(input_dim=input_dim, seq_len=seq_len)
            self._init_synthetic_nominal_weights()
        else:
            self.model = model

        self.model.to(self.device)
        self.model.eval()

    def _init_synthetic_nominal_weights(self) -> None:
        """Initializes weights with small orthogonal/xavier variance for stable reconstruction."""
        torch.manual_seed(42)
        for name, param in self.model.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.orthogonal_(param, gain=0.8)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)

    def evaluate_sequence(
        self,
        sequence: Union[np.ndarray, torch.Tensor],
        return_reconstruction: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluates a multivariate sequence window for anomalies.

        Args:
            sequence: Sequence array of shape (T, D) or (1, T, D)
            return_reconstruction: Whether to include reconstructed array in result.

        Returns:
            Dict containing:
              - anomaly_score: float (mean squared reconstruction error)
              - is_anomaly: bool (True if score > threshold)
              - threshold: float
              - channel_errors: Dict[str, float] per-channel MSE
              - top_contributors: List[Tuple[str, float]] ranked highest error channels
              - reconstruction: Optional[np.ndarray]
        """
        if isinstance(sequence, np.ndarray):
            seq_np = sequence
            if seq_np.ndim == 2:
                seq_tensor = torch.from_numpy(seq_np).float().unsqueeze(0)
            elif seq_np.ndim == 3:
                seq_tensor = torch.from_numpy(seq_np).float()
            else:
                raise ValueError(f"Sequence array must be 2D or 3D, got ndim={seq_np.ndim}")
        elif isinstance(sequence, torch.Tensor):
            if sequence.dim() == 2:
                seq_tensor = sequence.float().unsqueeze(0)
            elif sequence.dim() == 3:
                seq_tensor = sequence.float()
            else:
                raise ValueError(f"Sequence tensor must be 2D or 3D, got dim={sequence.dim()}")
        else:
            raise TypeError(f"Expected numpy ndarray or torch Tensor, got {type(sequence).__name__}")

        if seq_tensor.size(-1) != self.input_dim:
            raise ValueError(
                f"Feature dimension mismatch: expected input_dim={self.input_dim}, got {seq_tensor.size(-1)}"
            )

        seq_tensor = seq_tensor.to(self.device)

        with torch.no_grad():
            x_hat = self.model(seq_tensor)
            residuals = (seq_tensor - x_hat) ** 2
            score = float(residuals.mean().item())
            channel_errs = residuals.mean(dim=(0, 1)).cpu().numpy()

        is_anom = score > self.threshold

        channel_dict = {
            self.channel_names[i] if i < len(self.channel_names) else f"ch_{i}": float(channel_errs[i])
            for i in range(len(channel_errs))
        }

        # Rank sensors by contribution
        ranked_contributors = sorted(channel_dict.items(), key=lambda kv: kv[1], reverse=True)

        res: Dict[str, Any] = {
            "model_type": "LSTM_AUTOENCODER",
            "anomaly_score": round(score, 6),
            "is_anomaly": is_anom,
            "threshold": self.threshold,
            "score_margin": round(score - self.threshold, 6),
            "channel_errors": {k: round(v, 6) for k, v in channel_dict.items()},
            "top_contributors": ranked_contributors[:5],
        }

        if return_reconstruction:
            res["reconstruction"] = x_hat.squeeze(0).cpu().numpy()

        return res
