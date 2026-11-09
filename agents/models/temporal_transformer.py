"""
Temporal Transformer Prognostics Model
======================================
Architectural component for Stage 2 Prognostics Agent:
Provides multi-head self-attention sequence modeling for continuous
Remaining Useful Life (RUL) forecasting over turbofan degradation trajectories.

Theory & Math:
- Input projection: z_0 = X * W_in + b_in, where X in R^{B x T x D}, z_0 in R^{B x T x d_model}.
- Positional Encoding: PE_{(pos, 2i)} = sin(pos / 10000^{2i/d_model}), PE_{(pos, 2i+1)} = cos(pos / 10000^{2i/d_model}).
- Multi-Head Self-Attention:
    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
- Layer Normalization with Pre-LN (norm_first=True) for gradient stability.
- Temporal sequence pooling: Aggregates degradation representations across temporal cycle window.
- Regression Head: Non-negative bounded prediction RUL_hat = ReLU(W_out * z_pooled + b_out).
- Epistemic uncertainty estimation via Monte Carlo Dropout (MCDO).
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding module.
    """

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (Batch, Seq_Len, d_model)
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TemporalTransformer(nn.Module):
    """
    Deep Transformer encoder architecture for temporal RUL prediction.
    """

    def __init__(
        self,
        input_dim: int = 14,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        max_rul: float = 125.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.max_rul = max_rul

        # Linear projection to embedding space
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model, dropout=dropout)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(d_model)

        # Attention pooling layer across sequence dimension
        self.pool_attn = nn.Linear(d_model, 1)

        # RUL Regression Head
        self.regressor = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (Batch, Seq_Len, Input_Dim)
        Returns:
            rul: Tensor of shape (Batch, 1) non-negative RUL
        """
        # 1. Project input to d_model
        h = self.input_proj(x)
        # 2. Add positional encoding
        h = self.pos_encoder(h)
        # 3. Pass through self-attention stack
        h_enc = self.transformer_encoder(h)
        h_enc = self.norm(h_enc)

        # 4. Attention pooling over temporal dimension
        attn_weights = F.softmax(self.pool_attn(h_enc), dim=1)  # (Batch, Seq_Len, 1)
        pooled = torch.sum(h_enc * attn_weights, dim=1)  # (Batch, d_model)

        # 5. Predict continuous RUL with non-negativity constraint
        raw_rul = self.regressor(pooled)
        rul = F.relu(raw_rul)  # Enforce RUL >= 0.0
        return rul

    def extract_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts temporal attention pooling weights for interpretability."""
        h = self.input_proj(x)
        h = self.pos_encoder(h)
        h_enc = self.norm(self.transformer_encoder(h))
        attn_weights = F.softmax(self.pool_attn(h_enc), dim=1)
        return attn_weights.squeeze(-1)  # (Batch, Seq_Len)


class TemporalTransformerPrognosticator:
    """
    High-level prognostic inference engine wrapping the Temporal Transformer model.
    Provides batch and single-sequence RUL estimation with Monte Carlo Dropout uncertainty.
    """

    def __init__(
        self,
        input_dim: int = 14,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        model: Optional[TemporalTransformer] = None,
        max_rul: float = 125.0,
        device: Optional[str] = None,
    ):
        self.input_dim = input_dim
        self.d_model = d_model
        self.max_rul = max_rul
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if model is None:
            self.model = TemporalTransformer(
                input_dim=input_dim,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                max_rul=max_rul,
            )
            self._init_stable_weights()
        else:
            self.model = model

        self.model.to(self.device)
        self.model.eval()

    def _init_stable_weights(self) -> None:
        """Initializes weights with Xavier normal for attention stability."""
        torch.manual_seed(42)
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def predict_rul(
        self,
        sequence: Union[np.ndarray, torch.Tensor],
        enable_mc_dropout: bool = False,
        mc_samples: int = 15,
    ) -> Dict[str, Any]:
        """
        Estimates Remaining Useful Life (RUL) for a temporal sequence.

        Args:
            sequence: Array of shape (T, D) or (1, T, D)
            enable_mc_dropout: If True, computes epistemic uncertainty via Monte Carlo Dropout.
            mc_samples: Number of stochastic forward passes.

        Returns:
            Dict containing predicted RUL, epistemic standard deviation, and temporal attention.
        """
        if isinstance(sequence, np.ndarray):
            seq_np = sequence
            if seq_np.ndim == 2:
                tensor = torch.from_numpy(seq_np).float().unsqueeze(0)
            elif seq_np.ndim == 3:
                tensor = torch.from_numpy(seq_np).float()
            else:
                raise ValueError(f"Expected 2D or 3D sequence array, got ndim={seq_np.ndim}")
        elif isinstance(sequence, torch.Tensor):
            tensor = sequence.float()
            if tensor.dim() == 2:
                tensor = tensor.unsqueeze(0)
        else:
            raise TypeError(f"Expected numpy ndarray or torch Tensor, got {type(sequence).__name__}")

        if tensor.size(-1) != self.input_dim:
            raise ValueError(
                f"Feature dimension mismatch: expected input_dim={self.input_dim}, got {tensor.size(-1)}"
            )

        tensor = tensor.to(self.device)

        # Point estimate in eval mode
        self.model.eval()
        with torch.no_grad():
            point_pred = float(self.model(tensor).squeeze().item())
            point_pred_clipped = min(max(0.0, point_pred), self.max_rul)
            attn_weights = self.model.extract_attention_weights(tensor).squeeze(0).cpu().numpy()

        uncertainty_std = 0.0
        sample_preds = []
        if enable_mc_dropout:
            # Enable dropout during inference for Monte Carlo estimation
            self.model.train()
            with torch.no_grad():
                for _ in range(mc_samples):
                    samp = float(self.model(tensor).squeeze().item())
                    sample_preds.append(min(max(0.0, samp), self.max_rul))
            self.model.eval()
            uncertainty_std = float(np.std(sample_preds))

        # Classify uncertainty tier
        if uncertainty_std <= 5.0:
            uncertainty_tier = "LOW"
        elif uncertainty_std <= 15.0:
            uncertainty_tier = "MODERATE"
        else:
            uncertainty_tier = "HIGH"

        return {
            "model_type": "TEMPORAL_TRANSFORMER",
            "predicted_rul": round(point_pred_clipped, 2),
            "epistemic_std": round(uncertainty_std, 2),
            "uncertainty_tier": uncertainty_tier,
            "temporal_attention": [round(float(w), 4) for w in attn_weights],
            "max_attention_cycle_offset": int(np.argmax(attn_weights)),
        }
