"""
dl_regime.models.transformer
============================
Encoder-only Transformer for regime detection.

Uses a linear input projection, sinusoidal positional encoding, and
a stack of TransformerEncoder layers.  The [CLS] token approach is
avoided for simplicity — instead the last timestep's representation
is projected to the regime logit.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from dl_regime.models.base import BaseRegimeModule


class _PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (non-learnable)."""
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerRegimeModel(BaseRegimeModule):
    """Encoder-only Transformer for binary regime classification.

    Args:
        input_size:      Number of input features.
        d_model:         Embedding dimension.
        nhead:           Number of attention heads.
        num_layers:      Number of TransformerEncoder layers.
        dim_feedforward: FFN hidden dimension.
        dropout:         Dropout probability.
        learning_rate:   Adam learning rate.

    Example::

        model = TransformerRegimeModel(
            input_size=5,
            d_model=64,
            nhead=4,
            num_layers=3,
        )
    """
    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        ) -> None:
        super().__init__(learning_rate=learning_rate)
        self.save_hyperparameters()

        self._input_proj = nn.Linear(input_size, d_model)
        self._pos_enc = _PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self._encoder  = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
        )
        self._dropout  = nn.Dropout(dropout)
        self._fc       = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Float32 tensor ``(batch, seq_len, input_size)``.

        Returns:
            Dict with ``logit`` (batch,) and ``regime_prob`` (batch,).
        """
        out   = self._pos_enc(self._input_proj(x))
        out   = self._encoder(out)
        last  = self._dropout(out[:, -1, :])
        logit = self._fc(last).squeeze(-1)

        return {
            "logit"      : logit,
            "regime_prob": torch.sigmoid(logit),
        }
