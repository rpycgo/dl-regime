"""
dl_regime.models.lstm
=====================
LSTM-based directional regime detection model.

Uses a stacked LSTM followed by a linear projection to produce
3-class directional regime logits (flat / long / short).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from dl_regime.models.base import BaseRegimeModule


class LSTMRegimeModel(BaseRegimeModule):
    """Stacked LSTM for 3-class directional regime classification.

    Args:
        input_size:    Number of input features.
        hidden_size:   LSTM hidden state dimension.
        num_layers:    Number of stacked LSTM layers.
        dropout:       Dropout probability (applied between LSTM layers).
        bidirectional: Whether to use bidirectional LSTM.
        learning_rate: Adam learning rate.

    Example::

        model = LSTMRegimeModel(input_size=5, hidden_size=128, num_layers=2)
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
        learning_rate: float = 1e-3,
    ) -> None:
        super().__init__(learning_rate=learning_rate)
        self.save_hyperparameters()

        self._lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        fc_input = hidden_size * (2 if bidirectional else 1)
        self._fc = nn.Linear(fc_input, self.NUM_CLASSES)
        self._dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Float32 tensor ``(batch, seq_len, input_size)``.

        Returns:
            Dict with ``logits`` (batch, 3) and ``regime_prob`` (batch,).
        """
        out, _ = self._lstm(x)
        last   = self._dropout(out[:, -1, :])
        logits = self._fc(last)
        probs  = torch.softmax(logits, dim=-1)

        return {
            "logits"     : logits,
            "regime_prob": probs[:, 1],   # P(Long)
        }
