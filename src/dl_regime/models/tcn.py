"""
dl_regime.models.tcn
====================
Temporal Convolutional Network (TCN) for directional regime detection.

Implements dilated causal convolutions with residual connections.
The receptive field grows exponentially with depth, allowing the model
to capture long-range temporal dependencies without recurrence.

Reference:
    Bai et al. (2018) "An Empirical Evaluation of Generic Convolutional
    and Recurrent Networks for Sequence Modeling"
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

from dl_regime.models.base import BaseRegimeModule


class _CausalConv1d(nn.Module):
    """Causal dilated Conv1d with weight normalisation."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self._conv = weight_norm(
            nn.Conv1d(
                in_channels, out_channels, kernel_size,
                padding=padding, dilation=dilation,
            )
        )
        self._chomp = padding
        self._relu  = nn.ReLU()
        self._drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._conv(x)
        out = out[:, :, : -self._chomp] if self._chomp else out
        return self._drop(self._relu(out))


class _ResidualBlock(nn.Module):
    """Two causal conv layers with a residual skip connection."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self._conv1 = _CausalConv1d(
            in_channels, out_channels, kernel_size, dilation, dropout,
        )
        self._conv2 = _CausalConv1d(
            out_channels, out_channels, kernel_size, dilation, dropout,
        )
        self._skip  = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self._relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._relu(self._conv2(self._conv1(x)) + self._skip(x))


class TCNRegimeModel(BaseRegimeModule):
    """Temporal Convolutional Network for 3-class directional regime classification.

    Args:
        input_size:    Number of input features.
        num_channels:  List of output channel sizes per residual block.
        kernel_size:   Convolution kernel size.
        dilation_base: Base for exponential dilation growth.
        dropout:       Dropout probability.
        learning_rate: Adam learning rate.

    Example::

        model = TCNRegimeModel(
            input_size=5,
            num_channels=[64, 64, 128, 128],
        )
    """
    def __init__(
        self,
        input_size: int,
        num_channels: list[int] | None = None,
        kernel_size: int = 3,
        dilation_base: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        ) -> None:
        super().__init__(learning_rate=learning_rate)
        self.save_hyperparameters()

        if num_channels is None:
            num_channels = [64, 64, 128, 128]

        layers: list[nn.Module] = []
        in_ch = input_size
        for i, out_ch in enumerate(num_channels):
            dilation = dilation_base ** i
            layers.append(
                _ResidualBlock(in_ch, out_ch, kernel_size, dilation, dropout),
            )
            in_ch = out_ch

        self._network = nn.Sequential(*layers)
        self._fc      = nn.Linear(in_ch, self.NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Float32 tensor ``(batch, seq_len, input_size)``.

        Returns:
            Dict with ``logits`` (batch, 3) and ``regime_prob`` (batch,).
        """
        out    = self._network(x.permute(0, 2, 1))
        logits = self._fc(out[:, :, -1])
        probs  = torch.softmax(logits, dim=-1)

        return {
            "logits"     : logits,
            "regime_prob": probs[:, 1],   # P(Long)
        }
