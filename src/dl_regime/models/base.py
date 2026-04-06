"""
dl_regime.models.base
=====================
Abstract PyTorch Lightning base module for all regime detection models.

Standardises:
- 3-class CrossEntropy training and validation steps
- Softmax output → class probabilities in [0, 1]
- predict_step for inference (no gradient)
- Shared optimizer (Adam with configurable lr)

Label convention:
    0 = flat
    1 = long
    2 = short

All concrete models inherit this and only implement ``forward()``
which must return logits of shape (batch, 3).
"""
from __future__ import annotations

from abc import abstractmethod

import lightning as L
import torch
import torch.nn as nn
from torch.optim import Adam


class BaseRegimeModule(L.LightningModule):
    """Abstract base for directional regime classification models.

    Subclasses must implement :meth:`forward` which takes a float32
    tensor of shape ``(batch, seq_len, n_features)`` and returns a
    dict with:

    * ``logits``      — raw logit tensor ``(batch, 3)``
    * ``regime_prob`` — softmax probability for Long class ``(batch,)``

    Args:
        learning_rate: Adam learning rate.

    Example::

        class MyModel(BaseRegimeModule):
            def forward(self, x):
                ...
                return {"logits": logits, "regime_prob": probs[:, 1]}
    """
    NUM_CLASSES = 3  # 0=flat, 1=long, 2=short

    def __init__(self, learning_rate: float = 1e-3) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self._criterion = nn.CrossEntropyLoss()

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Float32 tensor of shape ``(batch, seq_len, n_features)``.

        Returns:
            Dict with keys:

            * ``logits``      — raw logit tensor ``(batch, 3)``
            * ``regime_prob`` — P(Long) = softmax(logits)[:, 1] ``(batch,)``
        """
        pass

    # ------------------------------------------------------------------
    # Lightning steps
    # ------------------------------------------------------------------

    def training_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
        ) -> torch.Tensor:
        out = self(batch["x"])
        loss = self._criterion(out["logits"], batch["y"])
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
        ) -> None:
        out = self(batch["x"])
        loss = self._criterion(out["logits"], batch["y"])
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def predict_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
        ) -> torch.Tensor:
        """Return class probabilities for inference.

        Returns:
            Float32 tensor of shape ``(batch, 3)`` — softmax probabilities
            for [flat, long, short].
        """
        with torch.no_grad():
            return torch.softmax(self(batch["x"])["logits"], dim=-1)

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> Adam:
        return Adam(self.parameters(), lr=self.learning_rate)
