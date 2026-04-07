"""
dl_regime.models.base
=====================
Abstract PyTorch Lightning base module for all regime detection models.

Standardises:
- Binary cross-entropy training and validation steps
- Sigmoid output → regime_prob in [0, 1]
- predict_step for inference (no gradient)
- Shared optimizer (Adam with configurable lr)

All concrete models inherit this and only implement ``forward()``.
"""
from __future__ import annotations

from abc import abstractmethod

import lightning as L
import torch
import torch.nn as nn
from torch.optim import Adam


class BaseRegimeModule(L.LightningModule):
    """Abstract base for binary regime classification models.

    Subclasses must implement :meth:`forward` which takes a float32
    tensor of shape ``(batch, seq_len, n_features)`` and returns a
    dict with:

    * ``logit``       — raw logit tensor ``(batch,)``
    * ``regime_prob`` — sigmoid probability tensor ``(batch,)``

    Args:
        learning_rate: Adam learning rate.

    Example::

        class MyModel(BaseRegimeModule):
            def forward(self, x):
                ...
                return {"logit": logit, "regime_prob": torch.sigmoid(logit)}
    """
    def __init__(self, learning_rate: float = 1e-3) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self._criterion = nn.BCEWithLogitsLoss()

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

            * ``logit``       — raw logit tensor ``(batch,)``
            * ``regime_prob`` — sigmoid probability tensor ``(batch,)``
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
        out  = self(batch["x"])
        loss = self._criterion(out["logit"], batch["y"])
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
        ) -> None:
        out  = self(batch["x"])
        loss = self._criterion(out["logit"], batch["y"])
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def predict_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
        ) -> torch.Tensor:
        """Return sigmoid regime probability for inference.

        Returns:
            Float32 tensor of shape ``(batch,)`` with values in [0, 1].
        """
        with torch.no_grad():
            return self(batch["x"])["regime_prob"]

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> Adam:
        return Adam(self.parameters(), lr=self.learning_rate)
