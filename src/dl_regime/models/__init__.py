"""
dl_regime.models
================
PyTorch Lightning regime detection models.

All models inherit from ``BaseRegimeModule`` which standardises
training, validation, and prediction steps so they can be swapped
into the WFA trainer without any changes.

Exports
-------
BaseRegimeModule   : Lightning base with shared train/val/predict logic
LSTMRegimeModel    : Bidirectional-optional LSTM
TCNRegimeModel     : Temporal Convolutional Network
TransformerRegimeModel : Encoder-only Transformer
"""
from dl_regime.models.base import BaseRegimeModule
from dl_regime.models.lstm import LSTMRegimeModel
from dl_regime.models.tcn import TCNRegimeModel
from dl_regime.models.transformer import TransformerRegimeModel

__all__ = [
    "BaseRegimeModule",
    "LSTMRegimeModel",
    "TCNRegimeModel",
    "TransformerRegimeModel",
]
