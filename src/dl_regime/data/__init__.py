"""
dl_regime.data
==============
PyTorch Dataset for regime detection.

Exports
-------
RegimeDataset
    Sliding-window dataset that converts preprocessed OHLCV DataFrames
    into (X, y) pairs for supervised regime classification.
"""

from dl_regime.data.dataset import RegimeDataset

__all__ = ["RegimeDataset"]
