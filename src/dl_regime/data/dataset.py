"""
dl_regime.data.dataset
======================
PyTorch Dataset for walk-forward regime classification.

Converts a preprocessed OHLCV DataFrame into sliding-window
(X, y) pairs where:

- X: sequence of input features over ``seq_len`` bars
- y: directional regime label at the last bar of the sequence
     0 = flat, 1 = long, 2 = short
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class RegimeDataset(Dataset):
    """Sliding-window dataset for directional regime classification.

    Args:
        df:       Preprocessed ``DataFrame`` containing feature columns
                  and a ``regime_label`` column (0 / 1 / 2).
        features: List of column names to use as input features.
        seq_len:  Number of bars per input sequence.
        scaler:   Optional fitted ``StandardScaler``. When ``None``
                  a new scaler is fitted on this dataset. Pass a
                  pre-fitted scaler when constructing the test set
                  to prevent data leakage.

    Example::

        train_ds = RegimeDataset(train_df, features, seq_len=288)
        test_ds  = RegimeDataset(test_df, features, seq_len=288,
                                 scaler=train_ds.scaler)
        train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    """
    def __init__(
        self,
        df: pd.DataFrame,
        features: list[str],
        seq_len: int = 288,
        scaler: StandardScaler | None = None,
        ) -> None:
        self.seq_len = seq_len
        self.features = features

        # Validate columns
        missing = set(features) - set(df.columns)
        if missing:
            raise KeyError(f"RegimeDataset: missing feature columns {missing}.")
        if "regime_label" not in df.columns:
            raise KeyError(
                "RegimeDataset: 'regime_label' column missing. "
                "Generate labels via FutureReturnLabelGenerator first."
            )

        # Drop rows with NaN in features or label
        df = df[features + ["regime_label"]].dropna().copy()

        # Feature scaling
        X_raw = df[features].values.astype(np.float32)
        if scaler is None:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_raw)
        else:
            self.scaler = scaler
            X_scaled = self.scaler.transform(X_raw)

        self._X = X_scaled
        # int64 required by CrossEntropyLoss
        self._y = df["regime_label"].values.astype(np.int64)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return max(0, len(self._X) - self.seq_len)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return sample dict for index *idx*.

        Returns:
            Dict with keys:

            * ``x`` — float32 tensor of shape ``(seq_len, n_features)``
            * ``y`` — int64 scalar tensor (0, 1, or 2)
        """
        x = torch.from_numpy(self._X[idx: idx + self.seq_len])
        y = torch.tensor(self._y[idx + self.seq_len - 1], dtype=torch.long)

        return {
            "x": x,
            "y": y,
        }
