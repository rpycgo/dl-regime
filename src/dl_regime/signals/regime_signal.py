"""
dl_regime.signals.regime_signal
================================
Label generation for DL regime models.

FutureReturnLabelGenerator
    Generates directional 3-class labels from future price returns.

    Labels:
        0 = flat  (|return| <= threshold)
        1 = long  (return >  threshold)
        2 = short (return < -threshold)

    This makes the DL models genuine directional predictors rather than
    pure volatility detectors, enabling direct Long/Short signal generation
    without relying on a direction gate.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FutureReturnLabelGenerator:
    """Generate directional 3-class labels from future price returns.

    A bar is labelled:
        1 (long)  when future_return >  threshold
        2 (short) when future_return < -threshold
        0 (flat)  otherwise

    Args:
        horizon:   Number of bars to look ahead (default: 12 = 1 hour
                   at 5-min bars).
        threshold: Minimum absolute forward return to label as directional
                   (default: 0.003 = 0.3%).
        col_close: Name of the close-price column.

    Example::

        label_gen = FutureReturnLabelGenerator(horizon=12, threshold=0.003)
        train_df = label_gen.generate(train_df)
    """
    LABEL_FLAT  = 0
    LABEL_LONG  = 1
    LABEL_SHORT = 2

    def __init__(
        self,
        horizon: int = 12,
        threshold: float = 0.003,
        col_close: str = "Close",
        ) -> None:
        self._horizon   = horizon
        self._threshold = threshold
        self._col_close = col_close

    @property
    def horizon(self) -> int:
        return self._horizon

    @property
    def threshold(self) -> float:
        return self._threshold

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ``regime_label`` column to *df*.

        Labels are derived from the signed forward return::

            future_return = close.shift(-horizon) / close - 1
            regime_label  = 1 if return >  threshold
                          = 2 if return < -threshold
                          = 0 otherwise

        Args:
            df: Training DataFrame with a close-price column.

        Returns:
            ``df`` with ``regime_label`` (0 / 1 / 2) added and tail rows
            (where forward return is unavailable) dropped.
        """
        df = df.copy()
        close = df[self._col_close]
        future_return = close.shift(-self._horizon) / close - 1.0
        df["future_return"] = future_return

        label = np.full(len(df), self.LABEL_FLAT, dtype=np.int64)
        label[future_return >  self._threshold] = self.LABEL_LONG
        label[future_return < -self._threshold] = self.LABEL_SHORT
        df["regime_label"] = label

        n_before = len(df)
        df = df.dropna(subset=["future_return"]).copy()
        n_dropped = n_before - len(df)
        if n_dropped > 0:
            logger.debug(
                "FutureReturnLabelGenerator: dropped %d tail rows (horizon=%d).",
                n_dropped, self._horizon,
            )

        long_rate  = (df["regime_label"] == self.LABEL_LONG).mean()
        short_rate = (df["regime_label"] == self.LABEL_SHORT).mean()
        logger.debug(
            "Label distribution — long: %.1f%%, short: %.1f%%, flat: %.1f%%",
            long_rate * 100,
            short_rate * 100,
            (1 - long_rate - short_rate) * 100,
        )

        return df


# Backward-compat alias
RegimeLabelGenerator = FutureReturnLabelGenerator
