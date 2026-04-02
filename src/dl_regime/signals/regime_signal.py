"""
dl_regime.signals.regime_signal
================================
Label generation for DL regime models.

FutureReturnLabelGenerator
    Generates binary labels from future price returns.  A bar is
    labelled 1 (profitable entry) if the absolute forward return over
    ``horizon`` bars exceeds ``threshold``.  This makes the DL
    benchmark fully independent of MDRS-SDE — no regime_prob or
    MCMC estimates are used for supervision.

Signal generation and backtesting are handled downstream by the
quant-research backtesting module — this package is responsible
only for training and prediction.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FutureReturnLabelGenerator:
    """Generate binary labels from future price returns.

    A bar is labelled 1 (profitable entry) when the absolute forward
    return over ``horizon`` bars exceeds ``threshold``.  This creates
    a supervision signal that is **completely independent** of MDRS-SDE,
    making the DL models genuine external benchmarks.

    Using absolute return means the DL model learns to predict
    *volatility breakouts* (either direction), which is the same
    semantic as MDRS-SDE's regime probability — high regime prob
    indicates a breakout opportunity regardless of direction.

    Args:
        horizon:   Number of bars to look ahead (default: 12 = 1 hour
                   at 5-min bars).
        threshold: Minimum absolute forward return to label as positive
                   (default: 0.003 = 0.3%).
        col_close: Name of the close-price column.

    Example::

        label_gen = FutureReturnLabelGenerator(horizon=12, threshold=0.003)
        train_df = label_gen.generate(train_df)
    """
    def __init__(
        self,
        horizon: int = 12,
        threshold: float = 0.003,
        col_close: str = "Close",
    ) -> None:
        self._horizon = horizon
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

        Labels are derived from the absolute forward return::

            future_return = close.shift(-horizon) / close - 1
            regime_label  = (|future_return| > threshold) ? 1 : 0

        Rows where the forward return cannot be computed (tail of the
        dataset) are dropped.

        Args:
            df: Training DataFrame with a close-price column.

        Returns:
            ``df`` with ``regime_label`` (0 / 1) added and tail rows
            (where forward return is unavailable) dropped.
        """
        df = df.copy()
        close = df[self._col_close]
        future_return = close.shift(-self._horizon) / close - 1.0
        df["future_return"] = future_return
        df["regime_label"] = (future_return.abs() > self._threshold).astype(int)

        # Drop rows where future return is NaN (tail)
        n_before = len(df)
        df = df.dropna(subset=["future_return"]).copy()
        n_dropped = n_before - len(df)
        if n_dropped > 0:
            logger.debug(
                "FutureReturnLabelGenerator: dropped %d tail rows "
                "(horizon=%d).",
                n_dropped,
                self._horizon,
            )

        return df


# Backward-compat alias
RegimeLabelGenerator = FutureReturnLabelGenerator
