"""
dl_regime.signals.regime_signal
================================
Label generation and signal pipeline for DL regime models.

Two classes:

RegimeLabelGenerator
    Generates binary regime labels per WFA window from the STRS-SDE
    regime_prob.  Labels are window-specific (adaptive gamma) so DL
    learns the same classification task as STRS-SDE.

DlRegimeSignalGenerator
    Converts DL model output (regime_prob array) into ``signal`` /
    ``confidence`` columns using the same sticky filter + ADX gate
    as ``mdrs_sde.signals.RegimeSignalGenerator``.

    Uses **fixed execution params** (no SNR scaling) for fair
    comparison with DL models that lack MCMC posterior estimates.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _sticky_filter(
    binary_signals: np.ndarray,
    minimum_duration: int,
) -> np.ndarray:
    mask = (
        pd.Series(binary_signals)
        .rolling(window=minimum_duration)
        .sum()
        == minimum_duration
    )
    return mask.astype(int).values


class RegimeLabelGenerator:
    """Generate per-window binary regime labels from STRS-SDE output.

    Labels are derived from the STRS-SDE regime_prob produced for each
    WFA window so that DL models are trained to replicate the same
    adaptive threshold behaviour.

    Args:
        entry_threshold: Probability threshold above which a bar is
                         labelled as breakout regime (default: 0.5).

    Example::

        label_gen = RegimeLabelGenerator(entry_threshold=0.5)
        train_df = label_gen.generate(train_df, regime_prob_series)
    """
    def __init__(self, entry_threshold: float = 0.5) -> None:
        self._threshold = entry_threshold

    def generate(
        self,
        df: pd.DataFrame,
        regime_prob: pd.Series,
        ) -> pd.DataFrame:
        """Add ``regime_label`` column to *df*.

        Args:
            df:          Training DataFrame.
            regime_prob: STRS-SDE regime probability Series aligned to
                         ``df``'s index.

        Returns:
            ``df`` with ``regime_label`` (0 / 1) added.
        """
        df = df.copy()
        df["regime_label"] = (regime_prob > self._threshold).astype(int)

        return df


class DlRegimeSignalGenerator:
    """Convert DL regime_prob output to trading signals.

    Applies the same sticky filter + ADX gate + direction gate as
    ``mdrs_sde.signals.RegimeSignalGenerator`` but uses **fixed**
    execution params (no SNR scaling) for fair DL comparison.

    Args:
        risk_config:   ``[risk_management]`` section from
                       ``backtest_settings.toml``.
        filter_config: ``[filters]`` section.
        trade_config:  ``[trading_parameters]`` section.

    Example::

        gen = DlRegimeSignalGenerator(
            risk_config=bt_cfg["risk_management"],
            filter_config=bt_cfg["filters"],
            trade_config=bt_cfg["trading_parameters"],
        )
        signal_df = gen.generate(test_data, regime_prob_array)
        fixed_params = gen.get_fixed_params()
    """
    def __init__(
        self,
        risk_config: dict[str, Any],
        filter_config: dict[str, Any],
        trade_config: dict[str, Any],
    ) -> None:
        self._risk = risk_config
        self._filters = filter_config
        self._trade = trade_config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        test_data: pd.DataFrame,
        regime_prob: np.ndarray,
        ) -> pd.DataFrame:
        """Generate ``signal`` and ``confidence`` columns.

        Args:
            test_data:   Out-of-sample DataFrame with ``hybrid_z_score``,
                         ``Close``, ``ADX``, ``dynamic_resistance``,
                         ``dynamic_support``.
            regime_prob: Array of regime probabilities from DL model,
                         aligned to ``test_data``'s index.

        Returns:
            ``test_data`` with ``regime_prob``, ``confidence``,
            ``signal`` columns added.
        """
        df = test_data.copy()
        df["regime_prob"] = regime_prob
        df["confidence"] = regime_prob

        entry_thr = self._risk.get("entry_probability_threshold", 0.5)
        min_dur = self._risk.get("minimum_signal_duration", 5)
        use_sticky = self._filters.get("use_sticky", True)
        use_adx = self._filters.get("use_adx", True)
        adx_thr = self._trade.get("adx_threshold", 30)

        # Sticky filter
        binary = (df["regime_prob"] > entry_thr).astype(int).values
        sticky = (
            _sticky_filter(binary, min_dur) if use_sticky else binary
        )

        # Direction + ADX gate
        long_cond = df["Close"] > df["dynamic_resistance"]
        short_cond = df["Close"] < df["dynamic_support"]
        adx_pass = (
            df["ADX"] > adx_thr
            if use_adx
            else pd.Series(True, index=df.index)
        )

        df["signal"] = 0
        df.loc[sticky.astype(bool) & long_cond & adx_pass,  "signal"] = 1
        df.loc[sticky.astype(bool) & short_cond & adx_pass, "signal"] = -1

        return df

    def get_fixed_params(self) -> dict[str, Any]:
        """Return fixed execution params (no SNR scaling).

        Returns config default values directly so all DL models use
        identical execution parameters, enabling fair comparison.

        Returns:
            Execution parameter dict compatible with
            ``GenericBacktestEngine.run_backtest()``.
        """
        tp = self._trade

        return {
            "tp_long":             tp["tp_long"],
            "sl_long":             tp["sl_long"],
            "tp_short":            tp["tp_short"],
            "sl_short":            tp["sl_short"],
            "max_hold":            tp["max_hold_hours"],
            "trailing_start_long": tp["trailing_stop_start_ratio"],
            "trailing_start_short": tp["trailing_stop_start_ratio"],
        }
