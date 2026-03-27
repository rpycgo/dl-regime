"""
dl_regime.signals
=================
Signal generation pipeline for DL regime models.

Exports
-------
RegimeLabelGenerator
    Generates binary regime labels from STRS-SDE regime_prob
    for DL supervised training.
DlRegimeSignalGenerator
    Converts DL model output (regime_prob) into signal/confidence
    columns using the same sticky filter + ADX gate as mdrs-sde.
"""
from dl_regime.signals.regime_signal import (
    DlRegimeSignalGenerator,
    RegimeLabelGenerator,
)

__all__ = ["RegimeLabelGenerator", "DlRegimeSignalGenerator"]
