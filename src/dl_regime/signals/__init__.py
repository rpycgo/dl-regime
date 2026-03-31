"""
dl_regime.signals
=================
Label generation for DL regime models.

Exports
-------
FutureReturnLabelGenerator
    Generates binary labels from future price returns for
    DL supervised training (independent of MDRS-SDE).
RegimeLabelGenerator
    Backward-compatible alias for FutureReturnLabelGenerator.
"""
from dl_regime.signals.regime_signal import (
    FutureReturnLabelGenerator,
    RegimeLabelGenerator,
)

__all__ = ["FutureReturnLabelGenerator", "RegimeLabelGenerator"]
