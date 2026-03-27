"""
dl_regime.trainer
=================
Walk-forward training and backtesting orchestration.

Exports
-------
WfaTrainer
    Orchestrates the expanding-window WFA loop:
    fit (or load checkpoint) → predict → backtest → MLflow log.
"""
from dl_regime.trainer.wfa_trainer import WfaTrainer

__all__ = ["WfaTrainer"]
