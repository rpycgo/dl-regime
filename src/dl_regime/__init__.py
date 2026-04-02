"""
dl_regime
=========
Deep learning regime detection benchmarks for Bitcoin perpetual futures.

Provides LSTM, TCN, and Transformer implementations that share the same
fit() / predict() interface as mdrs_sde.models.MdrsModeler, enabling
direct walk-forward comparison within the quant-research backtesting
framework.

All models:
- Accept the same input features as STRS-SDE (hybrid_z_score, log_return, ...)
- Output regime_prob in [0, 1] — same semantic as STRS-SDE sigmoid output
- Pass through the same sticky filter + ADX gate in RegimeSignalGenerator
- Use fixed execution params (no SNR scaling) for fair comparison

Quickstart
----------
::

    from dl_regime import get_default_config_path
    from dl_regime.trainer.wfa_trainer import WfaTrainer

    trainer = WfaTrainer(model_name="lstm", config_path=get_default_config_path())
    trainer.run(full_data, train_data)
"""

from importlib.resources import files

__version__ = "0.1.0"
__all__ = ["get_default_config_path"]


def get_default_config_path() -> str:
    """Return absolute path to the package-bundled default config.

    Returns:
        Absolute path string to ``dl_regime/configs/default_config.toml``.
    """
    return str(
        files("dl_regime.configs").joinpath("default_config.toml")
    )
