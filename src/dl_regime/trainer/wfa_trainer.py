"""
dl_regime.trainer.wfa_trainer
==============================
Walk-forward analysis trainer for DL regime models.

Expanding-window WFA loop:

    for each test window:
        1. Slice expanding train set
        2. Generate regime labels from STRS-SDE (per-window k, gamma)
        3. Load checkpoint if exists, else train and save
        4. Predict regime_prob on test set
        5. Convert to signal via DlRegimeSignalGenerator (fixed params)
        6. Run backtest via GenericBacktestEngine
        7. Log metrics to MLflow

All three DL models (LSTM, TCN, Transformer) share this trainer.
The model architecture is selected via ``model_name`` argument.
"""
from __future__ import annotations

import json
import logging
import pathlib
from dataclasses import dataclass, field
from typing import Any

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
from dateutil.relativedelta import relativedelta
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader

from dl_regime.data.dataset import RegimeDataset
from dl_regime.models import (
    LSTMRegimeModel,
    TCNRegimeModel,
    TransformerRegimeModel,
)
from dl_regime.signals.regime_signal import (
    DlRegimeSignalGenerator,
    RegimeLabelGenerator,
)

logger = logging.getLogger(__name__)

_MODEL_REGISTRY = {
    "lstm":        LSTMRegimeModel,
    "tcn":         TCNRegimeModel,
    "transformer": TransformerRegimeModel,
}


@dataclass
class WindowResult:
    """Output container for one WFA window."""
    window_label: str
    trades: pd.DataFrame
    regime_prob: pd.Series
    checkpoint_path: pathlib.Path
    metrics: dict[str, float] = field(default_factory=dict)


class WfaTrainer:
    """Expanding-window WFA trainer for DL regime models.

    Args:
        model_name:   One of ``"lstm"``, ``"tcn"``, ``"transformer"``.
        config:       Merged config dict (``default_config.toml`` +
                      model-specific yaml).
        bt_config:    Parsed ``backtest_settings.toml``.
        mdrs_modeler: Fitted ``MdrsModeler`` instance used to generate
                      per-window regime labels.  Must expose
                      ``estimate_parameters()``.

    Example::

        trainer = WfaTrainer(
            model_name="lstm",
            config=config,
            bt_config=bt_cfg,
            mdrs_modeler=modeler,
        )
        all_trades, summaries = trainer.run(full_data, train_data)
    """
    def __init__(
        self,
        model_name: str,
        config: dict[str, Any],
        bt_config: dict[str, Any],
        mdrs_modeler: Any,
        ) -> None:
        if model_name not in _MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Choose from: {list(_MODEL_REGISTRY)}"
            )
        self._model_name = model_name
        self._cfg = config
        self._bt_cfg = bt_config
        self._modeler = mdrs_modeler

        wfa = bt_config["walk_forward_settings"]
        self._start = pd.Timestamp(wfa["start_date"])
        self._end = pd.Timestamp(wfa["end_date"])
        self._test_months: int = wfa.get("testing_months", 1)

        train_cfg = config.get("training", {})
        self._seq_len: int = train_cfg.get("seq_len", 60)
        self._batch_size: int = train_cfg.get("batch_size", 256)
        self._max_epochs: int = train_cfg.get("max_epochs", 100)
        self._lr: float = train_cfg.get("learning_rate", 1e-3)
        self._patience: int = train_cfg.get("early_stopping_patience", 10)
        self._val_split: float = train_cfg.get("val_split", 0.2)
        self._seed: int = train_cfg.get("random_seed", 42)
        self._num_workers: int = train_cfg.get("num_workers", 0)

        self._features: list[str] = config.get("model", {}).get(
            "input_features",
            ["hybrid_z_score", "log_return", "direction_indicator",
             "volume_z_score", "absolute_return_z_score"],
        )

        ckpt_cfg = config.get("checkpoint", {})
        self._ckpt_dir = pathlib.Path(
            ckpt_cfg.get("dirpath", "checkpoints")
        ) / model_name

        mlflow_cfg = config.get("mlflow", {})
        mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri", "mlruns"))
        mlflow.set_experiment(mlflow_cfg.get("experiment_name", "dl-regime-btc"))

        self._label_gen = RegimeLabelGenerator(
            entry_threshold=bt_config.get("risk_management", {}).get(
                "entry_probability_threshold", 0.5
            )
        )
        self._signal_gen = DlRegimeSignalGenerator(
            risk_config=bt_config.get("risk_management", {}),
            filter_config=bt_config.get("filters", {}),
            trade_config=bt_config.get("trading_parameters", {}),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        full_data: pd.DataFrame,
        train_data: pd.DataFrame | None = None,
        ) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
        """Execute the full expanding-window WFA.

        Args:
            full_data:  Complete preprocessed dataset.
            train_data: MCMC-eligible subset (in-zone rows) used for
                        STRS-SDE label generation.  Falls back to
                        ``full_data`` when ``None``.

        Returns:
            ``(all_trades, metric_summaries)``
        """
        from backtesting.engines.engine import GenericBacktestEngine
        from backtesting.engines.performance import PerformanceAnalyzer

        if train_data is None:
            train_data = full_data

        engine = GenericBacktestEngine(config=self._bt_cfg)
        test_starts = pd.date_range(
            start=self._start, end=self._end, freq="MS"
        )

        logger.info(
            "WFA: %d windows | model=%s", len(test_starts), self._model_name
        )

        results: list[WindowResult] = []
        for ts in test_starts:
            result = self._process_window(ts, train_data, full_data, engine)
            if result is not None:
                results.append(result)

        return self._aggregate(results)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _process_window(
        self,
        test_start: pd.Timestamp,
        train_data: pd.DataFrame,
        full_data: pd.DataFrame,
        engine: Any,
        ) -> WindowResult | None:
        label = test_start.strftime("%Y-%m-%d")
        test_end = min(
            test_start + relativedelta(months=self._test_months)
            - pd.Timedelta(seconds=1),
            self._end,
        )

        # Expanding window — all data before test_start
        train_end = test_start - pd.Timedelta(seconds=1)
        train_slice = full_data.loc[:train_end].copy()
        test_slice = full_data.loc[test_start:test_end].copy()

        if len(train_slice) < 1000:
            logger.warning("Window %s skipped — insufficient rows.", label)
            return None

        # Generate per-window labels from STRS-SDE
        in_zone_slice = train_data.loc[:train_end].copy()
        if len(in_zone_slice) < 100:
            logger.warning(
                "Window %s: insufficient in-zone rows for STRS-SDE label.",
                label,
            )
            return None

        _trace, _summary, estimates = self._modeler.estimate_parameters(
            z_values=in_zone_slice["hybrid_z_score"].values,
            returns_scaled=in_zone_slice["log_return"].values * 100,
            direction=in_zone_slice["direction_indicator"].values,
        )
        if estimates is None:
            logger.warning("Window %s: STRS-SDE estimation failed.", label)
            return None

        k = float(estimates["k"])
        gamma = float(estimates["gamma"])
        entry_thr = self._bt_cfg.get("risk_management", {}).get(
            "entry_probability_threshold", 0.5
        )
        train_slice["regime_prob_sde"] = 1.0 / (
            1.0 + np.exp(-k * (train_slice["hybrid_z_score"] - gamma))
        )
        train_slice = self._label_gen.generate(
            train_slice, train_slice["regime_prob_sde"]
        )

        # Load or train model
        ckpt_path = self._ckpt_dir / label / "model.ckpt"
        model = self._load_or_train(train_slice, label, ckpt_path)
        if model is None:
            return None

        # Predict
        regime_prob = self._predict(model, test_slice)

        # Signal + backtest
        signal_df = self._signal_gen.generate(test_slice, regime_prob)
        fixed_params = self._signal_gen.get_fixed_params()
        trades = engine.run_backtest(signal_df, fixed_params)

        # Metrics
        metrics: dict[str, float] = {}
        if not trades.empty:
            from backtesting.engines.performance import PerformanceAnalyzer
            m, _eq, _dd = PerformanceAnalyzer.calculate_metrics(trades)
            if m:
                metrics = {
                    "sharpe": m["sharpe_ratio"],
                    "mdd": m["max_drawdown_pct"],
                    "return": m["total_return_pct"],
                    "win_rate": m["win_rate_pct"],
                }

        # MLflow logging
        with mlflow.start_run(
            run_name=f"{self._model_name}_{label}", nested=True
        ):
            mlflow.log_params({
                "model": self._model_name,
                "window": label,
                "k": k,
                "gamma": gamma,
            })
            mlflow.log_metrics(metrics)
            mlflow.pytorch.log_model(model, "model")

        return WindowResult(
            window_label=label,
            trades=trades,
            regime_prob=pd.Series(
                regime_prob, index=test_slice.index, name="regime_prob"
            ),
            checkpoint_path=ckpt_path,
            metrics=metrics,
        )

    def _load_or_train(
        self,
        train_df: pd.DataFrame,
        window_label: str,
        ckpt_path: pathlib.Path,
        ) -> Any | None:
        if ckpt_path.exists():
            logger.info("Loading checkpoint: %s", ckpt_path)
            return self._load_checkpoint(ckpt_path)

        logger.info("Training window %s …", window_label)

        return self._train(train_df, window_label, ckpt_path)

    def _train(
        self,
        train_df: pd.DataFrame,
        window_label: str,
        ckpt_path: pathlib.Path,
        ) -> Any | None:
        # Train / val split
        n = len(train_df)
        val_n = max(1, int(n * self._val_split))
        train_part = train_df.iloc[:-val_n]
        val_part = train_df.iloc[-val_n:]

        train_ds = RegimeDataset(train_part, self._features, self._seq_len)
        val_ds = RegimeDataset(
            val_part, self._features, self._seq_len,
            scaler=train_ds.scaler,
        )

        if len(train_ds) < self._seq_len or len(val_ds) < self._seq_len:
            logger.warning(
                "Window %s: dataset too small after seq_len slicing.",
                window_label,
            )
            return None

        train_loader = DataLoader(
            train_ds, batch_size=self._batch_size,
            shuffle=True, num_workers=self._num_workers,
        )
        val_loader = DataLoader(
            val_ds, batch_size=self._batch_size,
            shuffle=False, num_workers=self._num_workers,
        )

        model = self._build_model(len(self._features))

        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        callbacks = [
            EarlyStopping(
                monitor="val_loss", patience=self._patience, mode="min"
            ),
            ModelCheckpoint(
                dirpath=str(ckpt_path.parent),
                filename="model",
                monitor="val_loss",
                save_top_k=1,
                mode="min",
            ),
        ]

        trainer = Trainer(
            max_epochs=self._max_epochs,
            callbacks=callbacks,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
        )
        trainer.fit(model, train_loader, val_loader)

        # Save metadata
        meta_path = ckpt_path.parent / "metadata.json"
        with open(meta_path, "w") as fh:
            json.dump({"window": window_label, "model": self._model_name}, fh)

        return model

    def _load_checkpoint(self, ckpt_path: pathlib.Path) -> Any:
        model_cls = _MODEL_REGISTRY[self._model_name]

        return model_cls.load_from_checkpoint(str(ckpt_path))

    def _build_model(self, input_size: int) -> Any:
        model_cls = _MODEL_REGISTRY[self._model_name]
        cfg = self._cfg.get(self._model_name, {})

        return model_cls(
            input_size=input_size,
            learning_rate=self._lr,
            **{k: v for k, v in cfg.items()
               if k not in ("input_size", "learning_rate")},
        )

    def _predict(
        self,
        model: Any,
        test_df: pd.DataFrame,
        ) -> np.ndarray:
        """Run inference and return regime_prob array."""
        # Build dataset with training scaler — use dummy labels
        test_df = test_df.copy()
        test_df["regime_label"] = 0

        # Reuse scaler stored in model hparams is not straightforward;
        # fit a new scaler on test features (acceptable for inference)
        ds = RegimeDataset(test_df, self._features, self._seq_len)
        loader = DataLoader(
            ds, batch_size=self._batch_size,
            shuffle=False, num_workers=self._num_workers,
        )

        trainer = Trainer(
            enable_progress_bar=False,
            logger=False,
        )
        preds = trainer.predict(model, loader)
        prob = torch.cat(preds).numpy()

        # Pad the first seq_len - 1 rows with 0.5 (no signal)
        pad = np.full(self._seq_len - 1, 0.5, dtype=np.float32)

        return np.concatenate([pad, prob])

    @staticmethod
    def _aggregate(
        results: list[WindowResult],
        ) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
        trade_frames = [r.trades for r in results if not r.trades.empty]
        summaries = {r.window_label: r.metrics for r in results}

        all_trades = (
            pd.concat(trade_frames)
            .sort_values("entry_time")
            .reset_index(drop=True)
            if trade_frames
            else pd.DataFrame()
        )

        return all_trades, summaries
