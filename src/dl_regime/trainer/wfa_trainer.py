"""
dl_regime.trainer.wfa_trainer
==============================
Walk-forward analysis trainer for DL regime models.

Rolling-window WFA loop:

    for each test window:
        1. Slice rolling 3-month train set
        2. Generate binary labels from future returns (independent of
           MDRS-SDE — no regime_prob or MCMC estimates involved)
        3. Load checkpoint if exists, else train and save
        4. Predict regime_prob on test set
        5. Save predictions and log to MLflow

Signal generation and backtesting are handled downstream by the
quant-research backtesting module.

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
from dl_regime.signals.regime_signal import FutureReturnLabelGenerator

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
    regime_prob: pd.Series
    checkpoint_path: pathlib.Path
    metrics: dict[str, float] = field(default_factory=dict)


class WfaTrainer:
    """Rolling-window WFA trainer for DL regime models.

    Labels are generated from **future price returns** so that the DL
    benchmark is fully independent of MDRS-SDE.  No ``MdrsModeler`` or
    MCMC estimation is required.

    This trainer is responsible only for **training and prediction**.
    Signal generation and backtesting are handled by the quant-research
    backtesting module.

    Args:
        model_name: One of ``"lstm"``, ``"tcn"``, ``"transformer"``.
        config:     Merged config dict (``default_config.toml`` +
                    model-specific yaml).

    Example::

        trainer = WfaTrainer(model_name="lstm", config=config)
        predictions, summaries = trainer.run(full_data)
    """
    def __init__(
        self,
        model_name: str,
        config: dict[str, Any],
        ) -> None:
        if model_name not in _MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Choose from: {list(_MODEL_REGISTRY)}"
            )
        self._model_name = model_name
        self._cfg = config

        # WFA window settings
        wfa = config.get("walk_forward_settings", {})
        self._start = pd.Timestamp(wfa["start_date"])
        self._end = pd.Timestamp(wfa["end_date"])
        self._test_months = wfa.get("testing_months", 1)
        self._train_months = wfa.get("training_months", 3)

        # Training hyper-params
        train_cfg = config.get("training", {})
        self._seq_len: int = train_cfg.get("seq_len", 60)
        self._batch_size: int = train_cfg.get("batch_size", 256)
        self._max_epochs: int = train_cfg.get("max_epochs", 100)
        self._lr: float = train_cfg.get("learning_rate", 1e-3)
        self._patience: int = train_cfg.get("early_stopping_patience", 10)
        self._val_split: float = train_cfg.get("val_split", 0.2)
        self._seed: int = train_cfg.get("random_seed", 42)
        self._num_workers: int = train_cfg.get("num_workers", 0)

        # Input features
        self._features: list[str] = config.get("model", {}).get(
            "input_features",
            ["hybrid_z_score", "log_return", "direction_indicator",
             "volume_z_score", "absolute_return_z_score"],
        )

        # Checkpoint directory
        ckpt_cfg = config.get("checkpoint", {})
        self._ckpt_dir = pathlib.Path(
            ckpt_cfg.get("dirpath", "checkpoints")
        ) / model_name

        # MLflow
        mlflow_cfg = config.get("mlflow", {})
        mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri", "mlruns"))
        mlflow.set_experiment(mlflow_cfg.get("experiment_name", "dl-regime-btc"))

        # Label generator — future-return based (MDRS-SDE independent)
        label_cfg = config.get("label", {})
        self._label_gen = FutureReturnLabelGenerator(
            horizon=label_cfg.get("horizon", 12),
            threshold=label_cfg.get("threshold", 0.003),
            col_close=label_cfg.get("col_close", "Close"),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        full_data: pd.DataFrame,
        ) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
        """Execute the full rolling-window WFA.

        Args:
            full_data: Complete preprocessed dataset (must contain
                       feature columns and ``Close``).

        Returns:
            ``(predictions_df, metric_summaries)`` where
            ``predictions_df`` has columns ``[window, regime_prob]``
            indexed by the original datetime index.
        """
        test_starts = pd.date_range(
            start=self._start, end=self._end, freq="MS"
        )

        logger.info(
            "WFA: %d windows | model=%s", len(test_starts), self._model_name
        )

        results: list[WindowResult] = []
        for ts in test_starts:
            result = self._process_window(ts, full_data)
            if result is not None:
                results.append(result)

        return self._aggregate(results)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _process_window(
        self,
        test_start: pd.Timestamp,
        full_data: pd.DataFrame,
        ) -> WindowResult | None:
        label = test_start.strftime("%Y-%m-%d")
        test_end = min(
            test_start + relativedelta(months=self._test_months)
            - pd.Timedelta(seconds=1),
            self._end,
        )

        # Rolling window — 3 months before test_start (mirrors quant-research WalkForwardRunner)
        train_end   = test_start - pd.Timedelta(seconds=1)
        train_start = train_end - relativedelta(months=self._train_months)
        train_slice = full_data.loc[train_start:train_end].copy()
        test_slice = full_data.loc[test_start:test_end].copy()

        if len(train_slice) < 1000:
            logger.warning("Window %s skipped — insufficient rows.", label)
            return None

        # ── Label generation (future-return based, MDRS-SDE independent) ──
        train_slice = self._label_gen.generate(train_slice)

        pos_rate = train_slice["regime_label"].mean()
        logger.info(
            "Window %s: %d train rows, positive label rate %.1f%%",
            label, len(train_slice), pos_rate * 100,
        )

        # Load or train model
        ckpt_path = self._ckpt_dir / label / "model.ckpt"
        model = self._load_or_train(train_slice, label, ckpt_path)
        if model is None:
            return None

        # Predict on test set
        regime_prob = self._predict(model, test_slice)

        # MLflow logging
        with mlflow.start_run(
            run_name=f"{self._model_name}_{label}", nested=True
        ):
            mlflow.log_params({
                "model": self._model_name,
                "window": label,
                "label_horizon": self._label_gen.horizon,
                "label_threshold": self._label_gen.threshold,
                "train_rows": len(train_slice),
                "test_rows": len(test_slice),
                "positive_label_rate": round(pos_rate, 4),
            })
            mlflow.pytorch.log_model(model, "model")

        return WindowResult(
            window_label=label,
            regime_prob=pd.Series(
                regime_prob, index=test_slice.index, name="regime_prob"
            ),
            checkpoint_path=ckpt_path,
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
        # Train / val split (temporal — no shuffling of split boundary)
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
            enable_progress_bar=True,
            enable_model_summary=True,
            logger=True,
        )
        trainer.fit(model, train_loader, val_loader)

        # Save metadata
        meta_path = ckpt_path.parent / "metadata.json"
        with open(meta_path, "w") as fh:
            json.dump({
                "window": window_label,
                "model": self._model_name,
                "label_horizon": self._label_gen.horizon,
                "label_threshold": self._label_gen.threshold,
            }, fh)

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
        """Run inference and return regime_prob array aligned to test_df.

        Handles NaN rows (dropped by RegimeDataset) and seq_len padding
        so the output length always matches ``len(test_df)``.
        """
        test_df = test_df.copy()
        test_df["regime_label"] = 0

        # Identify valid (non-NaN) rows before dataset drops them
        valid_mask = test_df[self._features + ["regime_label"]].notna().all(axis=1)
        n_valid = valid_mask.sum()

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

        # len(ds) = n_valid - seq_len → prob covers indices [seq_len-1 .. n_valid-1]
        # Fill full-length array: 0.5 for non-predictable positions
        full_prob = np.full(len(test_df), 0.5, dtype=np.float32)
        valid_indices = np.where(valid_mask.values)[0]

        # Map model outputs to their original positions
        for i, p in enumerate(prob):
            original_idx = valid_indices[i + self._seq_len]
            full_prob[original_idx] = p

        return full_prob

    @staticmethod
    def _aggregate(
        results: list[WindowResult],
        ) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
        """Combine per-window predictions into a single DataFrame.

        Returns:
            ``(predictions_df, summaries)`` where ``predictions_df``
            concatenates all window regime_prob Series with a
            ``window`` column, and ``summaries`` maps window labels
            to metric dicts.
        """
        pred_frames = []
        summaries = {}
        for r in results:
            pdf = r.regime_prob.to_frame()
            pdf["window"] = r.window_label
            pred_frames.append(pdf)
            summaries[r.window_label] = r.metrics

        predictions = (
            pd.concat(pred_frames).sort_index()
            if pred_frames
            else pd.DataFrame(columns=["regime_prob", "window"])
        )

        return predictions, summaries
