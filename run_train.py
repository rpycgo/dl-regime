"""
run_train.py
============
CLI entry-point for DL regime model walk-forward training.

Trains LSTM / TCN / Transformer via expanding-window WFA and saves
per-window checkpoints to ``checkpoints/<model>/<window>/model.ckpt``
and regime_prob predictions to ``results/<model>_predictions.csv``.

All experiments are tracked in MLflow (``mlruns/``).

Signal generation and backtesting are handled downstream by the
quant-research backtesting module.

Usage
-----
::

    python run_train.py --model lstm
    python run_train.py --model tcn  --csv data/btcusdt_5m.csv
    python run_train.py --model transformer --horizon 12 --threshold 0.003
"""
from __future__ import annotations

import argparse
import logging
import pathlib
import sys
import tomllib

import pandas as pd
import yaml

_ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "src"))

from dl_regime import get_default_config_path
from dl_regime.trainer.wfa_trainer import WfaTrainer

from mdrs_sde import get_default_config_path as mdrs_config_path
from mdrs_sde.data.preprocessor import Preprocessor

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DL regime model WFA trainer")
    p.add_argument(
        "--model", type=str, required=True,
        choices=["lstm", "tcn", "transformer"],
        help="Model architecture to train.",
    )
    p.add_argument(
        "--csv", type=str, default=None,
        help="Path to pre-processed OHLCV CSV (5-min bars).",
    )
    p.add_argument(
        "--start", type=str, default=None,
        help="WFA start date override (ISO-8601).",
    )
    p.add_argument(
        "--end", type=str, default=None,
        help="WFA end date override (ISO-8601).",
    )
    p.add_argument(
        "--horizon", type=int, default=None,
        help="Label horizon override (bars ahead for future return).",
    )
    p.add_argument(
        "--threshold", type=float, default=None,
        help="Label threshold override (min absolute return for positive label).",
    )
    p.add_argument(
        "--config-dir", type=str, default="configs",
        help="Directory containing model yaml configs.",
    )
    return p.parse_args()


def _load_config(model: str, config_dir: str) -> dict:
    """Merge default_config.toml with model-specific yaml."""
    with open(get_default_config_path(), "rb") as fh:
        base = tomllib.load(fh)

    yaml_path = _ROOT / config_dir / f"{model}.yaml"
    if yaml_path.exists():
        with open(yaml_path) as fh:
            override = yaml.safe_load(fh)
        # Deep merge — override wins
        for k, v in override.items():
            if isinstance(v, dict) and k in base:
                base[k].update(v)
            else:
                base[k] = v

    return base


def main() -> int:
    args = _parse_args()

    # ── Config ──
    cfg = _load_config(args.model, args.config_dir)

    # WFA date overrides
    wfa = cfg.setdefault("walk_forward_settings", {})
    if args.start:
        wfa["start_date"] = args.start
    if args.end:
        wfa["end_date"] = args.end

    # Label overrides
    label_cfg = cfg.setdefault("label", {})
    if args.horizon is not None:
        label_cfg["horizon"] = args.horizon
    if args.threshold is not None:
        label_cfg["threshold"] = args.threshold

    # ── Data ──
    if args.csv:
        csv_path = pathlib.Path(args.csv)
    else:
        csv_path = _ROOT / "data" / "btcusdt_futures_5m.csv"

    if not csv_path.exists():
        log.error("CSV not found: %s", csv_path)
        return 1

    log.info("Loading data from %s", csv_path)
    raw = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    if raw.index.tz is not None:
        raw.index = raw.index.tz_convert("UTC").tz_localize(None)

    # ── Preprocessing (mdrs-sde feature engineering) ──
    with open(mdrs_config_path(), "rb") as fh:
        mdrs_cfg = tomllib.load(fh)

    # Slice to analysis window (avoid processing unnecessary early data)
    wfa_cfg = cfg.get("walk_forward_settings", {})
    data_start = wfa_cfg.get("data_start_date")
    if data_start:
        raw = raw.loc[data_start:]
        log.info("Sliced to analysis window from %s: %d rows.", data_start, len(raw))

    pre = Preprocessor(settings=mdrs_cfg["event_detection"])
    full_data = pre.run_full_pipeline(raw)
    full_data = full_data.dropna(subset=cfg["model"]["input_features"])
    log.info("Preprocessor applied: %d rows.", len(full_data))

    # ── WFA Training ──
    trainer = WfaTrainer(
        model_name=args.model,
        config=cfg,
    )

    log.info(
        "Starting WFA training: model=%s, horizon=%s, threshold=%s",
        args.model,
        cfg.get("label", {}).get("horizon", 12),
        cfg.get("label", {}).get("threshold", 0.003),
    )
    predictions, summaries = trainer.run(full_data)

    if predictions.empty:
        log.warning("No predictions generated.")
        return 1

    # Save predictions for downstream backtesting (quant-research)
    out_dir = _ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{args.model}_predictions.csv"
    predictions.to_csv(out_path)
    log.info("Predictions → %s (%d rows)", out_path, len(predictions))

    return 0


if __name__ == "__main__":
    sys.exit(main())
