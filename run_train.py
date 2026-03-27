"""
run_train.py
============
CLI entry-point for DL regime model walk-forward training.

Trains LSTM / TCN / Transformer via expanding-window WFA and saves
per-window checkpoints to ``checkpoints/<model>/<window>/model.ckpt``.
All experiments are tracked in MLflow (``mlruns/``).

Usage
-----
::

    python run_train.py --model lstm
    python run_train.py --model tcn  --csv data/btcusdt_1m.csv
    python run_train.py --model transformer --start 2024-01-01 --end 2026-01-31
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

from mdrs_sde.data.preprocessor import Preprocessor
from mdrs_sde.data.dataset_builder import DatasetBuilder
from mdrs_sde.models.sde_model import MdrsModeler

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
        help="Path to OHLCV CSV (1-minute bars). Skips data fetch.",
    )
    p.add_argument(
        "--events", type=str, default="data/events_btc_5m.toml",
        help="Path to events TOML file for STRS-SDE label generation.",
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

    # Config
    with open(_ROOT / "configs" / "backtest_settings.toml", "rb") as fh:
        bt_cfg = tomllib.load(fh)
    with open(_ROOT / "configs" / "data_settings.toml", "rb") as fh:
        ds_cfg = tomllib.load(fh)
    with open(_ROOT / "src" / "mdrs_sde" / "configs" / "default_config.toml", "rb") as fh:
        model_cfg = tomllib.load(fh)

    wfa_cfg = bt_cfg["walk_forward_settings"].copy()
    if args.start:
        wfa_cfg["start_date"] = args.start
    if args.end:
        wfa_cfg["end_date"] = args.end
    bt_cfg["walk_forward_settings"] = wfa_cfg

    dl_cfg = _load_config(args.model, args.config_dir)

    # Data
    if args.csv:
        raw = pd.read_csv(args.csv, index_col=0, parse_dates=True)
    else:
        csv_path = (
            _ROOT / "data"
            / ds_cfg["binance_collection"]["output_filename"]
        )
        if not csv_path.exists():
            log.error("CSV not found: %s", csv_path)
            return 1
        raw = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    if raw.index.tz is not None:
        raw.index = raw.index.tz_convert("UTC").tz_localize(None)

    pre = Preprocessor(settings=ds_cfg["event_detection"])
    full_data = pre.run_full_pipeline(raw)

    builder = DatasetBuilder(project_root=_ROOT)
    events = builder.load_events(pathlib.Path(args.events).name)
    full_data = builder.apply_event_tagging(full_data, events)
    train_data = builder.slice_training_data(full_data)

    # STRS-SDE modeler for per-window label generation
    merged_cfg = {**model_cfg, **bt_cfg}
    modeler = MdrsModeler(config=merged_cfg)

    # WFA training
    trainer = WfaTrainer(
        model_name=args.model,
        config=dl_cfg,
        bt_config=bt_cfg,
        mdrs_modeler=modeler,
    )

    log.info("Starting WFA training: model=%s", args.model)
    all_trades, summaries = trainer.run(full_data, train_data)

    if all_trades.empty:
        log.warning("No trades generated.")
        return 1

    out_dir = _ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    all_trades.to_csv(out_dir / f"{args.model}_trade_results.csv", index=False)
    log.info("Trades → results/%s_trade_results.csv", args.model)

    return 0


if __name__ == "__main__":
    sys.exit(main())
