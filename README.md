# dl-regime

Deep learning regime detection baselines for Bitcoin perpetual futures.

Provides LSTM, TCN, and Transformer models that predict volatility breakout regimes, serving as independent benchmarks against the [MDRS-SDE](https://github.com/rpycgo-research/mdrs-sde) model.

## Architecture

```
dl-regime/                  ← Training & prediction
├── run_train.py            ← CLI: WFA training
├── src/dl_regime/
│   ├── models/             ← LSTM, TCN, Transformer (PyTorch Lightning)
│   ├── trainer/            ← Expanding-window WFA trainer
│   ├── signals/            ← Future-return label generator
│   ├── data/               ← Sliding-window dataset
│   └── configs/            ← Default hyperparameters
└── configs/                ← Model-specific YAML overrides

quant-research/             ← Backtesting (separate repo)
└── src/backtesting/
    └── models/adapters/
        └── dl_regime.py    ← Loads checkpoints → runs backtest
```

Training and backtesting are decoupled:

- **dl-regime** handles model training and checkpoint generation
- **[quant-research](https://github.com/rpycgo/quant-research)** handles backtesting via `DlRegimeCryptoAdapter`, using the same `WalkForwardRunner` and `GenericBacktestEngine` as MDRS-SDE

## Installation

### Prerequisites

- Python >=3.11, <3.14
- [uv](https://docs.astral.sh/uv/) (recommended package manager)
- CUDA 12.4 compatible GPU

### Setup

```bash
git clone https://github.com/rpycgo/dl-regime.git
cd dl-regime
uv sync
```

`uv sync` resolves all dependencies from `pyproject.toml` including:

- `torch==2.6.0` from the PyTorch CUDA 12.4 index
- `mdrs-sde` from GitHub (preprocessing pipeline)
- `lightning`, `mlflow`, `scikit-learn`, `pandas`, `pyyaml`

### As a dependency (for quant-research)

```toml
# In quant-research/pyproject.toml
[project]
dependencies = [
    "dl-regime",
]

[tool.uv.sources]
dl-regime = { git = "https://github.com/rpycgo/dl-regime.git" }
```

## Usage

### Training

```bash
# Train LSTM with default settings
python run_train.py --model lstm

# Train TCN with custom label settings
python run_train.py --model tcn --horizon 12 --threshold 0.003

# Train Transformer with custom data and date range
python run_train.py --model transformer --csv data/btcusdt_futures_5m.csv \
    --start 2024-01-01 --end 2026-01-31
```

Outputs:
- `checkpoints/<model>/<window>/model.ckpt` — per-window model weights
- `results/<model>_predictions.csv` — regime probability predictions
- `mlruns/` — MLflow experiment tracking

### Backtesting (via quant-research)

```bash
cd quant-research
python run_backtest.py --model dl_regime_lstm_btc --symbol BTCUSDT
python run_backtest.py --model dl_regime_tcn_btc --symbol BTCUSDT
python run_backtest.py --model dl_regime_transformer_btc --symbol BTCUSDT
```

## Label Design

DL models are supervised with **future-return based labels**, fully independent of MDRS-SDE:

```
future_return = close.shift(-horizon) / close - 1
label = 1  if |future_return| > threshold  else 0
```

Default: `horizon=12` (1 hour at 5-min bars), `threshold=0.003` (0.3%).

Models learn to predict volatility breakouts (either direction), matching the semantic of MDRS-SDE's regime probability without sharing any supervision signal.

## Models

| Model | Architecture | Reference |
|-------|-------------|-----------|
| LSTM | Bidirectional-optional stacked LSTM | Hochreiter & Schmidhuber (1997) |
| TCN | Dilated causal convolutions with residual connections | Bai et al. (2018) |
| Transformer | Encoder-only with sinusoidal positional encoding | Vaswani et al. (2017) |

All models output `regime_prob ∈ [0, 1]` via sigmoid, trained with BCE loss.

## Configuration

Hierarchical config: `default_config.toml` → model-specific `configs/<model>.yaml` → CLI overrides.

Key settings in `default_config.toml`:

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `[label]` | `horizon` | 12 | Forward bars for return calculation |
| `[label]` | `threshold` | 0.003 | Min absolute return for positive label |
| `[training]` | `seq_len` | 60 | Input sequence length (bars) |
| `[training]` | `max_epochs` | 100 | Maximum training epochs |
| `[walk_forward_settings]` | `start_date` | 2024-01-01 | WFA start |
| `[walk_forward_settings]` | `testing_months` | 1 | Test window size |

## Project Context

This package provides DL baselines for the paper:

> *Persistence-Aware Regime Signal Stabilization: An Intelligent Trading System for Bitcoin Perpetual Futures*

The benchmarking design ensures fair comparison: DL models and MDRS-SDE share the same input features, backtesting engine, and evaluation metrics, differing only in the modeling approach (neural network vs. Bayesian SDE estimation with structural stabilization).
