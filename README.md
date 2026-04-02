# dl-regime

Deep learning regime detection library for Bitcoin perpetual futures.

Provides LSTM, TCN, and Transformer implementations that share the same walk-forward training interface, producing `regime_prob` predictions consumed downstream by the `quant-research` backtesting platform.

## Overview

`dl-regime` is a **pure library**. It owns model implementation, walk-forward training, and label generation. All execution entry points (`qr-dl-train`, `qr-backtest`) live in [quant-research](https://github.com/rpycgo/quant-research).

### Architecture

```
dl-regime                  quant-research
──────────────────         ──────────────────────────────
models/                →   strategies/dl_regime/
  LSTMRegimeModel            cli/train.py  (qr-dl-train)
  TCNRegimeModel             strategy.py   (qr-backtest)
  TransformerRegimeModel
trainer/
  WfaTrainer
signals/
  FutureReturnLabelGenerator
data/
  RegimeDataset
```

### Dependency

`dl-regime` depends on [mdrs-sde](https://github.com/rpycgo-research/mdrs-sde) for microstructure feature engineering (`Preprocessor`). All DL models share the same input features as MDRS-SDE, enabling direct walk-forward comparison within the `quant-research` backtesting framework.

```
quant-research → dl-regime → mdrs-sde
quant-research → mdrs-sde
```

## Models

All models inherit `BaseRegimeModule` (PyTorch Lightning) and implement a common interface:

- Input: `(batch, seq_len, n_features)` float32 tensor
- Output: `regime_prob` in `[0, 1]` — same semantic as MDRS-SDE sigmoid output

| Model | Class | Description |
|---|---|---|
| LSTM | `LSTMRegimeModel` | Stacked LSTM with dropout |
| TCN | `TCNRegimeModel` | Temporal Convolutional Network |
| Transformer | `TransformerRegimeModel` | Encoder-only Transformer |

## Label generation

Labels are derived from **future price returns**, independent of MDRS-SDE:

```
future_return = close.shift(-horizon) / close - 1
regime_label  = (|future_return| > threshold) ? 1 : 0
```

This makes DL models genuine external benchmarks — no MCMC estimates are used for supervision.

## Installation

```bash
uv add "dl-regime @ git+https://github.com/rpycgo/dl-regime.git"
```

> Training and backtesting are executed via `quant-research`. See [quant-research](https://github.com/rpycgo/quant-research) for usage.

## Library usage

```python
from dl_regime import get_default_config_path
from dl_regime.trainer.wfa_trainer import WfaTrainer
import tomllib

with open(get_default_config_path(), "rb") as f:
    config = tomllib.load(f)

trainer = WfaTrainer(model_name="lstm", config=config)
predictions, summaries = trainer.run(full_data)
```

## Requirements

- Python `>=3.11, <3.14`
- PyTorch `2.6.0` (CUDA 12.4)
- Lightning `>=2.6.1`
- mlflow `>=3.10.1`
- mdrs-sde (git)