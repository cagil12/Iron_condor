# SPX 0DTE Iron Condor Validation Pipeline

Scientific validation framework for SPX 0DTE Iron Condor strategies (Quant V2.0).

## Features

- **4-leg Iron Condor** with conservative bid/ask fills
- **Delta-based strike selection** with IV solver fallback
- **Walk-Forward Analysis** with rolling train/test splits
- **Regime breakdown** (HV, Range, Gap)
- **Investment-grade metrics**: Sharpe(RoR), CVaR, MaxDD, Profit Factor

## Installation

```bash
pip install -e .
```

## Usage

```bash
# Set PYTHONPATH
$env:PYTHONPATH = "path/to/parent/folder"

# Run WFA with mock data
python scripts/run_walk_forward.py --mock

# Run data quality check
python scripts/run_data_check.py --mock
```

## Project Structure

```
src/
  data/       # Schema, loaders, data quality
  pricing/    # IV solver, Greeks
  strategy/   # Condor builder, simulator, exits
  research/   # Grid search, WFA, metrics, regimes
  utils/      # Logging, config
configs/      # YAML configurations
tests/        # Unit tests
scripts/      # CLI entry points
```

## Configuration

See `configs/base.yaml` for default parameters and `configs/experiment_wfa.yaml` for WFA-specific settings.

## License

MIT
