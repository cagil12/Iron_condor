# SPX/XSP 0DTE Iron Condor Validation Pipeline

Scientific validation framework for SPX 0DTE Iron Condor strategies (Quant V2.0).
Now supports **Live/Paper Execution on XSP** via Interactive Brokers (IBKR).

## Features

- **4-leg Iron Condor** with conservative bid/ask fills
- **Delta-based strike selection** (0.10 Delta) with Real-time Greeks
- **Live Execution Engine** (IBKR TWS) with Combo/BAG Orders
- **Forensic Journaling**: Captures Slippage, MAE, and Bid/Ask Snapshots
- **Walk-Forward Analysis** with rolling train/test splits
- **Investment-grade metrics**: Sharpe(RoR), CVaR, MaxDD, Profit Factor

## Installation

```bash
pip install -e .
```

## Usage

### Live/Paper Trading (IBKR)

```bash
# Run Live Monitor (Paper Mode by default)
python run_live_monitor.py

# Run Live Monitor (LIVE MONEY - CAUTION)
python run_live_monitor.py --live
```

### Research & Backtesting

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
  data/       # IBKR Connector, Schema, loaders
  pricing/    # IV solver, Greeks
  strategy/   # LiveExecutor, Condor builder, simulator
  research/   # Grid search, WFA, metrics, regimes
  utils/      # TradeJournal, logging, config
configs/      # YAML configurations
tests/        # Unit tests
scripts/      # CLI entry points
```

## Configuration

See `configs/base.yaml` for default parameters and `configs/experiment_wfa.yaml` for WFA-specific settings.

## License

MIT
