# Implementation Plan: SPX 0DTE Iron Condor Validation

## Goal Description
Build a robust, scientific Python pipeline to validate an SPX 0DTE Iron Condor strategy. The system will handle intraday options data, enforce realistic trade fills (bid/ask), perform In-Sample optimization and Out-of-Sample Walk-Forward Analysis (WFA), and produce audit-ready metrics and artifacts.

## User Review Required
> [!IMPORTANT]
> - **Data Source**: Currently assuming a generic schema. Actual usage will require adapting `loaders.py` to the specific vendor format (e.g., CBOE, Databento).
> - **Mock Mode**: Initial development will use a `MockDataGenerator` to verify logic in the absence of the full intraday dataset.
> - **Performance**: Processing minute-level options chains for WFA can be computationally intensive; parallelization might be needed later.

## Proposed Changes

### Project Structure
Root: `ingresarios_options_research/`
Standard Python package structure with `src/`, `tests/`, `configs/`.

### Component: Data Layer (`src/data`)
#### [NEW] `schema.py`
- Define `Quote`, `OptionChain`, `SpotPrice` dataclasses/structures using `pydantic` or optimized `numpy`/`pandas` types.
- Enforce strict types: `timestamp`, `strike` (float), `option_type` ('C'/'P').

#### [NEW] `loaders.py`
- Functions to ingest parquet/csv data into the defined Schema.
- filtering for 0DTE.

#### [NEW] `data_quality.py`
- methods to check for missing 1m snapshots, crossed quotes, missing strikes.

### Component: Strategy Logic (`src/strategy` & `src/pricing`)
#### [NEW] `iv_solver.py`
- Newton-Raphson solver to extract Implied Volatility from price if missing.
- Greeks calculation (Delta) if missing from vendor.

#### [NEW] `condor_builder.py`
- Logic to select strikes based on Delta (e.g., 5-15 delta).
- Logic to select wings based on Width points.
- "Liquidity Gates": Check bid/ask spreads, reject bad trades.

#### [NEW] `exits.py`
- Minute-by-minute evaluation of conditions:
    - TP (Take Profit)
    - SL (Stop Loss provided as multiplier of credit)
    - Time Exit (e.g., 15:45)

#### [NEW] `simulator.py`
- The core event loop:
    1. Filter day's data.
    2. At entry time, call `condor_builder`.
    3. If filled, iterate minute-by-minute calling `exits`.
    4. Log trade result.

### Component: Research & Analysis (`src/research`)
#### [NEW] `metrics.py`
- Calculate RoR, Sharpe, MaxDD, Win Rate.
- Custom score: `score = sharpe(RoR) - λ*MaxDD ...`

#### [NEW] `walk_forward.py`
- Rolling window logic (Train on N days, Test on M days).
- Parameter optimization grid.

#### [NEW] `regimes.py`
- Helper to classify days/periods into Volatility Regimes (IV/HV) and Range Regimes.

### Component: Scripts (`scripts/`)
- `run_data_check.py`
- `run_grid_search_is.py`
- `run_walk_forward.py`

## Verification Plan

### Automated Tests
- `pytest` suite in `tests/` covering:
    - Strike selection logic (delta accuracy).
    - PnL calculations (checking bid/ask math).
    - WFA splitting logic (ensuring no look-ahead bias).

### Manual Verification
- **Mock Run**: Execute `scripts/run_walk_forward.py` with `MockDataGenerator` enabled.
- **Audit**: Inspect `trades.parquet` output to verify individual trade entry/exit prices match the simulated "market" quotes.
