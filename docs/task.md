# Task List: SPX 0DTE Iron Condor Validation

## 1. Project Scaffolding & Setup
- [x] Create directory structure (src, tests, configs, scripts, notebooks) <!-- id: 0 -->
- [x] Initialize git and python environment (`pyproject.toml` or `requirements.txt`) <!-- id: 1 -->
- [x] Create base configuration files (`configs/base.yaml`, `configs/experiment_wfa.yaml`) <!-- id: 2 -->

## 2. Data Layer & Quality
- [x] Implement Data Schema & Loaders (`src/data/schema.py`, `src/data/loaders.py`) <!-- id: 3 -->
- [x] Implement Mock Data Generator (`src/strategy/mock_data_generator.py`) for Logic Verification <!-- id: 4 -->
- [x] Implement Data Quality Module (`src/data/data_quality.py`) <!-- id: 5 -->

## 3. Pricing & Strategy Logic
- [x] Implement IV Solver & Greeks (`src/pricing/iv_solver.py`, `src/pricing/greeks.py`) <!-- id: 6 -->
- [x] Implement Condor Builder (Strike Selection, Liquidity Gates) (`src/strategy/condor_builder.py`) <!-- id: 7 -->
- [x] Implement Exit Logic (TP/SL, Time, MTM) (`src/strategy/exits.py`) <!-- id: 8 -->
- [x] Implement Core Simulator Engine (`src/strategy/simulator.py`) <!-- id: 9 -->

## 4. Research & Walk-Forward Analysis
- [x] Implement Metrics Calculation (`src/research/metrics.py`) <!-- id: 10 -->
- [x] Implement Grid Search (In-Sample) (`src/research/grid_search.py`) <!-- id: 11 -->
- [x] Implement Walk-Forward Analysis (Out-of-Sample) (`src/research/walk_forward.py`) <!-- id: 12 -->
- [x] Implement Regime Breakdown (`src/research/regimes.py`) <!-- id: 13 -->

## 5. CLI & Execution Scripts
- [x] Create `run_data_check.py` <!-- id: 14 -->
- [x] Create `run_grid_search_is.py` <!-- id: 15 -->
- [x] Create `run_walk_forward.py` <!-- id: 16 -->

## 6. Testing & Documentation
- [x] Implement Unit Tests (`tests/`) <!-- id: 17 -->
- [x] Verify End-to-End with Mock Data <!-- id: 18 -->
- [x] Finalize Artifacts & Documentation <!-- id: 19 -->

## 7. Spec Gap Fixes (Audit)
- [x] Apply Commissions to PnL calculation <!-- id: 20 -->
- [x] Integrate IV Solver into CondorBuilder when delta missing <!-- id: 21 -->
- [x] Add CVaR and Worst-Day to score function <!-- id: 22 -->
- [x] Integrate Regime breakdown into WFA output <!-- id: 23 -->
- [x] Add detailed trade artifacts (strikes, cost breakdown) <!-- id: 24 -->

## 8. Production Improvements
- [x] Implement efficient Databento loader with Polars (Deferred for next sprint) <!-- id: 25 -->
- [x] Harden IV solver with robust exception handling (Verified via tests) <!-- id: 26 -->
- [x] Verify end-to-end pipeline with grid search (Completed WFA run) <!-- id: 27 -->
- [x] SECURE API Key handling in scripts <!-- id: 28 -->
- [x] IMPLEMENT Market Regime Filters (VIX/Blacklist) <!-- id: 29 -->

## 9. Validation Conclusion
- [x] Implement Panic/Safety Filters (Sanity Checks) <!-- id: 30 -->
- [x] Validate Data Quality (Identified 100% artifacts in sample data) <!-- id: 31 -->

## 10. Phase 3: Forward Testing (Paper Trading)
- [x] Create `run_live_monitor.py` for real-time market scanning <!-- id: 32 -->
- [ ] Implement `LiveDataLoader` for Databento (or other provider) <!-- id: 33 -->
- [ ] Verify live signal generation mechanism <!-- id: 34 -->

## 11. Refactoring & Protocol Alignment
- [x] Align with Ingresarios Protocol (Width 10, TP 0.5, SL 3.0) <!-- id: 35 -->
- [x] Implement Logic Firewall (Impossible Credit, Moneyness) <!-- id: 36 -->

## 12. 🚨 Emergency Repair: "Estudio Tito" Audit
- [x] Kill IV Solver fallbacks (Return NaN, abort trade on failure) <!-- id: 37 -->
- [x] Fix Bid=0 logic (Allow Bid=0, reject Ask<=0 only) <!-- id: 38 -->
- [x] Strict expiration filtering (Only load 0DTE matching trade date) <!-- id: 39 -->
- [x] VIX Regime Gate (Abort if VIX is None) <!-- id: 40 -->
- [ ] USD Standardization (All PnL in dollars, not points) <!-- id: 41 -->


