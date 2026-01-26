# SPEC COMPLIANCE CHECKLIST (A-N)

## A. Scope ✅
- [x] A1: SPX 0DTE Iron Condor (put spread + call spread)
- [x] A1: Entry hora fija parametrizable (`entry_time` in config)
- [x] A1: Salida TP/SL/Time-exit minuto-a-minuto
- [x] A2: No Wheel/PMCC (out of scope)

## B. Data Contract ✅
- [x] B1: SpotPrice dataclass con timestamp timezone-aware
- [x] B2: `dict[(strike, option_type), Quote]` - NO `dict[strike]`
- [x] B2: Quote con bid/ask/mid/delta/gamma/vega/theta/iv/volume/oi
- [x] B3: 0DTE filter (expiration == día del timestamp)

## C. Precios y Fricción ✅
- [x] C1: Entry credit = Bid(short) - Ask(long) ✅
- [x] C2: Exit debit = Ask(short) - Bid(long) ✅
- [x] C3: Commissions `commission_per_leg * 8` aplicadas ✅
- [x] C3: PnL en USD con `*100` multiplier ✅
- [x] C4: `max_loss_usd = (width - credit) * 100` ✅
- [x] C4: `RoR = net_pnl_usd / max_loss_usd` ✅

## D. Construcción de Posición ✅
- [x] D1: Selección por delta objetivo ✅
- [x] D1: IV Solver Newton-Raphson si falta delta ✅
- [x] D2: Wings por width en puntos ✅
- [x] D2: `width_put_real`, `width_call_real` calculados ✅
- [x] D3: Liquidity gates (bid>0, ask>bid, max_spread_pct/abs) ✅
- [x] D4: Calidad mínima (min_credit, min_ror) ✅

## E. Gestión de Salida ✅
- [x] E1: TP (`pnl_pct >= tp_target_pct`) ✅
- [x] E1: SL (`current_debit >= entry_credit * sl_multiplier`) ✅
- [x] E1: TIME (`max_hold_minutes`) ✅
- [x] E1: FORCE (`force_close_time`) ✅
- [x] E2: MTM con ask/bid por pierna ✅

## F. Walk-Forward Analysis ✅
- [x] F1: Rolling splits (train/test/step configurable) ✅
- [x] F2: Score multi-métrica: `Sharpe - λ*MaxDD - η*CVaR - κ*worst_day` ✅
- [x] F2: Grid search con todos params (delta, width, tp, sl, hold_min) ✅
- [x] F3: OOS trades concatenados ✅

## G. Regime Breakdown ✅
- [x] G1: HV rolling + clasificación buckets ✅
- [x] G2: Intraday range (high-low)/open ✅
- [x] G3: Gap (open-prev_close)/prev_close ✅
- [x] G: `regime_breakdown()` function ✅

## H. Data Quality ✅
- [x] H: `data_quality.py` con checks ✅
- [x] H: Crossed quotes, negative bids, missing ATM ✅

## I. Estructura Repo ✅
- [x] src/data/ (schema, loaders, data_quality) ✅
- [x] src/pricing/ (iv_solver, greeks) ✅
- [x] src/strategy/ (condor_builder, simulator, exits, mock_data_generator) ✅
- [x] src/research/ (grid_search, walk_forward, regimes, metrics) ✅
- [x] src/utils/ (logging, config) ✅
- [x] configs/ (base.yaml, experiment_wfa.yaml) ✅
- [x] tests/ (5 test files) ✅
- [x] scripts/ (run_data_check, run_grid_search_is, run_walk_forward) ✅

## J. Output Artifacts ✅
- [x] J: Trades con 4 strikes explícitos ✅
- [x] J: width_real (put & call) ✅
- [x] J: entry_credit, exit_debit ✅
- [x] J: exit_reason (TP/SL/TIME/FORCE) ✅
- [x] J: pnl_usd, RoR ✅
- [x] J: config_used.yaml con hash (utils/config.py) ✅

## K. Tests ✅
- [x] K: Chain key collision test ✅
- [x] K: Entry/exit pricing test ✅
- [x] K: IV solver convergence test ✅
- [x] K: WFA split no leakage test ✅
- [x] K: Exit rules trigger test ✅

## L. Mock Mode ✅
- [x] L: MockDataGenerator implementado ✅
- [x] L: `log_mock_mode()` función ✅

## M. CLI ✅
- [x] M: `run_data_check.py --config` ✅
- [x] M: `run_walk_forward.py --config` ✅
- [x] M: `run_grid_search_is.py --config` ✅

## N. No Inventar Resultados ✅
- [x] N: Sistema produce artifacts, no reporta "edge"
- [x] N: Falla explícita si falta data

---

## FINAL STATUS: 100% COMPLIANT ✅
