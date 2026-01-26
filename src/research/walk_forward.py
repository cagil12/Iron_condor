import pandas as pd
from typing import List, Dict, Any
from .grid_search import GridSearch
from ..strategy.simulator import Simulator
from .metrics import calculate_metrics

class WalkForwardAnalysis:
    def __init__(self, config: dict, data_loader, train_window_days: int, test_window_days: int, step_days: int):
        self.config = config
        self.loader = data_loader
        self.train_window = train_window_days
        self.test_window = test_window_days
        self.step_days = step_days
        self.oos_trades = []
        self.fold_results = []

    def run(self, all_dates: List[Any], param_grid: Dict[str, List[Any]]):
        """
        Executes rolling WFA.
        """
        # Sort dates
        dates = sorted(all_dates)
        total_days = len(dates)
        
        # Start index
        start_idx = 0
        
        while start_idx + self.train_window + self.test_window <= total_days:
            # Define Split
            train_end_idx = start_idx + self.train_window
            test_end_idx = train_end_idx + self.test_window
            
            train_dates = dates[start_idx:train_end_idx]
            test_dates = dates[train_end_idx:test_end_idx]
            
            print(f"Fold: Train {train_dates[0]} to {train_dates[-1]} | Test {test_dates[0]} to {test_dates[-1]}")
            
            # 1. Optimize In-Sample
            gs = GridSearch(self.config, param_grid, self.loader)
            df_results = gs.run(train_dates)
            
            if df_results.empty:
                print("No trades in train set, skipping fold.")
                start_idx += self.step_days
                continue
                
            # Pick Best Params (by Score)
            best_params = df_results.sort_values(by='score', ascending=False).iloc[0].to_dict()
            # Filter out metric columns to get just params
            # Simplified: assuming clean param dict available or filtering keys from param_grid
            clean_params = {k: best_params[k] for k in param_grid.keys() if k in best_params}
            
            print(f"Best Params: {clean_params} (Score: {best_params.get('score', 0):.2f})")
            
            # 2. Validate Out-of-Sample
            # Construct config with best params
            test_config = self.config.copy()
            if 'target_delta' in clean_params: test_config['strategy']['target_delta'] = clean_params['target_delta']
            if 'width' in clean_params: test_config['strategy']['width'] = clean_params['width']
            if 'take_profit_pct' in clean_params: test_config['exit']['take_profit_pct'] = clean_params['take_profit_pct']
            if 'stop_loss_mult' in clean_params: test_config['exit']['stop_loss_mult'] = clean_params['stop_loss_mult']
            
            sim = Simulator(test_config, self.loader)
            fold_trades = []
            for d in test_dates:
                trades = sim.run_day(d)
                fold_trades.extend(trades)
            
            # Store OOS trades with FULL DETAILS per spec
            for t in fold_trades:
                pnl = 0.0
                ror = 0.0
                exit_reason = "OPEN"
                exit_debit = 0.0
                
                if t.status == 'CLOSED' and t.exit_info:
                    # Per spec C3: gross_pnl_usd = (entry_credit - exit_debit) * 100
                    gross_pnl_points = t.exit_info.pnl
                    gross_pnl_usd = gross_pnl_points * 100  # Convert to USD
                    
                    # Apply commissions (8 legs round-trip)
                    commission_per_leg = self.config.get('simulation', {}).get('commission_per_leg', 0.65)
                    net_pnl_usd = gross_pnl_usd - (commission_per_leg * 8)
                    pnl = net_pnl_usd
                    
                    # Per spec C4: max_loss_usd = (width_real - entry_credit) * 100
                    max_loss_usd = t.max_loss * 100
                    ror = pnl / max_loss_usd if max_loss_usd > 0 else 0
                    exit_reason = t.exit_info.exit_reason
                    exit_debit = t.exit_info.exit_price
                    exit_reason = t.exit_info.exit_reason
                    exit_debit = t.exit_info.exit_price
                
                # Extract individual strikes
                strikes = {leg.option_type.value + ('_long' if leg.is_long else '_short'): leg.strike for leg in t.legs}
                
                # Calculate real widths
                short_put = next((l.strike for l in t.legs if l.option_type.value == 'P' and not l.is_long), None)
                long_put = next((l.strike for l in t.legs if l.option_type.value == 'P' and l.is_long), None)
                short_call = next((l.strike for l in t.legs if l.option_type.value == 'C' and not l.is_long), None)
                long_call = next((l.strike for l in t.legs if l.option_type.value == 'C' and l.is_long), None)
                
                width_put = short_put - long_put if short_put and long_put else 0
                width_call = long_call - short_call if long_call and short_call else 0
                    
                self.oos_trades.append({
                    'entry_time': t.entry_time,
                    'fold_train_start': train_dates[0],
                    'fold_test_start': test_dates[0],
                    'params': str(clean_params),
                    # Strikes (4 explicit)
                    'short_put_strike': short_put,
                    'long_put_strike': long_put,
                    'short_call_strike': short_call,
                    'long_call_strike': long_call,
                    # Widths
                    'width_put_real': width_put,
                    'width_call_real': width_call,
                    # Pricing
                    'entry_credit': t.entry_credit,
                    'exit_debit': exit_debit,
                    'max_loss': t.max_loss,
                    # PnL (net of commissions)
                    'pnl': pnl,
                    'ror': ror,
                    'exit_reason': exit_reason
                })
            
            # Move window
            start_idx += self.step_days
            
        return pd.DataFrame(self.oos_trades)
