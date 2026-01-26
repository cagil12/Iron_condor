import itertools
import pandas as pd
from typing import Dict, List, Any
from ..strategy.simulator import Simulator
from .metrics import calculate_metrics, calculate_score

class GridSearch:
    def __init__(self, base_config: dict, param_grid: dict, data_loader):
        self.base_config = base_config
        self.param_grid = param_grid # {'target_delta': [0.1, 0.15], ...}
        self.loader = data_loader
        self.results = []

    def generate_combinations(self) -> List[Dict[str, Any]]:
        keys = self.param_grid.keys()
        values = self.param_grid.values()
        combinations = []
        for instance in itertools.product(*values):
            combinations.append(dict(zip(keys, instance)))
        return combinations

    def run(self, train_dates: List[Any], metric_weights: Dict[str, float] = None) -> pd.DataFrame:
        """
        Runs the grid search over the specified train_dates.
        """
        combinations = self.generate_combinations()
        print(f"Starting Grid Search: {len(combinations)} combinations over {len(train_dates)} dates.")
        
        for i, params in enumerate(combinations):
            # Update config with current params
            current_config = self.base_config.copy()
            # Deep update strategy/exit params
            # Simplified: assuming params keys match strategy/exit config structure flatly or handled here
            # For now, simplistic update:
            if 'target_delta' in params: current_config['strategy']['target_delta'] = params['target_delta']
            if 'width' in params: current_config['strategy']['width'] = params['width']
            if 'take_profit_pct' in params: current_config['exit']['take_profit_pct'] = params['take_profit_pct']
            if 'stop_loss_mult' in params: current_config['exit']['stop_loss_mult'] = params['stop_loss_mult']
            
            # Run Simulation
            sim = Simulator(current_config, self.loader)
            all_trades = []
            for d in train_dates:
                trades = sim.run_day(d)
                all_trades.extend(trades)
            
            # Convert to DF for Metrics
            if all_trades:
                trades_data = []
                for t in all_trades:
                    pnl = 0.0
                    ror = 0.0
                    if t.status == 'CLOSED' and t.exit_info:
                        pnl = t.exit_info.pnl
                        ror = t.exit_info.pnl_pct # Or pnl/max_loss
                        # Recalculate RoR strictly based on Max Loss
                        if t.max_loss > 0:
                            ror = pnl / t.max_loss
                            
                    trades_data.append({
                        'entry_time': t.entry_time,
                        'pnl': pnl,
                        'max_loss': t.max_loss,
                        'ror': ror
                    })
                df_trades = pd.DataFrame(trades_data)
                if not df_trades.empty:
                    df_trades['entry_time'] = pd.to_datetime(df_trades['entry_time'])
            else:
                df_trades = pd.DataFrame()

            metrics = calculate_metrics(df_trades)
            score = calculate_score(metrics) # user weights can be passed here
            
            result_row = params.copy()
            result_row.update(metrics)
            result_row['score'] = score
            self.results.append(result_row)
            
        return pd.DataFrame(self.results)
