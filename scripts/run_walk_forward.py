import argparse
import yaml
import sys
import pandas as pd
from datetime import date, timedelta
from ingresarios_options_research.src.strategy.mock_data_generator import MockDataGenerator
from ingresarios_options_research.src.research.walk_forward import WalkForwardAnalysis

def main():
    parser = argparse.ArgumentParser(description="Run Walk-Forward Analysis (WFA)")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to config file")
    parser.add_argument("--mock", action="store_true", help="Use Mock Data")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    if args.mock:
        # Mock Loader wrapper
        class MockLoader:
            def generate_day(self, d):
                return MockDataGenerator(d).generate_day(d)
        loader = MockLoader()
        
        # Simulate 30 days of data for WFA
        # Train 10, Test 5, Step 5
        all_dates = [date(2023, 1, 1) + timedelta(days=i) for i in range(30)]
    else:
        sys.exit(1)
        
    param_grid = config.get('grid_search', {})
    
    # Simple WFA params for testing
    wfa = WalkForwardAnalysis(
        config, 
        loader, 
        train_window_days=10, 
        test_window_days=5, 
        step_days=5
    )
    
    oos_results = wfa.run(all_dates, param_grid)
    
    print("\n--- OOS Walk-Forward Results ---")
    if not oos_results.empty:
        print(oos_results.head())
        print(f"Total OOS Trades: {len(oos_results)}")
        print(f"Total PnL: {oos_results['pnl'].sum():.2f}")
    else:
        print("No OOS trades generated.")
        
    oos_results.to_parquet("wfa_trades.parquet")
    print("OOS trades saved to wfa_trades.parquet")

if __name__ == "__main__":
    main()
