import argparse
import yaml
import sys
import pandas as pd
from datetime import date, timedelta, datetime
from src.strategy.mock_data_generator import MockDataGenerator
from src.research.walk_forward import WalkForwardAnalysis

def main():
    parser = argparse.ArgumentParser(description="Run Walk-Forward Analysis (WFA)")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to config file")
    parser.add_argument("--mock", action="store_true", help="Use Mock Data")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Path to data directory")
    parser.add_argument("--start-date", type=str, default="2023-01-09", help="Start Date (YYYY-MM-DD)")
    parser.add_argument("--days", type=int, default=5, help="Number of days to run")
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
        from src.data.loaders import DataLoader
        loader = DataLoader(data_path=args.data_dir)
        
        # Real Data Dates
        start_dt = datetime.strptime(args.start_date, "%Y-%m-%d").date()
        all_dates = [start_dt + timedelta(days=i) for i in range(args.days)]
        
        # Print info
        print(f"Running Real Data WFA from {start_dt} for {args.days} days")
        print(f"Data Dir: {args.data_dir}")
        
    param_grid = config.get('grid_search', {})
    
    # Simple WFA params for testing
    # Adjust windows for short data sample
    train_window = 10 if args.mock else 3
    test_window = 5 if args.mock else 1
    step = 5 if args.mock else 1

    wfa = WalkForwardAnalysis(
        config, 
        loader, 
        train_window_days=train_window, 
        test_window_days=test_window, 
        step_days=step
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
