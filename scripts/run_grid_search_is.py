import argparse
import yaml
import sys
import pandas as pd
from datetime import date, timedelta
from ingresarios_options_research.src.strategy.mock_data_generator import MockDataGenerator
from ingresarios_options_research.src.research.grid_search import GridSearch

def main():
    parser = argparse.ArgumentParser(description="Run Just-In-Time Grid Search (IS)")
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
        
        # Use simple mock dates
        train_dates = [date(2023, 1, 1) + timedelta(days=i) for i in range(5)]
    else:
        print("Real data loader not implemented.")
        sys.exit(1)
        
    param_grid = config.get('grid_search', {})
    
    gs = GridSearch(config, param_grid, loader)
    results = gs.run(train_dates)
    
    print("\n--- Grid Search Results ---")
    print(results.sort_values(by='score', ascending=False).head())
    
    results.to_parquet("grid_search_results.parquet")
    print("Results saved to grid_search_results.parquet")

if __name__ == "__main__":
    main()
