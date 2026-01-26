import argparse
import yaml
import sys
import pandas as pd
from datetime import datetime, date
from ingresarios_options_research.src.data.data_quality import DataQualityChecker
from ingresarios_options_research.src.data.loaders import DataLoader
from ingresarios_options_research.src.strategy.mock_data_generator import MockDataGenerator

def main():
    parser = argparse.ArgumentParser(description="Run Data Quality Checks")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to config file")
    parser.add_argument("--mock", action="store_true", help="Use Mock Data")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Running Data Check with config: {args.config}")
    
    if args.mock:
        print("Using MOCK DATA generator...")
        gen = MockDataGenerator(date(2023, 1, 1), days=1)
        # Mock generator yields chains for a day
        # We need to simulate the day loop
        day_gen = gen.generate_day(date(2023, 1, 1))
    else:
        print("Real data loader not yet fully implemented, please use --mock")
        sys.exit(1)
        
    checker = DataQualityChecker()
    
    count = 0
    for chain in day_gen:
        checker.check_chain_integrity(chain)
        count += 1
        if count % 60 == 0:
            print(f"Processed {count} snapshots...")
            
    report_df = checker.generate_report()
    print("\n--- Data Quality Report ---")
    if report_df.empty:
        print("No issues found!")
    else:
        print(f"Found {len(report_df)} issues.")
        print(report_df['issue_description'].value_counts())
        
    # Save report
    report_df.to_csv("data_quality_report.csv", index=False)
    print("Report saved to data_quality_report.csv")

if __name__ == "__main__":
    main()
