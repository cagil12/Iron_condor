"""
Databento Data Downloader

Downloads SPXW/SPX 0DTE options data from Databento API.
Saves as parquet files ready for the backtesting pipeline.
"""

import os
import argparse
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional

def get_databento_api_key() -> str:
    """
    Securely load Databento API key from environment variable.
    Raises ValueError if not configured.
    """
    api_key = os.getenv("DATABENTO_API_KEY")
    if not api_key:
        raise ValueError(
            "DATABENTO_API_KEY environment variable is not set.\n"
            "Please configure it using one of these methods:\n"
            "  1. Export in shell: export DATABENTO_API_KEY='your-key'\n"
            "  2. Set in .env file: DATABENTO_API_KEY=your-key\n"
            "  3. Windows: setx DATABENTO_API_KEY your-key\n"
            "Get your API key from: https://databento.com/portal"
        )
    return api_key


def download_options_data(
    symbol: str = "SPXW",  # SPXW for 0DTE SPX options
    start_date: str = None,
    end_date: str = None,
    output_dir: str = "./data/raw",
    schema: str = "cbbo-1m",  # Consolidated BBO 1-minute (bid/ask)
    auto_confirm: bool = False,
):
    """
    Download options data from Databento.
    
    Args:
        symbol: Root symbol (SPXW for SPX weeklies, SPY for SPY options)
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        output_dir: Directory to save parquet files
        schema: Databento schema (mbp-1 for bid/ask, ohlcv-1m for OHLC)
    """
    try:
        import databento as db
    except ImportError:
        print("Please install databento: pip install databento")
        return
    
    # Secure API key loading with validation
    api_key = get_databento_api_key()
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize client
    client = db.Historical(api_key)
    
    # Parse dates
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    # Databento uses exclusive end dates - add 1 day if same as start
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    if end_dt <= start_dt:
        end_dt = start_dt + timedelta(days=1)
        end_date = end_dt.strftime("%Y-%m-%d")
    
    print(f"Downloading {symbol} options data from {start_date} to {end_date}")
    print(f"Schema: {schema}")
    print(f"Output: {output_path}")
    
    # Get cost estimate first
    try:
        cost = client.metadata.get_cost(
            dataset="OPRA.PILLAR",  # US options data
            symbols=[symbol],
            schema=schema,
            start=start_date,
            end=end_date,
        )
        print(f"\nEstimated cost: ${cost:.2f}")
        
        if not auto_confirm:
            confirm = input("Proceed with download? (y/n): ")
            if confirm.lower() != 'y':
                print("Download cancelled.")
                return
        else:
            print("Auto-confirmed. Proceeding...")
            
    except Exception as e:
        print(f"Could not get cost estimate: {e}")
        print("Proceeding anyway...")
    
    # Download data
    try:
        data = client.timeseries.get_range(
            dataset="OPRA.PILLAR",
            symbols=[symbol],
            schema=schema,
            start=start_date,
            end=end_date,
        )
        
        # Convert to DataFrame and save
        df = data.to_df()
        
        if df.empty:
            print("No data returned!")
            return
        
        print(f"Downloaded {len(df):,} rows")
        
        # Save as single file
        output_file = output_path / f"{symbol}_{start_date}_{end_date}.parquet"
        df.to_parquet(output_file)
        print(f"Saved to: {output_file}")
        
        # Also save split by date for easier processing
        df['date'] = df.index.date
        for trade_date, group in df.groupby('date'):
            date_file = output_path / f"{trade_date}.parquet"
            group.to_parquet(date_file)
            print(f"  Saved: {date_file} ({len(group):,} rows)")
        
        print("\nDownload complete!")
        
    except Exception as e:
        print(f"Error downloading data: {e}")
        raise


def download_spot_data(
    symbol: str = "SPX",
    start_date: str = None,
    end_date: str = None,
    output_dir: str = "./data/raw/spot",
):
    """Download underlying spot/index data."""
    try:
        import databento as db
    except ImportError:
        print("Please install databento: pip install databento")
        return
    
    api_key = get_databento_api_key()
    client = db.Historical(api_key)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    print(f"Downloading {symbol} spot data from {start_date} to {end_date}")
    
    try:
        # Use XNAS.ITCH for index data or equity for SPY
        dataset = "XNAS.ITCH" if symbol in ["SPX", "NDX", "VIX"] else "XNAS.ITCH"
        
        data = client.timeseries.get_range(
            dataset=dataset,
            symbols=[symbol],
            schema="ohlcv-1m",
            start=start_date,
            end=end_date,
        )
        
        df = data.to_df()
        output_file = output_path / f"{symbol}_{start_date}_{end_date}.parquet"
        df.to_parquet(output_file)
        print(f"Saved spot data to: {output_file}")
        
    except Exception as e:
        print(f"Error downloading spot data: {e}")


def list_available_symbols():
    """List available option symbols on Databento."""
    try:
        import databento as db
    except ImportError:
        print("Please install databento: pip install databento")
        return
    
    api_key = get_databento_api_key()
    client = db.Historical(api_key)
    
    # List datasets first
    print("Checking available datasets...")
    try:
        # For options, we use OPRA datasets
        # The main datasets for options are:
        # - OPRA.PILLAR (OPRA for options)
        # - DBEQ.BASIC (equities for underlying)
        
        print("\nFor SPX 0DTE options, use:")
        print("  Symbol: SPXW (SPX Weeklies)")
        print("  Dataset: OPRA.PILLAR")
        print("")
        print("For SPY 0DTE options, use:")
        print("  Symbol: SPY")
        print("  Dataset: OPRA.PILLAR")
        print("")
        print("Example download command:")
        print("  python scripts/download_databento.py --symbol SPXW --start 2024-01-15 --end 2024-01-15")
        
    except Exception as e:
        print(f"API error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Download options data from Databento")
    parser.add_argument("--symbol", default="SPXW", help="Root symbol (SPXW, SPY, etc.)")
    parser.add_argument("--start", default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", default="./data/raw", help="Output directory")
    parser.add_argument("--schema", default="cbbo-1m", help="Data schema (cbbo-1m, ohlcv-1m)")
    parser.add_argument("--list-symbols", action="store_true", help="List available symbols")
    parser.add_argument("--spot", action="store_true", help="Download spot data instead")
    parser.add_argument("-y", "--yes", action="store_true", help="Auto-confirm download")
    
    args = parser.parse_args()
    
    if args.list_symbols:
        list_available_symbols()
    elif args.spot:
        download_spot_data(args.symbol, args.start, args.end, args.output + "/spot")
    else:
        download_options_data(args.symbol, args.start, args.end, args.output, args.schema, args.yes)


if __name__ == "__main__":
    main()
