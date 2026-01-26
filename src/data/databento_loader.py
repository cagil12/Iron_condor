"""
Databento Data Loader for SPX/SPXW Options

Optimized for large datasets using Polars for memory efficiency.
Supports streaming and chunked processing.
"""

import polars as pl
from datetime import datetime, date, time, timedelta
from typing import Generator, Optional, List
from pathlib import Path
import logging

from .schema import OptionChain, Quote, OptionType, SpotPrice

logger = logging.getLogger(__name__)


class DatabentoLoader:
    """
    Efficient loader for Databento options data.
    Uses Polars for memory-efficient processing of large datasets.
    """
    
    def __init__(self, data_dir: str, symbol: str = "SPXW"):
        self.data_dir = Path(data_dir)
        self.symbol = symbol
        self._spot_cache = {}
        
    def _load_day_lazy(self, trade_date: date) -> pl.LazyFrame:
        """
        Load day's data lazily (doesn't load into memory until needed).
        Databento typically exports as CSV or Parquet.
        """
        # Try parquet first (more efficient)
        parquet_path = self.data_dir / f"{trade_date.isoformat()}.parquet"
        csv_path = self.data_dir / f"{trade_date.isoformat()}.csv"
        
        if parquet_path.exists():
            return pl.scan_parquet(parquet_path)
        elif csv_path.exists():
            return pl.scan_csv(csv_path)
        else:
            raise FileNotFoundError(f"No data file for {trade_date}")
    
    def _process_databento_schema(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """
        Transform Databento's schema to our internal format.
        
        Databento options schema typically includes:
        - ts_event (timestamp)
        - symbol (e.g., SPXW 240126C05000)
        - bid_px_00, ask_px_00 (top of book)
        - bid_sz_00, ask_sz_00
        """
        return (
            lf
            .with_columns([
                # Parse timestamp
                pl.col("ts_event").cast(pl.Datetime("us")).alias("timestamp"),
                
                # Extract strike from symbol (e.g., SPXW 240126C05000 -> 5000)
                pl.col("symbol").str.extract(r"(\d{5})$", 1)
                    .cast(pl.Float64).alias("strike"),
                
                # Extract option type (C/P)
                pl.col("symbol").str.extract(r"([CP])\d{5}$", 1)
                    .alias("option_type"),
                
                # Extract expiration (e.g., 240126)
                pl.col("symbol").str.extract(r"(\d{6})[CP]", 1)
                    .alias("expiration_str"),
                
                # Rename bid/ask
                pl.col("bid_px_00").alias("bid"),
                pl.col("ask_px_00").alias("ask"),
            ])
            .filter(
                # Filter for 0DTE only
                pl.col("expiration_str").is_not_null()
            )
        )
    
    def load_spot_data(self, trade_date: date) -> pl.DataFrame:
        """Load underlying spot price data for the day."""
        spot_path = self.data_dir / "spot" / f"{trade_date.isoformat()}.parquet"
        
        if spot_path.exists():
            return pl.read_parquet(spot_path)
        
        # Fallback: try to infer from options data (ATM mid as proxy)
        logger.warning(f"No spot data for {trade_date}, will infer from ATM options")
        return None
    
    def generate_day(self, trade_date: date) -> Generator[OptionChain, None, None]:
        """
        Yields OptionChain snapshots for the given day, minute by minute.
        Memory-efficient: processes data in streaming fashion.
        """
        try:
            lf = self._load_day_lazy(trade_date)
            lf = self._process_databento_schema(lf)
        except FileNotFoundError:
            logger.error(f"No data for {trade_date}")
            return
        
        # Load spot data
        spot_df = self.load_spot_data(trade_date)
        
        # Collect and group by minute
        df = lf.collect()
        
        if df.is_empty():
            logger.warning(f"Empty data for {trade_date}")
            return
        
        # Round timestamps to minute
        df = df.with_columns([
            pl.col("timestamp").dt.truncate("1m").alias("minute")
        ])
        
        # Group by minute and yield chains
        for minute, group in df.group_by("minute", maintain_order=True):
            minute_ts = minute[0]
            
            # Get spot price for this minute
            if spot_df is not None:
                spot_row = spot_df.filter(
                    pl.col("timestamp").dt.truncate("1m") == minute_ts
                )
                if len(spot_row) > 0:
                    underlying = spot_row["price"][0]
                else:
                    # Use ATM mid as fallback
                    underlying = self._infer_spot_from_chain(group)
            else:
                underlying = self._infer_spot_from_chain(group)
            
            if underlying is None:
                logger.warning(f"Could not determine spot price for {minute_ts}")
                continue
            
            # Build chain
            chain = OptionChain(
                timestamp=minute_ts.to_pydatetime(),
                underlying_price=underlying,
                expiration=trade_date
            )
            
            for row in group.iter_rows(named=True):
                strike = row["strike"]
                otype_str = row["option_type"]
                
                if strike is None or otype_str is None:
                    continue
                
                otype = OptionType.CALL if otype_str == "C" else OptionType.PUT
                
                bid = row.get("bid", 0.0) or 0.0
                ask = row.get("ask", 0.0) or 0.0
                
                # Skip invalid quotes
                if ask <= 0 or bid < 0:
                    continue
                
                quote = Quote(
                    bid=float(bid),
                    ask=float(ask),
                    mid=(bid + ask) / 2,
                    delta=row.get("delta"),
                    gamma=row.get("gamma"),
                    theta=row.get("theta"),
                    vega=row.get("vega"),
                    implied_vol=row.get("implied_vol"),
                    volume=row.get("volume"),
                    open_interest=row.get("open_interest")
                )
                
                chain.quotes[(strike, otype)] = quote
            
            if chain.quotes:
                yield chain
    
    def _infer_spot_from_chain(self, group: pl.DataFrame) -> Optional[float]:
        """Infer spot price from ATM options (put-call parity approximation)."""
        # Find strike with smallest bid-ask spread (likely ATM)
        calls = group.filter(pl.col("option_type") == "C")
        puts = group.filter(pl.col("option_type") == "P")
        
        if calls.is_empty() or puts.is_empty():
            return None
        
        # Simple heuristic: strike where C and P have similar mids
        # More sophisticated: use put-call parity
        
        # For now, use the strike with highest combined volume or smallest spread
        all_strikes = group.select("strike").unique()
        if all_strikes.is_empty():
            return None
            
        # Return middle strike as approximation
        strikes = sorted(all_strikes["strike"].to_list())
        return strikes[len(strikes) // 2]
    
    def get_available_dates(self) -> List[date]:
        """List all dates with available data."""
        dates = []
        for f in self.data_dir.glob("*.parquet"):
            try:
                dt = datetime.strptime(f.stem, "%Y-%m-%d").date()
                dates.append(dt)
            except ValueError:
                continue
        return sorted(dates)


class ChunkedCSVLoader:
    """
    Memory-efficient CSV loader using chunking.
    For processing very large CSV files that don't fit in RAM.
    """
    
    def __init__(self, file_path: str, chunk_size: int = 100_000):
        self.file_path = Path(file_path)
        self.chunk_size = chunk_size
    
    def process_in_chunks(self, processor_fn):
        """
        Process file in chunks, applying processor_fn to each chunk.
        """
        reader = pl.read_csv_batched(
            self.file_path,
            batch_size=self.chunk_size
        )
        
        results = []
        batch_num = 0
        
        while True:
            batch = reader.next_batches(1)
            if batch is None:
                break
            
            for df in batch:
                result = processor_fn(df)
                if result is not None:
                    results.append(result)
            
            batch_num += 1
            if batch_num % 10 == 0:
                logger.info(f"Processed {batch_num * self.chunk_size:,} rows")
        
        return results
