"""
vix_loader.py

Downloads and provides historical VIX data for regime filtering.
Uses yfinance to fetch VIX (^VIX) data from Yahoo Finance.
"""
import pandas as pd
from datetime import date, datetime
from pathlib import Path
from typing import Optional

class VixLoader:
    """
    Loads VIX data from local cache or downloads from Yahoo Finance.
    """
    
    def __init__(self, cache_path: str = "data/vix_cache.parquet"):
        self.cache_path = Path(cache_path)
        self._vix_data: Optional[pd.DataFrame] = None
        self._load_cache()
    
    def _load_cache(self):
        """Load VIX data from local cache if available."""
        if self.cache_path.exists():
            try:
                self._vix_data = pd.read_parquet(self.cache_path)
                print(f"VIX cache loaded: {len(self._vix_data)} days")
            except Exception as e:
                print(f"Failed to load VIX cache: {e}")
                self._vix_data = None
    
    def download(self, start_date: str = "2025-01-01", end_date: str = None) -> pd.DataFrame:
        """
        Download VIX data from Yahoo Finance and cache locally.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD), defaults to today
            
        Returns:
            DataFrame with VIX data
        """
        try:
            import yfinance as yf
        except ImportError:
            print("yfinance not installed. Run: pip install yfinance")
            return pd.DataFrame()
        
        if end_date is None:
            end_date = date.today().strftime("%Y-%m-%d")
        
        print(f"Downloading VIX data from {start_date} to {end_date}...")
        
        vix = yf.Ticker("^VIX")
        df = vix.history(start=start_date, end=end_date)
        
        if df.empty:
            print("No VIX data returned from Yahoo Finance")
            return pd.DataFrame()
        
        # Clean up index
        df.index = pd.to_datetime(df.index).date
        df.index.name = 'date'
        
        # Keep only Close price (that's VIX level)
        df = df[['Close']].rename(columns={'Close': 'vix'})
        
        # Cache locally
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self.cache_path)
        print(f"VIX data cached to {self.cache_path} ({len(df)} days)")
        
        self._vix_data = df
        return df
    
    def get_vix(self, trade_date: date) -> Optional[float]:
        """
        Get VIX value for a specific date.
        
        Args:
            trade_date: The date to lookup
            
        Returns:
            VIX value or None if not available
        """
        if self._vix_data is None or self._vix_data.empty:
            return None
        
        # Handle datetime input
        if isinstance(trade_date, datetime):
            trade_date = trade_date.date()
        
        if trade_date in self._vix_data.index:
            return float(self._vix_data.loc[trade_date, 'vix'])
        
        # Try to find closest prior date (T-1)
        prior_dates = [d for d in self._vix_data.index if d < trade_date]
        if prior_dates:
            closest = max(prior_dates)
            return float(self._vix_data.loc[closest, 'vix'])
        
        return None
    
    def ensure_data(self, start_date: str = "2025-10-01"):
        """Ensure VIX data is available, downloading if necessary."""
        if self._vix_data is None or self._vix_data.empty:
            self.download(start_date=start_date)
        return self._vix_data is not None and not self._vix_data.empty


# Singleton instance for easy import
_vix_loader: Optional[VixLoader] = None

def get_vix_loader(cache_path: str = "data/vix_cache.parquet") -> VixLoader:
    """Get or create the singleton VIX loader."""
    global _vix_loader
    if _vix_loader is None:
        _vix_loader = VixLoader(cache_path)
    return _vix_loader
