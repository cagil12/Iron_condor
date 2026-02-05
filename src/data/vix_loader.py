"""
vix_loader.py

Downloads and provides historical VIX data for regime filtering.
Uses yfinance for historical data, IBKR for real-time.
"""
import pandas as pd
from datetime import date, datetime
from pathlib import Path
from typing import Optional, Any

class VixLoader:
    """
    Loads VIX data from IBKR (real-time) or local cache (fallback).
    """
    
    def __init__(self, cache_path: str = "data/vix_cache.parquet", ib_connector: Any = None):
        self.cache_path = Path(cache_path)
        self._vix_data: Optional[pd.DataFrame] = None
        self._ib_connector = ib_connector
        self._load_cache()
    
    def set_ib_connector(self, connector: Any):
        """Set the IBKR connector for real-time VIX fetching."""
        self._ib_connector = connector
        
    def _load_cache(self):
        """Load VIX data from local cache if available."""
        if self.cache_path.exists():
            try:
                self._vix_data = pd.read_parquet(self.cache_path)
                print(f"VIX cache loaded: {len(self._vix_data)} days")
            except Exception as e:
                print(f"Failed to load VIX cache: {e}")
                self._vix_data = None
    
    def get_vix_from_ibkr(self) -> Optional[float]:
        """
        Get real-time VIX from IBKR connection.
        
        Returns:
            Current VIX value or None if unavailable
        """
        if self._ib_connector is None:
            return None
            
        try:
            from ib_insync import Index
            
            ib = self._ib_connector.ib
            if not ib.isConnected():
                return None
            
            # Create VIX Index contract
            vix_contract = Index('VIX', 'CBOE')
            ib.qualifyContracts(vix_contract)
            
            # Request market data (snapshot)
            ticker = ib.reqMktData(vix_contract, '', False, False)
            
            # Wait briefly for data
            ib.sleep(1)
            
            # Get last price
            vix_value = None
            if ticker.last and ticker.last > 0:
                vix_value = ticker.last
            elif ticker.close and ticker.close > 0:
                vix_value = ticker.close
            elif ticker.bid and ticker.bid > 0:
                vix_value = ticker.bid
                
            # Cancel market data subscription
            ib.cancelMktData(vix_contract)
            
            if vix_value:
                print(f"📈 VIX (LIVE from IBKR): {vix_value:.2f}")
                return float(vix_value)
            
            return None
            
        except Exception as e:
            print(f"⚠️ Failed to get VIX from IBKR: {e}")
            return None
    
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
        Get VIX value. Tries IBKR first (real-time), then falls back to cache.
        
        Args:
            trade_date: The date to lookup (used for cache fallback)
            
        Returns:
            VIX value or None if not available
        """
        # 1. Try IBKR real-time first
        live_vix = self.get_vix_from_ibkr()
        if live_vix is not None:
            return live_vix
        
        # 2. Fallback to cache
        if self._vix_data is None or self._vix_data.empty:
            print("⚠️ No VIX data in cache")
            return None
        
        # Handle datetime input
        if isinstance(trade_date, datetime):
            trade_date = trade_date.date()
        
        if trade_date in self._vix_data.index:
            cached_vix = float(self._vix_data.loc[trade_date, 'vix'])
            print(f"📊 VIX (CACHE): {cached_vix:.2f}")
            return cached_vix
        
        # Try to find closest prior date (T-1)
        prior_dates = [d for d in self._vix_data.index if d < trade_date]
        if prior_dates:
            closest = max(prior_dates)
            cached_vix = float(self._vix_data.loc[closest, 'vix'])
            print(f"📊 VIX (CACHE T-1): {cached_vix:.2f}")
            return cached_vix
        
        return None
    
    def ensure_data(self, start_date: str = "2025-10-01"):
        """Ensure VIX data is available, downloading if necessary."""
        if self._vix_data is None or self._vix_data.empty:
            self.download(start_date=start_date)
        return self._vix_data is not None and not self._vix_data.empty


# Singleton instance for easy import
_vix_loader: Optional[VixLoader] = None

def get_vix_loader(cache_path: str = "data/vix_cache.parquet", ib_connector: Any = None) -> VixLoader:
    """Get or create the singleton VIX loader."""
    global _vix_loader
    if _vix_loader is None:
        _vix_loader = VixLoader(cache_path, ib_connector)
    elif ib_connector is not None:
        _vix_loader.set_ib_connector(ib_connector)
    return _vix_loader
