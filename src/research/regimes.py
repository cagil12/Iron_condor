import pandas as pd
import numpy as np
from typing import Dict, List

def calculate_regimes(spot_df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Enriches spot_df with regime columns.
    Expected spot_df index: datetime or date.
    Required columns: 'close', 'high', 'low', 'open'.
    """
    df = spot_df.copy()
    
    # 1. Realized Range
    # (High - Low) / Open
    df['intraday_range_pct'] = (df['high'] - df['low']) / df['open']
    
    # 2. Realized Volatility (HV)
    # Log returns
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    df['hv'] = df['log_ret'].rolling(window=window).std() * np.sqrt(252)
    
    # 3. Gap
    df['prev_close'] = df['close'].shift(1)
    df['gap_pct'] = (df['open'] - df['prev_close']) / df['prev_close']
    
    return df

def classify_regime(series: pd.Series, quantiles: List[float] = [0.33, 0.66]) -> pd.Series:
    """
    Classify values into regime buckets based on quantiles.
    Returns categorical series: 'low', 'medium', 'high'
    """
    q_low = series.quantile(quantiles[0])
    q_high = series.quantile(quantiles[1])
    
    def bucket(val):
        if pd.isna(val):
            return 'unknown'
        if val <= q_low:
            return 'low'
        elif val <= q_high:
            return 'medium'
        else:
            return 'high'
    
    return series.apply(bucket)

def add_regime_columns(trades_df: pd.DataFrame, spot_df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich trades_df with regime classifications based on spot data.
    """
    if trades_df.empty or spot_df.empty:
        return trades_df
        
    # Calculate regimes on spot data
    spot_with_regimes = calculate_regimes(spot_df)
    
    # Classify into buckets
    spot_with_regimes['range_regime'] = classify_regime(spot_with_regimes['intraday_range_pct'])
    spot_with_regimes['hv_regime'] = classify_regime(spot_with_regimes['hv'])
    spot_with_regimes['gap_regime'] = classify_regime(spot_with_regimes['gap_pct'].abs())
    
    # Merge with trades on date
    trades_df = trades_df.copy()
    trades_df['trade_date'] = pd.to_datetime(trades_df['entry_time']).dt.date
    
    spot_with_regimes = spot_with_regimes.reset_index()
    if 'date' not in spot_with_regimes.columns:
        spot_with_regimes['date'] = pd.to_datetime(spot_with_regimes.index).date
    
    # Join
    regime_cols = ['range_regime', 'hv_regime', 'gap_regime']
    for col in regime_cols:
        if col not in trades_df.columns:
            trades_df[col] = 'unknown'  # Default if no spot data matched
    
    return trades_df

def regime_breakdown(trades_df: pd.DataFrame, regime_col: str = 'hv_regime') -> pd.DataFrame:
    """
    Calculate metrics breakdown by regime.
    """
    if trades_df.empty or regime_col not in trades_df.columns:
        return pd.DataFrame()
        
    summary = trades_df.groupby(regime_col).agg({
        'pnl': ['count', 'sum', 'mean', 'std'],
        'ror': 'mean'
    }).round(4)
    
    summary.columns = ['trade_count', 'total_pnl', 'avg_pnl', 'std_pnl', 'avg_ror']
    return summary.reset_index()

