import pandas as pd
import numpy as np
from typing import List, Dict

def calculate_metrics(trades_df: pd.DataFrame, initial_capital: float = 10000.0) -> Dict[str, float]:
    """
    Calculates Investment-Grade metrics from a DataFrame of trades.
    trades_df must have: 'entry_time', 'pnl', 'max_loss' (for RoR).
    """
    if trades_df.empty:
        return {
            "total_trades": 0,
            "total_pnl": 0.0,
            "win_rate": 0.0,
            "sharpe": 0.0,
            "max_dd": 0.0,
            "ror_avg": 0.0,
            "profit_factor": 0.0
        }
    
    # 1. Trade Level Metrics
    total_trades = len(trades_df)
    total_pnl = trades_df['pnl'].sum()
    winners = trades_df[trades_df['pnl'] > 0]
    win_rate = len(winners) / total_trades
    
    gross_profit = winners['pnl'].sum()
    gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else np.nan
    
    # RoR per trade = PnL / Max Risk
    # Assuming 'max_loss' column exists and is > 0
    if 'max_loss' in trades_df.columns:
        trades_df['ror'] = trades_df['pnl'] / trades_df['max_loss']
        avg_trade_ror = trades_df['ror'].mean()
    else:
        avg_trade_ror = 0.0

    # 2. Time Series Metrics (Daily)
    # Aggregating by Day for Sharpe/DD
    trades_df['date'] = pd.to_datetime(trades_df['entry_time']).dt.date
    daily_pnl = trades_df.groupby('date')['pnl'].sum()
    
    # Fill missing days with 0?? Or assume days with no trades are flat?
    # For now, using days traded.
    
    daily_returns = daily_pnl # Absolute PnL
    # Theoretically Sharpe should be on Returns %, so we need a capital base.
    # Using simple PnL std dev for now or assuming fixed capital.
    # Spec requested: Sharpe on RoR? Or Equity Curve?
    # Spec: "Sharpe sobre RoR"
    
    if 'ror' in trades_df.columns:
        # Daily RoR sum
        daily_ror = trades_df.groupby('date')['ror'].sum()
        mean_daily_ror = daily_ror.mean()
        std_daily_ror = daily_ror.std()
        
        # Annualized Sharpe (assuming 252 days)
        if std_daily_ror > 0:
            sharpe_ror = (mean_daily_ror * 252) / (std_daily_ror * np.sqrt(252))
        else:
            sharpe_ror = 0.0
    else:
        sharpe_ror = 0.0

    # Max Drawdown
    # Construct Equity Curve
    equity = daily_pnl.cumsum() + initial_capital
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    max_dd = drawdown.min() # Negative value

    return {
        "total_trades": total_trades,
        "total_pnl": total_pnl,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_ror_per_trade": avg_trade_ror,
        "sharpe_ror": sharpe_ror,
        "max_dd_pct": max_dd,
        "cvar_95": calculate_cvar(trades_df['pnl'], 0.95) if not trades_df.empty else 0.0,
        "worst_day_pnl": daily_pnl.min() if len(daily_pnl) > 0 else 0.0
    }

def calculate_cvar(pnl_series: pd.Series, confidence: float = 0.95) -> float:
    """
    Calculate Conditional Value at Risk (Expected Shortfall).
    CVaR is the expected loss given that the loss exceeds VaR.
    """
    if pnl_series.empty:
        return 0.0
    var_threshold = pnl_series.quantile(1 - confidence)
    tail_losses = pnl_series[pnl_series <= var_threshold]
    return tail_losses.mean() if len(tail_losses) > 0 else 0.0

def calculate_score(
    metrics: Dict[str, float], 
    lambda_dd: float = 1.0, 
    eta_cvar: float = 0.5,
    kappa_worst: float = 0.1
) -> float:
    """
    Custom optimization objective per spec:
    score = sharpe(RoR) - λ*MaxDD - η*CVaR95 - κ*worst_day_penalty
    """
    sharpe = metrics.get('sharpe_ror', 0.0)
    max_dd = abs(metrics.get('max_dd_pct', 0.0))
    cvar = abs(metrics.get('cvar_95', 0.0))
    worst_day = abs(metrics.get('worst_day_pnl', 0.0))
    
    return sharpe - lambda_dd * max_dd - eta_cvar * cvar - kappa_worst * worst_day

def apply_commissions(gross_pnl: float, commission_per_leg: float = 0.65, num_legs: int = 4) -> float:
    """
    Apply round-trip commissions (entry + exit = 8 legs for IC).
    net_pnl = gross_pnl - commission_per_leg * 8
    """
    total_commission = commission_per_leg * num_legs * 2  # Round-trip
    return gross_pnl - total_commission

