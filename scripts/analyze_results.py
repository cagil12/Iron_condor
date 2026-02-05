import pandas as pd
import numpy as np

def calculate_metrics(file_path):
    try:
        df = pd.read_parquet(file_path)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return

    if df.empty:
        print("No trades found.")
        return

    print("--- Detailed Backtest Metrics (60-Day WFA) ---\n")
    
    # Recalculate PnL to avoid potential scaling bugs in parquet
    # PnL = (Credit - Debit) * 100 (multiplier)
    df['pnl_calculated'] = (df['entry_credit'] - df['exit_debit']) * 100
    
    # 1. Trade Counts
    total_trades = len(df)
    winning_trades = df[df['pnl_calculated'] > 0]
    losing_trades = df[df['pnl_calculated'] <= 0]
    
    num_winners = len(winning_trades)
    num_losers = len(losing_trades)
    win_rate = (num_winners / total_trades) * 100 if total_trades > 0 else 0
    
    # 2. PnL Stats
    total_pnl = df['pnl_calculated'].sum()
    avg_pnl = df['pnl_calculated'].mean()
    
    gross_profit = winning_trades['pnl_calculated'].sum()
    gross_loss = abs(losing_trades['pnl_calculated'].sum())
    
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    avg_win = winning_trades['pnl_calculated'].mean() if num_winners > 0 else 0
    avg_loss = losing_trades['pnl_calculated'].mean() if num_losers > 0 else 0
    
    # 3. Risk Metrics
    # Max Drawdown from equity curve
    df['cumulative_pnl'] = df['pnl_calculated'].cumsum()
    df['peak'] = df['cumulative_pnl'].cummax()
    df['drawdown'] = df['cumulative_pnl'] - df['peak']
    max_drawdown = df['drawdown'].min()
    
    # Expectancy
    expectancy = (win_rate/100 * avg_win) + ((1 - win_rate/100) * avg_loss)

    print(f"Total Trades:       {total_trades}")
    print(f"Win Rate:           {win_rate:.2f}% ({num_winners}W - {num_losers}L)")
    print(f"Total PnL:          ${total_pnl:,.2f}")
    print(f"Average Trade:      ${avg_pnl:,.2f}")
    print(f"Profit Factor:      {profit_factor:.2f}")
    print(f"Expectancy:         ${expectancy:,.2f}")
    print(f"Max Drawdown:       ${max_drawdown:,.2f}")
    print(f"Avg Win:            ${avg_win:,.2f}")
    print(f"Avg Loss:           ${avg_loss:,.2f}")
    
    # --- VIX REGIME ANALYSIS (NEW) ---
    print("\n" + "="*40)
    print("--- Performance by VIX Regime ---")
    print("="*40)
    
    # Detect VIX column
    vix_col = None
    if 'vix_value' in df.columns:
        vix_col = 'vix_value'
    elif 'vix' in df.columns:
        vix_col = 'vix'
        
    if vix_col:
        def bucket_vix(v):
            if pd.isna(v): return 'Unknown'
            if v < 15: return 'Low (<15)'
            if v <= 25: return 'Normal (15-25)'
            return 'High (>25)'
            
        df['vix_regime'] = df[vix_col].apply(bucket_vix)
        
        # Aggregation
        regime_stats = df.groupby('vix_regime').agg(
            Count=('pnl_calculated', 'count'),
            Win_Rate=('pnl_calculated', lambda x: (x > 0).mean() * 100),
            Total_PnL=('pnl_calculated', 'sum'),
            Avg_PnL=('pnl_calculated', 'mean')
        ).round(2)
        
        # Sort in logical order
        order_map = {'Low (<15)': 1, 'Normal (15-25)': 2, 'High (>25)': 3, 'Unknown': 4}
        regime_stats['sort_key'] = regime_stats.index.map(order_map)
        regime_stats = regime_stats.sort_values('sort_key').drop(columns='sort_key')
        
        # Print optimized table for console
        print(regime_stats.to_string())
        
    else:
        print("⚠️ No VIX data found in trade records. Skipping regime analysis.")

    print("\n--- Trade List (Last 20) ---")
    # Format entry_time to string
    if 'entry_time' in df.columns:
        df['Date'] = df['entry_time'].astype(str).str[:10]
    
    cols_to_show = ['Date', 'entry_credit', 'exit_debit', 'pnl_calculated', 'exit_reason']
    if vix_col:
        cols_to_show.append(vix_col)
        
    print(df[cols_to_show].tail(20).to_string(index=False))

if __name__ == "__main__":
    calculate_metrics("wfa_trades.parquet")
