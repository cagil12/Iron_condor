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
    # print(f"Win Rate:           {win_rate:.1f}% ({num_winners} W - {num_losers} L)")
    print(f"Win Rate:           {win_rate:.2f}% ({num_winners}W - {num_losers}L)")
    print(f"Total PnL:          ${total_pnl:,.2f}")
    print(f"Average Trade:      ${avg_pnl:,.2f}")
    print(f"Profit Factor:      {profit_factor:.2f}")
    print(f"Expectancy:         ${expectancy:,.2f}")
    print(f"Max Drawdown:       ${max_drawdown:,.2f}")
    print(f"Avg Win:            ${avg_win:,.2f}")
    print(f"Avg Loss:           ${avg_loss:,.2f}")
    
    print("\n--- Trade List ---")
    # Format entry_time to string
    if 'entry_time' in df.columns:
        df['Date'] = df['entry_time'].astype(str).str[:10]
    
    print(df[['Date', 'entry_credit', 'exit_debit', 'pnl_calculated', 'exit_reason']].to_string(index=False))

if __name__ == "__main__":
    calculate_metrics("wfa_trades.parquet")
