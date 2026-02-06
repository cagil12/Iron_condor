import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from pathlib import Path

# Set dark theme for charts
plt.style.use('dark_background')

# File Paths
DATA_DIR = Path('data')
JOURNAL_PATH = DATA_DIR / 'trade_journal.csv'
PLOTS_DIR = DATA_DIR / 'plots'
PLOT_OUTPUT = PLOTS_DIR / 'vrp_analysis.png'

def load_data():
    """Load and preprocess trade journal."""
    if not JOURNAL_PATH.exists():
        print(f"❌ Journal not found at {JOURNAL_PATH}")
        sys.exit(1)

    df = pd.read_csv(JOURNAL_PATH)
    
    # Check for critical columns
    required_cols = ['status', 'iv_entry_atm', 'rv_duration', 'final_pnl_usd', 'timestamp']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"❌ Missing columns in CSV: {missing}")
        sys.exit(1)

    # Filter Closed Trades
    df_closed = df[df['status'] == 'CLOSED'].copy()
    
    if df_closed.empty:
        print("⚠️ No CLOSED trades found. Cannot generate VRP analysis.")
        # Proceed to generate an empty placeholder plot with warning
        return df_closed

    # Convert numeric
    numeric_cols = ['iv_entry_atm', 'rv_duration', 'final_pnl_usd']
    for col in numeric_cols:
        df_closed[col] = pd.to_numeric(df_closed[col], errors='coerce')
        
    # Convert timestamp
    df_closed['timestamp'] = pd.to_datetime(df_closed['timestamp'])
    
    # Calculate Metrics
    # Harvest Ratio: IV (sold) / RV (realized)
    # Only valid if RV > 0
    df_closed['harvest_ratio'] = df_closed.apply(
        lambda row: row['iv_entry_atm'] / row['rv_duration'] if row['rv_duration'] > 0 else None, 
        axis=1
    )
    
    df_closed['vrp_spread'] = df_closed['iv_entry_atm'] - df_closed['rv_duration']
    
    return df_closed

def plot_vrp_analysis(df):
    """Generate VRP Dashboard Plot."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Setup Figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), gridspec_kw={'height_ratios': [2, 1]})
    
    # Filter valid entries for plotting (must have RV data)
    df_plot = df.dropna(subset=['rv_duration']) if not df.empty else df
    df_plot = df_plot[df_plot['rv_duration'] > 0]
    
    has_data = len(df_plot) >= 3
    
    # --- SUBPLOT 1: SCATTER (IV vs RV) ---
    if has_data:
        # Scatter Points
        scatter = ax1.scatter(
            df_plot['iv_entry_atm'], 
            df_plot['rv_duration'],
            c=df_plot['final_pnl_usd'] > 0, 
            cmap='RdYlGn', 
            s=df_plot['final_pnl_usd'].abs().clip(50, 300),
            alpha=0.8,
            edgecolors='w'
        )
    else:
        # Placeholder text
        ax1.text(0.5, 0.5, "INSUFFICIENT DATA\nNeed automated exits to populate RV", 
                 ha='center', va='center', fontsize=16, color='gray', transform=ax1.transAxes)

    # Reference Line (y=x)
    lims = [0, 0.5] # Default limits (0% to 50% Vol)
    if has_data:
        max_vol = max(df_plot['iv_entry_atm'].max(), df_plot['rv_duration'].max()) * 1.1
        lims = [0, max_vol]
        
    ax1.plot(lims, lims, 'w--', alpha=0.5, label='Fair Value (IV=RV)')
    ax1.set_xlim(lims)
    ax1.set_ylim(lims)
    
    # Zones
    ax1.text(0.6, 0.1, "POSITIVE VRP ZONE\n(Result: Edge)", transform=ax1.transAxes, color='#2ecc71', fontsize=10, ha='center', alpha=0.7)
    ax1.text(0.1, 0.8, "NEGATIVE VRP ZONE\n(Result: Drag)", transform=ax1.transAxes, color='#e74c3c', fontsize=10, ha='left', alpha=0.7)
    
    ax1.set_title("VRP Analysis: Implied vs Realized Volatility", fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel("Implied Volatility (IV) at Entry (Annualized)")
    ax1.set_ylabel("Realized Volatility (RV) during Hold (Annualized)")
    ax1.grid(True, alpha=0.2)
    
    # --- SUBPLOT 2: VRP SPREAD OVER TIME ---
    if has_data:
        colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in df_plot['vrp_spread']]
        ax2.bar(df_plot['timestamp'], df_plot['vrp_spread'], color=colors, alpha=0.7, width=0.04)
        ax2.axhline(0, color='white', linestyle='--', alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "Waiting for automated trade history...", 
                 ha='center', va='center', fontsize=12, color='gray', transform=ax2.transAxes)
        
    ax2.set_title("VRP Spread per Trade (IV - RV)", fontsize=12, pad=10)
    ax2.set_ylabel("Spread (IV - RV)")
    ax2.set_xlabel("Trade Date")
    ax2.grid(True, alpha=0.2)
    
    # --- STATISTICS BOX ---
    textstr = "No Closed Trades"
    if not df.empty:
        total_pnl = df['final_pnl_usd'].sum()
        win_rate = (len(df[df['final_pnl_usd'] > 0]) / len(df) * 100)
        
        harvest_text = "N/A"
        vrp_text = "N/A"
        
        if not df_plot.empty:
            avg_harvest = df_plot['harvest_ratio'].mean()
            avg_vrp = df_plot['vrp_spread'].mean()
            harvest_text = f"{avg_harvest:.2f}"
            vrp_text = f"{avg_vrp:.2%}"
            
        textstr = '\n'.join((
            f"Total PnL: ${total_pnl:.2f}",
            f"Win Rate: {win_rate:.1f}%",
            f"Trades: {len(df)}",
            f"Avg Harvest: {harvest_text}",
            f"Avg VRP Spread: {vrp_text}"
        ))
        
    props = dict(boxstyle='round', facecolor='#2c3e50', alpha=0.8, edgecolor='none')
    ax1.text(0.02, 0.95, textstr, transform=ax1.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    # Save
    plt.tight_layout()
    plt.savefig(PLOT_OUTPUT, dpi=150)
    print(f"   📊 Plot saved to: {PLOT_OUTPUT}")
    
    return textstr

def print_summary(df, stats_text):
    print("\n" + "═" * 40)
    print("  VRP DASHBOARD SUMMARY")
    print("═" * 40)
    
    valid_rv = df[df['rv_duration'] > 0] if 'rv_duration' in df.columns and not df.empty else []
    
    print(f"  Total Closed Trades:  {len(df)}")
    print(f"  Valid RV Data Points: {len(valid_rv)}")
    
    # Parse stats from text block for console display
    labels = ["Total PnL", "Win Rate", "Avg Harvest", "Avg VRP Spread"]
    for line in stats_text.split('\n'):
        if any(l in line for l in labels):
            print(f"  {line}")
            
    print("═" * 40 + "\n")

if __name__ == "__main__":
    print("🧪 Generatng VRP Dashboard...")
    df = load_data()
    stats = plot_vrp_analysis(df)
    print_summary(df, stats)
