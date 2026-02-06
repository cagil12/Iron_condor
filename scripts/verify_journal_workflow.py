
import sys
import os
import pandas as pd
from datetime import datetime

# Add project root to path
sys.path.insert(0, '.')

from src.utils.journal import TradeJournal

def main():
    print("🧪 JOURNAL INTEGRITY TEST (What-If)")
    print("-" * 50)
    
    # 1. Setup Test Journal
    test_path = "data/journal/test_journal.csv"
    if os.path.exists(test_path):
        os.remove(test_path)
        
    journal = TradeJournal(journal_path=test_path)
    print("✅ Journal Initialized")
    
    # 2. Simulate OPEN Trade
    print("📝 Logging OPEN Trade...")
    trade_id = journal.log_trade_open(
        spot_price=580.50,
        vix_value=18.5,
        short_put_strike=570,
        short_call_strike=590,
        wing_width=10,
        entry_credit=0.50,
        max_profit_usd=50.0,
        max_loss_usd=950.0,
        delta_net=-0.02,
        initial_credit=0.50, # New Field
        iv_entry_atm=0.155,  # New Field
        reasoning="Test Entry"
    )
    print(f"   Trade ID: {trade_id}")
    
    # 3. Verify OPEN persistence
    df = pd.read_csv(test_path)
    row = df[df['trade_id'] == trade_id].iloc[0]
    
    assert row['status'] == 'OPEN'
    assert float(row['initial_credit']) == 0.50
    assert float(row['iv_entry_atm']) == 0.155
    print("✅ OPEN Entry Verified (New fields present)")
    
    # 4. Simulate CLOSE Trade
    print("📝 Logging CLOSE Trade...")
    rv_sim = 0.081234 # 6 decimals test
    spread_max = 1.25
    
    journal.log_trade_close(
        trade_id=trade_id,
        exit_reason="TP_TEST",
        final_pnl_usd=25.50,
        entry_timestamp=datetime.now(),
        max_spread_val=spread_max,
        rv_duration=rv_sim
    )
    
    # 5. Verify CLOSE persistence
    df = pd.read_csv(test_path)
    row = df[df['trade_id'] == trade_id].iloc[0]
    
    assert row['status'] == 'CLOSED'
    assert row['exit_reason'] == 'TP_TEST'
    assert float(row['final_pnl_usd']) == 25.50
    assert float(row['max_spread_val']) == 1.25
    
    # Check Precision
    rv_actual = float(row['rv_duration'])
    print(f"   RV Expected: {rv_sim:.6f}")
    print(f"   RV Saved:    {rv_actual:.6f}")
    
    assert abs(rv_actual - rv_sim) < 0.000001
    print("✅ CLOSE Entry Verified (RV precision confirmed)")
    
    print("-" * 50)
    print("🎉 ALL CHECKS PASSED. Journal system is robust.")
    
    # Cleanup
    if os.path.exists(test_path):
        os.remove(test_path)

if __name__ == "__main__":
    main()
