
import sys
import time
from datetime import datetime
from ib_insync import IB, LimitOrder, Contract

# Add project root to path
sys.path.insert(0, '.')

from src.data.ib_connector import IBConnector
from src.utils.config import get_live_config

def main():
    print("🚨 EMERGENCY CLOSE PROTOCOL INITIATED")
    print("-" * 50)
    
    connector = IBConnector()
    
    try:
        # Connect
        connector.connect()
        ib = connector.ib
        print(f"✅ Connected to TWS")
        
        # Get Positions
        positions = ib.positions()
        xsp_positions = [p for p in positions if p.contract.symbol == 'XSP' and p.position != 0]
        
        if not xsp_positions:
            print("⚪ No open XSP positions found. System clean.")
            return

        print(f"🔥 Found {len(xsp_positions)} open XSP legs. Closing immediately...")
        
        closing_trades = []
        for p in xsp_positions:
            contract = p.contract
            contract.exchange = 'SMART' # Force SMART
            ib.qualifyContracts(contract) # Auto-fill details
            
            qty = abs(p.position)
            action = 'SELL' if p.position > 0 else 'BUY'
            
            print(f"   PLEASE CLOSE: {contract.localSymbol} ({action} {qty})")
            
            # USE LIMIT ORDER "AT MARKET" (0.00 or 0.01) to bypass MarketData restriction?? 
            # Actually, Limit at 0.00 often works as MARKET protection.
            # Or use a very low price for SELL (0.01) and high for BUY (10.00).
            price = 0.01 if action == 'SELL' else 50.0
            order = LimitOrder(action, qty, price)
            order.tif = 'DAY' # FIX: Explicitly set TIF to DAY to match preset
            
            trade = ib.placeOrder(contract, order)
            closing_trades.append(trade)
            
        # Wait for fills
        print("   ⏳ Waiting for fills (max 15s)...")
        waited = 0
        while waited < 15:
            ib.sleep(1)
            pending = [t for t in closing_trades if t.orderStatus.status != 'Filled']
            if not pending:
                print("   ✅ ALL POSITIONS CLOSED.")
                break
            waited += 1
            
        # Final status check
        remaining = [t for t in closing_trades if t.orderStatus.status != 'Filled']
        if remaining:
             print(f"   ❌ WARNING: {len(remaining)} orders failed to fill.")
             for t in remaining:
                 print(f"      {t.contract.localSymbol}: {t.orderStatus.status} - {t.orderStatus.whyHeld}")
                 
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        connector.disconnect()
        print("🔌 Disconnected.")

if __name__ == "__main__":
    main()
