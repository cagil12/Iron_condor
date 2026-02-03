#!/usr/bin/env python3
"""
verify_combo_logic.py

Test script to verify IBKR Combo (BAG) order construction and Margin Impact.
Uses `whatIfOrder` to check if the trade would be accepted without executing it.
"""
import sys
import time
sys.path.insert(0, '.')

from src.data.ib_connector import IBConnector
from src.strategy.execution import LiveExecutor
from src.utils.config import get_live_config
from ib_insync import LimitOrder

def main():
    print("🚀 COMBOS/BAG LOGIC VERIFICATION")
    print("=================================")
    
    config = get_live_config()
    connector = IBConnector(config.get('ibkr', {}))
    
    # Connect
    print("🔌 Connecting to TWS...")
    success, msg = connector.connect(paper=False) # Connect to LIVE for margin check
    if not success:
        print(f"❌ Connection failed: {msg}")
        return

    try:
        executor = LiveExecutor(connector)
        
        # 1. Get current spot
        spot = connector.get_live_price('XSP')
        if not spot:
            print("❌ Could not get XSP spot price")
            return
            
        print(f"📊 XSP Spot: ${spot:.2f}")
        
        # 2. Define Iron Condor params (approx closest strikes)
        # 10 delta approx strikes
        center = round(spot)
        short_put = center - 10
        short_call = center + 10
        width = 1.0
        
        long_put = short_put - width
        long_call = short_call + width
        
        print(f"🎯 Test Setup: {short_put}P/{short_call}C (Wings {width})")
        
        # 3. Build Contracts (needed for conId)
        from datetime import date, timedelta
        expiry = (date.today() + timedelta(days=(4-date.today().weekday()) % 7)).strftime('%Y%m%d')
        if expiry == date.today().strftime('%Y%m%d'):
             # If today is Friday, use next Friday
             expiry = (date.today() + timedelta(days=7)).strftime('%Y%m%d')
             
        print(f"📅 Expiry: {expiry}")
        
        print("🔨 Building legs...")
        c_short_put = executor.build_option_contract(short_put, 'P', expiry)
        c_long_put = executor.build_option_contract(long_put, 'P', expiry)
        c_short_call = executor.build_option_contract(short_call, 'C', expiry)
        c_long_call = executor.build_option_contract(long_call, 'C', expiry)
        
        # 4. Define Combo Legs
        legs = [
            {'contract': c_short_put, 'ratio': 1, 'action': 'SELL'},
            {'contract': c_long_put,  'ratio': 1, 'action': 'BUY'},
            {'contract': c_short_call,'ratio': 1, 'action': 'SELL'},
            {'contract': c_long_call, 'ratio': 1, 'action': 'BUY'}
        ]
        
        bag_contract = executor.build_combo_contract(legs)
        print(f"📦 BAG Contract created: {bag_contract}")
        
        # 5. Create Test Order (Limit Order with negative price = Credit)
        # Assume credit of $0.50 ($50)
        credit = 0.50
        limit_price = -credit 
        
        order = LimitOrder('BUY', 1, limit_price)
        order.tif = 'DAY'
        
        print(f"📝 Test Order: BUY 1 Combo @ ${limit_price} (Credit ${credit})")
        
        # 6. Check Margin (whatIf)
        print("\n🛡️ Running Margin Check (whatIfOrder)...")
        state = connector.ib.whatIfOrder(bag_contract, order)
        
        init_margin = float(state.initMarginChange)
        maint_margin = float(state.maintMarginChange)
        commission = float(state.commission) if state.commission else 0.0
        
        print("-" * 40)
        print(f"💵 Initial Margin Change: ${init_margin:.2f}")
        print(f"💵 Maint Margin Change:   ${maint_margin:.2f}")
        print(f"💸 Est. Commission:       ${commission:.2f}")
        print("-" * 40)
        
        if init_margin < 200:
            print("✅ MARGIN CHECK PASSED! (< $200)")
        else:
            print("❌ MARGIN TOO HIGH! (> $200)")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        connector.disconnect()
        print("\n🔌 Disconnected")

if __name__ == "__main__":
    main()
