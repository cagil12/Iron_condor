#!/usr/bin/env python3
"""
test_combo_structure.py

CRITICAL VALIDATION SCRIPT
Tests IBKR Combo (BAG) order structure and Margin Impact without execution.
"""
import sys
import time
from datetime import datetime, date, timedelta
sys.path.insert(0, '.')

from src.data.ib_connector import IBConnector
from src.strategy.execution import LiveExecutor
from src.utils.config import get_live_config
from ib_insync import Contract, LimitOrder, ComboLeg

def main():
    print("🚀 IBKR COMBO (BAG) STRUCTURE VALIDATION")
    print("========================================")
    
    config = get_live_config()
    connector = IBConnector(config.get('ibkr', {}))
    
    print("🔌 Connecting to TWS (LIVE)...")
    success, msg = connector.connect(paper=False)
    if not success:
        print(f"❌ Connection failed: {msg}")
        return

    try:
        executor = LiveExecutor(connector)
        
        # 1. Get Reference Price (Spot)
        spot = connector.get_live_price('XSP')
        if not spot:
            print("⚠️ Could not get XSP spot, using fallback 690")
            spot = 690.0
            
        print(f"📊 Reference Spot: ${spot:.2f}")
        
        # 2. Build Test Params (ATM +/- 10)
        # We need strikes that likely exist.
        center = round(spot / 5) * 5 # Round to nearest 5
        short_put = center - 10
        short_call = center + 10
        width = 1.0
        
        long_put = short_put - width
        long_call = short_call + width
        
        print(f"🎯 Test Structure: {short_put}P/{short_call}C (Wings {width})")
        
        # 3. Resolve Contracts
        # Use next Friday to ensure liquidity/existence
        today = date.today()
        days_ahead = 4 - today.weekday()  # Friday = 4
        if days_ahead <= 0:
            days_ahead += 7
        next_friday = today + timedelta(days=days_ahead)
        expiry = next_friday.strftime('%Y%m%d')
        
        print(f"📅 Expiry: {expiry}")
        
        print("🔨 Qualifying legs...")
        c_short_put = executor.build_option_contract(short_put, 'P', expiry)
        c_long_put = executor.build_option_contract(long_put, 'P', expiry)
        c_short_call = executor.build_option_contract(short_call, 'C', expiry)
        c_long_call = executor.build_option_contract(long_call, 'C', expiry)
        
        # 4. Construct BAG
        print("📦 Constructing BAG...")
        contract = Contract()
        contract.symbol = 'XSP'
        contract.secType = 'BAG'
        contract.currency = 'USD'
        contract.exchange = 'SMART'
        
        combo_legs = []
        legs_def = [
            (c_short_put, 'SELL'), # Short Put
            (c_long_put, 'BUY'),   # Long Put
            (c_short_call, 'SELL'),# Short Call
            (c_long_call, 'BUY')   # Long Call
        ]
        
        for leg_c, action in legs_def:
            if not leg_c.conId:
                connector.ib.qualifyContracts(leg_c) # Must qualify
            
            leg = ComboLeg() # Correct usage (class directly)
            leg.conId = leg_c.conId
            leg.ratio = 1
            leg.action = action
            leg.exchange = 'SMART'
            combo_legs.append(leg)
            
        contract.comboLegs = combo_legs
        
        # 5. Create Order
        # Price -0.50 (Credit $50) - Realistic test price
        limit_price = -0.50
        order = LimitOrder('BUY', 1, limit_price)
        order.tif = 'DAY'
        
        print(f"📝 Order: BUY 1 BAG @ ${limit_price} (Credit $0.50)")
        
        # 6. WHAT-IF VALIDATION
        print("\n🛡️ RUNNING WHAT-IF VALIDATION...")
        state = connector.ib.whatIfOrder(contract, order)
        
        init_margin = float(state.initMarginChange)
        maint_margin = float(state.maintMarginChange)
        comm = float(state.commission) if state.commission else 0.0
        
        print("-" * 40)
        print(f"💵 Initial Margin Change: ${init_margin:.2f}")
        print(f"💵 Maint Margin Change:   ${maint_margin:.2f}")
        print(f"💸 Est. Commission:       ${comm:.2f}")
        print("-" * 40)
        
        # Validation Logic
        MAX_MARGIN = 250.0
        
        if init_margin < MAX_MARGIN:
             print("✅ VALIDATION PASSED: Margin within limits.")
        else:
             print(f"❌ VALIDATION FAILED: Margin ${init_margin:.2f} too high (Exp < {MAX_MARGIN})")
             
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        connector.disconnect()
        print("\n🔌 Disconnected")

if __name__ == "__main__":
    main()
