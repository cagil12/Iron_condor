import asyncio
from datetime import datetime
from ib_insync import *
import pandas as pd
import sys

# Connect
ib = IB()
try:
    ib.connect('127.0.0.1', 7497, clientId=111)
except Exception as e:
    print(f"Connection failed: {e}")
    sys.exit(1)

print("✅ Connected to IBKR")

# Trade Details
# 10:36 AM ET = 15:36 UTC
trade_time = datetime(2026, 2, 9, 15, 36, 0) 
con_ids = {
    '684P': 841471209,
    '704C': 842236402,
    '682P': 841471172,
    '706C': 843130228
}

async def fetch_data():
    results = {}
    
    # 1. Get Underlying Spot Price (XSP)
    xsp = Index('XSP', 'CBOE')
    await ib.qualifyContractsAsync(xsp)
    
    # Use UTC format: YYYYMMDD-HH:MM:SS
    end_time_str = trade_time.strftime('%Y%m%d-%H:%M:%S')
    
    print(f"⏳ Fetching Historical Data ending {end_time_str} (UTC)...")
    
    # Historical Spot
    spot_bars = await ib.reqHistoricalDataAsync(
        xsp, 
        endDateTime=end_time_str,
        durationStr='60 S',
        barSizeSetting='1 min',
        whatToShow='MIDPOINT',
        useRTH=True
    )
    
    if spot_bars:
        spot_price = spot_bars[-1].close
        print(f"   📉 XSP Spot at {spot_bars[-1].date}: {spot_price}")
        results['spot'] = spot_price
    else:
        print("   ❌ Could not get Spot Price")
        results['spot'] = None

    # 2. Get IV for each leg
    for name, con_id in con_ids.items():
        contract = Contract(conId=con_id)
        await ib.qualifyContractsAsync(contract)
        
        iv_bars = await ib.reqHistoricalDataAsync(
            contract,
            endDateTime=end_time_str,
            durationStr='60 S',
            barSizeSetting='1 min',
            whatToShow='OPTION_IMPLIED_VOLATILITY', # Tries to get IV
            useRTH=True
        )
        
        if iv_bars:
            iv = iv_bars[-1].close
            print(f"   📊 {name} ({contract.localSymbol}) IV: {iv}")
            results[f"{name}_iv"] = iv
        else:
            print(f"   ⚠️ {name} IV not available. Trying MIDPOINT...")
            # Fallback to Price to calculate IV later?
            price_bars = await ib.reqHistoricalDataAsync(
                contract,
                endDateTime=end_time_str,
                durationStr='60 S',
                barSizeSetting='1 min',
                whatToShow='MIDPOINT',
                useRTH=True
            )
            if price_bars:
                 print(f"   💰 {name} Mid Price: {price_bars[-1].close}")
                 results[f"{name}_price"] = price_bars[-1].close
            else:
                 print(f"   ❌ No data for {name}")

    return results

# Run
data = ib.run(fetch_data())
print("\n--- Final Data ---")
print(data)

ib.disconnect()
