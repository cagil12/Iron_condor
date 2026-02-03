#!/usr/bin/env python3
"""
test_market_data.py

Diagnostic script to verify reception of LIVE Greeks (Delta, Gamma) from IBKR.
Connects, fetches XSP option chain, and identifies 10-Delta strikes using real market data.
"""
import sys
import time
from datetime import datetime
import pandas as pd
from typing import List, Optional

sys.path.insert(0, '.')

try:
    from ib_insync import IB, Index, Option, util
except ImportError:
    print("❌ ib_insync not found. Run: pip install ib_insync")
    sys.exit(1)

def main():
    print("🚀 IBKR LIVE MARKET DATA & GREEKS TEST")
    print("=======================================")
    
    ib = IB()
    
    # 1. Connect (Try LIVE 7496 first, then PAPER 7497)
    connected = False
    ports = [7496, 7497]
    
    for port in ports:
        try:
            print(f"🔌 Attempting connection on port {port}...")
            ib.connect('127.0.0.1', port, clientId=777)
            connected = True
            print(f"✅ Connected to Port {port}")
            break
        except Exception as e:
            print(f"   ⚠️ Connection failed onto {port}")
    
    if not connected:
        print("❌ Could not connect to any TWS instance.")
        return

    try:
        # 2. Get Underlying (XSP)
        contract = Index('XSP', 'CBOE')
        ib.qualifyContracts(contract)
        
        print("⏳ Getting XSP Spot Price...")
        ib.reqMarketDataType(1)  # Live
        # ib.reqMarketDataType(3) # Delayed (Uncomment if needed)
        
        ticker = ib.reqMktData(contract, '', False, False)
        for _ in range(10):
            ib.sleep(0.5)
            if ticker.last or ticker.close:
                break
                
        spot = ticker.last if ticker.last else ticker.close
        if not spot:
            print("❌ No spot price found. Using backup estimation: 695")
            spot = 695.0
            
        print(f"📊 XSP Spot: ${spot:.2f}")
        
        # 3. Get Option Chain (Nearest Expiry)
        print("🔗 Fetching Option Chain...")
        chains = ib.reqSecDefOptParams(contract.symbol, '', contract.secType, contract.conId)
        
        # Filter for SMART exchange
        smart_chains = [c for c in chains if c.exchange == 'SMART']
        if not smart_chains:
            # Fallback to CBOE logic if SMART not explicit in chain params (common for indices)
            smart_chains = chains 
            
        if not smart_chains:
             print("❌ No option chain found.")
             return

        chain = smart_chains[0]
        expirations = sorted(chain.expirations)
        strikes = sorted(chain.strikes)
        
        # Select nearest expiry
        # Find first expiry that is today or future
        today_str = datetime.now().strftime('%Y%m%d')
        target_expiry = None
        for exp in expirations:
            if exp >= today_str:
                target_expiry = exp
                break
        
        if not target_expiry:
            print("❌ No valid expirations found.")
            return
            
        print(f"📅 Selected Expiry: {target_expiry}")
        
        # 4. Filter Strikes (Around Spot)
        # We need roughly +/- 30 points
        center_strike = min(strikes, key=lambda x: abs(x - spot))
        selected_strikes = [k for k in strikes if abs(k - center_strike) <= 30]
        
        print(f"🎯 Scanning {len(selected_strikes)} strikes around {center_strike}...")
        
        # 5. Build Contracts & Request Data
        contracts = []
        for k in selected_strikes:
            contracts.append(Option('XSP', target_expiry, k, 'P', 'SMART'))
            contracts.append(Option('XSP', target_expiry, k, 'C', 'SMART'))
            
        contracts = ib.qualifyContracts(*contracts)
        
        print("📡 Requesting Greeks (waiting 4s)...")
        tickers = []
        for c in contracts:
            t = ib.reqMktData(c, '', False, False) 
            tickers.append(t)
            
        ib.sleep(4) # Allow data to flow
        
        # 6. Analyze Data
        print("\n🧐 ANALYSIS RESULTS:")
        print(f"{'Strike':<8} | {'Type':<4} | {'Price':<8} | {'Delta':<8} | {'Gamma':<8} | {'Notes'}")
        print("-" * 65)
        
        best_put = None
        best_call = None
        min_put_diff = 999
        min_call_diff = 999
        
        results = []
        
        for t in tickers:
            k = t.contract.strike
            right = t.contract.right
            
            # Price
            prica = t.midpoint() if t.midpoint() > 0 else (t.close if t.close else 0)
            
            # Greeks
            delta = None
            gamma = None
            
            # IBKR transmits modelGreeks usually
            if t.modelGreeks:
                delta = t.modelGreeks.delta
                gamma = t.modelGreeks.gamma
            
            display_delta = f"{delta:.3f}" if delta is not None else "NaN"
            display_gamma = f"{gamma:.3f}" if gamma is not None else "NaN"
            display_price = f"${prica:.2f}"
            
            # Selection Logic
            note = ""
            if delta is not None:
                if right == 'P':
                    # Target -0.10
                    diff = abs(delta - (-0.10))
                    if diff < min_put_diff:
                        min_put_diff = diff
                        best_put = t
                else: # Call
                    # Target 0.10
                    diff = abs(delta - 0.10)
                    if diff < min_call_diff:
                        min_call_diff = diff
                        best_call = t
            
            results.append({
                'strike': k,
                'right': right,
                'price': display_price,
                'delta': display_delta,
                'gamma': display_gamma
            })
        
        # Sort and print
        results.sort(key=lambda x: (x['strike'], x['right']))
        
        for r in results:
            # Highlight best candidates
            note = ""
            if best_put and float(r['strike']) == best_put.contract.strike and r['right'] == 'P':
                note = "★ BEST PUT (-0.10)"
            if best_call and float(r['strike']) == best_call.contract.strike and r['right'] == 'C':
                note = "★ BEST CALL (0.10)"
                
            print(f"{r['strike']:<8} | {r['right']:<4} | {r['price']:<8} | {r['delta']:<8} | {r['gamma']:<8} | {note}")
            
        # Summary
        print("-" * 65)
        if best_put and best_put.modelGreeks:
             print(f"✅ Found Put: {best_put.contract.strike}P (Delta: {best_put.modelGreeks.delta:.3f})")
        else:
             print("❌ Put Delta not found")
             
        if best_call and best_call.modelGreeks:
             print(f"✅ Found Call: {best_call.contract.strike}C (Delta: {best_call.modelGreeks.delta:.3f})")
        else:
             print("❌ Call Delta not found")

    except Exception as e:
        print(f"❌ Critical Error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        ib.disconnect()
        print("\n🔌 Disconnected")

if __name__ == "__main__":
    main()
