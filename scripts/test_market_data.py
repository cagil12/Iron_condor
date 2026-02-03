#!/usr/bin/env python3
"""
test_market_data.py

OPTIMIZED: Respects IBKR's 100 instrument limit.
Tests reception of LIVE Greeks (Delta, Gamma) from IBKR.
"""
import sys
from datetime import datetime

sys.path.insert(0, '.')

try:
    from ib_insync import IB, Index, Option
except ImportError:
    print("❌ ib_insync not found. Run: pip install ib_insync")
    sys.exit(1)

# IBKR Limit
MAX_INSTRUMENTS = 100
STRIKE_RANGE = 15  # +/- 15 points from spot

def main():
    print("🚀 IBKR LIVE MARKET DATA & GREEKS TEST (OPTIMIZED)")
    print("=" * 55)
    
    ib = IB()
    
    # 1. Connect
    connected = False
    for port in [7496, 7497]:
        try:
            print(f"🔌 Trying port {port}...")
            ib.connect('127.0.0.1', port, clientId=778)
            connected = True
            print(f"✅ Connected (Port {port})")
            break
        except Exception:
            pass
    
    if not connected:
        print("❌ Could not connect to TWS.")
        return

    try:
        # 2. Get Spot Price FIRST
        contract = Index('XSP', 'CBOE')
        ib.qualifyContracts(contract)
        
        ib.reqMarketDataType(1)  # Live
        ticker = ib.reqMktData(contract, '', False, False)
        
        for _ in range(10):
            ib.sleep(0.5)
            if ticker.last or ticker.close:
                break
                
        spot = ticker.last if ticker.last else ticker.close
        if not spot:
            print("❌ No spot price. Using fallback 694.")
            spot = 694.0
            
        print(f"📊 XSP Spot: ${spot:.2f}")
        
        # 3. Get Option Chain (Metadata only - no quota used)
        print("🔗 Fetching option chain metadata...")
        chains = ib.reqSecDefOptParams(contract.symbol, '', contract.secType, contract.conId)
        
        if not chains:
            print("❌ No option chain found.")
            return
            
        chain = chains[0]
        all_strikes = sorted(chain.strikes)
        expirations = sorted(chain.expirations)
        
        # Find nearest expiry (today or next trading day)
        today_str = datetime.now().strftime('%Y%m%d')
        target_expiry = None
        for exp in expirations:
            if exp >= today_str:
                target_expiry = exp
                break
        
        if not target_expiry:
            print("❌ No valid expiration found.")
            return
            
        print(f"📅 Expiry: {target_expiry}")
        
        # 4. AGGRESSIVE FILTERING (Python side)
        lower_bound = spot - STRIKE_RANGE
        upper_bound = spot + STRIKE_RANGE
        
        filtered_strikes = [k for k in all_strikes if lower_bound < k < upper_bound]
        
        print(f"🎯 Filtering: {len(all_strikes)} total → {len(filtered_strikes)} in range [{lower_bound:.0f} - {upper_bound:.0f}]")
        
        # Safety check
        num_contracts = len(filtered_strikes) * 2  # Puts + Calls
        if num_contracts > MAX_INSTRUMENTS:
            print(f"⚠️ Still too many ({num_contracts}). Reducing further...")
            # Take only closest 20 strikes
            center = min(filtered_strikes, key=lambda x: abs(x - spot))
            filtered_strikes = [k for k in filtered_strikes if abs(k - center) <= 10]
            num_contracts = len(filtered_strikes) * 2
            
        print(f"📦 Requesting data for {num_contracts} contracts...")
        
        # 5. Build Contracts & Request Data
        contracts = []
        for k in filtered_strikes:
            contracts.append(Option('XSP', target_expiry, k, 'P', 'SMART'))
            contracts.append(Option('XSP', target_expiry, k, 'C', 'SMART'))
            
        qualified = ib.qualifyContracts(*contracts)
        
        print("📡 Requesting Greeks (waiting 4s)...")
        tickers = []
        for c in qualified:
            t = ib.reqMktData(c, '', False, False)
            tickers.append(t)
            
        ib.sleep(4)
        
        # 6. Analyze & Display
        print(f"\n{'Strike':<8} | {'Type':<4} | {'Price':<8} | {'Delta':<8} | {'Notes'}")
        print("-" * 55)
        
        best_put = None
        best_call = None
        min_put_diff = 999
        min_call_diff = 999
        
        results = []
        
        for t in tickers:
            k = t.contract.strike
            right = t.contract.right
            price = t.midpoint() if t.midpoint() > 0 else (t.close if t.close else 0)
            
            delta = None
            if t.modelGreeks:
                delta = t.modelGreeks.delta
                
            display_delta = f"{delta:.3f}" if delta is not None else "NaN"
            display_price = f"${price:.2f}"
            
            # Find best matches
            if delta is not None:
                if right == 'P':
                    diff = abs(delta - (-0.10))
                    if diff < min_put_diff:
                        min_put_diff = diff
                        best_put = t
                else:
                    diff = abs(delta - 0.10)
                    if diff < min_call_diff:
                        min_call_diff = diff
                        best_call = t
                        
            results.append({'strike': k, 'right': right, 'price': display_price, 'delta': display_delta})
        
        # Sort and print
        results.sort(key=lambda x: (x['strike'], x['right']))
        
        for r in results:
            note = ""
            if best_put and r['strike'] == best_put.contract.strike and r['right'] == 'P':
                note = "★ BEST PUT"
            if best_call and r['strike'] == best_call.contract.strike and r['right'] == 'C':
                note = "★ BEST CALL"
            print(f"{r['strike']:<8} | {r['right']:<4} | {r['price']:<8} | {r['delta']:<8} | {note}")
        
        # Summary
        print("-" * 55)
        if best_put and best_put.modelGreeks:
            print(f"✅ Best Put:  {best_put.contract.strike}P (Delta: {best_put.modelGreeks.delta:.3f})")
        if best_call and best_call.modelGreeks:
            print(f"✅ Best Call: {best_call.contract.strike}C (Delta: {best_call.modelGreeks.delta:.3f})")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        ib.disconnect()
        print("\n🔌 Disconnected")

if __name__ == "__main__":
    main()
