from ib_insync import *
import sys

def debug_contract():
    ib = IB()
    try:
        # Connect to Live TWS
        ib.connect('127.0.0.1', 7496, clientId=20, timeout=15)
        print("🔌 Connected to TWS")
        
        # 1. Search for XSP Index
        print("\n🔍 Searching for XSP Index...")
        xsp_idx = Index('XSP', 'CBOE')
        contracts = ib.reqContractDetails(xsp_idx)
        
        if not contracts:
            print("❌ No Index contract found for Index('XSP', 'CBOE')")
            # Try wider search
            xsp_idx = Index('XSP')
            contracts = ib.reqContractDetails(xsp_idx)
            
        for cd in contracts:
            c = cd.contract
            print(f"   Found: {c.symbol} {c.secType} @ {c.exchange} (ConId: {c.conId})")
            
            # Try getting market data for the first valid one
            print(f"   >> Requesting data for ConId {c.conId}...")
            ib.reqMarketDataType(3) # Delayed
            ticker = ib.reqMktData(c, '', snapshot=False)
            ib.sleep(4)
            print(f"      Last: {ticker.last}, Close: {ticker.close}, Bid: {ticker.bid}, Ask: {ticker.ask}")
            ib.cancelMktData(c)

        # 2. Search for XSP Option (Feb 20 '26, Strike 697 Call)
        # Screenshot shows: FEB 20 '26, Strike 697, Call Bid ~8.96
        print("\n🔍 Searching for XSP Option (Exp 20260220, Strike 697, Call)...")
        opt = Option('XSP', '20260220', 697, 'C', 'SMART')
        opt_contracts = ib.reqContractDetails(opt)
        
        if not opt_contracts:
            print("❌ No Option contract found")
        
        for cd in opt_contracts:
            c = cd.contract
            print(f"   Found: {c.symbol} {c.secType} @ {c.exchange} (ConId: {c.conId}, Local: {c.localSymbol})")
            
            # Request data
            print(f"   >> Requesting data...")
            ib.reqMarketDataType(3) # Delayed
            ticker = ib.reqMktData(c, '', snapshot=False)
            ib.sleep(4)
            print(f"      Bid: {ticker.bid}, Ask: {ticker.ask}, Last: {ticker.last}")
            ib.cancelMktData(c)

    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        ib.disconnect()
        print("\n🔌 Disconnected")

if __name__ == "__main__":
    debug_contract()
