
import sys
import time
from datetime import datetime
from ib_insync import IB, util

# Add project root to path
sys.path.insert(0, '.')

from src.data.ib_connector import IBConnector
from src.utils.config import get_live_config

def main():
    print("🔍 DIAGNOSTIC: Checking IBKR Positions & Permissions")
    print("-" * 50)
    
    config = get_live_config()
    connector = IBConnector()
    
    try:
        # Connect
        connector.connect()
        ib = connector.ib
        print(f"✅ Connected to TWS")
        print(f"   Account: {config.get('account_id')}")
        
        # Get Positions
        positions = ib.positions()
        if not positions:
            print("⚪ No open positions found.")
        else:
            print(f"📊 Found {len(positions)} open positions:")
            for p in positions:
                contract = p.contract
                print(f"   • {contract.symbol} {contract.right} {contract.strike} {contract.lastTradeDateOrContractMonth}")
                print(f"     ID: {contract.conId} | Exch: {contract.exchange} | Pos: {p.position} | Cost: {p.avgCost}")
                
                # Check contract details
                details = ib.reqContractDetails(contract)
                if details:
                    d = details[0]
                    print(f"     Valid Exchanges: {d.validExchanges}")
                    print(f"     Order Types: {d.orderTypes}")
        
        print("-" * 50)
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        connector.disconnect()
        print("🔌 Disconnected.")

if __name__ == "__main__":
    main()
