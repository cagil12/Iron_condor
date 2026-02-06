
import sys
import time
from ib_insync import IB

# Add project root to path
sys.path.insert(0, '.')

from src.data.ib_connector import IBConnector
from src.utils.config import get_live_config

def main():
    print("🧹 ORPHAN ORDER CLEANUP")
    print("-" * 50)
    
    # Random Client ID
    import random
    config = get_live_config()
    config['client_id'] = 890 + random.randint(1, 99)
    if 'ibkr' in config:
        config['paper_port'] = config['ibkr'].get('paper_port', 7497) 
        config['live_port'] = config['ibkr'].get('live_port', 7496)
        config['host'] = config['ibkr'].get('host', '127.0.0.1')
        config['timeout'] = config['ibkr'].get('timeout', 30)
    else:
        config['paper_port'] = 7497 # Fallback
        
    connector = IBConnector(config=config)
    
    try:
        connector.connect()
        ib = connector.ib
        print(f"✅ Connected (Client {config['client_id']})")
        
        # Req Global Open Orders (to see orphans from other Client IDs)
        ib.reqAllOpenOrders()
        time.sleep(1)
        
        orders = ib.openOrders()
        print(f"🔍 Found {len(orders)} open orders.")
        
        if not orders:
            print("✅ No open orders. Clean.")
            return

        print("🔥 Invoking Global Cancel (reqGlobalCancel)...")
        ib.reqGlobalCancel()
        time.sleep(1)

        print("🔪 Aggressively cancelling individual orders...")
        for o in orders:
            print(f"   🗑️ Cancelling Order {o.orderId} (Client {o.clientId})...")
            ib.cancelOrder(o)
            
        # Verify
        time.sleep(2)
        remaining = ib.openOrders()
        if remaining:
            print(f"⚠️ {len(remaining)} orders still lingering.")
        else:
            print("✅ All orders cancelled.")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        connector.disconnect()
        print("🔌 Disconnected.")

if __name__ == "__main__":
    main()
