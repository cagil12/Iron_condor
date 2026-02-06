
import sys
import time
from datetime import datetime
from ib_insync import IB, util

# Add project root to path
sys.path.insert(0, '.')

from src.data.ib_connector import IBConnector
from src.utils.config import get_live_config

def main():
    print("💰 PnL VERIFICATION (Today's Session)")
    print("-" * 50)
    
    # Use random offset to avoid conflict
    import random
    config = get_live_config()
    config['client_id'] = 900 + random.randint(1, 99) 
    
    # Inject missing keys expected by IBConnector flat config
    if 'ibkr' in config:
        config['paper_port'] = config['ibkr'].get('paper_port', 7497)
        config['live_port'] = config['ibkr'].get('live_port', 7496)
        config['host'] = config['ibkr'].get('host', '127.0.0.1')
        config['timeout'] = config['ibkr'].get('timeout', 30)
    else:
        # Fallback defaults
        config['paper_port'] = 7497
        config['live_port'] = 7496
        config['host'] = '127.0.0.1'
        config['timeout'] = 30
    
    connector = IBConnector(config=config)
    
    try:
        # Connect
        connector.connect()
        ib = connector.ib
        print(f"✅ Connected to TWS (Client {config['client_id']})")
        
        # Request Executions
        print("🔍 Fetching executions for today...")
        execs = ib.reqExecutions()
        
        total_pnl = 0.0
        xsp_execs = []
        
        for fill in execs:
            contract = fill.contract
            exec_detail = fill.execution
            
            # Filter for XSP and Today
            # Exec time is string 'YYYYMMDD HH:MM:SS' or similar. 
            # We assume all executions returned are relevant if we just rely on reqExecutions() default (all reports accessible).
            # But let's filter by Symbol XSP.
            
            if contract.symbol == 'XSP':
                impact = 0.0
                multiplier = float(contract.multiplier) if contract.multiplier else 100.0
                price = exec_detail.price
                shares = exec_detail.shares
                side = exec_detail.side # 'BOT' or 'SLD'
                
                # Cash Flow = -Price * Quantity (if Buy) or +Price * Quantity (if Sell)
                # PnL logic:
                # Buy 1 @ 10: Cash -1000
                # Sell 1 @ 15: Cash +1500
                # Net: +500
                
                direction = -1 if side == 'BOT' else 1
                cash_flow = price * shares * multiplier * direction
                
                total_pnl += cash_flow
                xsp_execs.append(fill)
                
                print(f"   {exec_detail.time.strftime('%H:%M:%S')} {side} {shares} {contract.right}{contract.strike} @ {price:.2f} = ${cash_flow:.2f}")

        print("-" * 50)
        print(f"📊 Total Realized PnL (Cash Flow): ${total_pnl:.2f}")
        
        if len(xsp_execs) == 0:
             print("⚠️ No executions found for XSP today.")
        else:
             print(f"✅ Found {len(xsp_execs)} fills.")
             
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        connector.disconnect()
        print("🔌 Disconnected.")

if __name__ == "__main__":
    main()
