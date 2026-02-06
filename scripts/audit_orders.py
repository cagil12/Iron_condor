from ib_insync import *
import yaml
import os
from datetime import datetime

def audit_orders():
    # Load config directly
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'base.yaml')
    print(f"📂 Loading config from: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"🔍 Config Keys: {list(config.keys())}")
    
    # Flatten IBKR config if nested
    if 'ibkr' in config:
        print("Found 'ibkr' section, flattening...")
        for k, v in config['ibkr'].items():
            if k not in config:
                config[k] = v
    
    # Defaults
    if 'paper_port' not in config: config['paper_port'] = 7497
    if 'live_port' not in config: config['live_port'] = 7496
    if 'host' not in config: config['host'] = '127.0.0.1'
    if 'client_id' not in config: config['client_id'] = 999 

    ib = IB()
    try:
        # Determine port
        is_paper = config.get('paper_mode', True)
        port = config['paper_port'] if is_paper else config['live_port']
        print(f"🔌 Connecting to TWS on port {port} (Paper={is_paper})...")
        
        ib.connect(config['host'], port, clientId=777) # Use 777 for audit
        print("✅ Connected!")

        print("\n--- 1. OPEN ORDERS (reqAllOpenOrders) ---")
        open_orders = ib.reqAllOpenOrders()
        if not open_orders:
            print("No active open orders found.")
        else:
            for o in open_orders:
                print(f"⚠️ OPEN: ID={o.orderId}, Action={o.action}, Type={o.orderType}, Status=?")

        print("\n--- 2. RECENT EXECUTIONS (reqExecutions) ---")
        exec_filter = ExecutionFilter()
        executions = ib.reqExecutions(exec_filter)
        if not executions:
            print("No executions found in this session.")
        else:
            # Sort by time
            executions.sort(key=lambda x: x.execution.time if x.execution.time else '')
            for fill in executions:
                print(f"💰 FILL: Time={fill.execution.time}, Symbol={fill.contract.localSymbol}, Side={fill.execution.side}, Shares={fill.execution.shares}, Price={fill.execution.price}, Ref={fill.execution.orderRef}, ID={fill.execution.orderId}")

        print("\n--- 3. COMPLETED ORDERS (reqCompletedOrders) ---")
        try:
            completed_orders = ib.reqCompletedOrders(False) # Positional arg
            if not completed_orders:
                print("No completed orders history retrieved.")
            else:
                print(f"Found {len(completed_orders)} completed/cancelled/filled orders.")
                
                def get_time(t):
                    if t.log and len(t.log) > 0:
                        return t.log[-1].time
                    return datetime.min

                # Sort by time
                completed_orders.sort(key=get_time)

                for t in completed_orders[-40:]: # Show last 40
                    try:
                        o = t.order
                        s = t.orderStatus
                        log_time = t.log[-1].time if t.log else 'N/A'
                        print(f"📜 HIST: Time={log_time}, ID={o.orderId}, Status={s.status}, Filled={s.filled}/{o.totalQuantity}, Action={o.action}, Type={o.orderType}, Ref={o.orderRef}, PermId={o.permId}")
                    except Exception as e:
                        print(f"Error parsing order: {e}")
        except Exception as e:
             print(f"Could not fetch completed orders: {e}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ ERROR: {e}")
    finally:
        ib.disconnect()
        print("\n🔌 Disconnected.")

if __name__ == "__main__":
    audit_orders()
