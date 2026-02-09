
from ib_insync import *
import sys

def audit_history():
    ib = IB()
    try:
        print("🔌 Connecting to IBKR (Client ID 98)...")
        ib.connect('127.0.0.1', 7497, clientId=98)
        
        print("\n📜 --- EXECUTION HISTORY (Today) ---")
        execs = ib.reqExecutions()
        if not execs:
            print("❌ No executions found on server.")
        else:
            for fill in execs:
                # fill is a Fill object (contract, execution, commissionReport, time)
                e = fill.execution
                c = fill.contract
                print(f"   • Time: {e.time} | Side: {e.side} | Qty: {e.shares} | Sym: {c.symbol} | Price: {e.price} | OrderId: {e.orderId}")
                
        print("\n🧮 --- NET POSITION CALCULATOR (From Executions) ---")
        net_pos = {}
        for fill in execs:
            c = fill.contract
            qty = fill.execution.shares
            side = fill.execution.side
            
            if c.symbol not in net_pos: net_pos[c.symbol] = {}
            if c.conId not in net_pos[c.symbol]: net_pos[c.symbol][c.conId] = {'net': 0, 'desc': f"{c.right} {c.strike}"}
            
            # BOT = +1, SLD = -1
            direction = 1 if side == 'BOT' else -1
            net_pos[c.symbol][c.conId]['net'] += (qty * direction)
            
        for sym, buckets in net_pos.items():
            print(f"   Symbol: {sym}")
            for conId, data in buckets.items():
                status = "OPEN" if data['net'] != 0 else "CLOSED"
                print(f"     • {data['desc']} (ID {conId}): Net {data['net']} [{status}]")

    except Exception as e:
        print(f"❌ Error during audit: {e}")
    finally:
        ib.disconnect()
        print("\n🔌 Disconnected.")

if __name__ == "__main__":
    audit_history()
