from ib_insync import *
from datetime import datetime

def check_exits():
    ib = IB()
    try:
        # Use a different clientID to avoid conflict
        ib.connect('127.0.0.1', 7497, clientId=98) 
        
        print("\n--- Executions Today for XSP ---")
        exec_filter = ExecutionFilter(symbol='XSP')
        executions = ib.reqExecutions(exec_filter)
        
        today = datetime.now().date()
        today_execs = [e for e in executions if e.execution.time.date() == today]
        
        if not today_execs:
            print("No executions found for today.")
        
        # Sort by time
        today_execs.sort(key=lambda e: e.execution.time)
        
        for ex in today_execs:
            # Convert to local time
            dt_utc = ex.execution.time
            dt_local = dt_utc.astimezone()
            
            print(f"Time: {dt_local.strftime('%H:%M:%S')} | Action: {ex.execution.side} | Strike: {ex.contract.strike}{ex.contract.right} | Price: ${ex.execution.price:.2f}")

        print("\n--- Current Positions ---")
        positions = ib.positions()
        for p in positions:
            if p.contract.symbol == 'XSP':
                print(f"Position: {p.position} | Contract: {p.contract.lastTradeDateOrContractMonth} {p.contract.strike}{p.contract.right}")
        
        if not any(p.contract.symbol == 'XSP' for p in positions):
            print("No XSP positions found currently.")
            
        ib.disconnect()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_exits()
