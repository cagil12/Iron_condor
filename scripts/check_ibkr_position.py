
from ib_insync import *
import sys

def check_positions():
    ib = IB()
    try:
        # Use a different client ID to strictly avoid conflict with the live monitor (Client ID 1)
        print("🔌 Connecting to IBKR (Client ID 99)...")
        ib.connect('127.0.0.1', 7497, clientId=99)
        
        print("\n🔍 --- IBKR PORTFOLIO AUDIT ---")
        positions = ib.positions()
        portfolio = ib.portfolio()
        
        if not positions:
            print("❌ No open positions found on IBKR server.")
        else:
            print(f"✅ Found {len(positions)} open position(s):")
            for p in portfolio:
                print(f"   • {p.contract.symbol} {p.contract.secType} (ConID: {p.contract.conId})")
                print(f"     Qty: {p.position} | AvgCost: {p.averageCost:.2f} | MktPrice: {p.marketPrice:.2f}")
                print(f"     MktValue: {p.marketValue:.2f} | UnrealizedPnL: {p.unrealizedPNL:.2f} | RealizedPnL: {p.realizedPNL:.2f}")
                
        print("\n🔍 --- IBKR OPEN ORDERS AUDIT ---")
        orders = ib.reqOpenOrders()
        if not orders:
             print("ℹ️ No active open orders (TP/SL not yet resident or managed locally).")
        else:
             for o in orders:
                 print(f"   • {o.orderId} {o.action} {o.totalQuantity} {o.orderType} @ {o.lmtPrice}")
                 print(f"     Status: {o.orderStatus.status}")

        print("\n🔍 --- ACCOUNT SUMMARY ---")
        tags = ib.accountSummary()
        u_pnl = next((t.value for t in tags if t.tag == 'UnrealizedPnL'), 'N/A')
        net_liq = next((t.value for t in tags if t.tag == 'NetLiquidation'), 'N/A')
        print(f"   Net Liq: ${net_liq} | Tot Unrealized PnL: ${u_pnl}")

    except Exception as e:
        print(f"❌ Error during audit: {e}")
    finally:
        ib.disconnect()
        print("\n🔌 Disconnected.")

if __name__ == "__main__":
    check_positions()
