"""
Simple Confirmation Audit: List all XSP positions and open orders.
"""
from ib_insync import *
import pandas as pd

def audit_state():
    ib = IB()
    try:
        ib.connect('127.0.0.1', 7497, clientId=299)
    except:
        try:
            ib.connect('127.0.0.1', 7496, clientId=299)
        except:
            print("❌ Could not connect to IBKR")
            return

    print("\n" + "="*50)
    print("🔍 FINAL AUDIT: XSP POSITIONS & ORDERS")
    print("="*50)

    # 1. POSITIONS
    positions = ib.positions()
    xsp_pos = [p for p in positions if p.contract.symbol == 'XSP' and p.position != 0]
    
    if not xsp_pos:
        print("\n✅ NO OPEN POSITIONS. ACCOUNT IS FLAT.")
    else:
        print(f"\n⚠️ FOUND {len(xsp_pos)} OPEN POSITIONS:")
        print(f"{'Qty':<6} | {'Strike':<8} | {'Right':<5} | {'Cost':<8} | {'ConId'}")
        print("-" * 50)
        for p in xsp_pos:
            c = p.contract
            print(f"{p.position:<6} | {c.strike:<8} | {c.right:<5} | {p.avgCost:<8.2f} | {c.conId}")

    # 2. ORDERS
    orders = ib.openTrades()
    xsp_orders = [t for t in orders if t.contract.symbol == 'XSP']
    
    if not xsp_orders:
        print("\n✅ NO OPEN ORDERS.")
    else:
        print(f"\n⚠️ FOUND {len(xsp_orders)} OPEN ORDERS:")
        for t in xsp_orders:
            print(f"{t.order.action} {t.order.totalQuantity} {t.contract.localSymbol} ({t.orderStatus.status})")

    # 3. SUMMARY
    if not xsp_pos and not xsp_orders:
        print("\n✅✅ CONFIRMED: EVERYTHING IS CLOSED.")
    elif len(xsp_pos) == 1 and abs(xsp_pos[0].position) > 10:
         print("\n✅ CONFIRMED: Only Orphan position remains.")
    else:
         print("\n❌ WARNING: Some items remain open.")

    ib.disconnect()

if __name__ == "__main__":
    audit_state()
