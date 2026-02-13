"""Final cleanup with long cooldown to break delayed-fill loop."""
from ib_insync import *
import time as t

ib = IB()
ib.connect('127.0.0.1', 7497, clientId=206)
ib.sleep(2)

# STEP 1: CANCEL EVERYTHING
print("Step 1: Cancel everything...")
ib.reqGlobalCancel()
ib.sleep(3)

# Force cancel any remaining
trades = ib.openTrades()
for tr in trades:
    try:
        ib.cancelOrder(tr.order)
    except:
        pass
ib.sleep(3)

print(f"Open orders: {len(ib.openTrades())}")

# STEP 2: WAIT for delayed fills to settle
print("\nStep 2: Waiting 30s for delayed fills to settle...")
for i in range(30, 0, -5):
    ib.sleep(5)
    positions = ib.positions()
    xsp = [p for p in positions if p.contract.symbol == 'XSP' and p.position != 0]
    non_orphan = [p for p in xsp if abs(p.position) < 10]
    print(f"  {i}s remaining... Non-orphan positions: {len(non_orphan)}")
    for p in non_orphan:
        print(f"    {p.contract.strike}{p.contract.right} qty={p.position}")

# STEP 3: Clean state audit
print("\nStep 3: Clean state audit...")
positions = ib.positions()
xsp = [p for p in positions if p.contract.symbol == 'XSP' and p.position != 0]
closeable = [p for p in xsp if abs(p.position) < 10]

print(f"Total XSP: {len(xsp)} | Closeable: {len(closeable)}")
for p in xsp:
    tag = "ORPHAN" if abs(p.position) >= 10 else "CLOSE"
    print(f"  [{tag}] {p.contract.strike}{p.contract.right} qty={p.position}")

if not closeable:
    print("\n✅ Only orphan remains! Bot-closeable positions are clear.")
    ib.disconnect()
    exit()

# STEP 4: Close remaining with fresh state
print("\nStep 4: Closing remaining positions...")
for pos in closeable:
    c = pos.contract
    c.exchange = 'SMART'
    ib.qualifyContracts(c)
    
    action = 'SELL' if pos.position > 0 else 'BUY'
    abs_qty = abs(pos.position)
    
    ticker = ib.reqMktData(c, '', snapshot=True)
    ib.sleep(3)
    bid = ticker.bid if ticker.bid and ticker.bid > 0 else 0
    ask = ticker.ask if ticker.ask and ticker.ask > 0 else 0
    
    if action == 'SELL':
        price = max(round(bid - 0.02, 2), 0.01) if bid > 0 else 0.01
    else:
        price = round(ask + 0.05, 2) if ask > 0 else 0.50
    
    print(f"\n  {action} {abs_qty}x {c.strike}{c.right} @ ${price} (bid={bid} ask={ask})")
    
    order = LimitOrder(action, abs_qty, price)
    order.tif = 'DAY'
    trade = ib.placeOrder(c, order)
    
    start = t.time()
    while (t.time() - start) < 30:
        ib.sleep(1)
        s = trade.orderStatus.status
        if s in ('Filled', 'Cancelled', 'Inactive'):
            break
    
    s = trade.orderStatus.status
    if s == 'Filled':
        print(f"  ✅ FILLED @ {trade.orderStatus.avgFillPrice}")
    else:
        print(f"  ❌ {s}")
        for log in trade.log:
            if log.errorCode > 0:
                print(f"    {log.message}")

# FINAL
ib.sleep(3)
final = [p for p in ib.positions() if p.contract.symbol == 'XSP' and p.position != 0]
print(f"\n{'='*50}")
print(f"FINAL: {len(final)} positions")
for p in final:
    print(f"  {p.contract.strike}{p.contract.right} qty={p.position}")

orphan_only = all(abs(p.position) >= 10 for p in final)
if orphan_only:
    print("✅ Only orphan 704C remains. Will expire EOD.")
elif not final:
    print("✅ COMPLETELY FLAT!")
    
ib.disconnect()
print("Done.")
