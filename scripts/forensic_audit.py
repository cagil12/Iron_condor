"""Forensic audit: Get all XSP executions and commissions for today."""
from ib_insync import *
from datetime import datetime

ib = IB()
ib.connect('127.0.0.1', 7497, clientId=250)
ib.sleep(2)

# Get today's executions
executions = ib.reqExecutions(ExecutionFilter(symbol='XSP'))
today = datetime.now().date()

print("=" * 90)
print("FORENSIC EXECUTION TRACE — 2026-02-13")
print("=" * 90)

# Group by orderId
from collections import defaultdict
by_order = defaultdict(list)

for e in sorted(executions, key=lambda x: x.execution.time):
    ex = e.execution
    c = e.contract
    t = ex.time
    time_str = t.strftime("%H:%M:%S")
    print(f"  {time_str} | {ex.side:4} {ex.shares:5.0f}x {c.strike:.0f}{c.right} @ ${ex.price:.4f} | OrderId: {ex.orderId} | Acct: {ex.acctNumber}")
    by_order[ex.orderId].append(e)

# Commission summary
print()
print("=" * 90)
print("COMMISSION & REALIZED PnL SUMMARY")
print("=" * 90)

fills = ib.fills()
total_commission = 0
total_realized_pnl = 0
for f in fills:
    cr = f.commissionReport
    if cr.commission > 0:
        total_commission += cr.commission
        total_realized_pnl += cr.realizedPNL
        c = f.contract
        print(f"  {c.strike:.0f}{c.right} | Commission: ${cr.commission:.2f} | Realized PnL: ${cr.realizedPNL:.2f}")

print(f"\n  TOTAL Commission: ${total_commission:.2f}")
print(f"  TOTAL Realized PnL: ${total_realized_pnl:.2f}")
print(f"  NET IMPACT: ${total_realized_pnl - total_commission:.2f}")

# Group analysis
print()
print("=" * 90)
print("ORDER GROUP ANALYSIS")
print("=" * 90)
for oid, execs in sorted(by_order.items()):
    legs = []
    for e in execs:
        ex = e.execution
        c = e.contract
        legs.append(f"{ex.side} {ex.shares:.0f}x {c.strike:.0f}{c.right} @ ${ex.price:.4f}")
    time_str = execs[0].execution.time.strftime("%H:%M:%S")
    print(f"\n  OrderId {oid} ({time_str}):")
    for l in legs:
        print(f"    {l}")

ib.disconnect()
print("\nDone.")
