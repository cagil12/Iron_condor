#!/usr/bin/env python3
"""Check today's trade history from IBKR."""
import sys
sys.path.insert(0, '.')
from src.data.ib_connector import IBConnector
from src.utils.config import get_live_config

config = get_live_config()
conn = IBConnector(config.get('ibkr', {}))
ok, msg = conn.connect(paper=False)

if ok:
    print("HISTORIAL DE EJECUCIONES HOY")
    print("=" * 60)
    
    fills = conn.ib.fills()
    
    total_bought = 0
    total_sold = 0
    total_comm = 0
    
    for f in fills:
        time_str = str(f.execution.time)[:8]
        side = "COMPRA" if f.execution.side == "BOT" else "VENTA"
        symbol = f.contract.localSymbol
        price = f.execution.price
        qty = int(f.execution.shares)
        value = price * 100 * qty
        comm = 0
        if f.commissionReport and f.commissionReport.commission:
            comm = f.commissionReport.commission
        
        print(f"[{time_str}] {side:6} | {symbol[-18:]:18} | {qty} @ ${price:.2f} = ${value:.2f}")
        
        if f.execution.side == "BOT":
            total_bought += value
        else:
            total_sold += value
        total_comm += comm
    
    print("=" * 60)
    print(f"Total Comprado: ${total_bought:.2f}")
    print(f"Total Vendido:  ${total_sold:.2f}")
    print(f"Comisiones:     ${total_comm:.2f}")
    print("-" * 30)
    pnl = total_sold - total_bought - total_comm
    emoji = "🟢" if pnl >= 0 else "🔴"
    print(f"P&L NETO:       {emoji} ${pnl:.2f}")
    print("=" * 60)
    
    conn.disconnect()
