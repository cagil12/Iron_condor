"""Emergency: Cancel ALL open orders and audit remaining positions."""
from ib_insync import *
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def cancel_all_and_audit():
    ib = IB()
    try:
        ib.connect('127.0.0.1', 7497, clientId=102)
        logger.info("🔌 Connected")
    except:
        try:
            ib.connect('127.0.0.1', 7496, clientId=102)
        except:
            logger.error("❌ Cannot connect")
            return

    # 1. Cancel ALL open orders
    open_orders = ib.openOrders()
    trades = ib.openTrades()
    logger.warning(f"🛑 CANCELING {len(trades)} OPEN TRADES...")
    
    for trade in trades:
        try:
            ib.cancelOrder(trade.order)
            logger.info(f"   Canceled: {trade.contract.localSymbol} {trade.order.action} qty={trade.order.totalQuantity}")
        except Exception as e:
            logger.error(f"   Cancel failed: {e}")
    
    ib.sleep(3)
    
    # 2. Audit remaining positions
    logger.info("\n" + "="*60)
    logger.info("📊 CLEAN AUDIT AFTER ORDER CANCELLATION:")
    logger.info("="*60)
    
    positions = ib.positions()
    xsp_pos = [p for p in positions if p.contract.symbol == 'XSP' and p.position != 0]
    
    ic_legs = []
    orphans = []
    
    for p in xsp_pos:
        c = p.contract
        ib.qualifyContracts(c)
        qty = p.position
        strike = c.strike
        right = c.right
        local = c.localSymbol
        pnl = p.unrealizedPNL if hasattr(p, 'unrealizedPNL') else 'N/A'
        
        if abs(qty) == 1:
            ic_legs.append(p)
            tag = "IC LEG"
        else:
            orphans.append(p)
            tag = "ORPHAN"
        
        logger.info(f"   [{tag}] {strike}{right} | Qty: {qty} | ConID: {c.conId} | AvgCost: {p.avgCost}")
    
    logger.info(f"\n   SUMMARY: {len(ic_legs)} IC legs + {len(orphans)} orphans = {len(xsp_pos)} total")
    
    # Check remaining open orders
    remaining = ib.openTrades()
    logger.info(f"   REMAINING ORDERS: {len(remaining)}")
    
    ib.disconnect()
    logger.info("🔌 Disconnected")

if __name__ == "__main__":
    cancel_all_and_audit()
