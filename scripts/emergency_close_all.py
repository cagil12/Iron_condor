from ib_insync import *
import logging
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def close_all_positions():
    """
    EMERGENCY SCRIPT: Close ALL XSP positions at MARKET price.
    Use with caution.
    """
    try:
        ib = IB()
        # Connect to same port as monitor (try live 7496 first, then paper 7497)
        try:
            ib.connect('127.0.0.1', 7496, clientId=101) # Client 101 for emergency
            logger.info("🔌 Connected to LIVE TWS (7496)")
        except:
            try:
                ib.connect('127.0.0.1', 7497, clientId=101)
                logger.info("🔌 Connected to PAPER TWS (7497)")
            except:
                logger.error("❌ Cannot connect to TWS. Is it open?")
                return

        # Get positions
        positions = ib.positions()
        xsp_positions = [p for p in positions if p.contract.symbol == 'XSP']

        if not xsp_positions:
            logger.info("✅ No XSP positions found. Portfolio is clean.")
            return

        logger.warning(f"⚠️ FOUND {len(xsp_positions)} XSP POSITIONS TO CLOSE!")
        
        # Sort so we close SHORT positions (negative qty) first to reduce margin
        # Short positions have negative size -> We need to BUY (+1)
        # Long positions have positive size -> We need to SELL (-1)
        xsp_positions.sort(key=lambda p: p.position) 
        
        for p in xsp_positions:
            contract = p.contract
            ib.qualifyContracts(contract)
            
            # Opposing action
            is_short = p.position < 0
            action = 'BUY' if is_short else 'SELL'
            qty = abs(p.position)
            
            logger.info(f"   🚨 CLOSING {contract.localSymbol}: {action} {qty} @ MKT ({'Short' if is_short else 'Long'})")
            
            order = MarketOrder(action, qty)
            trade = ib.placeOrder(contract, order)
            ib.sleep(2) # Increased pause for margin recalc

            
        logger.info("⏳ Waiting for fills...")
        ib.sleep(5)
        
        # Verification
        positions_after = ib.positions()
        xsp_left = [p for p in positions_after if p.contract.symbol == 'XSP' and p.position != 0]
        
        if not xsp_left:
            logger.info("✅ SUCCESS: All XSP positions flattened.")
        else:
            logger.warning(f"⚠️ WARNING: {len(xsp_left)} positions still remain. Run again!")

        ib.disconnect()

    except Exception as e:
        logger.error(f"❌ Error: {e}")

if __name__ == "__main__":
    close_all_positions()
