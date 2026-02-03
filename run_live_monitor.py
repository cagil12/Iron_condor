#!/usr/bin/env python3
"""
run_live_monitor.py

Main execution loop for live XSP Iron Condor trading via IBKR.
Connects to TWS, monitors for opportunities, and executes trades.

Usage:
    python run_live_monitor.py          # Paper trading
    python run_live_monitor.py --live   # Live trading (USE WITH CAUTION)
"""
import sys
import time
import argparse
from datetime import datetime, time as dt_time
from typing import Optional

# Add project root to path
sys.path.insert(0, '.')

from src.data.ib_connector import IBConnector
from src.data.vix_loader import get_vix_loader
from src.strategy.condor_builder import CondorBuilder
from src.strategy.execution import LiveExecutor, IronCondorPosition
from src.utils.journal import TradeJournal
from src.utils.config import get_live_config


# ══════════════════════════════════════════════════════════════════════════════
# ASCII ART LOGO
# ══════════════════════════════════════════════════════════════════════════════

LOGO = r"""
    ___    _   ____________ _______  ___ _    ______________  __
   /   |  / | / /_  __/  _// ____/ |/ / | |  / /  _/_  __/\ \/ /
  / /| | /  |/ / / /  / / / / __ |   /  | | / // /  / /    \  / 
 / ___ |/ /|  / / / _/ / / /_/ //   |   | |/ // /  / /     / /  
/_/  |_/_/ |_/ /_/ /___/ \____//_/|_|   |___/___/ /_/     /_/   
                                                                 
                    ╔═══════════════════════════╗
                    ║   XSP IRON CONDOR BOT     ║
                    ║   $200 Capital Mode       ║
                    ╚═══════════════════════════╝
"""

TRADE_BANNER = """
╔════════════════ NEW TRADE DETECTED ════════════════╗
║ 🟢 OPENING IRON CONDOR (XSP)                       ║
║ -------------------------------------------------- ║
║ 🎯 Short Strikes: {sp}P / {sc}C{pad1}║
║ 🛡️  Wings: {width} wide{pad2}║
║ 💰 Est. Credit: ${credit:.2f} (${credit_usd:.2f}){pad3}║
║ 📊 VIX: {vix:.2f} | Spot: {spot:.2f}{pad4}║
║ ⚡ Delta Net: {delta:.4f}{pad5}║
╚════════════════════════════════════════════════════╝
"""


def print_trade_banner(
    short_put: float,
    short_call: float,
    width: float,
    credit: float,
    vix: float,
    spot: float,
    delta: float
):
    """Print a formatted trade detection banner."""
    # Calculate padding for alignment
    sp_str = f"{short_put:.0f}"
    sc_str = f"{short_call:.0f}"
    credit_usd = credit * 100
    
    # Pad strings to fill box width
    def pad(content: str, target_len: int = 48) -> str:
        current = len(content)
        return " " * max(0, target_len - current)
    
    line1 = f"🎯 Short Strikes: {sp_str}P / {sc_str}C"
    line2 = f"🛡️  Wings: {width:.1f} wide"
    line3 = f"💰 Est. Credit: ${credit:.2f} (${credit_usd:.2f})"
    line4 = f"📊 VIX: {vix:.2f} | Spot: {spot:.2f}"
    line5 = f"⚡ Delta Net: {delta:.4f}"
    
    print("╔════════════════ NEW TRADE DETECTED ════════════════╗")
    print("║ 🟢 OPENING IRON CONDOR (XSP)                       ║")
    print("║ -------------------------------------------------- ║")
    print(f"║ {line1}{pad(line1)}║")
    print(f"║ {line2}{pad(line2)}║")
    print(f"║ {line3}{pad(line3)}║")
    print(f"║ {line4}{pad(line4)}║")
    print(f"║ {line5}{pad(line5)}║")
    print("╚════════════════════════════════════════════════════╝")


def is_market_open() -> bool:
    """Check if US options market is open (9:30 AM - 4:00 PM ET)."""
    now = datetime.now()
    market_open = dt_time(9, 30)
    market_close = dt_time(16, 0)
    
    # Simple weekday check (0=Monday, 6=Sunday)
    if now.weekday() >= 5:  # Weekend
        return False
    
    current_time = now.time()
    return market_open <= current_time <= market_close


def get_next_friday_expiry() -> str:
    """Get the next Friday expiration in YYYYMMDD format."""
    from datetime import date, timedelta
    today = date.today()
    days_ahead = 4 - today.weekday()  # Friday = 4
    if days_ahead <= 0:
        days_ahead += 7
    next_friday = today + timedelta(days=days_ahead)
    return next_friday.strftime('%Y%m%d')


def find_trade_opportunity(
    connector: IBConnector,
    config: dict,
    vix_value: float
) -> Optional[dict]:
    """
    Scan market for Iron Condor opportunity.
    
    Returns trade setup dict if found, None otherwise.
    """
    # Get current XSP spot price
    spot = connector.get_live_price('XSP')
    if not spot or spot <= 0:
        print("⏳ Waiting for spot price...")
        return None
    
    # VIX filter
    vix_min = config.get('risk_filters', {}).get('vix_threshold_min', 12)
    vix_max = config.get('risk_filters', {}).get('vix_threshold_max', 30)
    
    if vix_value < vix_min or vix_value > vix_max:
        print(f"⚠️ VIX ({vix_value:.1f}) outside range [{vix_min}-{vix_max}]")
        return None
    
    # Calculate strikes based on delta target
    # For XSP ~580-600 range: ~10 delta is roughly 8-10 points OTM
    target_delta = config.get('target_delta', 0.10)
    
    # Simplified strike selection (in production, use Greeks calculation)
    # For 10 delta on XSP, roughly 1.5-2% OTM
    otm_distance = spot * 0.015  # 1.5% OTM
    
    short_put = round(spot - otm_distance)
    short_call = round(spot + otm_distance)
    
    # Get option quotes to estimate credit
    expiry = get_next_friday_expiry()
    
    bid_put, ask_put = connector.get_option_quote('XSP', expiry, short_put, 'P')
    bid_call, ask_call = connector.get_option_quote('XSP', expiry, short_call, 'C')
    
    if not all([bid_put, bid_call]):
        print("⏳ Waiting for option quotes...")
        return None
    
    # Estimate credit (mid of bid)
    put_credit = bid_put
    call_credit = bid_call
    total_credit = put_credit + call_credit
    
    # Minimum credit filter
    min_credit = config.get('min_credit', 0.10)
    if total_credit < min_credit:
        print(f"⚠️ Credit too low: ${total_credit:.2f} < ${min_credit:.2f}")
        return None
    
    return {
        'spot': spot,
        'vix': vix_value,
        'short_put': short_put,
        'short_call': short_call,
        'expiry': expiry,
        'credit': total_credit,
        'delta_net': 0.0,  # Would need proper Greek calculation
    }


def main():
    parser = argparse.ArgumentParser(description='XSP Iron Condor Live Monitor')
    parser.add_argument('--live', action='store_true', 
                        help='Connect to live trading (default: paper)')
    parser.add_argument('--no-trade', action='store_true',
                        help='Monitor only, do not execute trades')
    args = parser.parse_args()
    
    # Print logo
    print(LOGO)
    
    # Load config
    config = get_live_config()
    
    # Initialize components
    connector = IBConnector(config.get('ibkr', {}))
    journal = TradeJournal()
    vix_loader = get_vix_loader()
    
    # Print journal summary
    journal.print_summary()
    
    try:
        # Connect to TWS
        mode = "LIVE" if args.live else "PAPER"
        print(f"\n🔌 Connecting to TWS ({mode})...")
        
        if args.live:
            print("⚠️  LIVE MODE - Real money at risk!")
            time.sleep(3)
        
        success, msg = connector.connect(paper=not args.live)
        if not success:
            print(f"❌ {msg}")
            return
        
        print(f"✅ {msg}")
        capital = connector.get_account_value()
        print(f"💰 Account Value: ${capital:,.2f}")
        
        # Initialize executor
        executor = LiveExecutor(connector)
        
        # Main loop
        print("\n" + "═" * 50)
        print("  🔍 SCANNING FOR OPPORTUNITIES")
        print("  Press Ctrl+C to stop")
        print("═" * 50 + "\n")
        
        scan_interval = 60  # seconds
        
        while True:
            try:
                # Check market hours
                if not is_market_open():
                    now = datetime.now().strftime("%H:%M:%S")
                    print(f"[{now}] Market closed. Waiting...", end="\r")
                    time.sleep(60)
                    continue
                
                # Skip if we have an active position
                if executor.has_active_position():
                    print("📊 Active position - monitoring exits...")
                    executor.monitor_position(check_interval=10)
                    continue
                
                # Get VIX
                vix_value = vix_loader.get_vix(datetime.now().date())
                if vix_value is None:
                    vix_value = 15.0  # Default fallback
                
                # Scan for opportunity
                now = datetime.now().strftime("%H:%M:%S")
                print(f"[{now}] Scanning... (VIX: {vix_value:.1f})")
                
                trade_setup = find_trade_opportunity(connector, config, vix_value)
                
                if trade_setup:
                    # Print banner
                    print_trade_banner(
                        short_put=trade_setup['short_put'],
                        short_call=trade_setup['short_call'],
                        width=executor.WING_WIDTH,
                        credit=trade_setup['credit'],
                        vix=trade_setup['vix'],
                        spot=trade_setup['spot'],
                        delta=trade_setup['delta_net']
                    )
                    
                    if args.no_trade:
                        print("📋 NO-TRADE MODE: Would execute trade here")
                    else:
                        # Execute trade
                        position = executor.execute_iron_condor(
                            short_put=trade_setup['short_put'],
                            short_call=trade_setup['short_call'],
                            expiry=trade_setup['expiry'],
                            spot=trade_setup['spot'],
                            vix=trade_setup['vix'],
                            delta_net=trade_setup['delta_net']
                        )
                        
                        if position:
                            # Log to journal
                            trade_id = journal.log_trade_open(
                                spot_price=position.spot_at_entry,
                                vix_value=position.vix_at_entry,
                                short_put_strike=position.short_put_strike,
                                short_call_strike=position.short_call_strike,
                                wing_width=executor.WING_WIDTH,
                                entry_credit=position.entry_credit,
                                max_profit_usd=position.max_profit,
                                max_loss_usd=position.max_loss,
                                delta_net=position.delta_net,
                                reasoning=f"VIX={trade_setup['vix']:.1f}, Spot={trade_setup['spot']:.2f}"
                            )
                            
                            # Monitor until exit
                            executor.monitor_position()
                            
                            # Log close
                            final_pnl = executor.get_position_pnl() or 0
                            journal.log_trade_close(
                                trade_id=trade_id,
                                exit_reason="Automated",
                                final_pnl_usd=final_pnl,
                                entry_timestamp=position.entry_time
                            )
                
                # Wait before next scan
                time.sleep(scan_interval)
                
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"⚠️ Error in scan loop: {e}")
                time.sleep(10)
        
    except KeyboardInterrupt:
        print("\n\n👋 Stopping monitor...")
    finally:
        connector.disconnect()
        journal.print_summary()
        print("🔌 Disconnected from TWS")


if __name__ == "__main__":
    main()
