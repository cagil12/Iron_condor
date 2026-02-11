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
    gross: float,
    protection: float,
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
    line3 = f"💰 Net Credit: ${credit:.2f} (${credit_usd:.0f})"
    line4 = f"   (Gross: ${gross:.2f} | Prot: -${protection:.2f})"
    line5 = f"📊 VIX: {vix:.2f} | Spot: {spot:.2f}"
    line6 = f"⚡ Delta Net: {delta:.4f}"
    
    print("╔════════════════ NEW TRADE DETECTED ════════════════╗")
    print("║ 🟢 OPENING IRON CONDOR (XSP)                       ║")
    print("║ -------------------------------------------------- ║")
    print(f"║ {line1}{pad(line1)}║")
    print(f"║ {line2}{pad(line2)}║")
    print(f"║ {line3}{pad(line3)}║")
    print(f"║ {line4}{pad(line4)}║")
    print(f"║ {line5}{pad(line5)}║")
    print(f"║ {line6}{pad(line6)}║")
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
    # return True # FOR TESTING ONLY


def is_entry_time_reached(entry_time_str: str = "10:00") -> bool:
    """
    Check if we've reached the entry time window.
    
    Args:
        entry_time_str: Entry time in HH:MM format (default 10:00)
    
    Returns:
        True if current time >= entry time
    """
    now = datetime.now()
    try:
        hour, minute = map(int, entry_time_str.split(':'))
        entry_time = dt_time(hour, minute)
        return now.time() >= entry_time
    except:
        return True  # Default to allow if parsing fails


def get_0dte_expiry() -> str:
    """
    Get TODAY's expiration for 0DTE trading.
    
    SPX/XSP have daily expirations (Mon-Fri).
    For 0DTE strategy, expiry is ALWAYS today.
    
    Safety: If called on weekend (should never happen due to market_open check),
    returns next Monday.
    
    Returns:
        Expiry string in YYYYMMDD format
    """
    from datetime import date, timedelta
    today = date.today()
    weekday = today.weekday()
    
    if weekday == 5:  # Saturday -> next Monday
        return (today + timedelta(days=2)).strftime('%Y%m%d')
    elif weekday == 6:  # Sunday -> next Monday
        return (today + timedelta(days=1)).strftime('%Y%m%d')
    else:
        return today.strftime('%Y%m%d')  # Mon-Fri: TODAY


def find_trade_opportunity(
    connector: IBConnector,
    config: dict,
    vix_value: float
) -> Optional[dict]:
    """
    Scan market for Iron Condor opportunity.
    
    Returns trade setup dict if found, None otherwise.
    """
    # Fix 4: Initialize missing variables
    best_put_strike = None
    best_call_strike = None
    best_put_delta = -0.10
    best_call_delta = 0.10
    min_put_diff = 999.0
    min_call_diff = 999.0
    target_delta = config.get('target_delta', 0.10)
    min_credit = config.get('min_credit', 0.20)

    # Get current XSP spot price
    spot = connector.get_live_price('XSP')
    if not spot or spot <= 0:
        print("⏳ Waiting for spot price...")
        return None
    
    # VIX filter
    vix_min = config.get('min_vix', 10.0)
    vix_max = config.get('max_vix', 25.0)
    
    if vix_value < vix_min or vix_value > vix_max:
        print(f"⚠️ VIX ({vix_value:.1f}) outside range [{vix_min}-{vix_max}]")
        return None
    
    # Get expiry first
    expiry = get_0dte_expiry()
    
    # Use Delta-based strike selection (NEW)
    target_delta = config.get('target_delta', 0.10)
    
    # This requires executor - we'll pass it or create temporarily
    # For now, use the simple approach inline
    from ib_insync import Index, Option
    
    try:
        # Smart Delta Selection
        print(f"   🎯 Finding strikes with Delta ~{target_delta}...")
        
        underlying = Index('XSP', 'CBOE')
        connector.ib.qualifyContracts(underlying)
        
        chains = connector.ib.reqSecDefOptParams('XSP', '', 'IND', underlying.conId)
        if chains:
            chain = chains[0]
            all_strikes = sorted(chain.strikes)
            
            # Filter to spot +/- 15
            filtered_strikes = [k for k in all_strikes if (spot - 15) < k < (spot + 15)]
            
            if filtered_strikes:
                # Build and request data
                contracts = []
                for k in filtered_strikes:
                    contracts.append(Option('XSP', expiry, k, 'P', 'SMART'))
                    contracts.append(Option('XSP', expiry, k, 'C', 'SMART'))
                
                connector.ib.qualifyContracts(*contracts)
                
                tickers = [connector.ib.reqMktData(c, '', False, False) for c in contracts]
                connector.ib.sleep(3)
                
                # Find best delta matches (using variables initialized at top of function)
                
                for t in tickers:
                    if t.modelGreeks and t.modelGreeks.delta is not None:
                        delta = t.modelGreeks.delta
                        strike = t.contract.strike
                        
                        if t.contract.right == 'P':
                            diff = abs(delta - (-target_delta))
                            if diff < min_put_diff:
                                min_put_diff = diff
                                best_put_strike = strike
                                best_put_delta = delta
                        else:
                            diff = abs(delta - target_delta)
                            if diff < min_call_diff:
                                min_call_diff = diff
                                best_call_strike = strike
                                best_call_delta = delta
                
                # Cleanup
                for t in tickers:
                    try:
                        connector.ib.cancelMktData(t.contract)
                    except:
                        pass
                
                if best_put_strike and best_call_strike:
                    short_put = best_put_strike
                    short_call = best_call_strike
                    delta_put = best_put_delta
                    delta_call = best_call_delta
                    selection_method = "DELTA_TARGET"
                    print(f"   ✅ Delta selection: {short_put}P (Δ={delta_put:.3f}), {short_call}C (Δ={delta_call:.3f})")
                else:
                    raise ValueError("No valid delta strikes found")
            else:
                raise ValueError("No strikes in range")
        else:
            raise ValueError("No option chain")
            
    except Exception as e:
        print(f"   ⚠️ Delta selection failed ({e}). Using distance fallback.")
        otm_distance = spot * 0.015
        short_put = round(spot - otm_distance)
        short_call = round(spot + otm_distance)
        delta_put = -0.10
        delta_call = 0.10
        selection_method = "OTM_DISTANCE_PCT"
    
    # Get option quotes to estimate NET credit
    # We need to account for the cost of the Long Wings
    WING_WIDTH = config.get('wing_width', 2.0)
    long_put = short_put - WING_WIDTH
    long_call = short_call + WING_WIDTH
    
    # Quotes for Shorts (Bid)
    bid_short_put, _ = connector.get_option_quote('XSP', expiry, short_put, 'P')
    bid_short_call, _ = connector.get_option_quote('XSP', expiry, short_call, 'C')
    
    # Quotes for Longs (Ask) - Cost of protection
    _, ask_long_put = connector.get_option_quote('XSP', expiry, long_put, 'P')
    _, ask_long_call = connector.get_option_quote('XSP', expiry, long_call, 'C')
    
    if None in [bid_short_put, bid_short_call, ask_long_put, ask_long_call]:
        print("⏳ Waiting for option quotes (all legs)...")
        return None
    
    # Calculate Net Credit
    # Conservative: Sell at Bid, Buy at Ask
    gross_credit = bid_short_put + bid_short_call
    protection_cost = ask_long_put + ask_long_call
    net_credit_est = gross_credit - protection_cost
    
    # Minimum credit filter
    min_credit = config.get('min_credit', 0.20)
    if net_credit_est < min_credit:
        print(f"⚠️ Credit too low: ${net_credit_est:.2f} < ${min_credit:.2f}")
        return None
    
    return {
        'spot': spot,
        'vix': vix_value,
        'short_put': short_put,
        'short_call': short_call,
        'expiry': expiry,
        'credit': net_credit_est,
        'gross_credit': gross_credit,       # NEW
        'protection_cost': protection_cost, # NEW
        'delta_net': 0.0,
        'delta_put': delta_put,
        'delta_call': delta_call,
        'selection_method': selection_method,
        'target_delta': target_delta,
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
        
        # FIX 7: TWS Setting Reminder
        print("\n" + "!" * 60)
        print("⚠️  FIX 7: VERIFY TWS SETTINGS!")
        print("   TWS → Global Configuration → API → Settings")
        print("   ☑️ 'Download open orders on connection' MUST BE ENABLED")
        print("!" * 60 + "\n")
        
        capital = connector.get_account_value()
        print(f"💰 Account Value: ${capital:,.2f}")
        
        # Enable real-time VIX fetching now that we have IBKR connection
        vix_loader.set_ib_connector(connector)
        
        # Initialize executor
        executor = LiveExecutor(connector, journal)
        
        # FIX 2: Startup Reconciliation
        executor.startup_reconciliation()
        
        # ATTEMPT RECOVERY OF EXISTING POSITIONS via IBKR
        # (Already handled by load_state in __init__, but we can call it if we want manual sync)
        executor.recover_active_position()
        
        # Main loop
        print("\n" + "═" * 50)
        print("  🔍 SCANNING FOR OPPORTUNITIES")
        print("  Press Ctrl+C to stop")
        print("═" * 50 + "\n")
        
        scan_interval = 60  # seconds
        
        while True:
            try:
                # -----------------------------------------------------------
                # SAFETY: KILL SWITCH L1 (Equity Protection)
                # -----------------------------------------------------------
                if connector.ib.isConnected():
                    try:
                        # Filter by specific account if possible, or grab first NetLiquidation
                        target_acct = config.get('account_id')
                        summary = connector.ib.accountSummary()
                        
                        net_liq_item = next((x for x in summary if x.tag == 'NetLiquidation' and (not target_acct or x.account == target_acct)), None)
                        
                        if net_liq_item:
                            equity = float(net_liq_item.value)
                            min_equity = config.get('min_account_value', 1400.0)
                            
                            if equity < min_equity:
                                print(f"\n💀 KILL SWITCH L1 ACTIVATED: Equity (${equity:,.2f}) < Limit (${min_equity:,.2f}). System SHUTDOWN.")
                                sys.exit(1)
                    except Exception as e_safety:
                         pass # Don't crash on transient data issues

                # -----------------------------------------------------------
                # ACTIVE POSITION CHECK: Monitor existing trades FIRST
                # -----------------------------------------------------------
                if executor.has_active_position():
                    # Robustness Check: If we detect positions on IBKR but don't have internal state
                    if not executor.active_position:
                        print("⚠️ Detected unmanaged positions. Attempting recovery...")
                        executor.recover_active_position()
                        
                        if not executor.active_position:
                            print("❌ Recovery failed or inconsistent state. Pausing 60s to avoid spam...")
                            print("   (Please check TWS manually if this persists)")
                            time.sleep(60)
                            continue
                    
                    print("📊 Active position - monitoring exits...")
                    executor.monitor_position(check_interval=10)
                    continue

                # -----------------------------------------------------------
                # TIME GATE: No trading before start_time (10:00 AM ET)
                # -----------------------------------------------------------
                start_time_str = config.get('start_time', config.get('entry_time', '10:00'))
                try:
                    start_h, start_m = map(int, start_time_str.split(':'))
                    start_time_obj = dt_time(start_h, start_m)
                    current_time = datetime.now().time()
                    
                    if current_time < start_time_obj:
                        now_str = datetime.now().strftime("%H:%M:%S")
                        print(f"[{now_str}] ⏳ MARKET WARMUP: Waiting until {start_time_str} AM... (Current: {now_str[:5]})")
                        time.sleep(60)
                        continue
                        # pass
                except Exception:
                    pass  # Fallback if parsing fails

                # Check market hours
                if not is_market_open():
                    now = datetime.now().strftime("%H:%M:%S")
                    print(f"[{now}] Market closed. Waiting...", end="\r")
                    time.sleep(60)
                    continue
                
                
                # Get VIX
                vix_value = vix_loader.get_vix(datetime.now().date())
                if vix_value is None:
                    now_str = datetime.now().strftime("%H:%M:%S")
                    print(f"[{now_str}] ⚠️ VIX UNAVAILABLE - Cannot trade without VIX visibility. Retrying next cycle.")
                    time.sleep(scan_interval)
                    continue
                
                # VIX SAFETY CHECK (HARD LIMIT)
                max_vix = config.get('max_vix', 25.0)
                if vix_value > max_vix:
                    now_str = datetime.now().strftime("%H:%M:%S")
                    print(f"[{now_str}] 🛑 MARKET DANGER: VIX ({vix_value:.2f}) > Limit ({max_vix}). System HALTED.")
                    time.sleep(60)
                    continue
                
                
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
                        gross=trade_setup.get('gross_credit', 0.0), # NEW
                        protection=trade_setup.get('protection_cost', 0.0), # NEW
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
                             # Position created and logged internally by executor
                             print(f"📋 Position internal log ID: {position.trade_id}")
                            
                             # Monitor until exit
                             executor.monitor_position()
                            
                             # Log close (Handled by executor close_position)

                
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
