"""
execution.py

Live Execution Engine for XSP Iron Condors via IBKR.
Handles order placement, position monitoring, and exit management.

SAFETY: Hardcoded limits for small account ($200 capital).
"""
import time
from datetime import datetime
import json
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from ib_insync import IB, Option, Order, LimitOrder, Trade as IBTrade, Contract, ComboLeg

from src.data.ib_connector import IBConnector
from src.utils.config import get_live_config


@dataclass
class IronCondorPosition:
    """Represents an active Iron Condor position."""
    entry_time: datetime
    short_put_strike: float
    long_put_strike: float
    short_call_strike: float
    long_call_strike: float
    entry_credit: float  # Per contract in dollars
    qty: int
    max_profit: float  # entry_credit * qty * 100
    max_loss: float    # (wing_width - entry_credit) * qty * 100
    spot_at_entry: float
    vix_at_entry: float
    delta_net: float
    snapshot_json: str = "{}"  # NEW
    
    # Greeks Snapshot
    delta_put: float = 0.0
    delta_call: float = 0.0
    theta: float = 0.0
    gamma: float = 0.0
    
    # Order IDs for tracking
    order_ids: List[int] = None
    
    def __post_init__(self):
        if self.order_ids is None:
            self.order_ids = []


class LiveExecutor:
    """
    Live execution engine for XSP Iron Condors.
    
    Safety Features:
    - WING_WIDTH = 1.0 (hardcoded for $200 account)
    - QTY = 1 (no pyramiding)
    - Active position check before entry
    """
    
    # HARDCODED SAFETY LIMITS
    WING_WIDTH = 1.0       # Fixed wing width for small account
    MAX_QTY = 1            # Only 1 contract at a time
    TAKE_PROFIT_PCT = 0.50 # 50% of max profit
    STOP_LOSS_MULT = 2.0   # 2x the credit received
    ORDER_TIMEOUT = 10     # Seconds to wait for fill
    CHASE_TICKS = 3        # Number of times to chase price
    TICK_SIZE = 0.01       # XSP option tick size
    
    def __init__(self, connector: IBConnector):
        self.connector = connector
        self.ib = connector.ib
        self.config = get_live_config()
        self.active_position: Optional[IronCondorPosition] = None
        
    def has_active_position(self) -> bool:
        """Check if there's an active position."""
        # Check our internal tracking
        if self.active_position is not None:
            return True
        
        # Also verify with IBKR
        positions = self.ib.positions()
        xsp_positions = [p for p in positions if p.contract.symbol == 'XSP']
        return len(xsp_positions) > 0
    
    def build_option_contract(self, strike: float, right: str, expiry: str) -> Option:
        """Build an XSP option contract."""
        contract = Option('XSP', expiry, strike, right, 'SMART')
        self.ib.qualifyContracts(contract)
        return contract
    
    def get_mid_price(self, contract: Option) -> Optional[float]:
        """Get mid price for an option contract."""
        ticker = self.ib.reqMktData(contract, '', snapshot=False)
        self.ib.sleep(1)
        
        if ticker.bid and ticker.ask and ticker.bid > 0 and ticker.ask > 0:
            mid = (ticker.bid + ticker.ask) / 2
            self.ib.cancelMktData(contract)
            return round(mid, 2)
        
        self.ib.cancelMktData(contract)
        return None
    
    def find_delta_strikes(
        self, 
        spot: float, 
        expiry: str, 
        target_delta: float = 0.10
    ) -> Dict[str, Any]:
        """
        Find optimal strikes using real-time Delta from IBKR.
        
        Uses Smart Filter: Only requests data for strikes within Spot +/- 20.
        Falls back to distance heuristic if Greeks unavailable.
        
        Returns:
            dict with 'short_put', 'short_call', 'delta_put', 'delta_call', 'method'
        """
        STRIKE_RANGE = 20  # +/- 20 from spot
        TARGET_PUT_DELTA = -target_delta  # e.g., -0.10
        TARGET_CALL_DELTA = target_delta   # e.g., 0.10
        
        print(f"   🎯 Finding strikes with Delta ~{target_delta} (Smart Filter)...")
        
        try:
            # 1. Get option chain metadata (no quota used)
            from ib_insync import Index
            underlying = Index('XSP', 'CBOE')
            self.ib.qualifyContracts(underlying)
            
            chains = self.ib.reqSecDefOptParams('XSP', '', 'IND', underlying.conId)
            if not chains:
                raise ValueError("No option chain found")
                
            chain = chains[0]
            all_strikes = sorted(chain.strikes)
            
            # 2. Smart Filter: Only strikes in range
            lower_bound = spot - STRIKE_RANGE
            upper_bound = spot + STRIKE_RANGE
            filtered_strikes = [k for k in all_strikes if lower_bound < k < upper_bound]
            
            print(f"   📊 Filtered to {len(filtered_strikes)} strikes in [{lower_bound:.0f} - {upper_bound:.0f}]")
            
            if len(filtered_strikes) == 0:
                raise ValueError("No strikes in range")
            
            # 3. Build contracts for Puts and Calls
            put_contracts = []
            call_contracts = []
            
            for k in filtered_strikes:
                put_contracts.append(Option('XSP', expiry, k, 'P', 'SMART'))
                call_contracts.append(Option('XSP', expiry, k, 'C', 'SMART'))
            
            all_contracts = put_contracts + call_contracts
            self.ib.qualifyContracts(*all_contracts)
            
            # 4. Request market data
            print(f"   📡 Requesting Greeks for {len(all_contracts)} contracts...")
            tickers = []
            for c in all_contracts:
                t = self.ib.reqMktData(c, '', False, False)
                tickers.append(t)
            
            self.ib.sleep(3)  # Wait for Greeks
            
            # 5. Find best matches
            best_put = None
            best_call = None
            min_put_diff = 999
            min_call_diff = 999
            
            for t in tickers:
                if not t.modelGreeks or t.modelGreeks.delta is None:
                    continue
                    
                delta = t.modelGreeks.delta
                strike = t.contract.strike
                right = t.contract.right
                
                if right == 'P':
                    diff = abs(delta - TARGET_PUT_DELTA)
                    if diff < min_put_diff:
                        min_put_diff = diff
                        best_put = {'strike': strike, 'delta': delta}
                else:  # Call
                    diff = abs(delta - TARGET_CALL_DELTA)
                    if diff < min_call_diff:
                        min_call_diff = diff
                        best_call = {'strike': strike, 'delta': delta}
            
            # 6. Cleanup subscriptions
            for t in tickers:
                try:
                    self.ib.cancelMktData(t.contract)
                except:
                    pass
            
            # 7. Validate results
            if not best_put or not best_call:
                raise ValueError("Could not find valid Delta strikes")
            
            # Additional Safety: Check for NaN
            if best_put.get('delta') is None or best_call.get('delta') is None:
                raise ValueError("Received NaN Delta values")

            print(f"   ✅ Found: {best_put['strike']}P (Δ={best_put['delta']:.3f}), {best_call['strike']}C (Δ={best_call['delta']:.3f})")
            
            return {
                'short_put': best_put['strike'],
                'short_call': best_call['strike'],
                'delta_put': best_put['delta'],
                'delta_call': best_call['delta'],
                'method': 'DELTA_TARGET'
            }
            
        except Exception as e:
            print(f"   ⚠️ Delta selection failed: {e}") 
            print("   ⚠️ Greeks data unavailable. Reverting to Distance Heuristic (1.5%).")
            
            # FALLBACK: Distance-based selection (1.5% OTM)
            otm_distance = spot * 0.015
            short_put = round(spot - otm_distance)
            short_call = round(spot + otm_distance)
            
            return {
                'short_put': short_put,
                'short_call': short_call,
                'delta_put': -0.10,  # Estimated
                'delta_call': 0.10,  # Estimated
                'method': 'OTM_DISTANCE_PCT'
            }

    
    def build_combo_contract(self, legs: List[Dict[str, Any]]) -> Contract:
        """
        Build a BAG (Combo) contract.
        
        Args:
            legs: List of dicts with 'contract', 'ratio', 'action'
            
        Returns:
            IB Contract object with secType='BAG'
        """
        contract = Contract()
        contract.symbol = 'XSP'
        contract.secType = 'BAG'
        contract.currency = 'USD'
        contract.exchange = 'SMART'
        
        combo_legs = []
        for leg in legs:
            # Must qualify contract to get conId
            c = leg['contract']
            if not c.conId:
                self.ib.qualifyContracts(c)
            
            combo_leg = ComboLeg()
            combo_leg.conId = c.conId
            combo_leg.ratio = leg['ratio']
            combo_leg.action = leg['action']
            combo_leg.exchange = 'SMART'
            combo_legs.append(combo_leg)
            
        contract.comboLegs = combo_legs
        return contract

    def check_margin_impact(self, contract: Contract, order: Order) -> bool:
        """
        Check if order is within margin limits using whatIfOrder.
        
        Returns:
            True if safe to proceed, False if margin too high
        """
        print("   🛡️ Checking margin impact...")
        try:
            state = self.ib.whatIfOrder(contract, order)
            
            # Initial Margin Change
            init_margin = float(state.initMarginChange)
            
            # Max Risk Tolerance ($200 account -> ~$100-150 max margin allowed)
            # Iron Condor 1.0 wide should be ~$100 margin
            MAX_MARGIN = 180.0 
            
            print(f"   💰 Estimated Margin: ${init_margin:.2f}")
            
            if float(init_margin) > MAX_MARGIN:
                print(f"   ⚠️ MARGIN TOO HIGH: ${init_margin:.2f} > ${MAX_MARGIN}")
                return False
                
            return True
        except Exception as e:
            print(f"   ⚠️ Could not verify margin: {e}")
            return False  # Fail safe

    def execute_iron_condor(
        self,
        short_put: float,
        short_call: float,
        expiry: str,
        spot: float,
        vix: float,
        delta_net: float
    ) -> Optional[IronCondorPosition]:
        """
        Execute an Iron Condor order using IBKR BAG (Combo).
        """
        # SAFETY CHECK: No pyramiding
        if self.has_active_position():
            print("⚠️ ABORT: Active position exists. No pyramiding allowed.")
            return None
        
        # Calculate wing strikes
        long_put = short_put - self.WING_WIDTH
        long_call = short_call + self.WING_WIDTH
        
        # Build individual contracts
        c_short_put = self.build_option_contract(short_put, 'P', expiry)
        c_long_put = self.build_option_contract(long_put, 'P', expiry)
        c_short_call = self.build_option_contract(short_call, 'C', expiry)
        c_long_call = self.build_option_contract(long_call, 'C', expiry)
        
        # Get prices for credit estimation and snapshot
        contracts_map = {
            'short_put': c_short_put, 'long_put': c_long_put,
            'short_call': c_short_call, 'long_call': c_long_call
        }
        
        prices = {}
        snapshot_data = {}
        print("⏳ Getting leg prices (Bid/Ask) for snapshot...")
        
        for name, contract in contracts_map.items():
            # Request live ticker
            ticker = self.ib.reqMktData(contract, '', False, False)
            self.ib.sleep(1.0) # Wait for data
            
            if ticker.bid and ticker.ask and ticker.bid > 0:
                mid = (ticker.bid + ticker.ask) / 2
                prices[name] = mid
                snapshot_data[f"{name}_bid"] = ticker.bid
                snapshot_data[f"{name}_ask"] = ticker.ask
                snapshot_data[f"{name}_mid"] = mid
            else:
                print(f"❌ Could not get Price/Bid/Ask for {name} (Bid={ticker.bid}, Ask={ticker.ask})")
                self.ib.cancelMktData(contract)
                return None
                
            self.ib.cancelMktData(contract)
            
            # Capture Greeks if available
            greeks = ticker.modelGreeks
            if greeks:
                if name == 'short_put':
                    snapshot_data['delta_put'] = greeks.delta
                    snapshot_data['theta_put'] = greeks.theta
                    snapshot_data['gamma_put'] = greeks.gamma
                elif name == 'short_call':
                    snapshot_data['delta_call'] = greeks.delta
                    snapshot_data['theta_call'] = greeks.theta
                    snapshot_data['gamma_call'] = greeks.gamma

            
        # Calculate Net Credit (using Mids for Limit Price estimation)
        # Note: We use Mid for execution limit, but logic checks against conservative if needed
        credit_put_spread = prices['short_put'] - prices['long_put']
        credit_call_spread = prices['short_call'] - prices['long_call']
        total_credit = credit_put_spread + credit_call_spread
        
        snapshot_json_str = json.dumps(snapshot_data)
        
        if total_credit <= 0:
            print(f"❌ Invalid credit: ${total_credit:.2f}")
            return None
            
        print(f"📊 Calculated Mid Credit: ${total_credit:.2f} per contract")
        
        # 1. Prepare Combo Legs
        # Convention: We BUY the Strategy. Value = Net Credit (Negative Price)
        # To match the strategy payoff:
        # - Short Put: SELL
        # - Long Put: BUY 
        # - Short Call: SELL
        # - Long Call: BUY
        
        legs = [
            {'contract': c_short_put, 'ratio': 1, 'action': 'SELL'},
            {'contract': c_long_put,  'ratio': 1, 'action': 'BUY'},
            {'contract': c_short_call,'ratio': 1, 'action': 'SELL'},
            {'contract': c_long_call, 'ratio': 1, 'action': 'BUY'}
        ]
        
        bag_contract = self.build_combo_contract(legs)
        
        # 2. Create Limit Order
        # IBKR Convention: For Credit Strategies (like Iron Condor), using BUY with Negative Limit Price 
        # is the robust standard to ensure execution as a package (Debit of negative amount = Credit).
        # We stick to this to avoid ambiguity.
        limit_price = -round(total_credit, 2)
        
        print(f"📦 Combo Constructed: BUY 1 BAG @ ${limit_price} (Credit)")
        order = LimitOrder('BUY', self.MAX_QTY, limit_price)
        order.tif = 'DAY'
        
        # 3. Margin Guard (What-If)
        print("🛡️ Checking Margin Impact...")
        try:
            state = self.ib.whatIfOrder(bag_contract, order)
            init_margin = float(state.initMarginChange)
            
            print(f"   💵 Init Margin Change: ${init_margin:.2f}")
            
            # Max accepted margin: ~$100 (risk) + buffer = $250
            MAX_MARGIN_ACCEPTED = 250.0
            
            if init_margin > MAX_MARGIN_ACCEPTED:
                print(f"⚠️ MARGIN REJECT: Expected < ${MAX_MARGIN_ACCEPTED}, Got ${init_margin:.2f}")
                return None
                
            print("   ✅ Margin Check Passed")
            
        except Exception as e:
            print(f"⚠️ Margin Check Failed: {e}")
            return None # Conservative: Don't trade if check fails
            
        # 4. Execute
        try:
            print(f"🚀 Sending Combo Order...")
            trade = self.ib.placeOrder(bag_contract, order)
            # WAIT FOR FILL (Fixes duplicate logging bug)
            print(f"⏳ Waiting for fill (max 45s)...")
            start_wait = time.time()
            filled = False
            
            while (time.time() - start_wait) < 45:
                self.ib.sleep(1)
                status = trade.orderStatus.status
                if status == 'Filled':
                    filled = True
                    break
                if status in ['Cancelled', 'Inactive', 'ApiCancelled']:
                    print(f"❌ Order Cancelled/Inactive: {status}")
                    return None
            
            if not filled:
                print(f"⚠️ Order timeout (Status: {trade.orderStatus.status}). Cancelling...")
                self.ib.cancelOrder(order)
                return None
            
            # Success - Record Position
            print(f"📋 Order Filled! Exec Price: {trade.orderStatus.avgFillPrice}")
            
            max_profit = total_credit * self.MAX_QTY * 100
            max_loss = (self.WING_WIDTH - total_credit) * self.MAX_QTY * 100
            
            # Extract Greeks from snapshot (safely)
            delta_put = snapshot_data.get('delta_put', 0.0) or 0.0
            delta_call = snapshot_data.get('delta_call', 0.0) or 0.0
            
            # Sum up theta/gamma from short legs (dominant)
            theta_total = (snapshot_data.get('theta_put', 0.0) or 0.0) + (snapshot_data.get('theta_call', 0.0) or 0.0)
            gamma_total = (snapshot_data.get('gamma_put', 0.0) or 0.0) + (snapshot_data.get('gamma_call', 0.0) or 0.0)
            
            self.active_position = IronCondorPosition(
                entry_time=datetime.now(),
                short_put_strike=short_put,
                long_put_strike=long_put,
                short_call_strike=short_call,
                long_call_strike=long_call,
                entry_credit=total_credit,
                qty=self.MAX_QTY,
                max_profit=max_profit,
                max_loss=max_loss,
                spot_at_entry=spot,
                vix_at_entry=vix,
                delta_net=delta_net,
                snapshot_json=snapshot_json_str,
                order_ids=[trade.order.orderId],
                # New Greeks
                delta_put=delta_put,
                delta_call=delta_call,
                theta=theta_total,
                gamma=gamma_total
            )
            print("✅ Iron Condor Opened (Combo) & Filled")
            return self.active_position

                
        except Exception as e:
            print(f"❌ Execution Error: {e}")
            return None
    
    def get_position_pnl(self) -> Optional[float]:
        """Get current PnL for active position."""
        if not self.active_position:
            return None
        
        # Get current portfolio PnL from IBKR
        portfolio = self.ib.portfolio()
        xsp_pnl = sum(
            item.unrealizedPNL 
            for item in portfolio 
            if item.contract.symbol == 'XSP'
        )
        
        return xsp_pnl
    
    def check_exit_conditions(self) -> Optional[str]:
        """
        Check if exit conditions are met.
        
        Returns:
            Exit reason or None if no exit needed
        """
        if not self.active_position:
            return None
        
        pnl = self.get_position_pnl()
        if pnl is None:
            return None
        
        max_profit = self.active_position.max_profit
        take_profit_target = max_profit * self.TAKE_PROFIT_PCT
        stop_loss_target = -self.active_position.entry_credit * 100 * self.STOP_LOSS_MULT
        
        if pnl >= take_profit_target:
            return f"TAKE_PROFIT (PnL: ${pnl:.2f} >= ${take_profit_target:.2f})"
        
        if pnl <= stop_loss_target:
            return f"STOP_LOSS (PnL: ${pnl:.2f} <= ${stop_loss_target:.2f})"
        
        return None
    
    def close_position(self, reason: str) -> float:
        """
        Close the active position.
        
        Args:
            reason: Reason for closing
            
        Returns:
            Final PnL
        """
        if not self.active_position:
            return 0.0
        
        print(f"\n🔴 CLOSING POSITION: {reason}")
        
        # Get final PnL before closing
        final_pnl = self.get_position_pnl() or 0.0
        
        # Close all XSP positions
        positions = self.ib.positions()
        for pos in positions:
            if pos.contract.symbol == 'XSP' and pos.position != 0:
                action = 'SELL' if pos.position > 0 else 'BUY'
                qty = abs(pos.position)
                
                # Market order to close
                order = LimitOrder(action, qty, 0)  # Will need proper pricing
                order.orderType = 'MKT'  # Override to market
                self.ib.placeOrder(pos.contract, order)
        
        self.ib.sleep(5)  # Wait for fills
        
        print(f"✅ Position closed. Final PnL: ${final_pnl:.2f}")
        
        # Clear active position
        closed_position = self.active_position
        self.active_position = None
        
        return final_pnl
    
    def monitor_position(self, check_interval: float = 10.0):
        """
        Monitor active position for exit conditions.
        
        Args:
            check_interval: Seconds between checks
        """
        if not self.active_position:
            print("No active position to monitor.")
            return
        
        print(f"\n📡 Monitoring position (checking every {check_interval}s)...")
        
        while self.active_position:
            pnl = self.get_position_pnl()
            if pnl is not None:
                max_profit = self.active_position.max_profit
                pnl_pct = (pnl / max_profit) * 100 if max_profit > 0 else 0
                print(f"   PnL: ${pnl:.2f} ({pnl_pct:.1f}% of max)", end="\r")
            
            exit_reason = self.check_exit_conditions()
            if exit_reason:
                self.close_position(exit_reason)
                break
            
            self.ib.sleep(check_interval)
