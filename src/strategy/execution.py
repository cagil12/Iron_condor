"""
execution.py

Live Execution Engine for XSP Iron Condors via IBKR.
Handles order placement, position monitoring, and exit management.

SAFETY: Hardcoded limits for small account ($200 capital).
"""
import time as time_module
from datetime import datetime, time, timedelta
import json
from typing import Optional, Dict, Any, List
import argparse
import sys
import logging
from pathlib import Path
import numpy as np # NEW: For RV calculation
from dataclasses import dataclass

from ib_insync import IB, Option, Order, LimitOrder, MarketOrder, Trade as IBTrade, Contract, ComboLeg

from src.data.ib_connector import IBConnector
from src.utils.config import get_live_config
from src.utils.journal import TradeJournal


@dataclass
class IronCondorPosition:
    """Represents an active Iron Condor position."""
    trade_id: int # NEW
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
    legs: List[Dict[str, Any]] = None  # NEW: Store conId and action for BAG closure
    
    def __post_init__(self):
        if self.order_ids is None:
            self.order_ids = []
        if self.legs is None:
            self.legs = []


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
    # STOP_LOSS_MULT removed from hardcoded constants - loaded from config
    ORDER_TIMEOUT = 10     # Seconds to wait for fill
    CHASE_TICKS = 3        # Number of times to chase price
    TICK_SIZE = 0.01       # XSP option tick size
    FORCE_CLOSE_TIME = time(15, 45)  # Hard EOD exit at 3:45 PM
    STATE_FILE = Path("state.json")
    MAX_MARGIN_ACCEPTED = 250.0     # Maximum initial margin for 1 contract
    
    def __init__(self, connector: IBConnector, journal: Optional[TradeJournal] = None):
        self.connector = connector
        self.ib = connector.ib
        self.logger = logging.getLogger(__name__)
        # If logger has no handlers, add a simple stream handler
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            sh = logging.StreamHandler()
            sh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(sh)
            
        self.config = get_live_config()
        self.active_position: Optional[IronCondorPosition] = None
        self.journal = journal
        
        # Load dynamic settings
        self.STOP_LOSS_MULT = self.config.get('stop_loss_mult', 3.0)
        self.logger.info(f"🔧 Executor Config: SL={self.STOP_LOSS_MULT}x")
        
        # FIX 3: Load state at startup
        self.load_state()

    def save_state(self):
        """Persistir estado a disco para sobrevivir reinicios (FIX 3)"""
        try:
            state = {
                'active_position': self.active_position.__dict__ if self.active_position else None,
                'last_updated': datetime.now().isoformat()
            }
            # Handle non-serializable objects in __dict__ if any (delta/datetime handled by default str logic)
            self.STATE_FILE.write_text(json.dumps(state, indent=2, default=str))
            self.logger.debug("State saved to disk")
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")

    def load_state(self):
        """Cargar estado previo al iniciar (FIX 3)"""
        if self.STATE_FILE.exists():
            try:
                state = json.loads(self.STATE_FILE.read_text())
                pos_data = state.get('active_position')
                if pos_data:
                    # Convert strings back to datetime objects
                    pos_data['entry_time'] = datetime.fromisoformat(pos_data['entry_time'])
                    self.active_position = IronCondorPosition(**pos_data)
                    self.logger.info(f"Loaded active position from {state.get('last_updated')}")
                else:
                    self.active_position = None
            except Exception as e:
                self.logger.error(f"Failed to load state: {e}")
                self.active_position = None

    def startup_reconciliation(self):
        """Cancelar órdenes huérfanas y reconciliar posiciones al iniciar (FIX 2)"""
        self.logger.info("REMINDER: Verify TWS setting 'Download open orders on connection' is enabled")
        self.logger.info("TWS → Global Configuration → API → Settings → ☑️ Download open orders")
        
        # 1. Ver qué hay abierto
        open_orders = self.ib.reqAllOpenOrders()
        if open_orders:
            self.logger.warning(f"Found {len(open_orders)} open orders at startup")
            for trade in open_orders:
                self.logger.warning(
                    f"  {trade.contract.symbol} {trade.order.action} "
                    f"qty={trade.order.totalQuantity} "
                    f"status={trade.orderStatus.status}"
                )
        
        # 2. Cancelar TODAS las órdenes abiertas (API + TWS + otros clients)
        self.ib.reqGlobalCancel()
        self.ib.sleep(2)
        
        # 3. Reconciliar posiciones reales contra state
        positions = self.ib.positions()
        # Log posiciones encontradas para reconciliación manual si es necesario
        for pos in positions:
            if pos.contract.symbol == 'XSP' and pos.contract.secType == 'OPT':
                self.logger.info(
                    f"  Position: {pos.contract.symbol} {pos.contract.strike} "
                    f"{pos.contract.right} qty={pos.position}"
                )
        
    def recover_active_position(self):
        """
        Attempt to recover active position state from IBKR.
        Ensures 'legs' are populated for Atomic Closure (FIX 4).
        """
        self.logger.info("🔍 Checking for existing positions to resume...")
        positions = self.ib.positions()
        xsp_positions = [p for p in positions if p.contract.symbol == 'XSP' and p.position != 0]
        
        if not xsp_positions:
            self.logger.info("   ✅ No existing positions found. Ready for new entries.")
            return

        self.logger.warning(f"   ⚠️ Found {len(xsp_positions)} existing XSP legs. Attempting recovery...")
        
        try:
            puts = []
            calls = []
            total_credit_collected = 0.0
            qty = 0
            
            for p in xsp_positions:
                c = p.contract
                if c.right == 'P':
                    puts.append(p)
                elif c.right == 'C':
                    calls.append(p)
                
                # Credit/Debit calculation
                if p.position < 0:
                    total_credit_collected += p.avgCost
                else:
                    total_credit_collected -= p.avgCost
                qty = abs(int(p.position))

            puts.sort(key=lambda p: p.contract.strike)
            calls.sort(key=lambda p: p.contract.strike)
            
            if len(puts) != 2 or len(calls) != 2:
                self.logger.error(f"   ❌ Complex structure (P:{len(puts)}, C:{len(calls)}). Manual required.")
                return

            long_put = puts[0]
            short_put = puts[1]
            short_call = calls[0]
            long_call = calls[1]
            
            if not (long_put.position > 0 and short_put.position < 0 and 
                    short_call.position < 0 and long_call.position > 0):
                 self.logger.error("   ❌ Structure mismatch (Long/Short logic error).")
                 return
                 
            entry_credit = total_credit_collected / qty / 100
            max_profit = total_credit_collected
            wing_width = short_put.contract.strike - long_put.contract.strike
            max_loss = (wing_width * qty * 100) - max_profit
            
            # Populate legs for Atomic Closure (FIX 4 compatibility)
            recovered_legs = [
                {'conId': long_put.contract.conId, 'action': 'BUY', 'strike': long_put.contract.strike, 'right': 'P'},
                {'conId': short_put.contract.conId, 'action': 'SELL', 'strike': short_put.contract.strike, 'right': 'P'},
                {'conId': short_call.contract.conId, 'action': 'SELL', 'strike': short_call.contract.strike, 'right': 'C'},
                {'conId': long_call.contract.conId, 'action': 'BUY', 'strike': long_call.contract.strike, 'right': 'C'},
            ]
            
            # Entry time recovery
            entry_time = None
            try:
                from ib_insync import ExecutionFilter
                exec_filter = ExecutionFilter(symbol='XSP')
                executions = self.ib.reqExecutions(exec_filter)
                today = datetime.now().date()
                xsp_execs_today = [e for e in executions if e.execution.time.date() == today]
                if xsp_execs_today:
                    earliest = min(xsp_execs_today, key=lambda e: e.execution.time)
                    entry_time = earliest.execution.time.astimezone().replace(tzinfo=None)
                else:
                    entry_time = datetime.now().replace(hour=10, minute=0, second=0, microsecond=0)
            except Exception:
                entry_time = datetime.now().replace(hour=10, minute=0, second=0, microsecond=0)
            
            self.active_position = IronCondorPosition(
                trade_id=0,
                entry_time=entry_time,
                short_put_strike=short_put.contract.strike,
                long_put_strike=long_put.contract.strike,
                short_call_strike=short_call.contract.strike,
                long_call_strike=long_call.contract.strike,
                entry_credit=entry_credit,
                qty=qty,
                max_profit=max_profit,
                max_loss=max_loss,
                spot_at_entry=0.0,
                vix_at_entry=0.0,
                delta_net=0.0,
                snapshot_json="{}",
                legs=recovered_legs
            )
            
            hold_min = (datetime.now() - entry_time).total_seconds() / 60
            self.logger.info(f"   ✅ Position Recovered! Strikes: {short_put.contract.strike}P/{short_call.contract.strike}C | Hold: {hold_min:.0f}m")
            
        except Exception as e:
            self.logger.error(f"   ❌ Recovery Failed: {e}")

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
            
            if isinstance(state, list):
                if not state:
                    self.logger.warning("⚠️ Margin Check returned empty list. Data unavailable.")
                    return False
                state = state[0]
            
            # Initial Margin Change
            init_margin = float(state.initMarginChange)
            
            self.logger.info(f"   💰 Estimated Margin: ${init_margin:.2f}")
            
            if init_margin > self.MAX_MARGIN_ACCEPTED:
                self.logger.warning(f"   ⚠️ MARGIN TOO HIGH: ${init_margin:.2f} > ${self.MAX_MARGIN_ACCEPTED}")
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
        Includes Chase Logic (FIX 1, 6b) and Stores Leg Info (FIX 4).
        """
        # SAFETY CHECK: No pyramiding
        if self.has_active_position():
            self.logger.warning("⚠️ ABORT: Active position exists. No pyramiding allowed.")
            return None
        
        # Calculate wing strikes
        long_put_strike = short_put - self.WING_WIDTH
        long_call_strike = short_call + self.WING_WIDTH
        
        # Build individual contracts
        # FIX 8: Qualify individual legs FIRST
        legs_contracts = [
            Option('XSP', expiry, long_put_strike, 'P', 'SMART'),
            Option('XSP', expiry, short_put, 'P', 'SMART'),
            Option('XSP', expiry, short_call, 'C', 'SMART'),
            Option('XSP', expiry, long_call_strike, 'C', 'SMART')
        ]
        qualified_legs = self.ib.qualifyContracts(*legs_contracts)
        
        c_long_put = qualified_legs[0]
        c_short_put = qualified_legs[1]
        c_short_call = qualified_legs[2]
        c_long_call = qualified_legs[3]
        
        # Get prices for credit estimation and snapshot
        contracts_map = {
            'short_put': c_short_put, 'long_put': c_long_put,
            'short_call': c_short_call, 'long_call': c_long_call
        }
        
        prices = {}
        snapshot_data = {}
        self.logger.info("⏳ Getting leg prices (Bid/Ask) for snapshot...")
        
        for name, contract in contracts_map.items():
            ticker = self.ib.reqMktData(contract, '', False, False)
            self.ib.sleep(1.0)
            
            if ticker.bid and ticker.ask and ticker.bid > 0:
                mid = (ticker.bid + ticker.ask) / 2
                prices[name] = mid
                snapshot_data[f"{name}_bid"] = ticker.bid
                snapshot_data[f"{name}_ask"] = ticker.ask
                snapshot_data[f"{name}_mid"] = mid
            else:
                self.logger.error(f"❌ Could not get Price/Bid/Ask for {name} (Bid={ticker.bid}, Ask={ticker.ask})")
                self.ib.cancelMktData(contract)
                return None
                
            self.ib.cancelMktData(contract)
            
            greeks = ticker.modelGreeks
            if greeks:
                if name == 'short_put':
                    snapshot_data['delta_put'] = greeks.delta
                    snapshot_data['theta_put'] = greeks.theta
                    snapshot_data['gamma_put'] = greeks.gamma
                    snapshot_data['iv_put'] = greeks.impliedVol
                elif name == 'short_call':
                    snapshot_data['delta_call'] = greeks.delta
                    snapshot_data['theta_call'] = greeks.theta
                    snapshot_data['gamma_call'] = greeks.gamma
                    snapshot_data['iv_call'] = greeks.impliedVol

        # Calculate Net Credit
        credit_put_spread = prices['short_put'] - prices['long_put']
        credit_call_spread = prices['short_call'] - prices['long_call']
        total_credit = credit_put_spread + credit_call_spread
        
        if total_credit <= 0:
            self.logger.error(f"❌ Invalid credit: ${total_credit:.2f}")
            return None
            
        self.logger.info(f"📊 Calculated Mid Credit: ${total_credit:.2f} per contract")
        
        # 1. Prepare BAG (Combo) Contract
        bag = Contract()
        bag.symbol = 'XSP'
        bag.secType = 'BAG'
        bag.currency = 'USD'
        bag.exchange = 'SMART'
        bag.comboLegs = []
        
        # Actions: Long Put (BUY), Short Put (SELL), Short Call (SELL), Long Call (BUY)
        legs_setup = [
            (c_long_put, 'BUY'), (c_short_put, 'SELL'),
            (c_short_call, 'SELL'), (c_long_call, 'BUY')
        ]
        
        for contract, action in legs_setup:
            leg = ComboLeg()
            leg.conId = contract.conId
            leg.ratio = 1
            leg.action = action
            leg.exchange = 'SMART'
            bag.comboLegs.append(leg)
            
        # 2. Margin Guard (What-If)
        limit_price = -round(total_credit, 2)
        test_order = LimitOrder('BUY', self.MAX_QTY, limit_price)
        
        try:
            state = self.ib.whatIfOrder(bag, test_order)
            if isinstance(state, list):
                if not state:
                    self.logger.error("⚠️ Margin Check returned empty list. Data unavailable.")
                    return None
                state = state[0]
                
            init_margin = float(state.initMarginChange)
            if init_margin > self.MAX_MARGIN_ACCEPTED:
                self.logger.warning(f"   ⚠️ MARGIN REJECT: Expected < ${self.MAX_MARGIN_ACCEPTED}, Got ${init_margin:.2f}")
                return None
            self.logger.info(f"   ✅ Margin Check Passed: ${init_margin:.2f}")
            
        except Exception as e:
            self.logger.error(f"⚠️ Margin Check Failed: {e}")
            return None
            
        # 3. CHASE LOOP (FIX 1, FIX 6b)
        filled = False   # FIX 1: Initialized
        trade = None
        
        for chase in range(self.CHASE_TICKS + 1):
            if chase > 0:
                # FIX 6b: Cancel previous to avoid Error 105
                self.logger.info(f"   🔄 Chase #{chase}: Cancelling previous order...")
                if trade:
                    self.ib.cancelOrder(trade.order)
                    self.ib.sleep(1)
                
                # Adjust Price: Less negative = easier fill
                limit_price += self.TICK_SIZE
                if limit_price > 0: limit_price = 0.0
                
                # FIX 6b: Create NEW object (never modify existing lmtPrice)
                order = LimitOrder('BUY', self.MAX_QTY, round(limit_price, 2))
                order.tif = 'DAY'
            else:
                # First attempt
                order = test_order
                order.tif = 'DAY'
            
            self.logger.info(f"   📡 Placing order at ${limit_price:.2f} (Chase {chase})")
            trade = self.ib.placeOrder(bag, order)
            
            # Wait for fill with timeout
            start_wait = time_module.time()
            while (time_module.time() - start_wait) < 15:
                self.ib.sleep(1)
                if trade.orderStatus.status == 'Filled':
                    filled = True
                    break
                if trade.orderStatus.status in ['Cancelled', 'Inactive', 'ApiCancelled']:
                    break
            
            if filled:
                break
        
        if not filled:
            self.logger.warning("❌ Timeout after chase. Cancelling.")
            if trade:
                self.ib.cancelOrder(trade.order)
                self.ib.sleep(1)
            return None
            
        # Success - Record Position
        credit_received = abs(trade.order.lmtPrice)
        max_profit = credit_received * self.MAX_QTY * 100
        max_loss = (self.WING_WIDTH - credit_received) * self.MAX_QTY * 100
        
        delta_put = snapshot_data.get('delta_put', 0.0) or 0.0
        delta_call = snapshot_data.get('delta_call', 0.0) or 0.0
        theta_total = (snapshot_data.get('theta_put', 0.0) or 0.0) + (snapshot_data.get('theta_call', 0.0) or 0.0)
        gamma_total = (snapshot_data.get('gamma_put', 0.0) or 0.0) + (snapshot_data.get('gamma_call', 0.0) or 0.0)
        
        method = "DELTA_TARGET" if delta_net != 0.0 else "OTM_DISTANCE_PCT"
        iv_est = snapshot_data.get('iv_put') or snapshot_data.get('iv_call') or (vix / 100.0)
        
        # Log to Journal
        trade_id = 0
        if self.journal:
            trade_id = self.journal.log_trade_open(
                spot_price=spot, vix_value=vix,
                short_put_strike=short_put, short_call_strike=short_call,
                wing_width=self.WING_WIDTH, entry_credit=credit_received,
                initial_credit=credit_received, iv_entry_atm=iv_est,
                max_profit_usd=max_profit, max_loss_usd=max_loss,
                delta_net=delta_put + delta_call, delta_put=delta_put, delta_call=delta_call,
                theta=theta_total, gamma=gamma_total, selection_method=method,
                target_delta=0.10, otm_distance_pct="1.5%" if method == "OTM_DISTANCE_PCT" else "N/A",
                snapshot_json=json.dumps(snapshot_data), 
                reasoning=f"Method: {method} | VIX: {vix:.1f} | Delta: P{delta_put:.2f}/C{delta_call:.2f} | Credit: ${credit_received:.2f}"
            )
        
        self.active_position = IronCondorPosition(
            trade_id=trade_id, entry_time=datetime.now(),
            short_put_strike=short_put, long_put_strike=long_put_strike,
            short_call_strike=short_call, long_call_strike=long_call_strike,
            entry_credit=credit_received, qty=self.MAX_QTY,
            max_profit=max_profit, max_loss=max_loss,
            spot_at_entry=spot, vix_at_entry=vix, delta_net=delta_net,
            snapshot_json=json.dumps(snapshot_data),
            legs=[
                {'conId': c_long_put.conId, 'action': 'BUY', 'strike': c_long_put.strike, 'right': 'P'},
                {'conId': c_short_put.conId, 'action': 'SELL', 'strike': c_short_put.strike, 'right': 'P'},
                {'conId': c_short_call.conId, 'action': 'SELL', 'strike': c_short_call.strike, 'right': 'C'},
                {'conId': c_long_call.conId, 'action': 'BUY', 'strike': c_long_call.strike, 'right': 'C'},
            ],
            delta_put=delta_put, delta_call=delta_call, theta=theta_total, gamma=gamma_total
        )
        
        self.save_state()
        self.logger.info(f"✅ Iron Condor Opened [ID:{trade_id}] and State Saved")
        return self.active_position

    def close_position_atomic(self) -> bool:
        """Cerrar Iron Condor como combo atómico (BAG order) (FIX 4)"""
        if not self.active_position or not self.active_position.legs:
            self.logger.warning("No active position legs found for atomic closure")
            return False
        
        self.logger.info("📉 Attempting atomic closure (BAG order)...")
        
        # Construir combo inverso
        close_combo = Contract()
        close_combo.symbol = "XSP"
        close_combo.secType = "BAG"
        close_combo.currency = "USD"
        close_combo.exchange = "SMART"
        close_combo.comboLegs = []
        
        for leg_info in self.active_position.legs:
            leg = ComboLeg()
            leg.conId = leg_info['conId']
            leg.ratio = 1
            # DO NOT INVERT ACTION HERE. 
            # IBKR Logic: Sell Order on Strategy(Buy A, Sell B) -> Sell A, Buy B.
            leg.action = leg_info['action'] 
            leg.exchange = "SMART"
            close_combo.comboLegs.append(leg)
        
        # Para cierre: usar market order o limit agresivo
        close_order = MarketOrder(action='SELL', totalQuantity=self.active_position.qty)
        
        trade = self.ib.placeOrder(close_combo, close_order)
        
        # Esperar fill
        start = time_module.time()
        while (time_module.time() - start) < 30:
            self.ib.sleep(1)
            if trade.orderStatus.status == 'Filled':
                self.logger.info("✅ BAG order filled successfully.")
                return True
            if trade.orderStatus.status in ['Cancelled', 'Inactive', 'ApiCancelled']:
                break
        
        # Si BAG falla, ENTONCES intentar legs individuales como fallback
        self.logger.warning("BAG close failed, attempting individual legs fallback")
        return self.close_position_individual_fallback()

    def close_position_individual_fallback(self) -> bool:
        """Cierre de patas individuales si falla el BAG order (FIX 4)"""
        self.logger.info("📉 Fallback: Closing individual legs...")
        positions = self.ib.positions()
        xsp_positions = [p for p in positions if p.contract.symbol == 'XSP' and p.position != 0]
        
        if not xsp_positions:
            return True

        all_filled = True
        for pos in xsp_positions:
            contract = pos.contract
            contract.exchange = 'SMART'
            self.ib.qualifyContracts(contract)
            
            action = 'SELL' if pos.position > 0 else 'BUY'
            order = MarketOrder(action, abs(pos.position))
            trade = self.ib.placeOrder(contract, order)
            
            # Wait for fill with timeout
            start_wait = time_module.time()
            while (time_module.time() - start_wait) < 20:
                self.ib.sleep(1)
                if trade.orderStatus.status == 'Filled':
                    break
            
            if trade.orderStatus.status != 'Filled':
                self.logger.error(f"❌ Failed to close leg {contract.strike}{contract.right}")
                all_filled = False
        
        if all_filled:
            return True
        return False
    
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
        
        # 1. Take Profit
        if pnl >= take_profit_target:
            return f"TP_50 [PnL ${pnl:.0f} >= ${take_profit_target:.0f}]"
        
        # 2. Stop Loss
        if pnl <= stop_loss_target:
            return f"SL_{int(self.STOP_LOSS_MULT)}X [PnL ${pnl:.0f} <= ${stop_loss_target:.0f}]"
        
        # 3. EOD Force Close (time of day cutoff)
        now_time = datetime.now().time()
        if now_time >= self.FORCE_CLOSE_TIME:
            return f"EOD_TIME [Time {now_time.strftime('%H:%M')} >= {self.FORCE_CLOSE_TIME}]"
        
        return None
    
    def close_position(self, reason: str, max_spread_val: float = 0.0, rv_duration: float = 0.0) -> bool:
        """
        Close the active position with blocking confirmation.
        
        Args:
            reason: Reason for closing
            max_spread_val: Max observed spread value (for journal)
            rv_duration: Realized Volatility annualized (for journal)
            
        Returns:
            True if closed successfully, False if failed/timeout
        """
        if not self.active_position:
            return True
        
        self.logger.info(f"\n🔴 CLOSING POSITION: {reason} | RV: {rv_duration:.2f}%")
        
        # Get final PnL estimate
        final_pnl = self.get_position_pnl() or 0.0
        
        # Use Atomic Closure (BAG Order)
        success = self.close_position_atomic()
        
        if success:
            self.logger.info(f"✅ EXIT CONFIRMED via Atomic BAG. Final PnL: ${final_pnl:.2f}")
             # Final Journal Update
            if self.journal and self.active_position:
                self.journal.log_trade_close(
                    trade_id=self.active_position.trade_id,
                    exit_reason=reason,
                    final_pnl_usd=final_pnl,
                    entry_timestamp=self.active_position.entry_time,
                    max_spread_val=max_spread_val,
                    rv_duration=rv_duration
                )
            self.active_position = None
            self.save_state()
            return True
        else:
            self.logger.error("❌ EXIT FAILED - Atomic Closure failed. Position might be partially open.")
            return False


    
    def monitor_position(self, check_interval: float = 10.0):
        """
        Monitor active position for exit conditions.
        
        Args:
            check_interval: Seconds between checks
        """
        from datetime import time as dt_time
        
        if not self.active_position:
            print("No active position to monitor.")
            return
        
        print(f"\n📡 Monitoring position (checking every {check_interval}s)...")
        
        # Market close time
        # Market close time
        market_close = dt_time(16, 0)
        
        max_spread_val = 0.0
        spot_prices = [] # NEW: for RV calc
        
        while self.active_position:
            # Update Max Spread Value (Tail Risk)
            # PnL = (EntryCredit - CurrentSpread) * 100 * Qty
            # CurrentSpread = EntryCredit - (PnL / (100 * self.active_position.qty))
            pnl = self.get_position_pnl()
            if pnl is not None:
                current_spread_cost = self.active_position.entry_credit - (pnl / (100 * self.active_position.qty))
                if current_spread_cost > max_spread_val:
                    max_spread_val = current_spread_cost
            
            # Get current spot
            spot = self.connector.get_live_price('XSP') or 0.0
            if spot > 0:
                spot_prices.append(spot)
            
            # Get VIX
            vix = self.connector.get_live_price('VIX') or 0.0
            
            # Calculate distances to strikes
            short_put = self.active_position.short_put_strike
            short_call = self.active_position.short_call_strike
            dist_put = spot - short_put
            dist_call = short_call - spot
            
            # Danger zone detection (Global scope)
            min_distance = min(dist_put, dist_call)
            
            # Calculate percentages
            max_profit = self.active_position.max_profit
            max_loss = self.active_position.max_loss
            pnl_val = pnl if pnl is not None else 0.0
            
            tp_pct = (pnl_val / max_profit * 100) if max_profit > 0 else 0
            sl_pct = (pnl_val / max_loss * 100) if max_loss > 0 else 0 # Careful, max_loss is positive number
            # Actually max_loss is max loss amount. SL hit is negative PnL. 
            # Let's stick to TP% of MaxProfit.
            
            # Time held
            now = datetime.now()
            
            # Calculate hold time
            hold_time = (now - self.active_position.entry_time).total_seconds() / 60
            
            if pnl is not None:
                max_profit = self.active_position.max_profit
                tp_target = max_profit * self.TAKE_PROFIT_PCT
                # Show % of TP target reached (100% = hit target)
                tp_pct = (pnl / tp_target) * 100 if tp_target > 0 else 0
                # Also show % of max profit for context
                max_pct = (pnl / max_profit) * 100 if max_profit > 0 else 0
                
                # Calculate SL progress
                sl_target = -self.active_position.entry_credit * 100 * self.STOP_LOSS_MULT
                sl_pct = (pnl / sl_target) * 100 if sl_target < 0 else 0
                
                # Danger zone detection
                # min_distance = min(dist_put, dist_call) # Moved up
                current_spread_cost = self.active_position.entry_credit - (pnl / (100 * self.active_position.qty))
                if current_spread_cost > max_spread_val:
                    max_spread_val = current_spread_cost
            
            # Danger zone detection
            danger = "⚠️ DANGER" if min_distance < 5 else ""
            
            # Enhanced output: TP% means progress to TP target, Hold shows time in trade
            now_str = now.strftime("%H:%M:%S")
            # Calculate max potential profit pct realized
            max_pct = (pnl_val / max_profit * 100)
            
            base_output = f"[{now_str}] XSP: {spot:.2f} | VIX: {vix:.1f} | PnL: ${pnl_val:.2f} ({max_pct:.0f}%Max) | TP:{tp_pct:.0f}% SL:{sl_pct:.0f}% | {short_put:.0f}P({dist_put:+.0f}) {short_call:.0f}C({dist_call:+.0f}) | Hold:{hold_time:.0f}m | MaxSprd:{max_spread_val:.2f}"
            print(f"{danger} {base_output}" if danger else base_output, end="\r")
            
            exit_reason = self.check_exit_conditions()
            if exit_reason:
                # Calculate Realized Volatility (RV)
                rv_duration = 0.0
                if len(spot_prices) > 2:
                     try:
                        prices_arr = np.array(spot_prices)
                        log_returns = np.log(prices_arr[1:] / prices_arr[:-1])
                        std_dev = np.std(log_returns)
                        # Annualize: sqrt(252 trading days * 6.5 hours * 3600 seconds / interval)
                        # We use the actual sampling interval (check_interval)
                        annualization_factor = np.sqrt(252 * 6.5 * 3600 / check_interval)
                        rv_duration = std_dev * annualization_factor
                     except Exception as e:
                        print(f"⚠️ RV Calc Error: {e}")
                
                success = self.close_position(exit_reason, max_spread_val, rv_duration)
                if success:
                    break
                else:
                    print("⚠️ Exit failed. Retrying monitoring loop...")
                    self.ib.sleep(1)
            
            self.ib.sleep(check_interval)

"""
IBKR Combo Order Price Convention (from official docs):
https://www.ibkrguides.com/traderworkstation/notes-on-combination-orders.htm

| Operation                    | action | lmtPrice   | Meaning                    |
|------------------------------|--------|------------|----------------------------|
| Open Iron Condor (credit)    | BUY    | negative   | Buy combo, receive credit  |
| Chase (easier fill)          | BUY    | += TICK    | Less negative = less credit|
| Close Iron Condor (debit)    | SELL   | negative   | Sell combo, pay debit      |
| Close Iron Condor (residual) | SELL   | positive   | Sell combo, receive cash   |

Chase direction for BUY credit: += TICK_SIZE (less negative = easier fill)
DO NOT modify combo order lmtPrice — Error 105. Cancel + new order.
DO NOT qualifyContracts() on BAG — Error 321. Build from qualified legs.
"""

