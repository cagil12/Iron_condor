"""
execution.py

Live Execution Engine for XSP Iron Condors via IBKR.
Handles order placement, position monitoring, and exit management.

SAFETY: Hardcoded limits for small account ($200 capital).
"""
import time as time_module
from datetime import datetime, time, timedelta
import json
import csv
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
    - WING_WIDTH = 2.0 (default for current $200-risk profile, configurable)
    - QTY = 1 (no pyramiding)
    - Active position check before entry
    """
    
    # HARDCODED SAFETY LIMITS
    WING_WIDTH = 2.0       # 2-point wings (Default, overwritten by config)
    MAX_QTY = 1            # Only 1 contract at a time
    TAKE_PROFIT_PCT = 1.00 # Default: hold winners to expiry (overwritten by config)
    # STOP_LOSS_MULT removed from hardcoded constants - loaded from config
    ORDER_TIMEOUT = 10     # Seconds to wait for fill
    CHASE_TICKS = 8        # Number of times to chase price (FIX 3)
    TICK_SIZE = 0.05       # Combo tick size (aggressive for BAG orders) (FIX 3)
    TERMINAL_STATES = {"Filled", "Cancelled", "ApiCancelled", "Inactive", "ApiPending"}
    FORCE_CLOSE_TIME = time(15, 45)  # Hard EOD exit at 3:45 PM
    STATE_FILE = Path("state.json")
    MAX_MARGIN_ACCEPTED = 400.0     # Maximum initial margin for 1 contract (wing=2 theoretical=$200, buffer for IBKR haircuts)
    
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
        self.SYMBOL: str = str(self.config.get('symbol', 'XSP'))
        self.active_position: Optional[IronCondorPosition] = None
        self.journal = journal
        self.consecutive_losses = 0
        self.streak_pause_until = None
        self.cumulative_pnl = 0.0
        self.pnl_high_water_mark = 0.0
        self.dd_pause_until = None
        # Track hold-to-expiry signatures already finalized today to prevent
        # re-recovery loops when IBKR still reports expired 0DTE option legs.
        self.expired_hold_signatures: List[str] = []
        self.last_expiry_cleanup_key: Optional[str] = None
        # Tracks in-flight entry order ID to prevent chase re-submit races.
        self.pending_entry_order_id: Optional[int] = None
        
        # Load dynamic settings
        self.TAKE_PROFIT_PCT = float(self.config.get('take_profit_pct', self.TAKE_PROFIT_PCT))
        self.STOP_LOSS_MULT = float(self.config.get('stop_loss_mult', 2.0))
        self.WING_WIDTH = float(self.config.get('wing_width', 2.0))  # FIX 1
        self.HOLD_TO_EXPIRY_MODE = self.TAKE_PROFIT_PCT >= 1.00
        self.logger.info(
            f"HOLD_TO_EXPIRY_MODE={self.HOLD_TO_EXPIRY_MODE} "
            f"(tp_pct={self.TAKE_PROFIT_PCT:.2f})"
        )

        # Commission expectations (IBKR Fixed defaults: 0.65/leg x 4 legs).
        self.COMMISSION_PER_LEG = float(self.config.get('commission_per_leg', 0.65))
        self.LEGS_PER_IC = int(self.config.get('legs_per_ic', 4))
        self.OPEN_COMMISSION = self.COMMISSION_PER_LEG * self.LEGS_PER_IC
        self.ROUND_TRIP_COMMISSION = self.OPEN_COMMISSION * 2.0

        # Hold-to-expiry mode pays opening fee on wins, round-trip on active closes.
        if self.HOLD_TO_EXPIRY_MODE:
            self.expected_commission_win = self.OPEN_COMMISSION
            self.expected_commission_loss = self.ROUND_TRIP_COMMISSION
        else:
            self.expected_commission_win = self.ROUND_TRIP_COMMISSION
            self.expected_commission_loss = self.ROUND_TRIP_COMMISSION

        if self.HOLD_TO_EXPIRY_MODE and abs(self.STOP_LOSS_MULT - 2.0) < 1e-9:
            self.exit_strategy_label = "hold_to_expiry_sl2x"
        else:
            tp_label = int(round(self.TAKE_PROFIT_PCT * 100))
            sl_label = f"{self.STOP_LOSS_MULT:g}".replace(".", "p")
            self.exit_strategy_label = f"tp{tp_label}_sl{sl_label}x"

        self.logger.info(
            f"🔧 Executor Config: TP={self.TAKE_PROFIT_PCT:.2f} | "
            f"SL={self.STOP_LOSS_MULT:.1f}x | Width={self.WING_WIDTH:.1f} | "
            f"ExitStrategy={self.exit_strategy_label}"
        )
        self.logger.info(
            f"💸 Expected commissions: win=${self.expected_commission_win:.2f} | "
            f"loss=${self.expected_commission_loss:.2f} | "
            f"open=${self.OPEN_COMMISSION:.2f} | round_trip=${self.ROUND_TRIP_COMMISSION:.2f}"
        )
        if self.HOLD_TO_EXPIRY_MODE:
            self.logger.info(
                "📌 TP mode is hold-to-expiry (tp_pct >= 1.00): skipping TP close orders; "
                "only SL/EOD can trigger active close."
            )
        
        # FIX 3: Load state at startup
        self.load_state()
        self.logger.info(
            f"🛡️ Kill Switches: "
            f"L1={'ON' if self.config.get('dd_kill_enabled') else 'OFF'} "
            f"(DD>{self.config.get('dd_max_pct', 0.15) * 100:.0f}%) | "
            f"L3={'ON' if self.config.get('vix_gate_enabled') else 'OFF'} "
            f"(VIX>{self.config.get('vix_gate_threshold', 30):.0f}) | "
            f"L5={'ON' if self.config.get('streak_stop_enabled') else 'OFF'} "
            f"({self.config.get('streak_max_losses', 3)} losses)"
        )
        if self.dd_pause_until:
            self.logger.warning(f"⚠️ L1 ACTIVE: Trading halted until {self.dd_pause_until}")
        if self.streak_pause_until:
            self.logger.warning(f"⚠️ L5 ACTIVE: Streak pause until {self.streak_pause_until}")

    def save_state(self):
        """Persistir estado a disco para sobrevivir reinicios (FIX 3)"""
        try:
            state = {
                'active_position': self.active_position.__dict__ if self.active_position else None,
                'consecutive_losses': self.consecutive_losses,
                'streak_pause_until': self.streak_pause_until,
                'cumulative_pnl': self.cumulative_pnl,
                'pnl_high_water_mark': self.pnl_high_water_mark,
                'dd_pause_until': self.dd_pause_until,
                'expired_hold_signatures': self.expired_hold_signatures,
                'last_expiry_cleanup_key': self.last_expiry_cleanup_key,
                'pending_entry_order_id': self.pending_entry_order_id,
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
                self.consecutive_losses = int(state.get('consecutive_losses', 0) or 0)
                self.streak_pause_until = state.get('streak_pause_until', None)
                self.cumulative_pnl = float(state.get('cumulative_pnl', 0.0) or 0.0)
                self.pnl_high_water_mark = float(state.get('pnl_high_water_mark', 0.0) or 0.0)
                self.dd_pause_until = state.get('dd_pause_until', None)
                signatures = state.get('expired_hold_signatures', []) or []
                if isinstance(signatures, list):
                    self.expired_hold_signatures = [str(s) for s in signatures if s]
                else:
                    self.expired_hold_signatures = []
                self.last_expiry_cleanup_key = state.get('last_expiry_cleanup_key', None)
                pending = state.get('pending_entry_order_id', None)
                self.pending_entry_order_id = int(pending) if pending not in (None, "", 0, "0") else None
            except Exception as e:
                self.logger.error(f"Failed to load state: {e}")
                self.active_position = None
                self.consecutive_losses = 0
                self.streak_pause_until = None
                self.cumulative_pnl = 0.0
                self.pnl_high_water_mark = 0.0
                self.dd_pause_until = None
                self.expired_hold_signatures = []
                self.last_expiry_cleanup_key = None
                self.pending_entry_order_id = None

    def _wait_for_terminal(self, order: Order, timeout: float = 10.0) -> str:
        """
        Poll until order reaches a terminal state.

        Returns:
            "Filled", "Cancelled", specific terminal state, or "Timeout".
        """
        order_id = int(getattr(order, 'orderId', 0) or 0)
        if order_id <= 0:
            return "Timeout"

        deadline = time_module.time() + timeout
        while time_module.time() < deadline:
            try:
                open_trades = self.ib.openTrades()
            except Exception:
                open_trades = []

            tracked = None
            for ot in open_trades:
                if int(getattr(ot.order, 'orderId', 0) or 0) == order_id:
                    tracked = ot
                    break

            if tracked is None:
                try:
                    fills = self.ib.fills()
                except Exception:
                    fills = []
                for fill in fills:
                    exec_obj = getattr(fill, 'execution', None)
                    if exec_obj and int(getattr(exec_obj, 'orderId', 0) or 0) == order_id:
                        return "Filled"
                return "Cancelled"

            status = str(getattr(tracked.orderStatus, 'status', '') or '')
            if status in self.TERMINAL_STATES:
                return "Filled" if status == "Filled" else status

            self.ib.sleep(0.25)

        return "Timeout"

    def _register_fill_from_race(self, order: Order) -> None:
        """
        Recover active position when a cancel-targeted entry order fills.
        """
        order_id = int(getattr(order, 'orderId', 0) or 0)
        target_symbol = str(getattr(self, 'SYMBOL', self.config.get('symbol', '')))

        try:
            fills = self.ib.fills()
        except Exception:
            fills = []
        order_fills = [
            f for f in fills
            if int(getattr(getattr(f, 'execution', None), 'orderId', 0) or 0) == order_id
            and getattr(getattr(f, 'contract', None), 'symbol', '') == target_symbol
        ]
        leg_fills = [
            f for f in order_fills
            if getattr(getattr(f, 'contract', None), 'secType', '') in ('OPT', 'FOP')
        ]

        legs_by_conid: Dict[int, Dict[str, Any]] = {}
        for fill in leg_fills:
            contract = fill.contract
            exec_obj = fill.execution
            side = str(getattr(exec_obj, 'side', '')).upper()
            action = 'SELL' if side == 'SLD' else 'BUY'
            legs_by_conid[int(contract.conId)] = {
                'conId': int(contract.conId),
                'action': action,
                'strike': float(contract.strike),
                'right': str(contract.right),
            }

        recovered_legs = list(legs_by_conid.values())
        short_put = long_put = short_call = long_call = None
        if recovered_legs:
            sp = sorted([l for l in recovered_legs if l['right'] == 'P' and l['action'] == 'SELL'], key=lambda x: x['strike'])
            lp = sorted([l for l in recovered_legs if l['right'] == 'P' and l['action'] == 'BUY'], key=lambda x: x['strike'])
            sc = sorted([l for l in recovered_legs if l['right'] == 'C' and l['action'] == 'SELL'], key=lambda x: x['strike'])
            lc = sorted([l for l in recovered_legs if l['right'] == 'C' and l['action'] == 'BUY'], key=lambda x: x['strike'])
            if sp and lp and sc and lc:
                short_put, long_put = sp[-1], lp[0]
                short_call, long_call = sc[0], lc[-1]

        if not (short_put and long_put and short_call and long_call):
            # Fallback to live positions if leg fills are unavailable/incomplete.
            positions = [
                p for p in self.ib.positions()
                if p.contract.symbol == target_symbol and p.contract.secType == 'OPT' and p.position != 0 and abs(p.position) == 1
            ]
            puts = sorted([p for p in positions if p.contract.right == 'P'], key=lambda p: p.contract.strike)
            calls = sorted([p for p in positions if p.contract.right == 'C'], key=lambda p: p.contract.strike)
            if len(puts) == 2 and len(calls) == 2:
                long_put_p, short_put_p = puts[0], puts[1]
                short_call_p, long_call_p = calls[0], calls[1]
                short_put = {'conId': int(short_put_p.contract.conId), 'action': 'SELL', 'strike': float(short_put_p.contract.strike), 'right': 'P'}
                long_put = {'conId': int(long_put_p.contract.conId), 'action': 'BUY', 'strike': float(long_put_p.contract.strike), 'right': 'P'}
                short_call = {'conId': int(short_call_p.contract.conId), 'action': 'SELL', 'strike': float(short_call_p.contract.strike), 'right': 'C'}
                long_call = {'conId': int(long_call_p.contract.conId), 'action': 'BUY', 'strike': float(long_call_p.contract.strike), 'right': 'C'}
                recovered_legs = [long_put, short_put, short_call, long_call]
            else:
                self.pending_entry_order_id = None
                self.save_state()
                self.logger.error(
                    f"🛑 Race fill registered: could not reconstruct 4-leg structure for order {order_id}."
                )
                return

        qty = 1
        if leg_fills:
            try:
                qty = max(int(abs(getattr(f.execution, 'shares', 1) or 1)) for f in leg_fills)
            except Exception:
                qty = 1
        qty = max(qty, 1)

        combo_credit = abs(float(getattr(order, 'lmtPrice', 0.0) or 0.0))
        verified_credit = None
        if leg_fills:
            leg_total = 0.0
            for fill in leg_fills:
                exec_obj = fill.execution
                shares = float(getattr(exec_obj, 'shares', 0.0) or 0.0)
                price = float(getattr(exec_obj, 'price', 0.0) or 0.0)
                if str(getattr(exec_obj, 'side', '')).upper() == 'SLD':
                    leg_total += price * shares
                else:
                    leg_total -= price * shares
            if leg_total > 0:
                raw_credit = leg_total / qty
                if raw_credit > (self.WING_WIDTH * 5):
                    raw_credit /= 100.0
                verified_credit = round(raw_credit, 2)

        entry_credit = verified_credit if verified_credit is not None else combo_credit
        if entry_credit <= 0:
            entry_credit = combo_credit if combo_credit > 0 else 0.01

        wing_width = max(float(short_put['strike']) - float(long_put['strike']), 0.01)
        max_profit = entry_credit * qty * 100
        max_loss = max((wing_width - entry_credit) * qty * 100, 0.0)

        spot = 0.0
        vix = 0.0
        try:
            spot = float(self.connector.get_live_price(target_symbol) or 0.0)
        except Exception:
            spot = 0.0
        try:
            vix = float(self.connector.get_live_price('VIX') or 0.0)
        except Exception:
            vix = 0.0

        entry_time = datetime.now()
        if order_fills:
            try:
                entry_time = min(f.time for f in order_fills).astimezone().replace(tzinfo=None)
            except Exception:
                entry_time = datetime.now()

        method = "RACE_RECOVERY"
        trade_id = 0
        if self.journal:
            try:
                trade_id = self.journal.log_trade_open(
                    spot_price=spot,
                    vix_value=vix,
                    short_put_strike=float(short_put['strike']),
                    short_call_strike=float(short_call['strike']),
                    wing_width=wing_width,
                    entry_credit=entry_credit,
                    initial_credit=entry_credit,
                    iv_entry_atm=(vix / 100.0 if vix > 0 else 0.0),
                    max_profit_usd=max_profit,
                    max_loss_usd=max_loss,
                    delta_net=0.0,
                    delta_put=0.0,
                    delta_call=0.0,
                    theta=0.0,
                    gamma=0.0,
                    selection_method=method,
                    target_delta=self.config.get('target_delta', 0.10),
                    otm_distance_pct="N/A",
                    snapshot_json=json.dumps({}),
                    commissions_est=self.expected_commission_win,
                    reasoning=(
                        f"Method: {method} | ExitStrategy: {self.exit_strategy_label} | "
                        f"RecoveredFromOrder: {order_id} | Credit: ${entry_credit:.2f}"
                    ),
                )
            except Exception as e:
                self.logger.warning(f"🛑 Race fill registered: failed to journal OPEN row ({e})")

        self.active_position = IronCondorPosition(
            trade_id=trade_id,
            entry_time=entry_time,
            short_put_strike=float(short_put['strike']),
            long_put_strike=float(long_put['strike']),
            short_call_strike=float(short_call['strike']),
            long_call_strike=float(long_call['strike']),
            entry_credit=entry_credit,
            qty=qty,
            max_profit=max_profit,
            max_loss=max_loss,
            spot_at_entry=spot,
            vix_at_entry=vix,
            delta_net=0.0,
            snapshot_json=json.dumps({}),
            legs=[long_put, short_put, short_call, long_call],
            delta_put=0.0,
            delta_call=0.0,
            theta=0.0,
            gamma=0.0,
        )
        self.pending_entry_order_id = None
        self.save_state()
        self.logger.warning(
            f"🛑 Race fill registered: order {order_id} recovered as "
            f"{self.active_position.short_put_strike:.0f}P/{self.active_position.short_call_strike:.0f}C "
            f"credit=${entry_credit:.2f} qty={qty}"
        )

    def _resolve_stale_pending_entry_on_startup(self) -> None:
        """Resolve stale in-flight entry order lock from previous run."""
        if not self.pending_entry_order_id:
            return

        order_id = int(self.pending_entry_order_id)
        resolved = "Cancelled"

        # Check live open orders first.
        try:
            open_trade = next(
                (t for t in self.ib.openTrades() if int(getattr(t.order, 'orderId', 0) or 0) == order_id),
                None,
            )
        except Exception:
            open_trade = None

        if open_trade is not None:
            resolved = self._wait_for_terminal(open_trade.order, timeout=10.0)
        else:
            try:
                has_fill = any(
                    int(getattr(getattr(f, 'execution', None), 'orderId', 0) or 0) == order_id
                    for f in self.ib.fills()
                )
            except Exception:
                has_fill = False
            resolved = "Filled" if has_fill else "Cancelled"

        if resolved == "Filled":
            class _RecoveredOrder:
                def __init__(self, oid: int):
                    self.orderId = oid
                    self.lmtPrice = 0.0

            self._register_fill_from_race(_RecoveredOrder(order_id))
        else:
            self.pending_entry_order_id = None
            self.save_state()

        self.logger.warning(
            f"⚠️ Stale pending_entry_order_id detected on startup: {order_id}. Resolved as {resolved}."
        )

    def _build_positions_signature(self, positions: List[Any]) -> str:
        """Build deterministic signature for a set of live IBKR option positions."""
        if not positions:
            return ""
        parts = []
        for p in positions:
            c = p.contract
            parts.append(
                f"{int(getattr(c, 'conId', 0))}:{getattr(c, 'right', '')}:"
                f"{float(getattr(c, 'strike', 0.0)):.1f}:{int(p.position)}"
            )
        return "|".join(sorted(parts))

    def _build_legs_signature(self, legs: List[Dict[str, Any]]) -> str:
        """Build deterministic signature for internally-tracked IC legs."""
        if not legs:
            return ""
        parts = []
        for leg in legs:
            conid = int(leg.get('conId', 0) or 0)
            right = str(leg.get('right', ''))
            strike = float(leg.get('strike', 0.0) or 0.0)
            action = str(leg.get('action', ''))
            qty = -1 if action.upper() == 'SELL' else 1
            parts.append(f"{conid}:{right}:{strike:.1f}:{qty}")
        return "|".join(sorted(parts))

    def _signature_day_key(self, signature: str, ts: Optional[datetime] = None) -> str:
        """Attach date to a signature to prevent cross-day suppression."""
        if not signature:
            return ""
        day = (ts or datetime.now()).strftime("%Y-%m-%d")
        return f"{day}|{signature}"

    def _is_expired_signature_blocked(self, signature: str, ts: Optional[datetime] = None) -> bool:
        key = self._signature_day_key(signature, ts)
        return bool(key) and key in self.expired_hold_signatures

    def _register_expired_signature(self, signature: str, ts: Optional[datetime] = None):
        key = self._signature_day_key(signature, ts)
        if not key:
            return
        if key not in self.expired_hold_signatures:
            self.expired_hold_signatures.append(key)
            # Keep state bounded
            if len(self.expired_hold_signatures) > 100:
                self.expired_hold_signatures = self.expired_hold_signatures[-100:]

    def _find_open_trade_id(
        self,
        short_put_strike: float,
        short_call_strike: float,
        wing_width: float,
    ) -> int:
        """
        Resolve recovered position to an existing OPEN trade in journal.
        Returns 0 when no matching OPEN row is found.
        """
        if not self.journal:
            return 0
        journal_path = getattr(self.journal, 'journal_path', None)
        if not journal_path:
            return 0
        path_obj = Path(journal_path)
        if not path_obj.exists():
            return 0

        try:
            with open(path_obj, 'r', newline='', encoding='utf-8') as f:
                rows = list(csv.DictReader(f))
        except Exception as e:
            self.logger.warning(f"Could not read journal for recovery trade_id lookup: {e}")
            return 0

        for row in reversed(rows):
            try:
                status = str(row.get('status', '')).strip().upper()
                if status != 'OPEN':
                    continue
                sp = float(row.get('short_put_strike', 0.0) or 0.0)
                sc = float(row.get('short_call_strike', 0.0) or 0.0)
                ww = float(row.get('wing_width', 0.0) or 0.0)
                if (
                    abs(sp - short_put_strike) <= 0.01
                    and abs(sc - short_call_strike) <= 0.01
                    and abs(ww - wing_width) <= 0.01
                ):
                    return int(row.get('trade_id', 0) or 0)
            except Exception:
                continue
        return 0

    def _finalize_hold_to_expiry(
        self,
        final_pnl: float,
        max_spread_val: float,
        min_pnl_observed: float,
        max_pnl_observed: float,
    ):
        """Finalize state/journal for hold-to-expiry without requiring a close order."""
        if not self.active_position:
            return

        signature = self._build_legs_signature(self.active_position.legs or [])
        cleanup_key = self._signature_day_key(signature)
        if cleanup_key and self.last_expiry_cleanup_key == cleanup_key:
            self.logger.warning(
                "⚠️ Duplicate hold-to-expiry cleanup detected for same position signature; skipping duplicate accounting."
            )
            self.active_position = None
            self.save_state()
            return

        self.logger.info(
            "✅ Market close reached in hold-to-expiry mode; no TP close order sent."
        )

        trade_id = int(self.active_position.trade_id or 0)
        if self.journal and trade_id > 0:
            self.journal.log_trade_close(
                trade_id=trade_id,
                exit_reason="EXPIRED_HOLD_TO_EXPIRY",
                final_pnl_usd=final_pnl,
                entry_timestamp=self.active_position.entry_time,
                max_spread_val=max_spread_val,
                rv_duration=0.0,
                max_adverse_excursion=min_pnl_observed,
                max_favorable_excursion=max_pnl_observed,
            )
            self._update_kill_switch_trackers(final_pnl)
        elif self.journal and trade_id <= 0:
            self.logger.warning(
                "⚠️ Skipping journal/DD update for recovered position with trade_id<=0 (unmapped position)."
            )
        else:
            self._update_kill_switch_trackers(final_pnl)

        if cleanup_key:
            self.last_expiry_cleanup_key = cleanup_key
        if signature:
            self._register_expired_signature(signature)
        self.active_position = None
        self.save_state()

    def _update_kill_switch_trackers(self, final_pnl: float):
        """Update L5 streak tracking and L1 drawdown tracking after a trade is closed."""
        # === L5: STREAK TRACKING ===
        if final_pnl < 0:
            self.consecutive_losses += 1
            self.logger.info(f"📉 L5 STREAK: Loss #{self.consecutive_losses}")
            if self.consecutive_losses >= self.config.get('streak_max_losses', 3):
                pause_days = self.config.get('streak_pause_days', 2)
                pause_until = (datetime.now() + timedelta(days=pause_days)).strftime('%Y-%m-%d')
                self.streak_pause_until = pause_until
                self.logger.warning(
                    f"🛑 L5 STREAK STOP: {self.consecutive_losses} consecutive losses. "
                    f"Pausing until {pause_until}."
                )
        else:
            if self.consecutive_losses > 0:
                self.logger.info(f"✅ L5 STREAK RESET: Win after {self.consecutive_losses} losses")
            self.consecutive_losses = 0
            self.streak_pause_until = None

        # === L1: DRAWDOWN TRACKING ===
        self.cumulative_pnl += final_pnl
        if self.cumulative_pnl > self.pnl_high_water_mark:
            self.pnl_high_water_mark = self.cumulative_pnl

        current_dd = self.pnl_high_water_mark - self.cumulative_pnl
        max_capital = self.config.get('max_capital', 2000.0)
        dd_threshold = max_capital * self.config.get('dd_max_pct', 0.15)
        self.logger.info(
            f"📊 L1 DD TRACKER: Cumulative=${self.cumulative_pnl:.2f}, "
            f"HWM=${self.pnl_high_water_mark:.2f}, DD=${current_dd:.2f}/${dd_threshold:.0f}"
        )
        if self.config.get('dd_kill_enabled', True) and current_dd >= dd_threshold:
            pause_days = self.config.get('dd_pause_days', 5)
            pause_until = (datetime.now() + timedelta(days=pause_days)).strftime('%Y-%m-%d')
            self.dd_pause_until = pause_until
            self.logger.critical(
                f"🚨 L1 PORTFOLIO KILL SWITCH: DD=${current_dd:.2f} >= ${dd_threshold:.0f} "
                f"({self.config.get('dd_max_pct', 0.15) * 100:.0f}% of ${max_capital:.0f}). "
                f"ALL TRADING HALTED until {pause_until}."
            )

    def startup_reconciliation(self):
        """Cancelar órdenes huérfanas y reconciliar posiciones al iniciar (FIX 2)"""
        self.logger.info("REMINDER: Verify TWS setting 'Download open orders on connection' is enabled")
        self.logger.info("TWS → Global Configuration → API → Settings → ☑️ Download open orders")

        if self.pending_entry_order_id:
            self._resolve_stale_pending_entry_on_startup()
        
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
        
        # POSITION ISOLATION (FIX 9): Filter out foreign positions
        # Our IC always trades qty=1/-1. Any position with abs(qty) > 1 is foreign.
        orphan_positions = [p for p in xsp_positions if abs(p.position) > 1]
        if orphan_positions:
            for op in orphan_positions:
                self.logger.warning(f"   ⚠️ Ignoring foreign position: {op.contract.strike} {op.contract.right} qty={op.position}")
        xsp_positions = [p for p in xsp_positions if abs(p.position) == 1]
        
        if not xsp_positions:
            self.logger.info("   ✅ No existing positions found. Ready for new entries.")
            return

        live_signature = self._build_positions_signature(xsp_positions)
        if self._is_expired_signature_blocked(live_signature):
            self.logger.info(
                "   ℹ️ Ignoring previously finalized hold-to-expiry position signature."
            )
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
                trade_id=self._find_open_trade_id(
                    short_put.contract.strike,
                    short_call.contract.strike,
                    wing_width,
                ),
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
            if self.active_position.trade_id > 0:
                self.logger.info(
                    f"   ✅ Position Recovered! TradeID={self.active_position.trade_id} "
                    f"Strikes: {short_put.contract.strike}P/{short_call.contract.strike}C | Hold: {hold_min:.0f}m"
                )
            else:
                self.logger.warning(
                    f"   ⚠️ Position recovered without OPEN journal row "
                    f"(trade_id=0): {short_put.contract.strike}P/{short_call.contract.strike}C"
                )
            
        except Exception as e:
            self.logger.error(f"   ❌ Recovery Failed: {e}")

    def has_active_position(self) -> bool:
        """Check if there's an active position managed by us (FIX 9)."""
        # Check our internal tracking (primary source of truth)
        if self.active_position is not None:
            return True
        
        # Also verify with IBKR — but only count positions with qty=1/-1
        # Foreign positions (manual trades, orphans) have different quantities
        positions = self.ib.positions()
        xsp_positions = [p for p in positions 
                        if p.contract.symbol == 'XSP' 
                        and abs(p.position) == 1]
        if not xsp_positions:
            return False

        live_signature = self._build_positions_signature(xsp_positions)
        if self._is_expired_signature_blocked(live_signature):
            self.logger.info(
                "ℹ️ Ignoring already-finalized hold-to-expiry signature in has_active_position()."
            )
            return False
        return True
    
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
        # === KILL SWITCH GATES (order matters: L1 > L5 > L3) ===
        # L1: Portfolio drawdown pause gate
        if self.config.get('dd_kill_enabled', True) and self.dd_pause_until:
            pause_date = datetime.strptime(self.dd_pause_until, '%Y-%m-%d').date()
            if datetime.now().date() < pause_date:
                current_dd = self.pnl_high_water_mark - self.cumulative_pnl
                self.logger.critical(
                    f"🚨 L1 PORTFOLIO HALTED: DD=${current_dd:.2f}. "
                    f"Resuming on {self.dd_pause_until}. No trading."
                )
                return None
            self.logger.warning(
                f"⚠️ L1 PAUSE EXPIRED. Resuming cautiously. "
                f"DD was ${self.pnl_high_water_mark - self.cumulative_pnl:.2f}. "
                f"Consider manual review before continuing."
            )
            self.dd_pause_until = None
            self.save_state()

        # L5: Consecutive losses pause gate
        if self.config.get('streak_stop_enabled', True) and self.streak_pause_until:
            pause_date = datetime.strptime(self.streak_pause_until, '%Y-%m-%d').date()
            if datetime.now().date() < pause_date:
                self.logger.warning(
                    f"🛑 L5 STREAK PAUSED: {self.consecutive_losses} consecutive losses. "
                    f"Resuming on {self.streak_pause_until}. No trade today."
                )
                return None
            self.logger.info("✅ L5 STREAK PAUSE EXPIRED. Resuming trading.")
            self.streak_pause_until = None
            self.save_state()

        # SAFETY CHECK: No pyramiding (FIX 9)
        # Relies on internal state, not IBKR portfolio queries.
        # This prevents foreign/orphan positions from blocking new trades.
        if self.active_position:
            self.logger.warning("⚠️ ABORT: Active position exists. No pyramiding allowed.")
            return None

        # === L3: VIX REGIME GATE ===
        if self.config.get('vix_gate_enabled', True):
            vix_threshold = self.config.get('vix_gate_threshold', 30.0)
            if vix > vix_threshold:
                self.logger.warning(
                    f"🛑 L3 VIX GATE: VIX={vix:.1f} > {vix_threshold:.0f}. "
                    f"Skipping today. High vol regime = elevated gamma risk."
                )
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
        
        min_credit_required = self.config.get('min_credit', 0.18)
        if total_credit <= 0:
            self.logger.error(f"❌ Invalid credit: ${total_credit:.2f}")
            return None
        
        if total_credit < min_credit_required:
            self.logger.warning(f"⚠️ Credit too low at execution: ${total_credit:.2f} < ${min_credit_required:.2f}. ABORT.")
            return None
            
        self.logger.info(f"📊 Calculated Mid Credit: ${total_credit:.2f} per contract (min: ${min_credit_required:.2f})")
        
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
        
        # Detect paper mode: paper port is typically 7497, live is 7496
        # Also check account ID pattern if needed, but port is a good proxy here based on config
        is_paper = self.ib.client.port == 7497
        
        try:
            state = self.ib.whatIfOrder(bag, test_order)
            if isinstance(state, list):
                if not state:
                    if is_paper:
                        self.logger.warning("⚠️ Margin Check empty (Paper mode) — BYPASSING. Would block in LIVE.")
                    else:
                        self.logger.error("⚠️ Margin Check returned empty list. ABORTING (Live mode).")
                        return None
                else:
                    state = state[0]
                    init_margin = float(state.initMarginChange)
                    if init_margin > self.MAX_MARGIN_ACCEPTED:
                        self.logger.warning(f"   ⚠️ MARGIN REJECT: Expected < ${self.MAX_MARGIN_ACCEPTED}, Got ${init_margin:.2f}")
                        return None
                    self.logger.info(f"   ✅ Margin Check Passed: ${init_margin:.2f}")
            else:
                init_margin = float(state.initMarginChange)
                if init_margin > self.MAX_MARGIN_ACCEPTED:
                    self.logger.warning(f"   ⚠️ MARGIN REJECT: Expected < ${self.MAX_MARGIN_ACCEPTED}, Got ${init_margin:.2f}")
                    return None
                self.logger.info(f"   ✅ Margin Check Passed: ${init_margin:.2f}")
            
        except Exception as e:
            if is_paper:
                self.logger.warning(f"⚠️ Margin Check Failed (Paper mode): {e} — BYPASSING.")
            else:
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
                    prev_order = trade.order
                    self.ib.cancelOrder(prev_order)
                    terminal_status = self._wait_for_terminal(prev_order, timeout=10.0)

                    if terminal_status == "Filled":
                        self.logger.warning(
                            f"🛑 Chase race detected: order {prev_order.orderId} filled during cancel. "
                            "Position already open. Aborting chase."
                        )
                        self._register_fill_from_race(prev_order)
                        return self.active_position
                    if terminal_status == "Timeout":
                        self.logger.error(
                            f"❌ Cancel timeout for order {prev_order.orderId} after 10s. "
                            "Aborting chase — manual review required."
                        )
                        self.pending_entry_order_id = None
                        self.save_state()
                        return None
                    self.pending_entry_order_id = None
                    self.save_state()
                
                # Adjust Price: Less negative = easier fill
                limit_price += self.TICK_SIZE
                min_credit_floor = self.config.get('min_credit', 0.18)
                if abs(limit_price) < min_credit_floor:
                    self.logger.warning(f"   🛑 Chase stopped: credit ${abs(limit_price):.2f} < floor ${min_credit_floor}. Aborting.")
                    if trade:
                        prev_order = trade.order
                        self.ib.cancelOrder(prev_order)
                        terminal_status = self._wait_for_terminal(prev_order, timeout=10.0)
                        if terminal_status == "Filled":
                            self.logger.warning(
                                f"🛑 Chase race detected: order {prev_order.orderId} filled during cancel. "
                                "Position already open. Aborting chase."
                            )
                            self._register_fill_from_race(prev_order)
                            return self.active_position
                    self.pending_entry_order_id = None
                    self.save_state()
                    return None
                    
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
            real_id = int(getattr(trade.order, 'orderId', 0) or 0)
            if real_id > 0:
                self.pending_entry_order_id = real_id
                self.save_state()
            else:
                self.logger.warning(
                    "⚠️ placeOrder returned orderId=0. pending_entry_order_id lock not set for this attempt."
                )
            
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
                prev_order = trade.order
                self.ib.cancelOrder(prev_order)
                terminal_status = self._wait_for_terminal(prev_order, timeout=10.0)
                if terminal_status == "Filled":
                    self.logger.warning(
                        f"🛑 Chase race detected: order {prev_order.orderId} filled during cancel. "
                        "Position already open. Aborting chase."
                    )
                    self._register_fill_from_race(prev_order)
                    return self.active_position
            self.pending_entry_order_id = None
            self.save_state()
            return None

        self.pending_entry_order_id = None
        self.save_state()
            
        # Success - Verify real credit from individual leg fills
        combo_credit = abs(trade.order.lmtPrice)

        # Attempt to get actual fill prices per leg from executions
        verified_credit = None
        try:
            self.ib.sleep(2)  # Allow execution reports to propagate
            fills = trade.fills
            if fills and len(fills) > 0:
                # Prefer per-leg option fills to avoid mixing combo-level BAG execution
                leg_fills = [
                    f for f in fills
                    if getattr(getattr(f, 'contract', None), 'secType', '') in ('OPT', 'FOP')
                ]
                fills_to_sum = leg_fills if leg_fills else fills

                # Sum credits: SELL legs contribute positive, BUY legs negative
                leg_total = 0.0
                for fill in fills_to_sum:
                    # In a combo fill, each leg's execution has price and side
                    exec_obj = fill.execution
                    if exec_obj.side == 'SLD':
                        leg_total += exec_obj.price * exec_obj.shares
                    else:  # BOT
                        leg_total -= exec_obj.price * exec_obj.shares

                if leg_total > 0:
                    raw_credit = leg_total / max(self.MAX_QTY, 1)
                    # Some IB reports can express option shares with multiplier semantics.
                    if raw_credit > (self.WING_WIDTH * 5):
                        self.logger.info(
                            f"📐 Credit scale correction detected: raw=${raw_credit:.2f} "
                            f"-> adjusted=${(raw_credit / 100.0):.2f}"
                        )
                        raw_credit /= 100.0
                    verified_credit = round(raw_credit, 2)
                    self.logger.info(
                        f"✅ Credit verified from fills: ${verified_credit:.2f} "
                        f"(combo price was ${combo_credit:.2f}, "
                        f"delta=${abs(verified_credit - combo_credit):.2f})"
                    )
        except Exception as e:
            self.logger.warning(f"⚠️ Could not verify credit from fills: {e}")

        # Use verified credit if available, otherwise fall back to combo price
        credit_received = verified_credit if verified_credit is not None else combo_credit
        if verified_credit is not None and abs(verified_credit - combo_credit) > 0.02:
            self.logger.warning(
                f"⚠️ CREDIT DISCREPANCY: combo=${combo_credit:.2f} vs fills=${verified_credit:.2f}. "
                f"Using fills value for all PnL/SL calculations."
            )

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
                target_delta=self.config.get('target_delta', 0.10), otm_distance_pct="1.5%" if method == "OTM_DISTANCE_PCT" else "N/A",
                snapshot_json=json.dumps(snapshot_data),
                commissions_est=self.expected_commission_win,
                reasoning=(
                    f"Method: {method} | ExitStrategy: {self.exit_strategy_label} | "
                    f"VIX: {vix:.1f} | Delta: P{delta_put:.2f}/C{delta_call:.2f} | "
                    f"Credit: ${credit_received:.2f} | "
                    f"EstComm(win/loss): ${self.expected_commission_win:.2f}/${self.expected_commission_loss:.2f}"
                )
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
        
        # BAG failed — MUST cancel explicitly and wait before fallback
        self.logger.warning("BAG close failed. Cancelling order before fallback...")
        try:
            self.ib.cancelOrder(trade.order)
        except Exception as e:
            self.logger.warning(f"Cancel BAG order exception (may already be dead): {e}")
        # Close path: sleep(2) is acceptable here - duplicate close risk is lower than duplicate entry.
        # _wait_for_terminal is intentionally only required on entry idempotency path.
        self.ib.sleep(2)

        # Belt-and-suspenders: global cancel to clear any residual orders
        self.ib.reqGlobalCancel()
        self.ib.sleep(2)

        self.logger.info("BAG order cancelled. Proceeding to individual leg fallback...")
        return self.close_position_individual_fallback()

    def close_position_individual_fallback(self) -> bool:
        """Cierre de patas individuales si falla el BAG order (FIX 4 + FIX 9)"""
        self.logger.info("📉 Fallback: Closing individual legs...")
        
        # POSITION ISOLATION (FIX 9): ONLY close legs matching our IC's conIds
        # NEVER close unfiltered positions — this caused cascade contamination
        if not (self.active_position and self.active_position.legs):
            self.logger.error("❌ ABORT fallback: No active position legs to reference. Refusing to close blindly.")
            return False
        
        ic_conids = {leg['conId'] for leg in self.active_position.legs}
        positions = self.ib.positions()
        xsp_positions = [p for p in positions 
                       if p.contract.symbol == 'XSP' 
                       and p.position != 0 
                       and p.contract.conId in ic_conids]
        self.logger.info(f"   Filtered to {len(xsp_positions)} IC legs (ignored {len([p for p in positions if p.contract.symbol == 'XSP' and p.position != 0]) - len(xsp_positions)} foreign)")
        
        if not xsp_positions:
            return True

        all_filled = True
        FALLBACK_CHASE_STEPS = 3
        FALLBACK_TICK = 0.05

        for pos in xsp_positions:
            contract = pos.contract
            contract.exchange = 'SMART'
            self.ib.qualifyContracts(contract)

            action = 'SELL' if pos.position > 0 else 'BUY'
            qty = abs(pos.position)

            # Get current market price for aggressive limit
            ticker = self.ib.reqMktData(contract, '', False, False)
            self.ib.sleep(1)

            if action == 'BUY':
                # Closing a short leg: pay the ask + buffer
                base_price = ticker.ask if (ticker.ask and ticker.ask > 0) else 0.10
                limit_price = round(base_price + FALLBACK_TICK, 2)
            else:
                # Closing a long leg: sell at bid - buffer (floor at 0.01)
                base_price = ticker.bid if (ticker.bid and ticker.bid > 0) else 0.01
                limit_price = round(max(base_price - FALLBACK_TICK, 0.01), 2)

            self.ib.cancelMktData(contract)

            leg_filled = False
            trade = None

            for chase in range(FALLBACK_CHASE_STEPS + 1):
                if chase > 0:
                    # Cancel previous order before re-submitting
                    if trade:
                        try:
                            self.ib.cancelOrder(trade.order)
                        except Exception:
                            pass
                        # Exit chase: sleep(1) is acceptable here; this is not an entry idempotency risk.
                        self.ib.sleep(1)
                    # Widen price to be more aggressive
                    if action == 'BUY':
                        limit_price = round(limit_price + FALLBACK_TICK, 2)
                    else:
                        limit_price = round(max(limit_price - FALLBACK_TICK, 0.01), 2)

                order = LimitOrder(action, qty, limit_price)
                order.tif = 'DAY'
                self.logger.info(
                    f"   Fallback leg {contract.strike}{contract.right} "
                    f"{action} @ ${limit_price:.2f} (chase {chase})"
                )
                trade = self.ib.placeOrder(contract, order)

                start_wait = time_module.time()
                while (time_module.time() - start_wait) < 10:
                    self.ib.sleep(1)
                    if trade.orderStatus.status == 'Filled':
                        leg_filled = True
                        break
                    if trade.orderStatus.status in ['Cancelled', 'Inactive', 'ApiCancelled']:
                        break

                if leg_filled:
                    self.logger.info(f"   ✅ Leg {contract.strike}{contract.right} filled @ ${limit_price:.2f}")
                    break

            if not leg_filled:
                self.logger.error(f"❌ Failed to close leg {contract.strike}{contract.right} after {FALLBACK_CHASE_STEPS + 1} attempts")
                # Cancel any dangling order for this leg
                if trade:
                    try:
                        self.ib.cancelOrder(trade.order)
                    except Exception:
                        pass
                all_filled = False

        return all_filled
    
    def get_position_pnl(self) -> Optional[float]:

        """Get current PnL for active position."""
        if not self.active_position:
            return None
        
        # Get current portfolio PnL from IBKR
        portfolio = self.ib.portfolio()
        
        # POSITION ISOLATION (FIX 9): Only sum PnL for legs belonging to our IC
        # Prevents foreign positions from contaminating PnL and triggering false SL/TP
        if self.active_position.legs:
            ic_conids = {leg['conId'] for leg in self.active_position.legs}
            xsp_pnl = sum(
                item.unrealizedPNL 
                for item in portfolio 
                if item.contract.symbol == 'XSP' and item.contract.conId in ic_conids
            )
        else:
            # No legs available — return 0 to avoid false triggers
            self.logger.warning("⚠️ PnL: No IC legs to reference. Returning $0 (safe default).")
            xsp_pnl = 0.0
        
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
        
        # 1. Take Profit (disabled in hold-to-expiry mode to avoid phantom closes)
        if not self.HOLD_TO_EXPIRY_MODE and pnl >= take_profit_target:
            tp_label = int(round(self.TAKE_PROFIT_PCT * 100))
            return f"TP_{tp_label} [PnL ${pnl:.0f} >= ${take_profit_target:.0f}]"
        
        # 2. Stop Loss
        if pnl <= stop_loss_target:
            return f"SL_{int(self.STOP_LOSS_MULT)}X [PnL ${pnl:.0f} <= ${stop_loss_target:.0f}]"
        
        # 3. EOD Force Close (time of day cutoff)
        now_time = datetime.now().time()
        if now_time >= self.FORCE_CLOSE_TIME:
            if self.HOLD_TO_EXPIRY_MODE:
                # In hold-to-expiry mode, avoid closing nearly worthless winners.
                # PnL formula inversion:
                # current_spread_cost = entry_credit - (pnl / (100 * qty))
                qty = max(self.active_position.qty, 1)
                current_spread_cost = self.active_position.entry_credit - (pnl / (100 * qty))
                if current_spread_cost <= self.TICK_SIZE:
                    return None
            return f"EOD_TIME [Time {now_time.strftime('%H:%M')} >= {self.FORCE_CLOSE_TIME}]"
        
        return None
    
    def close_position(
        self,
        reason: str,
        max_spread_val: float = 0.0,
        rv_duration: float = 0.0,
        max_adverse_excursion: float = 0.0,
        max_favorable_excursion: float = 0.0,
    ) -> bool:
        """
        Close the active position with blocking confirmation.
        
        Args:
            reason: Reason for closing
            max_spread_val: Max observed spread value (for journal)
            rv_duration: Realized Volatility annualized (for journal)
            max_adverse_excursion: Lowest PnL observed during trade
            max_favorable_excursion: Highest PnL observed during trade
            
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
                    rv_duration=rv_duration,
                    max_adverse_excursion=max_adverse_excursion,
                    max_favorable_excursion=max_favorable_excursion,
                )
            self._update_kill_switch_trackers(final_pnl)
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
        market_close = dt_time(16, 0)
        
        max_spread_val = 0.0
        min_pnl_observed = 0.0  # Track MAE (Max Adverse Excursion)
        max_pnl_observed = 0.0  # Track MFE (Max Favorable Excursion)
        last_known_pnl = 0.0
        spot_prices = [] # NEW: for RV calc
        
        while self.active_position:
            now = datetime.now()

            # FIX 11: Hard expiry deadline BEFORE any IBKR-dependent fetches.
            if self.HOLD_TO_EXPIRY_MODE and now.time() >= market_close:
                final_pnl = last_known_pnl
                try:
                    latest_pnl = self.get_position_pnl()
                    if latest_pnl is not None:
                        final_pnl = latest_pnl
                except Exception as e:
                    self.logger.warning(
                        f"⚠️ Could not refresh PnL at expiry; using last known PnL=${last_known_pnl:.2f}. Error: {e}"
                    )
                self._finalize_hold_to_expiry(
                    final_pnl,
                    max_spread_val,
                    min_pnl_observed,
                    max_pnl_observed,
                )
                break

            try:
                # Update Max Spread Value (Tail Risk)
                # PnL = (EntryCredit - CurrentSpread) * 100 * Qty
                # CurrentSpread = EntryCredit - (PnL / (100 * self.active_position.qty))
                pnl = self.get_position_pnl()
                if pnl is not None:
                    last_known_pnl = pnl
                    current_spread_cost = self.active_position.entry_credit - (pnl / (100 * self.active_position.qty))
                    if current_spread_cost > max_spread_val:
                        max_spread_val = current_spread_cost
                    if pnl < min_pnl_observed:
                        min_pnl_observed = pnl
                    if pnl > max_pnl_observed:
                        max_pnl_observed = pnl
                else:
                    pnl = last_known_pnl

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

                # Danger zone detection
                min_distance = min(dist_put, dist_call)

                # Calculate percentages
                max_profit = self.active_position.max_profit
                pnl_val = pnl if pnl is not None else 0.0
                tp_target = max_profit * self.TAKE_PROFIT_PCT
                tp_pct = (pnl_val / tp_target * 100) if tp_target > 0 else 0
                sl_target = -self.active_position.entry_credit * 100 * self.STOP_LOSS_MULT
                sl_pct = (pnl_val / sl_target * 100) if sl_target < 0 else 0
                max_pct = (pnl_val / max_profit * 100) if max_profit > 0 else 0

                # Time held
                now = datetime.now()
                hold_time = (now - self.active_position.entry_time).total_seconds() / 60

                danger = "⚠️ DANGER" if min_distance < 5 else ""
                now_str = now.strftime("%H:%M:%S")
                base_output = (
                    f"[{now_str}] XSP: {spot:.2f} | VIX: {vix:.1f} | "
                    f"PnL: ${pnl_val:.2f} ({max_pct:.0f}%Max) | TP:{tp_pct:.0f}% SL:{sl_pct:.0f}% | "
                    f"{short_put:.0f}P({dist_put:+.0f}) {short_call:.0f}C({dist_call:+.0f}) | "
                    f"Hold:{hold_time:.0f}m | MaxSprd:{max_spread_val:.2f}"
                )
                print(f"{danger} {base_output}" if danger else base_output, flush=True)

                exit_reason = self.check_exit_conditions()
                if exit_reason:
                    # Calculate Realized Volatility (RV)
                    rv_duration = 0.0
                    if len(spot_prices) > 2:
                        try:
                            prices_arr = np.array(spot_prices)
                            log_returns = np.log(prices_arr[1:] / prices_arr[:-1])
                            std_dev = np.std(log_returns)
                            # Annualize using actual sampling interval
                            annualization_factor = np.sqrt(252 * 6.5 * 3600 / check_interval)
                            rv_duration = std_dev * annualization_factor
                        except Exception as e:
                            print(f"⚠️ RV Calc Error: {e}")

                    success = self.close_position(
                        exit_reason,
                        max_spread_val,
                        rv_duration,
                        min_pnl_observed,
                        max_pnl_observed,
                    )
                    if success:
                        break
                    print("⚠️ Exit failed. Retrying monitoring loop...")
                    try:
                        self.ib.sleep(1)
                    except Exception:
                        time_module.sleep(1)

            except Exception as e:
                self.logger.error(f"⚠️ Monitor loop exception: {e}")
                now = datetime.now()
                if self.HOLD_TO_EXPIRY_MODE and now.time() >= market_close:
                    self.logger.warning(
                        "⚠️ Exception at/after market close. Forcing hold-to-expiry cleanup with last known PnL."
                    )
                    self._finalize_hold_to_expiry(
                        last_known_pnl,
                        max_spread_val,
                        min_pnl_observed,
                        max_pnl_observed,
                    )
                    break
                try:
                    self.ib.sleep(1)
                except Exception:
                    time_module.sleep(1)
                continue

            try:
                self.ib.sleep(check_interval)
            except Exception as e:
                self.logger.warning(f"⚠️ Monitor sleep interrupted: {e}")
                now = datetime.now()
                if self.HOLD_TO_EXPIRY_MODE and now.time() >= market_close:
                    self._finalize_hold_to_expiry(
                        last_known_pnl,
                        max_spread_val,
                        min_pnl_observed,
                        max_pnl_observed,
                    )
                    break
                time_module.sleep(min(check_interval, 1.0))

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

