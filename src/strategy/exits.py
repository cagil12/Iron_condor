from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List
from ..data.schema import OptionChain, OptionType # Fixed import
# Note: Leg and IronCondorTrade are imported inside functions or referenced dynamically to avoid circular import if possible, 
# but better to put shared types in a separate module if needed. For now assuming weak reference or passed object.

@dataclass
class TradeExit:
    timestamp: datetime
    exit_price: float
    exit_reason: str # 'TP', 'SL', 'TIME', 'FORCE', 'LIQ_FAIL'
    pnl: float
    pnl_pct: float

class ExitManager:
    def __init__(self, config: dict):
        self.tp_pct = config.get('take_profit_pct', 0.50)
        self.sl_multiplier = config.get('stop_loss_mult', 2.5)
        self.max_hold_min = config.get('max_hold_minutes', 45)
        self.force_close_time_str = config.get('force_close_time', "15:45")
        
    def _parse_time(self, t_str):
        return datetime.strptime(t_str, "%H:%M").time()

    def calculate_debit(self, trade, chain: OptionChain) -> Optional[float]:
        total_debit = 0.0
        for leg in trade.legs:
            q = chain.get_quote(leg.strike, leg.option_type)
            if not q:
                return None # Missing quote
            
            # Close Long: Sell at Bid
            # Close Short: Buy at Ask
            price = q.bid if leg.is_long else q.ask
            
            if leg.is_long:
                total_debit -= price # Credit (Cash In)
            else:
                total_debit += price # Debit (Cash Out)
                
        return total_debit # Net cost to close

    def check_exit(self, trade, chain: OptionChain) -> Optional[TradeExit]:
        # 1. Compute current cost to close (Mark to Market)
        current_debit = self.calculate_debit(trade, chain)
        
        if current_debit is None:
            # Policy for missing quotes: Skip or Force Close?
            # User spec: "skip minuto con warning_quotes" or abort. 
            # For simplicity, if quotes missing near end of day, might be dangerous.
            # Returning None implies holding.
            return None
        
        pnl = trade.entry_credit - current_debit
        pnl_pct = pnl / trade.entry_credit
        
        # 2. Check TP
        if pnl_pct >= self.tp_pct:
            return TradeExit(chain.timestamp, current_debit, 'TP', pnl, pnl_pct)
            
        # 3. Check SL
        # SL condition: Debit >= Credit * Multiplier
        # Example: Credit 1.0, Mult 3.0 => Exit if Debit >= 3.0 (Loss 2.0)
        if current_debit >= trade.entry_credit * self.sl_multiplier:
            return TradeExit(chain.timestamp, current_debit, 'SL', pnl, pnl_pct)
            
        # 4. Check Time Hold
        elapsed_min = (chain.timestamp - trade.entry_time).total_seconds() / 60
        if elapsed_min >= self.max_hold_min:
            return TradeExit(chain.timestamp, current_debit, 'TIME', pnl, pnl_pct)
            
        # 5. Check Force Close Time
        # Assuming chain.timestamp is datetime
        force_time = self._parse_time(self.force_close_time_str)
        if chain.timestamp.time() >= force_time:
             return TradeExit(chain.timestamp, current_debit, 'FORCE', pnl, pnl_pct)
             
        return None
