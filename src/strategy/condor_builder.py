from dataclasses import dataclass
from typing import Optional, Tuple, List
from .exits import TradeExit
from ..data.schema import OptionChain, OptionType, Quote
import numpy as np

@dataclass
class Leg:
    option_type: OptionType
    strike: float
    is_long: bool # True if bought, False if sold
    entry_price: float # Bid if sell, Ask if buy (conservative)
    
    @property
    def direction(self) -> int:
        return 1 if self.is_long else -1

@dataclass
class IronCondorTrade:
    entry_time: object # datetime
    legs: List[Leg]
    entry_credit: float
    max_loss: float
    
    # Tracking
    status: str = "OPEN"
    exit_info: Optional[object] = None # TradeExit

    @property
    def short_put(self) -> Leg:
        return next(l for l in self.legs if l.option_type == OptionType.PUT and not l.is_long)
    
    @property
    def check_width(self) -> float:
        # Simplified width check
        sp = self.short_put.strike
        lp = next(l for l in self.legs if l.option_type == OptionType.PUT and l.is_long).strike
        return abs(sp - lp)

class CondorBuilder:
    def __init__(self, config: dict):
        self.target_delta = config.get('target_delta', 0.10)
        self.width = config.get('width', 20)
        self.min_credit = config.get('min_credit', 0.50)
        self.min_ror = config.get('min_ror', 0.05)
        self.max_spread_pct = config.get('max_spread_pct', 0.10)
        self.max_spread_abs = config.get('max_spread_abs', 1.0) # Points

    def find_strike_by_delta(self, chain: OptionChain, otype: OptionType, target_delta_abs: float) -> Optional[float]:
        """
        Find strike closest to target delta.
        If vendor delta is missing, infer delta via IV solver.
        """
        from ..pricing.iv_solver import implied_volatility
        from ..pricing.greeks import calculate_greeks
        
        candidates = []
        spot = chain.underlying_price
        
        # Time to expiry (approximate: assume 0DTE, ~6.5 hours left at 10:00)
        # More precise: calculate from timestamp to 16:00
        # For now, use a fixed estimate of 0.02 years (~7 hours)
        T = 0.02  # ~7 hours in year fraction
        r = 0.05  # Risk-free rate assumption (should be config)
        
        for (strike, ot), quote in chain.quotes.items():
            if ot != otype:
                continue
                
            # If delta is provided by vendor, use it
            if quote.delta is not None:
                candidates.append((strike, quote.delta))
            else:
                # Infer delta via IV solver
                mid = (quote.bid + quote.ask) / 2
                if mid <= 0:
                    continue
                    
                try:
                    iv = implied_volatility(mid, spot, strike, T, r, otype.value)
                    if iv > 0:
                        greeks = calculate_greeks(spot, strike, T, r, iv, otype.value)
                        inferred_delta = greeks['delta']
                        candidates.append((strike, inferred_delta))
                except Exception:
                    continue  # Skip if solver fails
        
        if not candidates:
            return None
            
        # Target: -delta for Put, +delta for Call
        target = -target_delta_abs if otype == OptionType.PUT else target_delta_abs
        
        # Sort by distance to target
        best_strike = min(candidates, key=lambda x: abs(x[1] - target))[0]
        return best_strike


    def find_wing(self, chain: OptionChain, otype: OptionType, short_strike: float, width: float) -> Optional[float]:
        # Long Put <= Short Put - Width
        # Long Call >= Short Call + Width
        
        candidates = [k[0] for k in chain.quotes.keys() if k[1] == otype]
        candidates.sort()
        
        if otype == OptionType.PUT:
            target = short_strike - width
            # Find largest strike <= target
            valid = [k for k in candidates if k <= target]
            if valid: return max(valid) # Closest to short
        else:
            target = short_strike + width
            # Find smallest strike >= target
            valid = [k for k in candidates if k >= target]
            if valid: return min(valid)
            
        return None

    def validate_quote(self, quote: Quote, spot: float) -> bool:
        if not quote.valid_spread():
            return False
            
        spread = quote.ask - quote.bid
        # Check absolute spread
        if spread > self.max_spread_abs:
            return False
        
        # Check relative spread
        mid = (quote.ask + quote.bid) / 2
        if mid > 0:
            if spread / mid > self.max_spread_pct:
                return False
                
        return True

    def build_trade(self, chain: OptionChain) -> Optional[IronCondorTrade]:
        # 1. Select Shorts
        short_put_k = self.find_strike_by_delta(chain, OptionType.PUT, self.target_delta)
        short_call_k = self.find_strike_by_delta(chain, OptionType.CALL, self.target_delta)
        
        if short_put_k is None or short_call_k is None:
            return None # Can't find deltas
            
        # 2. Select Longs
        long_put_k = self.find_wing(chain, OptionType.PUT, short_put_k, self.width)
        long_call_k = self.find_wing(chain, OptionType.CALL, short_call_k, self.width)
        
        if long_put_k is None or long_call_k is None:
            return None # Can't find wings
            
        # 3. Get Quotes and Validate
        legs = []
        specs = [
            (short_put_k, OptionType.PUT, False),
            (long_put_k, OptionType.PUT, True),
            (short_call_k, OptionType.CALL, False),
            (long_call_k, OptionType.CALL, True)
        ]
        
        total_credit = 0.0
        
        for strike, otype, is_long in specs:
            q = chain.get_quote(strike, otype)
            if not q or not self.validate_quote(q, chain.underlying_price):
                return None # Failed validation
            
            # Pricing: Sell at Bid, Buy at Ask
            price = q.ask if is_long else q.bid
            
            leg = Leg(otype, strike, is_long, price)
            legs.append(leg)
            
            if is_long:
                total_credit -= price
            else:
                total_credit += price
                
        # 4. Check Economics
        if total_credit < self.min_credit:
            return None
            
        # Calc Max Loss (assuming symmetric width for simplicity in check, but using real logic)
        # Real width for Put side
        width_put = short_put_k - long_put_k
        width_call = long_call_k - short_call_k
        max_width = max(width_put, width_call) # Risk based on wider side
        
        max_loss = max_width - total_credit
        if max_loss <= 0: return None # Arbitrage? unlikely with realistic data
        
        ror = total_credit / max_loss
        if ror < self.min_ror:
            return None
            
        return IronCondorTrade(
            entry_time=chain.timestamp,
            legs=legs,
            entry_credit=total_credit,
            max_loss=max_loss
        )
