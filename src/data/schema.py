from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Tuple, Optional, List

class OptionType(str, Enum):
    CALL = 'C'
    PUT = 'P'

@dataclass(frozen=True)
class SpotPrice:
    timestamp: datetime
    price: float
    # Optional OHLC if available
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None

@dataclass
class Quote:
    bid: float
    ask: float
    mid: Optional[float] = None
    
    # Greeks & IV (Vendor provided or calculated)
    implied_vol: Optional[float] = None
    delta: Optional[float] = None
    gamma: Optional[float] = None
    vega: Optional[float] = None
    theta: Optional[float] = None
    
    # Metadata
    volume: Optional[int] = None
    open_interest: Optional[int] = None

    def valid_spread(self) -> bool:
        """Returns True if bid > 0 and ask > bid."""
        return self.bid > 0 and self.ask > self.bid

@dataclass
class OptionChain:
    timestamp: datetime
    underlying_price: float
    expiration: datetime.date
    # Key: (strike, option_type) -> Quote
    quotes: Dict[Tuple[float, OptionType], Quote] = field(default_factory=dict)
    
    def get_quote(self, strike: float, option_type: OptionType) -> Optional[Quote]:
        return self.quotes.get((strike, option_type))

    @property
    def strikes(self) -> List[float]:
        """Returns sorted unique list of strikes in the chain."""
        return sorted(list(set(k[0] for k in self.quotes.keys())))
