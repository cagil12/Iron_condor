import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date, time
from typing import List, Generator
from ..data.schema import OptionChain, Quote, OptionType
from ..pricing.greeks import calculate_greeks

class MockDataGenerator:
    """
    Generates synthetic intraday data for testing purposes.
    """
    def __init__(self, 
                 start_date: date, 
                 days: int = 1, 
                 spot_start: float = 5000.0,
                 volatility: float = 0.15,
                 risk_free_rate: float = 0.05):
        self.start_date = start_date
        self.days = days
        self.spot_start = spot_start
        self.volatility = volatility
        self.r = risk_free_rate
        
    def generate_day(self, trade_date: date) -> Generator[OptionChain, None, None]:
        # Generate minutes from 09:30 to 16:00
        market_open = datetime.combine(trade_date, time(9, 30))
        market_close = datetime.combine(trade_date, time(16, 0))
        
        current_time = market_open
        current_spot = self.spot_start # Simple random walk
        
        # Strikes: ATM +/- 500 points, every 5 points
        # For simplicity, fixed set of strikes per day
        strikes = np.arange(self.spot_start - 300, self.spot_start + 300, 5)
        
        while current_time <= market_close:
            # Random walk for spot
            shock = np.random.normal(0, self.volatility / np.sqrt(252*390)) * current_spot
            current_spot += shock
            self.spot_start = current_spot # Persist for next step
            
            chain = OptionChain(
                timestamp=current_time,
                underlying_price=current_spot,
                expiration=trade_date # 0DTE
            )
            
            # Time to expiry in years
            # (16:00 - current_time)
            seconds_to_close = (market_close - current_time).total_seconds()
            if seconds_to_close <= 0:
                T = 1e-6 # Avoid zero
            else:
                T = seconds_to_close / (365 * 24 * 3600)
            
            for K in strikes:
                for otype_str in ['C', 'P']:
                    otype = OptionType.CALL if otype_str == 'C' else OptionType.PUT
                    
                    # Fair value
                    greeks = calculate_greeks(current_spot, K, T, self.r, self.volatility, otype_str)
                    fair = greeks['price']
                    
                    # Spread Logic: wider for OTM, tighter for ATM
                    # Mock spread: 0.10 + 0.05 * fair
                    spread = 0.10 + 0.01 * fair
                    bid = max(0.0, fair - spread/2)
                    ask = fair + spread/2
                    
                    q = Quote(
                        bid=bid,
                        ask=ask,
                        mid=(bid+ask)/2,
                        implied_vol=self.volatility,
                        delta=greeks['delta'],
                        gamma=greeks['gamma'],
                        theta=greeks['theta'],
                        vega=greeks['vega']
                    )
                    chain.quotes[(K, otype)] = q
            
            yield chain
            current_time += timedelta(minutes=1)

if __name__ == "__main__":
    # Smoke test
    gen = MockDataGenerator(date(2023, 1, 1)).generate_day(date(2023, 1, 1))
    first_chain = next(gen)
    print(f"Generated chain at {first_chain.timestamp} with Spot={first_chain.underlying_price:.2f}")
    print(f"Quotes count: {len(first_chain.quotes)}")
