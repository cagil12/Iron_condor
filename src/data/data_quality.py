import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
from .schema import OptionChain, OptionType

class DataQualityChecker:
    def __init__(self):
        self.issues = []

    def check_chain_integrity(self, chain: OptionChain) -> Dict[str, Any]:
        """
        Validates a single option chain snapshot.
        Checks for:
        - Crossed quotes (ask <= bid)
        - Negative prices
        - Missing ATM strikes (heuristic)
        """
        report = {
            "timestamp": chain.timestamp,
            "total_quotes": len(chain.quotes),
            "crossed_quotes": 0,
            "negative_bids": 0,
            "missing_atm": False
        }
        
        if not chain.quotes:
            self.issues.append(f"Empty chain at {chain.timestamp}")
            return report

        spot = chain.underlying_price
        strikes = chain.strikes
        
        # Check simple ATM range coverage (e.g. within 1% of spot)
        if strikes: # Guard against completely empty strikes
            min_strike = min(strikes)
            max_strike = max(strikes)
            if spot < min_strike or spot > max_strike:
                 report["missing_atm"] = True
                 self.issues.append(f"Spot {spot} outside strike range [{min_strike}, {max_strike}] at {chain.timestamp}")

        for (strike, otype), quote in chain.quotes.items():
            if quote.bid < 0:
                report["negative_bids"] += 1
            
            if quote.ask <= quote.bid:
                 report["crossed_quotes"] += 1
        
        return report

    def generate_report(self) -> pd.DataFrame:
        """Returns a summary of issues found."""
        return pd.DataFrame(self.issues, columns=["issue_description"])
