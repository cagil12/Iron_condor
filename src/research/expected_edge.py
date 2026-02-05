import numpy as np
import scipy.stats as stats
from typing import Dict

class EdgeCalculator:
    """
    Calculates Probability of Profit (POP) and Expected Value (EV)
    using Black-Scholes assumptions (Log-normal distribution).
    """

    @staticmethod
    def calculate_metrics(
        spot_price: float,
        short_put_strike: float,
        short_call_strike: float,
        credit_received: float,
        wing_width: float,
        iv: float,
        days_to_expiration: float
    ) -> Dict[str, float]:
        """
        Calculate POP and EV for an Iron Condor.

        Args:
            spot_price: Current underlying price
            short_put_strike: Short Put Strike
            short_call_strike: Short Call Strike
            credit_received: Net credit (e.g., 0.15)
            wing_width: Distance between long and short legs (used for Max Loss)
            iv: Implied Volatility (annualized, e.g., 0.15 for 15%)
            days_to_expiration: Time to expiry in days (e.g., 0.5 for half day)

        Returns:
            Dict containing 'pop', 'ev', 'edge_ratio'
        """
        # Time in years
        t = days_to_expiration / 365.0
        
        # Avoid division by zero
        if t <= 0:
            t = 0.0001 # Small epsilon
            
        # Volatility scaled to time period
        sigma_t = iv * np.sqrt(t)
        
        if sigma_t == 0:
            return {'pop': 0.0, 'ev': 0.0, 'edge_ratio': 0.0}

        # Breakevens
        be_lower = short_put_strike - credit_received
        be_upper = short_call_strike + credit_received
        
        # Calculate Log-returns required to hit breakevens
        # ln(K / S)
        # Standardize: Z = (ln(K/S) - (r - 0.5*sigma^2)*t) / sigma_t
        # Assuming r=0 (risk free rate negligible for 0DTE short term)
        
        r = 0.0
        drift = (r - 0.5 * iv**2) * t
        
        # Probability of expiring BELOW lower breakeven
        z_lower = (np.log(be_lower / spot_price) - drift) / sigma_t
        prob_below_lower = stats.norm.cdf(z_lower)
        
        # Probability of expiring ABOVE upper breakeven
        z_upper = (np.log(be_upper / spot_price) - drift) / sigma_t
        prob_above_upper = 1 - stats.norm.cdf(z_upper)
        
        # POP: Probability of staying BETWEEN breakevens
        # P(Profit > 0)
        prob_profit = 1 - (prob_below_lower + prob_above_upper)
        
        # Max Loss (Risk per contract)
        # Max Loss = Width - Credit
        max_loss = wing_width - credit_received
        if max_loss <= 0:
             max_loss = 0.0001 # Should not happen in defined risk unless credit > width
        
        # Expected Value (Simplified)
        # EV = (Credit * POP) - (Max_Loss * (1 - POP))
        ev = (credit_received * prob_profit) - (max_loss * (1 - prob_profit))
        
        # ROI / Edge Ratio
        # Edge Ratio = EV / Risk
        edge_ratio = ev / max_loss
        
        return {
            'pop': round(prob_profit * 100, 2), # Percentage
            'ev': round(ev, 4),
            'edge_ratio': round(edge_ratio, 4),
            'max_loss': round(max_loss, 2),
            'max_profit': round(credit_received, 2),
            'breakeven_lower': round(be_lower, 2),
            'breakeven_upper': round(be_upper, 2)
        }

if __name__ == "__main__":
    # Test Case
    print("--- Edge Calculator Validation ---")
    
    # Example: XSP 500, IV 15%, Credit 0.15, Width 1.0, 1 Day to Exp
    spot = 500.0
    iv = 0.15 
    credit = 0.15
    width = 1.0 
    dte = 1.0
    
    # Strikes: 10 delta approx? Let's say 490/510
    s_put = 495.0
    s_call = 505.0
    
    res = EdgeCalculator.calculate_metrics(
        spot, s_put, s_call, credit, width, iv, dte
    )
    
    print(f"INPUTS: Spot={spot}, Strikes={s_put}/{s_call}, Credit={credit}, Width={width}, IV={iv}, DTE={dte}")
    print(f"RESULTS: {res}")
    
    # Expected behavior:
    # High POP (wide strikes), Positive EV if credit is good relative to risk
    if res['pop'] > 50:
        print("✅ POP calculation seems reasonable (>50% for OTM)")
    else:
        print("⚠️ POP is low, check logic.")
