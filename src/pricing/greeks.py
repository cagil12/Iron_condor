import numpy as np
from scipy.stats import norm
from dataclasses import dataclass
from typing import Optional

# Constants
N = norm.pdf
N_prime = norm.pdf
CDF = norm.cdf

def d1_d2(S, K, T, r, sigma):
    """Calculate d1 and d2 for Black-Scholes."""
    if T <= 0 or sigma <= 0:
        return 0.0, 0.0
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2

def calculate_greeks(S: float, K: float, T: float, r: float, sigma: float, option_type: str):
    """
    Returns dictionary of greeks: delta, gamma, theta, vega, rho.
    option_type: 'C' or 'P'
    """
    greeks = {
        'delta': 0.0,
        'gamma': 0.0,
        'theta': 0.0,
        'vega': 0.0,
        'rho': 0.0,
        'price': 0.0
    }
    
    if T <= 0:
        # Expiry logic
        intrinsic = max(0, S - K) if option_type == 'C' else max(0, K - S)
        greeks['price'] = intrinsic
        if option_type == 'C':
            greeks['delta'] = 1.0 if S > K else 0.0
        else:
            greeks['delta'] = -1.0 if S < K else 0.0
        return greeks

    d1, d2 = d1_d2(S, K, T, r, sigma)

    # Common factors
    sqrt_T = np.sqrt(T)
    
    if option_type == 'C':
        greeks['price'] = S * CDF(d1) - K * np.exp(-r * T) * CDF(d2)
        greeks['delta'] = CDF(d1)
        greeks['rho'] = K * T * np.exp(-r * T) * CDF(d2)
        greeks['theta'] = (- (S * N(d1) * sigma) / (2 * sqrt_T) 
                           - r * K * np.exp(-r * T) * CDF(d2))
    else:
        greeks['price'] = K * np.exp(-r * T) * CDF(-d2) - S * CDF(-d1)
        greeks['delta'] = CDF(d1) - 1
        greeks['rho'] = -K * T * np.exp(-r * T) * CDF(-d2)
        greeks['theta'] = (- (S * N(d1) * sigma) / (2 * sqrt_T) 
                           + r * K * np.exp(-r * T) * CDF(-d2))

    greeks['gamma'] = N(d1) / (S * sigma * sqrt_T)
    greeks['vega'] = S * sqrt_T * N(d1) # Usually /100 in practice but keeping raw here
    
    return greeks
