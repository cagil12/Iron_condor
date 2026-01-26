import numpy as np
from scipy.stats import norm
from .greeks import d1_d2, CDF

def bsm_price(option_type: str, S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0:
        return max(0, S - K) if option_type == 'C' else max(0, K - S)
    
    d1, d2 = d1_d2(S, K, T, r, sigma)
    
    if option_type == 'C':
        return S * CDF(d1) - K * np.exp(-r * T) * CDF(d2)
    else:
        return K * np.exp(-r * T) * CDF(-d2) - S * CDF(-d1)

def vega_bsm(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0: return 0.0
    d1, _ = d1_d2(S, K, T, r, sigma)
    return S * np.sqrt(T) * norm.pdf(d1)

def implied_volatility(
    price: float, 
    S: float, 
    K: float, 
    T: float, 
    r: float, 
    option_type: str,
    tol: float = 1e-5,
    max_iter: int = 100
) -> float:
    """
    Calculate IV using Newton-Raphson.
    """
    sigma = 0.5 # Initial guess
    
    for i in range(max_iter):
        p = bsm_price(option_type, S, K, T, r, sigma)
        diff = price - p
        
        if abs(diff) < tol:
            return sigma
            
        v = vega_bsm(S, K, T, r, sigma)
        
        if v == 0:
            break # Avoid division by zero
            
        sigma = sigma + diff / v
        
    return sigma  # Return best guess even if not converged
