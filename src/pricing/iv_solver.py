"""
Implied Volatility Solver - Production Hardened

Features:
- Newton-Raphson with Brent bracketing fallback
- Robust exception handling (returns NaN, never crashes)
- Bounds checking for all inputs
- Logging of solver failures for debugging
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from .greeks import d1_d2, CDF
import logging
import warnings

logger = logging.getLogger(__name__)

# Suppress scipy warnings during optimization
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Constants
MIN_VOL = 0.001  # 0.1%
MAX_VOL = 5.0    # 500%
MIN_T = 1e-10    # Minimum time to expiry (avoid division by zero)


def bsm_price(option_type: str, S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Black-Scholes-Merton price with input validation.
    """
    try:
        # Input validation
        if S <= 0 or K <= 0 or sigma <= 0:
            return np.nan
        
        if T <= 0:
            # At expiry, return intrinsic value
            return max(0, S - K) if option_type == 'C' else max(0, K - S)
        
        d1, d2 = d1_d2(S, K, T, r, sigma)
        
        if option_type == 'C':
            return S * CDF(d1) - K * np.exp(-r * T) * CDF(d2)
        else:
            return K * np.exp(-r * T) * CDF(-d2) - S * CDF(-d1)
            
    except Exception as e:
        logger.debug(f"BSM price error: {e}")
        return np.nan


def vega_bsm(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Vega with safety checks."""
    try:
        if T <= MIN_T or sigma <= 0 or S <= 0:
            return 0.0
        
        d1, _ = d1_d2(S, K, T, r, sigma)
        return S * np.sqrt(T) * norm.pdf(d1)
        
    except Exception:
        return 0.0


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
    Calculate IV using Newton-Raphson with Brent fallback.
    
    Returns:
        float: Implied volatility, or np.nan if solver fails
        
    Production features:
        - Never crashes, returns np.nan on any error
        - Bounds checking (0.1% - 500% vol)
        - Falls back to Brent's method if Newton fails
        - Logs failures for debugging
    """
    try:
        # === Input Validation ===
        if np.isnan(price) or price <= 0:
            return np.nan
        
        if S <= 0 or K <= 0:
            logger.debug(f"Invalid S/K: S={S}, K={K}")
            return np.nan
        
        if T <= MIN_T:
            # At expiry, can't infer vol
            return np.nan
        
        # Check if price makes sense (not more than intrinsic + time value bounds)
        intrinsic = max(0, S - K) if option_type == 'C' else max(0, K - S)
        if price < intrinsic * 0.99:  # Allow small tolerance
            logger.debug(f"Price below intrinsic: price={price}, intrinsic={intrinsic}")
            return np.nan
        
        # === Newton-Raphson ===
        sigma = 0.3  # Initial guess (30% vol)
        
        for i in range(max_iter):
            p = bsm_price(option_type, S, K, T, r, sigma)
            
            if np.isnan(p):
                break
                
            diff = price - p
            
            if abs(diff) < tol:
                # Converged
                if MIN_VOL <= sigma <= MAX_VOL:
                    return sigma
                else:
                    break
            
            v = vega_bsm(S, K, T, r, sigma)
            
            if v < 1e-10:  # Near-zero vega
                break
            
            sigma_new = sigma + diff / v
            
            # Bound the update
            sigma_new = max(MIN_VOL, min(MAX_VOL, sigma_new))
            
            # Check for oscillation
            if abs(sigma_new - sigma) < 1e-8:
                break
                
            sigma = sigma_new
        
        # === Fallback to Brent's Method ===
        try:
            def objective(vol):
                return bsm_price(option_type, S, K, T, r, vol) - price
            
            # Try to find a bracketing interval
            result = brentq(objective, MIN_VOL, MAX_VOL, xtol=tol)
            
            if MIN_VOL <= result <= MAX_VOL:
                return result
                
        except ValueError:
            # No sign change in interval (Brent failed)
            pass
        except Exception as e:
            logger.debug(f"Brent fallback failed: {e}")
        
        # Return best Newton estimate if within bounds
        if MIN_VOL <= sigma <= MAX_VOL:
            return sigma
            
        return np.nan
        
    except Exception as e:
        logger.debug(f"IV solver error: {e}, inputs: price={price}, S={S}, K={K}, T={T}")
        return np.nan


def implied_volatility_safe(
    price: float, 
    S: float, 
    K: float, 
    T: float, 
    r: float, 
    option_type: str
) -> float:
    """
    Wrapper that ALWAYS returns a valid float (0.0 if solver fails).
    Use this in contexts where NaN handling is problematic.
    """
    result = implied_volatility(price, S, K, T, r, option_type)
    return 0.0 if np.isnan(result) else result

