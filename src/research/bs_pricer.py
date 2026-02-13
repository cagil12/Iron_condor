"""Black-Scholes utilities for synthetic 0DTE Iron Condor backtesting."""

from __future__ import annotations

from math import exp, isfinite, log, sqrt
from typing import Dict

import numpy as np
from scipy.stats import norm

XSP_MULTIPLIER = 100.0


def _validate_option_type(option_type: str) -> str:
    """Normalize and validate option type token."""
    opt = option_type.lower().strip()
    if opt not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")
    return opt


def _intrinsic_value(S: float, K: float, option_type: str) -> float:
    """Return intrinsic value for a call or put."""
    if option_type == "call":
        return max(0.0, S - K)
    return max(0.0, K - S)


def bs_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> float:
    """
    Black-Scholes price for a European option.

    Notes:
    - For 0DTE use fractional-year T (e.g. ~0.00366 at 10:00 AM entry).
    - If T <= 0 or sigma <= 0, returns intrinsic value.
    """
    opt = _validate_option_type(option_type)
    if S <= 0 or K <= 0:
        raise ValueError("S and K must be positive")
    if T <= 0 or sigma <= 0:
        return _intrinsic_value(S, K, opt)

    sqrt_t = sqrt(T)
    vol_term = sigma * sqrt_t
    if vol_term <= 0:
        return _intrinsic_value(S, K, opt)

    d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / vol_term
    d2 = d1 - vol_term

    if opt == "call":
        price = S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
    else:
        price = K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return float(max(0.0, price))


def bs_delta(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> float:
    """Black-Scholes delta (put deltas are negative)."""
    opt = _validate_option_type(option_type)
    if S <= 0 or K <= 0:
        raise ValueError("S and K must be positive")
    if T <= 0 or sigma <= 0:
        intrinsic = _intrinsic_value(S, K, opt)
        if intrinsic <= 0:
            return 0.0
        return 1.0 if opt == "call" else -1.0

    sqrt_t = sqrt(T)
    vol_term = sigma * sqrt_t
    if vol_term <= 0:
        return 0.0

    d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / vol_term
    if opt == "call":
        return float(norm.cdf(d1))
    return float(norm.cdf(d1) - 1.0)


def find_strike_by_delta(
    S: float,
    T: float,
    r: float,
    sigma: float,
    target_delta: float,
    option_type: str,
    strike_step: float = 1.0,
) -> float:
    """
    Find strike closest to target delta using bisection.

    Search range is S ± 5*sigma*sqrt(T)*S, rounded to strike_step.
    """
    opt = _validate_option_type(option_type)
    if S <= 0:
        raise ValueError("S must be positive")
    if strike_step <= 0:
        raise ValueError("strike_step must be positive")

    if opt == "put" and target_delta >= 0:
        raise ValueError("put target_delta must be negative")
    if opt == "call" and target_delta <= 0:
        raise ValueError("call target_delta must be positive")

    # If option has no time/vol, choose nearest OTM boundary strike.
    if T <= 0 or sigma <= 0 or not isfinite(T) or not isfinite(sigma):
        raw = S + strike_step if opt == "call" else S - strike_step
        return round(raw / strike_step) * strike_step

    sigma_t = sigma * sqrt(T)
    width = max(strike_step, 5.0 * sigma_t * S)
    lo = max(strike_step, S - width)
    hi = max(lo + strike_step, S + width)

    # Delta is monotonic decreasing in K for both call and put.
    def f(k: float) -> float:
        return bs_delta(S, k, T, r, sigma, opt) - target_delta

    f_lo = f(lo)
    f_hi = f(hi)

    if f_lo < 0 and f_hi < 0:
        strike = lo
    elif f_lo > 0 and f_hi > 0:
        strike = hi
    else:
        a, b = lo, hi
        for _ in range(100):
            m = 0.5 * (a + b)
            f_m = f(m)
            if abs(f_m) < 1e-6:
                strike = m
                break
            if f_m > 0:
                a = m
            else:
                b = m
        else:
            strike = 0.5 * (a + b)

    rounded = round(strike / strike_step) * strike_step
    if opt == "call":
        rounded = max(rounded, S + strike_step)
    else:
        rounded = min(rounded, S - strike_step)
    return float(max(strike_step, rounded))


def estimate_ic_credit(
    S: float,
    short_put: float,
    short_call: float,
    wing_width: float,
    T: float,
    r: float,
    sigma: float,
    bid_ask_haircut: float = 0.25,
) -> Dict[str, float]:
    """
    Estimate Iron Condor credit using Black-Scholes prices.

    Net credit is haircut-adjusted to model spread/slippage costs.
    """
    if wing_width <= 0:
        raise ValueError("wing_width must be positive")
    if short_put >= short_call:
        raise ValueError("short_put must be below short_call")
    if not (0 <= bid_ask_haircut < 1):
        raise ValueError("bid_ask_haircut must be in [0, 1)")

    long_put = short_put - wing_width
    long_call = short_call + wing_width

    short_put_price = bs_price(S, short_put, T, r, sigma, "put")
    long_put_price = bs_price(S, long_put, T, r, sigma, "put")
    short_call_price = bs_price(S, short_call, T, r, sigma, "call")
    long_call_price = bs_price(S, long_call, T, r, sigma, "call")

    put_credit_theoretical = short_put_price - long_put_price
    call_credit_theoretical = short_call_price - long_call_price
    theoretical_credit = put_credit_theoretical + call_credit_theoretical

    net_credit = theoretical_credit * (1.0 - bid_ask_haircut)
    gross_credit_usd = net_credit * XSP_MULTIPLIER
    max_loss_usd = max(0.0, (wing_width - net_credit) * XSP_MULTIPLIER)

    risk_reward_ratio = np.inf
    if gross_credit_usd > 0:
        risk_reward_ratio = max_loss_usd / gross_credit_usd

    denom = max_loss_usd + gross_credit_usd
    breakeven_winrate = max_loss_usd / denom if denom > 0 else np.nan

    return {
        "net_credit": float(net_credit),
        "theoretical_credit": float(theoretical_credit),
        "gross_credit_usd": float(gross_credit_usd),
        "max_loss_usd": float(max_loss_usd),
        "risk_reward_ratio": float(risk_reward_ratio),
        "breakeven_winrate": float(breakeven_winrate),
        "short_put_price": float(short_put_price),
        "short_call_price": float(short_call_price),
        "long_put_price": float(long_put_price),
        "long_call_price": float(long_call_price),
        "put_credit_theoretical": float(put_credit_theoretical),
        "call_credit_theoretical": float(call_credit_theoretical),
        "put_credit_net": float(put_credit_theoretical * (1.0 - bid_ask_haircut)),
        "call_credit_net": float(call_credit_theoretical * (1.0 - bid_ask_haircut)),
    }
