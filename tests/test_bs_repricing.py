import numpy as np
import pytest

from src.research import synthetic_backtest as sb


def _tte_hours(hours: float) -> float:
    return hours / (sb.TRADING_DAYS_PER_YEAR * 390.0)


def test_reprice_ic_spread_far_from_strikes_is_near_zero():
    # Sanity: both spreads far OTM should be close to zero cost-to-close.
    spread_cost = sb.reprice_ic_spread(
        S_t=100.0,
        K_short_put=90.0,
        K_long_put=88.0,
        K_short_call=110.0,
        K_long_call=112.0,
        TTE=_tte_hours(3.0),
        sigma=0.20,
        r=0.0,
    )
    assert spread_cost >= 0.0
    assert spread_cost < 0.02


def test_reprice_ic_spread_exceeds_linear_proxy_near_short_strike():
    # Near-short but no touch: linear proxy stays at 0 while BS captures gamma/theta value.
    S_t = 90.5
    short_put = 90.0
    long_put = 88.0
    short_call = 110.0
    long_call = 112.0
    TTE = _tte_hours(3.0)
    sigma = 0.25

    bs_cost = sb.reprice_ic_spread(S_t, short_put, long_put, short_call, long_call, TTE, sigma, 0.0)
    linear_cost = sb._linear_spread_cost_proxy_vectorized(
        np.array([S_t]),
        np.array([short_put]),
        np.array([long_put]),
        np.array([short_call]),
        np.array([long_call]),
    )[0]

    assert linear_cost == pytest.approx(0.0, abs=1e-12)
    assert bs_cost > linear_cost


def test_spread_cost_fallback_activates_below_5_minutes():
    S_t = np.array([90.2, 100.0])
    short_put = np.array([90.0, 95.0])
    long_put = np.array([88.0, 93.0])
    short_call = np.array([110.0, 105.0])
    long_call = np.array([112.0, 107.0])
    sigma = np.array([0.30, 0.20])
    tte_tiny = sb.MIN_BS_REPRICE_TTE_YEARS / 2.0

    linear = sb._linear_spread_cost_proxy_vectorized(S_t, short_put, long_put, short_call, long_call)
    with_fallback = sb._spread_cost_with_fallback_vectorized(
        S_t,
        short_put,
        long_put,
        short_call,
        long_call,
        tte_tiny,
        sigma,
        0.0,
        use_bs_repricing=True,
    )

    np.testing.assert_allclose(with_fallback, linear, atol=0.0, rtol=0.0)
