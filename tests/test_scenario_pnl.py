import itertools

import numpy as np
import pandas as pd
import pytest

from src.research import synthetic_backtest as sb


SHORT_PUT = 4950.0
SHORT_CALL = 5050.0
WING_WIDTH = 2.0
LONG_PUT = SHORT_PUT - WING_WIDTH
LONG_CALL = SHORT_CALL + WING_WIDTH

ENTRY_CREDIT_PER_SHARE = 0.50
ENTRY_CREDIT_USD = ENTRY_CREDIT_PER_SHARE * 100.0
MAX_LOSS_USD = (WING_WIDTH - ENTRY_CREDIT_PER_SHARE) * 100.0

OPEN_COMMISSION = 2.60
ROUND_TRIP_COMMISSION = 5.20


def _base_config(tp_pct: float = 0.50, sl_mult: float = 3.0, min_credit: float = 0.0) -> dict:
    return {
        "dates": {"start": "2025-01-01", "end": "2025-12-31"},
        "default": {
            "wing_width": WING_WIDTH,
            "delta_target": 0.10,
            "min_credit": min_credit,
            "bid_ask_haircut": 0.0,
            "tp_pct": tp_pct,
            "sl_mult": sl_mult,
            "entry_hour": 10.0,
            "multiplier": 100,
        },
        "entry_filters": {"min_vix": 0.0, "max_vix": 100.0},
        "strike_selection": {"strike_step": 1.0},
        "credit_filters": {"max_risk_reward": 1000.0},
        "timing": {"risk_free_rate": 0.05, "vix_source": "vix_open"},
        "commissions": {
            "pricing_plan": "fixed",
            "fixed": {"per_contract": 0.65},
            "legs_per_ic": 4,
            "model": "conditional",
            "flat_amount": 5.0,
        },
        "iv_scaling_factor": 1.0,
    }


def _one_row(open_px: float, high_px: float, low_px: float, close_px: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2025-01-06"),
                "spx_open": open_px,
                "spx_high": high_px,
                "spx_low": low_px,
                "spx_close": close_px,
                "vix_open": 20.0,
                "vix_high": 21.0,
                "vix_low": 19.0,
                "vix_close": 20.0,
            }
        ]
    )


def _run_one_day(open_px: float, high_px: float, low_px: float, close_px: float, tp_pct: float = 0.50, sl_mult: float = 3.0):
    cfg = _base_config(tp_pct=tp_pct, sl_mult=sl_mult)
    out = sb._simulate_dataframe(_one_row(open_px, high_px, low_px, close_px), cfg).iloc[0]
    assert not bool(out["skipped"]), f"unexpected skip: {out['skip_reason']}"
    return out


@pytest.fixture
def fixed_strikes_and_credit(monkeypatch):
    def fake_delta_to_strike(S, T, r, sigma, target_delta, option_type, strike_step):
        S = np.asarray(S, dtype=float)
        strike = SHORT_PUT if option_type.lower() == "put" else SHORT_CALL
        return np.full(S.shape, strike, dtype=float)

    def fake_bs_price(S, K, T, r, sigma, option_type):
        K = np.asarray(K, dtype=float)
        out = np.full(K.shape, np.nan, dtype=float)
        opt = option_type.lower()

        if opt == "put":
            out[np.isclose(K, SHORT_PUT)] = 1.00
            out[np.isclose(K, LONG_PUT)] = 0.75
        elif opt == "call":
            out[np.isclose(K, SHORT_CALL)] = 1.00
            out[np.isclose(K, LONG_CALL)] = 0.75
        else:
            raise AssertionError(f"Unexpected option_type={option_type}")

        if np.isnan(out).any():
            raise AssertionError(f"Unexpected strike in pricing patch: {K[np.isnan(out)]}")
        return out

    monkeypatch.setattr(sb, "_delta_to_strike_vectorized", fake_delta_to_strike)
    monkeypatch.setattr(sb, "_bs_price_vectorized", fake_bs_price)


def test_clean_win_all_scenarios(fixed_strikes_and_credit):
    out = _run_one_day(open_px=5000.0, high_px=5010.0, low_px=4990.0, close_px=5005.0, tp_pct=0.50, sl_mult=3.0)

    assert not bool(out["put_breached"])
    assert not bool(out["call_breached"])
    assert out["hold_to_expiry_pre_commission"] == pytest.approx(ENTRY_CREDIT_USD, abs=1e-9)
    assert out["outcome_hold_to_expiry"] == pytest.approx(ENTRY_CREDIT_USD - OPEN_COMMISSION, abs=1e-9)
    assert out["outcome_worst_case"] == pytest.approx(ENTRY_CREDIT_USD - ROUND_TRIP_COMMISSION, abs=1e-9)
    assert out["outcome_tp50_or_expiry"] == pytest.approx((0.50 * ENTRY_CREDIT_USD) - ROUND_TRIP_COMMISSION, abs=1e-9)
    assert out["outcome_tp50_sl_capped"] == pytest.approx((0.50 * ENTRY_CREDIT_USD) - ROUND_TRIP_COMMISSION, abs=1e-9)


def test_put_breach_at_settlement(fixed_strikes_and_credit):
    out = _run_one_day(open_px=5000.0, high_px=5005.0, low_px=4940.0, close_px=4945.0, tp_pct=0.50, sl_mult=3.0)

    assert bool(out["put_breached"])
    assert not bool(out["call_breached"])
    assert out["hold_to_expiry_pre_commission"] == pytest.approx(-MAX_LOSS_USD, abs=1e-9)
    assert out["outcome_hold_to_expiry"] == pytest.approx(-MAX_LOSS_USD - ROUND_TRIP_COMMISSION, abs=1e-9)
    assert out["outcome_worst_case"] == pytest.approx(-MAX_LOSS_USD - ROUND_TRIP_COMMISSION, abs=1e-9)
    assert out["outcome_tp50_or_expiry"] == pytest.approx(-MAX_LOSS_USD - ROUND_TRIP_COMMISSION, abs=1e-9)
    assert out["outcome_tp50_sl_capped"] == pytest.approx(-MAX_LOSS_USD - ROUND_TRIP_COMMISSION, abs=1e-9)


def test_call_breach_at_settlement(fixed_strikes_and_credit):
    out = _run_one_day(open_px=5000.0, high_px=5060.0, low_px=4995.0, close_px=5055.0, tp_pct=0.50, sl_mult=3.0)

    assert not bool(out["put_breached"])
    assert bool(out["call_breached"])
    assert out["hold_to_expiry_pre_commission"] == pytest.approx(-MAX_LOSS_USD, abs=1e-9)
    assert out["outcome_hold_to_expiry"] == pytest.approx(-MAX_LOSS_USD - ROUND_TRIP_COMMISSION, abs=1e-9)
    assert out["outcome_worst_case"] == pytest.approx(-MAX_LOSS_USD - ROUND_TRIP_COMMISSION, abs=1e-9)
    assert out["outcome_tp50_or_expiry"] == pytest.approx(-MAX_LOSS_USD - ROUND_TRIP_COMMISSION, abs=1e-9)
    assert out["outcome_tp50_sl_capped"] == pytest.approx(-MAX_LOSS_USD - ROUND_TRIP_COMMISSION, abs=1e-9)


def test_both_strikes_touched_same_bar(fixed_strikes_and_credit):
    out = _run_one_day(open_px=5000.0, high_px=5055.0, low_px=4945.0, close_px=5000.0, tp_pct=0.50, sl_mult=3.0)

    assert bool(out["put_breached"])
    assert bool(out["call_breached"])
    assert out["outcome_hold_to_expiry"] == pytest.approx(ENTRY_CREDIT_USD - OPEN_COMMISSION, abs=1e-9)
    assert out["outcome_worst_case"] == pytest.approx(-MAX_LOSS_USD - ROUND_TRIP_COMMISSION, abs=1e-9)
    assert out["outcome_tp50_or_expiry"] == pytest.approx(ENTRY_CREDIT_USD - ROUND_TRIP_COMMISSION, abs=1e-9)
    assert out["outcome_tp50_sl_capped"] == pytest.approx(ENTRY_CREDIT_USD - ROUND_TRIP_COMMISSION, abs=1e-9)


def test_sl_capped_vs_max_loss(fixed_strikes_and_credit):
    out = _run_one_day(open_px=5000.0, high_px=5005.0, low_px=4920.0, close_px=4925.0, tp_pct=0.50, sl_mult=2.0)

    sl_cap_usd = 2.0 * ENTRY_CREDIT_USD
    assert out["outcome_hold_to_expiry"] == pytest.approx(-MAX_LOSS_USD - ROUND_TRIP_COMMISSION, abs=1e-9)
    assert out["outcome_worst_case"] == pytest.approx(-MAX_LOSS_USD - ROUND_TRIP_COMMISSION, abs=1e-9)
    assert out["tp50_sl_capped_pre_commission"] == pytest.approx(-sl_cap_usd, abs=1e-9)
    assert out["outcome_tp50_sl_capped"] == pytest.approx(-sl_cap_usd - ROUND_TRIP_COMMISSION, abs=1e-9)


def test_tp_detection_near_strike_no_touch(fixed_strikes_and_credit):
    out = _run_one_day(open_px=5000.0, high_px=5049.9, low_px=4950.1, close_px=5000.0, tp_pct=0.50, sl_mult=3.0)

    assert not bool(out["put_breached"])
    assert not bool(out["call_breached"])
    assert out["tp50_or_expiry_pre_commission"] == pytest.approx(0.50 * ENTRY_CREDIT_USD, abs=1e-9)
    assert out["outcome_tp50_or_expiry"] == pytest.approx((0.50 * ENTRY_CREDIT_USD) - ROUND_TRIP_COMMISSION, abs=1e-9)


def test_exact_strike_touch_counts_as_touch(fixed_strikes_and_credit):
    out = _run_one_day(open_px=5000.0, high_px=5040.0, low_px=4950.0, close_px=5000.0, tp_pct=0.50, sl_mult=3.0)

    assert bool(out["put_breached"])
    assert out["tp50_or_expiry_pre_commission"] == pytest.approx(ENTRY_CREDIT_USD, abs=1e-9)
    assert out["outcome_tp50_or_expiry"] == pytest.approx(ENTRY_CREDIT_USD - ROUND_TRIP_COMMISSION, abs=1e-9)


def test_zero_credit_is_skipped(fixed_strikes_and_credit, monkeypatch):
    def zero_credit_bs(S, K, T, r, sigma, option_type):
        K = np.asarray(K, dtype=float)
        return np.full(K.shape, 1.0, dtype=float)

    monkeypatch.setattr(sb, "_bs_price_vectorized", zero_credit_bs)
    cfg = _base_config(tp_pct=0.50, sl_mult=3.0, min_credit=0.0)
    out = sb._simulate_dataframe(_one_row(5000.0, 5010.0, 4990.0, 5005.0), cfg).iloc[0]

    assert bool(out["skipped"])
    assert out["skip_reason"] == "invalid_credit"


def test_expiry_at_short_strike_boundary(fixed_strikes_and_credit):
    out = _run_one_day(open_px=5000.0, high_px=5005.0, low_px=4951.0, close_px=SHORT_PUT, tp_pct=0.50, sl_mult=3.0)

    assert not bool(out["put_breached"])
    assert out["hold_to_expiry_pre_commission"] == pytest.approx(ENTRY_CREDIT_USD, abs=1e-9)
    assert out["outcome_hold_to_expiry"] == pytest.approx(ENTRY_CREDIT_USD - OPEN_COMMISSION, abs=1e-9)


def test_hold_to_expiry_win_commission(fixed_strikes_and_credit):
    out = _run_one_day(open_px=5000.0, high_px=5010.0, low_px=4990.0, close_px=5001.0, tp_pct=0.50, sl_mult=3.0)
    assert out["commission_hold_to_expiry"] == pytest.approx(OPEN_COMMISSION, abs=1e-9)


def test_hold_to_expiry_loss_commission(fixed_strikes_and_credit):
    out = _run_one_day(open_px=5000.0, high_px=5005.0, low_px=4940.0, close_px=4945.0, tp_pct=0.50, sl_mult=3.0)
    assert out["commission_hold_to_expiry"] == pytest.approx(ROUND_TRIP_COMMISSION, abs=1e-9)


def test_tp50_always_roundtrip_commission(fixed_strikes_and_credit):
    win = _run_one_day(open_px=5000.0, high_px=5010.0, low_px=4990.0, close_px=5002.0, tp_pct=0.50, sl_mult=3.0)
    loss = _run_one_day(open_px=5000.0, high_px=5060.0, low_px=4995.0, close_px=5055.0, tp_pct=0.50, sl_mult=3.0)

    for row in (win, loss):
        assert row["commission_tp50_or_expiry"] == pytest.approx(ROUND_TRIP_COMMISSION, abs=1e-9)
        assert row["commission_tp50_sl_capped"] == pytest.approx(ROUND_TRIP_COMMISSION, abs=1e-9)


def test_hold_to_expiry_geq_worst_case_deterministic_grid(fixed_strikes_and_credit):
    highs = [5005.0, 5049.0, 5050.0, 5055.0]
    lows = [4995.0, 4951.0, 4950.0, 4945.0]
    closes = [4945.0, 4950.0, 5000.0, 5050.0, 5055.0]

    checked = 0
    for low_px, high_px, close_px in itertools.product(lows, highs, closes):
        if high_px < low_px:
            continue
        if not (low_px <= close_px <= high_px):
            continue
        out = _run_one_day(open_px=5000.0, high_px=high_px, low_px=low_px, close_px=close_px, tp_pct=0.50, sl_mult=3.0)
        assert out["outcome_hold_to_expiry"] >= out["outcome_worst_case"] - 1e-9
        checked += 1

    assert checked >= 20


def test_tp50_sl_capped_not_strictly_between_bounds(fixed_strikes_and_credit):
    untouched = _run_one_day(open_px=5000.0, high_px=5010.0, low_px=4990.0, close_px=5003.0, tp_pct=0.50, sl_mult=2.0)
    touched_loss = _run_one_day(open_px=5000.0, high_px=5005.0, low_px=4920.0, close_px=4925.0, tp_pct=0.50, sl_mult=2.0)

    # Untouched winners: TP cap can make tp50_sl_capped worse than worst_case.
    assert untouched["outcome_tp50_sl_capped"] < untouched["outcome_worst_case"]
    # Touched losers: SL cap can make tp50_sl_capped better than hold_to_expiry.
    assert touched_loss["outcome_tp50_sl_capped"] > touched_loss["outcome_hold_to_expiry"]
