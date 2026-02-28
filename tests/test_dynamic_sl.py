import numpy as np
import pandas as pd
import pytest

from src.research import synthetic_backtest as sb


def _tte_from_hours(hours: float) -> float:
    return hours / (sb.TRADING_DAYS_PER_YEAR * sb.TRADING_HOURS_PER_DAY)


def _base_cfg() -> dict:
    return {
        "dates": {"start": "2025-01-01", "end": "2025-12-31"},
        "default": {
            "wing_width": 2.0,
            "delta_target": 0.10,
            "min_credit": 0.0,
            "bid_ask_haircut": 0.0,
            "tp_pct": 1.0,  # disable TP for fixed-SL equivalence checks
            "sl_mult": 2.0,
            "entry_hour": 10.0,
            "multiplier": 100,
        },
        "entry_filters": {"min_vix": 0.0, "max_vix": 100.0},
        "strike_selection": {"strike_step": 1.0},
        "credit_filters": {"max_risk_reward": 1000.0},
        "timing": {"risk_free_rate": 0.05, "vix_source": "vix_open"},
        "pricing": {"use_bs_repricing": True},
        "dynamic_sl": {
            "enabled": True,
            "sl_floor": 2.0,
            "sl_ceiling": 2.0,
            "alpha": 0.0,
            "beta": 0.0,
        },
        "commissions": {
            "pricing_plan": "fixed",
            "fixed": {"per_contract": 0.65},
            "legs_per_ic": 4,
            "model": "conditional",
            "flat_amount": 5.0,
        },
        "iv_scaling_factor": 1.0,
    }


def _one_row(open_px: float, high_px: float, low_px: float, close_px: float, vix_open: float = 25.0) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2025-01-06"),
                "spx_open": open_px,
                "spx_high": high_px,
                "spx_low": low_px,
                "spx_close": close_px,
                "vix_open": vix_open,
                "vix_high": vix_open + 1.0,
                "vix_low": vix_open - 1.0,
                "vix_close": vix_open,
            }
        ]
    )


def test_calc_dynamic_sl_range():
    p = np.array([0.0, 0.2, 0.5, 0.8, 1.0])
    t = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    out = sb.calc_dynamic_sl(p, t, sl_floor=1.5, sl_ceiling=4.0, alpha=1.0, beta=0.5)
    assert np.all(np.isfinite(out))
    assert np.all(out >= 1.5 - 1e-12)
    assert np.all(out <= 4.0 + 1e-12)


def test_calc_dynamic_sl_monotonic():
    p = np.array([0.2, 0.5, 0.8])
    t = np.ones_like(p)
    out = sb.calc_dynamic_sl(p, t, sl_floor=1.5, sl_ceiling=4.0, alpha=1.0, beta=0.5)
    assert out[0] < out[1] < out[2]


def test_calc_P_OTM_far_otm():
    p_min, p_put, p_call = sb.calc_P_OTM_vectorized(
        np.array([100.0]),
        np.array([90.0]),
        np.array([110.0]),
        _tte_from_hours(3.0),
        np.array([0.20]),
        0.0,
    )
    assert p_put[0] > 0.90
    assert p_call[0] > 0.90
    assert p_min[0] > 0.90


def test_calc_P_OTM_near_strike():
    p_min, p_put, p_call = sb.calc_P_OTM_vectorized(
        np.array([90.05]),
        np.array([90.0]),
        np.array([110.0]),
        _tte_from_hours(3.0),
        np.array([0.25]),
        0.0,
    )
    assert 0.50 <= p_put[0] <= 0.60
    assert p_call[0] > 0.95
    assert p_min[0] == pytest.approx(p_put[0], abs=1e-12)


def test_calc_P_OTM_tte_guard():
    tiny_t = sb.MIN_BS_REPRICE_TTE_YEARS / 2.0
    p_min, p_put, p_call = sb.calc_P_OTM_vectorized(
        np.array([100.0]),
        np.array([90.0]),
        np.array([110.0]),
        tiny_t,
        np.array([0.20]),
        0.0,
    )
    assert np.isnan(p_min[0])
    assert np.isnan(p_put[0])
    assert np.isnan(p_call[0])


def test_dynamic_sl_loss_guard():
    # Force threshold below credit so a positive-PnL mark would trigger without the loss guard.
    n = 1
    out = sb._build_dynamic_sl_annotations_and_pnl(
        skipped=np.array([False]),
        S_open=np.array([90.2]),
        high=np.array([90.2]),
        low=np.array([90.2]),
        close=np.array([90.2]),
        sigma=np.array([0.25]),
        short_put=np.array([90.0]),
        long_put=np.array([88.0]),
        short_call=np.array([110.0]),
        long_call=np.array([112.0]),
        entry_credit=np.array([0.30]),
        settlement_pre_commission=np.array([30.0]),
        T_entry=_tte_from_hours(3.0),
        r=0.0,
        xsp_multiplier=100.0,
        use_bs_repricing=True,
        sl_floor=0.50,
        sl_ceiling=0.50,
        alpha=1.0,
        beta=1.0,
    )
    assert out["dynamic_sl_triggered"].shape == (n,)
    assert bool(out["dynamic_sl_triggered"][0]) is False


def test_fixed_sl_equivalence():
    # Constant 2x dynamic SL should match tp50_sl_capped SL trigger condition when TP is disabled.
    cfg = _base_cfg()
    row = _one_row(open_px=5000.0, high_px=5065.0, low_px=4940.0, close_px=5030.0, vix_open=30.0)
    out = sb._simulate_dataframe(row, cfg).iloc[0]
    assert not bool(out["skipped"]), f"unexpected skip: {out['skip_reason']}"
    assert bool(out["dynamic_sl_triggered"]) == bool(out["tp50_sl_capped_sl_trigger"])
