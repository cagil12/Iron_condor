import math

from src.strategy.execution import (
    calc_current_spread_cost_from_pnl,
    calc_sl_pct_cost_basis,
    calc_sl_threshold_cost,
)


def _legacy_def_a_trigger(entry_credit: float, pnl: float, qty: int, sl_mult: float) -> bool:
    target = -(entry_credit * sl_mult * 100.0 * qty)
    return pnl <= target


def _def_b_trigger(entry_credit: float, pnl: float, qty: int, sl_mult: float) -> bool:
    spread = calc_current_spread_cost_from_pnl(entry_credit, pnl, qty)
    threshold = calc_sl_threshold_cost(entry_credit, sl_mult)
    assert spread is not None and threshold is not None
    return spread >= threshold


def test_def_b_equivalence():
    # Def B sl_mult=3.0 should match old Def A sl_mult=2.0
    credits = [0.18, 0.25, 0.29, 0.35]
    pnls = [-200, -100, -70, -58, -50, -35, -29, -25, -10, 0, 5]
    for credit in credits:
        for pnl in pnls:
            a = _legacy_def_a_trigger(credit, pnl, 1, 2.0)
            b = _def_b_trigger(credit, pnl, 1, 3.0)
            assert a == b, f"credit={credit} pnl={pnl}: DefA(2x) != DefB(3x)"


def test_sl_pct_at_trigger_is_100():
    entry_credit = 0.25
    sl_mult = 3.0
    threshold = calc_sl_threshold_cost(entry_credit, sl_mult)
    assert threshold == 0.75
    sl_pct = calc_sl_pct_cost_basis(0.75, threshold)
    assert math.isclose(sl_pct, 100.0, rel_tol=0, abs_tol=1e-9)


def test_trade4_no_trigger():
    # Trade #4 replay per user prompt under Def B sl_mult=3.0.
    entry_credit = 0.25
    pnl = -29.08
    spread = calc_current_spread_cost_from_pnl(entry_credit, pnl, 1)
    threshold = calc_sl_threshold_cost(entry_credit, 3.0)
    assert spread is not None and threshold is not None
    assert math.isclose(spread, 0.5408, rel_tol=0, abs_tol=1e-4)
    assert math.isclose(calc_sl_pct_cost_basis(spread, threshold), 72.1066666667, rel_tol=0, abs_tol=1e-3)
    assert spread < threshold  # no trigger


def test_trade3_no_trigger():
    # Trade #3 replay per user prompt under Def B sl_mult=3.0.
    entry_credit = 0.29
    pnl = -35.95
    spread = calc_current_spread_cost_from_pnl(entry_credit, pnl, 1)
    threshold = calc_sl_threshold_cost(entry_credit, 3.0)
    assert spread is not None and threshold is not None
    assert math.isclose(spread, 0.6495, rel_tol=0, abs_tol=1e-4)
    assert math.isclose(calc_sl_pct_cost_basis(spread, threshold), 74.6551724138, rel_tol=0, abs_tol=1e-3)
    assert spread < threshold  # no trigger


def test_sl_mult_2_tighter():
    entry_credit = 0.25
    pnl = -29.08
    assert _def_b_trigger(entry_credit, pnl, 1, 2.0) is True
    assert _def_b_trigger(entry_credit, pnl, 1, 3.0) is False
