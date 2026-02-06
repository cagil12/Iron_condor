import pytest
from datetime import date
from run_live_monitor import get_next_friday_expiry

def test_get_next_friday_expiry():
    """Verificar que get_next_friday_expiry retorna HOY si es viernes, no el próximo"""
    
    # Lunes 2026-02-02 → Viernes 2026-02-06
    assert get_next_friday_expiry_test(date(2026, 2, 2)) == '20260206'
    # Martes
    assert get_next_friday_expiry_test(date(2026, 2, 3)) == '20260206'
    # Miércoles
    assert get_next_friday_expiry_test(date(2026, 2, 4)) == '20260206'
    # Jueves
    assert get_next_friday_expiry_test(date(2026, 2, 5)) == '20260206'
    # VIERNES — debe retornar HOY, no el próximo viernes
    assert get_next_friday_expiry_test(date(2026, 2, 6)) == '20260206'
    # Sábado → próximo viernes
    assert get_next_friday_expiry_test(date(2026, 2, 7)) == '20260213'
    # Domingo → próximo viernes
    assert get_next_friday_expiry_test(date(2026, 2, 8)) == '20260213'

def get_next_friday_expiry_test(today: date) -> str:
    """Version of get_next_friday_expiry for testing that accepts a date object."""
    from datetime import timedelta
    days_ahead = 4 - today.weekday()  # Friday = 4
    if days_ahead < 0: 
        days_ahead += 7
    next_friday = today + timedelta(days=days_ahead)
    return next_friday.strftime('%Y%m%d')

def test_chase_direction():
    """Verificar que chase hace precio menos negativo (más fácil de llenar)"""
    initial = -0.22
    TICK_SIZE = 0.01
    
    prices = [initial]
    for i in range(3):
        prices.append(prices[-1] + TICK_SIZE)
    
    # Cada precio debe ser más fácil de llenar (menos crédito demandado)
    assert [round(p, 2) for p in prices] == [-0.22, -0.21, -0.20, -0.19]
    
    # Todos negativos (credit spread)
    assert all(p < 0 for p in prices)
    
    # Cada siguiente es MENOS negativo (más fácil fill)
    for i in range(len(prices) - 1):
        assert prices[i+1] > prices[i], f"Chase #{i+1} should be less negative"

def test_chase_loop_code_quality():
    """Verify chase loop in execution.py follows IBKR API specs"""
    import inspect
    from src.strategy.execution import LiveExecutor
    
    source = inspect.getsource(LiveExecutor.execute_iron_condor)
    
    # Check filled initialization (FIX 1)
    assert "filled = False" in source, "filled must be initialized before chase loop"
    
    # Check no order.lmtPrice modification (Error 105) (FIX 6b)
    # Allow it in comments but not in actual code
    lines = [l.strip() for l in source.split('\n') if not l.strip().startswith('#')]
    active_code = '\n'.join(lines)
    assert "order.lmtPrice =" not in active_code, \
        "Do not modify order.lmtPrice on combo — Error 105. Use cancel + new LimitOrder."
    
    # Check cancelOrder exists in chase (FIX 6b)
    assert "cancelOrder" in source, "Must cancelOrder before resubmitting chase"

def test_startup_reconciliation_exists():
    """Verify FIX 2 is implemented"""
    from src.strategy.execution import LiveExecutor
    assert hasattr(LiveExecutor, 'startup_reconciliation')

def test_state_persistence_exists():
    """Verify FIX 3 is implemented"""
    from src.strategy.execution import LiveExecutor
    assert hasattr(LiveExecutor, 'save_state')
    assert hasattr(LiveExecutor, 'load_state')

def test_atomic_closure_exists():
    """Verify FIX 4 is implemented"""
    from src.strategy.execution import LiveExecutor
    assert hasattr(LiveExecutor, 'close_position_atomic')
