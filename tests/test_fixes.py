import pytest
from datetime import date
import datetime as datetime_module
from pathlib import Path
from types import SimpleNamespace
import json
from run_live_monitor import get_0dte_expiry

def _get_0dte_expiry_for(today: date, monkeypatch) -> str:
    class FakeDate(date):
        @classmethod
        def today(cls):
            return today

    monkeypatch.setattr(datetime_module, 'date', FakeDate)
    return get_0dte_expiry()


def test_get_0dte_expiry(monkeypatch):
    """Verificar lógica 0DTE: Mon-Fri=HOY, fin de semana=lunes siguiente."""

    # Mon-Fri => hoy
    assert _get_0dte_expiry_for(date(2026, 2, 2), monkeypatch) == '20260202'  # Monday
    assert _get_0dte_expiry_for(date(2026, 2, 6), monkeypatch) == '20260206'  # Friday

    # Weekend => next Monday
    assert _get_0dte_expiry_for(date(2026, 2, 7), monkeypatch) == '20260209'  # Saturday
    assert _get_0dte_expiry_for(date(2026, 2, 8), monkeypatch) == '20260209'  # Sunday

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
    
    # Check cancel path exists in chase (FIX 6b / Fix 12 refactor)
    assert (
        "_cancel_entry_order_and_wait" in source or "cancelOrder" in source
    ), "Must cancel before resubmitting chase (directly or via helper)."

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


class _CaptureLogger:
    def __init__(self):
        self.records = []

    def _log(self, level, message, *args):
        if args:
            try:
                message = message % args
            except Exception:
                message = f"{message} {' '.join(str(a) for a in args)}"
        self.records.append((level, str(message)))

    def info(self, message, *args):
        self._log("info", message, *args)

    def warning(self, message, *args):
        self._log("warning", message, *args)

    def error(self, message, *args):
        self._log("error", message, *args)

    def critical(self, message, *args):
        self._log("critical", message, *args)

    def debug(self, message, *args):
        self._log("debug", message, *args)


class _FakeIBForChase:
    def __init__(self):
        self.client = SimpleNamespace(port=7497)
        self._next_order_id = 9000
        self.place_calls = []
        self.cancel_calls = []
        self._positions = []
        self._fills = []
        self.positions_calls = 0

    def placeOrder(self, _contract, order):
        if not getattr(order, "orderId", 0):
            order.orderId = self._next_order_id
            self._next_order_id += 1
        trade = SimpleNamespace(
            order=order,
            orderStatus=SimpleNamespace(status="Submitted", filled=0.0, remaining=1.0),
            fills=[],
        )
        self.place_calls.append((order.orderId, float(getattr(order, "lmtPrice", 0.0) or 0.0)))
        return trade

    def cancelOrder(self, order):
        self.cancel_calls.append(int(getattr(order, "orderId", 0) or 0))

    def openTrades(self):
        return []

    def fills(self):
        return self._fills

    def positions(self):
        self.positions_calls += 1
        return list(self._positions)

    def sleep(self, _seconds):
        return None


def _make_state_machine_executor(tmp_path: Path):
    from src.strategy.execution import LiveExecutor, ChaseState

    fake_ib = _FakeIBForChase()
    ex = LiveExecutor.__new__(LiveExecutor)
    ex.ib = fake_ib
    ex.connector = SimpleNamespace(get_live_price=lambda _symbol: 0.0)
    ex.logger = _CaptureLogger()
    ex.config = {"symbol": "XSP", "target_delta": 0.10}
    ex.SYMBOL = "XSP"
    ex.active_position = None
    ex.journal = None
    ex.consecutive_losses = 0
    ex.streak_pause_until = None
    ex.cumulative_pnl = 0.0
    ex.pnl_high_water_mark = 0.0
    ex.dd_pause_until = None
    ex.expired_hold_signatures = []
    ex.last_expiry_cleanup_key = None
    ex.pending_entry_order_id = None
    ex.emergency_halt = False
    ex.MAX_QTY = 1
    ex.WING_WIDTH = 2.0
    ex.STATE_FILE = tmp_path / "state_fix12_machine.json"
    ex.chase_state = ChaseState.IDLE
    ex._active_chase_order_id = None
    ex._chase_state_ts = 0.0
    ex._ib_order_callbacks_registered = False
    return ex, fake_ib, ChaseState


def _opt_position(symbol: str, strike: float, right: str, qty: int):
    return SimpleNamespace(
        contract=SimpleNamespace(symbol=symbol, secType="OPT", strike=strike, right=right),
        position=qty,
    )


def test_fix12_state_machine_normal_chase_flow(tmp_path):
    """Test 1: submit -> fill -> FILLED state -> position verification OK."""
    ex, fake_ib, ChaseState = _make_state_machine_executor(tmp_path)
    order = SimpleNamespace(orderId=0, lmtPrice=-0.35, totalQuantity=1)

    trade = ex._submit_chase_order(SimpleNamespace(), order, limit_price=-0.35, chase=0)
    assert ex.chase_state == ChaseState.PENDING_FILL

    trade.orderStatus.status = "Filled"
    fake_ib._positions = [
        _opt_position("XSP", 682, "P", 1),
        _opt_position("XSP", 684, "P", -1),
        _opt_position("XSP", 693, "C", -1),
        _opt_position("XSP", 695, "C", 1),
    ]
    ex._on_order_status(trade)

    assert ex.chase_state == ChaseState.FILLED
    assert ex._verify_position_state(expected_qty=1) == "ok"


def test_fix12_state_machine_cancel_then_resubmit_no_race(monkeypatch, tmp_path):
    """Test 2: cancel confirmed -> IDLE -> resubmit allowed -> second fill possible."""
    ex, fake_ib, ChaseState = _make_state_machine_executor(tmp_path)
    order1 = SimpleNamespace(orderId=0, lmtPrice=-0.30, totalQuantity=1)
    ex._submit_chase_order(SimpleNamespace(), order1, limit_price=-0.30, chase=0)

    monkeypatch.setattr(ex, "_wait_for_terminal", lambda _order, timeout=10.0: "Cancelled")
    status = ex._cancel_entry_order_and_wait(order1, reason="test_retry", timeout=10.0)
    assert status == "Cancelled"
    assert ex.chase_state == ChaseState.IDLE

    order2 = SimpleNamespace(orderId=0, lmtPrice=-0.25, totalQuantity=1)
    ex._submit_chase_order(SimpleNamespace(), order2, limit_price=-0.25, chase=1)
    assert len(fake_ib.place_calls) == 2
    assert ex.chase_state == ChaseState.PENDING_FILL


def test_fix12_state_machine_fill_during_cancel_blocks_resubmit(monkeypatch, tmp_path):
    """Test 3: fill during cancel -> FILLED_DURING_CANCEL and no new submit in same sequence."""
    ex, fake_ib, ChaseState = _make_state_machine_executor(tmp_path)
    order = SimpleNamespace(orderId=0, lmtPrice=-0.30, totalQuantity=1)
    ex._submit_chase_order(SimpleNamespace(), order, limit_price=-0.30, chase=0)
    fake_ib._positions = [
        _opt_position("XSP", 682, "P", 1),
        _opt_position("XSP", 684, "P", -1),
        _opt_position("XSP", 693, "C", -1),
        _opt_position("XSP", 695, "C", 1),
    ]

    monkeypatch.setattr(ex, "_wait_for_terminal", lambda _order, timeout=10.0: "Filled")
    status = ex._cancel_entry_order_and_wait(order, reason="credit_low", timeout=10.0)

    assert status == "Filled"
    assert ex.chase_state == ChaseState.FILLED_DURING_CANCEL
    assert len(fake_ib.place_calls) == 1


def test_fix12_state_machine_cancel_timeout_verifies_positions(monkeypatch, tmp_path):
    """Test 4: cancel timeout triggers position verification path (positions() call)."""
    ex, _fake_ib, _ChaseState = _make_state_machine_executor(tmp_path)
    order = SimpleNamespace(orderId=0, lmtPrice=-0.30, totalQuantity=1)
    ex._submit_chase_order(SimpleNamespace(), order, limit_price=-0.30, chase=0)

    monkeypatch.setattr(ex, "_wait_for_terminal", lambda _order, timeout=10.0: "Timeout")
    status = ex._cancel_entry_order_and_wait(order, reason="timeout_test", timeout=10.0)

    assert status == "Timeout"
    assert ex.ib.positions_calls >= 1


def test_fix12_state_machine_position_mismatch_triggers_emergency_halt(tmp_path):
    """Test 5: qty mismatch (IBKR>expected) logs CRITICAL and persists emergency_halt."""
    ex, fake_ib, _ChaseState = _make_state_machine_executor(tmp_path)
    fake_ib._positions = [
        _opt_position("XSP", 682, "P", 2),
        _opt_position("XSP", 684, "P", -2),
        _opt_position("XSP", 693, "C", -2),
        _opt_position("XSP", 695, "C", 2),
    ]

    result = ex._verify_position_state(expected_qty=1)
    assert result == "mismatch"
    assert ex.emergency_halt is True
    assert any(level == "critical" and "POSITION MISMATCH" in msg for level, msg in ex.logger.records)

    state = json.loads(ex.STATE_FILE.read_text())
    assert state["emergency_halt"] is True


def test_fix12_state_machine_invariant_blocks_submit_outside_idle(tmp_path):
    """Test 6: submit_order() invariant — only allowed from IDLE state."""
    ex, fake_ib, ChaseState = _make_state_machine_executor(tmp_path)
    ex.chase_state = ChaseState.AWAITING_CANCEL

    with pytest.raises(RuntimeError):
        ex._submit_chase_order(SimpleNamespace(), SimpleNamespace(orderId=0, lmtPrice=-0.20), limit_price=-0.20, chase=1)

    assert fake_ib.place_calls == []
