from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from src.strategy.execution import IronCondorPosition, LiveExecutor


class CaptureLogger:
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

    def debug(self, message, *args):
        self._log("debug", message, *args)

    def critical(self, message, *args):
        self._log("critical", message, *args)


class FakeIB:
    def __init__(self, entry_statuses, force_zero_order_id=False):
        self.client = SimpleNamespace(port=7497)
        self._entry_statuses = list(entry_statuses)
        self._force_zero_order_id = bool(force_zero_order_id)
        self._next_order_id = 1000
        self._next_conid = 500000
        self.place_orders = []
        self.cancel_orders = []
        self._fills = []
        self._open_trades = []

    def qualifyContracts(self, *contracts):
        for contract in contracts:
            if not getattr(contract, "conId", 0):
                self._next_conid += 1
                contract.conId = self._next_conid
        return list(contracts)

    def reqMktData(self, contract, *_args, **_kwargs):
        # Deterministic quote map for 684/682 puts and 693/695 calls.
        if contract.right == "P":
            if contract.strike >= 684:
                bid, ask = 0.29, 0.31
            else:
                bid, ask = 0.14, 0.16
            delta = -0.12
        else:
            if contract.strike <= 693:
                bid, ask = 0.31, 0.33
            else:
                bid, ask = 0.11, 0.13
            delta = 0.14
        greeks = SimpleNamespace(delta=delta, theta=-0.1, gamma=0.01, impliedVol=0.2)
        return SimpleNamespace(bid=bid, ask=ask, modelGreeks=greeks, contract=contract)

    def cancelMktData(self, _contract):
        return None

    def whatIfOrder(self, _contract, _order):
        # Paper mode behavior in this bot: empty list can be bypassed.
        return []

    def placeOrder(self, _contract, order):
        if not getattr(order, "orderId", 0):
            if self._force_zero_order_id:
                order.orderId = 0
            else:
                order.orderId = self._next_order_id
                self._next_order_id += 1

        idx = len(self.place_orders)
        status = self._entry_statuses[idx] if idx < len(self._entry_statuses) else self._entry_statuses[-1]
        fills = []
        if status == "Filled":
            exec_obj = SimpleNamespace(
                orderId=order.orderId,
                side="BOT",
                shares=1.0,
                price=abs(float(getattr(order, "lmtPrice", 0.2) or 0.2)),
            )
            fill_contract = SimpleNamespace(symbol="XSP", secType="BAG", right="", strike=0.0, conId=0)
            fills = [SimpleNamespace(execution=exec_obj, contract=fill_contract, time=datetime.now(timezone.utc))]

        trade = SimpleNamespace(
            order=order,
            orderStatus=SimpleNamespace(
                status=status,
                filled=1.0 if status == "Filled" else 0.0,
                remaining=0.0 if status == "Filled" else 1.0,
            ),
            fills=fills,
        )
        self.place_orders.append(trade)
        return trade

    def cancelOrder(self, order):
        self.cancel_orders.append(int(getattr(order, "orderId", 0) or 0))

    def sleep(self, _seconds):
        return None

    def openTrades(self):
        return self._open_trades

    def fills(self):
        return self._fills

    def reqAllOpenOrders(self):
        return []

    def reqGlobalCancel(self):
        return None

    def positions(self):
        return []


def _make_executor(fake_ib: FakeIB, tmp_path: Path) -> LiveExecutor:
    ex = LiveExecutor.__new__(LiveExecutor)
    ex.ib = fake_ib
    ex.connector = SimpleNamespace(get_live_price=lambda _symbol: 0.0)
    ex.logger = CaptureLogger()
    ex.config = {"min_credit": 0.18, "target_delta": 0.10}
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
    ex.WING_WIDTH = 2.0
    ex.MAX_QTY = 1
    ex.CHASE_TICKS = 1
    ex.TICK_SIZE = 0.05
    ex.STOP_LOSS_MULT = 2.0
    ex.TAKE_PROFIT_PCT = 1.0
    ex.HOLD_TO_EXPIRY_MODE = True
    ex.expected_commission_win = 2.60
    ex.expected_commission_loss = 5.20
    ex.exit_strategy_label = "hold_to_expiry_sl2x"
    ex.STATE_FILE = tmp_path / "state_fix12.json"
    return ex


def _dummy_active_position() -> IronCondorPosition:
    return IronCondorPosition(
        trade_id=999,
        entry_time=datetime.now(),
        short_put_strike=684.0,
        long_put_strike=682.0,
        short_call_strike=693.0,
        long_call_strike=695.0,
        entry_credit=0.30,
        qty=1,
        max_profit=30.0,
        max_loss=170.0,
        spot_at_entry=690.0,
        vix_at_entry=19.0,
        delta_net=0.0,
        snapshot_json="{}",
        legs=[],
    )


def test_fix12_cancel_arrives_before_fill_single_position(monkeypatch, tmp_path):
    """
    1) Cancel arrives before fill -> single position opened, no duplicate submit race.
    """
    ex = _make_executor(FakeIB(["Submitted", "Filled"]), tmp_path)
    race_calls = []

    monkeypatch.setattr(ex, "_wait_for_terminal", lambda _order, timeout=10.0: "Cancelled")
    monkeypatch.setattr(ex, "_register_fill_from_race", lambda _order: race_calls.append("race"))

    position = ex.execute_iron_condor(
        short_put=684.0,
        short_call=693.0,
        expiry="20260220",
        spot=690.0,
        vix=19.0,
        delta_net=0.0,
    )

    assert position is not None
    assert ex.active_position is not None
    assert len(ex.ib.place_orders) == 2
    assert race_calls == []
    assert ex.pending_entry_order_id is None


def test_fix12_fill_arrives_before_cancel_registers_race_no_second_submit(monkeypatch, tmp_path):
    """
    2) Fill arrives before cancel -> register race fill and do NOT submit next chase order.
    """
    ex = _make_executor(FakeIB(["Submitted"]), tmp_path)
    race_calls = []

    def _register(_order):
        race_calls.append(int(getattr(_order, "orderId", 0) or 0))
        ex.active_position = _dummy_active_position()
        ex.pending_entry_order_id = None

    monkeypatch.setattr(ex, "_wait_for_terminal", lambda _order, timeout=10.0: "Filled")
    monkeypatch.setattr(ex, "_register_fill_from_race", _register)

    position = ex.execute_iron_condor(
        short_put=684.0,
        short_call=693.0,
        expiry="20260220",
        spot=690.0,
        vix=19.0,
        delta_net=0.0,
    )

    assert len(ex.ib.place_orders) == 1
    assert race_calls, "Expected race registration when cancel target fills."
    assert position is ex.active_position


def test_fix12_cancel_timeout_aborts_cleanly(monkeypatch, tmp_path):
    """
    3) Cancel timeout -> abort cleanly, no extra submit, pending lock cleared.
    """
    ex = _make_executor(FakeIB(["Submitted"]), tmp_path)
    monkeypatch.setattr(ex, "_wait_for_terminal", lambda _order, timeout=10.0: "Timeout")
    monkeypatch.setattr(ex, "_register_fill_from_race", lambda _order: (_ for _ in ()).throw(RuntimeError("must not register")))

    position = ex.execute_iron_condor(
        short_put=684.0,
        short_call=693.0,
        expiry="20260220",
        spot=690.0,
        vix=19.0,
        delta_net=0.0,
    )

    assert position is None
    assert ex.active_position is None
    assert len(ex.ib.place_orders) == 1
    assert ex.pending_entry_order_id is None
    assert any("Cancel timeout" in msg for lvl, msg in ex.logger.records if lvl == "error")


def test_fix12_startup_stale_pending_filled_recovers(monkeypatch, tmp_path):
    """
    4) Startup with stale pending_entry_order_id (Filled) -> recovery path is called.
    """
    fake_ib = FakeIB(["Submitted"])
    order_id = 555
    fake_ib._fills = [
        SimpleNamespace(
            execution=SimpleNamespace(orderId=order_id),
            contract=SimpleNamespace(symbol="XSP", secType="BAG"),
            time=datetime.now(timezone.utc),
        )
    ]
    ex = _make_executor(fake_ib, tmp_path)
    ex.pending_entry_order_id = order_id

    called = []

    def _register(order):
        called.append(int(getattr(order, "orderId", 0) or 0))
        ex.pending_entry_order_id = None

    monkeypatch.setattr(ex, "_register_fill_from_race", _register)
    ex.startup_reconciliation()

    assert called == [order_id]
    assert ex.pending_entry_order_id is None
    assert any(
        "Stale pending_entry_order_id detected on startup" in msg and "Resolved as Filled" in msg
        for lvl, msg in ex.logger.records
        if lvl == "warning"
    )


def test_fix12_startup_stale_pending_cancelled_clears_lock(tmp_path):
    """
    5) Startup with stale pending_entry_order_id (Cancelled/not found) -> lock cleared.
    """
    ex = _make_executor(FakeIB(["Submitted"]), tmp_path)
    ex.pending_entry_order_id = 777

    ex.startup_reconciliation()

    assert ex.pending_entry_order_id is None
    assert any(
        "Stale pending_entry_order_id detected on startup" in msg and "Resolved as Cancelled" in msg
        for lvl, msg in ex.logger.records
        if lvl == "warning"
    )


def test_fix12_placeorder_orderid_zero_warns_and_keeps_lock_clear(tmp_path):
    """
    6) placeOrder returns orderId=0 -> no crash, warning logged, lock stays clear.
    """
    ex = _make_executor(FakeIB(["Filled"], force_zero_order_id=True), tmp_path)

    position = ex.execute_iron_condor(
        short_put=684.0,
        short_call=693.0,
        expiry="20260220",
        spot=690.0,
        vix=19.0,
        delta_net=0.0,
    )

    assert position is not None
    assert ex.pending_entry_order_id is None
    assert any(
        "pending_entry_order_id lock not set" in msg
        for lvl, msg in ex.logger.records
        if lvl == "warning"
    )


def test_fix12_register_fill_from_race_uses_executor_symbol(tmp_path):
    """
    7) _register_fill_from_race must use executor.SYMBOL (not hardcoded XSP).
    """
    fake_ib = FakeIB(["Submitted"])
    ex = _make_executor(fake_ib, tmp_path)
    ex.SYMBOL = "SPX"

    order_id = 4242
    now = datetime.now(timezone.utc)
    fake_ib._fills = [
        SimpleNamespace(
            execution=SimpleNamespace(orderId=order_id, side="BOT", shares=1.0, price=0.10),
            contract=SimpleNamespace(symbol="SPX", secType="OPT", right="P", strike=671.0, conId=1001),
            time=now,
        ),
        SimpleNamespace(
            execution=SimpleNamespace(orderId=order_id, side="SLD", shares=1.0, price=0.20),
            contract=SimpleNamespace(symbol="SPX", secType="OPT", right="P", strike=673.0, conId=1002),
            time=now,
        ),
        SimpleNamespace(
            execution=SimpleNamespace(orderId=order_id, side="SLD", shares=1.0, price=0.19),
            contract=SimpleNamespace(symbol="SPX", secType="OPT", right="C", strike=688.0, conId=1003),
            time=now,
        ),
        SimpleNamespace(
            execution=SimpleNamespace(orderId=order_id, side="BOT", shares=1.0, price=0.01),
            contract=SimpleNamespace(symbol="SPX", secType="OPT", right="C", strike=690.0, conId=1004),
            time=now,
        ),
    ]

    ex._register_fill_from_race(SimpleNamespace(orderId=order_id, lmtPrice=0.28))

    assert ex.active_position is not None
    assert ex.active_position.short_put_strike == 673.0
    assert ex.active_position.short_call_strike == 688.0
