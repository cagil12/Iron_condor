import csv
import json
from pathlib import Path

from src.strategy.execution import LiveExecutor


class _CaptureLogger:
    def __init__(self):
        self.records = []

    def _log(self, level, msg, *args):
        if args:
            try:
                msg = msg % args
            except Exception:
                msg = f"{msg} {' '.join(str(a) for a in args)}"
        self.records.append((level, str(msg)))

    def info(self, msg, *args): self._log("info", msg, *args)
    def warning(self, msg, *args): self._log("warning", msg, *args)
    def error(self, msg, *args): self._log("error", msg, *args)
    def critical(self, msg, *args): self._log("critical", msg, *args)


def _mk_executor(tmp_path: Path) -> LiveExecutor:
    ex = LiveExecutor.__new__(LiveExecutor)
    ex.logger = _CaptureLogger()
    ex.STATE_FILE = tmp_path / "state.json"
    ex.TRADE_JOURNAL_PATH = tmp_path / "trade_journal.csv"
    return ex


def _write_state(path: Path, active_position):
    path.write_text(json.dumps({"active_position": active_position}, indent=2))


def _write_journal(path: Path, rows):
    fields = ["trade_id", "status", "entry_credit"]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def test_recovery_from_state_json(tmp_path):
    ex = _mk_executor(tmp_path)
    _write_state(ex.STATE_FILE, {"trade_id": 7, "entry_credit": 0.29})
    _write_journal(ex.TRADE_JOURNAL_PATH, [{"trade_id": 7, "status": "OPEN", "entry_credit": "0.25"}])

    credit = ex._recover_entry_credit(7, fallback_avgcost_credit=0.24)
    assert credit == 0.29
    assert any("from state.json" in msg for level, msg in ex.logger.records if level == "info")


def test_recovery_from_journal(tmp_path):
    ex = _mk_executor(tmp_path)
    _write_state(ex.STATE_FILE, {"trade_id": 7, "entry_credit": 0})
    _write_journal(ex.TRADE_JOURNAL_PATH, [{"trade_id": 7, "status": "OPEN", "entry_credit": "0.29"}])

    credit = ex._recover_entry_credit(7, fallback_avgcost_credit=0.24)
    assert credit == 0.29
    assert any("from journal" in msg for level, msg in ex.logger.records if level == "info")


def test_recovery_fallback_avgcost(tmp_path):
    ex = _mk_executor(tmp_path)
    _write_state(ex.STATE_FILE, None)
    _write_journal(ex.TRADE_JOURNAL_PATH, [{"trade_id": 7, "status": "CLOSED", "entry_credit": "0.29"}])

    credit = ex._recover_entry_credit(7, fallback_avgcost_credit=0.24)
    assert credit == 0.24
    assert any("Falling back to IBKR avgCost" in msg for level, msg in ex.logger.records if level == "warning")


def test_recovery_zero_credit_skipped_to_journal(tmp_path):
    ex = _mk_executor(tmp_path)
    _write_state(ex.STATE_FILE, {"trade_id": 7, "entry_credit": 0})
    _write_journal(ex.TRADE_JOURNAL_PATH, [{"trade_id": 7, "status": "OPEN", "entry_credit": "0.31"}])

    credit = ex._recover_entry_credit(7, fallback_avgcost_credit=0.22)
    assert credit == 0.31


def test_recovery_no_open_trade_in_journal_fallback(tmp_path):
    ex = _mk_executor(tmp_path)
    _write_state(ex.STATE_FILE, {"trade_id": 7, "entry_credit": -1})
    _write_journal(ex.TRADE_JOURNAL_PATH, [{"trade_id": 7, "status": "CLOSED", "entry_credit": "0.31"}])

    credit = ex._recover_entry_credit(7, fallback_avgcost_credit=0.26)
    assert credit == 0.26
