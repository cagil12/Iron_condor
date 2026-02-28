import json
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WATCHDOG_PS1 = REPO_ROOT / "scripts" / "background_watchdog.ps1"


def _run_ps_json(ps_body: str):
    cmd = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        ps_body,
    ]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, check=True)
    stdout = proc.stdout.strip()
    if not stdout:
        raise AssertionError(f"No JSON output from PowerShell.\nSTDERR:\n{proc.stderr}")
    return json.loads(stdout)


def _ps_prelude() -> str:
    p = str(WATCHDOG_PS1).replace("\\", "\\\\")
    return f". '{p}' -NoLoop\n"


def test_stale_detection_xsp_zero():
    body = _ps_prelude() + r"""
$snap = [pscustomobject]@{
  has_log = $true; has_market_line = $true; latest_xsp = 0.0; latest_vix = 19.4;
  latest_line_age_seconds = 10; reconnect_error_count = 0
}
$now = Get-Date "2026-02-26T11:30:00"
$r1 = Get-StalenessEvaluation -Now $now -HasActivePosition $true -InStalenessWindow $true -Snapshot $snap -IntervalSec 60 -ConsecutiveStaleChecks 0 -DataStale $false -LastValidDataTime $null
$r2 = Get-StalenessEvaluation -Now ($now.AddSeconds(60)) -HasActivePosition $true -InStalenessWindow $true -Snapshot $snap -IntervalSec 60 -ConsecutiveStaleChecks $r1.consecutive_stale_checks -DataStale $r1.data_stale -LastValidDataTime $r1.last_valid_data_time
$r2 | Select-Object status,critical,data_stale,consecutive_stale_checks,stale_seconds | ConvertTo-Json -Compress
"""
    data = _run_ps_json(body)
    assert data["status"] == "DATA_STALE"
    assert data["critical"] is True
    assert data["data_stale"] is True
    assert data["consecutive_stale_checks"] >= 2


def test_stale_detection_vix_zero():
    body = _ps_prelude() + r"""
$snap = [pscustomobject]@{
  has_log = $true; has_market_line = $true; latest_xsp = 688.1; latest_vix = 0.0;
  latest_line_age_seconds = 15; reconnect_error_count = 0
}
$now = Get-Date "2026-02-26T11:30:00"
$r1 = Get-StalenessEvaluation -Now $now -HasActivePosition $true -InStalenessWindow $true -Snapshot $snap -IntervalSec 60 -ConsecutiveStaleChecks 0 -DataStale $false -LastValidDataTime $null
$r2 = Get-StalenessEvaluation -Now ($now.AddSeconds(60)) -HasActivePosition $true -InStalenessWindow $true -Snapshot $snap -IntervalSec 60 -ConsecutiveStaleChecks $r1.consecutive_stale_checks -DataStale $r1.data_stale -LastValidDataTime $r1.last_valid_data_time
$r2 | Select-Object status,critical,data_stale,consecutive_stale_checks | ConvertTo-Json -Compress
"""
    data = _run_ps_json(body)
    assert data["status"] == "DATA_STALE"
    assert data["critical"] is True
    assert data["data_stale"] is True


def test_stale_clears_on_recovery():
    body = _ps_prelude() + r"""
$snapBad = [pscustomobject]@{
  has_log = $true; has_market_line = $true; latest_xsp = 0.0; latest_vix = 0.0;
  latest_line_age_seconds = 20; reconnect_error_count = 0
}
$snapGood = [pscustomobject]@{
  has_log = $true; has_market_line = $true; latest_xsp = 688.5; latest_vix = 19.6;
  latest_line_age_seconds = 10; reconnect_error_count = 0
}
$now = Get-Date "2026-02-26T11:30:00"
$r1 = Get-StalenessEvaluation -Now $now -HasActivePosition $true -InStalenessWindow $true -Snapshot $snapBad -IntervalSec 60 -ConsecutiveStaleChecks 0 -DataStale $false -LastValidDataTime $null
$r2 = Get-StalenessEvaluation -Now ($now.AddSeconds(60)) -HasActivePosition $true -InStalenessWindow $true -Snapshot $snapBad -IntervalSec 60 -ConsecutiveStaleChecks $r1.consecutive_stale_checks -DataStale $r1.data_stale -LastValidDataTime $r1.last_valid_data_time
$r3 = Get-StalenessEvaluation -Now ($now.AddSeconds(120)) -HasActivePosition $true -InStalenessWindow $true -Snapshot $snapGood -IntervalSec 60 -ConsecutiveStaleChecks $r2.consecutive_stale_checks -DataStale $r2.data_stale -LastValidDataTime $r2.last_valid_data_time
$r3 | Select-Object status,critical,data_stale,consecutive_stale_checks,recovered | ConvertTo-Json -Compress
"""
    data = _run_ps_json(body)
    assert data["status"] == "OK"
    assert data["critical"] is False
    assert data["data_stale"] is False
    assert data["consecutive_stale_checks"] == 0
    assert data["recovered"] is True


def test_no_stale_without_position():
    body = _ps_prelude() + r"""
$snap = [pscustomobject]@{
  has_log = $true; has_market_line = $true; latest_xsp = 0.0; latest_vix = 0.0;
  latest_line_age_seconds = 10; reconnect_error_count = 0
}
$now = Get-Date "2026-02-26T11:30:00"
$r1 = Get-StalenessEvaluation -Now $now -HasActivePosition $false -InStalenessWindow $true -Snapshot $snap -IntervalSec 60 -ConsecutiveStaleChecks 0 -DataStale $false -LastValidDataTime $null
$r2 = Get-StalenessEvaluation -Now ($now.AddSeconds(60)) -HasActivePosition $false -InStalenessWindow $true -Snapshot $snap -IntervalSec 60 -ConsecutiveStaleChecks $r1.consecutive_stale_checks -DataStale $r1.data_stale -LastValidDataTime $r1.last_valid_data_time
$r2 | Select-Object status,critical,data_stale,consecutive_stale_checks | ConvertTo-Json -Compress
"""
    data = _run_ps_json(body)
    assert data["status"] == "DATA_STALE"
    assert data["critical"] is False


def test_no_stale_during_warmup():
    body = _ps_prelude() + r"""
$snap = [pscustomobject]@{
  has_log = $true; has_market_line = $true; latest_xsp = 0.0; latest_vix = 0.0;
  latest_line_age_seconds = 10; reconnect_error_count = 0
}
$now = Get-Date "2026-02-26T10:00:00"
$r1 = Get-StalenessEvaluation -Now $now -HasActivePosition $true -InStalenessWindow $false -Snapshot $snap -IntervalSec 60 -ConsecutiveStaleChecks 0 -DataStale $false -LastValidDataTime $null
$r2 = Get-StalenessEvaluation -Now ($now.AddSeconds(60)) -HasActivePosition $true -InStalenessWindow $false -Snapshot $snap -IntervalSec 60 -ConsecutiveStaleChecks $r1.consecutive_stale_checks -DataStale $r1.data_stale -LastValidDataTime $r1.last_valid_data_time
$r2 | Select-Object status,critical,data_stale,consecutive_stale_checks | ConvertTo-Json -Compress
"""
    data = _run_ps_json(body)
    assert data["status"] == "DATA_STALE"
    assert data["critical"] is False
    assert data["data_stale"] is False
    assert data["consecutive_stale_checks"] == 0
