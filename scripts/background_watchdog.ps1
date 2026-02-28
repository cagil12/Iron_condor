param(
    [string]$RepoRoot = "c:\Users\Usuario\.gemini\antigravity\scratch\ingresarios_options_research",
    [int]$IntervalSec = 60,
    [int]$TailLines = 20,
    [switch]$RunOnce,
    [switch]$NoLoop
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$outputsDir = Join-Path $RepoRoot "outputs"
$pidFile = Join-Path $outputsDir "background_watchdog.pid"
$logFile = Join-Path $outputsDir "background_watchdog.log"

$script:WatchdogState = @{
    consecutive_stale_checks = 0
    data_stale = $false
    last_valid_data_time = $null
}

function Get-MonitorProcess {
    $items = @()
    $procs = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue
    foreach ($p in $procs) {
        $cmd = [string]$p.CommandLine
        if (-not $cmd) { continue }
        if ($cmd -match "run_live_monitor\.py") {
            $startTime = $null
            try {
                $startTime = (Get-Process -Id $p.ProcessId -ErrorAction Stop).StartTime
            } catch {
                $startTime = Get-Date "2000-01-01"
            }
            $items += [pscustomobject]@{
                Id = [int]$p.ProcessId
                StartTime = $startTime
                CommandLine = $cmd
            }
        }
    }
    if ($items.Count -gt 0) {
        return ($items | Sort-Object StartTime -Descending | Select-Object -First 1)
    }
    return $null
}

function Get-UpdaterProcess {
    $items = @()
    $procs = Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" -ErrorAction SilentlyContinue
    foreach ($p in $procs) {
        $cmd = [string]$p.CommandLine
        if (-not $cmd) { continue }
        if ($cmd -match "bitacora_updater\.ps1") {
            $startTime = $null
            try {
                $startTime = (Get-Process -Id $p.ProcessId -ErrorAction Stop).StartTime
            } catch {
                $startTime = Get-Date "2000-01-01"
            }
            $items += [pscustomobject]@{
                Id = [int]$p.ProcessId
                StartTime = $startTime
                CommandLine = $cmd
            }
        }
    }
    if ($items.Count -gt 0) {
        return ($items | Sort-Object StartTime -Descending | Select-Object -First 1)
    }
    return $null
}

function Get-LatestMonitorLogPath {
    param([string]$OutputsDir)
    if (-not (Test-Path $OutputsDir)) { return $null }
    $f = Get-ChildItem -Path $OutputsDir -Filter "pos_monitor_live_*.log" -File -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($f) { return $f.FullName }
    return $null
}

function Parse-MonitorMarketLine {
    param([string]$Line)
    if ([string]::IsNullOrWhiteSpace($Line)) { return $null }

    # Current format:
    # [10:42:45] XSP: 688.09 | VIX: 19.7 | ...
    $patterns = @(
        "\[(\d{2}:\d{2}:\d{2})\]\s+XSP:\s*([-\d.]+)\s*\|\s*VIX:\s*([-\d.]+)",
        "(\d{2}:\d{2}:\d{2})\s*\|\s*XSP\s+([-\d.]+)\s*\|\s*VIX\s+([-\d.]+)"
    )
    foreach ($pattern in $patterns) {
        if ($Line -match $pattern) {
            return [pscustomobject]@{
                time_str = $matches[1]
                xsp = [double]$matches[2]
                vix = [double]$matches[3]
                raw = $Line
            }
        }
    }
    return $null
}

function Get-ActivePositionStatus {
    param([string]$RepoRoot)
    $statePath = Join-Path $RepoRoot "state.json"
    if (-not (Test-Path $statePath)) {
        return [pscustomobject]@{ active = $false; trade_id = $null }
    }
    try {
        $state = Get-Content -Path $statePath -Raw -Encoding UTF8 | ConvertFrom-Json
        $pos = $state.active_position
        if ($null -ne $pos) {
            return [pscustomobject]@{
                active = $true
                trade_id = $pos.trade_id
            }
        }
    } catch {
        return [pscustomobject]@{ active = $false; trade_id = $null }
    }
    return [pscustomobject]@{ active = $false; trade_id = $null }
}

function Test-StalenessWindow {
    param([datetime]$Now)
    # Only evaluate as critical during weekday session after warmup.
    if ($Now.DayOfWeek -in @([DayOfWeek]::Saturday, [DayOfWeek]::Sunday)) {
        return $false
    }
    $t = $Now.TimeOfDay
    $start = [TimeSpan]::FromHours(10) + [TimeSpan]::FromMinutes(5)
    $end = [TimeSpan]::FromHours(16) + [TimeSpan]::FromMinutes(5)
    return ($t -ge $start -and $t -le $end)
}

function Get-LatestMonitorSnapshot {
    param(
        [string]$OutputsDir,
        [int]$TailLines,
        [datetime]$Now
    )
    $path = Get-LatestMonitorLogPath -OutputsDir $OutputsDir
    if (-not $path) {
        return [pscustomobject]@{
            has_log = $false
            log_path = $null
            has_market_line = $false
            latest_xsp = $null
            latest_vix = $null
            latest_line_time = $null
            latest_line_age_seconds = $null
            reconnect_error_count = 0
            tail_count = 0
        }
    }

    $lines = @()
    try {
        $lines = @(Get-Content -Path $path -Tail $TailLines -Encoding UTF8 -ErrorAction Stop)
    } catch {
        $lines = @()
    }

    $reconnectCount = 0
    $parsed = $null
    for ($i = $lines.Count - 1; $i -ge 0; $i--) {
        $line = [string]$lines[$i]
        if ($line -match "Error (1100|1102)") {
            $reconnectCount += 1
        }
        if (-not $parsed) {
            $parsed = Parse-MonitorMarketLine -Line $line
        }
    }

    if (-not $parsed) {
        return [pscustomobject]@{
            has_log = $true
            log_path = $path
            has_market_line = $false
            latest_xsp = $null
            latest_vix = $null
            latest_line_time = $null
            latest_line_age_seconds = $null
            reconnect_error_count = $reconnectCount
            tail_count = $lines.Count
        }
    }

    $lineTime = $null
    $ageSec = $null
    try {
        $lineTime = Get-Date ("{0} {1}" -f $Now.ToString("yyyy-MM-dd"), $parsed.time_str)
        if ($lineTime -gt $Now.AddMinutes(5)) {
            $lineTime = $lineTime.AddDays(-1)
        }
        $ageSec = [int][Math]::Max(0, ($Now - $lineTime).TotalSeconds)
    } catch {
        $lineTime = $null
        $ageSec = $null
    }

    return [pscustomobject]@{
        has_log = $true
        log_path = $path
        has_market_line = $true
        latest_xsp = [double]$parsed.xsp
        latest_vix = [double]$parsed.vix
        latest_line_time = $lineTime
        latest_line_age_seconds = $ageSec
        reconnect_error_count = $reconnectCount
        tail_count = $lines.Count
        raw_line = $parsed.raw
    }
}

function Get-StalenessEvaluation {
    param(
        [datetime]$Now,
        [bool]$HasActivePosition,
        [bool]$InStalenessWindow,
        [object]$Snapshot,
        [int]$IntervalSec,
        [int]$ConsecutiveStaleChecks,
        [bool]$DataStale,
        [object]$LastValidDataTime
    )

    $result = [ordered]@{
        consecutive_stale_checks = $ConsecutiveStaleChecks
        data_stale = [bool]$DataStale
        last_valid_data_time = $LastValidDataTime
        status = "NO_LOG"
        critical = $false
        stale_seconds = 0
        xsp = $null
        vix = $null
        reconnect_storm = $false
        reconnect_error_count = 0
        recovered = $false
    }

    if ($null -eq $Snapshot -or -not $Snapshot.has_log) {
        return [pscustomobject]$result
    }

    $result.status = "NO_MARKET_LINE"
    $result.reconnect_error_count = [int]($Snapshot.reconnect_error_count | ForEach-Object { $_ })
    $result.reconnect_storm = ($result.reconnect_error_count -ge 3)

    if ($Snapshot.has_market_line) {
        $result.xsp = [double]$Snapshot.latest_xsp
        $result.vix = [double]$Snapshot.latest_vix

        $hasValidMarket = ($result.xsp -gt 0.0 -and $result.vix -gt 0.0)
        $freshEnough = $false
        if ($null -ne $Snapshot.latest_line_age_seconds) {
            $freshEnough = ([int]$Snapshot.latest_line_age_seconds -le 180)
        }

        if ($hasValidMarket -and $freshEnough) {
            $result.status = "OK"
            $result.consecutive_stale_checks = 0
            if ($result.data_stale) {
                $result.recovered = $true
            }
            $result.data_stale = $false
            $result.last_valid_data_time = $Now
            $result.stale_seconds = 0
            return [pscustomobject]$result
        }

        $result.status = "DATA_STALE"
        if ($InStalenessWindow) {
            # Increment once per watchdog cycle (not per individual symptom) to avoid
            # double counting invalid values + age in the same check.
            $result.consecutive_stale_checks = [int]$ConsecutiveStaleChecks + 1
            $result.data_stale = ($result.consecutive_stale_checks -ge 2)
        } else {
            $result.consecutive_stale_checks = 0
            $result.data_stale = $false
        }

        if ($result.data_stale) {
            if ($result.last_valid_data_time -is [datetime]) {
                $result.stale_seconds = [int][Math]::Max(0, ($Now - [datetime]$result.last_valid_data_time).TotalSeconds)
            } else {
                $result.stale_seconds = [int]($result.consecutive_stale_checks * $IntervalSec)
            }
        } elseif ($Snapshot.latest_line_age_seconds -is [int]) {
            $result.stale_seconds = [int]$Snapshot.latest_line_age_seconds
        }

        if ($HasActivePosition -and $InStalenessWindow -and $result.data_stale) {
            $result.critical = $true
        }
    }

    return [pscustomobject]$result
}

function Write-WatchdogLogLine {
    param(
        [string]$LogFile,
        [string]$Message
    )
    Add-Content -Path $LogFile -Value $Message -Encoding UTF8
}

function Invoke-WatchdogCheck {
    param(
        [string]$RepoRoot,
        [string]$OutputsDir,
        [string]$LogFile,
        [int]$IntervalSec,
        [int]$TailLines
    )

    $now = Get-Date
    try {
        $monitor = Get-MonitorProcess
        $updater = Get-UpdaterProcess

        $monitorStatus = if ($monitor) { "UP(pid=$($monitor.Id))" } else { "DOWN" }
        $updaterStatus = if ($updater) { "UP(pid=$($updater.Id))" } else { "DOWN" }

        $activePos = Get-ActivePositionStatus -RepoRoot $RepoRoot
        $inWindow = Test-StalenessWindow -Now $now
        $snapshot = Get-LatestMonitorSnapshot -OutputsDir $OutputsDir -TailLines $TailLines -Now $now

        $eval = Get-StalenessEvaluation `
            -Now $now `
            -HasActivePosition ([bool]$activePos.active) `
            -InStalenessWindow $inWindow `
            -Snapshot $snapshot `
            -IntervalSec $IntervalSec `
            -ConsecutiveStaleChecks ([int]$script:WatchdogState.consecutive_stale_checks) `
            -DataStale ([bool]$script:WatchdogState.data_stale) `
            -LastValidDataTime $script:WatchdogState.last_valid_data_time

        $script:WatchdogState.consecutive_stale_checks = [int]$eval.consecutive_stale_checks
        $script:WatchdogState.data_stale = [bool]$eval.data_stale
        $script:WatchdogState.last_valid_data_time = $eval.last_valid_data_time

        $dataStatus = ""
        switch ($eval.status) {
            "OK" {
                $dataStatus = ("data=OK xsp={0} vix={1}" -f `
                    ([double]$eval.xsp).ToString("0.0"), `
                    ([double]$eval.vix).ToString("0.0"))
                if ($eval.recovered) {
                    $dataStatus += " DATA_RECOVERED"
                }
            }
            "DATA_STALE" {
                $xspText = if ($null -ne $eval.xsp) { ([double]$eval.xsp).ToString("0.0") } else { "N/A" }
                $vixText = if ($null -ne $eval.vix) { ([double]$eval.vix).ToString("0.0") } else { "N/A" }
                $dataStatus = ("DATA_STALE xsp={0} vix={1} stale={2}s" -f $xspText, $vixText, [int]$eval.stale_seconds)
                if ($eval.critical) {
                    $dataStatus += " CRITICAL"
                }
            }
            "NO_MARKET_LINE" { $dataStatus = "data=NO_MARKET_LINE" }
            default { $dataStatus = "data=NO_LOG" }
        }

        if ($eval.reconnect_storm) {
            $dataStatus += (" RECONNECT_STORM:{0}" -f [int]$eval.reconnect_error_count)
        }

        $line = "[{0}] monitor={1} updater={2} | {3}" -f `
            ($now.ToString("yyyy-MM-dd HH:mm:ss")), `
            $monitorStatus, `
            $updaterStatus, `
            $dataStatus

        Write-WatchdogLogLine -LogFile $LogFile -Message $line
    } catch {
        Write-WatchdogLogLine -LogFile $LogFile -Message ("[{0}] error={1}" -f ($now.ToString("yyyy-MM-dd HH:mm:ss")), $_.Exception.Message)
    }
}

if (-not $NoLoop) {
    if (-not (Test-Path $outputsDir)) {
        New-Item -Path $outputsDir -ItemType Directory | Out-Null
    }

    Set-Content -Path $pidFile -Value $PID -Encoding ASCII
    Write-WatchdogLogLine -LogFile $logFile -Message ("[{0}] started pid={1} interval={2}s" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $PID, $IntervalSec)

    while ($true) {
        Invoke-WatchdogCheck -RepoRoot $RepoRoot -OutputsDir $outputsDir -LogFile $logFile -IntervalSec $IntervalSec -TailLines $TailLines
        if ($RunOnce) { break }
        Start-Sleep -Seconds $IntervalSec
    }
}
