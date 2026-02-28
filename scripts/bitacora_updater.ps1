param(
    [string]$RepoRoot = "c:\Users\Usuario\.gemini\antigravity\scratch\ingresarios_options_research",
    [int]$IntervalSec = 180
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$outputsDir = Join-Path $RepoRoot "outputs"
$stateFile = Join-Path $outputsDir "bitacora_updater.state"
$tradeHeaderStateFile = Join-Path $outputsDir "bitacora_updater.trade_header.state"
$pidFile = Join-Path $outputsDir "bitacora_updater.pid"
$runnerLog = Join-Path $outputsDir "bitacora_updater_runner.log"
$script:StopLossMult = $null

function Get-ConfiguredStopLossMult {
    param([string]$RepoRoot)

    if ($null -ne $script:StopLossMult -and $script:StopLossMult -gt 0) {
        return [double]$script:StopLossMult
    }

    $defaultMult = 3.0
    $configPath = Join-Path $RepoRoot "src\\utils\\config.py"
    if (-not (Test-Path $configPath)) {
        $script:StopLossMult = $defaultMult
        return $defaultMult
    }

    try {
        $raw = Get-Content -Path $configPath -Raw -Encoding UTF8
        if ($raw -match "'stop_loss_mult'\s*:\s*([0-9]+(?:\.[0-9]+)?)") {
            $val = [double]$matches[1]
            if ($val -gt 0) {
                $script:StopLossMult = $val
                return $val
            }
        }
    } catch {
        # fall through to default
    }

    $script:StopLossMult = $defaultMult
    return $defaultMult
}

function Get-LatestMonitorLog {
    $files = @(
        Get-ChildItem -Path (Join-Path $outputsDir "pos_monitor_*.log") -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -notlike "*.err.log" } |
            Sort-Object LastWriteTime -Descending
    )
    if ($files.Count -gt 0) {
        return $files[0].FullName
    }
    return $null
}

function Get-LatestExecutionLog {
    $patterns = @(
        "pos_monitor_live_*.log",
        "pos_monitor_startup_*.log",
        "paper_live_unbuf_*.err.log",
        "paper_live_*.err.log",
        "live_watch_*.err.log",
        "preopen_no_trade_*.err.log",
        "pos_monitor_*.err.log"
    )

    $files = @()
    foreach ($pattern in $patterns) {
        $files += Get-ChildItem -Path (Join-Path $outputsDir $pattern) -ErrorAction SilentlyContinue
    }

    if ($files.Count -gt 0) {
        return ($files | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
    }

    return $null
}

function Parse-WindowRows {
    param(
        [string]$LogPath,
        [datetime]$WindowStart
    )

    $pattern = '^(?:.+?\s+)?\[(\d{2}:\d{2}:\d{2})\]\s+XSP:\s+([0-9\.]+)\s+\|\s+VIX:\s+([0-9\.]+)\s+\|\s+PnL:\s+\$(-?[0-9\.]+)'
    $rows = @()
    $lines = Get-Content $LogPath -Tail 500 -ErrorAction SilentlyContinue
    foreach ($line in $lines) {
        if ($line -match $pattern) {
            $timeTxt = $matches[1]
            $xspVal = $matches[2]
            $vixVal = $matches[3]
            $pnlVal = $matches[4]
            $t = [datetime]::ParseExact($timeTxt, "HH:mm:ss", $null)
            $dt = Get-Date -Hour $t.Hour -Minute $t.Minute -Second $t.Second
            $pOtm = $null
            $pOtmGamma = $false
            if ($line -match 'P_OTM:(\d+\.\d+)(\(γ!\))?') {
                try { $pOtm = [double]$matches[1] } catch { $pOtm = $null }
                $pOtmGamma = -not [string]::IsNullOrWhiteSpace($matches[2])
            }
            if ($dt -ge $WindowStart) {
                $rows += [pscustomobject]@{
                    Time = $dt
                    XSP = [double]$xspVal
                    VIX = [double]$vixVal
                    PnL = [double]$pnlVal
                    POtm = $pOtm
                    POtmGamma = $pOtmGamma
                    Danger = $line.StartsWith([string][char]0x26A0) -or $line -like "*DANGER*"
                    Raw = $line
                }
            }
        }
    }
    return $rows
}

function Parse-ExecutionEvents {
    param(
        [string]$LogPath,
        [datetime]$WindowStart
    )

    $failPattern = '(?i)(failed|rejected|abort|not allowed|not filled|error 201|error 10349|margin check returned empty|exit failed|cancel race)'
    $attemptPattern = '(?i)(found setup|scanner found setup|attempt|placing|opening|execute_iron_condor)'
    $events = @()
    $lines = Get-Content $LogPath -Tail 800 -ErrorAction SilentlyContinue

    foreach ($line in $lines) {
        if ($line.Length -lt 19) {
            continue
        }

        $tsRaw = $line.Substring(0, 19)
        try {
            $dt = [datetime]::ParseExact(
                $tsRaw,
                "yyyy-MM-dd HH:mm:ss",
                [System.Globalization.CultureInfo]::InvariantCulture
            )
        } catch {
            continue
        }
        if ($dt -lt $WindowStart) {
            continue
        }

        $isFail = $line -match $failPattern
        $isAttempt = $line -match $attemptPattern
        if ($isFail -or $isAttempt) {
            $events += [pscustomobject]@{
                Time = $dt
                Fail = $isFail
                Attempt = $isAttempt
                Raw = $line
            }
        }
    }

    return $events
}

function Build-MarketComment {
    param(
        [double]$XspMove,
        [double]$VixMove,
        [int]$DangerCount,
        [int]$TotalRows
    )

    if ($XspMove -gt 0.40) {
        $trend = "sesgo alcista intraventana"
    } elseif ($XspMove -lt -0.40) {
        $trend = "sesgo bajista intraventana"
    } else {
        $trend = "mercado lateral intraventana"
    }

    if ($VixMove -gt 0.30) {
        $volTone = "volatilidad al alza"
    } elseif ($VixMove -lt -0.30) {
        $volTone = "volatilidad cediendo"
    } else {
        $volTone = "volatilidad estable"
    }

    $dangerPct = 0.0
    if ($TotalRows -gt 0) {
        $dangerPct = [Math]::Round((100.0 * $DangerCount / $TotalRows), 0)
    }

    if ($dangerPct -ge 60) {
        $riskTone = "alta (precio frecuente en zona DANGER)"
    } elseif ($dangerPct -ge 25) {
        $riskTone = "media"
    } else {
        $riskTone = "baja"
    }

    return "$trend; $volTone; presion de riesgo $riskTone."
}

function Get-OpenJournalTrade {
    param(
        [string]$RepoRoot
    )

    $journalPath = Join-Path $RepoRoot "data\trade_journal.csv"
    if (-not (Test-Path $journalPath)) {
        return $null
    }

    try {
        $rows = Import-Csv -Path $journalPath -ErrorAction Stop
    } catch {
        return $null
    }

    $openRows = @($rows | Where-Object { ([string]$_.status).Trim().ToUpperInvariant() -eq "OPEN" })
    if ($openRows.Count -eq 0) {
        return $null
    }

    $latest = $openRows |
        Sort-Object `
            @{ Expression = {
                    try { [int]$_.trade_id } catch { 0 }
                }
            }, `
            @{ Expression = { [string]$_.timestamp } } |
        Select-Object -Last 1

    if (-not $latest) {
        return $null
    }

    try {
        $tradeId = [int]$latest.trade_id
    } catch {
        $tradeId = 0
    }

    try {
        $entryCredit = [double]$latest.entry_credit
    } catch {
        $entryCredit = 0.0
    }

    return [pscustomobject]@{
        TradeId = $tradeId
        EntryCredit = $entryCredit
        Raw = $latest
    }
}

function Get-ActiveTradeSnapshot {
    param(
        [string]$RepoRoot
    )

    $statePath = Join-Path $RepoRoot "state.json"
    if (-not (Test-Path $statePath)) {
        return $null
    }

    try {
        $state = Get-Content $statePath -Raw -ErrorAction Stop | ConvertFrom-Json
    } catch {
        return $null
    }

    if (-not $state.active_position) {
        return $null
    }

    $ap = $state.active_position
    $journalOpen = Get-OpenJournalTrade -RepoRoot $RepoRoot

    try {
        $stateTradeId = [int]$ap.trade_id
    } catch {
        $stateTradeId = 0
    }
    try {
        $stateEntryCredit = [double]$ap.entry_credit
    } catch {
        $stateEntryCredit = 0.0
    }
    try {
        $qty = [double]$ap.qty
    } catch {
        $qty = 1.0
    }
    if ($qty -le 0) { $qty = 1.0 }

    try {
        $shortPut = [double]$ap.short_put_strike
        $shortCall = [double]$ap.short_call_strike
    } catch {
        $shortPut = 0.0
        $shortCall = 0.0
    }

    $journalTradeId = $null
    $journalEntryCredit = $null
    if ($journalOpen -ne $null) {
        $journalTradeId = $journalOpen.TradeId
        $journalEntryCredit = $journalOpen.EntryCredit
    }

    $slMult = Get-ConfiguredStopLossMult -RepoRoot $RepoRoot
    $slReal = $null
    if ($journalEntryCredit -ne $null) {
        $slReal = [Math]::Round(([double]$journalEntryCredit) * [double]$slMult, 4)
    }

    $discrepancyFlag = "N/A"
    if ($journalEntryCredit -ne $null) {
        $delta = [Math]::Abs(([double]$journalEntryCredit) - $stateEntryCredit)
        if ($delta -gt 0.01) {
            $discrepancyFlag = ("⚠️ MISMATCH: journal={0:0.00} vs state={1:0.00}" -f ([double]$journalEntryCredit), $stateEntryCredit)
        } else {
            $discrepancyFlag = "OK"
        }
    }

    return [pscustomobject]@{
        StateTradeId = $stateTradeId
        JournalTradeId = $journalTradeId
        StateEntryCredit = $stateEntryCredit
        JournalEntryCredit = $journalEntryCredit
        Qty = $qty
        ShortPut = $shortPut
        ShortCall = $shortCall
        SlMult = $slMult
        SlReal = $slReal
        DiscrepancyFlag = $discrepancyFlag
    }
}

function Get-PositionContext {
    param(
        [string]$RepoRoot,
        [double]$LastXsp,
        [double]$LastPnl
    )

    $tradeSnap = Get-ActiveTradeSnapshot -RepoRoot $RepoRoot
    if ($tradeSnap -eq $null) {
        return $null
    }

    try {
        $entryCredit = [double]$tradeSnap.StateEntryCredit
        $qty = [double]$tradeSnap.Qty
        $shortPut = [double]$tradeSnap.ShortPut
        $shortCall = [double]$tradeSnap.ShortCall

        $distPut = [Math]::Round($LastXsp - $shortPut, 2)
        $distCall = [Math]::Round($shortCall - $LastXsp, 2)
        $minDistance = [Math]::Round([Math]::Min($distPut, $distCall), 2)

        # For short IC: close cost ~ entry_credit - realized open profit per-share
        $spreadMid = [Math]::Round([Math]::Max($entryCredit - ($LastPnl / (100.0 * $qty)), 0.0), 4)

        $now = Get-Date
        $day = $now.ToString("yyyy-MM-dd")
        $t1600 = Get-Date "$day 16:00:00"
        $t1615 = Get-Date "$day 16:15:00"
        $tte1600 = [Math]::Round([Math]::Max(($t1600 - $now).TotalHours, 0.0), 2)
        $tte1615 = [Math]::Round([Math]::Max(($t1615 - $now).TotalHours, 0.0), 2)

        $corrSpread = $null
        $corrSlPct = $null
        if ($tradeSnap.JournalEntryCredit -ne $null) {
            $corrSpread = [Math]::Round([Math]::Max(([double]$tradeSnap.JournalEntryCredit) - ($LastPnl / (100.0 * $qty)), 0.0), 4)
            if ($tradeSnap.SlReal -ne $null -and [double]$tradeSnap.SlReal -gt 0) {
                $corrSlPct = [Math]::Round((100.0 * $corrSpread / [double]$tradeSnap.SlReal), 1)
            }
        }

        return [pscustomobject]@{
            TradeId = $tradeSnap.StateTradeId
            JournalTradeId = $tradeSnap.JournalTradeId
            EntryCredit = $entryCredit
            EntryCreditState = $tradeSnap.StateEntryCredit
            EntryCreditJournal = $tradeSnap.JournalEntryCredit
            Qty = $qty
            DistPut = $distPut
            DistCall = $distCall
            MinDistance = $minDistance
            SpreadMid = $spreadMid
            CorrSpread = $corrSpread
            CorrSlPct = $corrSlPct
            SlReal = $tradeSnap.SlReal
            DiscrepancyFlag = $tradeSnap.DiscrepancyFlag
            Tte1600 = $tte1600
            Tte1615 = $tte1615
        }
    } catch {
        return $null
    }
}

function Ensure-TradeContextHeader {
    param(
        [string]$DailyLogPath,
        [string]$TradeHeaderStateFile,
        [psobject]$TradeSnapshot
    )

    if ($TradeSnapshot -eq $null) {
        return $false
    }

    $tradeId = [int]$TradeSnapshot.StateTradeId
    if ($tradeId -le 0 -and $TradeSnapshot.JournalTradeId -ne $null) {
        $tradeId = [int]$TradeSnapshot.JournalTradeId
    }
    if ($tradeId -le 0) {
        return $false
    }

    $lastSeenTradeId = 0
    if (Test-Path $TradeHeaderStateFile) {
        try {
            $rawSeen = (Get-Content $TradeHeaderStateFile -ErrorAction Stop | Select-Object -First 1).Trim()
            if ($rawSeen) {
                $lastSeenTradeId = [int]$rawSeen
            }
        } catch {
            $lastSeenTradeId = 0
        }
    }

    if ($tradeId -eq $lastSeenTradeId) {
        return $false
    }

    $entryJournalTxt = "N/A"
    if ($TradeSnapshot.JournalEntryCredit -ne $null) {
        $entryJournalTxt = [string]::Format("{0:0.00}", [double]$TradeSnapshot.JournalEntryCredit)
    }
    $entryStateTxt = [string]::Format("{0:0.00}", [double]$TradeSnapshot.StateEntryCredit)
    $slRealTxt = "N/A"
    if ($TradeSnapshot.SlReal -ne $null) {
        $slRealTxt = [string]::Format("{0:0.00}", [double]$TradeSnapshot.SlReal)
    }
    $discFlag = [string]$TradeSnapshot.DiscrepancyFlag
    if ([string]::IsNullOrWhiteSpace($discFlag)) {
        $discFlag = "N/A"
    }

    $stamp = Get-Date -Format "HH:mm"
    $headerLines = @(
        "",
        "## Trade Context",
        "- detected_at: $stamp",
        "- trade_id_state: $($TradeSnapshot.StateTradeId)",
        "- trade_id_journal_open: $(if ($TradeSnapshot.JournalTradeId -ne $null) { $TradeSnapshot.JournalTradeId } else { 'N/A' })",
        "- entry_credit_journal: $entryJournalTxt",
        "- entry_credit_state: $entryStateTxt",
        "- discrepancy_flag: $discFlag",
        "- sl_mult_config: $([string]::Format('{0:0.##}', [double]$TradeSnapshot.SlMult))"
        "- sl_real: $slRealTxt"
    )
    Add-Content -Path $DailyLogPath -Value ($headerLines -join [Environment]::NewLine) -Encoding UTF8
    Set-Content -Path $TradeHeaderStateFile -Value $tradeId -Encoding ASCII
    return $true
}

function Append-BitacoraUpdate {
    param(
        [string]$DailyLogPath,
        [string]$Summary,
        [string]$MarketLine,
        [string]$AttemptsLine,
        [string]$RiskLine,
        [string]$StatusLine,
        [int]$WindowMinutes,
        [string]$ContextLine = ""
    )
    $stamp = Get-Date -Format "HH:mm"
    $entryLines = @(
        "",
        "### $stamp - Update monitor (+$WindowMinutes min)",
        "- Resumen ventana $WindowMinutes min: $Summary",
        "- Lectura mercado: $MarketLine",
        "- Intentos de apertura / fallas: $AttemptsLine",
        "- Riesgo intraminuto: $RiskLine",
        "- Estado: $StatusLine"
    )
    if ($ContextLine -and $ContextLine.Trim().Length -gt 0) {
        $entryLines += "- Contexto cuantitativo: $ContextLine"
    }
    $entry = $entryLines -join [Environment]::NewLine
    Add-Content -Path $DailyLogPath -Value $entry -Encoding UTF8
}

if (-not (Test-Path $outputsDir)) {
    New-Item -Path $outputsDir -ItemType Directory | Out-Null
}

Set-Content -Path $pidFile -Value $PID -Encoding ASCII
Add-Content -Path $runnerLog -Value ("[{0}] started pid={1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $PID) -Encoding UTF8

$lastKey = ""
if (Test-Path $stateFile) {
    try {
        $lastKey = (Get-Content $stateFile -ErrorAction Stop | Select-Object -First 1).Trim()
    } catch {
        $lastKey = ""
    }
}

while ($true) {
    try {
        $windowMin = [Math]::Max(3, [int][Math]::Ceiling($IntervalSec / 60.0))
        $today = Get-Date -Format "yyyy_MM_dd"
        $dailyLogPath = Join-Path $RepoRoot ("docs\daily_log_{0}.md" -f $today)
        if (-not (Test-Path $dailyLogPath)) {
            New-Item -Path $dailyLogPath -ItemType File | Out-Null
        }

        $activeTradeSnapshot = Get-ActiveTradeSnapshot -RepoRoot $RepoRoot
        $headerWritten = Ensure-TradeContextHeader `
            -DailyLogPath $dailyLogPath `
            -TradeHeaderStateFile $tradeHeaderStateFile `
            -TradeSnapshot $activeTradeSnapshot
        if ($headerWritten) {
            Add-Content -Path $runnerLog -Value ("[{0}] trade-context-header trade_id={1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $activeTradeSnapshot.StateTradeId) -Encoding UTF8
        }

        $logPath = Get-LatestMonitorLog
        if (-not $logPath) {
            Start-Sleep -Seconds ([Math]::Min($IntervalSec, 30))
            continue
        }

        $now = Get-Date
        $windowStart = $now.AddMinutes(-$windowMin)
        $rows = @(Parse-WindowRows -LogPath $logPath -WindowStart $windowStart)

        $execLogPath = Get-LatestExecutionLog
        $events = @()
        if ($execLogPath) {
            $events = @(Parse-ExecutionEvents -LogPath $execLogPath -WindowStart $windowStart)
        }

        if ($rows.Count -eq 0) {
            if ($events.Count -gt 0) {
                $attemptCount = @($events | Where-Object { $_.Attempt }).Count
                $failCount = @($events | Where-Object { $_.Fail }).Count
                $lastEvent = $events[$events.Count - 1]
                $key = "event|{0}|{1}|{2}" -f $lastEvent.Time.ToString("yyyy-MM-dd HH:mm:ss"), $attemptCount, $failCount

                if ($key -ne $lastKey) {
                    $attemptsLine = "intentos=$attemptCount | fallidos=$failCount"
                    $lastFail = @($events | Where-Object { $_.Fail } | Select-Object -Last 1)
                    if ($lastFail.Count -gt 0) {
                        $raw = [string]$lastFail[0].Raw
                        if ($raw.Length -gt 180) {
                            $raw = $raw.Substring(0, 180) + "..."
                        }
                        $attemptsLine = "$attemptsLine | ultimo fallo: $raw"
                    }

                    $summary = "sin posicion activa; sin snapshot PnL en ventana."
                    $marketLine = "pre-entrada: esperando ventana operativa o setup valido."
                    $riskLine = "N/A (sin filas de monitor de posicion)"
                    $statusLine = "sin posicion abierta; monitor en modo escaneo."

                    Append-BitacoraUpdate `
                        -DailyLogPath $dailyLogPath `
                        -Summary $summary `
                        -MarketLine $marketLine `
                        -AttemptsLine $attemptsLine `
                        -RiskLine $riskLine `
                        -StatusLine $statusLine `
                        -WindowMinutes $windowMin `
                        -ContextLine "DANGER se dispara con min_distance < 5.00 pts (sin posicion activa en ventana)."

                    Set-Content -Path $stateFile -Value $key -Encoding ASCII
                    $lastKey = $key
                    Add-Content -Path $runnerLog -Value ("[{0}] event-update attempts={1} fails={2}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $attemptCount, $failCount) -Encoding UTF8
                }
            }

            Start-Sleep -Seconds ([Math]::Min($IntervalSec, 30))
            continue
        }

        $first = $rows[0]
        $last = $rows[$rows.Count - 1]
        $key = "{0}|{1}|{2}" -f $last.Time.ToString("HH:mm:ss"), $last.XSP, $last.PnL
        if ($key -eq $lastKey) {
            Start-Sleep -Seconds $IntervalSec
            continue
        }

        $minPnl = ($rows | Measure-Object -Property PnL -Minimum).Minimum
        $maxPnl = ($rows | Measure-Object -Property PnL -Maximum).Maximum
        $dangerCount = @($rows | Where-Object { $_.Danger }).Count
        $xspMove = $last.XSP - $first.XSP
        $vixMove = $last.VIX - $first.VIX
        $marketLine = Build-MarketComment -XspMove $xspMove -VixMove $vixMove -DangerCount $dangerCount -TotalRows $rows.Count

        $summary = "inicio {0} -> fin {1} | XSP {2} | VIX {3} | PnL ult={4} rango=[{5}, {6}]" -f `
            $first.Time.ToString("HH:mm:ss"), `
            $last.Time.ToString("HH:mm:ss"), `
            $last.XSP, `
            $last.VIX, `
            ([string]::Format("{0:0.00}", $last.PnL)), `
            ([string]::Format("{0:0.00}", $minPnl)), `
            ([string]::Format("{0:0.00}", $maxPnl))

        $attemptsLine = "sin intentos ni fallas detectadas en ventana."
        if ($events.Count -gt 0) {
            $attemptCount = @($events | Where-Object { $_.Attempt }).Count
            $failCount = @($events | Where-Object { $_.Fail }).Count
            $attemptsLine = "intentos=$attemptCount | fallidos=$failCount"
            $lastFail = @($events | Where-Object { $_.Fail } | Select-Object -Last 1)
            if ($lastFail.Count -gt 0) {
                $raw = [string]$lastFail[0].Raw
                if ($raw.Length -gt 180) {
                    $raw = $raw.Substring(0, 180) + "..."
                }
                $attemptsLine = "$attemptsLine | ultimo fallo: $raw"
            }
        }

        $riskLine = "{0}/{1} ticks con DANGER" -f $dangerCount, $rows.Count
        $statusLine = "posicion sigue abierta; sin evento de cierre detectado en log."
        $contextLine = "DANGER se dispara con min_distance < 5.00 pts."
        $positionContext = Get-PositionContext -RepoRoot $RepoRoot -LastXsp $last.XSP -LastPnl $last.PnL
        if ($positionContext -ne $null) {
            $distPutTxt = [string]::Format("{0:+0.00;-0.00;0.00}", $positionContext.DistPut)
            $distCallTxt = [string]::Format("{0:+0.00;-0.00;0.00}", $positionContext.DistCall)
            $minDistTxt = [string]::Format("{0:0.00}", $positionContext.MinDistance)
            $spreadMidTxt = [string]::Format("{0:0.0000}", $positionContext.SpreadMid)
            $contextLine = "DANGER<5.00 pts | dist_put=$distPutTxt | dist_call=$distCallTxt | min_dist=$minDistTxt | TTE 16:00=$($positionContext.Tte1600)h / 16:15=$($positionContext.Tte1615)h | spread_mid_est=$spreadMidTxt"
            if ($positionContext.CorrSpread -ne $null) {
                $corrSprdTxt = [string]::Format("{0:0.0000}", [double]$positionContext.CorrSpread)
                $contextLine = "$contextLine | CorrSprd=$corrSprdTxt"
                if ($positionContext.CorrSlPct -ne $null) {
                    $slPctTxt = [string]::Format("{0:0.0}%", [double]$positionContext.CorrSlPct)
                    $contextLine = "$contextLine | SL%=$slPctTxt"
                }
            }
            if ($last.POtm -ne $null) {
                $pOtmTxt = [string]::Format("{0:0.00}", [double]$last.POtm)
                if ($last.POtmGamma) {
                    $pOtmTxt = "$pOtmTxt(γ!)"
                }
                $contextLine = "$contextLine | P_OTM=$pOtmTxt"
            }
        }
        Append-BitacoraUpdate `
            -DailyLogPath $dailyLogPath `
            -Summary $summary `
            -MarketLine $marketLine `
            -AttemptsLine $attemptsLine `
            -RiskLine $riskLine `
            -StatusLine $statusLine `
            -WindowMinutes $windowMin `
            -ContextLine $contextLine

        Set-Content -Path $stateFile -Value $key -Encoding ASCII
        $lastKey = $key
        Add-Content -Path $runnerLog -Value ("[{0}] update {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $summary) -Encoding UTF8
    } catch {
        $lineNo = $_.InvocationInfo.ScriptLineNumber
        $line = $_.InvocationInfo.Line.Trim()
        Add-Content -Path $runnerLog -Value ("[{0}] error line={1} msg={2} src={3}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $lineNo, $_.Exception.Message, $line) -Encoding UTF8
    }

    Start-Sleep -Seconds $IntervalSec
}
