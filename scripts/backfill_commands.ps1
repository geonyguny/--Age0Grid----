#requires -Version 5.1
[CmdletBinding()]
param(
  [switch]$DryRun,            # print plan only; do not run
  [switch]$RebuildStage3,     # rebuild snapshot/score/tables only
  [int]$LookbackMinutes = 180,# scan recent logs (minutes)
  [string]$TagsRegex = '.*'   # whitelist filter for tags (regex)
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Write-Info ($msg) { Write-Host "[INFO] $msg" -ForegroundColor Cyan }
function Write-Warn ($msg) { Write-Host "[WARN] $msg" -ForegroundColor Yellow }
function Write-Err  ($msg) { Write-Host "[ERR]  $msg" -ForegroundColor Red }

# workspace paths
$Root   = (Get-Location).Path
$OutDir = Join-Path $Root 'outputs'
$Logs   = Join-Path $OutDir '_logs'
$null   = New-Item -ItemType Directory -Force -Path $OutDir,$Logs | Out-Null

# console/file encoding hints (UTF-8, no BOM required)
try { chcp 65001 | Out-Null } catch {}
$OutputEncoding           = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)

# helpers
function As-Double($x){ try { [double]$x } catch { $null } }
function Sort-ByScoreDesc($rows){
  $rows | Sort-Object @{Expression={ As-Double $_.CompositeScore_benefit }; Descending=$true }
}

# paths
$PyExe     = ".\.venv\Scripts\python.exe"
$CliModule = "project.runner.cli"
$ScorePy   = ".\scripts\score_snapshot.py"

$MetricsCsv = Join-Path $Logs   'metrics.csv'
$Snapshot   = Join-Path $OutDir 'DEV_metrics_snapshot.csv'
$ScoredOut  = Join-Path $OutDir 'DEV_scored.csv'

$OPT_Sum = Join-Path $OutDir 'OPT_summary_benefit.csv'
$BH_Sum  = Join-Path $OutDir 'BH_summary_benefit.csv'
$OPT_Core= Join-Path $OutDir 'OPT_table_core.csv'
$BH_Core = Join-Path $OutDir 'BH_table_core.csv'
$Compare = Join-Path $OutDir 'OPT_BH_compare.csv'
$TopBoth = Join-Path $OutDir 'OPT_BH_top.csv'

if ($RebuildStage3) { Write-Info "Stage3-only rebuild requested." }

# 1) scan recent logs for tags with errors/warnings
$Cutoff = (Get-Date).AddMinutes(-[math]::Abs($LookbackMinutes))
$LogFiles = Get-ChildItem $Logs -Filter *.log -ErrorAction SilentlyContinue |
           Where-Object { $_.LastWriteTime -ge $Cutoff }

$BadTags = @()
foreach ($lf in $LogFiles) {
  $lines = Get-Content $lf.FullName -ErrorAction SilentlyContinue
  if (-not $lines) { continue }

  $hasErr  = $lines | Select-String -Pattern '^Traceback|\[ERR\]|^Exception\b|^\s*ERROR\b'
  $hasWarn = $lines | Select-String -Pattern 'unrecognized|not found|invalid argument|^WARNING\b|\[WARN\]'

  if ($hasErr -or $hasWarn) {
    $tag = [IO.Path]::GetFileNameWithoutExtension($lf.Name)
    if ($tag -match $TagsRegex) { $BadTags += $tag }
  }
}
$BadTags = $BadTags | Sort-Object -Unique

if (-not $BadTags -and -not $RebuildStage3) {
  Write-Info "No errored/warned tags to backfill."
} else {
  if ($BadTags) { Write-Info ("Backfill candidates: {0}" -f ($BadTags -join ', ')) }
  else          { Write-Info "No backfill candidates; Stage3 rebuild only." }
}

if ($DryRun) { Write-Info "DryRun: printing planned commands." }

# 2) re-run by tag (idempotent: last row per tag wins)
if (-not $DryRun) {
  foreach ($tag in $BadTags) {
    try {
      Write-Info ("Re-run tag: {0}" -f $tag)
      $psi = @{
        FilePath     = $PyExe
        ArgumentList = @('-m', $CliModule,
                         '--mode','rl','--method','rl',
                         '--print_mode','metrics',
                         '--tag', $tag)
        NoNewWindow  = $true
        Wait         = $true
        PassThru     = $true
      }
      $p = Start-Process @psi
      if ($p.ExitCode -ne 0) {
        Write-Warn ("Non-zero exit for tag {0}: {1}" -f $tag, $p.ExitCode)
      }
    } catch {
      Write-Err  ("Failed to re-run tag {0}: {1}" -f $tag, $_.Exception.Message)
    }
  }
} else {
  foreach ($tag in $BadTags) {
    Write-Host ("python -m {0} --mode rl --method rl --print_mode metrics --tag {1}" -f $CliModule, $tag)
  }
}

# 3) Stage3: snapshot / score / tables
Write-Info ("snapshot => {0}" -f $Snapshot)

if (Test-Path $MetricsCsv) {
  (Import-Csv $MetricsCsv |
    Group-Object tag,method,sex |
    ForEach-Object { $_.Group | Select-Object -Last 1 }) |
    Export-Csv $Snapshot -NoTypeInformation -Encoding UTF8
} else {
  Write-Warn ("metrics.csv not found: {0}" -f $MetricsCsv)
}

if ( (Test-Path $PyExe) -and (Test-Path $ScorePy) -and (Test-Path $Snapshot) ) {
  & $PyExe $ScorePy --src $Snapshot --metrics "EW,ES95" --weights "0.6,0.4" --es_mode wealth --out $ScoredOut
  Write-Info ("scored => {0}" -f $ScoredOut)
} else {
  Write-Warn "Skip scoring (python/score.py/snapshot missing)."
}

$opt = $null; $bh = $null

if (Test-Path $OPT_Sum) {
  $opt = Import-Csv $OPT_Sum | Select-Object tag,EW,ES95,Ruin,CompositeScore_benefit
  if ($opt -and $opt.Count -gt 0) {
    $opt | Sort-Object @{Expression={ As-Double $_.CompositeScore_benefit }; Descending=$true} |
      Export-Csv $OPT_Core -NoTypeInformation -Encoding UTF8
    Write-Info ("core(opt) => {0}" -f $OPT_Core)
  } else {
    Write-Warn "OPT_summary_benefit.csv exists but has no rows — skip OPT core."
    $opt = $null
  }
} else {
  Write-Warn "OPT_summary_benefit.csv not found — skip OPT core."
}

if (Test-Path $BH_Sum) {
  $bh = Import-Csv $BH_Sum | Select-Object tag,EW,ES95,Ruin,CompositeScore_benefit
  if ($bh -and $bh.Count -gt 0) {
    $bh | Sort-Object @{Expression={ As-Double $_.CompositeScore_benefit }; Descending=$true} |
      Export-Csv $BH_Core -NoTypeInformation -Encoding UTF8
    Write-Info ("core(bh) => {0}" -f $BH_Core)
  } else {
    Write-Warn "BH_summary_benefit.csv exists but has no rows — skip BH core."
    $bh = $null
  }
} else {
  Write-Warn "BH_summary_benefit.csv not found — skip BH core."
}

if ($opt -and $bh) {
  $cols = @('tag','EW','ES95','Ruin','CompositeScore_benefit')
  $opt2 = $opt | Select-Object -Property ($cols + @(@{ Name='Room'; Expression={ 'OPT' } }))
  $bh2  = $bh  | Select-Object -Property ($cols + @(@{ Name='Room'; Expression={ 'BH' } }))

  $all = $opt2 + $bh2
  if ($all -and $all.Count -gt 0) {
    $all | Sort-Object Room, @{Expression={ As-Double $_.CompositeScore_benefit }; Descending=$true} |
      Export-Csv $Compare -NoTypeInformation -Encoding UTF8
    Write-Info ("compare => {0}" -f $Compare)

    $top = @()
    $top += ($opt2 | Sort-ByScoreDesc | Select-Object -First 3)
    $top += ($bh2  | Sort-ByScoreDesc | Select-Object -First 3)
    if ($top.Count -gt 0) {
      $top | Export-Csv $TopBoth -NoTypeInformation -Encoding UTF8
      Write-Info ("top3+top3 => {0}" -f $TopBoth)
    } else {
      Write-Warn "top skipped (no rows after sorting)."
    }
  } else {
    Write-Warn "compare/top skipped (no rows)."
  }
} else {
  Write-Warn "compare/top skipped (OPT or BH summary missing)."
}

Write-Host "[DONE] backfill complete" -ForegroundColor Green
