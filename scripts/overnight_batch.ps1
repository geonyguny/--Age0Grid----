# =======================
# Overnight DECUM Batch (refactored v2: invariant args, empty-argv guard)
# =======================

[CmdletBinding()]
param(
  [string]   $Python       = ".\.venv\Scripts\python.exe",
  [string]   $CliMod       = "project.runner.cli",
  [switch]   $IncludeRL    = $false,
  [int]      $NPathsHi     = 2000,
  [int[]]    $Seeds        = @(11,12,13,14,15),
  [double[]] $ThetaList    = @(0.0,0.5,1.0,2.0)
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Note($msg){ Write-Host "[Note] $msg" -ForegroundColor Cyan }
function Ok($msg){ Write-Host "[OK] $msg" -ForegroundColor Green }
function Warn($msg){ Write-Host "[WARN] $msg" -ForegroundColor Yellow }

# Invariant stringifier for numeric args (0.5 → "0.5" regardless of locale)
$CI = [System.Globalization.CultureInfo]::InvariantCulture
function S($x){
  if ($null -eq $x) { return "" }
  if ($x -is [double] -or $x -is [single] -or $x -is [decimal]) {
    return [string]::Format($CI, "{0}", $x)
  }
  return [string]$x
}

# Run: hard guard against empty argv; echo exact command line
function Run([string[]]$argv){
  if (-not $argv -or $argv.Count -eq 0) { throw "Empty argv passed to Run()" }
  $cmdShown = "$Python " + ($argv -join ' ')
  Write-Host ">> $cmdShown" -ForegroundColor Gray
  & $Python @argv
  if ($LASTEXITCODE -ne 0) { throw "Command failed ($LASTEXITCODE): $cmdShown" }
}

# --- Preflight ---
Note "Please close any Excel files to avoid write conflicts."
if (-not (Test-Path ".\.venv\Scripts\python.exe")) { throw "Python venv not found: .\.venv\Scripts\python.exe" }
if ([string]::IsNullOrWhiteSpace($CliMod)) { throw "Cli module name is empty." }
if (-not (Test-Path ".\outputs")) { New-Item -ItemType Directory -Path .\outputs | Out-Null }
if (-not (Test-Path ".\outputs\_logs")) { New-Item -ItemType Directory -Path .\outputs\_logs | Out-Null }

# common: precompute "n?k" for tags
$nk = [int][math]::Round([double]$NPathsHi / 1000.0)

# --- 0) Snapshot (OAT_/2D_/AMB_) ---
if (Test-Path ".\outputs\_logs\metrics.csv") {
  Note "Build dev_metrics_snapshot.csv (OAT_*, 2D_*, AMB_*)"
  $M = Import-Csv .\outputs\_logs\metrics.csv | Where-Object {
    $_.tag -like "OAT_*" -or $_.tag -like "2D_*" -or $_.tag -like "AMB_*"
  }
  if ($M) {
    $M = $M | Group-Object tag,method,seed | ForEach-Object { $_.Group | Select-Object -Last 1 }
    $M | Export-Csv .\outputs\dev_metrics_snapshot.csv -NoTypeInformation
    Ok "outputs\dev_metrics_snapshot.csv"
  } else {
    Warn "No OAT_/2D_/AMB_ rows yet in _logs\metrics.csv"
  }
} else {
  Warn "metrics.csv not found yet; continuing."
}

# --- 1) High-precision OAT/2D ---
Note "High-precision re-eval for OAT (RL, hedge_sigma_k in {0,0.5,1.0})"
if ($IncludeRL) {
  foreach ($h in @(0.0, 0.5, 1.0)) {
    Run @(
      "-m", $CliMod,
      "--method","rl","--data_profile","dev","--market_mode","bootstrap",
      "--hedge","on","--hedge_mode","sigma","--hedge_sigma_k", (S $h),
      "--n_paths", (S $NPathsHi),
      "--tag", ("OAT_h{0}_n{1}k" -f (S $h), $nk),
      "--print_mode","summary","--autosave","on"
    )
  }
} else {
  Warn "RL block skipped (set -IncludeRL to enable)."
}

Note "High-precision re-eval for 2D (HJB)"
$Pairs = @("0.2,0.0","0.2,0.5","0.2,1.0","0.6,0.0","0.6,0.5","0.6,1.0")
foreach ($p in $Pairs) {
  $sp = $p.Split(",")
  $u  = [double]$sp[0]
  $h  = [double]$sp[1]
  $mix = "0.0,{0},{1}" -f (S $u), (S (1.0 - $u))
  Run @(
    "-m", $CliMod,
    "--method","hjb","--data_profile","dev","--market_mode","bootstrap",
    "--alpha_mix", $mix,
    "--hedge","on","--hedge_mode","sigma","--hedge_sigma_k",(S $h),
    "--n_paths",(S $NPathsHi),
    "--tag", ("2D_us{0}_h{1}_n{2}k" -f (S $u), (S $h), $nk),
    "--print_mode","summary","--autosave","on"
  )
}

# --- 2) Ambiguity sweep ---
Note "Theta ambiguity sweep (HJB)"
foreach ($t in $ThetaList) {
  foreach ($s in $Seeds) {
    Run @(
      "-m", $CliMod,
      "--method","hjb","--data_profile","dev","--market_mode","bootstrap",
      "--theta_ambiguity",(S $t),"--seed",(S $s),
      "--tag",("AMB_t{0}_s{1}" -f (S $t),(S $s)),
      "--print_mode","summary","--autosave","on"
    )
  }
}

# --- 3) Summaries & Reports ---
Note "Build _summary_scored.csv (prefer snapshot if exists)"
if (Test-Path ".\outputs\dev_metrics_snapshot.csv") {
  Run @(".\scripts\score_metrics.py","--src",".\outputs\dev_metrics_snapshot.csv","--out",".\outputs\_summary_scored.csv")
} elseif (Test-Path ".\outputs\_logs\metrics.csv") {
  Run @(".\scripts\score_metrics.py","--src",".\outputs\_logs\metrics.csv","--out",".\outputs\_summary_scored.csv")
} else {
  Warn "No metrics source for scoring."
}

Run @(".\scripts\make_decum_report.py","--tag","overnight","--frontier_topn","50")

if (Test-Path ".\outputs\_summary_scored.csv") {
  Run @(".\scripts\make_oat_lines.py","--tag_startswith","OAT_","--xcol","hedge_sigma_k","--metrics","EW,ES95")
  Run @(".\scripts\make_heatmaps.py","--src",".\outputs\_summary_scored.csv","--tag_startswith","2D_","--x","mix_us","--y","hedge_sigma_k","--zlist","EW,ES95,CompositeScore")
}

Run @(".\scripts\compare_dev_full.py","--abs_tol","0.03","--rel_tol","0.04")

# --- 4) Final snapshot (post batch) ---
if (Test-Path ".\outputs\_logs\metrics.csv") {
  Note "Refresh dev_metrics_snapshot.csv after batch"
  $M2 = Import-Csv .\outputs\_logs\metrics.csv | Where-Object {
    $_.tag -like "OAT_*" -or $_.tag -like "2D_*" -or $_.tag -like "AMB_*"
  }
  if ($M2) {
    $M2 = $M2 | Group-Object tag,method,seed | ForEach-Object { $_.Group | Select-Object -Last 1 }
    $M2 | Export-Csv .\outputs\dev_metrics_snapshot.csv -NoTypeInformation
    Ok "outputs\dev_metrics_snapshot.csv (refreshed)"
  }
}

Ok "Overnight batch finished."
[console]::beep(1000,300)
