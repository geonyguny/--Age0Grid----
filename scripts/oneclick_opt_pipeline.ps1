<# =====================================================================
 oneclick_opt_pipeline.ps1  (PS5-safe, ASCII-only messages)
 - Orchestrates the full pipeline by calling run_opt_design.ps1
 - Modes:
     defense = light profile, minimal coverage for 1st defense run
     heavy   = heavy profile, fuller coverage
 - All user-facing strings are ASCII to avoid parser issues on PS5.

 Save as UTF-8 with BOM.
===================================================================== #>

[CmdletBinding()]
param(
  [ValidateSet('defense','heavy')] [string]$Mode = 'defense',
  [ValidateSet('dev','full')]      [string]$DataProfile = 'dev',

  [double]$KR_MinLiving_Monthly = 1500000,  # KRW per month (real)
  [double]$InitWealth_Real      = 5e8,      # initial real wealth (KRW)

  # Optional overrides
  [string]$Seeds     = '',     # e.g., "7,9,11,13,17"; if empty use defaults per Mode
  [string]$PWGammas  = '',     # e.g., "0.85,0.80,0.70,0.60"; if empty use defaults per Mode

  # Paths
  [string]$ProjectRoot = '',   # if empty, use current
  [string]$RunScript   = '.\scripts\run_opt_design.ps1'
)

function Ensure-Dir($path) {
  if (-not (Test-Path $path)) { $null = New-Item -ItemType Directory -Force -Path $path }
}

# ----- Resolve project root -----
if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
  $ProjectRoot = (Get-Location).Path
}
Set-Location $ProjectRoot

# ----- Outputs & logs -----
$OUT   = Join-Path $ProjectRoot 'outputs'
$LOGS  = Join-Path $OUT '_logs'
$FIGS  = Join-Path $OUT 'figs'
Ensure-Dir $OUT; Ensure-Dir $LOGS; Ensure-Dir $FIGS

# ----- Mode presets -----
$profile = 'light'
$seedsStr = '7'
$pwStr    = '0.70,0.60'   # short set good for defense

if ($Mode -eq 'heavy') {
  $profile = 'heavy'
  $seedsStr = '7,9,11,13,17'
  $pwStr    = '0.85,0.80,0.70,0.60'
}

if (-not [string]::IsNullOrWhiteSpace($Seeds))    { $seedsStr = $Seeds }
if (-not [string]::IsNullOrWhiteSpace($PWGammas)) { $pwStr    = $PWGammas }

Write-Host "[OneClick] Starting pipeline" -ForegroundColor Cyan
Write-Host " - Mode         : $Mode"
Write-Host " - Profile      : $profile"
Write-Host " - DataProfile  : $DataProfile"
Write-Host " - Seeds        : $seedsStr"
Write-Host " - PWGammas     : $pwStr"
Write-Host " - Floor (KRW/m): $($KR_MinLiving_Monthly.ToString('N0'))"
Write-Host " - InitWealth   : $($InitWealth_Real.ToString('N0'))"
Write-Host " - ProjectRoot  : $ProjectRoot"
Write-Host " - Outputs      : $OUT"
Write-Host " - Logs         : $LOGS"
Write-Host " - Figs         : $FIGS"

# Safety notice (ASCII only)
Write-Host "[Guard] Close Excel files to avoid PermissionError on CSV/XLSX." -ForegroundColor Yellow

# ----- Run the full design pipeline -----
#   This script itself performs:
#   1D & 2D sweeps -> snapshot -> scoring -> OPT picks ->
#   behavioral overlays -> core tables/compare/top lists.
if (-not (Test-Path $RunScript)) {
  throw "Run script not found: $RunScript"
}

# Build argument list for run_opt_design.ps1
$runArgs = @(
  '-Profile', $profile,
  '-DataProfile', $DataProfile,
  '-Seeds', $seedsStr,
  '-KR_MinLiving_Monthly', $KR_MinLiving_Monthly,
  '-InitWealth_Real', $InitWealth_Real,
  '-PWGammas', $pwStr
)

Write-Host "[OneClick] Invoke: $RunScript $($runArgs -join ' ')" -ForegroundColor DarkCyan
& $RunScript @runArgs
if ($LASTEXITCODE -ne $null -and $LASTEXITCODE -ne 0) {
  Write-Warning "[OneClick] run_opt_design.ps1 returned non-zero exit code: $LASTEXITCODE"
}

# ----- Final hints (ASCII only) -----
$paperXlsx = Join-Path $OUT 'Paper_1D_2D_Tables.xlsx'  # optional if your Python makes it
Write-Host ""
Write-Host "[OneClick] Completed."
Write-Host "Artifacts (if produced by the downstream scripts):"
Write-Host " - Snapshots: outputs\DEV_metrics_snapshot.csv, OPT_metrics_snapshot.csv, BH_metrics_snapshot.csv"
Write-Host " - Summaries: outputs\*_summary.csv, *_summary_benefit.csv"
Write-Host " - Core tables: outputs\OPT_table_core.csv, outputs\BH_table_core.csv"
Write-Host " - Compare/top: outputs\OPT_BH_compare.csv, outputs\OPT_BH_top.csv"
Write-Host " - Figures dir: $FIGS (e.g., optimal_points.json, *.png)"
Write-Host " - Paper table: $paperXlsx (if your figure/table generator creates it)"
