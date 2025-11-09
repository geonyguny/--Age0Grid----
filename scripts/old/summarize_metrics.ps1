# scripts/summarize_metrics.ps1
param(
  [string]$Root = "D:\01_simul",
  [string]$Out  = ".\outputs",
  [string[]]$Tags = @("A_HJB_SMOKE","B_LOSS_ES","C_BEHAVIORAL_UTILITY","D_ACTION_BIAS","E_RULE_KGR_ANNUITY_MORT"),
  [string[]]$Prefixes = @("BU_","RL_SMOKE")
)

$ErrorActionPreference = "Stop"
Set-Location $Root
$OutputEncoding = [Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$MET = Join-Path $Out "_logs\metrics.csv"
if (-not (Test-Path $MET)) {
  Write-Host "[FAIL] metrics.csv not found: $MET" -ForegroundColor Red
  exit 1
}

Write-Host "=== Tail(20) ===" -ForegroundColor Cyan
Get-Content -Path $MET -Tail 20 | Write-Host

Write-Host "`n=== By Tag (explicit list) ===" -ForegroundColor Cyan
Import-Csv $MET |
  Where-Object { $_.tag -in $Tags } |
  Sort-Object ts |
  Format-Table ts,tag,method,es_mode,EW,ES95,EU,EU_per_year -AutoSize

Write-Host "`n=== By Prefix (BU_, RL_SMOKE, …) ===" -ForegroundColor Cyan
Import-Csv $MET |
  Where-Object {
    $t = $_.tag
    $match = $false
    foreach ($p in $Prefixes) { if ($t -like "$p*") { $match = $true; break } }
    $match
  } |
  Sort-Object ts |
  Format-Table ts,tag,EW,ES95,EU,EU_per_year,crra_gamma,u_scale -AutoSize
