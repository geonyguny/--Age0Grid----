# scripts/plot_frontier.ps1
param(
  [string]$Root = "D:\01_simul",
  [string]$Out  = ".\outputs"
)
$ErrorActionPreference = "Stop"
Set-Location $Root
$OutputEncoding = [Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$MET = Join-Path $Out "_logs\metrics.csv"
if (-not (Test-Path $MET)) {
  Write-Host "[FAIL] metrics.csv not found: $MET" -ForegroundColor Red
  exit 1
}

Write-Host "=== Build EW–ES95 frontier ===" -ForegroundColor Cyan
# project.eval.plot_frontier_from_csv 사용
py - << 'PYCODE'
from project.eval import plot_frontier_from_csv
import sys, os
csv = r"""%METRICS%"""
png = plot_frontier_from_csv(csv)
print(png or "no_points")
PYCODE
# 위 heredoc 안에서 %METRICS% 치환
(Get-Content -Raw -Path $PSCommandPath) | Out-Null
