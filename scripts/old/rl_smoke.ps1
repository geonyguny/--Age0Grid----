# scripts/rl_smoke.ps1
param(
  [string]$Root = "D:\01_simul",
  [string]$Out  = ".\outputs",
  [string]$Csv  = "D:\01_simul\project\data\market\kr_us_gold_bootstrap_mini.csv"
)

$ErrorActionPreference = "Stop"
Set-Location $Root
$OutputEncoding = [Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$MET = Join-Path $Out "_logs\metrics.csv"
if (-not (Test-Path (Split-Path $MET))) { New-Item -ItemType Directory -Force -Path (Split-Path $MET) | Out-Null }

$tag = "RL_SMOKE_E2_S2k"
Write-Host "=== RL smoke ($tag) ===" -ForegroundColor Cyan

py -m project.runner.cli `
  --asset KR --method rl `
  --market_mode bootstrap --market_csv $Csv --use_real_rf on `
  --outputs $Out --bands on `
  --es_mode wealth --rl_n_paths_eval 64 `
  --rl_epochs 2 --rl_steps_per_epoch 2000 `
  --lr 0.0007 --gae_lambda 0.95 --entropy_coef 0.01 --value_coef 0.5 --max_grad_norm 0.5 `
  --seeds 0 `
  --tag $tag --print_mode summary --quiet on

Write-Host "`n=== RL metrics (tail) ===" -ForegroundColor Green
if (Test-Path $MET) {
  Get-Content -Path $MET -Tail 10 | Write-Host
  "`n-- Filter RL row --" | Write-Host
  Import-Csv $MET |
    Where-Object { $_.tag -eq $tag } |
    Sort-Object ts |
    Format-Table ts,tag,method,EW,ES95,mean_WT,best_epoch,train_time_s,eval_time_s -AutoSize
} else {
  Write-Host "[WARN] metrics.csv not found: $MET" -ForegroundColor DarkYellow
}
