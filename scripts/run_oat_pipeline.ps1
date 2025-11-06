# scripts\run_oat_pipeline.ps1
param(
  # 런 프로파일
  [ValidateSet('dev','overnight')] [string]$Mode = 'dev',

  # A) σ-헤지 스윕
  [string]$HedgeValues = "0,0.5,1.0",
  [string]$HedgeSeeds  = "11,12",

  # B) κ 재학습(bias 계열)
  [string]$BiasValues  = "1.0 1.5 2.0",
  [string]$ExtraBias   = "--rl_epochs 10 --rl_steps_per_epoch 3000 --entropy_coef 0.0 --teacher_eps0 0.10 --teacher_decay 0.98",

  # C) Policy-Locked 평가
  [string]$LockedValues = "1.0 2.0",

  # D) 점수화
  [string]$ScoreWeights = "0.6,0.4",
  [ValidateSet('wealth','loss')] [string]$EsMode = 'wealth',

  # DryRun(모든 단계 드라이런)
  [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ScriptDir   = $PSScriptRoot
$ProjectRoot = Split-Path -Parent $ScriptDir
Set-Location $ProjectRoot

$run = Join-Path $ScriptDir 'run_batch_oat.ps1'

Write-Host "=== [A] Hedge sweep ===" -ForegroundColor Cyan
& $run -Var hedge_sigma_k -Values $HedgeValues -Method rl -Mode $Mode -Seeds $HedgeSeeds -DryRun:$DryRun

Write-Host "=== [B] Bias retrain ===" -ForegroundColor Cyan
& $run -Var bias_loss_aversion -Values $BiasValues -Method rl -Mode $Mode -CliMode rl -Extra $ExtraBias -DryRun:$DryRun

Write-Host "=== [C] Policy-Locked report ===" -ForegroundColor Cyan
& $run -Var bias_loss_aversion -Values $LockedValues -Method rl -Mode $Mode -PolicyLocked -DryRun:$DryRun

# D) 스냅샷·점수화·퀵리포트 (DryRun이면 스킵)
if (-not $DryRun) {
  $Out = '.\outputs'
  $Log = Join-Path $Out '_logs'
  $snap = Join-Path $Out 'DEV_metrics_snapshot.csv'
  Write-Host "=== [D] Snapshot & Scoring & QuickReport ===" -ForegroundColor Cyan

  # 최신 태그 한 줄씩만 추려 스냅샷
  (Import-Csv (Join-Path $Log 'metrics.csv') | Where-Object { $_.tag -like 'DEV_OAT_*' } |
    Group-Object tag,method,seed | ForEach-Object { $_.Group | Select-Object -Last 1 }) |
    Export-Csv $snap -NoTypeInformation

  # RL/HJB 점수화
  python .\scripts\score_snapshot.py --src $snap --tag_startswith DEV_OAT_ `
    --metrics EW,ES95 --weights $ScoreWeights --method rl  --es_mode $EsMode --out inplace
  python .\scripts\score_snapshot.py --src $snap --tag_startswith DEV_OAT_ `
    --metrics EW,ES95 --weights $ScoreWeights --method hjb --es_mode $EsMode --out inplace

  # 퀵리포트
  python .\scripts\make_quick_report.py
}

Write-Host "[OK] OAT pipeline completed." -ForegroundColor Green
