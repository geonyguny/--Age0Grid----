# scripts/verify_pipeline.ps1
#Requires -Version 5.1
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Write-Title($text) {
  Write-Host ""
  Write-Host "=== $text ===" -ForegroundColor Cyan
}

function Write-Section($text) {
  Write-Host ""
  Write-Host "---- $text ----" -ForegroundColor DarkCyan
}

# 루트 고정
$Root = Split-Path $PSScriptRoot -Parent
Set-Location $Root

# 경로 정의
$Py = "py"
$Cli = "project.runner.cli"
$Outputs = Join-Path $Root "outputs"
$LogsDir = Join-Path $Outputs "_logs"
$Metrics = Join-Path $LogsDir "metrics.csv"
$MarketCsv = Join-Path $Root "project\data\market\kr_us_gold_bootstrap_mini.csv"
$MortCsv   = Join-Path $Root "project\data\kidi_qx.csv"

# 보장
New-Item -ItemType Directory -Force -Path $Outputs | Out-Null
New-Item -ItemType Directory -Force -Path $LogsDir | Out-Null

function Wait-ForMetrics([int]$TimeoutSec = 30) {
  $sw = [Diagnostics.Stopwatch]::StartNew()
  while ($sw.Elapsed.TotalSeconds -lt $TimeoutSec) {
    if (Test-Path $Metrics) { return $true }
    Start-Sleep -Milliseconds 300
  }
  return $false
}

function Get-TagRows($tag) {
  if (-not (Test-Path $Metrics)) { return @() }
  try {
    return Import-Csv -Path $Metrics | Where-Object { $_.tag -eq $tag }
  } catch {
    # 헤더가 아직 없거나 파일 잠김 등
    return @()
  }
}

function Assert-TagExists($tag, [int]$minRows = 1) {
  $rows = Get-TagRows $tag
  if ($rows.Count -ge $minRows) {
    Write-Host "[PASS] metrics.csv 기록 확인 ($tag x$($rows.Count))" -ForegroundColor Green
    return $true
  } else {
    Write-Host "[FAIL] metrics.csv 기록 없음/부족 ($tag)" -ForegroundColor Red
    return $false
  }
}

function Run-Step($title, $argsArray, $tag, [int]$waitSec = 45) {
  Write-Title $title
  $cmd = @($Py, "-m", $Cli) + $argsArray
  Write-Host ">> $($cmd -join ' ')" -ForegroundColor DarkGray
  & $Py -m $Cli @argsArray
  if (-not (Wait-ForMetrics -TimeoutSec $waitSec)) {
    Write-Host "[warn] metrics.csv 생성 대기 시간 초과" -ForegroundColor Yellow
  }
  Assert-TagExists -tag $tag | Out-Null
}

Write-Section "환경 체크"
Write-Host "Root      : $Root"
Write-Host "Outputs   : $Outputs"
Write-Host "Metrics   : $Metrics"
Write-Host "MarketCsv : $MarketCsv"
if (Test-Path $MortCsv) { Write-Host "MortCsv   : $MortCsv" }

# (A) HJB 기본 평가 (wealth ES)
$tagA = "A_HJB_SMOKE"
Run-Step "A) HJB 기본 평가 (wealth ES)" @(
  "--asset","KR",
  "--method","hjb",
  "--market_mode","bootstrap",
  "--market_csv",$MarketCsv,
  "--use_real_rf","on",
  "--outputs",".\outputs",
  "--bands","on",
  "--n_paths","200",
  "--seeds","0",
  "--es_mode","wealth",
  "--tag",$tagA,
  "--print_mode","summary",
  "--quiet","on"
) $tagA 60

# (B) HJB 손실기준 ES(=CVaR)
$tagB = "B_LOSS_ES"
Run-Step "B) HJB 손실기준 ES(=CVaR)" @(
  "--asset","KR",
  "--method","hjb",
  "--market_mode","bootstrap",
  "--market_csv",$MarketCsv,
  "--use_real_rf","on",
  "--outputs",".\outputs",
  "--bands","on",
  "--n_paths","200",
  "--seeds","0",
  "--es_mode","loss",
  "--F_target","1.0",
  "--tag",$tagB,
  "--print_mode","summary",
  "--quiet","on"
) $tagB 60

# (C) 효용 레이어(behavioral utility) 체크
$tagC = "C_BEHAVIORAL_UTILITY"
Run-Step "C) 효용 레이어(behavioral utility)" @(
  "--asset","KR",
  "--method","hjb",
  "--market_mode","bootstrap",
  "--market_csv",$MarketCsv,
  "--use_real_rf","on",
  "--outputs",".\outputs",
  "--bands","on",
  "--n_paths","100",
  "--seeds","0",
  "--es_mode","wealth",
  "--report_utility","on",
  "--crra_gamma","3.0",
  "--u_scale","1.0",
  "--bh_on","on",
  "--la_k","1.2",
  "--beta","0.95",
  "--habit_phi","0.1",
  "--tag",$tagC,
  "--print_mode","summary",
  "--quiet","on"
) $tagC 60

# (D) 액션 편향(action-layer) 래퍼 체크
$tagD = "D_ACTION_BIAS"
Run-Step "D) 액션 편향 래퍼" @(
  "--asset","KR",
  "--method","hjb",
  "--market_mode","bootstrap",
  "--market_csv",$MarketCsv,
  "--use_real_rf","on",
  "--outputs",".\outputs",
  "--bands","on",
  "--n_paths","100",
  "--seeds","0",
  "--es_mode","wealth",
  "--bias_on","on",
  "--bias_loss_aversion","0.5",
  "--bias_prob_gamma","0.8",
  "--bias_myopia","0.2",
  "--bias_w_floor","0.1",
  "--bias_w_cap_shock","0.3",
  "--tag",$tagD,
  "--print_mode","summary",
  "--quiet","on"
) $tagD 60

# (E) 규칙(KGR) + 생명표 + 연금 오버레이
$tagE = "E_RULE_KGR_ANNUITY_MORT"
$annArgs = @(
  "--asset","KR",
  "--method","rule","--baseline","kgr",
  "--market_mode","bootstrap",
  "--market_csv",$MarketCsv,
  "--use_real_rf","on",
  "--outputs",".\outputs",
  "--bands","on",
  "--n_paths","120",
  "--seeds","0",
  "--es_mode","wealth",
  "--mortality","on",
  "--mort_table",$MortCsv,
  "--ann_on","on",
  "--ann_alpha","0.3",
  "--ann_index","real",
  "--tag",$tagE,
  "--print_mode","summary",
  "--quiet","on"
)
Run-Step "E) 규칙(KGR) + 생명표 + 연금 오버레이" $annArgs $tagE 60

Write-Host ""
Write-Host "==========================================="
Write-Host "            VERIFY SUMMARY"
Write-Host "==========================================="

foreach ($tag in @($tagA,$tagB,$tagC,$tagD,$tagE)) {
  $rows = Get-TagRows $tag
  if ($rows.Count -gt 0) {
    $last = $rows[-1]
    $ew = $last.EW; $es = $last.ES95; $ruin = $last.Ruin
    Write-Host ("{0,-26} : EW={1}  ES95={2}  Ruin={3}" -f $tag, $ew, $es, $ruin)
  } else {
    Write-Host ("{0,-26} : (no rows)" -f $tag) -ForegroundColor Yellow
  }
}
Write-Host "==========================================="
