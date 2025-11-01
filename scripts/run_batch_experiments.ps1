Param(
  [string]$Py = ".\.venv\Scripts\python.exe",
  [string]$Cli = "project.runner.cli",
  [string]$OutDir = ".\outputs",
  [string]$Tag = "compare",
  [string]$DataProfile = "dev",
  [string]$MarketMode = "bootstrap",
  [int]$PathsEval = 20000
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Invoke-Run($args) {
  Write-Host ">> $args"
  & $Py -m $Cli $args
}

Write-Host "=== (0) 사전 체크: Excel 닫힘/가상환경/경로 ==="
Write-Host "PWD: $(Get-Location)"
Write-Host "Python: $Py  CLI: $Cli"

# ─────────────────────────────────────────────────────────
# (1) OAT(단일변수) 스윕 예시
# ─────────────────────────────────────────────────────────
Write-Host "=== (1) OAT sweeps ==="

# 1-1) 위험자산 믹스 스윕 (KR/US/Gold)
$mixKR = 0.0,0.2,0.4,0.6,0.8,1.0
foreach ($kr in $mixKR) {
  $us = [math]::Max(0.0, 1.0 - $kr); $au = 0.0
  $args = @(
    "--method rule",
    "--baseline vpw",
    "--market_mode $MarketMode",
    "--data_profile $DataProfile",
    "--alpha_mix ""$kr,$us,$au""",
    "--tag ""mix_$kr""",
    "--n_paths_eval $PathsEval",
    "--print_mode summary"
  ) -join " "
  Invoke-Run $args
}

# 1-2) w_max 스윕
$wmaxs = 0.30,0.50,0.70,0.90,1.00
foreach ($w in $wmaxs) {
  $args = @(
    "--method hjb",
    "--market_mode $MarketMode",
    "--data_profile $DataProfile",
    "--w_max $w",
    "--tag ""wmax_$w""",
    "--n_paths_eval $PathsEval",
    "--print_mode summary"
  ) -join " "
  Invoke-Run $args
}

# 1-3) q_floor 스윕
$qfloors = 0.01,0.02,0.03,0.04
foreach ($q in $qfloors) {
  $args = @(
    "--method rl",
    "--market_mode $MarketMode",
    "--data_profile $DataProfile",
    "--q_floor $q",
    "--tag ""qfloor_$q""",
    "--n_paths_eval $PathsEval",
    "--print_mode summary"
  ) -join " "
  Invoke-Run $args
}

# 1-4) Hedge ratio 스윕 (sigma hedge)
$hedges = 0.0,0.25,0.50,0.75,1.0
foreach ($h in $hedges) {
  $args = @(
    "--method rule",
    "--baseline 4pct",
    "--market_mode $MarketMode",
    "--data_profile $DataProfile",
    "--hedge on",
    "--hedge_mode sigma",
    "--hedge_sigma_k $h",
    "--tag ""hedge_$h""",
    "--n_paths_eval $PathsEval",
    "--print_mode summary"
  ) -join " "
  Invoke-Run $args
}

# 1-5) Ambiguity(강건효용 θ) 스윕
$thetas = 0.0,0.5,1.0,2.0
foreach ($t in $thetas) {
  $args = @(
    "--method hjb",
    "--market_mode $MarketMode",
    "--data_profile $DataProfile",
    "--theta_ambiguity $t",
    "--tag ""amb_$t""",
    "--n_paths_eval $PathsEval",
    "--print_mode summary"
  ) -join " "
  Invoke-Run $args
}

# ─────────────────────────────────────────────────────────
# (2) 2D Heatmap 예시 (w_max × q_floor)
# ─────────────────────────────────────────────────────────
Write-Host "=== (2) 2D heatmap (w_max × q_floor) ==="
$wgrid = 0.30,0.50,0.70,0.90
$qgrid = 0.01,0.02,0.03,0.04
foreach ($w in $wgrid) {
  foreach ($q in $qgrid) {
    $args = @(
      "--method rl",
      "--market_mode $MarketMode",
      "--data_profile $DataProfile",
      "--w_max $w",
      "--q_floor $q",
      "--tag ""w$q""",
      "--n_paths_eval $PathsEval",
      "--print_mode summary"
    ) -join " "
    Invoke-Run $args
  }
}

# ─────────────────────────────────────────────────────────
# (3) Dynamic CVaR λ Calibration / Sweep
# ─────────────────────────────────────────────────────────
Write-Host "=== (3) CVaR λ sweep ==="
$lambdas = 0.25,0.5,0.8,1.2,1.6,2.0
foreach ($lam in $lambdas) {
  $args = @(
    "--method rl",
    "--market_mode $MarketMode",
    "--data_profile $DataProfile",
    "--alpha 0.95",
    "--lambda_term $lam",
    "--calib_fast on",
    "--calib_max_iter 8",
    "--tag ""lambda_$lam""",
    "--n_paths_eval $PathsEval",
    "--print_mode summary"
  ) -join " "
  Invoke-Run $args
}

# ─────────────────────────────────────────────────────────
# (4) 결과 요약/스코어링 및 우승 세트 추출
# ─────────────────────────────────────────────────────────
Write-Host "=== (4) Summary & Scoring ==="
# _summary_scored.csv는 내부 루틴으로 생성된다고 가정
$S = Import-Csv "$OutDir\_summary_scored.csv"
$grpNoSeed = { '{0}|{1}|{2}|{3}|{4}|{5}|{6}' -f $_.es_mode,$_.window,$_.hedge_ratio,$_.mix_kr,$_.mix_us,$_.mix_gold,$_.es_metric }

$winners = $S | Where-Object { $_.CompositeScore -ne '' } |
  Group-Object $grpNoSeed | ForEach-Object {
    $_.Group | Sort-Object {[double]$_.CompositeScore} | Select-Object -First 1
  }

$winners | Export-Csv "$OutDir\_winners.csv" -NoTypeInformation
Write-Host "Winners saved: $OutDir\_winners.csv"

# ─────────────────────────────────────────────────────────
# (5) 보고서/그림 생성
# ─────────────────────────────────────────────────────────
Write-Host "=== (5) Report & Figures ==="
& $Py "scripts\make_paper_figs.py" --outdir $OutDir --tag $Tag
Write-Host "DONE. 보고서: $OutDir\ALM_Executive_Report_$Tag.xlsx"
