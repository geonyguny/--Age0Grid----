# scripts\run_behavior_pair.ps1
param(
  [double]$Kappa     = 1.5,                  # 손실회피 강도(la_k)
  [int]   $Seed      = 0,
  [string]$TagPrefix = "DEV_OAT_la",
  [switch]$SkipAnchor,                       # 붙이면 Bias-OFF 앵커 생략
  [switch]$SkipReopt                         # 붙이면 Bias-ON 재최적화 생략
)

# -----------------------------
# 실행 환경 준비
# -----------------------------
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

# Python 경로(.venv 우선)
$py = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

# PYTHONPATH 설정
if (-not $env:PYTHONPATH) { $env:PYTHONPATH = "" }
$env:PYTHONPATH = "$RepoRoot;$RepoRoot\project;$RepoRoot\src;$env:PYTHONPATH"

# runner 모듈 확인
$RUNNER = & $py (Join-Path $RepoRoot "scripts\resolve_runner.py")
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($RUNNER) -or $RUNNER -eq "NO_MODULE") {
  throw "runner.cli 계열 모듈을 찾지 못함"
}

# -----------------------------
# 공통 인자(초미니 학습: 빠른 확인 위주)
# -----------------------------
$common = @(
  "-m", $RUNNER,
  "--method","rl",
  "--data_profile","dev",
  "--market_mode","bootstrap","--bootstrap_block","24",
  "--horizon_years","35",
  "--fee_annual","0.004",
  "--alpha","0.95",
  "--lambda_term","0.8",
  "--beta","0.996",
  "--mortality","on","--sex","M","--age0","65",
  "--rl_epochs","3",                 # 초미니 에폭(빠름)
  "--rl_steps_per_epoch","1024",     # 짧은 스텝(빠름)
  "--rl_n_paths_eval","200",         # 평가 경로 축소(빠름)
  "--bh_on","on",
  "--print_mode","summary",
  "--report_utility","on",
  "--autosave","on"
)

function Invoke-Run([string]$title, [string[]]$args) {
  Write-Host ">> $title"
  Write-Host ">> $py $($args -join ' ')"
  & $py @args
  if ($LASTEXITCODE -ne 0) { throw "$title failed." }
}

# -----------------------------
# (A) Bias-OFF 앵커 (Policy-Locked 비교용)
# -----------------------------
if (-not $SkipAnchor) {
  $tagA = "${TagPrefix}_k${Kappa}_seed${Seed}_biasOFF"
  $argsA = $common + @("--bias_on","off","--seed",$Seed,"--tag",$tagA)
  Invoke-Run "RL Bias-OFF  k=$Kappa  seed=$Seed  tag=$tagA" $argsA
}

# -----------------------------
# (B) Bias-ON 재최적화 (Re-Opt, la_k 사용)
# -----------------------------
if (-not $SkipReopt) {
  $tagB = "${TagPrefix}_k${Kappa}_seed${Seed}_biasON"
  $argsB = $common + @(
    "--bias_on","on",
    "--la_k",("$Kappa"),            # ★ 핵심: la_k로 적용
    "--seed",$Seed,
    "--tag",$tagB
  )
  Invoke-Run "RL Bias-ON   k=$Kappa  seed=$Seed  tag=$tagB" $argsB
}
