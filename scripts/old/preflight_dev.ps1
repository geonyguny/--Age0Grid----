# scripts/preflight_dev.ps1
# 목적:
# - DEV 프로필로 빠른 사전 점검(Preflight)
#   1) fixed 두 번 → 동일성
#   2) jitter 두 번 → 상이성
#   3) 골든 벨류(EW/ES95/mean_WT) 대조(없으면 생성, -UpdateGolden로 갱신)
#   4) wealth/loss 요약 출력(참고용, 실패조건 아님)

param(
  [switch]$UpdateGolden = $false
)

$ErrorActionPreference = "Stop"

# 0) 경로 준비
$projectRoot = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
if (-not $projectRoot) { $projectRoot = "." }
Set-Location $projectRoot

# 1) DEV CSV 확보(없으면 생성)
$devCsv = "project/data/market/kr_us_gold_bootstrap_mini.csv"
if (-not (Test-Path $devCsv)) {
  Write-Host "[preflight] making DEV csv: $devCsv"
  .\.venv\Scripts\python.exe scripts/make_dev_csv.py | Out-Null
}

# 2) 헬퍼: CLI 호출 → JSON
function RunJson([object]$argv) {
  $out = .\.venv\Scripts\python.exe -m project.runner.cli @argv
  if ($LASTEXITCODE -ne 0) { throw "CLI failed ($LASTEXITCODE):`n$out" }
  return ($out | ConvertFrom-Json)
}

# 3) 공통 인자(DEV 프로필, 빠른 실행)
$COMMON = @(
  "--method","rl","--asset","KR",
  "--market_mode","bootstrap","--data_profile","dev",
  "--use_real_rf","on","--outputs",".\outputs",
  "--rl_epochs","0","--rl_n_paths_eval","80","--seed","42",
  "--quiet","on","--print_mode","metrics","--metrics_keys","EW,ES95,Ruin,mean_WT","--no_paths"
)

# 4) fixed 동일성
$f1 = RunJson ($COMMON + @("--tag","dev_fixed1","--eval_seed_jitter","off"))
$f2 = RunJson ($COMMON + @("--tag","dev_fixed2","--eval_seed_jitter","off"))
if ($f1.EW -ne $f2.EW -or $f1.ES95 -ne $f2.ES95 -or $f1.mean_WT -ne $f2.mean_WT) {
  throw "FIXED MISMATCH: ($($f1.EW),$($f1.ES95),$($f1.mean_WT)) vs ($($f2.EW),$($f2.ES95),$($f2.mean_WT))"
}

# 5) jitter 상이성
$j1 = RunJson ($COMMON + @("--tag","dev_j1","--eval_seed_jitter","on"))
$j2 = RunJson ($COMMON + @("--tag","dev_j2","--eval_seed_jitter","on"))
if ($j1.EW -eq $j2.EW -and $j1.ES95 -eq $j2.ES95 -and $j1.mean_WT -eq $j2.mean_WT) {
  throw "JITTER NO-DIFF"
}

# 6) 골든 벨류 관리(파일 없으면 생성, 있으면 대조; -UpdateGolden로 갱신)
$goldenPath = "scripts/_golden_dev.json"
$tol = 1e-9
function Close([double]$a, [double]$b) { return [math]::Abs($a - $b) -le $tol }

$golden = $null
if (Test-Path $goldenPath) {
  try { $golden = Get-Content $goldenPath -Raw | ConvertFrom-Json } catch {}
}

if ($UpdateGolden -or -not $golden) {
  $payload = [ordered]@{
    EW        = [double]$f1.EW
    ES95      = [double]$f1.ES95
    mean_WT   = [double]$f1.mean_WT
    seed      = 42
    n_paths   = 80
    jitter    = "off"
    updatedAt = (Get-Date).ToString("s")
  }
  ($payload | ConvertTo-Json -Depth 4) | Set-Content -Encoding UTF8 $goldenPath
  if ($UpdateGolden) {
    Write-Host "[preflight] golden updated → $goldenPath"
  } else {
    Write-Host "[preflight] golden created → $goldenPath"
  }
} else {
  if (-not (Close ([double]$f1.EW) ([double]$golden.EW)) -or
      -not (Close ([double]$f1.ES95) ([double]$golden.ES95)) -or
      -not (Close ([double]$f1.mean_WT) ([double]$golden.mean_WT))) {
    throw ("DEV GOLDEN MISMATCH:`n" +
           " got=({0},{1},{2})`n exp=({3},{4},{5})" -f
           $f1.EW,$f1.ES95,$f1.mean_WT,$golden.EW,$golden.ES95,$golden.mean_WT)
  }
}

# 7) 참고용 요약 출력(wealth/loss) — 실패조건 아님
.\.venv\Scripts\python.exe -m project.runner.cli `
  --method rl --asset KR `
  --market_mode bootstrap --data_profile dev `
  --use_real_rf on --outputs .\outputs --seed 7 `
  --rl_epochs 0 --rl_n_paths_eval 80 `
  --print_mode summary --metrics_keys "EW,ES95,Ruin,mean_WT" --no_paths

.\.venv\Scripts\python.exe -m project.runner.cli `
  --method rl --asset KR `
  --market_mode bootstrap --data_profile dev `
  --use_real_rf on --outputs .\outputs --seed 7 `
  --rl_epochs 0 --rl_n_paths_eval 80 `
  --es_mode loss --F_target 1.0 `
  --print_mode summary --metrics_keys "ES95,Ruin,mean_WT" --no_paths

Write-Host "PRE-FLIGHT DEV: PASS"
