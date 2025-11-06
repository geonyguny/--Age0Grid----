param(
  [string]$Seeds    = "0",
  [string]$Policies = "4pct,cpb,vpw",
  [string]$Tag      = "DEV_BASE"
)

# -----------------------------
# 실행 환경 준비
# -----------------------------
# 리포지토리 루트(../)
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

# Python 경로(.venv 우선)
$py = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

# PYTHONPATH 설정(중복 허용)
if (-not $env:PYTHONPATH) { $env:PYTHONPATH = "" }
$env:PYTHONPATH = "$RepoRoot;$RepoRoot\project;$RepoRoot\src;$env:PYTHONPATH"

# runner 모듈 확인
$RUNNER = & $py (Join-Path $RepoRoot "scripts\resolve_runner.py")
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($RUNNER) -or $RUNNER -eq "NO_MODULE") {
  throw "runner.cli 계열 모듈을 찾지 못함"
}

# -----------------------------
# 파라미터 파싱
# -----------------------------
$seedList   = ($Seeds -split "[, ]+" | Where-Object { $_ -ne "" })
$policyList = ($Policies -split "[, ]+" | Where-Object { $_ -ne "" })

# 공통 인자(배열)
$common = @(
  "-m", $RUNNER,
  "--method","rule",
  "--horizon_years","35",
  "--market_mode","bootstrap","--bootstrap_block","24",
  "--q_floor","0.02",
  "--fee_annual","0.004",
  "--alpha","0.95",
  "--lambda_term","0.8",
  "--beta","0.996",
  "--mortality","on","--sex","M","--age0","65",
  "--autosave","on",
  "--print_mode","summary"
)

# -----------------------------
# 실행 루프
# -----------------------------
foreach ($s in $seedList) {
  foreach ($p in $policyList) {
    $tagThis = "${Tag}_${p}_seed${s}"
    $args = $common + @("--baseline",$p,"--seed",$s,"--tag",$tagThis)

    Write-Host ">> RULE baseline: $p  seed=$s  tag=$tagThis"
    & $py @args
    if ($LASTEXITCODE -ne 0) {
      Write-Warning "baseline 실패: policy=$p, seed=$s"
    }
  }
}
