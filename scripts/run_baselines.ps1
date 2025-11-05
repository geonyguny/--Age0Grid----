param(
  [string]$Seeds = "0",
  [string]$Policies = "4pct,cpb,vpw",
  [string]$Tag = "DEV_BASE"
)

# 1) PYTHONPATH 보강
if (-not $env:PYTHONPATH) { $env:PYTHONPATH = "" }
$env:PYTHONPATH = "$PWD;$PWD\project;$PWD\src;$env:PYTHONPATH"

# 2) runner 모듈 자동 탐색
$RUNNER = python .\scripts\resolve_runner.py
if ($LASTEXITCODE -ne 0 -or $RUNNER -eq "NO_MODULE") { throw "runner.cli 계열 모듈을 찾지 못함" }

# 3) 실행 루프
$seeds = $Seeds -split ','
$pols  = $Policies -split ','
foreach ($s in $seeds) {
  foreach ($p in $pols) {
    $tag = "${Tag}_${p}_seed${s}"
    $cmd = @(
      "python","-m",$RUNNER,
      "--method","rule","--baseline",$p,"--seed",$s,"--tag",$tag,
      "--horizon_years","35","--market_mode","bootstrap","--bootstrap_block","24",
      "--q_floor","0.02","--fee_annual","0.004",
      "--alpha","0.95","--lambda_term","0.8","--beta","0.996",
      "--mortality","on","--sex","M","--age0","65"
    )
    Write-Host ">>" ($cmd -join ' ')
    & $cmd
    if ($LASTEXITCODE -ne 0) { Write-Warning "baseline $p seed $s 실패" }
  }
}
