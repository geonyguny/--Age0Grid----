Param(
  [string]$PY = "$PWD\.venv\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
$ProgressPreference    = "SilentlyContinue"

if (!(Test-Path $PY)) { $PY = "python" }
New-Item -ItemType Directory -Force -Path .\outputs\_logs | Out-Null

# RL – 기본, 비용
& $PY -m project.runner.cli --method rl --data_profile dev --rl_epochs 0 --rl_n_paths_eval 5 --outputs .\outputs --tag dev_quick       --print_mode summary --market_mode bootstrap --bootstrap_block 12 --data_window : --h_FX 0.5 --fx_hedge_cost 0       2>&1 | Tee-Object -FilePath .\outputs\_logs\dev_quick.log
& $PY -m project.runner.cli --method rl --data_profile dev --rl_epochs 0 --rl_n_paths_eval 5 --outputs .\outputs --tag dev_quick_cost --print_mode summary --market_mode bootstrap --bootstrap_block 12 --data_window : --h_FX 0.5 --fx_hedge_cost 0.002   2>&1 | Tee-Object -FilePath .\outputs\_logs\dev_quick_cost.log

# Rule – VPW
& $PY -m project.runner.cli --method rule --baseline vpw --data_profile dev --outputs .\outputs --tag chk_vpw --print_mode summary --market_mode bootstrap --bootstrap_block 12 --data_window :  2>&1 | Tee-Object -FilePath .\outputs\_logs\chk_vpw.log

# HJB – 스모크(빠름)
& $PY -m project.runner.cli --method hjb --data_profile dev --outputs .\outputs --tag hjb_smoke --print_mode summary --market_mode bootstrap --bootstrap_block 12 --data_window : --no_paths  2>&1 | Tee-Object -FilePath .\outputs\_logs\hjb_smoke.log

# 요약/검증/스냅샷
& .\scripts\p0_check.ps1
