Param(
  [string]$Csv = ".\project\data\market\kr_us_gold_bootstrap_full_extended.csv",
  [string]$Outputs = ".\outputs"
)

$ErrorActionPreference = "Stop"

function RunJson($argv) {
  $out = .\.venv\Scripts\python.exe -m project.runner.cli @argv
  if ($LASTEXITCODE -ne 0) { throw "CLI failed ($LASTEXITCODE):`n$out" }
  return ($out | ConvertFrom-Json)
}

$COMMON = @(
  "--method","rl","--asset","KR",
  "--market_mode","bootstrap","--market_csv",$Csv,
  "--use_real_rf","on","--outputs",$Outputs,
  "--rl_epochs","0","--rl_n_paths_eval","200","--seed","42",
  "--quiet","on","--print_mode","metrics",
  "--metrics_keys","EW,ES95,Ruin,mean_WT","--no_paths"
)

$f1 = RunJson ($COMMON + @("--tag","reg_fixed1","--eval_seed_jitter","off"))
$f2 = RunJson ($COMMON + @("--tag","reg_fixed2","--eval_seed_jitter","off"))
if ($f1.EW -ne $f2.EW -or $f1.ES95 -ne $f2.ES95 -or $f1.mean_WT -ne $f2.mean_WT) {
  Write-Host "FIXED MISMATCH"
  exit 2
}

$j1 = RunJson ($COMMON + @("--tag","reg_jitter1","--eval_seed_jitter","on"))
$j2 = RunJson ($COMMON + @("--tag","reg_jitter2","--eval_seed_jitter","on"))
if ($j1.EW -eq $j2.EW -and $j1.ES95 -eq $j2.ES95 -and $j1.mean_WT -eq $j2.mean_WT) {
  Write-Host "JITTER NO-DIFF"
  exit 3
}

Write-Host "SMOKE REGRESSION: PASS"
