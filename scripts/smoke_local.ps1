# scripts\smoke_local.ps1
param()

$ErrorActionPreference = "Stop"

$COMMON = @(
  "--method","rl","--asset","KR",
  "--market_mode","bootstrap","--market_csv",".\project\data\market\kr_us_gold_bootstrap_full_extended.csv",
  "--use_real_rf","on","--outputs",".\outputs",
  "--rl_epochs","0","--rl_n_paths_eval","200","--seed","42",
  "--quiet","on","--print_mode","metrics","--metrics_keys","EW,ES95,Ruin,mean_WT","--no_paths"
)

function RunJson([object]$argv) {
  $out = .\.venv\Scripts\python.exe -m project.runner.cli @argv
  if ($LASTEXITCODE -ne 0) { throw "CLI fail $LASTEXITCODE`n$out" }
  return ($out | ConvertFrom-Json)
}

# fixed: identical
$f1 = RunJson ($COMMON + @("--tag","local_fixed1","--eval_seed_jitter","off"))
$f2 = RunJson ($COMMON + @("--tag","local_fixed2","--eval_seed_jitter","off"))
if ($f1.EW -ne $f2.EW -or $f1.ES95 -ne $f2.ES95 -or $f1.mean_WT -ne $f2.mean_WT) {
  throw "FIXED MISMATCH"
}

# jitter: must differ
$j1 = RunJson ($COMMON + @("--tag","local_j1","--eval_seed_jitter","on"))
$j2 = RunJson ($COMMON + @("--tag","local_j2","--eval_seed_jitter","on"))
if ($j1.EW -eq $j2.EW -and $j1.ES95 -eq $j2.ES95 -and $j1.mean_WT -eq $j2.mean_WT) {
  throw "JITTER NO-DIFF"
}

"SMOKE REGRESSION: PASS"
