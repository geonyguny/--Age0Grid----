param(
  [string]$Python = "python",
  [string]$Outputs = "./outputs",
  [string]$DataProfile = "dev",   # dev | full
  [string]$Asset = "KR",
  [switch]$Quiet
)

$ErrorActionPreference = "Stop"
$ts = Get-Date -Format "yyMMdd_HHmm"
$quietFlag = $(if ($Quiet) { "--quiet on" } else { "--quiet off" })

# 공통 베이스 인자
$base = @(
  "--asset", $Asset,
  "--market_mode", "bootstrap",
  "--data_profile", $DataProfile,
  "--outputs", $Outputs,
  $quietFlag
)

# 실행 목록(원하면 추가/삭제)
$jobs = @(
  @{ name="hjb_wealth"; args=@("--method","hjb","--horizon_years","35","--w_max","0.70","--fee_annual","0.004","--alpha","0.95","--lambda_term","0.0","--es_mode","wealth","--n_paths","200","--seeds","0","1","2","3","4","--tag","hjb_${ts}") },
  @{ name="hjb_loss";   args=@("--method","hjb","--horizon_years","35","--w_max","0.70","--fee_annual","0.004","--alpha","0.95","--lambda_term","0.0","--es_mode","loss","--F_target","1.0","--n_paths","300","--seeds","0","1","2","3","4","--tag","hjb_loss_${ts}") },
  @{ name="rule_kgr";   args=@("--method","rule","--baseline","kgr","--w_max","0.60","--q_floor","0.02","--tag","kgr_${ts}") },
  @{ name="rl_dev";     args=@("--method","rl","--rl_epochs","60","--rl_steps_per_epoch","2048","--rl_n_paths_eval","300","--gae_lambda","0.95","--entropy_coef","0.01","--value_coef","0.5","--lr","3e-4","--w_max","0.70","--fee_annual","0.004","--tag","rl_${ts}") }
)

# 로그 디렉토리
$logDir = Join-Path $Outputs "_logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Run-Job([string]$name, [string[]]$args) {
  $cmd = @("-m","project.runner.cli") + $base + $args + @("--print_mode","summary")
  $log = Join-Path $logDir ("night_" + $name + "_" + $ts + ".log")
  Write-Host "▶ $name" -ForegroundColor Cyan
  Write-Host "  $Python $($cmd -join ' ')"
  & $Python $cmd *>&1 | Tee-Object -FilePath $log
  if ($LASTEXITCODE -ne 0) {
    Write-Warning "Job '$name' failed (exit $LASTEXITCODE). See $log"
  } else {
    Write-Host "✔ $name done → $log" -ForegroundColor Green
  }
}

foreach ($j in $jobs) {
  Run-Job -name $j.name -args $j.args
}

Write-Host "All done." -ForegroundColor Green
