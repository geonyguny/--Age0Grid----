# run_overnight.ps1
param(
  [switch]$Parallel = $true,      # 병렬(on)/순차(off)
  [int]$MaxConcurrent = 3         # 병렬 동시 작업 개수
)

$ErrorActionPreference = 'Stop'
$LOG = ".\outputs\_logs\overnight.jsonl"
New-Item -ItemType Directory -Path ".\outputs\_logs" -ErrorAction SilentlyContinue | Out-Null
Remove-Item $LOG -ErrorAction SilentlyContinue

# 공통: JSON만 출력되게 stderr 무시, autosave 켜기
function New-Args {
  param([string[]]$parts)
  return @($parts + @('--quiet','on','--eta_mode','off','--autosave','on'))
}

# === 실행할 작업들 정의(원하면 자유롭게 추가/수정) ===
$tasks = @(
  # 1) HJB full 데이터, 시드 분할 평가 (각 2000 경로)
  @{ Name='hjb_full_s0'; Args= New-Args @('--method','hjb','--market_mode','bootstrap','--data_profile','full','--seeds','0','--n_paths','2000','--tag','hjb_full_s0') },
  @{ Name='hjb_full_s1'; Args= New-Args @('--method','hjb','--market_mode','bootstrap','--data_profile','full','--seeds','1','--n_paths','2000','--tag','hjb_full_s1') },
  @{ Name='hjb_full_s2'; Args= New-Args @('--method','hjb','--market_mode','bootstrap','--data_profile','full','--seeds','2','--n_paths','2000','--tag','hjb_full_s2') },
  @{ Name='hjb_full_s3'; Args= New-Args @('--method','hjb','--market_mode','bootstrap','--data_profile','full','--seeds','3','--n_paths','2000','--tag','hjb_full_s3') },
  @{ Name='hjb_full_s4'; Args= New-Args @('--method','hjb','--market_mode','bootstrap','--data_profile','full','--seeds','4','--n_paths','2000','--tag','hjb_full_s4') },

  # 2) CVaR 목표치로 λ 캘리브레이션
  @{ Name='calib_lambda_full'; Args= New-Args @('--method','hjb','--market_mode','bootstrap','--data_profile','full','--seeds','0','--n_paths','3000','--calib','on','--calib_param','lambda','--cvar_target','0.80','--cvar_tol','0.01','--lambda_min','0.0','--lambda_max','5.0','--calib_max_iter','12','--tag','calib_lambda_full') },

  # 3) 규칙형 베이스라인 스윕 (각 5 seeds × 1000 경로)
  @{ Name='rule_full_kgr';  Args= New-Args @('--method','rule','--baseline','kgr','--market_mode','bootstrap','--data_profile','full','--seeds','0','1','2','3','4','--n_paths','1000','--tag','rule_full_kgr') },
  @{ Name='rule_full_vpw';  Args= New-Args @('--method','rule','--baseline','vpw','--market_mode','bootstrap','--data_profile','full','--seeds','0','1','2','3','4','--n_paths','1000','--tag','rule_full_vpw') },
  @{ Name='rule_full_4pct'; Args= New-Args @('--method','rule','--baseline','4pct','--market_mode','bootstrap','--data_profile','full','--seeds','0','1','2','3','4','--n_paths','1000','--tag','rule_full_4pct') },

  # 4) 헤지 파라미터 그리드 (mode×k 조합)
  @{ Name='hjb_hedge_sigma_k0.0'; Args= New-Args @('--method','hjb','--market_mode','bootstrap','--data_profile','full','--hedge','on','--hedge_mode','sigma','--hedge_sigma_k','0.0','--hedge_cost','0.005','--seeds','0','1','2','--n_paths','1000','--tag','hjb_full_hedge_sigma_k0.0') },
  @{ Name='hjb_hedge_sigma_k0.2'; Args= New-Args @('--method','hjb','--market_mode','bootstrap','--data_profile','full','--hedge','on','--hedge_mode','sigma','--hedge_sigma_k','0.2','--hedge_cost','0.005','--seeds','0','1','2','--n_paths','1000','--tag','hjb_full_hedge_sigma_k0.2') },
  @{ Name='hjb_hedge_sigma_k0.5'; Args= New-Args @('--method','hjb','--market_mode','bootstrap','--data_profile','full','--hedge','on','--hedge_mode','sigma','--hedge_sigma_k','0.5','--hedge_cost','0.005','--seeds','0','1','2','--n_paths','1000','--tag','hjb_full_hedge_sigma_k0.5') },
  @{ Name='hjb_hedge_down_k0.2'; Args= New-Args @('--method','hjb','--market_mode','bootstrap','--data_profile','full','--hedge','on','--hedge_mode','downside','--hedge_sigma_k','0.2','--hedge_cost','0.005','--seeds','0','1','2','--n_paths','1000','--tag','hjb_full_hedge_down_k0.2') }
)

# === 실행 엔진 ===
$PY = 'python'
$ENTRY = @('-m','project.runner.cli')

function Invoke-Task {
  param($t)
  $stamp = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
  "[$stamp] START  $($t.Name)" | Out-File -FilePath $LOG -Append -Encoding utf8
  & $PY @($ENTRY + $t.Args) 2>$null |
    Out-File -FilePath $LOG -Append -Encoding utf8
  $stamp2 = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
  "[$stamp2] FINISH $($t.Name)" | Out-File -FilePath $LOG -Append -Encoding utf8
}

if (-not $Parallel) {
  foreach ($t in $tasks) { Invoke-Task $t }
} else {
  $jobs = @()
  foreach ($t in $tasks) {
    while (($jobs | Where-Object { $_.State -eq 'Running' }).Count -ge $MaxConcurrent) {
      Start-Sleep -Seconds 5
      $jobs | Receive-Job -Keep | Out-Null
      $jobs = $jobs | Where-Object { $_.State -in 'Running','NotStarted' }
    }
    $jobs += Start-Job -Name $t.Name -ArgumentList $t,$PY,$ENTRY,$LOG -ScriptBlock {
      param($task,$py,$entry,$log)
      $stamp = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
      "[$stamp] START  $($task.Name)" | Out-File -FilePath $log -Append -Encoding utf8
      & $py @($entry + $task.Args) 2>$null |
        Out-File -FilePath $log -Append -Encoding utf8
      $stamp2 = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
      "[$stamp2] FINISH $($task.Name)" | Out-File -FilePath $log -Append -Encoding utf8
    }
  }
  # 모든 작업 종료 대기 + 로그 수거
  Wait-Job -Job $jobs | Out-Null
  $jobs | Receive-Job | Out-Null
  $jobs | Remove-Job
}

"Batch done. Log → $LOG" | Write-Host
