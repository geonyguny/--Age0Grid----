# UTF-8 + 작업 폴더
chcp 65001 > $null
Set-Location D:\01_simul

# 타임스탬프 & 로그
$TS  = Get-Date -Format "yyyyMMdd_HHmm"
$LOG = ".\outputs\_logs\smoke_$TS.log"
New-Item (Split-Path $LOG) -ItemType Directory -ErrorAction SilentlyContinue | Out-Null
Start-Transcript -Path $LOG -Append

# 공통 옵션(해시테이블 스플래팅)
$OR = ".\outputs\night_$TS"
$COMMON = @{
  OutRoot        = $OR
  Seeds          = 2
  NPaths         = 500
  MarketMode     = 'iid'          # 빠른 스모크: bootstrap 미사용(원하면 변경)
  BootstrapBlock = 12
  Quiet          = 'on'
  XaiOn          = 'off'
  EtaMode        = 'history'
  EtaDb          = '.\outputs\_logs\eta_history.csv'
  SkipExisting   = $false
  DoSummary      = $true
}

# 1) RULE : 극단값만 빠르게 확인 (w=0, 1)
.\night_run.ps1 -Method rule -Baseline 4pct -Tag ("smk_rule_4pct_{0}" -f $TS) -WList 0,1 @COMMON

# 2) HJB : 파라미터 기본으로 1회 (짧은 스펙, i.i.d.)
.\night_run.ps1 -Method hjb -Tag ("smk_hjb_{0}" -f $TS) `
  -Lambda 0.8 -Alpha 0.95 -WMax 0.70 -QFloor 0.02 -Fee 0.004 -HorizonY 15 @COMMON

# 3) RL : 1 epoch 초경량
$RLARGS = @(
  "--rl_epochs","1",
  "--rl_steps_per_epoch","1024",
  "--rl_n_paths_eval","100",
  "--gae_lambda","0.95",
  "--entropy_coef","0.01",
  "--value_coef","0.5",
  "--lr","0.0003",
  "--max_grad_norm","0.5"
)
.\night_run.ps1 -Method rl -Tag ("smk_rl_{0}" -f $TS) @COMMON -ExtraArgs $RLARGS

# 4) 그림/표 생성
python .\scripts\make_paper_figs.py $OR

# 5) 요약 프린트(있을 때만)
Write-Host "====== SMOKE SUMMARY ($OR) ======"
if (Test-Path "$OR\night_summary_report.csv") {
  Import-Csv "$OR\night_summary_report.csv" |
    Sort method,baseline,w_fixed |
    Format-Table -Auto
} else {
  Write-Warning "요약 파일이 아직 없습니다. 로그($LOG)를 확인하세요."
}

Stop-Transcript
