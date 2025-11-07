<# ========================================================================
  run_opt_bh.ps1
  목적: 최적설계(OPT) → optimal_points.json → 행동편향(BH) → 스냅샷/점수/취합까지 일괄 수행
  리팩터링 요점
    - -DataProfile(dev|full) 추가하여 외부에서 데이터 규모 전환
    - -PWGammas "0.70,0.60,0.80,0.50" 등 가변 감마 그리드 지원(기본 "0.70,0.60")
    - benefit 점수(0.6*EW+0.4*ES95) 정렬 안정화([double] 캐스팅) 유지
    - optimal_points.json(OPT→BH 연계) 및 RAW/로그 dedup 고정 유지
    - 콘솔 UTF-8(한글) 및 Excel 권한 경고 유지
========================================================================= #>

[CmdletBinding()]
param(
  [ValidateSet("light","medium","heavy")] [string]$Profile = "light",
  [int[]]$OptSeeds = @(7),
  [bool]$DoBH = $true,
  [ValidateSet("dev","full")] [string]$DataProfile = "dev",
  [string]$PWGammas = "0.70,0.60",
  [string]$ProjectRoot = "G:\01_simul"
)

# -------------------- 환경/경로 & 콘솔 인코딩 --------------------
try { Set-Location -Path $ProjectRoot -ErrorAction Stop }
catch {
  Write-Host "프로젝트 루트가 존재하지 않습니다: $ProjectRoot" -ForegroundColor Red
  exit 1
}

try { chcp 65001 > $null } catch {}
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$PY   = ".\.venv\Scripts\python.exe"
$Out  = ".\outputs"
$Log  = Join-Path $Out "_logs"
$Figs = Join-Path $Out "figs"
New-Item -ItemType Directory -Force -Path $Out,$Log,$Figs | Out-Null

Write-Host "[Night v3] profile=$Profile; DoBH=$DoBH; seeds=[$($OptSeeds -join ',')]; data=$DataProfile; PWGammas=$PWGammas" -ForegroundColor Cyan
Write-Host "주의: Excel이 열려 있으면 PermissionError가 발생합니다. 경로: $ProjectRoot" -ForegroundColor Yellow

# -------------------- 러닝 프로파일 --------------------
# 공통(OPT 기본)
$COMMON = @(
  "--method","rl","--mode","rl","--data_profile",$DataProfile,
  "--market_mode","bootstrap","--bootstrap_block","24",
  "--eval_seed_jitter","off","--autosave","on","--print_mode","summary",
  "--bias_on","off","--cvar_stage","off","--es_mode","wealth"
)

# 프로필별 STEP(OPT)
$RL_LIGHT = @("--n_paths","1000","--rl_epochs","1","--rl_steps_per_epoch","256")
$RL_MED   = @("--n_paths","2000","--rl_epochs","2","--rl_steps_per_epoch","1024")
$RL_HEAVY = @("--n_paths","3000","--rl_epochs","3","--rl_steps_per_epoch","1024")

switch ($Profile) {
  "light"  { $RL = $RL_LIGHT  }
  "medium" { $RL = $RL_MED    }
  "heavy"  { $RL = $RL_HEAVY  }
}

# 프로필별 STEP(BH)
$BH_LIGHT = @("--n_paths","1000","--rl_epochs","1","--rl_steps_per_epoch","512")
$BH_MED   = @("--n_paths","2000","--rl_epochs","2","--rl_steps_per_epoch","1024")
$BH_HEAVY = @("--n_paths","3000","--rl_epochs","3","--rl_steps_per_epoch","1024")

switch ($Profile) {
  "light"  { $BHSTEP = $BH_LIGHT  }
  "medium" { $BHSTEP = $BH_MED    }
  "heavy"  { $BHSTEP = $BH_HEAVY  }
}

# BH 공통
$BHCOMMON_BASE = @(
  "--method","rl","--mode","rl","--data_profile",$DataProfile,
  "--market_mode","bootstrap","--bootstrap_block","24",
  "--eval_seed_jitter","off","--autosave","on","--print_mode","summary",
  "--bh_on","on","--bias_on","on","--cvar_stage","off","--es_mode","wealth",
  "--metrics_keys","EW,ES95,Ruin,la_k,habit_phi,bias_myopia,bias_prob_gamma,bias_loss_aversion,bias_w_floor,bias_w_cap_shock"
)

# -------------------- 유틸 함수 --------------------
function New-MetricsDedup {
  param([string]$LogDir)
  $src = Join-Path $LogDir "metrics.csv"
  $dst = Join-Path $LogDir "metrics_dedup.csv"
  if (!(Test-Path $src)) { throw "로그 없음: $src" }
  (Import-Csv $src | Group-Object tag,method,seed | % { $_.Group | Select-Object -Last 1 }) |
    Export-Csv $dst -NoTypeInformation
  return $dst
}

function New-Snapshot {
  param([string]$TagPrefix,[string]$OutDir,[string]$LogDir,[string]$SnapName)
  $snap = Join-Path $OutDir $SnapName
  (Import-Csv (Join-Path $LogDir 'metrics.csv') |
    ?{ $_.tag -like "$TagPrefix*" -and $_.method -eq 'rl' } |
    Group-Object tag,seed | % { $_.Group | Select-Object -Last 1 }) |
    Export-Csv $snap -NoTypeInformation
  return $snap
}

function Add-BenefitScore {
  param([string]$SnapshotCsv,[string]$OutCsv)
  $rows = Import-Csv $SnapshotCsv
  if ($rows.Count -eq 0) { Copy-Item $SnapshotCsv $OutCsv -Force; return }

  $EW_min = ($rows | Measure-Object -Property EW   -Minimum).Minimum
  $EW_max = ($rows | Measure-Object -Property EW   -Maximum).Maximum
  $ES_min = ($rows | Measure-Object -Property ES95 -Minimum).Minimum
  $ES_max = ($rows | Measure-Object -Property ES95 -Maximum).Maximum

  $normed = $rows | ForEach-Object {
    $ew=[double]$_.EW; $es=[double]$_.ES95
    $nEW = if ($EW_max - $EW_min -ne 0) { ($ew - $EW_min)/($EW_max - $EW_min) } else { 0.0 }
    $nES = if ($ES_max - $ES_min -ne 0) { ($es - $ES_min)/($ES_max - $ES_min) } else { 0.0 }
    $score = 0.6*$nEW + 0.4*$nES
    $_ | Add-Member CompositeScore_benefit $score -Force
    $_ | Add-Member norm_EW_b $nEW -Force
    $_ | Add-Member norm_ES95_b $nES -Force
    $_
  }
  $normed | Export-Csv $OutCsv -NoTypeInformation
}

function Publish-Room {
  param([ValidateSet("OPT","BH")]$Prefix,[string]$SnapName,[string]$SummaryName,[string]$SummaryBenefitName)

  $snap     = Join-Path $Out $SnapName
  $summary  = Join-Path $Out $SummaryName
  $summaryB = Join-Path $Out $SummaryBenefitName

  Copy-Item $snap $summary -Force

  $tmpB = [System.IO.Path]::GetTempFileName()
  Add-BenefitScore -SnapshotCsv $snap -OutCsv $tmpB
  Copy-Item $tmpB $summaryB -Force

  # RAW 고정(중첩 방지)
  $rawRoot = Join-Path $Out "${Prefix}_raw"
  if (Test-Path $rawRoot) { Remove-Item $rawRoot -Recurse -Force }
  New-Item $rawRoot -ItemType Directory -Force | Out-Null
  Get-ChildItem $Out -Directory | Where-Object {
    $_.Name -like "${Prefix}_*" -and $_.Name -notlike "${Prefix}_raw*"
  } | ForEach-Object {
    Copy-Item $_.FullName (Join-Path $rawRoot $_.Name) -Recurse -Force
  }

  $dedup = New-MetricsDedup -LogDir $Log
  Copy-Item $dedup (Join-Path $Log "${Prefix}_metrics_dedup.csv") -Force

  Write-Host "[$Prefix] publish → $summary | $summaryB | $rawRoot" -ForegroundColor Green
}

function Write-OptimalPointsJson {
  param([string]$OptSummary,[string]$BhSummary)

  $optRows = if (Test-Path $OptSummary) { Import-Csv $OptSummary } else { @() }
  $bhRows  = if ($BhSummary -and (Test-Path $BhSummary)) { Import-Csv $BhSummary } else { @() }

  $ensureBenefit = {
    param($rows)
    if (-not $rows) { return @() }
    if ($rows | Get-Member -Name CompositeScore_benefit -MemberType NoteProperty) { return $rows }
    $EW_min = ($rows | Measure-Object -Property EW   -Minimum).Minimum
    $EW_max = ($rows | Measure-Object -Property EW   -Maximum).Maximum
    $ES_min = ($rows | Measure-Object -Property ES95 -Minimum).Minimum
    $ES_max = ($rows | Measure-Object -Property ES95 -Maximum).Maximum
    return $rows | ForEach-Object {
      $ew=[double]$_.EW; $es=[double]$_.ES95
      $nEW = if ($EW_max - $EW_min -ne 0) { ($ew - $EW_min)/($EW_max - $EW_min) } else { 0.0 }
      $nES = if ($ES_max - $ES_min -ne 0) { ($es - $ES_min)/($ES_max - $ES_min) } else { 0.0 }
      $score = 0.6*$nEW + 0.4*$nES
      $_ | Add-Member CompositeScore_benefit $score -Force
      $_
    }
  }

  $optRows = & $ensureBenefit $optRows
  $bhRows  = & $ensureBenefit $bhRows

  $optRows = $optRows | ForEach-Object { $_.CompositeScore_benefit = [double]$_.CompositeScore_benefit; $_ }
  if ($bhRows) { $bhRows = $bhRows | ForEach-Object { $_.CompositeScore_benefit = [double]$_.CompositeScore_benefit; $_ } }

  $optTop   = $optRows | Sort-Object CompositeScore_benefit -Descending | Select-Object -First 1
  $optTop3  = $optRows | Sort-Object CompositeScore_benefit -Descending | Select-Object -First 3
  $bhTop    = if ($bhRows) { $bhRows | Sort-Object CompositeScore_benefit -Descending | Select-Object -First 1 } else { $null }
  $bhTop3   = if ($bhRows) { $bhRows | Sort-Object CompositeScore_benefit -Descending | Select-Object -First 3 } else { $null }

  $payload = [ordered]@{
    generated_at = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    criteria     = @{ es_mode="wealth"; score="CompositeScore_benefit"; weights=@(0.6,0.4) }
    OPT = if ($optTop) { @{ top_tag=$optTop.tag; top=$optTop; top3=$optTop3 } } else { $null }
    BH  = if ($bhTop)  { @{ top_tag=$bhTop.tag;  top=$bhTop;  top3=$bhTop3  } } else { $null }
  }

  $dst = Join-Path $Figs "optimal_points.json"
  ($payload | ConvertTo-Json -Depth 6) | Set-Content $dst -Encoding UTF8
  Write-Host "saved: $dst" -ForegroundColor Cyan
  return $dst
}

function Infer-BaseDesign {
  param([string]$Tag)
  if ($Tag -match "ann")   { return @{ mode="annuity"; cap="0.002" } }
  if ($Tag -match "cap05") { return @{ mode="vpw";    cap="0.005" } }
  if ($Tag -match "cap2")  { return @{ mode="vpw";    cap="0.02"  } }
  return @{ mode="vpw"; cap="0.005" }
}

# -------------------- A) OPT 러닝 --------------------
$profileLabel = $Profile.ToUpper()
$optTags = @()

if ($Profile -eq "light" -and $OptSeeds.Count -eq 1 -and $OptSeeds[0] -eq 7) {
  & $PY -m project.runner.cli --tag OPT_vpw_cap05  --cstar_mode vpw     --rl_q_cap 0.005 @COMMON --seed 7 @RL
  & $PY -m project.runner.cli --tag OPT_ann_tight  --cstar_mode annuity --rl_q_cap 0.002 @COMMON --seed 7 @RL
  $optTags += "OPT_vpw_cap05","OPT_ann_tight"
} else {
  Start-Transcript -Path (Join-Path $Log ("opt_"+(Get-Date -UFormat "%Y%m%d_%H%M")+".log")) | Out-Null
  foreach($s in $OptSeeds){
    $t1 = "OPT_vpw_cap05_S${s}_${profileLabel}"
    $t2 = "OPT_ann_tight_S${s}_${profileLabel}"
    & $PY -m project.runner.cli --tag $t1 --cstar_mode vpw     --rl_q_cap 0.005 @COMMON --seed $s @RL
    & $PY -m project.runner.cli --tag $t2 --cstar_mode annuity --rl_q_cap 0.002 @COMMON --seed $s @RL
    $optTags += $t1,$t2
  }
  Stop-Transcript | Out-Null
}

# 스냅샷/점수/요약 고정 + OPT만으로 optimal_points.json 1차 생성
$snap_opt = New-Snapshot -TagPrefix "OPT_" -OutDir $Out -LogDir $Log -SnapName "OPT_metrics_snapshot.csv"
& $PY .\scripts\score_snapshot.py --src $snap_opt --tag_startswith OPT_ --metrics EW,ES95 --weights 0.6,0.4 --es_mode wealth --out inplace
Publish-Room -Prefix "OPT" -SnapName "OPT_metrics_snapshot.csv" -SummaryName "OPT_summary.csv" -SummaryBenefitName "OPT_summary_benefit.csv"
$optJson = Write-OptimalPointsJson -OptSummary (Join-Path $Out "OPT_summary_benefit.csv") -BhSummary $null

# -------------------- B) BH 러닝 --------------------
if ($DoBH) {
  $optTopTag = (Get-Content $optJson | ConvertFrom-Json).OPT.top_tag
  $base = if ($optTopTag) { Infer-BaseDesign -Tag $optTopTag } else { @{ mode="vpw"; cap="0.005" } }

  $BHCOMMON = @($BHCOMMON_BASE + $BHSTEP)

  Start-Transcript -Path (Join-Path $Log ("bh_"+(Get-Date -UFormat "%Y%m%d_%H%M")+".log")) | Out-Null

  # 1) 손실회피 (la_k)
  & $PY -m project.runner.cli --tag ("BH_la_k2.5_"+$base.mode) --la_k 2.5 --cstar_mode $($base.mode) --rl_q_cap $($base.cap) @BHCOMMON
  & $PY -m project.runner.cli --tag ("BH_la_k3.0_"+$base.mode) --la_k 3.0 --cstar_mode $($base.mode) --rl_q_cap $($base.cap) @BHCOMMON

  # 2) 현재편향 (bias_myopia)
  & $PY -m project.runner.cli --tag ("BH_myopia_0.92_"+$base.mode) --bias_myopia 0.92 --cstar_mode $($base.mode) --rl_q_cap $($base.cap) @BHCOMMON
  & $PY -m project.runner.cli --tag ("BH_myopia_0.90_"+$base.mode) --bias_myopia 0.90 --cstar_mode $($base.mode) --rl_q_cap $($base.cap) @BHCOMMON
  & $PY -m project.runner.cli --tag ("BH_myopia_0.85_"+$base.mode) --bias_myopia 0.85 --cstar_mode $($base.mode) --rl_q_cap $($base.cap) @BHCOMMON

  # 3) 확률가중 (bias_prob_gamma) — 외부 스위치로 가변
  $gammas = $PWGammas -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" }
  foreach($g in $gammas){
    & $PY -m project.runner.cli --tag ("BH_pw_g"+$g+"_"+$base.mode) --bias_prob_gamma $g --cstar_mode $($base.mode) --rl_q_cap $($base.cap) @BHCOMMON
  }

  # 4) 습관 (habit_phi)
  & $PY -m project.runner.cli --tag ("BH_habit_phi0.25_"+$base.mode) --habit_phi 0.25 --cstar_mode $($base.mode) --rl_q_cap $($base.cap) @BHCOMMON

  # 5) 소비 바닥/캡 쇼크
  & $PY -m project.runner.cli --tag ("BH_wfloor_0.02_"+$base.mode)    --bias_w_floor 0.02     --cstar_mode $($base.mode) --rl_q_cap $($base.cap) @BHCOMMON
  & $PY -m project.runner.cli --tag ("BH_wcap_shock0.10_"+$base.mode) --bias_w_cap_shock 0.10 --cstar_mode $($base.mode) --rl_q_cap $($base.cap) @BHCOMMON

  Stop-Transcript | Out-Null

  # 스냅샷/점수/요약 고정 + optimal_points.json 최종 갱신(OPT+BH)
  $snap_bh = New-Snapshot -TagPrefix "BH_" -OutDir $Out -LogDir $Log -SnapName "BH_metrics_snapshot.csv"
  & $PY .\scripts\score_snapshot.py --src $snap_bh --tag_startswith BH_ --metrics EW,ES95 --weights 0.6,0.4 --es_mode wealth --out inplace
  Publish-Room -Prefix "BH" -SnapName "BH_metrics_snapshot.csv" -SummaryName "BH_summary.csv" -SummaryBenefitName "BH_summary_benefit.csv"
  Write-OptimalPointsJson -OptSummary (Join-Path $Out "OPT_summary_benefit.csv") -BhSummary (Join-Path $Out "BH_summary_benefit.csv")
}

Write-Host "[Night v3 DONE] 확인: outputs\figs\optimal_points.json, OPT/BH summary(_benefit).csv, *_raw, *_metrics_dedup.csv" -ForegroundColor Cyan
