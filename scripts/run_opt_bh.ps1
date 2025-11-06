param(
  [ValidateSet("light","medium","heavy")] [string]$Profile = "light",
  [int[]] $OptSeeds = @(7),              # OPT seeds (light은 1개 권장)
  [switch] $DoBH = $true,                # BH 실행 여부
  [string] $ProjectRoot = (Get-Location).Path
)

# --- 전역 설정/안내 ---
$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot
$PY   = ".\.venv\Scripts\python.exe"
$Out  = ".\outputs"
$Log  = Join-Path $Out "_logs"
$Figs = Join-Path $Out "figs"
New-Item -ItemType Directory -Force -Path $Out,$Log,$Figs | Out-Null

Write-Host "[Night v3] profile=$Profile; DoBH=$DoBH; seeds=[$($OptSeeds -join ',')]" -ForegroundColor Cyan
Write-Host "주의: Excel 닫기(파일 잠금 방지), 경로: $ProjectRoot" -ForegroundColor Yellow

# --- 러닝 프로파일(라이트/미디엄/헤비) ---
$COMMON = @(
  "--method","rl","--mode","rl","--data_profile","dev",
  "--market_mode","bootstrap","--bootstrap_block","24",
  "--eval_seed_jitter","off","--autosave","on","--print_mode","summary",
  "--bias_on","off","--cvar_stage","off","--es_mode","wealth"
)
$BH_BASE = @(
  "--method","rl","--mode","rl","--data_profile","dev",
  "--market_mode","bootstrap","--bootstrap_block","24",
  "--eval_seed_jitter","off","--autosave","on","--print_mode","summary",
  "--bias_on","on","--bh_on","on","--cvar_stage","off","--es_mode","wealth"
)

switch ($Profile) {
  "light"  { $OPT_TRAIN = @("--n_paths","1000","--rl_epochs","1","--rl_steps_per_epoch","256");
             $BH_TRAIN  = @("--seed","7","--n_paths","1000","--rl_epochs","1","--rl_steps_per_epoch","512") }
  "medium" { $OPT_TRAIN = @("--n_paths","3000","--rl_epochs","2","--rl_steps_per_epoch","1024");
             $BH_TRAIN  = @("--seed","7","--n_paths","3000","--rl_epochs","2","--rl_steps_per_epoch","1024") }
  "heavy"  { $OPT_TRAIN = @("--n_paths","3000","--rl_epochs","3","--rl_steps_per_epoch","1024");
             $BH_TRAIN  = @("--seed","7","--n_paths","3000","--rl_epochs","3","--rl_steps_per_epoch","1024") }
}

# --- 유틸 함수들 ---
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
  $snap = Join-Path $Out $SnapName
  $summary = Join-Path $Out $SummaryName
  $summaryB = Join-Path $Out $SummaryBenefitName
  Copy-Item $snap $summary -Force

  $tmpB = [System.IO.Path]::GetTempFileName()
  Add-BenefitScore -SnapshotCsv $snap -OutCsv $tmpB
  Copy-Item $tmpB $summaryB -Force

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

  $optTop  = $optRows | Sort-Object -Descending CompositeScore_benefit | Select-Object -First 1
  $optTop3 = $optRows | Sort-Object -Descending CompositeScore_benefit | Select-Object -First 3
  $bhTop   = if ($bhRows) { $bhRows | Sort-Object -Descending CompositeScore_benefit | Select-Object -First 1 } else { $null }
  $bhTop3  = if ($bhRows) { $bhRows | Sort-Object -Descending CompositeScore_benefit | Select-Object -First 3 } else { $null }

  $payload = [ordered]@{
    generated_at = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    criteria     = @{ es_mode="wealth"; score="CompositeScore_benefit"; weights=@(0.6,0.4) }
    OPT = @{ top_tag=$optTop.tag; top=$optTop; top3=$optTop3 }
    BH  = if ($bhRows) { @{ top_tag=$bhTop.tag; top=$bhTop; top3=$bhTop3 } } else { $null }
  }

  $dst = Join-Path $Figs "optimal_points.json"
  ($payload | ConvertTo-Json -Depth 6) | Set-Content $dst -Encoding UTF8
  Write-Host "saved: $dst" -ForegroundColor Cyan
  return $dst
}
function Infer-BaseDesign {
  param([string]$Tag)
  if ($Tag -match "ann")   { return @{ mode="annuity"; cap="0.002" } }
  if ($Tag -match "cap05") { return @{ mode="vpw";     cap="0.005" } }
  if ($Tag -match "cap2")  { return @{ mode="vpw";     cap="0.02"  } }
  return @{ mode="vpw"; cap="0.005" }
}

# --- A) OPT 대표 2종 (seed grid 지원) ---
$optTags = @()
foreach($s in $OptSeeds){
  & $PY -m project.runner.cli --tag ("OPT_vpw_cap05_S"+$s)  --cstar_mode vpw     --rl_q_cap 0.005 @COMMON @("--seed",$s) @OPT_TRAIN
  & $PY -m project.runner.cli --tag ("OPT_ann_tight_S"+$s) --cstar_mode annuity --rl_q_cap 0.002 @COMMON @("--seed",$s) @OPT_TRAIN
  $optTags += @("OPT_vpw_cap05_S$s","OPT_ann_tight_S$s")
}

# --- B) OPT 스냅샷/점수/요약 + optimal_points.json(OPT만) ---
$snap_opt = New-Snapshot -TagPrefix "OPT_" -OutDir $Out -LogDir $Log -SnapName "OPT_metrics_snapshot.csv"
& $PY .\scripts\score_snapshot.py --src $snap_opt --tag_startswith OPT_ --metrics EW,ES95 --weights 0.6,0.4 --es_mode wealth --out inplace
Publish-Room -Prefix "OPT" -SnapName "OPT_metrics_snapshot.csv" -SummaryName "OPT_summary.csv" -SummaryBenefitName "OPT_summary_benefit.csv"
$optJson = Write-OptimalPointsJson -OptSummary (Join-Path $Out "OPT_summary_benefit.csv") -BhSummary $null
$optTop  = (Get-Content $optJson | ConvertFrom-Json).OPT.top_tag
$base    = Infer-BaseDesign -Tag $optTop
$M = $base.mode; $C = $base.cap

# --- C) BH 6축(라이트/미디엄/헤비 동일 틀) ---
if ($DoBH) {
  $BH_COMMON_INH = $BH_BASE + @("--cstar_mode",$M,"--rl_q_cap",$C) + $BH_TRAIN

  # 1) Loss aversion
  & $PY -m project.runner.cli --tag ("BH_la_k2.5_"+$M)        --la_k 2.5               @BH_COMMON_INH
  & $PY -m project.runner.cli --tag ("BH_la_k3.0_"+$M)        --la_k 3.0               @BH_COMMON_INH

  # 2) Probability weighting (Prelec γ)
  & $PY -m project.runner.cli --tag ("BH_pw_g0.70_"+$M)       --bias_prob_gamma 0.70   @BH_COMMON_INH
  & $PY -m project.runner.cli --tag ("BH_pw_g0.60_"+$M)       --bias_prob_gamma 0.60   @BH_COMMON_INH

  # 3) Present bias (myopia 계수)
  & $PY -m project.runner.cli --tag ("BH_myopia_0.90_"+$M)    --bias_myopia 0.90       @BH_COMMON_INH
  & $PY -m project.runner.cli --tag ("BH_myopia_0.85_"+$M)    --bias_myopia 0.85       @BH_COMMON_INH

  # 4) Habit/Smoothing
  & $PY -m project.runner.cli --tag ("BH_habit_phi0.25_"+$M)  --habit_phi 0.25         @BH_COMMON_INH

  # 5) Policy floor & cap-shock
  & $PY -m project.runner.cli --tag ("BH_wfloor_0.02_"+$M)    --bias_w_floor 0.02      @BH_COMMON_INH
  & $PY -m project.runner.cli --tag ("BH_wcap_shock0.10_"+$M) --bias_w_cap_shock 0.10  @BH_COMMON_INH

  # --- D) BH 스냅샷/점수/요약 + optimal_points.json(OPT+BH) ---
  $snap_bh = New-Snapshot -TagPrefix "BH_" -OutDir $Out -LogDir $Log -SnapName "BH_metrics_snapshot.csv"
  & $PY .\scripts\score_snapshot.py --src $snap_bh --tag_startswith BH_ --metrics EW,ES95 --weights 0.6,0.4 --es_mode wealth --out inplace
  Publish-Room -Prefix "BH" -SnapName "BH_metrics_snapshot.csv" -SummaryName "BH_summary.csv" -SummaryBenefitName "BH_summary_benefit.csv"
  Write-OptimalPointsJson -OptSummary (Join-Path $Out "OPT_summary_benefit.csv") -BhSummary (Join-Path $Out "BH_summary_benefit.csv")
}

Write-Host "[Night v3 DONE] 확인: outputs\figs\optimal_points.json, OPT/BH summary(_benefit).csv, *_raw, *_metrics_dedup.csv" -ForegroundColor Cyan
