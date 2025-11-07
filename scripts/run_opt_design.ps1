<# =======================================================================
 run_opt_design.ps1  (PS5-safe, 2025-11-07, refactored)
 - 1D/2D sweeps -> snapshot -> scoring -> OPT/BH tables
 - Use ONLY flags that exist in cli.py --help
 - ASCII-only console messages, no fancy unicode
 ======================================================================= #>

[CmdletBinding()]
param(
  [ValidateSet('light','heavy')] [string]$Profile = 'light',
  [ValidateSet('dev','full')]    [string]$DataProfile = 'dev',

  [string]$Seeds    = '7',                # e.g. "7,9,11,13,17"
  [string]$PWGammas = '0.85,0.80,0.70,0.60',

  [double]$KR_MinLiving_Monthly = 1500000,
  [double]$InitWealth_Real      = 5e8,

  # ---- Grids (PS5: wrap with @()) ----
  [double[]]$AnnAlphaGrid  = @(0..10 | ForEach-Object { $_/10.0 }),
  [double[]]$RiskShareGrid = @(0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0),
  [double[]]$FeeGrid       = @(0.0000, 0.0040, 0.0080),
  [int[]]   $Age0Grid      = @(55..65),
  [double[]]$HedgeGrid     = @(0.00, 0.25, 0.50, 0.75, 1.00),
  [double[]]$VPWGrid       = @(0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06),

  [ValidateSet('KR','US','GLD','EQUAL3')] [string[]]$AssetMixes = @('EQUAL3','KR','US','GLD'),

  [ValidateSet('M','F')] [string[]]$Sexes = @('M','F'),
  [ValidateSet('BASE','COHORT')] [string[]]$MortCfg = @('BASE','COHORT')
)

# ---------- Paths ----------
$Root = (Get-Location).Path
$Out  = Join-Path $Root 'outputs'
$Logs = Join-Path $Out  '_logs'
$Figs = Join-Path $Out  'figs'
$null = New-Item -ItemType Directory -Force -Path $Out,$Logs,$Figs | Out-Null

Write-Host "[Guard] Close Excel to avoid PermissionError on CSV/XLSX." -ForegroundColor Yellow

# ---------- Utils ----------
function As-Double($x) {
  if ($null -eq $x -or "$x" -eq '') { return $null }
  try { return [double]$x } catch { return $null }
}

function Sort-ByScoreDesc($rows) {
  $rows | Sort-Object @{ Expression = {
      $v = $_.CompositeScore_benefit
      if ($null -eq $v -or "$v" -eq '') { As-Double $_.CompositeScore } else { As-Double $v }
    }; Descending = $true }
}

function Make-Tag([string]$MortId,[string]$BaseTag,[string]$Sex) {
  return "${BaseTag}_${MortId}_${Sex}"
}

# ---------- Derived ----------
$floor_yearly_real = $KR_MinLiving_Monthly * 12
Write-Host "[Info] Profile=$Profile Data=$DataProfile Seeds=$Seeds PWGammas=$PWGammas"
Write-Host "[Info] KR min living monthly = $($KR_MinLiving_Monthly.ToString('N0')) -> yearly $($floor_yearly_real.ToString('N0'))"
Write-Host "[Info] InitWealth_Real (KRW) = $($InitWealth_Real.ToString('N0'))"

# Mortality mapping to engine flags (use existing switches from cli.py help)
# cli.py shows: --mortality {on,off}, --mort_table <name>
$MortMap = @{
  'BASE'   = @('--mortality','on','--mort_table','base')
  'COHORT' = @('--mortality','on','--mort_table','cohort_2020')
}

# ---------- Core runner ----------
function Invoke-CLI {
  param(
    [string]$Tag,
    [string]$Sex,
    [hashtable]$MortArgs = @{},
    [hashtable]$Extra    = @{}
  )

  $args = @(
    '-m','project.runner.cli',
    '--mode','rl','--method','rl',
    '--data_profile', $DataProfile,
    '--sex', $Sex,
    '--seeds', $Seeds,
    '--print_mode','metrics',
    # NOTE: floor_on 은 presence-only 스위치 (값 주면 안됨)
    '--floor_on',
    '--f_min_real', ('{0:F0}' -f $floor_yearly_real)
    # ⬇️ 이 줄을 삭제합니다:
    # '--init_wealth_real', ('{0:F0}' -f $InitWealth_Real)
  )

  foreach ($k in $MortArgs.Keys) { $args += @($k, $MortArgs[$k]) }
  foreach ($k in $Extra.Keys)    { $args += @($k, $Extra[$k]) }

  if ($Tag -and $Tag -ne '') { $args += @('--tag', $Tag) }

  Write-Host "[RUN] $Tag" -ForegroundColor Cyan
  & .\.venv\Scripts\python.exe @args | Out-Null
}

# ===================================================
# 1) 1D sweeps
# ===================================================
Write-Host "`n[STAGE 1] 1D sweeps" -ForegroundColor Green
foreach ($sex in $Sexes) {
  foreach ($mort in $MortCfg) {
    $mortArgs = @{}
    if ($MortMap.ContainsKey($mort)) {
      $m = $MortMap[$mort]
      for ($i=0;$i -lt $m.Count;$i+=2) { $mortArgs[$m[$i]] = $m[$i+1] }
    }

    # (1) Annuity share (ann_alpha). cli.py has --ann_on {on,off} and --ann_alpha
    foreach ($a in $AnnAlphaGrid) {
      $tag = Make-Tag $mort 'DEV1D_ann' $sex
      $extra = @{ '--ann_alpha' = ('{0:F2}' -f $a) }
      if ($a -gt 0) { $extra['--ann_on'] = 'on' } else { $extra['--ann_on'] = 'off' }
      Invoke-CLI -Tag ("${tag}_{0:F1}" -f $a) -Sex $sex -MortArgs $mortArgs -Extra $extra
    }

    # (2) Risk share.
    # NOTE: If your engine does NOT have --risk_share, use --w_max as a proxy cap.
    foreach ($w in $RiskShareGrid) {
      $tag = Make-Tag $mort 'DEV1D_wrisk' $sex
      # prefer native flag if exists; fallback: w_max = desired cap
      $extra = @{}
      $extra['--w_max'] = ('{0:F2}' -f $w)
      Invoke-CLI -Tag ("${tag}_{0:F1}" -f $w) -Sex $sex -MortArgs $mortArgs -Extra $extra
    }

    # (3) Fee sensitivity (annual)
    foreach ($fee in $FeeGrid) {
      $tag = Make-Tag $mort 'DEV1D_fee' $sex
      Invoke-CLI -Tag ("${tag}_{0:F4}" -f $fee) -Sex $sex -MortArgs $mortArgs -Extra @{ '--fee_annual' = ('{0:F4}' -f $fee) }
    }

    # (4) Retirement age (age0)
    foreach ($age0 in $Age0Grid) {
      $tag = Make-Tag $mort 'DEV1D_age' $sex
      Invoke-CLI -Tag ("${tag}_${age0}") -Sex $sex -MortArgs $mortArgs -Extra @{ '--age0' = "$age0" }
    }

    # (5) Hedge ratio: cli.py shows --hedge {on,off} and --h_fx (or --h_FX)
    foreach ($hz in $HedgeGrid) {
      $tag = Make-Tag $mort 'DEV1D_hedge' $sex
      $on  = if ($hz -gt 0) { 'on' } else { 'off' }
      Invoke-CLI -Tag ("${tag}_{0:F2}" -f $hz) -Sex $sex -MortArgs $mortArgs -Extra @{
        '--hedge' = $on; '--h_fx' = ('{0:F2}' -f $hz)
      }
    }

    # (6) VPW target c*: cli.py has --cstar_mode vpw, --cstar_m
    foreach ($vpw in $VPWGrid) {
      $tag = Make-Tag $mort 'DEV1D_vpw' $sex
      Invoke-CLI -Tag ("${tag}_{0:F3}" -f $vpw) -Sex $sex -MortArgs $mortArgs -Extra @{
        '--cstar_mode'='vpw'; '--cstar_m' = ('{0:F3}' -f $vpw)
      }
    }

    # (7) Asset mixes (example mapping; adjust to your engine)
    foreach ($mix in $AssetMixes) {
      $tag = Make-Tag $mort ("DEV1D_mix_$mix") $sex
      $extra = @{}
      switch ($mix) {
        'EQUAL3' { $extra['--mix_mode']='fixed'; $extra['--w_kr']='0.3333'; $extra['--w_us']='0.3333'; $extra['--w_gold']='0.3333' }
        'KR'     { $extra['--mix_mode']='fixed'; $extra['--w_kr']='1.0';     $extra['--w_us']='0.0';     $extra['--w_gold']='0.0' }
        'US'     { $extra['--mix_mode']='fixed'; $extra['--w_kr']='0.0';     $extra['--w_us']='1.0';     $extra['--w_gold']='0.0' }
        'GLD'    { $extra['--mix_mode']='fixed'; $extra['--w_kr']='0.0';     $extra['--w_us']='0.0';     $extra['--w_gold']='1.0' }
      }
      Invoke-CLI -Tag $tag -Sex $sex -MortArgs $mortArgs -Extra $extra
    }
  }
}

# ===================================================
# 2) 2D sweeps
# ===================================================
Write-Host "`n[STAGE 2] 2D sweeps" -ForegroundColor Green
foreach ($sex in $Sexes) {
  foreach ($mort in $MortCfg) {
    $mortArgs = @{}
    if ($MortMap.ContainsKey($mort)) {
      $m = $MortMap[$mort]
      for ($i=0;$i -lt $m.Count;$i+=2) { $mortArgs[$m[$i]] = $m[$i+1] }
    }

    # (a) ann_alpha x risk (risk via w_max proxy)
    foreach ($a in $AnnAlphaGrid) {
      foreach ($w in $RiskShareGrid) {
        $tag = Make-Tag $mort 'DEV2D_ann_wrisk' $sex
        $extra = @{
          '--w_max'     = ('{0:F2}' -f $w)
          '--ann_alpha' = ('{0:F2}' -f $a)
          '--ann_on'    = $(if ($a -gt 0) { 'on' } else { 'off' })
        }
        Invoke-CLI -Tag ("${tag}_a{0:F1}_w{1:F1}" -f $a,$w) -Sex $sex -MortArgs $mortArgs -Extra $extra
      }
    }

    # (b) risk x hedge
    foreach ($w in $RiskShareGrid) {
      foreach ($hz in $HedgeGrid) {
        $tag = Make-Tag $mort 'DEV2D_wrisk_hedge' $sex
        $extra = @{
          '--w_max' = ('{0:F2}' -f $w)
          '--hedge' = $(if ($hz -gt 0) { 'on' } else { 'off' })
          '--h_fx'  = ('{0:F2}' -f $hz)
        }
        Invoke-CLI -Tag ("${tag}_w{0:F1}_h{1:F2}" -f $w,$hz) -Sex $sex -MortArgs $mortArgs -Extra $extra
      }
    }

    # (c) risk x cstar_m (VPW)
    foreach ($w in $RiskShareGrid) {
      foreach ($vpw in $VPWGrid) {
        $tag = Make-Tag $mort 'DEV2D_wrisk_c' $sex
        $extra = @{
          '--w_max'      = ('{0:F2}' -f $w)
          '--cstar_mode' = 'vpw'
          '--cstar_m'    = ('{0:F3}' -f $vpw)
        }
        Invoke-CLI -Tag ("${tag}_w{0:F1}_c{1:F3}" -f $w,$vpw) -Sex $sex -MortArgs $mortArgs -Extra $extra
      }
    }
  }
}

# ===================================================
# 3) Snapshot / Score / Tables
# ===================================================
Write-Host "`n[STAGE 3] Snapshot / Score / Tables" -ForegroundColor Green

$metricsCsv = Join-Path $Logs 'metrics.csv'
if (Test-Path $metricsCsv) {
  $snap = Join-Path $Out 'DEV_metrics_snapshot.csv'
  (Import-Csv $metricsCsv |
     Group-Object tag,method,sex | ForEach-Object { $_.Group | Select-Object -Last 1 }) |
     Export-Csv $snap -NoTypeInformation -Encoding UTF8
  Write-Host "[OK] snapshot => $snap"
} else {
  Write-Warning "metrics.csv not found: $metricsCsv"
}

# scoring (benefit weights are inside your scorer; override if needed)
& .\.venv\Scripts\python.exe .\scripts\score_snapshot.py `
  --src (Join-Path $Out 'DEV_metrics_snapshot.csv') `
  --metrics_keys "EW,ES95,Ruin" `
  --weights "0.6,0.4" `
  --es_mode "wealth" `
  --out_prefix (Join-Path $Out '') | Out-Null

$OPT_Sum = Join-Path $Out 'OPT_summary_benefit.csv'
$BH_Sum  = Join-Path $Out 'BH_summary_benefit.csv'
$OPT_Core= Join-Path $Out 'OPT_table_core.csv'
$BH_Core = Join-Path $Out 'BH_table_core.csv'
$Compare = Join-Path $Out 'OPT_BH_compare.csv'
$TopBoth = Join-Path $Out 'OPT_BH_top.csv'
$BestBy  = Join-Path $Out 'DEV_OPT_best_by_sex_mort.csv'

if (Test-Path $OPT_Sum) {
  Import-Csv $OPT_Sum |
    Select-Object tag,EW,ES95,Ruin,CompositeScore_benefit |
    Sort-Object @{Expression={ As-Double $_.CompositeScore_benefit }; Descending=$true} |
    Export-Csv $OPT_Core -NoTypeInformation -Encoding UTF8
  Write-Host "[OK] $OPT_Core"
}
if (Test-Path $BH_Sum) {
  Import-Csv $BH_Sum |
    Select-Object tag,EW,ES95,Ruin,CompositeScore_benefit |
    Sort-Object @{Expression={ As-Double $_.CompositeScore_benefit }; Descending=$true} |
    Export-Csv $BH_Core -NoTypeInformation -Encoding UTF8
  Write-Host "[OK] $BH_Core"
}

if ((Test-Path $OPT_Sum) -and (Test-Path $BH_Sum)) {
  $cols     = @('tag','EW','ES95','Ruin','CompositeScore_benefit')
  $optProps = $cols + @(@{Name='Room';Expression={'OPT'}})
  $bhProps  = $cols + @(@{Name='Room';Expression={'BH'}})

  $opt = Import-Csv $OPT_Sum | Select-Object -Property $optProps
  $bh  = Import-Csv $BH_Sum  | Select-Object -Property $bhProps
  $all = $opt + $bh

  $all | Sort-Object Room, @{Expression={ As-Double $_.CompositeScore_benefit }; Descending=$true} |
    Export-Csv $Compare -NoTypeInformation -Encoding UTF8
  Write-Host "[OK] compare => $Compare"

  $optTop3 = Sort-ByScoreDesc $opt | Select-Object -First 3
  $bhTop3  = Sort-ByScoreDesc $bh  | Select-Object -First 3
  ($optTop3 + $bhTop3) | Export-Csv $TopBoth -NoTypeInformation -Encoding UTF8
  Write-Host "[OK] top3+top3 => $TopBoth"
}

# Best by sex x mort (if missing columns, fallback to global top1)
if (Test-Path $OPT_Sum) {
  $sum = Import-Csv $OPT_Sum
  $hasSex  = ($sum | Get-Member -Name sex    -MemberType NoteProperty) -ne $null
  $hasMort = ($sum | Get-Member -Name mort_id -MemberType NoteProperty) -ne $null

  if (-not $hasSex -or -not $hasMort) {
    $best = Sort-ByScoreDesc $sum | Select-Object -First 1
    $best | Export-Csv $BestBy -NoTypeInformation -Encoding UTF8
    Write-Host "[WARN] sex/mort_id not found -> saved global top1: $BestBy"
  } else {
    $best =
      $sum | Group-Object sex, mort_id | ForEach-Object {
        $grp = $_.Group
        Sort-ByScoreDesc $grp | Select-Object -First 1
      }
    $best | Export-Csv $BestBy -NoTypeInformation -Encoding UTF8
    Write-Host "[OK] best by sex x mort => $BestBy"
  }
}

# Just notify optimal_points.json if exists
$PointsJson = Join-Path $Figs 'optimal_points.json'
if (Test-Path $PointsJson) {
  Write-Host "[OK] optimal_points.json -> $PointsJson"
} else {
  Write-Host "[Info] optimal_points.json not found (may be generated by engine)."
}

Write-Host "[DONE] design/score/summary complete" -ForegroundColor Green
