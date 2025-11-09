<# =======================================================================
 run_opt_design.ps1  (PS5-safe, 2025-11-09, refactored full)
 - 1D/2D sweeps -> snapshot -> scoring -> tables
 - Always --mode rl (even in Smoke; reduce epochs/paths only)
 - bias: always pass --bias_prob_gamma (>0), control with --bias_on
 - Flag-guards via SupportsFlag()
 - Per-run logs: outputs\_logs\<tag>.log (single redirection)
 - DEV_scored_clean.csv 자동 생성(빈 tag 제거)
 - FX hedge 자동 감지(--h_fx/--h_FX/--fx_hedge_ratio, --hedge/--fx_hedge_on, --fx_hedge_cost/--hedge_cost)
 - FX 플래그 없으면 레거시 헤지로 안전 대체(--hedge on/off, hedge_mode, hedge_sigma_k 스윕)
 - PS5.1 안전 패턴: 조건식의 -and/-or 우변에 함수 호출 금지(사전 변수 캐싱)
======================================================================= #>

[CmdletBinding()]
param(
  [ValidateSet('light','heavy')] [string]$Profile      = 'light',
  [ValidateSet('dev','full')]    [string]$DataProfile  = 'dev',
  [string]$Seeds                  = '7',
  [string]$PWGammas              = '0.85,0.80,0.70,0.60',

  [double]$KR_MinLiving_Monthly  = 1361000,
  [double]$InitWealth_Real       = 5e8,

  # Grids
  [double[]]$AnnAlphaGrid  = @(0..10 | ForEach-Object { $_/10.0 }),
  [double[]]$RiskShareGrid = @(0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0),
  [double[]]$FeeGrid       = @(0.0000,0.0040,0.0080),
  [int[]]   $Age0Grid      = @(55..65),
  [double[]]$HedgeGrid     = @(0.00,0.25,0.50,0.75,1.00),
  [double[]]$VPWGrid       = @(0.03,0.035,0.04,0.045,0.05,0.055,0.06),

  [ValidateSet('KR','US','GLD','EQUAL3')] [string[]]$AssetMixes = @('EQUAL3','KR','US','GLD'),
  [ValidateSet('M','F')]  [string[]]$Sexes   = @('M','F'),
  [ValidateSet('BASE','COHORT')] [string[]]$MortCfg = @('BASE'),

  # Behavior / Bias
  [ValidateSet('off','on')] [string]$Bias = 'off',
  [double]$BiasProbGamma = 0.65,

  # Smoke (preflight) — still rl; just smaller epochs/paths
  [switch]$Smoke,
  [int]$SmokePaths = 64,
  [int]$SmokeEpochs = 1,

  # ===== 레거시 헤지 기본 파라미터(대체 경로용) =====
  [ValidateSet('sigma','mu','downside')] [string]$LegacyHedgeMode = 'sigma',
  [double[]]$LegacyHedgeSigmas = @(0.10, 0.15, 0.20, 0.25),
  [double]$LegacyHedgeCost = 0.005,
  [double]$LegacyHedgeTx   = 0.000
)

# ---------- Paths ----------
$Root = (Get-Location).Path
$Out  = Join-Path $Root 'outputs'
$Logs = Join-Path $Out  '_logs'
$Figs = Join-Path $Out  'figs'
$null = New-Item -ItemType Directory -Force -Path $Out,$Logs,$Figs | Out-Null

Write-Host "[Guard] Close Excel to avoid PermissionError on CSV/XLSX." -ForegroundColor Yellow

# ---------- Utils ----------
function As-Double($x) { if ($null -eq $x -or "$x" -eq '') {return $null}; try {[double]$x} catch { $null } }
function Sort-ByScoreDesc($rows) {
  $rows | Sort-Object @{ Expression = {
    $v = $_.CompositeScore_benefit
    if ($null -eq $v -or "$v" -eq '') { As-Double $_.CompositeScore } else { As-Double $v }
  }; Descending = $true }
}
function Make-Tag([string]$MortId,[string]$BaseTag,[string]$Sex) { "${BaseTag}_${MortId}_${Sex}" }
function Get-SeedArgs([string]$SeedsCsv) {
  $parts = $SeedsCsv -split '[,\s]+' | Where-Object { $_ -ne '' }
  if ($parts.Count -eq 0) { return @('--seeds','7') }
  @('--seeds') + $parts
}

# Preload cli help once (for flag detection)
$global:CLI_HELP = (& .\.venv\Scripts\python.exe -m project.runner.cli -h 2>&1 | Out-String)
function SupportsFlag([string]$flag) {
  $pattern = ("--" + [regex]::Escape($flag) + "(\s|$|=)")
  return ($global:CLI_HELP -match $pattern)
}

# Asset mix → args (prefer alpha_kr/us/au; fallback alpha_mix)
function Build-MixArgs([string]$mix) {
  $a = @{}
  $hasKR = SupportsFlag('alpha_kr')
  $hasUS = SupportsFlag('alpha_us')
  $hasAU = SupportsFlag('alpha_au')
  $hasMix = SupportsFlag('alpha_mix')

  if ($hasKR -and $hasUS -and $hasAU) {
    switch ($mix) {
      'EQUAL3' { $a['--alpha_kr']='0.3333'; $a['--alpha_us']='0.3333'; $a['--alpha_au']='0.3333' }
      'KR'     { $a['--alpha_kr']='1.0';    $a['--alpha_us']='0.0';    $a['--alpha_au']='0.0' }
      'US'     { $a['--alpha_kr']='0.0';    $a['--alpha_us']='1.0';    $a['--alpha_au']='0.0' }
      'GLD'    { $a['--alpha_kr']='0.0';    $a['--alpha_us']='0.0';    $a['--alpha_au']='1.0' }
    }
  } elseif ($hasMix) {
    switch ($mix) {
      'EQUAL3' { $a['--alpha_mix']='kr:0.3333,us:0.3333,au:0.3333' }
      'KR'     { $a['--alpha_mix']='kr:1.0,us:0.0,au:0.0' }
      'US'     { $a['--alpha_mix']='kr:0.0,us:1.0,au:0.0' }
      'GLD'    { $a['--alpha_mix']='kr:0.0,us:0.0,au:1.0' }
    }
  }
  return $a
}

# Mortality mapping
$MortMap = @{
  'BASE'   = @('--mortality','on','--mort_table','base')
  'COHORT' = @('--mortality','on','--mort_table','cohort_2020')
}

# ---------- FX Hedge flags (robust detection) ----------
function Get-FxFlags {
  $has_h_fx            = SupportsFlag('h_fx')
  $has_h_FX            = SupportsFlag('h_FX')
  $has_fx_hedge_ratio  = SupportsFlag('fx_hedge_ratio')

  $ratio = $null
  if     ($has_h_fx)           { $ratio = '--h_fx' }
  elseif ($has_h_FX)           { $ratio = '--h_FX' }
  elseif ($has_fx_hedge_ratio) { $ratio = '--fx_hedge_ratio' }

  $has_hedge       = SupportsFlag('hedge')
  $has_fx_hedge_on = SupportsFlag('fx_hedge_on')

  $onoff = $null
  if     ($has_hedge)       { $onoff = '--hedge' }
  elseif ($has_fx_hedge_on) { $onoff = '--fx_hedge_on' }

  $cost = $null
  if     (SupportsFlag('fx_hedge_cost')) { $cost = '--fx_hedge_cost' }
  elseif (SupportsFlag('hedge_cost'))    { $cost = '--hedge_cost' }

  return @{ ratio = $ratio; onoff = $onoff; cost = $cost }
}
$FxFlags = Get-FxFlags

# ---------- Base args builder (ALWAYS mode rl) ----------
function New-BaseArgs {
  param(
    [string]$Sex,
    [hashtable]$MortArgs = @{},
    [hashtable]$Extra    = @{}
  )

  $args = @(
    '-m','project.runner.cli',
    '--mode','rl',
    '--method','rl',
    '--data_profile', $DataProfile,
    '--sex', $Sex,
    '--print_mode','metrics',
    '--floor_on','--f_min_real', ('{0:F0}' -f ($KR_MinLiving_Monthly*12))
  )

  # seeds
  $args += (Get-SeedArgs -SeedsCsv $Seeds)

  # bias: always pass gamma; toggle ON/OFF with --bias_on
  $has_bias_prob_gamma = SupportsFlag('bias_prob_gamma')
  $has_bias_on         = SupportsFlag('bias_on')
  if ($has_bias_prob_gamma) { $args += @('--bias_prob_gamma', ('{0:F3}' -f $BiasProbGamma)) }
  if ($has_bias_on)         { $args += @('--bias_on', $(if ($Bias -eq 'on') {'on'} else {'off'})) }

  # Smoke: reduce workload (still rl)
  $has_rl_epochs          = SupportsFlag('rl_epochs')
  $has_rl_steps_per_epoch = SupportsFlag('rl_steps_per_epoch')
  $has_n_paths            = SupportsFlag('n_paths')
  $has_quiet              = SupportsFlag('quiet')

  if ($Smoke) {
    if ($has_rl_epochs)          { $args += @('--rl_epochs', "$SmokeEpochs") }
    if ($has_rl_steps_per_epoch) { $args += @('--rl_steps_per_epoch','128') }
    if ($has_n_paths)            { $args += @('--n_paths', "$SmokePaths") }
    if ($has_quiet)              { $args += @('--quiet','on') }
  }

  foreach ($k in $MortArgs.Keys) { $args += @($k, $MortArgs[$k]) }
  foreach ($k in $Extra.Keys)    { $args += @($k, $Extra[$k]) }

  return ,$args
}

# ---------- Core runner with logging ----------
function Invoke-CLI {
  param(
    [string]$Tag,
    [string]$Sex,
    [hashtable]$MortArgs = @{},
    [hashtable]$Extra    = @{}
  )
  $args = New-BaseArgs -Sex $Sex -MortArgs $MortArgs -Extra $Extra
  if ($Tag) { $args += @('--tag', $Tag) }

  $log = Join-Path $Logs ("{0}.log" -f ($Tag -replace '[^\w\-\.]','_'))
  Write-Host "[RUN] $Tag  -> $log" -ForegroundColor Cyan

  & ".\.venv\Scripts\python.exe" @args *> $log
  $ec = $LASTEXITCODE
  if ($ec -ne 0) { Add-Content -Path $log -Value "[ERR] exit=$ec  tag=$Tag" -Encoding utf8 }
  return $ec
}

# ---------- Helpers: scoring clean-up ----------
function Export-CleanScored {
  $scr = Join-Path $Out 'DEV_scored.csv'
  $clean = Join-Path $Out 'DEV_scored_clean.csv'
  if (Test-Path $scr) {
    Import-Csv $scr | Where-Object { $_.tag -and $_.tag.Trim() -ne '' } |
      Export-Csv $clean -NoTypeInformation -Encoding UTF8
    Write-Host "[OK] cleaned => $clean"
  } else {
    Write-Host "[Info] DEV_scored.csv not found, skip cleaning."
  }
}

# ---------- Banners ----------
$floor_yearly_real = $KR_MinLiving_Monthly * 12
Write-Host "[Info] Profile=$Profile Data=$DataProfile Seeds=$Seeds PWGammas=$PWGammas"
Write-Host "[Info] KR min living monthly = $($KR_MinLiving_Monthly.ToString('N0')) -> yearly $($floor_yearly_real.ToString('N0'))"
Write-Host "[Info] InitWealth_Real (KRW) = $($InitWealth_Real.ToString('N0'))"
Write-Host "[Info] Bias=$Bias  BiasProbGamma=$BiasProbGamma  Smoke=$($Smoke.IsPresent)"

# ===================================================
# 0) Smoke preflight (optional)
# ===================================================
if ($Smoke) {
  Write-Host "`n[STAGE 0] Smoke preflight" -ForegroundColor Green
  $fail = 0
  foreach ($sex in @('M')) {
    $m = $MortMap['BASE']
    $mortArgs = @{}
    for ($i=0;$i -lt $m.Count;$i+=2) { $mortArgs[$m[$i]] = $m[$i+1] }

    # ann_alpha
    $has_ann_on = SupportsFlag('ann_on')
    $extra1 = @{'--ann_alpha'='0.10'}
    if ($has_ann_on) { $extra1['--ann_on'] = 'on' }
    $fail += (Invoke-CLI -Tag 'SMOKE_ann_0.1'   -Sex $sex -MortArgs $mortArgs -Extra $extra1)

    # wrisk via w_max proxy
    $fail += (Invoke-CLI -Tag 'SMOKE_wrisk_0.3' -Sex $sex -MortArgs $mortArgs -Extra @{ '--w_max'='0.30' })

    # hedge: FX 우선, 없으면 레거시 on/off 테스트
    $extra2 = @{}
    if ($FxFlags.ratio) { $extra2[$FxFlags.ratio] = '0.50' }
    if ($FxFlags.onoff) { $extra2[$FxFlags.onoff] = 'on' }

    if ($extra2.Count -gt 0) {
      $fail += (Invoke-CLI -Tag 'SMOKE_hedge_0.5' -Sex $sex -MortArgs $mortArgs -Extra $extra2)
    } else {
      $has_legacy_hedge = SupportsFlag('hedge')
      if ($has_legacy_hedge) {
        $has_hedge_cost    = SupportsFlag('hedge_cost')
        $has_hedge_tx      = SupportsFlag('hedge_tx')
        $has_hedge_sigma_k = SupportsFlag('hedge_sigma_k')
        foreach ($on in @('off','on')) {
          $x = @{ '--hedge'=$on; '--hedge_mode'=$LegacyHedgeMode }
          if ($has_hedge_cost)    { $x['--hedge_cost'] = ('{0:F3}' -f $LegacyHedgeCost) }
          if ($has_hedge_tx)      { $x['--hedge_tx']   = ('{0:F3}' -f $LegacyHedgeTx) }
          if ( ($on -eq 'on') -and $has_hedge_sigma_k) { $x['--hedge_sigma_k']='0.20' }
          $fail += (Invoke-CLI -Tag ("SMOKE_legacy_hedge_{0}" -f $on) -Sex $sex -MortArgs $mortArgs -Extra $x)
        }
      } else {
        Write-Host "[SMOKE] Skip hedge check: no FX/legacy flags found." -ForegroundColor Yellow
      }
    }

    # mix
    $mixArgs = Build-MixArgs 'EQUAL3'
    if ($mixArgs.Count -gt 0) { $fail += (Invoke-CLI -Tag 'SMOKE_mix_equal3' -Sex $sex -MortArgs $mortArgs -Extra $mixArgs) }
  }
  if ($fail -ne 0) {
    Write-Host  "[SMOKE] Preflight failed. Check logs under $Logs and fix flags." -ForegroundColor Yellow
    exit 1
  } else {
    Write-Host "[SMOKE] OK — flags good. Proceeding." -ForegroundColor Green
  }
}

# ===================================================
# 1) 1D sweeps
# ===================================================
Write-Host "`n[STAGE 1] 1D sweeps" -ForegroundColor Green
foreach ($sex in $Sexes) {
  foreach ($mort in $MortCfg) {
    $m = $MortMap[$mort]
    $mortArgs = @{}
    for ($i=0;$i -lt $m.Count;$i+=2) { $mortArgs[$m[$i]] = $m[$i+1] }

    $has_ann_on = SupportsFlag('ann_on')
    foreach ($a in $AnnAlphaGrid) {
      $tag = Make-Tag $mort 'DEV1D_ann' $sex
      $extra = @{ '--ann_alpha' = ('{0:F2}' -f $a) }
      if ($has_ann_on) { $extra['--ann_on'] = $(if ($a -gt 0) {'on'} else {'off'}) }
      Invoke-CLI -Tag ("${tag}_{0:F1}" -f $a) -Sex $sex -MortArgs $mortArgs -Extra $extra | Out-Null
    }

    foreach ($w in $RiskShareGrid) {
      $tag = Make-Tag $mort 'DEV1D_wrisk' $sex
      Invoke-CLI -Tag ("${tag}_{0:F1}" -f $w) -Sex $sex -MortArgs $mortArgs -Extra @{ '--w_max' = ('{0:F2}' -f $w) } | Out-Null
    }

    foreach ($fee in $FeeGrid) {
      $tag = Make-Tag $mort 'DEV1D_fee' $sex
      Invoke-CLI -Tag ("${tag}_{0:F4}" -f $fee) -Sex $sex -MortArgs $mortArgs -Extra @{ '--fee_annual' = ('{0:F4}' -f $fee) } | Out-Null
    }

    foreach ($age0 in $Age0Grid) {
      $tag = Make-Tag $mort 'DEV1D_age' $sex
      Invoke-CLI -Tag ("${tag}_${age0}") -Sex $sex -MortArgs $mortArgs -Extra @{ '--age0' = "$age0" } | Out-Null
    }

    # --- Hedge 1D: FX 있으면 비율 스윕, 없으면 레거시 on/off ---
    $hasFX     = ($FxFlags.ratio -ne $null)
    $hasLegacy = SupportsFlag('hedge')

    if ($hasFX) {
      foreach ($hz in $HedgeGrid) {
        $tag = Make-Tag $mort 'DEV1D_hedge' $sex
        $extra = @{}
        $extra[$FxFlags.ratio] = ('{0:F2}' -f $hz)
        if ($FxFlags.onoff) { $extra[$FxFlags.onoff] = $(if ($hz -gt 0){'on'} else {'off'}) }
        if ($FxFlags.cost -and $hz -gt 0) { $extra[$FxFlags.cost] = '0.000' }
        Invoke-CLI -Tag ("${tag}_{0:F2}" -f $hz) -Sex $sex -MortArgs $mortArgs -Extra $extra | Out-Null
      }
    } elseif ($hasLegacy) {
      $has_hedge_cost    = SupportsFlag('hedge_cost')
      $has_hedge_tx      = SupportsFlag('hedge_tx')
      $has_hedge_sigma_k = SupportsFlag('hedge_sigma_k')
      foreach ($on in @('off','on')) {
        $tag = Make-Tag $mort 'DEV1D_hedge' $sex
        $x = @{ '--hedge'=$on; '--hedge_mode'=$LegacyHedgeMode }
        if ($has_hedge_cost)    { $x['--hedge_cost'] = ('{0:F3}' -f $LegacyHedgeCost) }
        if ($has_hedge_tx)      { $x['--hedge_tx']   = ('{0:F3}' -f $LegacyHedgeTx) }
        if ( ($on -eq 'on') -and $has_hedge_sigma_k) { $x['--hedge_sigma_k'] = ('{0:F2}' -f $LegacyHedgeSigmas[0]) }
        Invoke-CLI -Tag ("${tag}_LEGACY_{0}" -f $on) -Sex $sex -MortArgs $mortArgs -Extra $x | Out-Null
      }
    } else {
      Write-Warning "[hedge] No FX nor legacy hedge flags; skipping hedge 1D."
    }

    $has_cstar_mode = SupportsFlag('cstar_mode')
    $has_cstar_m    = SupportsFlag('cstar_m')
    foreach ($vpw in $VPWGrid) {
      $tag = Make-Tag $mort 'DEV1D_vpw' $sex
      $extra = @{}
      if ($has_cstar_mode) { $extra['--cstar_mode'] = 'vpw' }
      if ($has_cstar_m)    { $extra['--cstar_m']    = ('{0:F3}' -f $vpw) }
      if ($extra.Count -gt 0) {
        Invoke-CLI -Tag ("${tag}_{0:F3}" -f $vpw) -Sex $sex -MortArgs $mortArgs -Extra $extra | Out-Null
      }
    }

    foreach ($mix in $AssetMixes) {
      $tag = Make-Tag $mort ("DEV1D_mix_$mix") $sex
      $extra = Build-MixArgs $mix
      if ($extra.Count -gt 0) { Invoke-CLI -Tag $tag -Sex $sex -MortArgs $mortArgs -Extra $extra | Out-Null }
    }
  }
}

# ===================================================
# 2) 2D sweeps
# ===================================================
Write-Host "`n[STAGE 2] 2D sweeps" -ForegroundColor Green
foreach ($sex in $Sexes) {
  foreach ($mort in $MortCfg) {
    $m = $MortMap[$mort]
    $mortArgs = @{}
    for ($i=0;$i -lt $m.Count;$i+=2) { $mortArgs[$m[$i]] = $m[$i+1] }

    $has_ann_on = SupportsFlag('ann_on')
    foreach ($a in $AnnAlphaGrid) {
      foreach ($w in $RiskShareGrid) {
        $tag = Make-Tag $mort 'DEV2D_ann_wrisk' $sex
        $extra = @{ '--w_max'=('{0:F2}' -f $w); '--ann_alpha'=('{0:F2}' -f $a) }
        if ($has_ann_on) { $extra['--ann_on'] = $(if ($a -gt 0) {'on'} else {'off'}) }
        Invoke-CLI -Tag ("${tag}_a{0:F1}_w{1:F1}" -f $a,$w) -Sex $sex -MortArgs $mortArgs -Extra $extra | Out-Null
      }
    }

    # --- Hedge 2D: wrisk × (FX ratio or legacy sigma_k) ---
    $hasFX     = ($FxFlags.ratio -ne $null)
    $hasLegacy = SupportsFlag('hedge')

    if ($hasFX) {
      foreach ($w in $RiskShareGrid) {
        foreach ($hz in $HedgeGrid) {
          $tag = Make-Tag $mort 'DEV2D_wrisk_hedge' $sex
          $extra = @{ '--w_max'=('{0:F2}' -f $w) }
          $extra[$FxFlags.ratio] = ('{0:F2}' -f $hz)
          if ($FxFlags.onoff) { $extra[$FxFlags.onoff] = $(if ($hz -gt 0){'on'} else {'off'}) }
          if ($FxFlags.cost -and $hz -gt 0) { $extra[$FxFlags.cost] = '0.000' }
          Invoke-CLI -Tag ("${tag}_w{0:F1}_h{1:F2}" -f $w,$hz) -Sex $sex -MortArgs $mortArgs -Extra $extra | Out-Null
        }
      }
    } elseif ($hasLegacy) {
      $has_hedge_cost    = SupportsFlag('hedge_cost')
      $has_hedge_tx      = SupportsFlag('hedge_tx')
      $has_hedge_sigma_k = SupportsFlag('hedge_sigma_k')
      foreach ($w in $RiskShareGrid) {
        foreach ($k in $LegacyHedgeSigmas) {
          $tag = Make-Tag $mort 'DEV2D_wrisk_hedgeSIG' $sex
          $extra = @{ '--w_max'=('{0:F2}' -f $w); '--hedge'='on'; '--hedge_mode'=$LegacyHedgeMode }
          if ($has_hedge_cost)    { $extra['--hedge_cost'] = ('{0:F3}' -f $LegacyHedgeCost) }
          if ($has_hedge_tx)      { $extra['--hedge_tx']   = ('{0:F3}' -f $LegacyHedgeTx) }
          if ($has_hedge_sigma_k) { $extra['--hedge_sigma_k'] = ('{0:F2}' -f $k) }
          Invoke-CLI -Tag ("${tag}_w{0:F1}_k{1:F2}" -f $w,$k) -Sex $sex -MortArgs $mortArgs -Extra $extra | Out-Null
        }
      }
    } else {
      Write-Warning "[hedge] No FX nor legacy hedge flags; skipping wrisk×hedge 2D."
    }

    $has_cstar_mode = SupportsFlag('cstar_mode')
    $has_cstar_m    = SupportsFlag('cstar_m')
    foreach ($w in $RiskShareGrid) {
      foreach ($vpw in $VPWGrid) {
        $tag = Make-Tag $mort 'DEV2D_wrisk_c' $sex
        $extra = @{ '--w_max'=('{0:F2}' -f $w) }
        if ($has_cstar_mode) { $extra['--cstar_mode'] = 'vpw' }
        if ($has_cstar_m)    { $extra['--cstar_m']    = ('{0:F3}' -f $vpw) }
        Invoke-CLI -Tag ("${tag}_w{0:F1}_c{1:F3}" -f $w,$vpw) -Sex $sex -MortArgs $mortArgs -Extra $extra | Out-Null
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
  (Import-Csv $metricsCsv | Group-Object tag,method,sex | ForEach-Object { $_.Group | Select-Object -Last 1 }) |
    Export-Csv $snap -NoTypeInformation -Encoding UTF8
  Write-Host "[OK] snapshot => $snap"
} else {
  Write-Warning "metrics.csv not found: $metricsCsv"
}

& .\.venv\Scripts\python.exe .\scripts\score_snapshot.py `
  --src (Join-Path $Out 'DEV_metrics_snapshot.csv') `
  --metrics "EW,ES95" `
  --weights "0.6,0.4" `
  --es_mode "wealth" `
  --out (Join-Path $Out 'DEV_scored.csv')

# --- Clean scored (drop blank tag rows)
Export-CleanScored

$OPT_Sum = Join-Path $Out 'OPT_summary_benefit.csv'
$BH_Sum  = Join-Path $Out 'BH_summary_benefit.csv'
$OPT_Core= Join-Path $Out 'OPT_table_core.csv'
$BH_Core = Join-Path $Out 'BH_table_core.csv'
$Compare = Join-Path $Out 'OPT_BH_compare.csv'
$TopBoth = Join-Path $Out 'OPT_BH_top.csv'
$BestBy  = Join-Path $Out 'DEV_OPT_best_by_sex_mort.csv'

if (Test-Path $OPT_Sum) {
  Import-Csv $OPT_Sum | Select-Object tag,EW,ES95,Ruin,CompositeScore_benefit |
    Sort-Object @{Expression={ As-Double $_.CompositeScore_benefit }; Descending=$true} |
    Export-Csv $OPT_Core -NoTypeInformation -Encoding UTF8
  Write-Host "[OK] $OPT_Core"
}
if (Test-Path $BH_Sum) {
  Import-Csv $BH_Sum | Select-Object tag,EW,ES95,Ruin,CompositeScore_benefit |
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

if (Test-Path $OPT_Sum) {
  $sum = Import-Csv $OPT_Sum
  $hasSex  = ($sum | Get-Member -Name sex     -MemberType NoteProperty) -ne $null
  $hasMort = ($sum | Get-Member -Name mort_id -MemberType NoteProperty) -ne $null
  if (-not $hasSex -or -not $hasMort) {
    Sort-ByScoreDesc $sum | Select-Object -First 1 | Export-Csv $BestBy -NoTypeInformation -Encoding UTF8
    Write-Host "[WARN] sex/mort_id missing -> saved global top1: $BestBy"
  } else {
    ($sum | Group-Object sex, mort_id | ForEach-Object { Sort-ByScoreDesc $_.Group | Select-Object -First 1 }) |
      Export-Csv $BestBy -NoTypeInformation -Encoding UTF8
    Write-Host "[OK] best by sex x mort => $BestBy"
  }
}

$PointsJson = Join-Path $Figs 'optimal_points.json'
if (Test-Path $PointsJson) { Write-Host "[OK] optimal_points.json -> $PointsJson" }
else { Write-Host "[Info] optimal_points.json not found." }

Write-Host "[DONE] design/score/summary complete" -ForegroundColor Green
