<# =======================================================================  
 run_opt_design.ps1  (PS5-safe, 2025-11-10, full refactor w/ CPI policy)
 - 기존 1D/2D 파이프라인 유지
 - CPI 정책 반영:
   * 기본 실질 모드: --use_real_rf on (글로벌)  ※ MAIN_ANN은 케이스별 명시
   * 명목 비교가 필요한 경우에만 --use_real_rf off
   * (옵션) 부트스트랩 CSV 스키마/블록 길이 사전 점검
 - NEW: -MainANN 스위치로 “공정가 vs 현실가(실질/명목)” 3세트 자동 실행
 - RL 고정(스모크 시 epochs/paths 축소만), per-run 로그 저장
 - DEV_scored_clean.csv 자동 생성(빈 tag 제거)
 - FX hedge 자동 감지(존재 시만 사용)
 - PowerShell 5.1 호환(해시 리터럴/조건문 정리)
======================================================================= #>

[CmdletBinding()]
param(
  [ValidateSet('light','heavy')] [string]$Profile      = 'light',
  [ValidateSet('dev','full')]    [string]$DataProfile  = 'dev',
  [string]$Seeds                  = '7',
  [string]$PWGammas              = '0.85,0.80,0.70,0.60',

  [double]$KR_MinLiving_Monthly  = 1361000,
  [double]$InitWealth_Real       = 5e8,

  # ===== CPI / Real-mode policy =====
  [ValidateSet('on','off')] [string]$UseRealRF = 'on',     # 기본 실질 모드
  [switch]$CheckCpiSchema,                                  # CSV 헤더/블록 길이 점검 스위치
  [string]$BootstrapCsv = "",                               # (선택) 점검할 CSV 경로
  [int]$BootstrapBlock = 24,                                # 월 단위 블록(기본 24)

  # Grids (기존 파이프라인용)
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

  # ===== MAIN_ANN 모드(신규) =====
  [switch]$MainANN,                         # ← 켜면 공정가/현실가 3세트 자동 실행
  [ValidateSet('iid','bootstrap')] [string]$MarketMode = 'bootstrap',
  [string]$MainANNSeed = '7',               # 재현성(단일 seed 우선)
  [int]$MainANNEpochs = 1,
  [int]$MainANNStepsPerEpoch = 256,
  [int]$MainANNPaths = 1000,
  [double]$MainANNPhi = 0.03,               # 현실가 로딩(φ_adval)
  [ValidateSet('BASE','cohort_2020')] [string]$MainANNMortFair = 'BASE',
  [ValidateSet('BASE','cohort_2020')] [string]$MainANNMortReal = 'cohort_2020'
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

# (옵션) CPI CSV 스키마/블록 길이 점검
function Test-CpiSchema {
  param([string]$CsvPath,[int]$Block)
  if (-not $CsvPath -or -not (Test-Path $CsvPath)) {
    Write-Host "[CPI] Skip schema check (file not provided/found)." -ForegroundColor Yellow
    return
  }
  $first = Get-Content -Path $CsvPath -TotalCount 1
  if (-not $first) { Write-Warning "[CPI] Empty CSV: $CsvPath"; return }
  $hdr = $first.Split(',') | ForEach-Object { $_.Trim() }
  $need = @('date','risky_nom','tbill_nom','cpi')
  $miss = @()
  foreach($h in $need){ if ($hdr -notcontains $h){ $miss += $h } }
  if ($miss.Count -gt 0) {
    Write-Warning ("[CPI] Missing headers: {0}  (file={1})" -f ($miss -join ', '), $CsvPath)
  } else {
    Write-Host "[CPI] Header OK: date,risky_nom,tbill_nom,cpi" -ForegroundColor Green
  }
  if ($Block -ne 24) {
    Write-Host ("[CPI] Note: bootstrap_block={0} (policy recommends 24 months)" -f $Block) -ForegroundColor Yellow
  } else {
    Write-Host "[CPI] bootstrap_block=24 verified (policy-aligned)." -ForegroundColor Green
  }
}

# CLI 도움말 1회 캐시(플래그 감지)
$global:CLI_HELP = (& .\.venv\Scripts\python.exe -m project.runner.cli -h 2>&1 | Out-String)
function SupportsFlag([string]$flag) { return $global:CLI_HELP -match ("--" + [regex]::Escape($flag) + "(\s|$|=)") }

# Asset mix → args
function Build-MixArgs([string]$mix) {
  $a = @{}
  $has_kr = SupportsFlag('alpha_kr'); $has_us = SupportsFlag('alpha_us'); $has_au = SupportsFlag('alpha_au')
  $has_mix = SupportsFlag('alpha_mix')
  if ($has_kr -and $has_us -and $has_au) {
    switch ($mix) {
      'EQUAL3' { $a['--alpha_kr']='0.3333'; $a['--alpha_us']='0.3333'; $a['--alpha_au']='0.3333' }
      'KR'     { $a['--alpha_kr']='1.0';    $a['--alpha_us']='0.0';    $a['--alpha_au']='0.0' }
      'US'     { $a['--alpha_kr']='0.0';    $a['--alpha_us']='1.0';    $a['--alpha_au']='0.0' }
      'GLD'    { $a['--alpha_kr']='0.0';    $a['--alpha_us']='0.0';    $a['--alpha_au']='1.0' }
    }
  } elseif ($has_mix) {
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

# FX hedge flags (존재 시만 사용)
function Get-FxFlags {
  $ratio = $null; $onoff = $null; $cost = $null
  if (SupportsFlag('h_fx')) { $ratio='--h_fx' }
  elseif (SupportsFlag('h_FX')) { $ratio='--h_FX' }
  elseif (SupportsFlag('fx_hedge_ratio')) { $ratio='--fx_hedge_ratio' }
  if (SupportsFlag('hedge')) { $onoff='--hedge' }
  elseif (SupportsFlag('fx_hedge_on')) { $onoff='--fx_hedge_on' }
  if (SupportsFlag('fx_hedge_cost')) { $cost='--fx_hedge_cost' }
  elseif (SupportsFlag('hedge_cost')) { $cost='--hedge_cost' }
  return @{ ratio=$ratio; onoff=$onoff; cost=$cost }
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
  $args += (Get-SeedArgs -SeedsCsv $Seeds)

  # CPI/Real-mode(글로벌): 엔진에 플래그가 있을 때만 추가
  if (SupportsFlag('use_real_rf')) { $args += @('--use_real_rf', $UseRealRF) }

  # Bias
  if (SupportsFlag('bias_prob_gamma')) { $args += @('--bias_prob_gamma', ('{0:F3}' -f $BiasProbGamma)) }
  if (SupportsFlag('bias_on'))         { $args += @('--bias_on', $(if ($Bias -eq 'on') {'on'} else {'off'})) }

  # Smoke workload 축소
  if ($Smoke) {
    if (SupportsFlag('rl_epochs'))          { $args += @('--rl_epochs', "$SmokeEpochs") }
    if (SupportsFlag('rl_steps_per_epoch')) { $args += @('--rl_steps_per_epoch','128') }
    if (SupportsFlag('n_paths'))            { $args += @('--n_paths', "$SmokePaths") }
    if (SupportsFlag('quiet'))              { $args += @('--quiet','on') }
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
Write-Host "[Info] use_real_rf(global) = $UseRealRF"
if ($CheckCpiSchema) { Test-CpiSchema -CsvPath $BootstrapCsv -Block $BootstrapBlock }

# ===================================================
# NEW) MAIN_ANN 모드: 공정가 vs 현실가(실질/명목) 3세트 실행
# ===================================================
if ($MainANN) {
  Write-Host "`n[MAIN_ANN] Annuity 현실화 비교 세트 실행" -ForegroundColor Green

  $Py = ".\.venv\Scripts\python.exe"
  $COMMON = @(
    '--method','rl','--mode','rl',
    '--rl_epochs', "$MainANNEpochs",
    '--rl_steps_per_epoch', "$MainANNStepsPerEpoch",
    '--n_paths', "$MainANNPaths",
    '--seed', "$MainANNSeed",
    '--eval_seed_jitter','off',
    '--print_mode','summary',
    '--data_profile','full','--market_mode', $MarketMode
  )
  if (SupportsFlag('bias_on')) { $COMMON += @('--bias_on','off') }

  # 1) FAIR(무로딩, BASE, 실질) ※ 케이스별 실질 명시
  $FAIR = @('--tag','MAIN_ANN_FAIR','--mortality','on','--mort_table',$MainANNMortFair,'--phi_adval','0.0','--ann_index','real')
  if (SupportsFlag('use_real_rf')) { $FAIR += @('--use_real_rf','on') }
  Write-Host "[RUN] MAIN_ANN_FAIR" -ForegroundColor Cyan
  & $Py @('-m','project.runner.cli') @FAIR @COMMON *> (Join-Path $Logs 'MAIN_ANN_FAIR.log')

  # 2) REAL(로딩 φ, 코호트, 실질)
  $phi = ('{0:F2}' -f $MainANNPhi)
  $REAL = @('--tag','MAIN_ANN_REAL','--mortality','on','--mort_table',$MainANNMortReal,'--phi_adval',$phi,'--ann_index','real')
  if (SupportsFlag('use_real_rf')) { $REAL += @('--use_real_rf','on') }
  Write-Host "[RUN] MAIN_ANN_REAL" -ForegroundColor Cyan
  & $Py @('-m','project.runner.cli') @REAL @COMMON *> (Join-Path $Logs 'MAIN_ANN_REAL.log')

  # 3) REAL_NOM(로딩 φ, 코호트, 명목연금 + 명목RF off)
  $REALN = @('--tag','MAIN_ANN_REAL_NOM','--mortality','on','--mort_table',$MainANNMortReal,'--phi_adval',$phi,'--ann_index','nominal')
  if (SupportsFlag('use_real_rf')) { $REALN += @('--use_real_rf','off') }
  Write-Host "[RUN] MAIN_ANN_REAL_NOM" -ForegroundColor Cyan
  & $Py @('-m','project.runner.cli') @REALN @COMMON *> (Join-Path $Logs 'MAIN_ANN_REAL_NOM.log')

  # 스냅샷/점수화
  $metricsCsv = Join-Path $Logs 'metrics.csv'
  $snap = Join-Path $Out 'MAIN_ANN_snapshot.csv'
  if (Test-Path $metricsCsv) {
    (Import-Csv $metricsCsv |
      Where-Object { $_.tag -in 'MAIN_ANN_FAIR','MAIN_ANN_REAL','MAIN_ANN_REAL_NOM' } |
      Group-Object tag,method,seed | ForEach-Object { $_.Group | Select-Object -Last 1 }) |
      Export-Csv $snap -NoTypeInformation -Encoding UTF8
    Write-Host "[OK] snapshot => $snap"
  } else {
    Write-Warning "[MAIN_ANN] metrics.csv not found: $metricsCsv"
  }

  & $Py .\scripts\score_snapshot.py `
    --src $snap `
    --tag_startswith MAIN_ANN_ `
    --metrics "EW,ES95" `
    --weights "0.6,0.4" `
    --es_mode "wealth" `
    --out "inplace"

  # README
  $readme = @"
# MAIN | Annuity 현실화 비교
- 환경: data_profile=full, market_mode=$MarketMode, seed=$MainANNSeed, n_paths=$MainANNPaths, RL(epochs=$MainANNEpochs, steps=$MainANNStepsPerEpoch), bias_off
- 케이스:
  - MAIN_ANN_FAIR        = 공정가(무로딩, $MainANNMortFair, 실질)
  - MAIN_ANN_REAL        = 로딩 $phi + $MainANNMortReal (실질)
  - MAIN_ANN_REAL_NOM    = 로딩 $phi + $MainANNMortReal (명목연금, 명목RF)
- 핵심지표: EW, ES95, RuinPct(=0), mean_WT
"@
  $readmePath = Join-Path $Out 'MAIN_ANN_README.md'
  $readme | Set-Content $readmePath -Encoding UTF8
  Write-Host "[OK] README  => $readmePath"

  Write-Host "[DONE] MAIN_ANN complete" -ForegroundColor Green
  exit 0
}

# ===================================================
# 0) Smoke preflight (optional, 기존)
# ===================================================
if ($Smoke) {
  Write-Host "`n[STAGE 0] Smoke preflight" -ForegroundColor Green
  $fail = 0
  foreach ($sex in @('M')) {
    $m = $MortMap['BASE']; $mortArgs=@{}; for ($i=0;$i -lt $m.Count;$i+=2){ $mortArgs[$m[$i]]=$m[$i+1] }

    $extra1 = @{'--ann_alpha'='0.10'}; if (SupportsFlag('ann_on')){ $extra1['--ann_on']='on' }
    $fail += (Invoke-CLI -Tag 'SMOKE_ann_0.1' -Sex $sex -MortArgs $mortArgs -Extra $extra1)

    $fail += (Invoke-CLI -Tag 'SMOKE_wrisk_0.3' -Sex $sex -MortArgs $mortArgs -Extra @{ '--w_max'='0.30' })

    $extra2 = @{}
    if ($FxFlags.ratio){ $extra2[$FxFlags.ratio]='0.50' }
    if ($FxFlags.onoff){ $extra2[$FxFlags.onoff]='on' }
    if ($extra2.Count -gt 0){
      $fail += (Invoke-CLI -Tag 'SMOKE_hedge_0.5' -Sex $sex -MortArgs $mortArgs -Extra $extra2)
    } else {
      Write-Host "[SMOKE] Skip hedge check: FX flags not found." -ForegroundColor Yellow
    }

    $mixArgs = Build-MixArgs 'EQUAL3'
    if ($mixArgs.Count -gt 0){ $fail += (Invoke-CLI -Tag 'SMOKE_mix_equal3' -Sex $sex -MortArgs $mortArgs -Extra $mixArgs) }
  }
  if ($fail -ne 0){ Write-Host "[SMOKE] Preflight failed. Check logs." -ForegroundColor Yellow; exit 1 }
  Write-Host "[SMOKE] OK — flags good. Proceeding." -ForegroundColor Green
}

# ===================================================
# 1) 1D sweeps (기존)
# ===================================================
Write-Host "`n[STAGE 1] 1D sweeps" -ForegroundColor Green
foreach ($sex in $Sexes) {
  foreach ($mort in $MortCfg) {
    $m = $MortMap[$mort]; $mortArgs=@{}; for ($i=0;$i -lt $m.Count;$i+=2){ $mortArgs[$m[$i]]=$m[$i+1] }

    foreach ($a in $AnnAlphaGrid) {
      $tag = Make-Tag $mort 'DEV1D_ann' $sex
      $extra = @{ '--ann_alpha' = ('{0:F2}' -f $a) }
      if (SupportsFlag('ann_on')) { $extra['--ann_on'] = $(if ($a -gt 0){'on'} else {'off'}) }
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
    foreach ($hz in $HedgeGrid) {
      $tag = Make-Tag $mort 'DEV1D_hedge' $sex
      $extra = @{}
      if ($FxFlags.ratio){ $extra[$FxFlags.ratio] = ('{0:F2}' -f $hz) }
      if ($FxFlags.onoff){ $extra[$FxFlags.onoff] = $(if ($hz -gt 0){'on'} else {'off'}) }
      if ($extra.Count -gt 0){ Invoke-CLI -Tag ("${tag}_{0:F2}" -f $hz) -Sex $sex -MortArgs $mortArgs -Extra $extra | Out-Null }
    }
    foreach ($vpw in $VPWGrid) {
      $tag = Make-Tag $mort 'DEV1D_vpw' $sex
      $extra = @{}
      if (SupportsFlag('cstar_mode')) { $extra['--cstar_mode'] = 'vpw' }
      if (SupportsFlag('cstar_m'))    { $extra['--cstar_m']    = ('{0:F3}' -f $vpw) }
      if ($extra.Count -gt 0){ Invoke-CLI -Tag ("${tag}_{0:F3}" -f $vpw) -Sex $sex -MortArgs $mortArgs -Extra $extra | Out-Null }
    }
    foreach ($mix in $AssetMixes) {
      $tag = Make-Tag $mort ("DEV1D_mix_$mix") $sex
      $extra = Build-MixArgs $mix
      if ($extra.Count -gt 0){ Invoke-CLI -Tag $tag -Sex $sex -MortArgs $mortArgs -Extra $extra | Out-Null }
    }
  }
}

# ===================================================
# 2) 2D sweeps (기존)
# ===================================================
Write-Host "`n[STAGE 2] 2D sweeps" -ForegroundColor Green
foreach ($sex in $Sexes) {
  foreach ($mort in $MortCfg) {
    $m = $MortMap[$mort]; $mortArgs=@{}; for ($i=0;$i -lt $m.Count;$i+=2){ $mortArgs[$m[$i]]=$m[$i+1] }

    foreach ($a in $AnnAlphaGrid) {
      foreach ($w in $RiskShareGrid) {
        $tag = Make-Tag $mort 'DEV2D_ann_wrisk' $sex
        $extra = @{ '--w_max'=('{0:F2}' -f $w); '--ann_alpha'=('{0:F2}' -f $a) }
        if (SupportsFlag('ann_on')) { $extra['--ann_on'] = $(if ($a -gt 0){'on'} else {'off'}) }
        Invoke-CLI -Tag ("${tag}_a{0:F1}_w{1:F1}" -f $a,$w) -Sex $sex -MortArgs $mortArgs -Extra $extra | Out-Null
      }
    }
    foreach ($w in $RiskShareGrid) {
      foreach ($hz in $HedgeGrid) {
        $tag = Make-Tag $mort 'DEV2D_wrisk_hedge' $sex
        $extra = @{ '--w_max'=('{0:F2}' -f $w) }
        if ($FxFlags.ratio) {
          $extra[$FxFlags.ratio] = ('{0:F2}' -f $hz)
          if ($FxFlags.onoff) { $extra[$FxFlags.onoff] = $(if ($hz -gt 0){'on'} else {'off'}) }
          Invoke-CLI -Tag ("${tag}_w{0:F1}_h{1:F2}" -f $w,$hz) -Sex $sex -MortArgs $mortArgs -Extra $extra | Out-Null
        }
      }
    }
    foreach ($w in $RiskShareGrid) {
      foreach ($vpw in $VPWGrid) {
        $tag = Make-Tag $mort 'DEV2D_wrisk_c' $sex
        $extra = @{ '--w_max'=('{0:F2}' -f $w) }
        if (SupportsFlag('cstar_mode')) { $extra['--cstar_mode'] = 'vpw' }
        if (SupportsFlag('cstar_m'))    { $extra['--cstar_m']    = ('{0:F3}' -f $vpw) }
        Invoke-CLI -Tag ("${tag}_w{0:F1}_c{1:F3}" -f $w,$vpw) -Sex $sex -MortArgs $mortArgs -Extra $extra | Out-Null
      }
    }
  }
}

# ===================================================
# 3) Snapshot / Score / Tables (기존)
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
