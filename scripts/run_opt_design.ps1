<# =======================================================================  
 run_opt_design.ps1  (PS5-safe, 2025-11-18, refactor + light/heavy + optimal json)
 - RunProfile(light/heavy)에 따라 1D/2D grid 기본값 자동 분기
 - 사용자가 grid 파라미터를 넘기면 RunProfile과 무관하게 그대로 사용
 - 바닥 처리: f_min_real 제거, 고정 비율 바닥(--q_floor)로 일원화(QFloorDefault=0.02)
 - 연금 로딩: --phi_adval (기본 0.05) + --ann_index (real/nominal) 자동 주입
 - RL 워크로드 명시(Epochs/Steps/NPaths), Smoke 모드 축소
 - FX 헤지 플래그 자동 감지
 - MAIN_ANN: 공정가/현실가(실질/명목) 3세트 실행 + ann_alpha 주입
 - NEW: -Skip1D, -Skip2D 스위치로 단계별 실행 제어
 - NEW: 전역 태그 중복 가드(동일 tag 재실행·중복 로그 방지)
 - NEW: 스냅샷 dedup + 실패 태그 제외 후 재스코어
 - NEW: Stage 4에서 make_1d_and_opt_report.py / make_optimal_summary.py 호출
         → outputs/figs/optimal_points.json 생성 여부 자동 체크
======================================================================= #>

[CmdletBinding()]
param(
  # 실행 프로파일 / 데이터셋 / 시장 모드
  [ValidateSet('light','heavy')]      [string]$RunProfile   = 'light',
  [ValidateSet('dev','full')]         [string]$DataProfile  = 'dev',
  [ValidateSet('iid','bootstrap')]    [string]$MarketMode   = 'bootstrap',

  # 시드 / 편향 관련
  [string]$Seeds = '7',
  [ValidateSet('off','on')]           [string]$Bias = 'off',
  [double]$BiasProbGamma = 0.65,

  # RL 워크로드(일반)
  [int]$Epochs = 6,
  [int]$StepsPerEpoch = 512,
  [int]$NPaths = 1500,

  # Smoke(프리플라이트) 축소 워크로드
  [switch]$Smoke,
  [int]$SmokeEpochs = 1,
  [int]$SmokeSteps = 128,
  [int]$SmokePaths = 64,

  # 바닥(엔진 스케일 비율 바닥)
  [double]$QFloorDefault = 0.02,

  # CPI / Real-mode 정책
  [ValidateSet('on','off')]           [string]$UseRealRF = 'on',
  [switch]$CheckCpiSchema,
  [string]$BootstrapCsv = "",
  [int]$BootstrapBlock = 24,

  # 연금(annuity) 로딩/지수
  [double]$AnnuityPhi = 0.05,
  [ValidateSet('real','nominal')]     [string]$AnnuityIndex = 'real',

  # 1D/2D 그리드 (기본값은 RunProfile에 따라 아래에서 다시 설정)
  [double[]]$AnnAlphaGrid  = @(),
  [Alias('WriskGrid')]
  [double[]]$RiskShareGrid = @(),
  [double[]]$FeeGrid       = @(),
  [int[]]   $Age0Grid      = @(),
  [double[]]$HedgeGrid     = @(),
  [double[]]$VPWGrid       = @(),

  [ValidateSet('KR','US','GLD','EQUAL3')] [string[]]$AssetMixes = @('EQUAL3','KR','US','GLD'),
  [ValidateSet('M','F')]  [string[]]$Sexes   = @('M','F'),
  [ValidateSet('BASE','COHORT')] [string[]]$MortCfg = @('BASE'),

  # 선택 실행 스위치
  [switch]$Skip1D,
  [switch]$Skip2D,

  # MAIN_ANN(공정가 vs 현실가) 세트
  [switch]$MainANN,
  [string]$MainANNSeed = '7',
  [int]$MainANNEpochs = 1,
  [int]$MainANNStepsPerEpoch = 256,
  [int]$MainANNPaths = 1000,
  [double]$MainANNAlpha = 0.3,
  [double]$MainANNPhi = 0.03,
  [ValidateSet('BASE','cohort_2020')] [string]$MainANNMortFair = 'BASE',
  [ValidateSet('BASE','cohort_2020')] [string]$MainANNMortReal = 'cohort_2020'
)

# ---------- RunProfile별 기본 grid 설정 ----------
if (-not $PSBoundParameters.ContainsKey('AnnAlphaGrid') -or $AnnAlphaGrid.Count -eq 0) {
  if ($RunProfile -eq 'light') {
    $AnnAlphaGrid = @(0.0, 0.25, 0.50, 0.75, 1.0)
  } else {
    $AnnAlphaGrid = @(0..10 | ForEach-Object { $_/10.0 })
  }
}

if (-not $PSBoundParameters.ContainsKey('RiskShareGrid') -or $RiskShareGrid.Count -eq 0) {
  if ($RunProfile -eq 'light') {
    $RiskShareGrid = @(0.20, 0.40, 0.60, 0.80, 1.00)
  } else {
    $RiskShareGrid = @(0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0)
  }
}

if (-not $PSBoundParameters.ContainsKey('FeeGrid') -or $FeeGrid.Count -eq 0) {
  $FeeGrid = @(0.0000, 0.0040, 0.0080)
}

if (-not $PSBoundParameters.ContainsKey('Age0Grid') -or $Age0Grid.Count -eq 0) {
  if ($RunProfile -eq 'light') {
    $Age0Grid = @(55)
  } else {
    $Age0Grid = @(55)
  }
}

if (-not $PSBoundParameters.ContainsKey('HedgeGrid') -or $HedgeGrid.Count -eq 0) {
  if ($RunProfile -eq 'light') {
    $HedgeGrid = @(0.00, 0.50, 1.00)
  } else {
    $HedgeGrid = @(0.00, 0.25, 0.50, 0.75, 1.00)
  }
}

if (-not $PSBoundParameters.ContainsKey('VPWGrid') -or $VPWGrid.Count -eq 0) {
  if ($RunProfile -eq 'light') {
    $VPWGrid = @(0.03, 0.05, 0.06)
  } else {
    $VPWGrid = @(0.03,0.035,0.04,0.045,0.05,0.055,0.06)
  }
}

# ---------- Paths ----------
$Root = (Get-Location).Path
$Out  = Join-Path $Root 'outputs'
$Logs = Join-Path $Out  '_logs'
$Figs = Join-Path $Out  'figs'
$null = New-Item -ItemType Directory -Force -Path $Out,$Logs,$Figs | Out-Null

Write-Host "[Guard] Close Excel to avoid PermissionError on CSV/XLSX." -ForegroundColor Yellow

# ---------- Utils ----------
function ConvertTo-Double {
  param($x)
  if ($null -eq $x -or "$x" -eq '') { return $null }
  try { [double]$x } catch { $null }
}

function Sort-ByScoreDesc {
  param($rows)
  $rows | Sort-Object @{
    Expression = {
      $v = $_.CompositeScore_benefit
      if ($null -eq $v -or "$v" -eq '') {
        ConvertTo-Double $_.CompositeScore
      } else {
        ConvertTo-Double $v
      }
    }
    Descending = $true
  }
}

function Make-Tag([string]$MortId,[string]$BaseTag,[string]$Sex){ "${BaseTag}_${MortId}_${Sex}" }
function Get-SeedArgs([string]$SeedsCsv){
  $parts = $SeedsCsv -split '[,\s]+' | Where-Object { $_ -ne '' }
  if($parts.Count -eq 0){ return @('--seeds','7') }
  @('--seeds') + $parts
}

# CPI CSV 스키마/블록 길이 점검(옵션)
function Test-CpiSchema {
  param([string]$CsvPath,[int]$Block)
  if(-not $CsvPath -or -not (Test-Path $CsvPath)){
    Write-Host "[CPI] Skip schema check." -ForegroundColor Yellow
    return
  }
  $first = Get-Content -Path $CsvPath -TotalCount 1
  if(-not $first){ Write-Warning "[CPI] Empty CSV: $CsvPath"; return }
  $hdr = $first.Split(',') | ForEach-Object { $_.Trim() }
  $need = @('date','risky_nom','tbill_nom','cpi')
  $miss = @()
  foreach($h in $need){ if($hdr -notcontains $h){ $miss += $h } }
  if($miss.Count -gt 0){
    Write-Warning ("[CPI] Missing headers: {0}" -f ($miss -join ', '))
  } else {
    Write-Host "[CPI] Header OK." -ForegroundColor Green
  }
  if($Block -ne 24){
    Write-Host "[CPI] bootstrap_block=$Block (policy recommends 24)." -ForegroundColor Yellow
  } else {
    Write-Host "[CPI] bootstrap_block=24 verified." -ForegroundColor Green
  }
}

# CLI 도움말(플래그 감지) – 일부 플래그에만 사용
$global:CLI_HELP = (& .\.venv\Scripts\python.exe -m project.runner.cli -h 2>&1 | Out-String)
function SupportsFlag([string]$flag){
  return $global:CLI_HELP -match ("--" + [regex]::Escape($flag) + "(\s|$|=)")
}

# Asset mix → args
function Build-MixArgs([string]$mix){
  $a = @{}
  $has_kr = SupportsFlag('alpha_kr'); $has_us = SupportsFlag('alpha_us'); $has_au = SupportsFlag('alpha_au')
  $has_mix = SupportsFlag('alpha_mix')
  if($has_kr -and $has_us -and $has_au){
    switch($mix){
      'EQUAL3' { $a['--alpha_kr']='0.3333'; $a['--alpha_us']='0.3333'; $a['--alpha_au']='0.3333' }
      'KR'     { $a['--alpha_kr']='1.0';    $a['--alpha_us']='0.0';    $a['--alpha_au']='0.0' }
      'US'     { $a['--alpha_kr']='0.0';    $a['--alpha_us']='1.0';    $a['--alpha_au']='0.0' }
      'GLD'    { $a['--alpha_kr']='0.0';    $a['--alpha_us']='0.0';    $a['--alpha_au']='1.0' }
    }
  } elseif($has_mix){
    switch($mix){
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

# FX hedge flags 자동 감지
function Get-FxFlags {
  $ratio=$null; $onoff=$null; $cost=$null
  if(SupportsFlag('h_fx')){ $ratio='--h_fx' }
  elseif(SupportsFlag('h_FX')){ $ratio='--h_FX' }
  elseif(SupportsFlag('fx_hedge_ratio')){ $ratio='--fx_hedge_ratio' }
  if(SupportsFlag('hedge')){ $onoff='--hedge' }
  elseif(SupportsFlag('fx_hedge_on')){ $onoff='--fx_hedge_on' }
  if(SupportsFlag('fx_hedge_cost')){ $cost='--fx_hedge_cost' }
  elseif(SupportsFlag('hedge_cost')){ $cost='--hedge_cost' }
  return @{ ratio=$ratio; onoff=$onoff; cost=$cost }
}
$FxFlags = Get-FxFlags

# ---------- 전역 태그 중복 가드 ----------
$global:RanTags = [System.Collections.Generic.HashSet[string]]::new()

# ---------- Base args builder ----------
function New-BaseArgs {
  param(
    [string]$Sex,
    [hashtable]$MortArgs = @{},
    [hashtable]$Extra    = @{}
  )
  $argList = @(
    '-m','project.runner.cli',
    '--mode','rl','--method','rl',
    '--data_profile', $DataProfile,
    '--market_mode',  $MarketMode,
    '--sex', $Sex,
    '--print_mode','metrics',
    '--metrics_keys','EW,ES95,Ruin,mean_WT,es_mode,use_real_rf'
  )

  if(SupportsFlag('q_floor'))       { $argList += @('--q_floor', ('{0:F3}' -f $QFloorDefault)) }
  if(SupportsFlag('use_real_rf'))   { $argList += @('--use_real_rf', $UseRealRF) }
  if(SupportsFlag('bias_on'))       { $argList += @('--bias_on', $(if($Bias -eq 'on'){'on'} else {'off'})) }
  if(SupportsFlag('bias_prob_gamma')) { $argList += @('--bias_prob_gamma', ('{0:F3}' -f $BiasProbGamma)) }

  if($Smoke){
    if(SupportsFlag('rl_epochs'))          { $argList += @('--rl_epochs', "$SmokeEpochs") }
    if(SupportsFlag('rl_steps_per_epoch')) { $argList += @('--rl_steps_per_epoch', "$SmokeSteps") }
    if(SupportsFlag('n_paths'))            { $argList += @('--n_paths', "$SmokePaths") }
    if(SupportsFlag('quiet'))              { $argList += @('--quiet','on') }
  } else {
    if(SupportsFlag('rl_epochs'))          { $argList += @('--rl_epochs', "$Epochs") }
    if(SupportsFlag('rl_steps_per_epoch')) { $argList += @('--rl_steps_per_epoch', "$StepsPerEpoch") }
    if(SupportsFlag('n_paths'))            { $argList += @('--n_paths', "$NPaths") }
  }

  if(SupportsFlag('phi_adval'))   { $argList += @('--phi_adval', ('{0:F2}' -f $AnnuityPhi)) }
  if(SupportsFlag('ann_index'))   { $argList += @('--ann_index', $AnnuityIndex) }

  $argList += (Get-SeedArgs -SeedsCsv $Seeds)

  foreach($k in $MortArgs.Keys){ $argList += @($k, $MortArgs[$k]) }
  foreach($k in $Extra.Keys){    $argList += @($k, $Extra[$k]) }
  return ,$argList
}

# ---------- Core runner ----------
function Invoke-CLI {
  param(
    [string]$Tag,[string]$Sex,[hashtable]$MortArgs=@{},[hashtable]$Extra=@{}
  )

  if ($Tag -and -not $global:RanTags.Add($Tag)) {
    Write-Host "[SKIP] duplicated tag: $Tag" -ForegroundColor Yellow
    return 0
  }

  $cliArgs = New-BaseArgs -Sex $Sex -MortArgs $MortArgs -Extra $Extra
  if($Tag){ $cliArgs += @('--tag', $Tag) }
  $log = Join-Path $Logs ("{0}.log" -f ($Tag -replace '[^\w\-\.]','_'))
  Write-Host "[RUN] $Tag  -> $log" -ForegroundColor Cyan
  & ".\.venv\Scripts\python.exe" @cliArgs *> $log
  $ec = $LASTEXITCODE
  if($ec -ne 0){ Add-Content -Path $log -Value "[ERR] exit=$ec  tag=$Tag" -Encoding utf8 }
  return $ec
}

# ---------- Helpers ----------
function Export-CleanScored {
  $scr = Join-Path $Out 'DEV_scored.csv'
  $clean = Join-Path $Out 'DEV_scored_clean.csv'
  if(Test-Path $scr){
    Import-Csv $scr | Where-Object { $_.tag -and $_.tag.Trim() -ne '' } |
      Export-Csv $clean -NoTypeInformation -Encoding UTF8
    Write-Host "[OK] cleaned => $clean"
  } else {
    Write-Host "[Info] DEV_scored.csv not found, skip cleaning."
  }
}

# ---------- Banners ----------
Write-Host "[Info] RunProfile=$RunProfile  Data=$DataProfile  Market=$MarketMode  Seeds=$Seeds"
Write-Host "[Info] Bias=$Bias  BiasProbGamma=$BiasProbGamma  Smoke=$($Smoke.IsPresent)"
Write-Host "[Info] use_real_rf=$UseRealRF  q_floor=$QFloorDefault"
Write-Host "[Info] Annuity: phi_adval=$AnnuityPhi  ann_index=$AnnuityIndex"
Write-Host "[Info] Grids: AnnAlpha=[$($AnnAlphaGrid -join ',')], wrisk=[$($RiskShareGrid -join ',')], hedge=[$($HedgeGrid -join ',')], VPW=[$($VPWGrid -join ',')]"
if($CheckCpiSchema){ Test-CpiSchema -CsvPath $BootstrapCsv -Block $BootstrapBlock }

# ===================================================
# MAIN_ANN: 공정가 vs 현실가(실질/명목) 3세트
# ===================================================
if($MainANN){
  Write-Host "`n[MAIN_ANN] Annuity 현실화 비교" -ForegroundColor Green
  $Py = ".\.venv\Scripts\python.exe"

  $COMMON = @('--method','rl','--mode','rl',
              '--rl_epochs',"$MainANNEpochs",'--rl_steps_per_epoch',"$MainANNStepsPerEpoch",
              '--n_paths',"$MainANNPaths",'--seed',"$MainANNSeed",
              '--eval_seed_jitter','off','--print_mode','summary',
              '--data_profile','full','--market_mode',$MarketMode)

  $alphaStr = ('{0:F2}' -f $MainANNAlpha)
  if (SupportsFlag('ann_alpha')) { $COMMON += @('--ann_alpha', $alphaStr) }
  if (SupportsFlag('ann_on'))    { $COMMON += @('--ann_on','on') }

  if(SupportsFlag('bias_on')){ $COMMON += @('--bias_on','off') }

  $FAIR = @('--tag','MAIN_ANN_FAIR',
            '--mortality','on','--mort_table',$MainANNMortFair,
            '--phi_adval','0.0','--ann_index','real')
  if(SupportsFlag('use_real_rf')){ $FAIR += @('--use_real_rf','on') }
  & $Py @('-m','project.runner.cli') @FAIR @COMMON *> (Join-Path $Logs 'MAIN_ANN_FAIR.log')

  $phi = ('{0:F2}' -f $MainANNPhi)
  $REAL = @('--tag','MAIN_ANN_REAL',
            '--mortality','on','--mort_table',$MainANNMortReal,
            '--phi_adval',$phi,'--ann_index','real')
  if(SupportsFlag('use_real_rf')){ $REAL += @('--use_real_rf','on') }
  & $Py @('-m','project.runner.cli') @REAL @COMMON *> (Join-Path $Logs 'MAIN_ANN_REAL.log')

  $REALN = @('--tag','MAIN_ANN_REAL_NOM',
             '--mortality','on','--mort_table',$MainANNMortReal,
             '--phi_adval',$phi,'--ann_index','nominal')
  if(SupportsFlag('use_real_rf')){ $REALN += @('--use_real_rf','off') }
  & $Py @('-m','project.runner.cli') @REALN @COMMON *> (Join-Path $Logs 'MAIN_ANN_REAL_NOM.log')

  $metricsCsv = Join-Path $Logs 'metrics.csv'
  $snap = Join-Path $Out 'MAIN_ANN_snapshot.csv'
  if(Test-Path $metricsCsv){
    (Import-Csv $metricsCsv |
      Where-Object { $_.tag -in 'MAIN_ANN_FAIR','MAIN_ANN_REAL','MAIN_ANN_REAL_NOM' } |
      Group-Object tag,method,seed | ForEach-Object { $_.Group | Select-Object -Last 1 }) |
      Export-Csv $snap -NoTypeInformation -Encoding UTF8
    Write-Host "[OK] snapshot => $snap"
  } else {
    Write-Warning "[MAIN_ANN] metrics.csv not found."
  }

  & $Py .\scripts\score_snapshot.py `
    --src $snap --tag_startswith MAIN_ANN_ `
    --metrics "EW,ES95" --weights "0.6,0.4" --es_mode "wealth" --out "inplace"

  $readme = @"
# MAIN | Annuity 현실화 비교
- 환경: data_profile=full, market_mode=$MarketMode, seed=$MainANNSeed, n_paths=$MainANNPaths, RL(epochs=$MainANNEpochs, steps=$MainANNStepsPerEpoch), bias_off
- 공통 annuity 비중: ann_alpha = $alphaStr
- 케이스:
  - MAIN_ANN_FAIR     = 공정가(무로딩, $MainANNMortFair, 실질)
  - MAIN_ANN_REAL     = 로딩 $phi + $MainANNMortReal (실질)
  - MAIN_ANN_REAL_NOM = 로딩 $phi + $MainANNMortReal (명목연금, 명목RF)
- 핵심지표: EW, ES95, RuinPct, mean_WT
"@
  $readme | Set-Content (Join-Path $Out 'MAIN_ANN_README.md') -Encoding UTF8
  Write-Host "[DONE] MAIN_ANN complete" -ForegroundColor Green
  exit 0
}

# ===================================================
# 0) Smoke preflight (옵션)
# ===================================================
if($Smoke){
  Write-Host "`n[STAGE 0] Smoke preflight" -ForegroundColor Green
  $fail = 0
  foreach($sex in @('M')){
    $m = $MortMap['BASE']; $mortArgs=@{}; for($i=0;$i -lt $m.Count;$i+=2){ $mortArgs[$m[$i]]=$m[$i+1] }

    $extra1 = @{'--ann_alpha'='0.10'}; if(SupportsFlag('ann_on')){ $extra1['--ann_on']='on' }
    $fail += (Invoke-CLI -Tag 'SMOKE_ann_0.1' -Sex $sex -MortArgs $mortArgs -Extra $extra1)

    $fail += (Invoke-CLI -Tag 'SMOKE_wrisk_0.3' -Sex $sex -MortArgs $mortArgs -Extra @{ '--w_max'='0.30' })

    $extra2=@{}; if($FxFlags.ratio){ $extra2[$FxFlags.ratio]='0.50' }; if($FxFlags.onoff){ $extra2[$FxFlags.onoff]='on' }
    if($extra2.Count -gt 0){ $fail += (Invoke-CLI -Tag 'SMOKE_hedge_0.5' -Sex $sex -MortArgs $mortArgs -Extra $extra2) }
    else { Write-Host "[SMOKE] Skip hedge check: FX flags not found." -ForegroundColor Yellow }

    $mixArgs = Build-MixArgs 'EQUAL3'
    if($mixArgs.Count -gt 0){ $fail += (Invoke-CLI -Tag 'SMOKE_mix_equal3' -Sex $sex -MortArgs $mortArgs -Extra $mixArgs) }
  }
  if($fail -ne 0){ Write-Host "[SMOKE] Preflight failed. See logs." -ForegroundColor Yellow; exit 1 }
  Write-Host "[SMOKE] OK — flags good." -ForegroundColor Green
}

# ===================================================
# 1) 1D sweeps
# ===================================================
if(-not $Skip1D){
  Write-Host "`n[STAGE 1] 1D sweeps (ann / wrisk / VPW + 기타)" -ForegroundColor Green
  foreach($sex in $Sexes){
    foreach($mort in $MortCfg){
      $m = $MortMap[$mort]; $mortArgs=@{}; for($i=0;$i -lt $m.Count;$i+=2){ $mortArgs[$m[$i]]=$m[$i+1] }

      # 1D ann_alpha
      foreach($a in $AnnAlphaGrid){
        $tag = Make-Tag $mort 'DEV1D_ann' $sex
        $extra = @{ '--ann_alpha' = ('{0:F2}' -f $a) }
        if(SupportsFlag('ann_on')){ $extra['--ann_on'] = $(if($a -gt 0){'on'} else {'off'}) }
        Invoke-CLI -Tag ("${tag}_{0:F1}" -f $a) -Sex $sex -MortArgs $mortArgs -Extra $extra | Out-Null
      }

      # 1D wrisk (w_max)
      foreach($w in $RiskShareGrid){
        $tag = Make-Tag $mort 'DEV1D_wrisk' $sex
        $extra = @{ '--w_max' = ('{0:F2}' -f $w) }
        Invoke-CLI -Tag ("${tag}_{0:F3}" -f $w) -Sex $sex -MortArgs $mortArgs -Extra $extra | Out-Null
      }

      # 1D fee
      foreach($fee in $FeeGrid){
        $tag = Make-Tag $mort 'DEV1D_fee' $sex
        $extra = @{ '--fee_annual' = ('{0:F4}' -f $fee) }
        Invoke-CLI -Tag ("${tag}_{0:F4}" -f $fee) -Sex $sex -MortArgs $mortArgs -Extra $extra | Out-Null
      }

      # 1D age0
      foreach($age0 in $Age0Grid){
        $tag = Make-Tag $mort 'DEV1D_age' $sex
        Invoke-CLI -Tag ("${tag}_${age0}") -Sex $sex -MortArgs $mortArgs -Extra @{ '--age0' = "$age0" } | Out-Null
      }

      # 1D hedge
      foreach($hz in $HedgeGrid){
        $tag = Make-Tag $mort 'DEV1D_hedge' $sex
        $extra=@{}
        if($FxFlags.ratio){ $extra[$FxFlags.ratio]=('{0:F2}' -f $hz) }
        if($FxFlags.onoff){ $extra[$FxFlags.onoff]=$(if($hz -gt 0){'on'} else {'off'}) }
        if($extra.Count -gt 0){
          Invoke-CLI -Tag ("${tag}_{0:F2}" -f $hz) -Sex $sex -MortArgs $mortArgs -Extra $extra | Out-Null
        }
      }

      # 1D VPW (cstar_mode=vpw, cstar_m)
      foreach($vpw in $VPWGrid){
        $tag = Make-Tag $mort 'DEV1D_vpw' $sex
        $extra = @{
          '--cstar_mode' = 'vpw'
          '--cstar_m'    = ('{0:F3}' -f $vpw)
        }
        Invoke-CLI -Tag ("${tag}_{0:F3}" -f $vpw) -Sex $sex -MortArgs $mortArgs -Extra $extra | Out-Null
      }

      # 1D asset mix
      foreach($mix in $AssetMixes){
        $tag = Make-Tag $mort ("DEV1D_mix_$mix") $sex
        $extra = Build-MixArgs $mix
        if($extra.Count -gt 0){
          Invoke-CLI -Tag $tag -Sex $sex -MortArgs $mortArgs -Extra $extra | Out-Null
        }
      }
    }
  }
} else {
  Write-Host "[STAGE 1] skipped by -Skip1D" -ForegroundColor Yellow
}

# ===================================================
# 2) 2D sweeps
# ===================================================
if(-not $Skip2D){
  Write-Host "`n[STAGE 2] 2D sweeps" -ForegroundColor Green
  foreach($sex in $Sexes){
    foreach($mort in $MortCfg){
      $m = $MortMap[$mort]; $mortArgs=@{}; for($i=0;$i -lt $m.Count;$i+=2){ $mortArgs[$m[$i]]=$m[$i+1] }

      # 2D (ann_alpha, wrisk)
      foreach($a in $AnnAlphaGrid){
        foreach($w in $RiskShareGrid){
          $tag = Make-Tag $mort 'DEV2D_ann_wrisk' $sex
          $extra = @{
            '--w_max'     = ('{0:F2}' -f $w)
            '--ann_alpha' = ('{0:F2}' -f $a)
          }
          if(SupportsFlag('ann_on')){ $extra['--ann_on']=$(if($a -gt 0){'on'} else {'off'}) }
          Invoke-CLI -Tag ("${tag}_a{0:F3}_w{1:F3}" -f $a,$w) -Sex $sex -MortArgs $mortArgs -Extra $extra | Out-Null
        }
      }

      # 2D (wrisk, hedge)
      foreach($w in $RiskShareGrid){
        foreach($hz in $HedgeGrid){
          $tag = Make-Tag $mort 'DEV2D_wrisk_hedge' $sex
          $extra = @{ '--w_max'=('{0:F2}' -f $w) }
          if($FxFlags.ratio){
            $extra[$FxFlags.ratio]=('{0:F2}' -f $hz)
            if($FxFlags.onoff){ $extra[$FxFlags.onoff]=$(if($hz -gt 0){'on'} else {'off'}) }
            Invoke-CLI -Tag ("${tag}_w{0:F3}_h{1:F3}" -f $w,$hz) -Sex $sex -MortArgs $mortArgs -Extra $extra | Out-Null
          }
        }
      }

      # 2D (wrisk, VPW)
      foreach($w in $RiskShareGrid){
        foreach($vpw in $VPWGrid){
          $tag = Make-Tag $mort 'DEV2D_wrisk_c' $sex
          $extra=@{
            '--w_max'     = ('{0:F2}' -f $w)
            '--cstar_mode'= 'vpw'
            '--cstar_m'   = ('{0:F3}' -f $vpw)
          }
          Invoke-CLI -Tag ("${tag}_w{0:F3}_c{1:F3}" -f $w,$vpw) -Sex $sex -MortArgs $mortArgs -Extra $extra | Out-Null
        }
      }
    }
  }
} else {
  Write-Host "[STAGE 2] skipped by -Skip2D" -ForegroundColor Yellow
}

# ===================================================
# 3) Snapshot / Score / Tables
# ===================================================
Write-Host "`n[STAGE 3] Snapshot / Score / Tables" -ForegroundColor Green
$metricsCsv = Join-Path $Logs 'metrics.csv'
$SnapRaw    = Join-Path $Out 'DEV_metrics_snapshot.csv'
$SnapDedup  = Join-Path $Out 'DEV_metrics_snapshot_dedup.csv'
$SnapClean  = Join-Path $Out 'DEV_metrics_snapshot_clean.csv'

if(Test-Path $metricsCsv){
  (Import-Csv $metricsCsv |
    Group-Object tag,method,sex | ForEach-Object { $_.Group | Select-Object -Last 1 }) |
    Export-Csv $SnapRaw -NoTypeInformation -Encoding UTF8
  Write-Host "[OK] snapshot => $SnapRaw"

  (Import-Csv $SnapRaw | Group-Object tag | ForEach-Object { $_.Group | Select-Object -Last 1 }) |
    Export-Csv $SnapDedup -NoTypeInformation -Encoding UTF8
  Write-Host "[OK] dedup => $SnapDedup"

  $failTags = Get-ChildItem $Logs\*.log |
    Where-Object { Select-String -Path $_.FullName -Pattern '[[]ERR[]] exit=' -Quiet } |
    ForEach-Object {
      (Get-Content $_.FullName | Select-String -Pattern 'tag=' | Select-Object -Last 1).Line -replace '.*tag=',''
    } | Sort-Object -Unique

  (Import-Csv $SnapDedup | Where-Object { ($_.tag -like 'DEV*') -and ($failTags -notcontains $_.tag) }) |
    Export-Csv $SnapClean -NoTypeInformation -Encoding UTF8
  Write-Host "[OK] clean snapshot => $SnapClean"
} else {
  Write-Warning "metrics.csv not found: $metricsCsv"
}

$ScoreSrc = if (Test-Path $SnapClean) { $SnapClean } elseif (Test-Path $SnapRaw) { $SnapRaw } else { $null }
if ($null -ne $ScoreSrc) {
  & .\.venv\Scripts\python.exe .\scripts\score_snapshot.py `
    --src $ScoreSrc `
    --metrics "EW,ES95" `
    --weights "0.6,0.4" `
    --es_mode "wealth" `
    --out (Join-Path $Out 'DEV_scored.csv')
  Export-CleanScored
} else {
  Write-Warning "[Score] Source snapshot not found. Skip scoring."
}

$OPT_Sum = Join-Path $Out 'OPT_summary_benefit.csv'
$BH_Sum  = Join-Path $Out 'BH_summary_benefit.csv'
$OPT_Core= Join-Path $Out 'OPT_table_core.csv'
$BH_Core = Join-Path $Out 'BH_table_core.csv'
$Compare = Join-Path $Out 'OPT_BH_compare.csv'
$TopBoth = Join-Path $Out 'OPT_BH_top.csv'
$BestBy  = Join-Path $Out 'DEV_OPT_best_by_sex_mort.csv'

if(Test-Path $OPT_Sum){
  Import-Csv $OPT_Sum | Select-Object tag,EW,ES95,Ruin,CompositeScore_benefit |
    Sort-Object @{Expression={ ConvertTo-Double $_.CompositeScore_benefit }; Descending=$true} |
    Export-Csv $OPT_Core -NoTypeInformation -Encoding UTF8
  Write-Host "[OK] $OPT_Core"
}
if(Test-Path $BH_Sum){
  Import-Csv $BH_Sum | Select-Object tag,EW,ES95,Ruin,CompositeScore_benefit |
    Sort-Object @{Expression={ ConvertTo-Double $_.CompositeScore_benefit }; Descending=$true} |
    Export-Csv $BH_Core -NoTypeInformation -Encoding UTF8
  Write-Host "[OK] $BH_Core"
}
if((Test-Path $OPT_Sum) -and (Test-Path $BH_Sum)){
  $cols=@('tag','EW','ES95','Ruin','CompositeScore_benefit')
  $optProps=$cols+@(@{Name='Room';Expression={'OPT'}})
  $bhProps =$cols+@(@{Name='Room';Expression={'BH'}})
  $opt=Import-Csv $OPT_Sum|Select-Object -Property $optProps
  $bh =Import-Csv $BH_Sum |Select-Object -Property $bhProps
  $all=$opt+$bh
  $all | Sort-Object Room, @{Expression={ ConvertTo-Double $_.CompositeScore_benefit }; Descending=$true} |
    Export-Csv $Compare -NoTypeInformation -Encoding UTF8
  Write-Host "[OK] compare => $Compare"
  $optTop3 = Sort-ByScoreDesc $opt | Select-Object -First 3
  $bhTop3  = Sort-ByScoreDesc $bh  | Select-Object -First 3
  ($optTop3+$bhTop3) | Export-Csv $TopBoth -NoTypeInformation -Encoding UTF8
  Write-Host "[OK] top3+top3 => $TopBoth"
}
if(Test-Path $OPT_Sum){
  $sum = Import-Csv $OPT_Sum
  $hasSex  = ($sum|Get-Member -Name sex     -MemberType NoteProperty) -ne $null
  $hasMort = ($sum|Get-Member -Name mort_id -MemberType NoteProperty) -ne $null
  if(-not $hasSex -or -not $hasMort){
    Sort-ByScoreDesc $sum | Select-Object -First 1 | Export-Csv $BestBy -NoTypeInformation -Encoding UTF8
    Write-Host "[WARN] sex/mort_id missing -> saved global top1: $BestBy"
  } else {
    ($sum | Group-Object sex, mort_id | ForEach-Object { Sort-ByScoreDesc $_.Group | Select-Object -First 1 }) |
      Export-Csv $BestBy -NoTypeInformation -Encoding UTF8
    Write-Host "[OK] best by sex x mort => $BestBy"
  }
}

# ===================================================
# 4) 1D Report / Optimal Summary / optimal_points.json
# ===================================================
Write-Host "`n[STAGE 4] 1D report / optimal summary" -ForegroundColor Green
$PyExe = ".\.venv\Scripts\python.exe"
$ScoredClean = Join-Path $Out 'DEV_scored_clean.csv'
$ScoredRaw   = Join-Path $Out 'DEV_scored.csv'

$SnapshotFor1D = $null
if (Test-Path $SnapClean) { $SnapshotFor1D = $SnapClean }
elseif (Test-Path $SnapDedup) { $SnapshotFor1D = $SnapDedup }
elseif (Test-Path $SnapRaw) { $SnapshotFor1D = $SnapRaw }

$ScoredFor1D = $null
if (Test-Path $ScoredClean) { $ScoredFor1D = $ScoredClean }
elseif (Test-Path $ScoredRaw) { $ScoredFor1D = $ScoredRaw }

if (Test-Path ".\scripts\make_1d_and_opt_report.py") {
  if (($null -ne $SnapshotFor1D) -and ($null -ne $ScoredFor1D)) {
    Write-Host "[STAGE 4] run make_1d_and_opt_report.py" -ForegroundColor Cyan
    & $PyExe .\scripts\make_1d_and_opt_report.py `
      --snapshot $SnapshotFor1D `
      --scored $ScoredFor1D `
      --outdir $Out
    if ($LASTEXITCODE -ne 0) {
      Write-Warning "[STAGE 4] make_1d_and_opt_report.py exit=$LASTEXITCODE (CLI 인자 확인 필요)"
    }
  } else {
    Write-Warning "[STAGE 4] snapshot/scored source not found, skip make_1d_and_opt_report.py."
  }
} else {
  Write-Host "[STAGE 4] make_1d_and_opt_report.py not found, skip." -ForegroundColor Yellow
}

if (Test-Path ".\scripts\make_optimal_summary.py") {
  Write-Host "[STAGE 4] run make_optimal_summary.py" -ForegroundColor Cyan
  & $PyExe .\scripts\make_optimal_summary.py
  if ($LASTEXITCODE -ne 0) {
    Write-Warning "[STAGE 4] make_optimal_summary.py exit=$LASTEXITCODE (CLI 인자 확인 필요)"
  }
} else { 
  Write-Host "[STAGE 4] make_optimal_summary.py not found, skip." -ForegroundColor Yellow
}

$PointsJson = Join-Path $Figs 'optimal_points.json'
if(Test-Path $PointsJson){
  Write-Host "[OK] optimal_points.json -> $PointsJson" -ForegroundColor Green
} else {
  Write-Host "[Info] optimal_points.json not found (요약 스크립트 CLI를 확인하세요)." -ForegroundColor Yellow
}

Write-Host "[DONE] design/score/summary complete" -ForegroundColor Green
