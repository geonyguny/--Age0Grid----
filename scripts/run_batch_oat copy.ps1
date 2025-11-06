param(
  # ───────── 실험 축(하위호환 포함) ─────────
  [ValidateSet('hedge_sigma_k','mix_us','loss_aversion','bias_loss_aversion')]
  [string]$Var,

  # 콤마/공백 구분 값들: "0,0.5,1.0" / "0 0.5 1.0"
  [Parameter(Mandatory=$true)]
  [string]$Values,

  # 방법론
  [ValidateSet('rl','hjb','both')]
  [string]$Method = 'both',

  # 실행 프로파일(dev=빠른점검 / overnight=대용량)
  [ValidateSet('dev','overnight')]
  [string]$Mode = 'dev',

  # CLI 실행 모드 강제: RL 캐시 회피 기본 rl 권장
  [ValidateSet('auto','rl','once','calib')]
  [string]$CliMode = 'rl',

  # 현재 엔진 미지원 → 경고 후 무시
  [switch]$Overwrite,

  # 사용자 추가 인자 (예: "--rl_epochs 8 --rl_steps_per_epoch 2000")
  [string]$Extra,

  # DryRun: 파이썬 실행 대신 커맨드만 출력
  [switch]$DryRun,

  # Seeds: 문자열/쉼표/공백/배열 모두 허용 (예: "11,12" / "11 12")
  [string]$Seeds = "11",

  # 정책 고정(평가/리포트) 모드
  [switch]$PolicyLocked,

  # ───────── 파이프라인 프리셋(ABCD) 전용 ─────────
  # preset=abcd → A) σ-헤지 → B) κ 재학습(bias, 학습강화) → C) Policy-Locked → D) 스냅샷·점수화·퀵리포트
  [ValidateSet('none','abcd')]
  [string]$Preset = 'none',

  # (A)에서 사용할 값
  [string]$HedgeValues = "0,0.5,1.0",
  [string]$HedgeSeeds  = "11,12",

  # (B)에서 사용할 값과 추가 인자(학습강화)
  [string]$BiasValues  = "1.0 1.5 2.0",
  [string]$ExtraBias   = "--rl_epochs 10 --rl_steps_per_epoch 3000 --entropy_coef 0.0 --teacher_eps0 0.10 --teacher_decay 0.98",

  # (C)에서 사용할 값(Policy-Locked)
  [string]$LockedValues = "1.0 2.0",

  # (D) 점수화 가중치/ES 모드
  [string]$ScoreWeights = "0.6,0.4",
  [ValidateSet('wealth','loss')]
  [string]$EsMode = 'wealth'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName 'System.Globalization' | Out-Null
$inv = [System.Globalization.CultureInfo]::InvariantCulture

# ── 경로/환경 ────────────────────────────────────────────
$ScriptDir   = $PSScriptRoot
$ProjectRoot = Split-Path -Parent $ScriptDir
Set-Location $ProjectRoot

$Py      = '.\.venv\Scripts\python.exe'
$OutRoot = '.\outputs'
$LogDir  = Join-Path $OutRoot '_logs'
New-Item -ItemType Directory -Force -Path $OutRoot,$LogDir | Out-Null

# ── 유틸 ────────────────────────────────────────────────
function ToInv([double]$x) { return $x.ToString($inv) }

function Split-List([string]$x) {
  if ([string]::IsNullOrWhiteSpace($x)) { return @() }
  return @($x -split '[,\s]+' | Where-Object { $_ -ne '' })
}

function Parse-Doubles([string]$csv) {
  $vals = New-Object System.Collections.Generic.List[double]
  foreach ($t in (Split-List $csv)) {
    try { [void]$vals.Add([double]::Parse($t, $inv)) }
    catch { throw "Values 파싱 실패: '$t' (콤마/공백 구분, 소수점은 . 사용)" }
  }
  if ($vals.Count -eq 0) { throw "Values가 비어 있습니다." }
  return @($vals.ToArray())
}

function Parse-Seeds([string]$seedsText) {
  $out = New-Object System.Collections.Generic.List[int]
  foreach ($t in (Split-List $seedsText)) {
    try { [void]$out.Add([int]::Parse($t, $inv)) }
    catch { throw "Seeds 파싱 실패: '$t' (예: 11,12 또는 '11 12')" }
  }
  if ($out.Count -eq 0) { $out.Add(11) }
  return @($out.ToArray())
}

# ── 모드별 파라미터 ─────────────────────────────────────
switch ($Mode) {
  'dev' {
    $profile    = 'dev'
    $nPathsRL   = 2000
    $nPathsHJB  = 2000
    $TagPrefix  = 'DEV'
  }
  'overnight' {
    $profile    = 'full'
    $nPathsRL   = 8000
    $nPathsHJB  = 30000
    $TagPrefix  = 'OVN'
  }
}

# ── 공통 출력 ───────────────────────────────────────────
function Info-Header([string]$v,[string]$vals,[string]$m,[string]$mode) {
  Write-Host "[BATCH OAT] Var=$v  Values=$vals  Method=$m  Mode=$mode" -ForegroundColor Cyan
  Write-Host "[PROFILE] $profile  [SEEDS] $([string]::Join(',', $SeedList))  [RL n_paths] $nPathsRL  [HJB n_paths] $nPathsHJB" -ForegroundColor DarkCyan
}

if ($Overwrite) { Write-Warning "현재 엔진은 --overwrite 인자를 지원하지 않습니다. (무시)" }

function Get-NPaths([string]$mth) { if ($mth -eq 'rl') { return $nPathsRL } else { return $nPathsHJB } }

function RunCmd([string]$exe, [string[]]$argv) {
  if ($DryRun) {
    Write-Host ($exe + ' ' + ($argv -join ' ')) -ForegroundColor DarkGray
    return 0
  }
  & $exe @argv
  return $LASTEXITCODE
}

function RunPy([string]$title, [string[]]$argv) {
  Write-Host ">> $title" -ForegroundColor Cyan
  $rc = RunCmd $Py $argv
  if ($rc -ne 0) { throw "FAILED: $title (exit=$rc)" }
}

function Build-ArgsFor([string]$mth, [double]$val, [int]$seed) {
  $args = @(
    '-m','project.runner.cli',
    '--method', $mth,
    '--data_profile', $profile,
    '--market_mode','bootstrap',
    '--n_paths', (Get-NPaths $mth).ToString(),
    '--seed', $seed.ToString(),
    '--print_mode','summary',
    '--autosave','on',
    '--hedge','on','--hedge_mode','sigma'
  )

  # RL은 학습 경로 강제(캐시 회피), HJB는 단회 평가
  switch ($mth) {
    'rl' {
      switch ($CliMode) {
        'rl'    { $args += @('--mode','rl') }
        'once'  { $args += @('--mode','once') }
        'calib' { $args += @('--mode','calib') }
        default { } # auto → 주입 안 함
      }
    }
    'hjb' { $args += @('--mode','once') }
  }

  $tag = $null

  switch ($Var) {
    'hedge_sigma_k' {
      $args += @('--hedge_sigma_k', (ToInv $val))
      $tag = "{0}_OAT_h{1}" -f $TagPrefix, (ToInv $val)
    }
    'mix_us' {
      if ($mth -ne 'hjb') { throw "Var=mix_us 는 hjb 전용입니다. 현재 method=$mth" }
      $us = [double]$val
      if ($us -lt 0.0 -or $us -gt 1.0) { throw "mix_us는 [0,1] 범위여야 합니다. 입력: $us" }
      $kr   = 0.0
      $gold = [math]::Round(1.0 - $us, 10)
      $alpha = "{0},{1},{2}" -f (ToInv $kr), (ToInv $us), (ToInv $gold) # (KR,US,AU)
      $args += @('--alpha_mix', $alpha, '--hedge_sigma_k','0')          # 교란 최소화
      $tag = "{0}_OAT_us{1}" -f $TagPrefix, (ToInv $us)
    }
    'loss_aversion' {
      # 레거시: κ 항상 주입(평가/학습 모두 반영)
      $args += @('--bh_on','on','--la_k', (ToInv $val))
      $tag = "{0}_OAT_la{1}" -f $TagPrefix, (ToInv $val)
    }
    'bias_loss_aversion' {
      # 신규: 호환 위해 dual injection 유지
      $args += @('--bh_on','on','--la_k', (ToInv $val))
      $args += @('--bias_on','on','--bias_loss_aversion', (ToInv $val))
      $tag = "{0}_OAT_la{1}" -f $TagPrefix, (ToInv $val)
    }
    default { throw "지원하지 않는 Var: $Var" }
  }

  if ($PolicyLocked) {
    # 정책 고정 리포팅 플래그(학습 미수행)
    $args += @('--report_utility','on','--cstar_mode','fixed','--cstar_m','0.5','--return_actor','on')
  }

  # 사용자 추가 인자
  if ($PSBoundParameters.ContainsKey('Extra') -and -not [string]::IsNullOrWhiteSpace($Extra)) {
    $args += @(Split-List $Extra)
  }

  if (-not $tag) { $tag = "{0}_OAT_{1}_{2}" -f $TagPrefix, $Var, (ToInv $val) }
  $args += @('--tag', $tag)
  return ,$args
}

# ── 개별 실행(기존 기능) ────────────────────────────────
$SeedList = Parse-Seeds $Seeds
$ValList  = Parse-Doubles $Values

function Run-ScalarBatch([string]$v,[string]$vals,[string]$m,[string]$mode) {
  Info-Header $v $vals $m $mode
  foreach ($seed in $SeedList) {
    foreach ($method in (if ($m -eq 'both') { @('rl','hjb') } else { @($m) })) {
      foreach ($value in (Parse-Doubles $vals)) {
        $args = Build-ArgsFor -mth $method -val $value -seed $seed
        $title = "OAT $v=$(ToInv $value)  method=$method  seed=$seed"
        RunPy -title $title -argv $args
      }
    }
  }
  Write-Host "[OK] OAT batch completed." -ForegroundColor Green
}

# ── 파이프라인 프리셋(ABCD) 구현 ────────────────────────
function Make-Snapshot-And-Score() {
  $Out = $OutRoot
  $Log = Join-Path $Out '_logs'
  $snap = Join-Path $Out ("{0}_metrics_snapshot.csv" -f $TagPrefix)

  $cmd1 = @'
(Import-Csv (Join-Path $Log 'metrics.csv') | ?{ $_.tag -like "' + $TagPrefix + '_OAT_*" } | `
  Group-Object tag,method,seed | % { $_.Group | Select-Object -Last 1 }) | `
  Export-Csv "' + $snap + '" -NoTypeInformation
'@

  if ($DryRun) {
    Write-Host "powershell -Command $cmd1" -ForegroundColor DarkGray
  } else {
    powershell -Command $cmd1 | Out-Null
    Write-Host "[OK] saved: $snap"
  }

  $scoreArgs = @(
    '--src', $snap, '--tag_startswith', ("{0}_OAT_" -f $TagPrefix),
    '--metrics', 'EW,ES95', '--weights', $ScoreWeights, '--es_mode', $EsMode, '--out', 'inplace'
  )

  # RL 점수화
  $scoreRL = @('scripts\score_snapshot.py') + $scoreArgs + @('--method','rl')
  if ($DryRun) { Write-Host "python $($scoreRL -join ' ')" -ForegroundColor DarkGray }
  else {
    & python @scoreRL; if ($LASTEXITCODE -ne 0) { throw "FAILED: score RL" }
  }

  # HJB 점수화
  $scoreHJB = @('scripts\score_snapshot.py') + $scoreArgs + @('--method','hjb')
  if ($DryRun) { Write-Host "python $($scoreHJB -join ' ')" -ForegroundColor DarkGray }
  else {
    & python @scoreHJB; if ($LASTEXITCODE -ne 0) { throw "FAILED: score HJB" }
  }

  # 퀵 리포트
  if ($DryRun) { Write-Host "python .\scripts\make_quick_report.py" -ForegroundColor DarkGray }
  else {
    & python .\scripts\make_quick_report.py
    if ($LASTEXITCODE -ne 0) { throw "FAILED: quick report" }
  }
}

if ($Preset -eq 'abcd') {
  # ---- A) σ-헤지(11,12 reseed) ----
  $Var = 'hedge_sigma_k'; $Values = $HedgeValues; $Method = 'rl'; $Seeds = $HedgeSeeds; $PolicyLocked = $false
  $SeedList = Parse-Seeds $Seeds
  Run-ScalarBatch -v $Var -vals $Values -m $Method -mode $Mode

  # ---- B) κ 재학습(bias, 학습강화) ----
  $Var = 'bias_loss_aversion'; $Values = $BiasValues; $Method = 'rl'; $Seeds = "11"; $PolicyLocked = $false
  $SeedList = Parse-Seeds $Seeds
  $bakExtra = $Extra; $Extra = $ExtraBias
  Run-ScalarBatch -v $Var -vals $Values -m $Method -mode $Mode
  $Extra = $bakExtra

  # ---- C) Policy-Locked(평가/리포트) ----
  $Var = 'bias_loss_aversion'; $Values = $LockedValues; $Method = 'rl'; $Seeds = "11"; $PolicyLocked = $true
  $SeedList = Parse-Seeds $Seeds
  Run-ScalarBatch -v $Var -vals $Values -m $Method -mode $Mode
  $PolicyLocked = $false

  # ---- D) 스냅샷·점수화·퀵리포트 ----
  Make-Snapshot-And-Score
  return
}

# ── (기존) 단일 배치 실행 경로 ──────────────────────────
Info-Header $Var $Values $Method $Mode
Run-ScalarBatch -v $Var -vals $Values -m $Method -mode $Mode
