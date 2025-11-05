param(
  # 실험 변수 키(하위호환 포함)
  [ValidateSet('hedge_sigma_k','mix_us','loss_aversion','bias_loss_aversion')]
  [string]$Var,

  # 콤마 구분 값들: 예) "0,0.5,1.0"
  [Parameter(Mandatory=$true)]
  [string]$Values,

  # 방법론 선택
  [ValidateSet('rl','hjb','both')]
  [string]$Method = 'both',

  # 실행 프로파일
  [ValidateSet('dev','overnight')]
  [string]$Mode = 'dev',

  # 임의 추가 인자 전달용 (예: -Extra "--validate on --something 1")
  [string]$Extra,

  # DryRun: 파이썬 실행 대신 명령만 출력
  [switch]$DryRun,

  # ✅ 여러 시드 지원 (예: -Seeds 11,12)
  [int[]]$Seeds = @(11)
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

Write-Host "[BATCH OAT] Var=$Var  Values=$Values  Method=$Method  Mode=$Mode" -ForegroundColor Cyan
Write-Host "[PROFILE] $profile  [SEEDS] $($Seeds -join ',')  [RL n_paths] $nPathsRL  [HJB n_paths] $nPathsHJB" -ForegroundColor DarkCyan

function Get-NPaths([string]$mth) {
  if ($mth -eq 'rl') { return $nPathsRL } else { return $nPathsHJB }
}

function RunPy([string]$title, [string[]]$argv, [switch]$dry) {
  Write-Host ">> $title" -ForegroundColor Cyan
  if ($dry) {
    $cmd = "$Py " + ($argv -join ' ')
    Write-Host $cmd -ForegroundColor DarkGray
    return
  }
  & $Py @argv
  if ($LASTEXITCODE -ne 0) { throw "FAILED: $title (exit=$LASTEXITCODE)" }
}

function ToInv([double]$x) { return $x.ToString($inv) }

function Split-Extra([string]$x) {
  if ([string]::IsNullOrWhiteSpace($x)) { return @() }
  return @($x -split '\s+' | Where-Object { $_ -ne '' })
}

function Build-ArgsFor([string]$mth, [double]$val, [int]$seed) {
  # 공통 인자
  $args = @(
    '-m','project.runner.cli',
    '--method', $mth,
    '--data_profile', $profile,
    '--market_mode','bootstrap',
    '--n_paths', (Get-NPaths $mth).ToString(),
    '--seed', $seed.ToString(),
    '--print_mode','summary',
    '--autosave','on',
    '--hedge','on','--hedge_mode','sigma'   # 기본: σ-헤지 on
  )

  $tag = $null

  switch ($Var) {
    'hedge_sigma_k' {
      # σ-헤지 강도 스윕
      $args += @('--hedge_sigma_k', (ToInv $val))
      $tag = "{0}_OAT_h{1}" -f $TagPrefix, (ToInv $val)
    }

    'mix_us' {
      if ($mth -ne 'hjb') {
        throw "Var=mix_us 는 hjb 전용입니다. 현재 method=$mth"
      }
      $us = [double]$val
      if ($us -lt 0.0 -or $us -gt 1.0) { throw "mix_us는 [0,1] 범위여야 합니다. 입력: $us" }
      $kr   = 0.0
      $gold = [math]::Round(1.0 - $us, 10)
      $alpha = "{0},{1},{2}" -f (ToInv $kr), (ToInv $us), (ToInv $gold)  # alpha_mix=(kr,us,gold)
      # 구성 스윕 시 헤지강도 0으로 고정(교란 최소화)
      $args += @('--alpha_mix', $alpha, '--hedge_sigma_k','0')
      $tag = "{0}_OAT_us{1}" -f $TagPrefix, (ToInv $us)
    }

    'loss_aversion' {        # 하위호환 키 (레거시 계열)
      # 정책 재학습 반영(학습 목적함수): bh/la 계열 플래그 사용
      $args += @('--bh_on','on','--la_k', (ToInv $val))
      $tag = "{0}_OAT_la{1}" -f $TagPrefix, (ToInv $val)
    }

    'bias_loss_aversion' {   # 신규 계열
      # 정책 재학습 반영(학습 목적함수): bias 계열 플래그 사용
      $args += @('--bias_on','on','--bias_loss_aversion', (ToInv $val))
      $tag = "{0}_OAT_la{1}" -f $TagPrefix, (ToInv $val)
    }

    default {
      throw "지원하지 않는 Var: $Var"
    }
  }

  # 사용자 추가 인자
  $extraArgs = Split-Extra $Extra
  if ($extraArgs -and $extraArgs.Length -gt 0) { $args += $extraArgs }

  if (-not $tag) { $tag = "{0}_OAT_{1}_{2}" -f $TagPrefix, $Var, (ToInv $val) }
  $args += @('--tag', $tag)
  return ,$args
}

# 값 파싱
$vals = [System.Collections.Generic.List[double]]::new()
foreach ($s in ($Values -split ',')) {
  $t = $s.Trim()
  if ($t -ne '') {
    try { [void]$vals.Add([double]::Parse($t, $inv)) }
    catch { throw "Values 파싱 실패: '$t' (콤마로 구분, 소수점은 . 사용)" }
  }
}
if ($vals.Count -eq 0) { throw "Values가 비어 있습니다." }

# 실행 대상 메서드 확정
$methods = @()
switch ($Method) {
  'both' { $methods = @('rl','hjb') }
  default { $methods = @($Method) }
}

# 실행 루프
foreach ($seed in $Seeds) {
  foreach ($m in $methods) {
    foreach ($v in $vals) {
      $args = Build-ArgsFor -mth $m -val $v -seed $seed
      $title = "OAT $Var=$(ToInv $v)  method=$m  seed=$seed"
      RunPy -title $title -argv $args -dry:$DryRun
    }
  }
}

Write-Host "[OK] OAT batch completed." -ForegroundColor Green
