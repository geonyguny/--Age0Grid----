param(
  # 실험 변수 키(하위호환 포함)
  [ValidateSet('hedge_sigma_k','mix_us','loss_aversion','bias_loss_aversion')]
  [string]$Var,

  # 콤마/공백 구분 값들: 예) "0,0.5,1.0" / "0 0.5 1.0"
  [Parameter(Mandatory=$true)]
  [string]$Values,

  # 방법론 선택
  [ValidateSet('rl','hjb','both')]
  [string]$Method = 'both',

  # 실행 프로파일(dev=빠른점검 / overnight=대용량)
  [ValidateSet('dev','overnight')]
  [string]$Mode = 'dev',

  # CLI 실행 모드 강제: auto|rl|once|calib (RL 캐시 방지 기본 rl 권장)
  [ValidateSet('auto','rl','once','calib')]
  [string]$CliMode = 'rl',

  # 현재 엔진 미지원 플래그 → 경고 후 무시
  [switch]$Overwrite,

  # 임의 추가 인자 전달용 (예: -Extra "--rl_epochs 8 --rl_steps_per_epoch 2000")
  [string]$Extra,

  # DryRun: 파이썬 실행 대신 명령만 출력
  [switch]$DryRun,

  # 여러 시드(문자열/쉼표/공백/배열 모두 허용): -Seeds "11,12" / -Seeds 11,12 / -Seeds "11 12"
  [string]$Seeds = "11",

  # 정책 고정 리포팅 모드 (학습 대신 평가/리포트 플래그만 부여)
  [switch]$PolicyLocked
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

$SeedList = Parse-Seeds $Seeds
$ValList  = Parse-Doubles $Values

Write-Host "[BATCH OAT] Var=$Var  Values=$Values  Method=$Method  Mode=$Mode" -ForegroundColor Cyan
Write-Host "[PROFILE] $profile  [SEEDS] $([string]::Join(',', $SeedList))  [RL n_paths] $nPathsRL  [HJB n_paths] $nPathsHJB" -ForegroundColor DarkCyan
if ($Overwrite) { Write-Warning "현재 엔진은 --overwrite 인자를 지원하지 않습니다. (무시)" }

function Get-NPaths([string]$mth) { if ($mth -eq 'rl') { return $nPathsRL } else { return $nPathsHJB } }

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

  # 모드 주입: RL은 기본적으로 학습 경로 강제, HJB는 단회 평가
  switch ($mth) {
    'rl'  {
      switch ($CliMode) {
        'rl'    { $args += @('--mode','rl') }
        'once'  { $args += @('--mode','once') }
        'calib' { $args += @('--mode','calib') }
        default { } # auto → 미주입
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
      $alpha = "{0},{1},{2}" -f (ToInv $kr), (ToInv $us), (ToInv $gold)  # (KR,US,AU)
      $args += @('--alpha_mix', $alpha, '--hedge_sigma_k','0')  # 교란 최소화
      $tag = "{0}_OAT_us{1}" -f $TagPrefix, (ToInv $us)
    }

    'loss_aversion' {
      # 레거시 계열: κ를 항상 주입(PolicyLocked 여부 무관) → 평가/학습 모두 반영 가능
      $args += @('--bh_on','on','--la_k', (ToInv $val))
      $tag = "{0}_OAT_la{1}" -f $TagPrefix, (ToInv $val)
    }

    'bias_loss_aversion' {
      # 신규 계열: 엔진 호환성을 위해 dual injection 유지 (레거시 + 신규)
      $args += @('--bh_on','on','--la_k', (ToInv $val))
      $args += @('--bias_on','on','--bias_loss_aversion', (ToInv $val))
      $tag = "{0}_OAT_la{1}" -f $TagPrefix, (ToInv $val)
    }

    default { throw "지원하지 않는 Var: $Var" }
  }

  # 정책 고정 리포팅 모드: 학습 없이 평가 유틸/소비 리포팅 강제
  if ($PolicyLocked) {
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

# 실행 대상 메서드 확정
$methods = @()
switch ($Method) {
  'both' { $methods = @('rl','hjb') }
  default { $methods = @($Method) }
}

# 실행 루프
foreach ($seed in $SeedList) {
  foreach ($m in $methods) {
    foreach ($v in $ValList) {
      $args = Build-ArgsFor -mth $m -val $v -seed $seed
      $title = "OAT $Var=$(ToInv $v)  method=$m  seed=$seed"
      RunPy -title $title -argv $args -dry:$DryRun
    }
  }
}

Write-Host "[OK] OAT batch completed." -ForegroundColor Green
