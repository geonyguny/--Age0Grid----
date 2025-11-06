param(
  [ValidateSet('hedge_sigma_k','mix_us','loss_aversion','bias_loss_aversion')]
  [string]$Var,

  [Parameter(Mandatory=$true)]
  [string]$Values,

  [ValidateSet('rl','hjb','both')]
  [string]$Method = 'both',

  [ValidateSet('dev','overnight')]
  [string]$Mode = 'dev',

  [ValidateSet('auto','rl','once','calib')]
  [string]$CliMode = 'rl',

  [switch]$Overwrite,
  [string]$Extra,
  [switch]$DryRun,

  [string]$Seeds = "11",

  [switch]$PolicyLocked
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName 'System.Globalization' | Out-Null
$inv = [System.Globalization.CultureInfo]::InvariantCulture

$ScriptDir   = $PSScriptRoot
$ProjectRoot = Split-Path -Parent $ScriptDir
Set-Location $ProjectRoot

$Py      = '.\.venv\Scripts\python.exe'
$OutRoot = '.\outputs'
$LogDir  = Join-Path $OutRoot '_logs'
New-Item -ItemType Directory -Force -Path $OutRoot,$LogDir | Out-Null

function ToInv([double]$x){ $x.ToString($inv) }
function Split-List([string]$x){
  if ([string]::IsNullOrWhiteSpace($x)) { return @() }
  @($x -split '[,\s]+' | ?{ $_ -ne '' })
}
function Parse-Doubles([string]$csv){
  $vals = [System.Collections.Generic.List[double]]::new()
  foreach($t in (Split-List $csv)){
    try { [void]$vals.Add([double]::Parse($t,$inv)) }
    catch { throw "Values 파싱 실패: '$t' (콤마/공백 구분, 소수점은 . 사용)" }
  }
  if($vals.Count -eq 0){ throw "Values가 비어 있습니다." }
  @($vals.ToArray())
}
function Parse-Seeds([string]$seedsText){
  $out = [System.Collections.Generic.List[int]]::new()
  foreach($t in (Split-List $seedsText)){
    try { [void]$out.Add([int]::Parse($t,$inv)) }
    catch { throw "Seeds 파싱 실패: '$t' (예: 11,12 또는 '11 12')" }
  }
  if($out.Count -eq 0){ $out.Add(11) }
  @($out.ToArray())
}

switch($Mode){
  'dev' { $profile='dev';  $nPathsRL=2000;  $nPathsHJB=2000;  $TagPrefix='DEV' }
  'overnight' { $profile='full'; $nPathsRL=8000; $nPathsHJB=30000; $TagPrefix='OVN' }
}

$SeedList = Parse-Seeds $Seeds
$ValList  = Parse-Doubles $Values

Write-Host "[BATCH OAT] Var=$Var  Values=$Values  Method=$Method  Mode=$Mode" -ForegroundColor Cyan
Write-Host "[PROFILE] $profile  [SEEDS] $([string]::Join(',', $SeedList))  [RL n_paths] $nPathsRL  [HJB n_paths] $nPathsHJB" -ForegroundColor DarkCyan
if($Overwrite){ Write-Warning "현재 엔진은 --overwrite 인자를 지원하지 않습니다. (무시)" }

function Get-NPaths([string]$mth){ if($mth -eq 'rl'){ $nPathsRL } else { $nPathsHJB } }

function RunPy([string]$title,[string[]]$argv,[switch]$dry){
  Write-Host ">> $title" -ForegroundColor Cyan
  if($dry){ $cmd = "$Py " + ($argv -join ' '); Write-Host $cmd -ForegroundColor DarkGray; return }
  & $Py @argv
  if($LASTEXITCODE -ne 0){ throw "FAILED: $title (exit=$LASTEXITCODE)" }
}

function Build-ArgsFor([string]$mth,[double]$val,[int]$seed){
  $args = @(
    '-m','project.runner.cli',
    '--method',$mth,
    '--data_profile',$profile,
    '--market_mode','bootstrap',
    '--n_paths',(Get-NPaths $mth).ToString(),
    '--seed',$seed.ToString(),
    '--print_mode','summary',
    '--autosave','on',
    '--hedge','on','--hedge_mode','sigma'
  )

  switch($mth){
    'rl' {
      switch($CliMode){
        'rl'    { $args += @('--mode','rl') }
        'once'  { $args += @('--mode','once') }
        'calib' { $args += @('--mode','calib') }
        default { } # auto → 미주입
      }
    }
    'hjb' { $args += @('--mode','once') }
  }

  $tag = $null
  switch($Var){
    'hedge_sigma_k' {
      $args += @('--hedge_sigma_k',(ToInv $val))
      $tag = "{0}_OAT_h{1}" -f $TagPrefix,(ToInv $val)
    }
    'mix_us' {
      if($mth -ne 'hjb'){ throw "Var=mix_us 는 hjb 전용입니다. 현재 method=$mth" }
      $us = [double]$val
      if($us -lt 0.0 -or $us -gt 1.0){ throw "mix_us는 [0,1] 범위여야 합니다. 입력: $us" }
      $kr=0.0; $gold=[math]::Round(1.0-$us,10)
      $alpha = "{0},{1},{2}" -f (ToInv $kr),(ToInv $us),(ToInv $gold)
      $args += @('--alpha_mix',$alpha,'--hedge_sigma_k','0')
      $tag = "{0}_OAT_us{1}" -f $TagPrefix,(ToInv $us)
    }
    'loss_aversion' {
      $args += @('--bh_on','on','--la_k',(ToInv $val))
      $tag = "{0}_OAT_la{1}" -f $TagPrefix,(ToInv $val)
    }
    'bias_loss_aversion' {
      $args += @('--bh_on','on','--la_k',(ToInv $val))
      $args += @('--bias_on','on','--bias_loss_aversion',(ToInv $val))
      $tag = "{0}_OAT_la{1}" -f $TagPrefix,(ToInv $val)
    }
    default { throw "지원하지 않는 Var: $Var" }
  }

  if($PolicyLocked){
    $args += @('--report_utility','on','--cstar_mode','fixed','--cstar_m','0.5','--return_actor','on')
  }

  if($PSBoundParameters.ContainsKey('Extra') -and -not [string]::IsNullOrWhiteSpace($Extra)){
    $args += @(Split-List $Extra)
  }

  if(-not $tag){ $tag = "{0}_OAT_{1}_{2}" -f $TagPrefix,$Var,(ToInv $val) }
  $args += @('--tag',$tag)
  ,$args
}

$methods = @(); switch($Method){ 'both' { $methods=@('rl','hjb') } default { $methods=@($Method) } }

foreach($seed in $SeedList){
  foreach($m in $methods){
    foreach($v in $ValList){
      $args = Build-ArgsFor -mth $m -val $v -seed $seed
      $title = "OAT $Var=$(ToInv $v)  method=$m  seed=$seed"
      RunPy -title $title -argv $args -dry:$DryRun
    }
  }
}

Write-Host "[OK] OAT batch completed." -ForegroundColor Green
