[CmdletBinding()]
Param(
  [Parameter()][string]$Py      = ".\.venv\Scripts\python.exe",
  [Parameter()][string]$Cli     = "project.runner.cli",
  [Parameter()][string]$Metrics = ".\outputs\_logs\metrics.csv",
  [Parameter()][string]$Winners = ".\outputs\_winners.csv"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ── Helpers (Approved verbs 사용)
function Test-Property {
  [CmdletBinding()]
  param(
    [Parameter(Mandatory)][object]$InputObject,
    [Parameter(Mandatory)][string]$Name
  )
  return $InputObject.PSObject.Properties.Match($Name).Count -gt 0
}

function Get-PropertyValue {
  [CmdletBinding()]
  param(
    [Parameter(Mandatory)][object]$InputObject,
    [Parameter(Mandatory)][string]$Name
  )
  if (Test-Property -InputObject $InputObject -Name $Name) { return [string]$InputObject.$Name }
  return $null
}

function Add-Argument {
  [CmdletBinding()]
  param(
    [Parameter(Mandatory)][ref]$TargetArray,
    [Parameter(Mandatory)][string]$Name,
    [Parameter()][string]$Value
  )
  if (-not [string]::IsNullOrWhiteSpace($Value)) {
    $TargetArray.Value += @($Name, $Value)
  }
}

function Get-MethodMap {
  [CmdletBinding()]
  param([Parameter(Mandatory)][string]$MetricsPath)
  $map = @{}
  if (Test-Path $MetricsPath) {
    try {
      $rows = Import-Csv $MetricsPath | Where-Object { $_.tag -and $_.method }
      foreach ($r in $rows) {
        $t = [string]$r.tag; $m = [string]$r.method
        if ($t -and $m -and -not $map.ContainsKey($t)) { $map[$t] = $m }
      }
    } catch { }
  }
  return $map
}

function Get-MethodFromMetrics {
  [CmdletBinding()]
  param(
    [Parameter(Mandatory)][string]$BaseTag,
    [Parameter()][hashtable]$MethodMap,
    [Parameter()][string]$MetricsPath
  )
  if ([string]::IsNullOrWhiteSpace($BaseTag)) { return $null }
  if ($MethodMap -and $MethodMap.ContainsKey($BaseTag)) { return $MethodMap[$BaseTag] }

  if (Test-Path $MetricsPath) {
    try {
      $rows = Import-Csv $MetricsPath | Where-Object { $_.tag -and $_.method }
      $cand = $rows | Where-Object { $_.tag -like "*$BaseTag*" } | Select-Object -Last 1
      if ($cand) { return [string]$cand.method }
    } catch { }
  }
  return $null
}

# ── 0) metrics.csv에서 tag→method 맵
$methodMap = Get-MethodMap -MetricsPath $Metrics

# ── 1) 완료된 tag 집합
$doneTags = New-Object 'System.Collections.Generic.HashSet[string]'
if (Test-Path $Metrics) {
  try {
    (Import-Csv $Metrics | Where-Object { $_.tag }) | ForEach-Object {
      [void]$doneTags.Add([string]$_.tag)
    }
  } catch { }
}

# ── 2) 우승 후보 로드(CompositeScore 내림차순 Top 15)
if (-not (Test-Path $Winners)) { throw "Winners CSV not found: $Winners" }
$W = Import-Csv $Winners |
     Sort-Object { [double]($_.CompositeScore) } -Descending |
     Select-Object -First 15

# ── 3) seed 세트
$seeds = 11,12,13,14,15

# ── 4) 실행 루프
foreach ($row in $W) {
  $baseTag = Get-PropertyValue -InputObject $row -Name "tag"
  if ([string]::IsNullOrWhiteSpace($baseTag)) { continue }

  # method 우선순위: winners → metrics 유추 → 태그 휴리스틱 → 기본 hjb
  $method = Get-PropertyValue -InputObject $row -Name "method"
  if ([string]::IsNullOrWhiteSpace($method)) {
    $method = Get-MethodFromMetrics -BaseTag $baseTag -MethodMap $methodMap -MetricsPath $Metrics
  }
  if ([string]::IsNullOrWhiteSpace($method)) {
    if ($baseTag -like "rob_*")                                  { $method = "hjb" }
    elseif ($baseTag -like "*_rl_*" -or $baseTag -like "*mini*") { $method = "rl" }
    else                                                         { $method = "hjb" } # 보수 기본값
  }

  foreach ($s in $seeds) {
    $tag = "fullchk_{0}_s{1}" -f $baseTag, $s
    if ($doneTags.Contains($tag)) {
      Write-Host "[skip] $tag (already in metrics)" -ForegroundColor Yellow
      continue
    }

    # ← $args(자동변수) 대신 $cliArgs 사용
    $cliArgs = @(
      "-m",              $Cli,
      "--method",        $method,
      "--data_profile",  "full",
      "--market_mode",   "bootstrap",
      "--tag",           $tag,
      "--seed",          "$s",
      "--print_mode",    "summary",
      "--autosave",      "on",
      "--n_paths",       "30000"
    )

    # 선택 인자
    Add-Argument ([ref]$cliArgs) "--baseline"        (Get-PropertyValue -InputObject $row -Name "baseline")
    Add-Argument ([ref]$cliArgs) "--es_mode"         (Get-PropertyValue -InputObject $row -Name "es_mode")
    Add-Argument ([ref]$cliArgs) "--w_max"           (Get-PropertyValue -InputObject $row -Name "w_max")
    Add-Argument ([ref]$cliArgs) "--q_floor"         (Get-PropertyValue -InputObject $row -Name "q_floor")
    Add-Argument ([ref]$cliArgs) "--lambda_term"     (Get-PropertyValue -InputObject $row -Name "lambda_term")
    Add-Argument ([ref]$cliArgs) "--theta_ambiguity" (Get-PropertyValue -InputObject $row -Name "theta_ambiguity")

    # mix (kr,us,gold 순으로 csv에 존재할 때만)
    $mk = Get-PropertyValue -InputObject $row -Name "mix_kr"
    $mu = Get-PropertyValue -InputObject $row -Name "mix_us"
    $mg = Get-PropertyValue -InputObject $row -Name "mix_gold"
    if ($mk -and $mu -and $mg) {
      $cliArgs += @("--alpha_mix", ("{0},{1},{2}" -f $mk,$mu,$mg))
    }

    # hedge
    $hr = Get-PropertyValue -InputObject $row -Name "hedge_ratio"
    if ($hr) {
      $cliArgs += @("--hedge","on","--hedge_mode","sigma","--hedge_sigma_k",$hr)
    }

    Write-Host ">> $Py $($cliArgs -join ' ')" -ForegroundColor Cyan
    & $Py @cliArgs
    [void]$doneTags.Add($tag)
  }
}
