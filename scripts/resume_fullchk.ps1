Param(
  [string]$Py = ".\.venv\Scripts\python.exe",
  [string]$Cli = "project.runner.cli",
  [string]$Metrics = ".\outputs\_logs\metrics.csv",
  [string]$Winners = ".\outputs\_winners.csv"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Has-Prop($row, $name) {
  return $row.PSObject.Properties.Match($name).Count -gt 0
}
function Get-PropVal($row, $name) {
  if (Has-Prop $row $name) { return [string]$row.$name } else { return $null }
}
function Add-Arg([ref]$A, [string]$name, [string]$val) {
  if (-not [string]::IsNullOrWhiteSpace($val)) { $A.Value += @($name, $val) }
}

# 0) metrics.csv에서 tag->method 맵 구축
$methodMap = @{}
if (Test-Path $Metrics) {
  try {
    $M = Import-Csv $Metrics | Where-Object { $_.tag -and $_.method }
    foreach ($m in $M) {
      $t = [string]$m.tag
      $md = [string]$m.method
      if ($t -and $md -and -not $methodMap.ContainsKey($t)) { $methodMap[$t] = $md }
    }
  } catch { }
}

function Infer-Method-FromMetrics([string]$baseTag) {
  if ([string]::IsNullOrWhiteSpace($baseTag)) { return $null }
  if ($methodMap.ContainsKey($baseTag)) { return $methodMap[$baseTag] }
  if (Test-Path $Metrics) {
    try {
      $M = Import-Csv $Metrics | Where-Object { $_.tag -and $_.method }
      $cand = $M | Where-Object { $_.tag -like "*$baseTag*" } | Select-Object -First 1
      if ($cand) { return [string]$cand.method }
    } catch { }
  }
  return $null
}

# 1) 완료된 tag 집합
$doneTags = New-Object 'System.Collections.Generic.HashSet[string]'
if (Test-Path $Metrics) {
  try {
    (Import-Csv $Metrics | Where-Object { $_.tag }) | ForEach-Object { [void]$doneTags.Add($_.tag) }
  } catch { }
}

# 2) 우승 후보 로드
if (!(Test-Path $Winners)) { throw "Winners CSV not found: $Winners" }
$W = Import-Csv $Winners | Sort-Object {[double]$_.CompositeScore} -Descending | Select-Object -First 15

# 3) seed 세트
$seeds = 11,12,13,14,15

foreach ($row in $W) {
  $baseTag = Get-PropVal $row "tag"
  if ([string]::IsNullOrWhiteSpace($baseTag)) { continue }

  # method 우선순위: winners 값 → metrics 유추 → 태그 휴리스틱
  $method = Get-PropVal $row "method"
  if ([string]::IsNullOrWhiteSpace($method)) { $method = Infer-Method-FromMetrics $baseTag }
  if ([string]::IsNullOrWhiteSpace($method)) {
    if ($baseTag -like "rob_*")                   { $method = "hjb" }
    elseif ($baseTag -like "*_rl_*" -or $baseTag -like "*mini*") { $method = "rl" }
    else                                          { $method = "hjb" } # 보수 기본값
  }

  foreach ($s in $seeds) {
    $tag = "fullchk_{0}_s{1}" -f $baseTag, $s
    if ($doneTags.Contains($tag)) {
      Write-Host "[skip] $tag (already in metrics)" -ForegroundColor Yellow
      continue
    }

    $args = @(
      "--method",        $method,
      "--data_profile",  "full",
      "--market_mode",   "bootstrap",
      "--tag",           $tag,
      "--seed",          "$s",
      "--print_mode",    "summary",
      "--autosave",      "on",
      "--n_paths",       "30000"
    )

    # 선택 인자 (존재+비공란일 때만 추가)
    Add-Arg ([ref]$args) "--baseline"        (Get-PropVal $row "baseline")
    Add-Arg ([ref]$args) "--es_mode"         (Get-PropVal $row "es_mode")
    Add-Arg ([ref]$args) "--w_max"           (Get-PropVal $row "w_max")
    Add-Arg ([ref]$args) "--q_floor"         (Get-PropVal $row "q_floor")
    Add-Arg ([ref]$args) "--lambda_term"     (Get-PropVal $row "lambda_term")
    Add-Arg ([ref]$args) "--theta_ambiguity" (Get-PropVal $row "theta_ambiguity")

    # mix
    $mk = Get-PropVal $row "mix_kr"
    $mu = Get-PropVal $row "mix_us"
    $mg = Get-PropVal $row "mix_gold"
    if ($mk -and $mu -and $mg) {
      $args += @("--alpha_mix", ("{0},{1},{2}" -f $mk,$mu,$mg))
    }

    # hedge
    $hr = Get-PropVal $row "hedge_ratio"
    if ($hr) { $args += @("--hedge","on","--hedge_mode","sigma","--hedge_sigma_k",$hr) }

    Write-Host ">> $Py -m $Cli $($args -join ' ')" -ForegroundColor Cyan
    & $Py -m $Cli @args
    [void]$doneTags.Add($tag)
  }
}
