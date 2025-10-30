# D:\01_simul\scripts\run_stress1.ps1
param(
  [string]   $Root         = "D:\01_simul",
  [string]   $Tag          = "stress1",
  [string]   $SeedSpec     = "0-19",                 # "0-19" 또는 "0,1,2,3,4"
  [string[]] $Methods      = @("hjb","rl","rule"),   # 실행할 메서드 집합
  [string]   $Block        = "6",                    # 부트스트랩 블록: "6" or "6m" or "12" or "2y" or "90d"
  [double]   $FeeAnnual    = 0.003,                  # 30 bps
  [double]   $FMinReal     = 0.0,                    # floor_real
  [string]   $MarketMode   = "bootstrap",            # "iid"|"bootstrap"
  [string]   $DataProfile  = "full",                 # "dev"|"full"
  [string]   $RuleBaseline = "cpb",                  # rule 전용: "cpb"|"vpw"|"kgr"|"4pct"
  [switch]   $Quiet,                                  # 콘솔 출력 최소화 (cli --quiet on)
  [switch]   $NoLog                                   # PowerShell Write-Host 최소화
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Expand-SeedSpec([string]$spec){
  if ([string]::IsNullOrWhiteSpace($spec)) { return @(0) }
  if ($spec -match '^\d+-\d+$') {
    $a,$b = $spec -split '-'
    return [int]$a..[int]$b
  }
  return ($spec -split '[,\s]+' | Where-Object { $_ -ne '' }) | ForEach-Object { [int]$_ }
}

function Convert-BlockToMonths([string]$b) {
  # 허용: "6" / "6m" / "12" / "2y" / "90d"
  $s = ($b ?? "").Trim().ToLower()
  if ($s -match '^\d+$') { return [int]$s }             # 그대로 개월
  if ($s -match '^\d+\s*m$') { return [int]($s -replace 'm','') }
  if ($s -match '^\d+\s*y$') { return [int]([int]($s -replace 'y','') * 12) }
  if ($s -match '^\d+\s*d$') {
    $days = [int]($s -replace 'd','')
    return [int][math]::Max(1, [math]::Round($days / 30.0))  # 대략 월 환산
  }
  throw "Invalid Block value: '$b' (use like 6 | 6m | 12 | 2y | 90d)"
}

function Log([string]$msg, [string]$color="Gray") {
  if (-not $NoLog) { Write-Host $msg -ForegroundColor $color }
}

# ── 환경 세팅
Set-Location $Root
$env:PYTHONPATH = $Root          # project를 최상위 패키지로 import
$python = Join-Path $Root ".venv\Scripts\python.exe"
$seeds  = Expand-SeedSpec $SeedSpec
$blockM = Convert-BlockToMonths $Block
$quietArg = if ($Quiet) { @("--quiet","on") } else { @("--quiet","off") }

# 유효성
if ($Methods.Count -eq 0) { throw "Methods 비어있음. 예: -Methods @('hjb','rl','rule')" }
if ($MarketMode -notin @("iid","bootstrap")) { throw "MarketMode must be 'iid' or 'bootstrap'." }
if ($DataProfile -and $DataProfile -notin @("dev","full")) { throw "DataProfile must be 'dev' or 'full'." }
if ($Methods -contains "rule" -and [string]::IsNullOrWhiteSpace($RuleBaseline)) {
  throw "rule 실행 시 -RuleBaseline 필요 (cpb|vpw|kgr|4pct)."
}

Log "== Stress batch start ==" "Cyan"
Log "Root        : $Root"
Log "Tag         : $Tag"
Log "Seeds       : $($seeds -join ', ')"
Log "Methods     : $($Methods -join ', ')"
Log "Block(month): $blockM  (input:$Block)"
Log "FeeAnnual   : $FeeAnnual"
Log "Floor(real) : $FMinReal"
Log "Mode/Profile: $MarketMode / $DataProfile"
if ($Methods -contains "rule") { Log "RuleBaseline: $RuleBaseline" }

foreach ($m in $Methods) {
  $args = @(
    "-m","project.runner.cli",
    "--mode","auto",
    "--method",$m,
    "--fee_annual",$FeeAnnual.ToString("0.########"),
    "--floor_on","--f_min_real",$FMinReal.ToString("0.########"),
    "--market_mode",$MarketMode,"--bootstrap_block",$blockM,
    "--data_profile",$DataProfile,
    "--outputs",(Join-Path $Root "outputs"),
    "--tag",$Tag,
    "--seeds"
  ) + $seeds + $quietArg

  if ($m -eq "rule") {
    $args += @("--baseline",$RuleBaseline)
  }

  Log ("▶ Running {0} … (seeds=[{1}])" -f $m, ($seeds -join ", ")) "Yellow"
  & $python @args
  if ($LASTEXITCODE -ne 0) {
    Log ("✗ {0} failed with exit {1}" -f $m, $LASTEXITCODE) "Red"
    throw "runner failed"
  }
  Log ("✓ {0} done" -f $m) "Green"
}

Log "== Done: $Tag ==" "Cyan"
