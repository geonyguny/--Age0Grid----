param(
  [ValidateSet("hjb","rl")] [string]$Method = "hjb",
  [ValidateSet("dev","overnight")] [string]$Mode = "dev",

  # 그리드 지정 (둘 중 하나 방식 사용)
  [string]$Mixes = "",            # 예: "0,0.1,0.2,0.25"
  [double]$MixFrom = [double]::NaN,
  [double]$MixTo   = [double]::NaN,
  [double]$MixStep = [double]::NaN,

  [string]$HVals = "",            # 예: "0.18,0.20,0.22,0.24,0.26,0.28,0.30,0.32,0.34"
  [double]$HFrom = [double]::NaN,
  [double]$HTo   = [double]::NaN,
  [double]$HStep = [double]::NaN,

  [string]$Seeds = "11,12,13",
  [int]$NpathsDev = 2000,
  [int]$NpathsOvernight = 5000,

  [string]$DataProfile = "dev",
  [string]$TagPrefix = "DEV_2D_",
  [switch]$DryRun,
  [string]$Extra = "",            # 추가 CLI 인자 문자열

  # Post 단계(스냅샷/스코어/최적점/히트맵) 실행 여부
  [switch]$Post,
  [string]$Weights = "0.6,0.4",
  [string]$ESMode = "wealth",
  [string]$OverlayPoints = ".\outputs\figs\optimal_points.json"
)

# ── 고정 경로/런타임 설정
$ErrorActionPreference = "Stop"
[System.Threading.Thread]::CurrentThread.CurrentCulture = 'en-US'
$Root = (Resolve-Path ".").Path
$Py   = ".\.venv\Scripts\python.exe"
$Out  = ".\outputs"
$Log  = Join-Path $Out "_logs"
$snap = Join-Path $Out "DEV_metrics_snapshot.csv"

function _ParseList([string]$s) {
  if ([string]::IsNullOrWhiteSpace($s)) { return @() }
  return $s.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" } | ForEach-Object { [double]$_ }
}
function _BuildRange([double]$a,[double]$b,[double]$c) {
  if ([double]::IsNaN($a) -or [double]::IsNaN($b) -or [double]::IsNaN($c)) { return @() }
  $vals = @(); $x = $a
  while ($x -le $b + 1e-12) { $vals += [math]::Round($x, 6); $x += $c }
  return $vals
}
function _Run($title, $argv, [switch]$dry) {
  if ($dry) { Write-Host $argv -ForegroundColor Yellow; return }
  Invoke-Expression $argv
}

# ── 격자 구성
$mixList = if ($Mixes) { _ParseList $Mixes } else { _BuildRange $MixFrom $MixTo $MixStep }
$hList   = if ($HVals) { _ParseList $HVals } else { _BuildRange $HFrom $HTo $HStep }
if (-not $mixList -or -not $hList) {
  throw "Mixes/HVals 또는 Range(MixFrom~To~Step, HFrom~To~Step)를 지정하세요."
}
$seedList = $Seeds.Split(",") | ForEach-Object { [int]($_.Trim()) }

# ── Npaths 결정
$N = if ($Mode -eq "overnight") { $NpathsOvernight } else { $NpathsDev }

Write-Host "[BATCH 2D] Method=$Method  Mode=$Mode  N=$N" -ForegroundColor Cyan
Write-Host "[PROFILE] $DataProfile  [SEEDS] $($seedList -join ', ')" -ForegroundColor DarkCyan
Write-Host "[GRID] mix_us=[$($mixList -join ', ')]  hedge_sigma_k=[$($hList -join ', ')]" -ForegroundColor DarkCyan
if ($Extra) { Write-Host "[EXTRA] $Extra" -ForegroundColor DarkYellow }

# ── 실행 루프
foreach ($s in $seedList) {
  foreach ($u in $mixList) {
    foreach ($h in $hList) {
      $alpha_mix = ("0.0,{0},{1}" -f $u, (1.0 - $u))
      $tag = ("{0}us{1}_h{2}" -f $TagPrefix, $u, $h)
      $cli = @(
        "$Py -m project.runner.cli",
        "--method $Method",
        "--data_profile $DataProfile",
        "--market_mode bootstrap",
        "--alpha_mix $alpha_mix",
        "--hedge on --hedge_mode sigma --hedge_sigma_k $h",
        "--n_paths $N --seed $s",
        "--tag $tag --print_mode summary --autosave on"
      ) -join " "
      if ($Extra) { $cli = "$cli $Extra" }
      Write-Host ">> RUN us=$u  h=$h  seed=$s  tag=$tag"
      _Run "sim" $cli -dry:$DryRun
    }
  }
}
Write-Host "[OK] 2D batch completed." -ForegroundColor Green

if ($Post) {
  # 스냅샷 최신본 갱신 (2D만 추출)
  $snapCmd = @"
(Import-Csv (Join-Path '$Log' 'metrics.csv') |
  Where-Object { `$_.tag -like '${TagPrefix}*' } |
  Group-Object tag,method,seed | ForEach-Object { `$_.Group | Select-Object -Last 1 }) |
  Export-Csv '$snap' -NoTypeInformation
"@
  Write-Host ">> SNAPSHOT update → $snap"
  if (-not $DryRun) { powershell -NoProfile -Command $snapCmd } else { Write-Host $snapCmd -ForegroundColor Yellow }

  # CompositeScore 생성 (메서드별)
  $score = "$Py .\scripts\score_snapshot.py --src $snap --tag_startswith $TagPrefix --metrics EW,ES95 --weights $Weights --method $Method --es_mode $ESMode --out inplace"
  Write-Host ">> SCORE → CompositeScore ($Weights, $Method, $ESMode)"
  _Run "score" $score -dry:$DryRun

  # 최적점 & 히트맵
  $find = "$Py .\scripts\find_optima.py --src $snap --tag_startswith $TagPrefix --x mix_us --y hedge_sigma_k --z CompositeScore --tiebreak ES95 --agg median"
  Write-Host ">> OPTIMA"
  _Run "find" $find -dry:$DryRun

  $hm = "$Py .\scripts\make_paper_figs.py oat-heatmap --src $snap --tag_startswith $TagPrefix --method $Method --es_mode $ESMode --x mix_us --y hedge_sigma_k --zlist EW,ES95,CompositeScore --agg median --annotate on --overlay_points $OverlayPoints --overlay_label best"
  Write-Host ">> HEATMAPS"
  _Run "heatmap" $hm -dry:$DryRun
}
