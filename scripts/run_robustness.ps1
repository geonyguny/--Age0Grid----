Param(
  [string] $Python     = ".\.venv\Scripts\python.exe",
  [string] $OutRoot    = ".\outputs",
  [string] $Profile    = "dev",
  [string] $TagPrefix  = "robust",
  [int]    $EvalPaths  = 100,
  [int]    $TimeoutSec = 900,        # 각 케이스 타임아웃(초)
  [int]    $Retry      = 0,          # 실패 시 재시도 횟수
  [switch] $MakeFigures,             # night_robust_* 요약/그림 생성
  [switch] $EstimateOnly,            # ETA만 출력하고 종료
  [int]    $EstimateSecPerCase = 30, # ETA 계산 기본값(초)

  # 스윕 파라미터(미지정 시 기본값 사용)
  [string[]] $Windows,
  [int[]]    $Blocks,
  [double[]] $HFXs,
  [double[]] $Costs
)

$ErrorActionPreference = "Stop"
$ProgressPreference    = "SilentlyContinue"

#---------------------------
# Helpers
#---------------------------
function New-StudyRoot {
  param([string]$BaseOut)
  $ts   = Get-Date -Format "yyyyMMdd_HHmm"
  $root = Join-Path $BaseOut ("_study\robust_" + $ts)
  New-Item -ItemType Directory -Force -Path $root | Out-Null
  return @{ Root = $root; Ts = $ts }
}

function Normalize-WindowKey {
  param([string]$w)
  if ([string]::IsNullOrWhiteSpace($w)) { return "all" }
  $k = $w -replace "[:\s]", "_" -replace "-", ""
  if ($k -eq "_" -or $k -eq "") { $k = "all" }
  return $k
}

function Invoke-NativeCapture {
  param(
    [string]  $Exe,
    [string[]]$Args,
    [int]     $Timeout = 0 # 0 = infinite
  )
  $psi = New-Object System.Diagnostics.ProcessStartInfo
  $psi.FileName               = $Exe
  $psi.Arguments              = ($Args -join ' ')
  $psi.UseShellExecute        = $false
  $psi.RedirectStandardOutput = $true
  $psi.RedirectStandardError  = $true
  $psi.CreateNoWindow         = $true

  $p = New-Object System.Diagnostics.Process
  $p.StartInfo = $psi
  [void]$p.Start()

  if ($Timeout -le 0) {
    $null = $p.WaitForExit()
  } else {
    if (-not $p.WaitForExit($Timeout * 1000)) {
      try { $p.Kill($true) } catch {}
      return @{ Text = ""; ExitCode = 124; TimedOut = $true; DurationSec = $Timeout }
    }
  }

  $stdout = $p.StandardOutput.ReadToEnd()
  $stderr = $p.StandardError.ReadToEnd()
  $text   = $stdout + "`n" + $stderr
  $dur    = ($p.ExitTime - $p.StartTime).TotalSeconds
  return @{ Text = $text; ExitCode = $p.ExitCode; TimedOut = $false; DurationSec = [int]$dur }
}

function Parse-Metrics {
  param([string[]]$Lines)
  $o = [ordered]@{ EW=$null; ES95=$null; Ruin=$null; MeanWT=$null }
  foreach($ln in $Lines){
    if ($o.EW     -eq $null -and $ln -match '^\s*EW\s*:\s*([0-9eE\.\-]+)')      { $o.EW     = [double]$Matches[1]; continue }
    if ($o.ES95   -eq $null -and $ln -match '^\s*ES95\s*:\s*([0-9eE\.\-]+)')    { $o.ES95   = [double]$Matches[1]; continue }
    if ($o.Ruin   -eq $null -and $ln -match '^\s*Ruin\s*:\s*([0-9eE\.\-]+)')    { $o.Ruin   = [double]$Matches[1]; continue }
    if ($o.MeanWT -eq $null -and $ln -match '^\s*mean_WT\s*:\s*([0-9eE\.\-]+)') { $o.MeanWT = [double]$Matches[1]; continue }
  }
  return $o
}

function Build-Tag {
  param([string]$Prefix, [string]$Profile, [string]$WinKey, [int]$Block, [double]$Hfx, [double]$Cost)
  # ex) robust_dev_w200001_201012_b24_h1_c0.002
  return ("{0}_{1}_w{2}_b{3}_h{4}_c{5}" -f $Prefix,$Profile,$WinKey,$Block,([string]$Hfx),([string]$Cost))
}

#---------------------------
# Init & grid
#---------------------------
$session   = New-StudyRoot -BaseOut $OutRoot
$StudyRoot = $session.Root
$Ts        = $session.Ts

if (-not $Windows) { $Windows = @("2000-01:2010-12","2010-01:2020-12",":") }
if (-not $Blocks)  { $Blocks  = @(12,24) }
if (-not $HFXs)    { $HFXs    = @(0,0.5,1.0) }
if (-not $Costs)   { $Costs   = @(0,0.002) }

$cases = @()
foreach($w in $Windows){ foreach($b in $Blocks){ foreach($h in $HFXs){ foreach($c in $Costs){
  $cases += [pscustomobject]@{ Window=$w; Block=$b; Hfx=$h; Cost=$c }
}}}}

Write-Host ("[ROBUST] out={0}  ts={1}  profile={2}" -f $StudyRoot,$Ts,$Profile) -ForegroundColor Cyan

if ($EstimateOnly) {
  $sec = $EstimateSecPerCase * $cases.Count
  Write-Host ("[ETA] {0} cases × {1}s ≈ {2}s (~{3} min)" -f $cases.Count,$EstimateSecPerCase,$sec,[math]::Ceiling($sec/60.0)) -ForegroundColor Yellow
  return
}

# 결과 CSV
$matrixCsv  = Join-Path $StudyRoot "robustness_matrix.csv"
$summaryCsv = Join-Path $StudyRoot "robust_summary.csv"
"ts,profile,window,block,hfx,cost,EW,ES95,Ruin,mean_WT,tag,exit_code,dur_sec,note" | Out-File -Encoding UTF8 $matrixCsv
"tag,EW,ES95,RuinPct,mean_WT,method,baseline,w_fixed" | Out-File -Encoding UTF8 $summaryCsv

#---------------------------
# Runner
#---------------------------
function Run-One {
  param([string]$Window, [int]$Block, [double]$Hfx, [double]$Cost)

  $winKey = Normalize-WindowKey $Window
  $tag    = Build-Tag -Prefix $TagPrefix -Profile $Profile -WinKey $winKey -Block $Block -Hfx $Hfx -Cost $Cost

  $args = @(
    "-m","project.runner.cli",
    "--method","rl",
    "--data_profile",$Profile,
    "--rl_epochs","0",
    "--rl_n_paths_eval","$EvalPaths",
    "--outputs",$OutRoot,
    "--tag",$tag,
    "--print_mode","summary",
    "--market_mode","bootstrap",
    "--bootstrap_block","$Block",
    "--data_window","$Window",
    "--h_FX","$Hfx",
    "--fx_hedge_cost","$Cost"
  )
  Write-Host ("[RUN] {0} {1}" -f $Python, ($args -join ' ')) -ForegroundColor Gray

  $attempt = 0
  $res = $null
  $note = ""
  do {
    $attempt++
    $res = Invoke-NativeCapture -Exe $Python -Args $args -Timeout $TimeoutSec
    if     ($res.TimedOut)          { $note = "timeout" }
    elseif ($res.ExitCode -ne 0)    { $note = "exit_$($res.ExitCode)" }
    else                            { $note = "ok" }
    if ($note -eq "ok") { break }
    if ($attempt -le $Retry) {
      Write-Host ("  -> retry {0}/{1}" -f $attempt,$Retry) -ForegroundColor Yellow
    }
  } while ($attempt -le $Retry)

  $lines = $res.Text -split "`r?`n"
  $m = Parse-Metrics $lines

  # matrix row
  $EW = if($m.EW     -ne $null){$m.EW}     else {""}
  $ES = if($m.ES95   -ne $null){$m.ES95}   else {""}
  $RU = if($m.Ruin   -ne $null){$m.Ruin}   else {""}
  $WT = if($m.MeanWT -ne $null){$m.MeanWT} else {""}
  $row = "{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13}" -f `
         $Ts,$Profile,$Window,$Block,$Hfx,$Cost,$EW,$ES,$RU,$WT,$tag,$res.ExitCode,$res.DurationSec,$note
  Add-Content -Encoding UTF8 $matrixCsv $row

  # summary row (EW/ES95 둘 다 있을 때만 기록 → 그림에서 공란 제외)
  if ($EW -ne "" -and $ES -ne "") {
    $sumRow = "{0},{1},{2},{3},{4},,,NA" -f $tag,$EW,$ES,$RU,$WT
    Add-Content -Encoding UTF8 $summaryCsv $sumRow
  }

  # per-tag metrics.csv (재활용 용이)
  try {
    $tagDir = Join-Path $OutRoot $tag
    New-Item -ItemType Directory -Force -Path $tagDir | Out-Null
    $mtext = "EW,ES95,RuinPct,mean_WT`n{0},{1},{2},{3}" -f $EW,$ES,$RU,$WT
    $mtext | Out-File -Encoding UTF8 (Join-Path $tagDir "metrics.csv")
  } catch {}
}

#---------------------------
# Sweep
#---------------------------
$idx = 0; $total = $cases.Count
foreach($c in $cases){
  $idx++
  Write-Host ("[{0}/{1}] window={2} block={3} hfx={4} cost={5}" -f $idx,$total,$c.Window,$c.Block,$c.Hfx,$c.Cost) -ForegroundColor DarkCyan
  Run-One -Window $c.Window -Block $c.Block -Hfx $c.Hfx -Cost $c.Cost
}

Write-Host "[DONE] matrix  : $matrixCsv"  -ForegroundColor Green
Write-Host "[DONE] summary : $summaryCsv" -ForegroundColor Green

#---------------------------
# (옵션) 요약 & 그림
#---------------------------
if ($MakeFigures) {
  $night = Join-Path $OutRoot ("night_robust_" + $Ts)
  New-Item -ItemType Directory -Force -Path $night | Out-Null

  # 1) 기본 clean 생성
  Copy-Item $summaryCsv (Join-Path $night "night_summary_clean.csv") -Force

  # 2) dedup 생성(동일 tag 여러 행 평균)
  $dedup = Join-Path $night "night_summary_dedup.csv"
  try {
    $py = @"
import pandas as pd, os, sys
OR = r"$night"
p = os.path.join(OR, "night_summary_clean.csv")
df = pd.read_csv(p)

# 최소 컬럼 채움(그림 스크립트 호환)
for c,d in [("method","rl"),("baseline",""),("w_fixed","NA")]:
    if c not in df.columns: df[c]=d

num_cols = [c for c in ["EW","ES95","RuinPct","mean_WT"] if c in df.columns]
if not num_cols or df.empty:
    # 빈 데이터이면 빈 dedup만 남기고 종료
    df.head(0).to_csv(os.path.join(OR,"night_summary_dedup.csv"), index=False)
    sys.exit(0)

agg = df.groupby("tag", as_index=False).agg({
    **{c:"mean" for c in num_cols},
    **{"method":"first","baseline":"first","w_fixed":"first"}
})
agg.to_csv(os.path.join(OR,"night_summary_dedup.csv"), index=False)
"@
    $py | & $Python -  | Out-Null
  } catch {}

  # 3) 그림: dedup를 일시 교체하여 플롯
  $clean = Join-Path $night "night_summary_clean.csv"
  $bak   = Join-Path $night "night_summary_clean.bak"
  if (Test-Path (Join-Path $night "night_summary_dedup.csv")) {
    Copy-Item $clean $bak -Force
    Copy-Item (Join-Path $night "night_summary_dedup.csv") $clean -Force
  }

  Write-Host "[OR] $night" -ForegroundColor Cyan
  try {
    & $Python .\scripts\make_paper_figs.py $night --paper_style on --frontier on --save_md on | Out-Host
  } catch {
    Write-Warning "make_paper_figs 실행 실패(무시 가능): $($_.Exception.Message)"
  } finally {
    if (Test-Path $bak) { Copy-Item $bak $clean -Force; Remove-Item $bak -Force }
  }
}
 