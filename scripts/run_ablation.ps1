Param(
  [string]$Python = ".\.venv\Scripts\python.exe",
  [string]$OutRoot = ".\outputs",
  [string]$TagPrefix = "ablation",
  [string]$Profile = "dev",
  [int]$EvalPaths = 120,
  [int]$Seed = 42
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

function Invoke-NativeCapture {
  param([string]$Exe, [string[]]$Args)
  $tmpOut = [System.IO.Path]::GetTempFileName()
  $tmpErr = [System.IO.Path]::GetTempFileName()
  try {
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $Exe
    $psi.Arguments = ($Args -join ' ')
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError  = $true
    $psi.CreateNoWindow = $true
    $p = New-Object System.Diagnostics.Process
    $p.StartInfo = $psi
    [void]$p.Start()
    $stdOut = $p.StandardOutput.ReadToEnd()
    $stdErr = $p.StandardError.ReadToEnd()
    $p.WaitForExit()
    # 합쳐서 텍스트 반환
    return @{ Text = ($stdOut + "`n" + $stdErr); ExitCode = $p.ExitCode }
  } finally {
    Remove-Item $tmpOut,$tmpErr -ErrorAction SilentlyContinue
  }
}

function Parse-MetricsFromLog {
  param([string[]]$lines)
  $o = [ordered]@{EW=$null; ES95=$null; RuinPct=$null; mean_WT=$null}
  foreach ($ln in $lines) {
    if ($o.EW      -eq $null -and $ln -match '^\s*EW\s*:\s*([0-9eE\.\-]+)')       { $o.EW      = [double]$Matches[1]; continue }
    if ($o.ES95    -eq $null -and $ln -match '^\s*ES95\s*:\s*([0-9eE\.\-]+)')     { $o.ES95    = [double]$Matches[1]; continue }
    if ($o.mean_WT -eq $null -and $ln -match '^\s*mean_WT\s*:\s*([0-9eE\.\-]+)')  { $o.mean_WT = [double]$Matches[1]; continue }
    if ($o.RuinPct -eq $null -and $ln -match '^\s*Ruin\s*:\s*([0-9eE\.\-]+)')     { $o.RuinPct = [double]$Matches[1]; continue }
  }
  return $o
}

$ts = Get-Date -Format "yyyyMMdd_HHmm"
$root = Join-Path $OutRoot ("_study\" + $TagPrefix + "_" + $ts)
New-Item -ItemType Directory -Force -Path $root | Out-Null

$summaryCsv = Join-Path $root "ablation_summary.csv"
"case,EW,ES95,Ruin,tag" | Out-File -Encoding UTF8 $summaryCsv

function Ensure-MetricsFiles {
  param([string]$tag, [hashtable]$m)
  $tagDir = Join-Path $OutRoot $tag
  New-Item -ItemType Directory -Force -Path $tagDir | Out-Null
  $csvPath = Join-Path $tagDir "metrics.csv"
  $ruinVal = if ($m.RuinPct -ne $null) { $m.RuinPct } else { 0.0 }
  $row = "EW,ES95,RuinPct,mean_WT`n{0},{1},{2},{3}" -f ($m.EW),($m.ES95),($ruinVal),($m.mean_WT)
  $row | Out-File -Encoding UTF8 $csvPath
}

function Run-OneCase {
  param([string]$name, [string]$extraArgs)
  $tag = "${TagPrefix}_$name"
  $args = @(
    "-m","project.runner.cli",
    "--method","rl",
    "--data_profile",$Profile,
    "--rl_epochs","0",
    "--rl_n_paths_eval","$EvalPaths",
    "--outputs",$OutRoot,
    "--tag",$tag,
    "--print_mode","summary"
  )
  if ($extraArgs) { $args += ($extraArgs -split '\s+') }

  Write-Host "[RUN] $Python $($args -join ' ')" -ForegroundColor Cyan
  $res = Invoke-NativeCapture -Exe $Python -Args $args
  $lines = ($res.Text -split "`r?`n")
  $m = Parse-MetricsFromLog $lines

  if ($m.EW -ne $null -and $m.ES95 -ne $null) {
    Ensure-MetricsFiles -tag $tag -m $m
    $EW   = "{0:F6}" -f $m.EW
    $ES95 = "{0:F6}" -f $m.ES95
    $Ruin = "{0:F4}" -f (if ($m.RuinPct -ne $null) { $m.RuinPct } else { 0.0 })
    Add-Content -Encoding UTF8 $summaryCsv "$name,$EW,$ES95,$Ruin,$tag"
  } else {
    Write-Warning "metrics parse failed for tag=$tag. 첫 줄 몇 개: `n$($lines | Select-Object -First 10 -join "`n")"
  }
}

#  케이스 정의 
Run-OneCase -name "base" -extraArgs "--bh_on off --bias_on off"
$kappas = @(1, 2, 3)
foreach ($k in $kappas) { Run-OneCase -name ("bh_lak_{0}" -f $k) -extraArgs "--bh_on on --la_k $k --bias_on off" }
$pgs = @(0.6, 0.8, 1.0)
foreach ($g in $pgs) { Run-OneCase -name ("bias_prob_{0}" -f $g) -extraArgs "--bh_on off --bias_on on --bias_prob_gamma $g" }
Run-OneCase -name "both_la2_prob0.8" -extraArgs "--bh_on on --la_k 2 --bias_on on --bias_prob_gamma 0.8"

#  플롯 (matplotlib 없으면 스킵) 
$plotPy = Join-Path $root "plot_frontier_tmp.py"
@"
import sys
csv, out_png = sys.argv[1:3]
try:
    import pandas as pd
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception as e:
    print('[NOTE] plotting skipped:', e); sys.exit(0)
df = pd.read_csv(csv)
if len(df)==0:
    print('[NOTE] no rows to plot'); sys.exit(0)
plt.figure()
plt.scatter(df['ES95'], df['EW'])
for _, r in df.iterrows():
    plt.annotate(str(r['case']), (r['ES95'], r['EW']), fontsize=7, alpha=0.6)
plt.xlabel('ES95 (wealth, low is bad)')
plt.ylabel('EW')
plt.title('Ablation Frontier (EW vs ES95)')
plt.tight_layout()
plt.savefig(out_png, dpi=180)
"@ | Out-File -Encoding UTF8 $plotPy

$null = Invoke-NativeCapture -Exe $Python -Args @($plotPy, $summaryCsv, (Join-Path $root "eff_frontier.png"))
Remove-Item $plotPy -Force -ErrorAction SilentlyContinue

Write-Host "[DONE] $summaryCsv" -ForegroundColor Green

#  미리보기 & 폴더 열기 
if (Test-Path $summaryCsv) {
  Write-Host "`n[PREVIEW] $summaryCsv" -ForegroundColor Green
  Import-Csv $summaryCsv | Format-Table -AutoSize
  $png = Join-Path $root "eff_frontier.png"
  if (Test-Path $png) { Start-Process $png | Out-Null }
  Start-Process $root | Out-Null
}
