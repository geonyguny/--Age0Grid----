Param(
  [string]$Python = ".\.venv\Scripts\python.exe",
  [string]$OutRoot = ".\outputs",
  [string]$Profile = "dev",
  [string]$TagPrefix = "robust",
  [int]$EvalPaths = 100
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

function Invoke-NativeCapture {
  param([string]$Exe, [string[]]$Args)
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
  $out = $p.StandardOutput.ReadToEnd()
  $err = $p.StandardError.ReadToEnd()
  $p.WaitForExit()
  return @{ Text = ($out + "`n" + $err); ExitCode = $p.ExitCode }
}

function Parse-Metrics {
  param([string[]]$Lines)
  $o = [ordered]@{ EW=$null; ES95=$null; Ruin=$null }
  foreach($ln in $Lines){
    if ($o.EW   -eq $null -and $ln -match '^\s*EW\s*:\s*([0-9eE\.\-]+)'){ $o.EW   = [double]$Matches[1]; continue }
    if ($o.ES95 -eq $null -and $ln -match '^\s*ES95\s*:\s*([0-9eE\.\-]+)'){ $o.ES95 = [double]$Matches[1]; continue }
    if ($o.Ruin -eq $null -and $ln -match '^\s*Ruin\s*:\s*([0-9eE\.\-]+)'){ $o.Ruin = [double]$Matches[1]; continue }
  }
  return $o
}

$ts = Get-Date -Format "yyyyMMdd_HHmm"
$root = Join-Path $OutRoot ("_study\robustness_" + $ts)
New-Item -ItemType Directory -Force -Path $root | Out-Null
$csv = Join-Path $root "robustness_matrix.csv"
"window,block,hfx,cost,EW,ES95,Ruin,tag" | Out-File -Encoding UTF8 $csv

function Run-One {
  param([string]$Window, [int]$Block, [double]$Hfx, [double]$Cost)
  $tag = "{0}_w{1}_b{2}_h{3}_c{4}" -f $TagPrefix,$Window,$Block,$Hfx,$Cost
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
  $res = Invoke-NativeCapture -Exe $Python -Args $args
  $lines = $res.Text -split "`r?`n"
  $m = Parse-Metrics $lines
  $EW = if($m.EW -ne $null){$m.EW}else{""}
  $ES= if($m.ES95 -ne $null){$m.ES95}else{""}
  $RU= if($m.Ruin -ne $null){$m.Ruin}else{""}
  Add-Content -Encoding UTF8 $csv ("{0},{1},{2},{3},{4},{5},{6},{7}" -f $Window,$Block,$Hfx,$Cost,$EW,$ES,$RU,$tag)
}

$wins = @("2000-01:2010-12","2010-01:2020-12",":")
$blocks = @(12,24)
$hfxs = @(0,0.5,1.0)
$costs = @(0,0.002)

foreach($w in $wins){ foreach($b in $blocks){ foreach($h in $hfxs){ foreach($c in $costs){ Run-One $w $b $h $c }}}}

Write-Host "[DONE] $csv" -ForegroundColor Green
