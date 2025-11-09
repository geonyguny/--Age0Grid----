# scripts/rules_compare.ps1
param(
  [string]$Root = "D:\01_simul",
  [string]$Out  = ".\outputs",
  [string]$Csv  = "D:\01_simul\project\data\market\kr_us_gold_bootstrap_mini.csv",
  [string]$Mort = "D:\01_simul\project\data\kidi_qx.csv",
  [switch]$WithAnnuity = $true,
  [switch]$WithMortality = $true
)

$ErrorActionPreference = "Stop"
Set-Location $Root
$OutputEncoding = [Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$MET = Join-Path $Out "_logs\metrics.csv"
if (-not (Test-Path (Split-Path $MET))) { New-Item -ItemType Directory -Force -Path (Split-Path $MET) | Out-Null }

$baselines = @("fourpct","cpb","vpw","kgr")
$tags = @()
foreach ($b in $baselines) {
  $tag = ("RULE_{0}{1}{2}" -f $b.ToUpper(),
    $(if ($WithMortality) { "_MORT" } else { "" }),
    $(if ($WithAnnuity)   { "_ANN"  } else { "" }))
  $tags += $tag

  Write-Host ">> $tag" -ForegroundColor Cyan
  $cmd = @(
    "py","-m","project.runner.cli",
    "--asset","KR","--method","rule","--baseline",$b,
    "--market_mode","bootstrap","--market_csv",$Csv,"--use_real_rf","on",
    "--outputs",$Out,"--bands","on","--n_paths","200","--seeds","0",
    "--es_mode","wealth","--tag",$tag,"--print_mode","summary","--quiet","on"
  )

  if ($WithMortality) { $cmd += @("--mortality","on","--mort_table",$Mort) }
  if ($WithAnnuity)   { $cmd += @("--ann_on","on","--ann_alpha","0.3","--ann_index","real") }

  & $cmd
}

Write-Host "`n=== RULES SUMMARY ===" -ForegroundColor Yellow
if (Test-Path $MET) {
  Import-Csv $MET |
    Where-Object { $_.tag -in $tags } |
    Sort-Object ts |
    Format-Table ts,tag,EW,ES95,AlivePathRate,y_ann,a_factor,P -AutoSize
} else {
  Write-Host "[WARN] metrics.csv not found: $MET" -ForegroundColor DarkYellow
}
