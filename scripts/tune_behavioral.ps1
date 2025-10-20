# scripts/tune_behavioral.ps1
param(
  [string]$Root = "D:\01_simul",
  [string]$Out  = ".\outputs",
  [string]$Csv  = "D:\01_simul\project\data\market\kr_us_gold_bootstrap_mini.csv"
)

$ErrorActionPreference = "Stop"
Set-Location $Root
$OutputEncoding = [Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$MET = Join-Path $Out "_logs\metrics.csv"
if (-not (Test-Path (Split-Path $MET))) { New-Item -ItemType Directory -Force -Path (Split-Path $MET) | Out-Null }

Write-Host "=== Behavioral utility sweep ===" -ForegroundColor Cyan
# 빠른 스윕(필요시 값 추가/수정)
$u_scales = @(0.02, 0.05)
$gammas   = @(2.0, 3.0)
$betas    = @(0.97, 0.99)
$habits   = @(0.0, 0.1)

foreach ($u in $u_scales) {
  foreach ($g in $gammas) {
    foreach ($b in $betas) {
      foreach ($h in $habits) {
        $tag = ("BU_u{0}_g{1}_b{2}_h{3}" -f $u, $g, $b, $h).Replace(".","p")
        Write-Host ">> $tag" -ForegroundColor Yellow
        py -m project.runner.cli `
          --asset KR --method hjb `
          --market_mode bootstrap --market_csv $Csv --use_real_rf on `
          --outputs $Out --bands on --n_paths 120 --seeds 0 `
          --es_mode wealth --report_utility on --crra_gamma $g --u_scale $u `
          --bh_on on --beta $b --habit_phi $h `
          --tag $tag --print_mode summary --quiet on
      }
    }
  }
}

Write-Host "`n=== Summary (latest 20 rows) ===" -ForegroundColor Green
if (Test-Path $MET) {
  Get-Content -Path $MET -Tail 20 | Write-Host
  "`n-- Filter by BU_* --" | Write-Host
  Import-Csv $MET |
    Where-Object { $_.tag -like "BU_*" } |
    Sort-Object ts |
    Format-Table ts,tag,EW,ES95,EU,EU_per_year,crra_gamma,u_scale -AutoSize
} else {
  Write-Host "[WARN] metrics.csv not found: $MET" -ForegroundColor DarkYellow
}
