<# ======================================================================
  scripts\run_all.ps1
  - Mode dev : 빠른 개발 검증 (경로수/seed 축소)
  - Mode overnight : 밤샘 대량 (full 프로파일, 경로수 확대)
  산출물:
    - outputs\_logs\metrics.csv (러너가 갱신)
    - outputs\figs\heatmap_*.png 및 *_pivot.csv
    - outputs\Paper_Decum_Report_<tag>.xlsx
    - outputs\dev_full_drift.xlsx
    - outputs\run_all_<timestamp>.log (Transcript)
====================================================================== #>

param(
  [ValidateSet('dev','overnight')]
  [string]$Mode = 'dev'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# ── 경로/환경 ────────────────────────────────────────────
$ScriptDir   = $PSScriptRoot
$ProjectRoot = Split-Path -Parent $ScriptDir
Set-Location $ProjectRoot

$Py      = '.\.venv\Scripts\python.exe'
$OutRoot = '.\outputs'
$LogDir  = Join-Path $OutRoot '_logs'
$FigDir  = Join-Path $OutRoot 'figs'
$SumCsv  = Join-Path $OutRoot '_summary_scored.csv'    # 시각화/드리프트에서 사용
$Metrics = Join-Path $LogDir  'metrics.csv'

New-Item -ItemType Directory -Force -Path $OutRoot,$LogDir,$FigDir | Out-Null

# ── 로깅 시작 ───────────────────────────────────────────
$ts = Get-Date -Format 'yyyyMMdd_HHmmss'
$Transcript = Join-Path $OutRoot ("run_all_{0}_{1}.log" -f $Mode,$ts)
Start-Transcript -Path $Transcript -Force | Out-Null
Write-Host ">>> RUN_ALL START  Mode=$Mode   $(Get-Date)" -ForegroundColor Green

# ── 공용 함수 ───────────────────────────────────────────
function Ensure-Exe([string]$PathOrExe) {
  if (-not (Test-Path $PathOrExe)) {
    throw "실행 파일을 찾을 수 없습니다: $PathOrExe"
  }
}
function RunPy([string]$title, [string[]]$argv) {
  Write-Host ">> $title" -ForegroundColor Cyan
  & $Py @argv
  if ($LASTEXITCODE -ne 0) { throw "FAILED: $title (exit=$LASTEXITCODE)" }
}
function Safe-Exists([string]$path) { return (Test-Path $path) }

Ensure-Exe $Py

# ── 모드별 파라미터 ─────────────────────────────────────
switch ($Mode) {
  'dev' {
    $profile    = 'dev'
    $nPathsRL   = 2000
    $nPathsHJB  = 2000
    $Seeds      = @(11)
    $TagPrefix  = 'DEV'
    $ReportTag  = 'dev_minset'
  }
  'overnight' {
    $profile    = 'full'
    $nPathsRL   = 8000    # RL은 8k, HJB 30k (성능/시간 밸런스)
    $nPathsHJB  = 30000
    $Seeds      = @(11,21,31)
    $TagPrefix  = 'OVN'
    $ReportTag  = 'overnight_full'
  }
}

# ── 0) 사전 안내 ────────────────────────────────────────
Write-Host "[PROFILE] $profile  [SEEDS] $($Seeds -join ',')  [RL n_paths] $nPathsRL  [HJB n_paths] $nPathsHJB" -ForegroundColor DarkGray

# ── 1) RL: OAT (hedge_sigma_k sweep) ─────────────────────
try {
  $Hs = @(0.0, 0.5, 1.0)
  foreach ($seed in $Seeds) {
    foreach ($h in $Hs) {
      $tag = "${TagPrefix}_OAT_h$h"
      RunPy "RL OAT h=$h seed=$seed" @(
        '-m','project.runner.cli',
        '--method','rl','--data_profile',$profile,'--market_mode','bootstrap',
        '--hedge','on','--hedge_mode','sigma','--hedge_sigma_k',"$h",
        '--n_paths',"$nPathsRL",'--seed',"$seed",'--tag',$tag,
        '--print_mode','summary','--autosave','on'
      )
    }
  }
} catch { Write-Host "[ERR] RL OAT: $($_.Exception.Message)"; throw }

# ── 2) HJB: 2D (mix_us × hedge_sigma_k) ──────────────────
try {
  $Pairs = @('0.2,0.0','0.2,0.5','0.6,0.0','0.6,0.5')
  foreach ($seed in $Seeds) {
    foreach ($pair in $Pairs) {
      $sp  = $pair.Split(',')
      $u   = [double]$sp[0]
      $h   = [double]$sp[1]
      $mix = "0.0,$u," + (1.0 - $u)   # alpha_mix = (kr,us,gold)
      $tag = "{0}_2D_us{1}_h{2}" -f $TagPrefix, $u, $h
      RunPy "HJB 2D us=$u h=$h seed=$seed" @(
        '-m','project.runner.cli',
        '--method','hjb','--data_profile',$profile,'--market_mode','bootstrap',
        '--alpha_mix',$mix,'--hedge','on','--hedge_mode','sigma','--hedge_sigma_k',"$h",
        '--n_paths',"$nPathsHJB",'--seed',"$seed",'--tag',$tag,
        '--print_mode','summary','--autosave','on'
      )
    }
  }
} catch { Write-Host "[ERR] HJB 2D: $($_.Exception.Message)"; throw }

# ── 3) 메트릭 스냅샷 (태그별 최신 레코드) ───────────────
try {
  if (Safe-Exists $Metrics) {
    $SnapCsv = Join-Path $OutRoot 'dev_metrics_snapshot.csv'
    $m = Import-Csv $Metrics |
      Where-Object { $_.tag -like "${TagPrefix}_OAT_*" -or $_.tag -like "${TagPrefix}_2D_*" } |
      Group-Object tag,method,seed | ForEach-Object { $_.Group | Select-Object -Last 1 }
    $m | Export-Csv $SnapCsv -NoTypeInformation
    Write-Host "[OK] Snapshot -> $SnapCsv"
  } else {
    Write-Host "[WARN] metrics.csv가 없어 스냅샷 생략"
  }
} catch { Write-Host "[ERR] Snapshot: $($_.Exception.Message)"; throw }

# ── 4) 시각화 ────────────────────────────────────────────
try {
  # OAT 라인 (RL)
  RunPy 'OAT lines' @(
    '.\scripts\make_oat_lines.py',
    '--tag_startswith',("${TagPrefix}_OAT_"),
    '--xcol','hedge_sigma_k','--metrics','EW,ES95'
  )

  # 2D 히트맵 (pivot CSV 저장 포함)
  RunPy '2D heatmaps (mean)' @(
    '.\scripts\make_heatmaps.py',
    '--tag_startswith',("${TagPrefix}_2D_"),
    '--x','mix_us','--y','hedge_sigma_k',
    '--zlist','EW,ES95,CompositeScore',
    '--agg','mean','--annotate','off','--save_pivots','on',
    '--dpi','180','--fig_w','6.6','--fig_h','4.8',
    '--vmin_max','ES95:0.25,1.0;EW:0.25,1.25;CompositeScore:-0.6,1.9'
  )
} catch { Write-Host "[ERR] Visualization: $($_.Exception.Message)"; throw }

# ── 5) 리포트 ────────────────────────────────────────────
try {
  RunPy "Make report ($ReportTag)" @(
    '.\scripts\make_decum_report.py','--tag',$ReportTag,'--frontier_topn','50'
  )
} catch { Write-Host "[ERR] Report: $($_.Exception.Message)"; throw }

# ── 6) 드리프트 체크(dev↔full) ──────────────────────────
try {
  RunPy 'Compare dev vs full (wealth/HJB/2D)' @(
    '.\scripts\compare_dev_full.py',
    '--tag_startswith',("${TagPrefix}_2D_"),
    '--method','hjb','--es_mode','wealth',
    '--abs_tol','0.03','--rel_tol','0.04','--round_mix','2','--round_h','2',
    '--include_profile_in_key','on'
  )
} catch { Write-Host "[ERR] Drift: $($_.Exception.Message)"; throw }

Write-Host ">>> DONE. Mode=$Mode | Transcript=$Transcript" -ForegroundColor Green
Stop-Transcript | Out-Null
