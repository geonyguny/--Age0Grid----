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
New-Item -ItemType Directory -Force -Path $OutRoot,$LogDir,$FigDir | Out-Null

# 로그 시작
$ts = Get-Date -Format 'yyyyMMdd_HHmmss'
$Transcript = Join-Path $OutRoot ("run_all_{0}_{1}.log" -f $Mode,$ts)
Start-Transcript -Path $Transcript -Force | Out-Null
Write-Host ">>> RUN_ALL START  Mode=$Mode   $(Get-Date)" -ForegroundColor Cyan

# ── 모드별 파라미터 ─────────────────────────────────────
switch ($Mode) {
  'dev' {
    $profile    = 'dev'
    $nPathsRL   = 2000
    $nPathsHJB  = 2000
    $Seeds      = @(11)
    $TagPrefix  = 'DEV'
  }
  'overnight' {
    $profile    = 'full'
    # RL 경로 수는 환경에 맞게 여기서만 조정
    $nPathsRL   = 8000     # (너무 커서 병목이면 8k~15k 권장)
    $nPathsHJB  = 30000
    $Seeds      = @(11,21,31)
    $TagPrefix  = 'OVN'
  }
}

Write-Host "[PROFILE] $profile  [SEEDS] $($Seeds -join ',')  [RL n_paths] $nPathsRL  [HJB n_paths] $nPathsHJB" -ForegroundColor DarkCyan

function RunPy([string]$title, [string[]]$argv) {
  Write-Host ">> $title" -ForegroundColor Cyan
  & $Py @argv
  if ($LASTEXITCODE -ne 0) { throw "FAILED: $title (exit=$LASTEXITCODE)" }
}

try {
  # ── 1) RL: OAT (hedge_sigma_k sweep) ───────────────────
  $Hs = @(0.0, 0.5, 1.0)
  foreach ($seed in $Seeds) {
    foreach ($h in $Hs) {
      $tag = "{0}_OAT_h{1}" -f $TagPrefix, $h
      $args = @(
        '-m','project.runner.cli',
        '--method','rl','--data_profile',$profile,'--market_mode','bootstrap',
        '--hedge','on','--hedge_mode','sigma','--hedge_sigma_k',"$h",
        '--n_paths',"$nPathsRL",'--seed',"$seed",'--tag',$tag,
        '--print_mode','summary','--autosave','on'
      )
      RunPy ("RL OAT h={0} seed={1}" -f $h,$seed) $args
    }
  }

  # ── 2) HJB: 2D (mix_us × hedge_sigma_k) ────────────────
  $Pairs = @('0.2,0.0','0.2,0.5','0.6,0.0','0.6,0.5')
  foreach ($seed in $Seeds) {
    foreach ($pair in $Pairs) {
      $sp  = $pair.Split(',')
      $u   = [double]$sp[0]
      $h   = [double]$sp[1]
      $mix = "0.0,$u," + (1.0 - $u)   # alpha_mix = (kr,us,gold)
      $tag = "{0}_2D_us{1}_h{2}" -f $TagPrefix, $u, $h
      $args = @(
        '-m','project.runner.cli',
        '--method','hjb','--data_profile',$profile,'--market_mode','bootstrap',
        '--alpha_mix',$mix,'--hedge','on','--hedge_mode','sigma','--hedge_sigma_k',"$h",
        '--n_paths',"$nPathsHJB",'--seed',"$seed",'--tag',$tag,
        '--print_mode','summary','--autosave','on'
      )
      RunPy ("HJB 2D us={0} h={1} seed={2}" -f $u,$h,$seed) $args
    }
  }

  # ── 3) 스냅샷 작성(히트맵/라인의 소스로 사용) ─────────
  $SnapCsv = Join-Path $OutRoot ("{0}_metrics_snapshot.csv" -f $TagPrefix)
  $ps = @"
`$m = Import-Csv '$LogDir\metrics.csv' |
  Where-Object { `$_.tag -like '${TagPrefix}_OAT_*' -or `$_.tag -like '${TagPrefix}_2D_*' } |
  Group-Object tag,method,seed | ForEach-Object { `$_.Group | Select-Object -Last 1 }
`$m | Export-Csv '$SnapCsv' -NoTypeInformation
"@
  powershell -NoProfile -Command $ps
  Write-Host "[OK] Snapshot -> $SnapCsv" -ForegroundColor Green

  # ── 4) 시각화 ──────────────────────────────────────────
  # OAT 라인 (스냅샷 기반)
  RunPy 'OAT lines' @(
    '.\scripts\make_oat_lines.py',
    '--src',$SnapCsv,
    '--tag_startswith',("${TagPrefix}_OAT_"),
    '--xcol','hedge_sigma_k','--metrics','EW,ES95'
  )

  # 2D 히트맵 (스냅샷 기반, median + annotate + pivot 저장)
  RunPy '2D heatmaps (median)' @(
    '.\scripts\make_heatmaps.py',
    '--src',$SnapCsv,
    '--tag_startswith',("${TagPrefix}_2D_"),
    '--x','mix_us','--y','hedge_sigma_k',
    '--zlist','EW,ES95,CompositeScore',
    '--annotate','on','--agg','median','--save_pivots','on',
    '--dpi','180','--fig_w','6.6','--fig_h','4.8'
  )

  # ── 5) 리포트 ──────────────────────────────────────────
  $ReportTag = "${TagPrefix}_report"
  RunPy ("Make report ({0})" -f $ReportTag) @(
    '.\scripts\make_decum_report.py',
    '--tag',$ReportTag,'--frontier_topn','50'
  )

  # ── 6) 드리프트 체크(dev↔full, wealth) ────────────────
  # overnight 모드에선 의미가 덜할 수 있지만 일관성 있게 실행
  RunPy 'Compare dev vs full (wealth)' @(
    '.\scripts\compare_dev_full.py',
    '--tag_startswith',("${TagPrefix}_2D_"),
    '--method','hjb','--es_mode','wealth',
    '--abs_tol','0.03','--rel_tol','0.04',
    '--round_mix','2','--round_h','2',
    '--include_profile_in_key','on'
  )

  Write-Host ">>> DONE. Mode=$Mode  |  Snapshot=$SnapCsv" -ForegroundColor Green
}
catch {
  Write-Host "[ERR] $($_.Exception.Message)" -ForegroundColor Red
  throw
}
finally {
  Stop-Transcript | Out-Null
}
