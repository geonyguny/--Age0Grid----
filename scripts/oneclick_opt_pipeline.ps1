<# =====================================================================
 oneclick_opt_pipeline.ps1  (ALL-IN-ONE for paper; PS5+ safe)
 - Stage A: run_opt_design.ps1 (design experiments)
 - Stage B: snapshot & score (idempotent)
 - Stage C: figures (user scripts if present; else fallback: make_paper_figs.py oat-heatmap)
 - Stage D: tables (user script preferred; else fallback path)
 - Stage E: optional PPT

 Notes
 - ASCII-only console text to avoid encoding glitches
 - Guarded Test-Path with parentheses to avoid '-and' parsing issues
 - Uses snapshot CSV for heatmaps
===================================================================== #>

[CmdletBinding()]
param(
  [ValidateSet('defense','full')] [string]$Mode = 'defense',
  [ValidateSet('dev','full')]     [string]$DataProfile = 'dev',

  [string]$Seeds    = '7',
  [string]$PWGammas = '0.85,0.80,0.70,0.60',

  # KR monthly minimum living cost (KRW) - KIHSA 2024 median individual
  [double]$KR_MinLiving_Monthly = 1361000,

  # Initial real wealth (KRW)
  [double]$InitWealth_Real = 5e8,

  # Post steps
  [bool]$DoFigs   = $true,
  [bool]$DoTables = $true,
  [bool]$DoPPT    = $false
)

# ---------- Paths ----------
$Root = (Get-Location).Path
$Out  = Join-Path $Root 'outputs'
$Logs = Join-Path $Out  '_logs'
$Figs = Join-Path $Out  'figs'
$null = New-Item -ItemType Directory -Force -Path $Out,$Logs,$Figs | Out-Null

Write-Host "[OneClick] Starting pipeline"
Write-Host (" - Mode         : {0}" -f $Mode)
Write-Host (" - Profile      : {0}" -f ($(if ($Mode -eq 'defense') {'light'} else {'heavy'})))
Write-Host (" - DataProfile  : {0}" -f $DataProfile)
Write-Host (" - Seeds        : {0}" -f $Seeds)
Write-Host (" - PWGammas     : {0}" -f $PWGammas)
Write-Host (" - Floor (KRW/m): {0}" -f $KR_MinLiving_Monthly.ToString('N0'))
Write-Host (" - InitWealth   : {0}" -f $InitWealth_Real.ToString('N0'))
Write-Host (" - ProjectRoot  : {0}" -f $Root)
Write-Host (" - Outputs      : {0}" -f $Out)
Write-Host (" - Logs         : {0}" -f $Logs)
Write-Host (" - Figs         : {0}" -f $Figs)
Write-Host "[Guard] Close Excel files to avoid PermissionError on CSV/XLSX."

# Common executables
$PyPath = Join-Path $Root ".venv\Scripts\python.exe"
$ScorePy = Join-Path $Root "scripts\score_snapshot.py"
$FigFallbackPy = Join-Path $Root "scripts\make_paper_figs.py"
$UserHeatA = Join-Path $Root "scripts\figs\make_2d_heatmaps.py"
$UserHeatB = Join-Path $Root "scripts\make_2d_heatmaps.py"
$UserCurveA = Join-Path $Root "scripts\figs\make_1d_curves.py"
$UserCurveB = Join-Path $Root "scripts\make_1d_curves.py"
$UserTablesA = Join-Path $Root "scripts\figs\make_paper_tables.py"
$UserTablesB = Join-Path $Root "scripts\make_paper_tables.py"

# ---------- Stage A: run_opt_design ----------
$profile = if ($Mode -eq 'defense') {'light'} else {'heavy'}
$runDesign = Join-Path $Root 'scripts\run_opt_design.ps1'
$runArgs = @(
  '-Profile', $profile,
  '-DataProfile', $DataProfile,
  '-Seeds', $Seeds,
  '-KR_MinLiving_Monthly', ("{0}" -f $KR_MinLiving_Monthly),
  '-InitWealth_Real',      ("{0}" -f $InitWealth_Real),
  '-PWGammas', $PWGammas
)

if (Test-Path $runDesign) {
  Write-Host "[OneClick] Invoke: .\scripts\run_opt_design.ps1 $($runArgs -join ' ')"
  & $runDesign @runArgs
  if ($LASTEXITCODE -ne 0) {
    Write-Warning ("[OneClick] run_opt_design.ps1 returned non-zero exit code: {0}" -f $LASTEXITCODE)
  }
} else {
  Write-Warning "[OneClick] scripts\run_opt_design.ps1 not found. Skipping Stage A."
}

# ---------- Stage B: snapshot & score ----------
$metricsCsv = Join-Path $Logs 'metrics.csv'
$snapCsv    = Join-Path $Out  'DEV_metrics_snapshot.csv'

if (Test-Path $metricsCsv) {
  Write-Host ("[OneClick] Snapshot latest rows -> {0}" -f $snapCsv)
  (Import-Csv $metricsCsv |
    Group-Object tag,method,sex | ForEach-Object { $_.Group | Select-Object -Last 1 }) |
    Export-Csv $snapCsv -NoTypeInformation -Encoding UTF8

  if ((Test-Path $PyPath) -and (Test-Path $ScorePy)) {
    Write-Host "[OneClick] Scoring snapshot (benefit weights inside scorer)."
    & $PyPath $ScorePy `
      --src $snapCsv `
      --out_prefix $Out `
      2>&1 | Out-Null
  } else {
    Write-Warning "[OneClick] score_snapshot.py not found or Python venv missing."
  }
} else {
  Write-Warning ("[OneClick] metrics.csv not found: {0}" -f $metricsCsv)
}

# ---------- Stage C: Figures ----------
if ($DoFigs) {
  $userHeat = $null
  if (Test-Path $UserHeatA) { $userHeat = $UserHeatA }
  elseif (Test-Path $UserHeatB) { $userHeat = $UserHeatB }

  $userCurves = $null
  if (Test-Path $UserCurveA) { $userCurves = $UserCurveA }
  elseif (Test-Path $UserCurveB) { $userCurves = $UserCurveB }

  if (Test-Path $PyPath) {
    if ($userHeat) {
      Write-Host ("[OneClick] Heatmaps via user script: {0}" -f $userHeat)
      & $PyPath $userHeat --logs $Logs --out $Figs 2>&1 | Out-Null
    }
    if ($userCurves) {
      Write-Host ("[OneClick] 1D curves via user script: {0}" -f $userCurves)
      & $PyPath $userCurves --logs $Logs --out $Figs 2>&1 | Out-Null
    }

    if (-not $userHeat -and -not $userCurves) {
      if (Test-Path $FigFallbackPy) {
        # Fallback uses our refactored make_paper_figs.py (oat-heatmap subcommand)
        Write-Host "[OneClick] Figures via fallback (make_paper_figs.py oat-heatmap)."
        if (Test-Path $snapCsv) {
          & $PyPath $FigFallbackPy oat-heatmap `
            --src $snapCsv `
            --outdir $Figs `
            --tag_startswith 'DEV2D_' `
            --x 'mix_us' --y 'hedge_sigma_k' `
            --zlist 'EW,ES95,CompositeScore' `
            --agg 'median' --annotate 'on' `
            --dpi 180 --fig_w 6.6 --fig_h 4.8 `
            --cmap 'viridis' `
            --method 'rl' --es_mode 'wealth' `
            2>&1 | Out-Null
        } else {
          Write-Warning "[OneClick] Snapshot not found; skip fallback figures."
        }
      } else {
        Write-Warning "[OneClick] No figure script found (user/fallback)."
      }
    }
  } else {
    Write-Warning "[OneClick] Python venv not found; skip figures."
  }
}

# ---------- Stage D: Tables ----------
if ($DoTables) {
  $paperXlsx = Join-Path $Out 'Paper_1D_2D_Tables.xlsx'
  $userTables = $null
  if (Test-Path $UserTablesA) { $userTables = $UserTablesA }
  elseif (Test-Path $UserTablesB) { $userTables = $UserTablesB }

  if (Test-Path $PyPath) {
    if ($userTables) {
      Write-Host ("[OneClick] Tables via user script: {0}" -f $userTables)
      & $PyPath $userTables --outputs $Out --dest $paperXlsx 2>&1 | Out-Null
    } elseif (Test-Path $UserTablesA) {
      Write-Host "[OneClick] Tables via fallback (scripts\figs\make_paper_tables.py)."
      & $PyPath $UserTablesA --outputs $Out --dest $paperXlsx 2>&1 | Out-Null
    } else {
      Write-Warning "[OneClick] No table script found (user/fallback)."
    }

    if (Test-Path $paperXlsx) {
      Write-Host ("Output: {0}" -f $paperXlsx)
    }
  } else {
    Write-Warning "[OneClick] Python venv not found; skip tables."
  }
}

# ---------- Stage E: PPT (optional) ----------
if ($DoPPT) {
  $UserPptA = Join-Path $Root "scripts\figs\make_paper_ppt.py"
  $UserPptB = Join-Path $Root "scripts\make_paper_ppt.py"
  $userPPT = $null
  if (Test-Path $UserPptA) { $userPPT = $UserPptA }
  elseif (Test-Path $UserPptB) { $userPPT = $UserPptB }

  if ($userPPT -and (Test-Path $PyPath)) {
    Write-Host ("[OneClick] PPT via user script: {0}" -f $userPPT)
    & $PyPath $userPPT --outputs $Out --figs $Figs 2>&1 | Out-Null
  } else {
    Write-Host "[OneClick] PPT step skipped."
  }
}

Write-Host "[OneClick] Completed."
Write-Host ("Artifacts:")
Write-Host (" - Snapshots: {0}\*_metrics_snapshot.csv" -f $Out)
Write-Host (" - Summaries: {0}\*_summary*.csv" -f $Out)
Write-Host (" - Core tables: {0}\OPT_table_core.csv, {0}\BH_table_core.csv" -f $Out)
Write-Host (" - Compare/top: {0}\OPT_BH_compare.csv, {0}\OPT_BH_top.csv" -f $Out)
Write-Host (" - Figures: {0} (heatmaps, bars, scatter, etc.)" -f $Figs)
Write-Host (" - Paper table: {0}\Paper_1D_2D_Tables.xlsx" -f $Out)
