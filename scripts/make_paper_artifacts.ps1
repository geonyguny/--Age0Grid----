<# ======================================================================
 make_paper_artifacts.ps1 — 1D/2D 표·그림 자동 생성 (논문용)
 - DEF/DEV 접두사 모두 지원: -TagPrefix DEF  혹은  -TagPrefix DEV
 - 입력: outputs\{TagPrefix}_metrics_snapshot.csv (없으면 logs에서 자동 생성)
 - 출력: outputs\figs\ *.png , outputs\Paper_1D_2D_Tables.xlsx, Paper_Quick_Summary.txt
====================================================================== #>

[CmdletBinding()]
param(
  [ValidateSet('DEF','DEV')] [string]$TagPrefix = 'DEF',
  [string]$OutDir = '.\outputs',
  [string]$LogsRel = '_logs\metrics.csv',
  [string]$SummaryCsv = ''   # (선택) *_summary_benefit.csv 경로 지정 가능
)

$PY = ".\.venv\Scripts\python.exe"
$OUT = (Resolve-Path $OutDir).Path
$LOG = Join-Path $OUT $LogsRel
$SnapCsv = Join-Path $OUT ("{0}_metrics_snapshot.csv" -f $TagPrefix)

# 스냅샷 없으면 logs로부터 최신행 추출
if (-not (Test-Path $SnapCsv)) {
  if (-not (Test-Path $LOG)) {
    Write-Error "metrics.csv가 없습니다: $LOG"; exit 1
  }
  (Import-Csv $LOG |
    Where-Object { $_.tag -like "$TagPrefix*" } |
    Group-Object tag,method,sex |
    ForEach-Object { $_.Group | Select-Object -Last 1 }) |
    Export-Csv $SnapCsv -NoTypeInformation -Encoding UTF8
  Write-Host "[OK] snapshot → $SnapCsv"
}

# 파이썬 플로터 실행
$Args = @(
  ".\scripts\mk_1D_2D.py",
  "--tag_prefix", $TagPrefix,
  "--metrics_csv", $SnapCsv,
  "--outdir", (Join-Path $OUT "figs")
)
if ($SummaryCsv -ne '') { $Args += @("--summary_csv",$SummaryCsv) }

Write-Host "[RUN] plotter: $($Args -join ' ')"
& $PY @Args
if ($LASTEXITCODE -ne 0) {
  Write-Error "mk_1D_2D.py 실행 실패"; exit 1
}
Write-Host "[DONE] 논문용 표·그림 생성 완료" -ForegroundColor Green
