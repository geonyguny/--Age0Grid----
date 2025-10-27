Param()

$ErrorActionPreference = "Stop"
$ProgressPreference    = "SilentlyContinue"

# 0) 보조: 로그→CSV 1행 추가
function Add-LogRow {
  param([Parameter(Mandatory=$true)][string]$Tag)

  $log = Join-Path .\outputs\_logs "$Tag.log"
  if (!(Test-Path $log)) { Write-Warning "로그 없음: $log"; return }

  $txt = Get-Content $log -Raw
  function Grab([string]$name){
    $m = [regex]::Match($txt, "(?m)^\s*$name\s*:\s*([0-9eE\.\-]+)")
    if ($m.Success) { $m.Groups[1].Value } else { "" }
  }

  $EW =  (Grab "EW");   $ES = (Grab "ES95")
  $RU =  (Grab "Ruin"); $WT = (Grab "mean_WT")
  if (($EW -eq "") -and ($ES -eq "") -and ($RU -eq "") -and ($WT -eq "")) {
    Write-Warning "메트릭을 찾지 못함: $Tag"
    return
  }

  $csv = ".\outputs\p0_summary.csv"
  if (!(Test-Path $csv)) {
    "tag,EW,ES95,RuinPct,mean_WT,method,baseline,w_fixed" | Out-File -Encoding UTF8 $csv
  }

  $low = $Tag.ToLower()
  $method = if ($low -match "vpw") { "rule" } elseif ($low -match "hjb") { "hjb" } else { "rl" }
  $baseline = if ($low -match "vpw") { "vpw" } else { "" }

  $row = "{0},{1},{2},{3},{4},{5},{6},{7}" -f $Tag,$EW,$ES,$RU,$WT,$method,$baseline,"NA"
  Add-Content -Encoding UTF8 $csv $row
  Write-Host "[p0] appended: $Tag"
}

# 1) 전체 재구축
function Rebuild-P0Summary {
  $outCsv = ".\outputs\p0_summary.csv"
  "tag,EW,ES95,RuinPct,mean_WT,method,baseline,w_fixed" | Out-File -Encoding UTF8 $outCsv

  $logs = Get-ChildItem .\outputs\_logs\*.log -ErrorAction SilentlyContinue
  foreach ($f in $logs) {
    $tag = [System.IO.Path]::GetFileNameWithoutExtension($f.Name)
    Add-LogRow $tag
  }

  $n = (Get-Content $outCsv | Measure-Object -Line).Lines
  Write-Host "[p0] rebuilt: $outCsv  rows=$(($n-1))"
}

# 2) 숫자 무결성 검사 + 정렬 미리보기
function Validate-And-Preview {
  $csv = ".\outputs\p0_summary.csv"
  if (!(Test-Path $csv)) { throw "p0_summary.csv 없음" }

  # 강제 헤더로 로드(헤더 꼬임 방지)
  $hdr = 'tag,EW,ES95,RuinPct,mean_WT,method,baseline,w_fixed'.Split(',')
  $rows = (Get-Content $csv | Select-Object -Skip 1) | ConvertFrom-Csv -Header $hdr

  # 숫자 파싱 체크
  foreach ($r in $rows) {
    [void][double]$r.EW; [void][double]$r.ES95; [void][double]$r.RuinPct; [void][double]$r.mean_WT
  }
  "OK: numeric parse passed" | Write-Host

  # 정렬 출력
  $rows | Sort-Object @{e='ES95';a=$true}, @{e='EW';a=$false} |
    Format-Table -AutoSize | Out-Host
}

# 3) 스냅샷 저장
function Snapshot-P0 {
  $stamp = Get-Date -Format "yyyyMMdd_HHmm"
  New-Item -ItemType Directory -Force -Path .\outputs\_snapshots | Out-Null
  Copy-Item .\outputs\p0_summary.csv .\outputs\_snapshots\p0_summary_$stamp.csv -Force
  Write-Host "[p0] snapshot saved: outputs/_snapshots/p0_summary_$stamp.csv"
}

# 실행 순서
Rebuild-P0Summary
Validate-And-Preview
Snapshot-P0
