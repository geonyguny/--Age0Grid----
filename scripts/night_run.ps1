# =========[ 공통 세팅 ]=========
$PY     = ".\.venv\Scripts\python.exe"
$MODULE = "project.runner.cli"

# 베이스 파라미터 (DONE v1 지침 기본값)
$BASE = @(
  "--method","hjb",
  "--market_mode","iid",
  "--horizon_years","35",
  "--alpha","0.95",
  "--beta","0.996",
  "--lambda_term","0.8",
  "--w_max","0.70",
  "--q_floor","0.02",
  "--phi_adval","0.004",
  "--autosave","on",
  "--quiet","on"
)

# 오버나이트용 런타임(시간 충분히)
$SEEDS   = 5       # 경계구간은 10으로 재런(맨 아래 bootstrap/redo 섹션 참고)
$NPATHS  = 2000    # 기계 상황 맞게 조절 (빠르게: 1000, 안정: 3000~5000 권장)

# 출력 루트
$OUTROOT = ".\outputs\night"

# =========[ 헬퍼 ]=========
function Invoke-Run {
  param(
    [string]$Tag,
    [string]$OutDir,
    [string[]]$ExtraArgs = @()
  )

  # 이미 완료된 태그는 스킵 (metrics.csv 존재 시)
  $metricsFile = Join-Path $OutDir "_logs\metrics.csv"
  if (Test-Path $metricsFile) {
    Write-Host "[SKIP] $Tag (metrics.csv exists)" -ForegroundColor Yellow
    return
  }

  New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

  # 스톱워치로 개별/누적 시간 표시
  $sw = [System.Diagnostics.Stopwatch]::StartNew()

  & $PY --% -m $MODULE `
     --seeds $SEEDS `
     --n_paths $NPATHS `
     --outputs $OutDir `
     --tag $Tag `
     @BASE @ExtraArgs

  $sw.Stop()
  Write-Host ("[DONE] {0}  took={1:c}" -f $Tag, $sw.Elapsed) -ForegroundColor Green
}

# =========[ 0) 워밍업 (JIT/캐시/초기화 편익) ]=========
& $PY --% -m $MODULE `
  --method hjb --market_mode iid --seeds 1 --n_paths 50 `
  --horizon_years 35 --alpha 0.95 --beta 0.996 --lambda_term 0.8 `
  --w_max 0.70 --q_floor 0.02 --phi_adval 0.004 `
  --w_fixed 0.5 `
  --outputs "$OUTROOT\_warmup" `
  --tag "warmup" --autosave off --quiet off

# =========[ 1) 단독효과 스윕 ]=========
# 1-1) 성별
$levels_sex = @("M","F")
foreach($sex in $levels_sex){
  Invoke-Run -Tag "sex_$sex" `
    -OutDir "$OUTROOT\sex" `
    -ExtraArgs @("--sex",$sex)
}

# 1-2) 인출 개시 연령
$levels_age0 = 55..65
foreach($age in $levels_age0){
  Invoke-Run -Tag "age0_$age" `
    -OutDir "$OUTROOT\age0" `
    -ExtraArgs @("--age0",$age)
}

# 1-3) 위험자산 고정비중 w_fixed
$levels_wfixed = @(0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0)
foreach($w in $levels_wfixed){
  Invoke-Run -Tag ("wfixed_{0:N1}" -f $w) `
    -OutDir "$OUTROOT\wfixed" `
    -ExtraArgs @("--w_fixed",$w)
}

# 1-4) 종신연금 가입비율(가정: --ann_on on + 비율 옵션; 프로젝트 옵션 명에 맞게 사용)
$levels_ann = @(0,25,50,75)
foreach($a in $levels_ann){
  Invoke-Run -Tag ("ann_{0}pct" -f $a) `
    -OutDir "$OUTROOT\annuity" `
    -ExtraArgs @("--ann_on","on","--ann_alpha","1.0","--ann_L",$a)    # ← 옵션명 프로젝트에 맞추어 조정
}

# 1-5) FX 헤지 비율(가정: --fx_hedge_ratio)
$levels_fx = @(0.0,0.5,1.0)
foreach($r in $levels_fx){
  Invoke-Run -Tag ("fxhedge_{0:N1}" -f $r) `
    -OutDir "$OUTROOT\fxhedge" `
    -ExtraArgs @("--fx_hedge_ratio",$r)
}

# 1-6) 수수료율 phi_adval
$levels_fee = @(0.002,0.004,0.006)
foreach($fee in $levels_fee){
  Invoke-Run -Tag ("fee_{0:N3}" -f $fee) `
    -OutDir "$OUTROOT\fee" `
    -ExtraArgs @("--phi_adval",$fee)
}

# 1-7) 변동성 헤지 on/off (sigma 기준)
$levels_sigmahedge = @("off","on")
foreach($h in $levels_sigmahedge){
  if($h -eq "off"){
    Invoke-Run -Tag "hedge_off" `
      -OutDir "$OUTROOT\hedge" `
      -ExtraArgs @("--hedge","off")
  } else {
    Invoke-Run -Tag "hedge_on_sigma" `
      -OutDir "$OUTROOT\hedge" `
      -ExtraArgs @("--hedge","on","--hedge_mode","sigma","--hedge_cost","0.005")
  }
}

# 1-8) 최소인출비율 q_floor
$levels_floor = @(0.01,0.02,0.03)
foreach($q in $levels_floor){
  Invoke-Run -Tag ("qfloor_{0:N2}" -f $q) `
    -OutDir "$OUTROOT\qfloor" `
    -ExtraArgs @("--q_floor",$q)
}

# =========[ 2) 대표 셀 bootstrap 강건성 체크(3–5개) ]=========
$bootstrap_cells = @(
  @{ tag="boot_age0_60";  dir="$OUTROOT\robust"; args=@("--age0","60","--bootstrap_block","12") },
  @{ tag="boot_wfixed_0.6"; dir="$OUTROOT\robust"; args=@("--w_fixed","0.6","--bootstrap_block","24") },
  @{ tag="boot_fee_0.004"; dir="$OUTROOT\robust"; args=@("--phi_adval","0.004","--bootstrap_block","12") }
)
foreach($cell in $bootstrap_cells){
  Invoke-Run -Tag $cell.tag -OutDir $cell.dir -ExtraArgs $cell.args
}

# =========[ 3) 산출물 합치기/요약 (metrics.csv 전부 통합) ]=========
$combined = "$OUTROOT\_summary\metrics_all.csv"
New-Item -ItemType Directory -Force -Path (Split-Path $combined) | Out-Null
Get-ChildItem -Path $OUTROOT -Recurse -Filter "metrics.csv" |
  ForEach-Object { Import-Csv $_.FullName } |
  Export-Csv -NoTypeInformation -Path $combined -Encoding UTF8

Write-Host "[SUMMARY] Combined metrics -> $combined" -ForegroundColor Cyan

# =========[ 4) (선택) 경계구간 재런: seeds=10로 덮어쓰기 ]=========
# 필요 시 아래 주석 해제하고 경계 레벨만 다시 실행
# $SEEDS = 10
# Invoke-Run -Tag "wfixed_0.5_s10" -OutDir "$OUTROOT\wfixed" -ExtraArgs @("--w_fixed","0.5")
# Invoke-Run -Tag "age0_60_s10"    -OutDir "$OUTROOT\age0"   -ExtraArgs @("--age0","60")

Write-Host "All jobs submitted. Good night 🌙" -ForegroundColor Magenta
