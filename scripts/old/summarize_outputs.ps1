param(
  [string]$Root = ".\outputs",
  [string[]]$Patterns = @("metrics.json","wealth_metrics.json","results.json"),
  [switch]$Recurse,
  [string]$OutCsv = ".\outputs\_summary.csv",
  [switch]$NoDedup,
  [int]$Round = 6,

  # 종합점수 가중치 (JSON → CompositeScore 경로에서만 사용)
  [double]$AlphaES = 0.5,
  [double]$BetaRuin = 0.3,
  [double]$GammaEW = 0.2,

  # 부가 출력
  [string]$OutCsvScored       = ".\outputs\_summary_scored.csv",
  [string]$OutCsvPairs        = ".\outputs\_pairwise_vs_best.csv",
  [string]$OutCsvMethod       = ".\outputs\_method_summary.csv",
  [string]$OutCsvMethodNoSeed = ".\outputs\_method_summary_noseed.csv",

  # 선택 스위치
  [switch]$SkipPairs,
  [switch]$SkipMethod,

  # (신규) Normalize 결과(rescored) 우선 사용
  [switch]$UseRescored,
  [string]$NormCsv = ".\outputs\_summary_scored_norm.csv",

  # (신규) Method 요약에서 seed를 그룹키에서 제외하고 별도 파일 저장
  [switch]$NoSeedForGroups,

  # (신규) 동률 판단 epsilon (Composite 기준)
  [double]$TieEps = 1e-4
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ─────────────────────────────────────────────────────────
# 공용 유틸
# ─────────────────────────────────────────────────────────
function Get-Prop {
  param($obj, [string[]]$names)
  foreach ($n in $names) {
    if ($null -ne $obj -and $obj.PSObject.Properties[$n]) { return $obj.$n }
  }
  return $null
}

function To-Double {
  param($val, [int]$Round)
  if ($null -eq $val) { return $null }
  if ($val -is [double]) { return [math]::Round($val, $Round) }
  if ($val -is [single]) { return [math]::Round([double]$val, $Round) }
  if ($val -is [int])    { return [math]::Round([double]$val, $Round) }
  $s = $val.ToString() -replace '[,\s]',''
  [double]$tmp = 0
  if ([double]::TryParse($s, [ref]$tmp)) { return [math]::Round($tmp, $Round) }
  return $null
}

function To-Int {
  param($val)
  if ($null -eq $val) { return $null }
  if ($val -is [int]) { return $val }
  $s = $val.ToString().Trim()
  [int]$tmp = 0
  if ([int]::TryParse($s, [ref]$tmp)) { return $tmp }
  [double]$d = 0
  if ([double]::TryParse($s, [ref]$d)) { return [int][math]::Round($d) }
  return $null
}

function Find-FirstExistingFile {
  param([string]$Dir, [string[]]$Patterns)
  foreach ($p in $Patterns) {
    $cand = Join-Path $Dir $p
    if (Test-Path $cand) { return $cand }
    $hits = Get-ChildItem -Path $Dir -Filter $p -File -ErrorAction SilentlyContinue
    if ($hits -and $hits.Count -gt 0) { return $hits[0].FullName }
  }
  return $null
}

function Parse-MetaFromTag {
  param([string]$tag)
  $meta = [ordered]@{
    es_metric   = $null
    hedge_ratio = $null
    mix_kr      = $null
    mix_us      = $null
    mix_gold    = $null
    window      = $null
    seed_tag    = $null
  }
  if ($tag -match '_(wealth|loss|cons)_') { $meta.es_metric = $Matches[1] }
  if ($tag -match '_h([0-9]+(?:\.[0-9]+)?)_') { $meta.hedge_ratio = [double]$Matches[1] }
  if ($tag -match '_m([0-9\.]+)-([0-9\.]+)-([0-9\.]+)_') {
    $meta.mix_kr   = [double]$Matches[1]
    $meta.mix_us   = [double]$Matches[2]
    $meta.mix_gold = [double]$Matches[3]
  }
  if ($tag -match '_w(FULL|[0-9]{4}-[0-9]{2}to)') { $meta.window = $Matches[1] }
  if ($tag -match '_s(\d+)') { $meta.seed_tag = [int]$Matches[1] }
  return [pscustomobject]$meta
}

function Get-MinMax {
  param($arr)
  $vals = @($arr | Where-Object { $_ -ne $null })
  if ($vals.Count -eq 0) { return @{min=$null; max=$null} }
  return @{
    min = ($vals | Measure-Object -Minimum | Select-Object -ExpandProperty Minimum)
    max = ($vals | Measure-Object -Maximum | Select-Object -ExpandProperty Maximum)
  }
}

function Norm {
  param($x, $min, $max)
  if ($x -eq $null -or $min -eq $null -or $max -eq $null) { return $null }
  if ($max -eq $min) { return 0.5 }
  return ($x - $min) / ($max - $min)
}

function Avg-Safe {
  param([object[]]$arr, [string]$prop)
  $vals = @($arr | ForEach-Object { $_.$prop } | Where-Object { $_ -ne $null })
  if ($vals.Count -eq 0) { return $null }
  $avg = ($vals | Measure-Object -Average | Select-Object -ExpandProperty Average)
  if ($avg -eq $null) { return $null }
  return [math]::Round([double]$avg, 6)
}

# 동일 환경 비교용 키
$EnvironmentKeysSeed   = @('es_mode','window','hedge_ratio','mix_kr','mix_us','mix_gold','es_metric','seed')
$EnvironmentKeysNoSeed = @('es_mode','window','hedge_ratio','mix_kr','mix_us','mix_gold','es_metric')

function Group-KeyBy {
  param($r, [string[]]$Keys)
  $vals = foreach ($k in $Keys) {
    $v = $r.$k
    if ($v -eq $null) { '' } else { $v }
  }
  return ($vals -join '|')
}

# 출력 경로 보장
foreach ($p in @($OutCsv, $OutCsvScored, $OutCsvPairs, $OutCsvMethod, $OutCsvMethodNoSeed)) {
  $dir = Split-Path -Path $p -Parent
  if ($dir -and -not (Test-Path $dir)) { [void](New-Item -ItemType Directory -Path $dir -Force) }
}

# ─────────────────────────────────────────────────────────
# 0) 입력 로드: (우선) normalized CSV → 실패 시 JSON 스캔
# ─────────────────────────────────────────────────────────
$FromNorm = $false
$rows = $null
$scored = $null

if ($UseRescored -and (Test-Path $NormCsv)) {
  try {
    $norm = Import-Csv $NormCsv
    if ($norm.Count -gt 0) {
      $hasRescored = $norm[0].PSObject.Properties.Name -contains 'Composite_rescored'
      $hasComp     = $norm[0].PSObject.Properties.Name -contains 'CompositeScore'
      if ($hasRescored -or $hasComp) {
        # CompositeUse 생성(우선순위: rescored → CompositeScore)
        $norm | ForEach-Object {
          $val = if ($hasRescored) { $_.Composite_rescored } elseif ($hasComp) { $_.CompositeScore } else { $null }
          $_ | Add-Member -Name CompositeUse -Value $val -MemberType NoteProperty -Force
        }
        $rows = $norm
        $FromNorm = $true
      } else {
        Write-Warning "NormCsv에 Composite_rescored/CompositeScore가 없어 JSON 경로로 폴백합니다."
      }
    }
  } catch {
    Write-Warning "NormCsv 로드 실패 → JSON 경로로 폴백: $($_.Exception.Message)"
  }
}

if (-not $FromNorm) {
  # JSON 스캔
  $dirs = if ($Recurse) { Get-ChildItem $Root -Directory -Recurse } else { Get-ChildItem $Root -Directory }
  $rows = New-Object System.Collections.Generic.List[psobject]

  foreach ($d in $dirs) {
    $file = Find-FirstExistingFile -Dir $d.FullName -Patterns $Patterns
    if (-not $file) { continue }
    try { $j = Get-Content $file -Raw | ConvertFrom-Json }
    catch { Write-Warning "JSON 파싱 실패: $file (`$($_.Exception.Message)`)" ; continue }

    # 지표 통합 추출
    $EW        = To-Double (Get-Prop $j @('EW','mean_WT','mean_wealth','expected_wealth','exp_wealth')) $Round
    $ES95      = To-Double (Get-Prop $j @(
                      'ES95','ES_95','es95',
                      'cons_ES95','cons_es95','ES95_cons','ES_cons',
                      'wealth_ES95','wealth_es95','ES95_wealth','ES_wealth',
                      'loss_ES95','loss_es95','ES95_loss','ES_loss'
                    )) $Round
    $Ruin      = To-Double (Get-Prop $j @('RuinPct','ruin_pct','Ruin','ruin','RuinRate','ruin_rate')) $Round
    $ESModeRaw = Get-Prop   $j @('es_mode','ES_mode','esMode','mode_es')
    $EUperYear = To-Double (Get-Prop $j @('EU_per_year','eu_per_year','EU_yearly','EUperYear')) $Round
    $AliveRate = To-Double (Get-Prop $j @('AlivePathRate','alive_path_rate','alive_rate','AliveRate')) $Round
    $HedgeHit  = To-Double (Get-Prop $j @('HedgeHit','hedge_hit','HedgeHitRate','hedge_hit_rate')) $Round
    $SeedRaw   = Get-Prop   $j @('seed','rng_seed','random_seed')
    $NPaths    = To-Int     (Get-Prop $j @('n_paths','NPaths','num_paths','paths','n_paths_eval','eval_episodes'))
    $MethodRaw = Get-Prop   $j @('method','algo','solver')
    $BiasOn    = Get-Prop   $j @('bias_on','bias','behavior_on','behavior')

    # 태그 메타
    $meta = Parse-MetaFromTag -tag $d.Name

    # es_mode 정규화 + es_metric 자동 유추
    $ESMode = $null
    if ($ESModeRaw -ne $null) { $ESMode = To-Int $ESModeRaw }
    if ($null -eq $meta.es_metric -and $ESMode -ne $null) {
      $meta.es_metric = switch ($ESMode) { 0 { 'wealth' } 1 { 'cons' } 2 { 'loss' } default { $null } }
    }

    # seed 보정
    $Seed = if ($SeedRaw -ne $null) { To-Int $SeedRaw } elseif ($meta.seed_tag -ne $null) { [int]$meta.seed_tag } else { $null }

    # method 소문자
    $Method = if ($MethodRaw) { $MethodRaw.ToString().Trim().ToLower() } else { $null }

    [void]$rows.Add([pscustomobject]@{
      tag            = $d.Name
      method         = $Method
      es_metric      = $meta.es_metric
      es_mode        = $ESMode
      mix_kr         = To-Double $meta.mix_kr $Round
      mix_us         = To-Double $meta.mix_us $Round
      mix_gold       = To-Double $meta.mix_gold $Round
      hedge_ratio    = To-Double $meta.hedge_ratio $Round
      window         = $meta.window
      ES95           = $ES95
      EW             = $EW
      Ruin           = $Ruin
      EU_per_year    = $EUperYear
      AlivePathRate  = $AliveRate
      HedgeHit       = $HedgeHit
      seed           = $Seed
      n_paths        = $NPaths
      bias_on        = $BiasOn
      source         = $file
    })
  }

  if ($rows.Count -eq 0) {
    Write-Warning "요약할 항목이 없습니다. 폴더($Root) 아래에 $($Patterns -join ', ') 중 하나가 있는지 확인하세요. -Recurse 스위치도 고려하세요."
    return
  }

  # 기본값 보정
  $rows = $rows | ForEach-Object {
    if ($_.es_metric -eq $null) { $_.es_metric = 'wealth' }
    $_
  }
}

# ─────────────────────────────────────────────────────────
# 1) 중복 제거 및 기본 요약 저장 (공통)
# ─────────────────────────────────────────────────────────
$sortSpec = @(
  @{ Expression = { if ($_.ES95 -eq $null) { [double]::PositiveInfinity } else { $_.ES95 } }; Ascending = $true },
  @{ Expression = { if ($_.EW   -eq $null) { [double]::NegativeInfinity } else { -$_.EW }  }; Ascending = $true },
  @{ Expression = { if ($_.Ruin -eq $null) { [double]::PositiveInfinity } else { $_.Ruin } }; Ascending = $true }
)

$base =
  if ($NoDedup) { $rows | Sort-Object $sortSpec }
  else {
    $rows | Group-Object tag | ForEach-Object { $_.Group | Sort-Object $sortSpec | Select-Object -First 1 } | Sort-Object $sortSpec
  }

$base |
  Select-Object tag, method, es_metric, es_mode, window,
                mix_kr, mix_us, mix_gold, hedge_ratio,
                ES95, EW, Ruin, EU_per_year, AlivePathRate, HedgeHit,
                seed, n_paths, bias_on, source |
  Export-Csv -NoTypeInformation -Encoding UTF8 -Path $OutCsv -Force

# ─────────────────────────────────────────────────────────
# 2) 정규화/점수/랭킹
#   - FromNorm==True → CompositeUse(=rescored 우선)로 랭킹만 계산/저장
#   - FromNorm==False → 기존 방식으로 CompositeScore 계산 후 저장
# ─────────────────────────────────────────────────────────
if ($FromNorm) {
  # 이미 정규화/CompositeUse 보유 → 필요한 랭킹만 생성하고 저장
  $scored = @($base)

  # CompositeUse가 없다면 NormCsv에서 복사
  $hasCompUse = ($scored | Select-Object -First 1).PSObject.Properties.Name -contains 'CompositeUse'
  if (-not $hasCompUse) {
    # NormCsv 로드한 rows에서 tag 기준 merge를 시도
    $map = @{}
    foreach ($n in $rows) { $map[$n.tag] = $n }
    foreach ($x in $scored) {
      $v = $null; if ($map.ContainsKey($x.tag)) { $v = $map[$x.tag].CompositeUse }
      $x | Add-Member -Name CompositeUse -Value $v -MemberType NoteProperty -Force
    }
  }

  # 랭킹(ES95/Ruin/CompositeUse: 낮을수록 좋음, EW: 높을수록 좋음)
  $idx=1; foreach ($x in ($scored | Sort-Object @{Expression={ if ($_.ES95 -eq $null) { [double]::PositiveInfinity } else { $_.ES95 } }; Ascending=$true })) { $x | Add-Member rank_ES95 $idx -Force; $idx++ }
  $idx=1; foreach ($x in ($scored | Sort-Object @{Expression={ if ($_.Ruin -eq $null) { [double]::PositiveInfinity } else { $_.Ruin } }; Ascending=$true })) { $x | Add-Member rank_Ruin $idx -Force; $idx++ }
  $idx=1; foreach ($x in ($scored | Sort-Object @{Expression={ if ($_.CompositeUse -eq $null) { [double]::PositiveInfinity } else { $_.CompositeUse } }; Ascending=$true })) { $x | Add-Member rank_CompositeScore $idx -Force; $idx++ }
  $idx=1; foreach ($x in ($scored | Sort-Object @{Expression={ if ($_.EW -eq $null) { [double]::NegativeInfinity } else { -$_.EW } }; Ascending=$true })) { $x | Add-Member rank_EW $idx -Force; $idx++ }

  $scored |
    Select-Object tag, method, es_metric, es_mode, window,
                  mix_kr, mix_us, mix_gold, hedge_ratio,
                  ES95, EW, Ruin,
                  # FromNorm 경로에서는 CompositeUse를 CompositeScore 컬럼명과 병행 저장(호환성)
                  @{Name='CompositeScore'; Expression={ $_.CompositeUse }},
                  EU_per_year, AlivePathRate, HedgeHit, seed, n_paths, bias_on, source,
                  rank_ES95, rank_EW, rank_Ruin, rank_CompositeScore |
    Export-Csv -NoTypeInformation -Encoding UTF8 -Path $OutCsvScored -Force
}
else {
  # 기존 방식으로 CompositeScore 계산
  $mmES  = Get-MinMax ($base | Select-Object -ExpandProperty ES95)
  $mmRui = Get-MinMax ($base | Select-Object -ExpandProperty Ruin)
  $mmEW  = Get-MinMax ($base | Select-Object -ExpandProperty EW)

  $withScore = foreach ($r in $base) {
    $ESn = Norm $r.ES95 $mmES.min $mmES.max
    $Run = Norm $r.Ruin $mmRui.min $mmRui.max
    $EWn = Norm $r.EW   $mmEW.min $mmEW.max
    $comp = $null
    if ($ESn -ne $null -and $Run -ne $null -and $EWn -ne $null) {
      # ES/Ruin 낮을수록 좋음, EW 높을수록 좋음 ⇒ EW는 (1 - EWn)
      $comp = ($AlphaES * $ESn) + ($BetaRuin * $Run) + ($GammaEW * (1 - $EWn))
    }
    [pscustomobject]@{
      tag=$r.tag; method=$r.method; es_metric=$r.es_metric; es_mode=$r.es_mode; window=$r.window
      mix_kr=$r.mix_kr; mix_us=$r.mix_us; mix_gold=$r.mix_gold; hedge_ratio=$r.hedge_ratio
      ES95=$r.ES95; EW=$r.EW; Ruin=$r.Ruin
      EU_per_year=$r.EU_per_year; AlivePathRate=$r.AlivePathRate; HedgeHit=$r.HedgeHit
      seed=$r.seed; n_paths=$r.n_paths; bias_on=$r.bias_on; source=$r.source
      ES95_norm=$ESn; Ruin_norm=$Run; EW_norm=$EWn; CompositeScore=$comp
    }
  }

  $scored = @($withScore)

  # 랭킹
  $idx=1; foreach ($x in ($scored | Sort-Object @{Expression={ if ($_.ES95 -eq $null) { [double]::PositiveInfinity } else { $_.ES95 } }; Ascending=$true })) { $x | Add-Member 'rank_ES95' $idx -Force; $idx++ }
  $idx=1; foreach ($x in ($scored | Sort-Object @{Expression={ if ($_.Ruin -eq $null) { [double]::PositiveInfinity } else { $_.Ruin } }; Ascending=$true })) { $x | Add-Member 'rank_Ruin' $idx -Force; $idx++ }
  $idx=1; foreach ($x in ($scored | Sort-Object @{Expression={ if ($_.CompositeScore -eq $null) { [double]::PositiveInfinity } else { $_.CompositeScore } }; Ascending=$true })) { $x | Add-Member 'rank_CompositeScore' $idx -Force; $idx++ }
  $idx=1; foreach ($x in ($scored | Sort-Object @{Expression={ if ($_.EW -eq $null) { [double]::NegativeInfinity } else { -$_.EW } }; Ascending=$true })) { $x | Add-Member 'rank_EW' $idx -Force; $idx++ }

  $scored |
    Select-Object tag, method, es_metric, es_mode, window,
                  mix_kr, mix_us, mix_gold, hedge_ratio,
                  ES95, EW, Ruin, ES95_norm, EW_norm, Ruin_norm, CompositeScore,
                  EU_per_year, AlivePathRate, HedgeHit, seed, n_paths, bias_on, source,
                  rank_ES95, rank_EW, rank_Ruin, rank_CompositeScore |
    Export-Csv -NoTypeInformation -Encoding UTF8 -Path $OutCsvScored -Force
}

# ─────────────────────────────────────────────────────────
# 3) 동일 환경 내 페어 비교(Seed 포함 그룹)
#   - FromNorm: CompositeUse(저장 시 CompositeScore로 에일리어스됨) 기준
#   - Else:     CompositeScore 기준
# ─────────────────────────────────────────────────────────
if (-not $SkipPairs) {
  $pairRows = New-Object System.Collections.Generic.List[psobject]
  $byGroup = $scored | Group-Object { Group-KeyBy $_ $EnvironmentKeysSeed }

  foreach ($g in $byGroup) {
    $members = @($g.Group | Where-Object {
      if ($FromNorm) { $_.CompositeScore -ne $null } else { $_.CompositeScore -ne $null }
    })
    if ($members.Count -eq 0) { continue }

    $ordered = if ($FromNorm) {
      $members | Sort-Object @{e={ [double]$_.CompositeScore }; Ascending=$true}
    } else {
      $members | Sort-Object @{e={ [double]$_.CompositeScore }; Ascending=$true}
    }
    $best = $ordered | Select-Object -First 1

    foreach ($m in $g.Group) {
      $dES  = $null; if ($m.ES95  -ne $null -and $best.ES95  -ne $null) { $dES  = $m.ES95  - $best.ES95 }
      $dEW  = $null; if ($m.EW    -ne $null -and $best.EW    -ne $null) { $dEW  = $m.EW    - $best.EW }
      $dRu  = $null; if ($m.Ruin  -ne $null -and $best.Ruin  -ne $null) { $dRu  = $m.Ruin  - $best.Ruin }
      $dCmp = $null; if ($m.CompositeScore -ne $null -and $best.CompositeScore -ne $null) { $dCmp = $m.CompositeScore - $best.CompositeScore }

      [void]$pairRows.Add([pscustomobject]@{
        group_key       = $g.Name
        es_mode         = $m.es_mode
        window          = $m.window
        hedge_ratio     = $m.hedge_ratio
        mix_kr          = $m.mix_kr
        mix_us          = $m.mix_us
        mix_gold        = $m.mix_gold
        es_metric       = $m.es_metric
        seed            = $m.seed

        tag             = $m.tag
        method          = $m.method
        ES95            = $m.ES95
        EW              = $m.EW
        Ruin            = $m.Ruin
        Composite       = $m.CompositeScore

        best_tag        = $best.tag
        best_method     = $best.method
        best_ES95       = $best.ES95
        best_EW         = $best.EW
        best_Ruin       = $best.Ruin
        best_Composite  = $best.CompositeScore

        delta_ES95      = $dES
        delta_EW        = $dEW
        delta_Ruin      = $dRu
        delta_Composite = $dCmp
      })
    }
  }

  if ($pairRows.Count -gt 0) {
    $pairRows | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $OutCsvPairs -Force
  } else {
    Write-Warning "내보낼 페어 비교가 없습니다. (pairRows 비어있음)"
    "" | Out-File -Encoding utf8 $OutCsvPairs
  }
}

# ─────────────────────────────────────────────────────────
# 4) Method 요약 (seed 포함 기준) + (옵션) seed 제외 기준 추가
#   - FromNorm: CompositeScore(=CompositeUse) 기준
#   - Else:     CompositeScore 기준
# ─────────────────────────────────────────────────────────
if (-not $SkipMethod) {

  # 4-1) seed 포함 그룹키
  $rankRows = New-Object System.Collections.Generic.List[psobject]
  foreach ($g in ($scored | Group-Object { Group-KeyBy $_ $EnvironmentKeysSeed })) {
    $members = @($g.Group | Where-Object { $_.CompositeScore -ne $null })
    if ($members.Count -eq 0) { continue }
    $ordered = $members | Sort-Object @{e={ [double]$_.CompositeScore }; Ascending=$true}
    $rnk = 1
    foreach ($m in $ordered) {
      [void]$rankRows.Add([pscustomobject]@{
        method     = $m.method
        tag        = $m.tag
        group_key  = $g.Name
        group_rank = $rnk
        ES95       = $m.ES95
        EW         = $m.EW
        Ruin       = $m.Ruin
        Composite  = $m.CompositeScore
      })
      $rnk++
    }
  }

  $methodSummaryRows = New-Object System.Collections.Generic.List[psobject]
  foreach ($mgroup in ($rankRows | Group-Object method)) {
    $grp  = @($mgroup.Group)
    $wins = (@($grp | Where-Object { $_.group_rank -eq 1 })).Count
    $cnt  = $grp.Count
    $nameForDisplay = if ([string]::IsNullOrWhiteSpace($mgroup.Name)) { '(null)' } else { $mgroup.Name }
    $win_rate = if ($cnt -gt 0) { [math]::Round($wins / $cnt, 6) } else { $null }
    $avgRank = Avg-Safe -arr $grp -prop 'group_rank'
    $avgES   = Avg-Safe -arr $grp -prop 'ES95'
    $avgEW   = Avg-Safe -arr $grp -prop 'EW'
    $avgRu   = Avg-Safe -arr $grp -prop 'Ruin'
    $avgCm   = Avg-Safe -arr $grp -prop 'Composite'

    [void]$methodSummaryRows.Add([pscustomobject]@{
      method          = $nameForDisplay
      samples         = $cnt
      win_count       = $wins
      win_rate        = $win_rate
      avg_group_rank  = $avgRank
      avg_ES95        = $avgES
      avg_EW          = $avgEW
      avg_Ruin        = $avgRu
      avg_Composite   = $avgCm
    })
  }

  $methodSummary =
    @($methodSummaryRows) |
    Sort-Object @{e='win_rate';Descending=$true}, @{e='avg_Composite';Ascending=$true}

  if ($methodSummary.Count -gt 0) {
    $methodSummary | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $OutCsvMethod -Force
  } else {
    "" | Out-File -Encoding utf8 $OutCsvMethod
  }

  # 4-2) (옵션) seed 제외 그룹키 버전 추가 산출
  if ($NoSeedForGroups) {
    $rankRows2 = New-Object System.Collections.Generic.List[psobject]
    foreach ($g in ($scored | Group-Object { Group-KeyBy $_ $EnvironmentKeysNoSeed })) {
      $members = @($g.Group | Where-Object { $_.CompositeScore -ne $null })
      if ($members.Count -eq 0) { continue }
      $ordered = $members | Sort-Object @{e={ [double]$_.CompositeScore }; Ascending=$true}
      $rnk = 1
      foreach ($m in $ordered) {
        [void]$rankRows2.Add([pscustomobject]@{
          method     = $m.method
          tag        = $m.tag
          group_key  = $g.Name
          group_rank = $rnk
          ES95       = $m.ES95
          EW         = $m.EW
          Ruin       = $m.Ruin
          Composite  = $m.CompositeScore
        })
        $rnk++
      }
    }

    $methodSummaryRows2 = New-Object System.Collections.Generic.List[psobject]
    foreach ($mgroup in ($rankRows2 | Group-Object method)) {
      $grp  = @($mgroup.Group)
      $wins = (@($grp | Where-Object { $_.group_rank -eq 1 })).Count
      $cnt  = $grp.Count
      $nameForDisplay = if ([string]::IsNullOrWhiteSpace($mgroup.Name)) { '(null)' } else { $mgroup.Name }
      $win_rate = if ($cnt -gt 0) { [math]::Round($wins / $cnt, 6) } else { $null }
      $avgRank = Avg-Safe -arr $grp -prop 'group_rank'
      $avgES   = Avg-Safe -arr $grp -prop 'ES95'
      $avgEW   = Avg-Safe -arr $grp -prop 'EW'
      $avgRu   = Avg-Safe -arr $grp -prop 'Ruin'
      $avgCm   = Avg-Safe -arr $grp -prop 'Composite'

      [void]$methodSummaryRows2.Add([pscustomobject]@{
        method          = $nameForDisplay
        samples         = $cnt
        win_count       = $wins
        win_rate        = $win_rate
        avg_group_rank  = $avgRank
        avg_ES95        = $avgES
        avg_EW          = $avgEW
        avg_Ruin        = $avgRu
        avg_Composite   = $avgCm
      })
    }

    $methodSummary2 =
      @($methodSummaryRows2) |
      Sort-Object @{e='win_rate';Descending=$true}, @{e='avg_Composite';Ascending=$true}

    if ($methodSummary2.Count -gt 0) {
      $methodSummary2 | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $OutCsvMethodNoSeed -Force
      Write-Host ("[OK] saved (method, noseed) -> " + (Resolve-Path $OutCsvMethodNoSeed))
    } else {
      "" | Out-File -Encoding utf8 $OutCsvMethodNoSeed
      Write-Host ("[WARN] empty (method, noseed) -> " + (Resolve-Path $OutCsvMethodNoSeed))
    }
  }
}

# ─────────────────────────────────────────────────────────
# 5) 화면 출력 (요약)
# ─────────────────────────────────────────────────────────
Write-Host ""
Write-Host ("Saved CSV (base)     -> " + (Resolve-Path $OutCsv))
Write-Host ("Saved CSV (scored)   -> " + (Resolve-Path $OutCsvScored))
if (-not $SkipPairs)  { Write-Host ("Saved CSV (pairs)    -> " + (Resolve-Path $OutCsvPairs)) }
if (-not $SkipMethod) {
  Write-Host ("Saved CSV (methods)  -> " + (Resolve-Path $OutCsvMethod))
  if ($NoSeedForGroups) {
    Write-Host ("Saved CSV (methods*) -> " + (Resolve-Path $OutCsvMethodNoSeed))
  }
}
Write-Host ""

# 참고: 표시용 테이블(CompositeScore 컬럼은 FromNorm일 때 CompositeUse를 재사용)
$scored |
  Select-Object tag, method, es_mode, es_metric, hedge_ratio, mix_kr, mix_us, mix_gold, window,
                ES95, EW, Ruin, CompositeScore, rank_ES95, rank_EW, rank_Ruin, rank_CompositeScore |
  Sort-Object @{e='CompositeScore';Ascending=$true} |
  Format-Table -AutoSize
