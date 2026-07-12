# Milevsky 연금화시점 분석 - 편향별 일괄 실행
# 사용법: chcp 65001 먼저 실행 후, 이 파일을 G:\01_simul 에 두고 .\milevsky_run_all.ps1 실행

$biasPatterns = @{
    "base"          = "rl5b_base_s0_*", "rl5_base_s0_*", "rl_rl_bias_base_*"
    "lossaversion"  = "rl5b_lossaversion_s0_*", "rl5_lossaversion_s0_*", "rl_rl_bias_lossaversion_*"
    "habit"         = "rl5b_habit_s0_*", "rl5_habit_s0_*", "rl_rl_bias_habit_*"
    "presentbias"   = "rl5b_presentbias_s0_*", "rl_rl_bias_presentbias_*"
    "ambiguity"     = "rl5b_ambiguity_s0_*", "rl_rl_bias_ambiguity_*"
    "regret"        = "rl5b_regret_s0_*", "rl5_regret_s0_*", "rl_rl_bias_regret_fixed_*", "rl_rl_bias_regret_*"
    "probdistort"   = "rl5b_probdistort_s0_*", "rl5_probdistort_s0_*", "rl_rl_bias_probdistort_*"
}

foreach ($key in $biasPatterns.Keys) {
    $patterns = $biasPatterns[$key]
    $folder = $null
    foreach ($p in $patterns) {
        # best.pt가 실제로 있는 폴더만 후보로 인정 (구버전 실험 제외)
        $found = Get-ChildItem "outputs\_logs" -Directory -ErrorAction SilentlyContinue |
                 Where-Object { $_.Name -like "*$p*" -and (Test-Path (Join-Path $_.FullName "best.pt")) } |
                 Sort-Object LastWriteTime -Descending |
                 Select-Object -First 1
        if ($found) { $folder = $found; break }
    }

    if (-not $folder) {
        Write-Host "[건너뜀] '$key' : best.pt가 있는 폴더를 찾지 못함 (해당 실험이 아직 없거나 완료 전)" -ForegroundColor Yellow
        continue
    }

    $ckpt = Join-Path $folder.FullName "best.pt"

    Write-Host "===== [$key] $($folder.Name) =====" -ForegroundColor Cyan
    python milevsky_timing_RL.py "$ckpt"

    $outFile = "milevsky_timing_RL_$key.json"
    if (Test-Path "milevsky_timing_RL_results.json") {
        Move-Item -Force "milevsky_timing_RL_results.json" $outFile
        Write-Host "저장됨: $outFile" -ForegroundColor Green
    } else {
        Write-Host "[경고] $key : 결과 파일이 생성되지 않음" -ForegroundColor Red
    }
    Write-Host ""
}

Write-Host "전체 완료. milevsky_timing_RL_*.json 파일들을 확인하세요." -ForegroundColor Cyan