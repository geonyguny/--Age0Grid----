# Milevsky 연금화시점 분석 - 최종판 (gamma=3/1 x 7개 조건)
# 사용법: chcp 65001 먼저 실행 후, milevsky_final_training_batch.ps1 학습 전부 완료 후 실행

$gammaSets = @{
    "g3" = 3.0
    "g1" = 1.0
}

$biasKeys = @("base", "lossaversion", "habit", "regret", "presentbias", "ambiguity", "probdistort")

foreach ($gkey in $gammaSets.Keys) {
    $gamma = $gammaSets[$gkey]
    foreach ($key in $biasKeys) {
        $pattern = "rl_final_${gkey}_${key}_s0_*"
        $folder = Get-ChildItem "outputs\_logs" -Directory -ErrorAction SilentlyContinue |
                  Where-Object { $_.Name -like "*$pattern*" -and (Test-Path (Join-Path $_.FullName "best.pt")) } |
                  Sort-Object LastWriteTime -Descending |
                  Select-Object -First 1

        if (-not $folder) {
            Write-Host "[건너뜀] ${gkey}/${key} : best.pt가 있는 폴더를 찾지 못함" -ForegroundColor Yellow
            continue
        }

        $ckpt = Join-Path $folder.FullName "best.pt"
        Write-Host "===== [${gkey}/${key}] $($folder.Name) (gamma=$gamma) =====" -ForegroundColor Cyan
        python milevsky_timing_RL.py "$ckpt" --gamma $gamma

        $outFile = "milevsky_timing_RL_${gkey}_${key}.json"
        if (Test-Path "milevsky_timing_RL_results.json") {
            Move-Item -Force "milevsky_timing_RL_results.json" $outFile
            Write-Host "저장됨: $outFile" -ForegroundColor Green
        } else {
            Write-Host "[경고] ${gkey}/${key} : 결과 파일이 생성되지 않음" -ForegroundColor Red
        }
        Write-Host ""
    }
}

Write-Host "전체 완료. milevsky_timing_RL_g3_*.json / milevsky_timing_RL_g1_*.json 파일들을 확인하세요." -ForegroundColor Cyan