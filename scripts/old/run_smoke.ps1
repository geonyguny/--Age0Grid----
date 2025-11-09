# scripts/run_smoke.ps1
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OUT = "./outputs"

Write-Host "=== (A) HJB 기본 실행 + 소비밴드/로그 ==="
py -m project.runner.cli --method hjb --asset KR --market_mode bootstrap --data_profile dev --outputs $OUT --n_paths 200 --bands on --print_mode summary --quiet off

Write-Host "`n=== (B) 손실기준 ES(=CVaR) ==="
py -m project.runner.cli --method hjb --asset KR --market_mode bootstrap --data_profile dev --es_mode loss --F_target 1.0 --alpha 0.95 --n_paths 300 --outputs $OUT --print_mode summary --quiet off

Write-Host "`n=== (C) 효용 리포팅 + 효용-레이어 편향(λ,β,φ) ==="
py -m project.runner.cli --method hjb --asset KR --market_mode bootstrap --data_profile dev --report_utility on --crra_gamma 3.0 --u_scale 1.0 --delta_annual 0.97 --bh_on on --la_k 1.3 --beta 0.85 --habit_phi 0.2 --n_paths 200 --outputs $OUT --print_mode summary --quiet off

Write-Host "`n=== (D) 액션-레이어 편향(근시 q 상향 + w 하한) ==="
py -m project.runner.cli --method hjb --asset KR --market_mode bootstrap --data_profile dev --bias_on on --bias_myopia 0.5 --bias_w_floor 0.2 --n_paths 200 --outputs $OUT --print_mode summary --quiet off

Write-Host "`n=== (E) 룰(K-GR) + 사망/연금 오버레이 ==="
$MORT = "D:\01_simul\project\data\kidi_qx.csv"  # 경로 확인
py -m project.runner.cli --method rule --baseline kgr --market_mode bootstrap --data_profile dev --mortality on --mort_table "$MORT" --age0 65 --ann_on on --ann_alpha 0.2 --ann_L 0.0 --ann_d 0 --ann_index real --n_paths 200 --outputs $OUT --print_mode summary --quiet off

Write-Host "`n=== (F) RL 스모크(빠른 확인용) ==="
py -m project.runner.cli --method rl --asset KR --market_mode bootstrap --data_profile dev --rl_epochs 1 --rl_steps_per_epoch 1024 --rl_n_paths_eval 64 --outputs $OUT --quiet off

Write-Host "`n완료! 결과는 $OUT\_logs\metrics.csv 에 누적됩니다."
