# 잔차 정책용 HJB 기준선(baseline) 재생성 - 자산=KR 정합
# rl_final_training_batch.ps1 이 --asset KR 로 학습하므로, 잔차 정책의 고정 기준선도
# KR-HJB 여야 "무편향 → 보정≈0(θ*≈0)" 성질이 성립한다.
# (기존 bc_g1.pt/bc_g3.pt 는 TDF 기준 + critic 미포함이라 부적합)
# 산출: bc_kr_g1.pt, bc_kr_g3.pt  (actor + critic 모두 포함)

python pretrain_bc.py --asset KR --crra_gamma 3.0 --w_max 0.70 --pension_rho 0.30 --q_cap 0.02 --epochs 300 --out bc_kr_g3.pt
python pretrain_bc.py --asset KR --crra_gamma 1.0 --w_max 0.70 --pension_rho 0.30 --q_cap 0.02 --epochs 300 --out bc_kr_g1.pt

Write-Host "완료: bc_kr_g3.pt / bc_kr_g1.pt 생성. 이제 rl_residual_training_batch.ps1 실행 가능." -ForegroundColor Cyan
