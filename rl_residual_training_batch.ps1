# 잔차 정책(Residual Policy) 학습 배치
# rl_final_training_batch.ps1 과 동일한 설정 + 잔차 정책 3옵션만 추가.
# 기준선: bc_kr_g3.pt / bc_kr_g1.pt (bc_regen_kr_baseline.ps1 로 먼저 생성할 것)
# 총 14개(gamma 2 x 조건 7). 산출 태그: rl_resid_<gkey>_<key>_s0
#
# 잔차 옵션:
#   --residual_policy on          : HJB(모방) 기준선 위에 작은 보정만 학습
#   --baseline_ckpt bc_kr_<g>.pt  : 고정 기준선(KR-HJB)
#   --residual_scale 0.15         : 보정폭 상한(±0.15, raw 액션 기준)
#   --residual_l2_coef 0.005      : 보정 L2 정규화(무편향 시 보정→0 유도)

$common = "--method rl --asset KR --market_mode iid --horizon_years 35 --w_max 0.70 --alpha 0.95 --report_utility on --delta_annual 0.9530 --rl_epochs 400 --rl_steps_per_epoch 2048 --rl_n_paths_eval 300 --seed 0 --rl_q_cap 0.02 --u_scale 0.0001 --entropy_coef 0.03 --lr 0.0001 --value_clip 100 --entropy_clip 0.2 --pension_rho 0.30 --train_random_annuity on --train_annuity_prob 0.5 --train_annuity_theta_max 0.8 --residual_policy on --residual_scale 0.15 --residual_l2_coef 0.005 --print_mode summary --metrics_keys `"EW,ES95,Ruin,mean_WT,EU,EU_per_year`""

# 조건별 편향 플래그 (rl_final_training_batch.ps1 과 동일)
$biasFlags = [ordered]@{
    "base"         = ""
    "lossaversion" = "--bh_on on --la_k 0.5"
    "habit"        = "--bh_on on --habit_phi 0.5"
    "regret"       = "--bh_on on --bh_regret_rho 0.5 --regret_c_ref_rate 0.12"
    "presentbias"  = "--bh_on on --beta 0.85"
    "ambiguity"    = "--bh_on on --theta_ambiguity 0.5"
    "probdistort"  = "--bias_on on --bias_prob_gamma 0.7"
}
$gammaSets = [ordered]@{ "g3" = 3.0; "g1" = 1.0 }

foreach ($gkey in $gammaSets.Keys) {
    $gamma = $gammaSets[$gkey]
    $baseline = "bc_kr_$gkey.pt"
    if (-not (Test-Path $baseline)) {
        Write-Host "[중단] $baseline 없음. 먼저 bc_regen_kr_baseline.ps1 실행 필요." -ForegroundColor Red
        continue
    }
    foreach ($key in $biasFlags.Keys) {
        $bias = $biasFlags[$key]
        $tag  = "rl_resid_${gkey}_${key}_s0"
        Write-Host "===== 학습 [$gkey/$key] (gamma=$gamma, baseline=$baseline) =====" -ForegroundColor Cyan
        $cmd = "python -m project.runner.cli $common --crra_gamma $gamma --baseline_ckpt $baseline $bias --tag $tag"
        Invoke-Expression $cmd
    }
}

Write-Host "잔차 정책 학습 전체 완료. 평가는 milevsky_timing_RL.py 로 각 best.pt 를 돌리세요." -ForegroundColor Green
