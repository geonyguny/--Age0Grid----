param([int]$Seed=0)

$env:PYTHONPATH = "$PWD;$PWD\project;$PWD\src;$env:PYTHONPATH"
$RUNNER = python .\scripts\resolve_runner.py
if ($LASTEXITCODE -ne 0 -or $RUNNER -eq "NO_MODULE") { throw "runner.cli 계열 모듈을 찾지 못함" }

# 1) lambda_term 스윕
foreach ($lam in "0.5","0.8","1.2") {
  & python -m $RUNNER --method rl --seed $Seed --lambda_term $lam `
    --tag "VAL_lambda_$lam" --market_mode bootstrap --bootstrap_block 24 --q_floor 0.02 --fee_annual 0.004 `
    --alpha 0.95 --beta 0.996 --horizon_years 35 --mortality on --sex M
}

# 2) w_max 스윕
foreach ($w in "0.50","0.70","1.00") {
  & python -m $RUNNER --method hjb --seed $Seed --w_max $w `
    --tag "VAL_wmax_$w" --market_mode bootstrap --bootstrap_block 24 --q_floor 0.02 --fee_annual 0.004 `
    --alpha 0.95 --lambda_term 0.8 --beta 0.996 --horizon_years 35 --mortality on --sex M
}

# 3) 수수료/바닥
foreach ($fee in "0.000","0.004","0.010") {
  & python -m $RUNNER --method rl --seed $Seed --fee_annual $fee `
    --tag "VAL_fee_$fee" --market_mode bootstrap --bootstrap_block 24 --q_floor 0.02 `
    --alpha 0.95 --lambda_term 0.8 --beta 0.996 --horizon_years 35 --mortality on --sex M
}
foreach ($qf in "0.00","0.02","0.05") {
  & python -m $RUNNER --method rl --seed $Seed --q_floor $qf `
    --tag "VAL_qfloor_$qf" --market_mode bootstrap --bootstrap_block 24 --fee_annual 0.004 `
    --alpha 0.95 --lambda_term 0.8 --beta 0.996 --horizon_years 35 --mortality on --sex M
}

# 4) sigma hedge 강도
foreach ($h in "0.00","0.10","0.25","0.40") {
  & python -m $RUNNER --method hjb --seed $Seed --hedge_sigma_k $h `
    --tag "VAL_hsig_$h" --market_mode bootstrap --bootstrap_block 24 --q_floor 0.02 --fee_annual 0.004 `
    --alpha 0.95 --lambda_term 0.8 --beta 0.996 --horizon_years 35 --mortality on --sex M
}
