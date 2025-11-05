param(
  [double]$Kappa = 1.5,
  [int]$Seed = 0,
  [string]$TagPrefix = "DEV_OAT_la"
)
$env:PYTHONPATH = "$PWD;$PWD\project;$PWD\src;$env:PYTHONPATH"
$RUNNER = python .\scripts\resolve_runner.py
if ($LASTEXITCODE -ne 0 -or $RUNNER -eq "NO_MODULE") { throw "runner.cli 계열 모듈을 찾지 못함" }

# Policy-Locked (학습 중립, 평가만 편향)
$tagPL = "${TagPrefix}_PL_k${Kappa}_s${Seed}"
& python -m $RUNNER --method rl --seed $Seed `
  --bias_on off --eval_bias_on on --bias_loss_aversion $Kappa `
  --tag $tagPL --market_mode bootstrap --bootstrap_block 24 `
  --q_floor 0.02 --fee_annual 0.004 --alpha 0.95 --lambda_term 0.8 --beta 0.996 `
  --horizon_years 35 --mortality on --sex M

# Re-opt (학습·평가 모두 편향)
$tagRO = "${TagPrefix}_RO_k${Kappa}_s${Seed}"
& python -m $RUNNER --method rl --seed $Seed `
  --bias_on on --eval_bias_on on --bias_loss_aversion $Kappa `
  --tag $tagRO --market_mode bootstrap --bootstrap_block 24 `
  --q_floor 0.02 --fee_annual 0.004 --alpha 0.95 --lambda_term 0.8 --beta 0.996 `
  --horizon_years 35 --mortality on --sex M
