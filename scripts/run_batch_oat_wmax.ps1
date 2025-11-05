param(
  [string]$Values = "0.70,1.00",
  [string]$Method = "hjb",
  [string]$Mode = "dev",
  [string]$TagPrefix = "DEV_OAT_wmax"
)
$env:PYTHONPATH = "$PWD;$PWD\project;$PWD\src;$env:PYTHONPATH"
$RUNNER = python .\scripts\resolve_runner.py
if ($LASTEXITCODE -ne 0 -or $RUNNER -eq "NO_MODULE") { throw "runner.cli 계열 모듈을 찾지 못함" }

$vals = $Values -split ','
foreach ($v in $vals) {
  $tag = "${TagPrefix}_${v}"
  & python -m $RUNNER --method $Method --tag $tag `
     --w_max $v --market_mode bootstrap --bootstrap_block 24 `
     --q_floor 0.02 --fee_annual 0.004 --alpha 0.95 --lambda_term 0.8 --beta 0.996 `
     --horizon_years 35 --mortality on --sex M
}
