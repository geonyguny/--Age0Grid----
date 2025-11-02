# scripts/run_theta_amb.ps1
# Hansen–Sargent 스타일 모호성(ambiguity) 파라미터 스위프 (DECUM 전용)
# 예) powershell -ExecutionPolicy Bypass -File .\scripts\run_theta_amb.ps1 -Autosave

param(
  [string]$Python      = ".\.venv\Scripts\python.exe",
  [string]$CliMod      = "project.runner.cli",
  [string]$DataProfile = "dev",
  [string]$MarketMode  = "bootstrap",
  [string]$Method      = "hjb",
  [double[]]$ThetaList = @(0.0, 0.5, 1.0, 2.0),
  [int[]]$Seeds        = @(11,12,13,14,15),
  [string]$TagPrefix   = "AMB",
  [switch]$Autosave
)

# ALM과 구동 컨텍스트 분리
$env:SIM_CONTEXT = "DECUM"

foreach ($t in $ThetaList) {
  foreach ($s in $Seeds) {
    $argsPlain = @(
      "--method",       $Method,
      "--data_profile", $DataProfile,
      "--market_mode",  $MarketMode,
      "--theta_ambiguity", "$t",
      "--seed",         "$s",
      "--tag",          ("{0}_t{1}_s{2}" -f $TagPrefix, $t, $s),
      "--print_mode",   "summary"
    )
    if ($Autosave) { $argsPlain += @("--autosave","on") }

    Write-Host ">> $Python -m $CliMod $($argsPlain -join ' ')" -ForegroundColor Cyan
    & $Python -m $CliMod @argsPlain

    if ($LASTEXITCODE -ne 0) {
      Write-Warning "실행 실패 (theta=$t, seed=$s). cli.py에 --theta_ambiguity 인자가 정의되어 있는지 확인하세요."
    }
  }
}
