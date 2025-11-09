# 짧은 경로수+여러 seed로 상위 후보 재평가
param(
  [string]$TagsFile = ".\tags_top5.txt",
  [int[]]$Seeds = @(0,1,2,3,4),
  [int]$NPaths = 500
)

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force | Out-Null

$tags = Get-Content $TagsFile | Where-Object { $_ -and $_.Trim().Length -gt 0 }
foreach($t in $tags){
  foreach($s in $Seeds){
    .\.venv\Scripts\python.exe -m runner.cli `
      --mode eval `
      --tag $t `
      --n_paths_eval $NPaths `
      --seed $s
  }
}

# 끝나면 요약 갱신
.\scripts\summarize_outputs.ps1 -OutDir .\outputs -CsvPath .\_summary.csv
