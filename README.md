# KR Decumulation Simulation (v2)

* HJB(2D: **q, w**) 및 RL 기반 인출 시뮬레이션.
* `project.runner.cli`로 일관된 실행/평가/메트릭 출력.

> **데이터 주의**
> 리포에는 대형 CSV를 포함하지 않습니다. 대신 **개발용(dev) 축약 CSV**를 자동 생성하여 바로 실행할 수 있습니다.

---

## Quickstart (DEV)

**Windows PowerShell**

```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt

# 개발용 CSV 생성 + 간단 검증
.\scripts\preflight_dev.ps1
```

**Linux / macOS (Bash)**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt

# 개발용 CSV 생성 + 간단 검증
pwsh -File scripts/preflight_dev.ps1   # PowerShell Core 사용
```

---

## 가장 빠른 실행 예시

### 1) 요약 출력 (wealth 모드)

**PowerShell**

```powershell
python -m project.runner.cli `
  --method rl `
  --market_mode bootstrap `
  --data_profile dev `
  --rl_epochs 0 --rl_n_paths_eval 80 --seed 42 `
  --print_mode summary `
  --metrics_keys "EW,ES95,Ruin,mean_WT" `
  --no_paths
```

**Bash**

```bash
python -m project.runner.cli \
  --method rl \
  --market_mode bootstrap \
  --data_profile dev \
  --rl_epochs 0 --rl_n_paths_eval 80 --seed 42 \
  --print_mode summary \
  --metrics_keys "EW,ES95,Ruin,mean_WT" \
  --no_paths
```

### 2) Loss 모드(ES95=CVaR) 파이프 확인

```powershell
python -m project.runner.cli `
  --method rl `
  --market_mode bootstrap `
  --data_profile dev `
  --es_mode loss --F_target 1.0 `
  --rl_epochs 0 --rl_n_paths_eval 80 --seed 7 `
  --print_mode summary `
  --metrics_keys "ES95,Ruin,mean_WT" `
  --no_paths
```

---

## 개발용 스모크(재현성 & 지터) 체크

**PowerShell**

```powershell
function RunJson($argv) {
  $o = .\.venv\Scripts\python.exe -m project.runner.cli @argv
  if ($LASTEXITCODE -ne 0) { throw "CLI failed ($LASTEXITCODE):`n$o" }
  $o | ConvertFrom-Json
}

$COMMON = @(
  "--method","rl","--asset","KR",
  "--market_mode","bootstrap","--data_profile","dev",
  "--use_real_rf","on","--outputs",".\outputs",
  "--rl_epochs","0","--rl_n_paths_eval","80","--seed","42",
  "--quiet","on","--print_mode","metrics",
  "--metrics_keys","EW,ES95,Ruin,mean_WT","--no_paths"
)

# fixed: 동일해야 함
$f1 = RunJson ($COMMON + @("--tag","dev_fixed1","--eval_seed_jitter","off"))
$f2 = RunJson ($COMMON + @("--tag","dev_fixed2","--eval_seed_jitter","off"))
if ($f1.EW -ne $f2.EW -or $f1.ES95 -ne $f2.ES95 -or $f1.mean_WT -ne $f2.mean_WT) { throw "FIXED MISMATCH" }

# jitter: 달라야 함
$j1 = RunJson ($COMMON + @("--tag","dev_j1","--eval_seed_jitter","on"))
$j2 = RunJson ($COMMON + @("--tag","dev_j2","--eval_seed_jitter","on"))
if ($j1.EW -eq $j2.EW -and $j1.ES95 -eq $j2.ES95 -and $j1.mean_WT -eq $j2.mean_WT) { throw "JITTER NO-DIFF" }

"DEV SMOKE: PASS"
```

* `--eval_seed_jitter on` 시 **평가시드에 시간 하위비트를 더해** 매 실행마다 약간 다른 결과를 강제합니다.
* `--eval_seed_jitter off` 시 완전 동일성을 보장합니다.

---

## 데이터 프로필 & 입력

* `--market_mode bootstrap`일 때 **다음 중 하나 필수**:

  * `--data_profile dev` : 개발용 축약 CSV(자동 생성됨)
  * `--data_profile full` : 전체 CSV (별도 보유 필요, 리포 미포함)
  * `--market_csv <path>` : 직접 경로 지정
* 로더는 스키마를 확인합니다(필수 컬럼: `date, ret_kr_eq, cpi_kr, rf_kr_nom`).
  `scripts/make_dev_csv.py`가 dev용 CSV를 만들어 줍니다.

---

## 출력 모드 요약

* `--print_mode full` : 전체 구조(JSON) 출력
* `--print_mode metrics` : 지정 키만 납작하게 출력

  * `--metrics_keys "EW,ES95,Ruin,mean_WT,es95_source"` 등
* `--print_mode summary` : 핵심 메타 + 선택 메트릭 요약
* `--no_paths` : 대용량 경로(`extra.eval_WT` 등) 생략

---

## 주요 옵션(발췌)

* `--es_mode {wealth,loss}` : wealth=부 기준 ES, loss=손실 (L=max(F−W,0)) 기반 ES
* `--F_target <float>` : loss 모드 기준선
* `--eval_seed_jitter {on,off}` : 평가 시드 지터 토글
* `--alpha_mix "a_kr,a_us,a_au"` : KR/US/Gold 혼합 가중(예: `--alpha_mix 0.4,0.4,0.2`)
  (미지정 시 균등 (1/3,1/3,1/3))
* `--h_FX <0..1>` / `--fx_hedge_cost <annual>` : 환헤지/비용
* `--use_real_rf {on,off}` : 실질무위험선호 토글
* RL 경량 실행: `--rl_epochs 0 --rl_n_paths_eval <N>` 로 평가 전용

---

## 로그 & 내보내기

* 모든 실행은 `outputs/_logs/metrics.csv`에 누적 기록(최신 설정보간).
* 필요한 경우 파이프라인으로 CSV/JSON 저장:

  ```powershell
  python -m project.runner.cli ... --print_mode metrics `
    --metrics_keys "EW,ES95,Ruin,mean_WT,es95_source" `
  | ConvertFrom-Json `
  | Export-Csv -NoTypeInformation -Encoding UTF8 -Path .\outputs\metrics_export.csv
  ```

---

## 테스트

```powershell
python -m pip install pytest
python -m pytest -q
```

---

## CI에 관하여 (선택)

* 데이터 파일(CSV)은 리포에 포함하지 않으므로, GitHub Actions는 기본적으로 **끄는 것을 권장**합니다.
* 필요 시 dev 프로필을 사용하는 **경량 스모크 워크플로우**만 활성화하세요(레포 보안/용량 정책에 따라 결정).

---

## 트러블슈팅

* `market_mode=bootstrap 사용 시 --market_csv 또는 --data_profile(dev|full) 필요.`
  → `--data_profile dev` 또는 `--market_csv <path>` 지정
* `schema missing required columns`
  → CSV 헤더 확인 (`date, ret_kr_eq, cpi_kr, rf_kr_nom` 필수). dev 프로필은 자동 충족.
* PowerShell에서 JSON 후처리 에러
  → `| ConvertFrom-Json` 이전에 `--print_mode metrics` 또는 `summary` 사용 권장.
* `pwsh` 명령을 찾지 못함
  → Windows PowerShell만 사용 중이면 `.\scripts\preflight_dev.ps1` 직접 실행(호출 시 `pwsh -File …` 대신).

---

## 변경 이력(요점)

* **평가 시드 지터** 도입: `--eval_seed_jitter on/off`
* **메트릭/요약 출력**: `--print_mode {metrics,summary}` + `--metrics_keys`
* **시장 메타/배선 고도화**: 혼합가중/환헤지/데이터 요약 자동 로그
* **dev 프로필**: 리포 내에서 즉시 실행 가능한 축약 CSV 파이프라인

> 질문/이슈는 PR 또는 이슈 트래커로 남겨주세요.

# 가상환경 파이썬 명령어
.\.venv\Scripts\python.exe
import os; os.system('cls')