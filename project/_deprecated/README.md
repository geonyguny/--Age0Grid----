# 이 폴더에 대해

이 폴더의 파일들은 실제 코드베이스 전체를 `grep`으로 전수 조사한 결과,
**어디에서도 import되지 않는 것이 확인된 파일**들입니다. 삭제하지 않고
여기로 옮겨두었으니, 필요하면 언제든 원래 위치로 되돌리시면 됩니다.

## 옮긴 이유 (파일별)

- `env.py.bak` : `project/env/` 패키지(retirement_env.py 포함)가 실제로 쓰이고
  있고, 같은 이름의 이 플랫 파일은 파이썬 임포트 규칙상 애초에 도달 불가능한
  코드였습니다 (`import project.env`는 항상 패키지 쪽으로 resolve됨).

- `rl.py.bak` : 코드베이스 어디서도 import하는 곳이 없었습니다.

- `trainer/rl_a2c.py.bak` : `runner/run.py`가 실제로 호출하는 건
  `trainer/rl_trainer.py`의 `RLTrainer` 클래스였고, 이 파일의 `train_rl()`
  함수는 호출되는 곳이 없었습니다.

- `trainer/rl_loss.py.bak`, `trainer/policy_io.py.bak` : 마찬가지로 어디서도
  import되지 않았습니다.

- `baselines.py.bak`, `policy/kgr_rule.py.bak` : 규칙기반(4%룰 등) 로직은
  `runner/run.py` 안에 직접 인라인으로 재구현되어 있고, 이 두 파일은
  더 이상 호출되지 않는 초기 버전으로 보입니다.

- `stats_module/`, `report_module/` : 코드베이스 전체에서 import하는 곳을
  찾지 못했습니다. 다만 이건 앞으로 리포트/통계 기능을 붙일 때 쓰실 계획이
  있으실 수도 있어서, 삭제 대신 폴더째로 옮겨만 두었습니다.

## 함께 지운 것 (부산물, 재생성 가능하므로 완전 삭제)

- 모든 `__pycache__/` 폴더
- `project/data/market/_cache/*.npz` (부트스트랩 계산 캐시 — 다시 돌리면 자동 재생성됨)
- `project/data/market_test_600m.csv` (project/data/market/ 안에 있는 동일 파일과 완전히 같은 내용의 중복본)

## 주의

혹시 이 폴더로 옮긴 파일 중 실제로 어딘가에서 아직 쓰고 계신 게 있다면
(예: 별도 스크립트에서 `python project/rl.py`처럼 직접 실행하시는 경우),
말씀해주시면 원위치로 되돌리거나 정식 경로로 다시 연결해드리겠습니다.
