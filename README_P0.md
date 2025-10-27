# P0 재현 가이드 (요약)

## 요구 전제
- dev 데이터: `.\data\dev\market.csv` (현재는 `sample_market_600m.csv` 복사본)
- 가상환경: `.\.venv` 활성화

## 최소 실행
1) RL(기본)  
   `python -m project.runner.cli --method rl --data_profile dev --rl_epochs 0 --rl_n_paths_eval 5 --outputs .\outputs --tag dev_quick --print_mode summary --market_mode bootstrap --bootstrap_block 12 --data_window : --h_FX 0.5 --fx_hedge_cost 0`
2) Rule(VPW)  
   `python -m project.runner.cli --method rule --baseline vpw --data_profile dev --outputs .\outputs --tag chk_vpw --print_mode summary --market_mode bootstrap --bootstrap_block 12 --data_window :`
3) HJB 스모크  
   `python -m project.runner.cli --method hjb --data_profile dev --outputs .\outputs --tag hjb_smoke --print_mode summary --market_mode bootstrap --bootstrap_block 12 --data_window : --no_paths`

## 로그→요약
- 로그는 `outputs/_logs/*.log`
- `scripts\p0_check.ps1` 실행: 로그 파싱→`outputs/p0_summary.csv` 생성/검증→스냅샷 저장

## 산출물
- `outputs/p0_summary.csv` (RL/Rule/HJB 최소 1행씩)
- 스냅샷: `outputs/_snapshots/p0_summary_*.csv`
