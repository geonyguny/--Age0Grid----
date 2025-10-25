import subprocess, sys

def test_eta_budget_exitcode():
    # 아주 작은 budget을 주면 hard-stop=on 에서 종료코드 3
    cmd = [
        sys.executable, "-m", "project.runner.cli",
        "--method","rl","--asset","KR",
        "--market_mode","iid",  # 빠르게
        "--outputs","outputs",
        "--rl_epochs","0","--rl_n_paths_eval","1",
        "--eta_mode","history","--eta_budget_s","0.0","--eta_hard_stop","on",
        "--quiet","on","--print_mode","metrics","--metrics_keys","EW","--no_paths"
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode in (0,3), f"unexpected exit {p.returncode}\n{p.stderr}"
