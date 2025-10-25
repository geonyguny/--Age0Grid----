import json, os, subprocess, sys, math

CSV = os.environ.get("SMOKE_CSV", "project/data/market/kr_us_gold_bootstrap_mini.csv")

def run_full(args):
    cmd = [sys.executable, "-m", "project.runner.cli"] + args
    out = subprocess.check_output(cmd, text=True)
    return json.loads(out)

def test_market_quick_stats_present():
    out = run_full([
        "--method","rl","--asset","KR",
        "--market_mode","bootstrap","--market_csv",CSV,
        "--use_real_rf","on","--outputs","outputs",
        "--rl_epochs","0","--rl_n_paths_eval","10","--seed","42",
        "--quiet","on","--print_mode","full","--no_paths","--tag","stats"
    ])
    m = out["metrics"]
    for k in ["market_len_ret","market_len_rf","market_len_dates"]:
        assert k in m and isinstance(m[k], int) and m[k] >= 0
    for k in ["ret_mean","rf_mean"]:
        assert k in m and (m[k] is None or math.isfinite(m[k]))
