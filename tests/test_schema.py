import json, os, subprocess, sys

CSV = os.environ.get("SMOKE_CSV", "project/data/market/kr_us_gold_bootstrap_mini.csv")

def run_json(args):
    cmd = [sys.executable, "-m", "project.runner.cli"] + args
    out = subprocess.check_output(cmd, text=True)
    return json.loads(out)

def test_metrics_schema():
    m = run_json([
        "--method","rl","--asset","KR",
        "--market_mode","bootstrap","--market_csv",CSV,
        "--use_real_rf","on","--outputs","outputs",
        "--rl_epochs","0","--rl_n_paths_eval","50","--seed","42",
        "--quiet","on","--print_mode","metrics",
        "--metrics_keys","EW,ES95,Ruin,mean_WT",
        "--no_paths","--tag","schema","--eval_seed_jitter","off"
    ])
    # 필수 키 & 타입
    for k in ["EW","ES95","Ruin","mean_WT"]:
        assert k in m, f"missing key {k}"
        assert isinstance(m[k], (int,float)), f"type mismatch {k}={type(m[k])}"
