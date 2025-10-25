import json, os, subprocess, sys

CSV = os.environ.get("SMOKE_CSV", "project/data/market/kr_us_gold_bootstrap_mini.csv")

def run_full(args):
    cmd = [sys.executable, "-m", "project.runner.cli"] + args
    out = subprocess.check_output(cmd, text=True)
    return json.loads(out)

def test_eval_seed_meta_fixed_and_jitter():
    base = [
        "--method","rl","--asset","KR",
        "--market_mode","bootstrap","--market_csv",CSV,
        "--use_real_rf","on","--outputs","outputs",
        "--rl_epochs","0","--rl_n_paths_eval","50","--seed","42",
        "--quiet","on","--print_mode","full","--no_paths"
    ]
    fixed = run_full(base + ["--tag","meta_fixed","--eval_seed_jitter","off"])
    jitter = run_full(base + ["--tag","meta_jitter","--eval_seed_jitter","on"])
    assert fixed["meta"]["eval_seed_mode"] == "fixed"
    assert jitter["meta"]["eval_seed_mode"] == "jitter"
    assert isinstance(fixed["meta"]["eval_seed_base"], int)
    assert isinstance(jitter["meta"]["eval_seed_base"], int)
