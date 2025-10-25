#!/usr/bin/env python
import json, os, subprocess, sys, shlex

CSV = os.environ.get("SMOKE_CSV", "project/data/market/kr_us_gold_bootstrap_mini.csv")
OUTPUTS = os.environ.get("SMOKE_OUT", "outputs")

COMMON = [
    "--method","rl","--asset","KR",
    "--market_mode","bootstrap","--market_csv",CSV,
    "--use_real_rf","on","--outputs",OUTPUTS,
    "--rl_epochs","0","--rl_n_paths_eval","200","--seed","42",
    "--quiet","on","--print_mode","metrics",
    "--metrics_keys","EW,ES95,Ruin,mean_WT","--no_paths"
]

def run_json(args):
    cmd = [sys.executable, "-m", "project.runner.cli"] + args
    out = subprocess.check_output(cmd, text=True)
    try:
        return json.loads(out)
    except json.JSONDecodeError as e:
        print(out)
        raise

def main():
    f1 = run_json(COMMON + ["--tag","reg_fixed1","--eval_seed_jitter","off"])
    f2 = run_json(COMMON + ["--tag","reg_fixed2","--eval_seed_jitter","off"])
    assert f1["EW"] == f2["EW"] and f1["ES95"] == f2["ES95"] and f1["mean_WT"] == f2["mean_WT"], "FIXED MISMATCH"

    j1 = run_json(COMMON + ["--tag","reg_jitter1","--eval_seed_jitter","on"])
    j2 = run_json(COMMON + ["--tag","reg_jitter2","--eval_seed_jitter","on"])
    assert not (j1["EW"] == j2["EW"] and j1["ES95"] == j2["ES95"] and j1["mean_WT"] == j2["mean_WT"]), "JITTER NO-DIFF"

    print("SMOKE REGRESSION: PASS")

if __name__ == "__main__":
    sys.exit(main())
