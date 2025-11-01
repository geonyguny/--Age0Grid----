import itertools, subprocess, sys, os, csv, json

PY = r".\.venv\Scripts\python.exe"
CLI = "project.runner.cli"
OUTDIR = r".\outputs"
TAG = "compare"
DATA_PROFILE = "dev"
MARKET_MODE = "bootstrap"
N_EVAL = "20000"

def run(args: list[str]):
    cmd = [PY, "-m", CLI] + args
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    os.makedirs("scripts", exist_ok=True)

    # (1) OAT sweeps
    for kr in [0.0,0.2,0.4,0.6,0.8,1.0]:
        us = max(0.0, 1.0-kr); au = 0.0
        run([
            "--method","rule","--baseline","vpw",
            "--market_mode", MARKET_MODE, "--data_profile", DATA_PROFILE,
            "--alpha_mix", f"{kr},{us},{au}",
            "--tag", f"mix_{kr}", "--n_paths", N_EVAL, "--print_mode","summary"
        ])

    for w in [0.30,0.50,0.70,0.90,1.00]:
        run([
            "--method","hjb","--market_mode",MARKET_MODE,"--data_profile",DATA_PROFILE,
            "--w_max",str(w),"--tag",f"wmax_{w}",
            "--n_paths",N_EVAL,"--print_mode","summary"
        ])

    for q in [0.01,0.02,0.03,0.04]:
        run([
            "--method","rl","--market_mode",MARKET_MODE,"--data_profile",DATA_PROFILE,
            "--q_floor",str(q),"--tag",f"qfloor_{q}",
            "--n_paths",N_EVAL,"--print_mode","summary"
        ])

    for h in [0.0,0.25,0.50,0.75,1.0]:
        run([
            "--method","rule","--baseline","4pct",
            "--market_mode",MARKET_MODE,"--data_profile",DATA_PROFILE,
            "--hedge","on","--hedge_mode","sigma","--hedge_sigma_k",str(h),
            "--tag",f"hedge_{h}","--n_paths",N_EVAL,"--print_mode","summary"
        ])

    for t in [0.0,0.5,1.0,2.0]:
        run([
            "--method","hjb","--market_mode",MARKET_MODE,"--data_profile",DATA_PROFILE,
            "--theta_ambiguity",str(t),
            "--tag",f"amb_{t}","--n_paths",N_EVAL,"--print_mode","summary"
        ])

    # (2) 2D heatmap
    for w,q in itertools.product([0.30,0.50,0.70,0.90],[0.01,0.02,0.03,0.04]):
        run([
            "--method","rl","--market_mode",MARKET_MODE,"--data_profile",DATA_PROFILE,
            "--w_max",str(w),"--q_floor",str(q),
            "--tag",f"w{w}_q{q}","--n_paths",N_EVAL,"--print_mode","summary"
        ])

    # (3) λ sweep
    for lam in [0.25,0.5,0.8,1.2,1.6,2.0]:
        run([
            "--method","rl","--market_mode",MARKET_MODE,"--data_profile",DATA_PROFILE,
            "--alpha","0.95","--lambda_term",str(lam),
            "--calib_fast","on","--calib_max_iter","8",
            "--tag",f"lambda_{lam}","--n_paths",N_EVAL,"--print_mode","summary"
        ])

    # (4) Figure/Report
    subprocess.run([PY, "scripts/make_paper_figs.py", "--outdir", OUTDIR, "--tag", TAG], check=True)
    print(f"DONE. 보고?? {OUTDIR}/ALM_Executive_Report_{TAG}.xlsx")

if __name__ == "__main__":
    main()
