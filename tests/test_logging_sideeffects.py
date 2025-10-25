import json, os, subprocess, sys, pathlib, stat

def run_full(args):
    cmd = [sys.executable, "-m", "project.runner.cli"] + args
    out = subprocess.check_output(cmd, text=True)
    return json.loads(out)

def test_metrics_csv_written_or_softfail(tmp_path):
    # outputs/_logs에 쓰기 권한 제거해도 프로세스는 살아야 함 (소프트 실패)
    outdir = tmp_path / "outs"
    logs = outdir / "_logs"
    logs.mkdir(parents=True)
    # 권한 제한
    try:
        logs.chmod(stat.S_IREAD)
    except Exception:
        pass  # 윈도우/CI에서 무시

    os.environ["SMOKE_OUT"] = str(outdir)
    res = run_full([
        "--method","rl","--asset","KR",
        "--market_mode","iid",
        "--outputs",str(outdir),
        "--rl_epochs","0","--rl_n_paths_eval","5","--seed","42",
        "--quiet","on","--print_mode","full","--no_paths","--tag","logtest"
    ])
    assert isinstance(res, dict)
