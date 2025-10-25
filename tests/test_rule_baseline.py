import json, subprocess, sys
PY = sys.executable
CLI = ["-m","project.runner.cli"]

def run_sum(args):
  out = subprocess.check_output([PY]+CLI+args, text=True)
  # summary 모드: 표준출력 파싱(간단 키:값)
  lines = [l.strip() for l in out.splitlines() if ":" in l]
  kv = {}
  for ln in lines:
    k,v = ln.split(":",1)
    kv[k.strip()] = v.strip()
  return kv

COMMON = [
  "--market_mode","bootstrap","--data_profile","dev",
  "--use_real_rf","on","--outputs","./outputs",
  "--print_mode","summary","--no_paths",
]

def test_vpw_sanity():
  kv = run_sum(["--method","rule","--baseline","vpw","--tag","t_vpw"]+COMMON)
  ew = float(kv["EW"])
  es95 = float(kv["ES95"])
  assert ew > 1.0
  assert es95 > 0.0
