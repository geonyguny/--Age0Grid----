param(
  [string]$Root = ".\outputs"
)

$ErrorActionPreference = "Stop"

# 1) 경로 정리
$rootAbs = Resolve-Path $Root | Select-Object -ExpandProperty Path
Write-Host "[run] Root = $rootAbs"

# 2) 임시 파이썬 스크립트 생성
$py = @"
import os, sys, csv, json
import pandas as pd

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
root = os.path.abspath(ROOT)

# 우선순위: _summary_scored.csv -> _summary.csv
p_scored = os.path.join(root, "_summary_scored.csv")
p_summary = os.path.join(root, "_summary.csv")
p_pair   = os.path.join(root, "_pairwise_vs_best.csv")

def _safe_read_csv(p):
    if not os.path.exists(p): return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None

def _minmax(s):
    try:
        v = pd.to_numeric(s, errors="coerce")
        lo, hi = v.min(), v.max()
        if pd.isna(lo) or pd.isna(hi): return pd.Series([0.5]*len(v))
        if hi - lo == 0: return pd.Series([0.5]*len(v))
        return (v - lo) / (hi - lo)
    except Exception:
        return pd.Series([0.5]*len(s))

def score_df(df):
    # 필수 컬럼이 없으면 가능한 것만 사용
    need_cols = ["EW","ES95","RuinPct","WinRate"]
    for c in need_cols:
        if c not in df.columns: df[c] = None

    s_ew   = _minmax(df["EW"])
    s_es   = _minmax(-pd.to_numeric(df["ES95"], errors="coerce"))   # ES95는 낮을수록 좋음 → 부호 반전
    s_ruin = _minmax(-pd.to_numeric(df["RuinPct"], errors="coerce"))# Ruin도 낮을수록 좋음 → 부호 반전
    s_win  = _minmax(df["WinRate"])

    # 가중치 (원하면 여기 수정)
    w = {"EW":0.35, "ES95":0.35, "RuinPct":0.20, "WinRate":0.10}
    comp = w["EW"]*s_ew + w["ES95"]*s_es + w["RuinPct"]*s_ruin + w["WinRate"]*s_win

    out = df.copy()
    out["Score_EW"]   = s_ew
    out["Score_ES95"] = s_es
    out["Score_Ruin"] = s_ruin
    out["Score_Win"]  = s_win
    out["CompositeScore"] = comp
    return out

# 3) 스코어링 소스 확보
df = _safe_read_csv(p_scored)
if df is None:
    base = _safe_read_csv(p_summary)
    if base is None:
        print(f"[ERR] not found: {p_scored} nor {p_summary}", file=sys.stderr)
        sys.exit(2)
    # 최소 컬럼만 있어도 동작하도록 방어
    df = score_df(base)
    df.to_csv(p_scored, index=False, encoding="utf-8-sig")
    print(f"[ok] wrote {p_scored}")
else:
    # CompositeScore 없으면 보강
    if "CompositeScore" not in df.columns:
        df = score_df(df)
        df.to_csv(p_scored, index=False, encoding="utf-8-sig")
        print(f"[ok] normalized & wrote {p_scored}")
    else:
        print(f"[ok] found {p_scored}")

# 4) pairwise vs best 생성 (panel·method 묶음 내 best 대비 Δ)
df = _safe_read_csv(p_scored)
panel_cols = [c for c in ["panel","Panel","tag","method"] if c in df.columns]
if not set(["method"]).issubset(panel_cols):
    # 최소한 method라도 있어야 함
    df["panel_key"] = "ALL"
    panel_cols = ["panel_key","method"]

grp_keys = [c for c in panel_cols if c != "CompositeScore"]

rows = []
for keys, g in df.groupby(grp_keys, dropna=False):
    # 같은 패널(또는 전체) 한정 best
    # 단, keys는 튜플일 수 있음 → dict로 바꿔 저장
    best = g.loc[g["CompositeScore"].idxmax()]
    for _, r in g.iterrows():
        d = {k:v for k,v in zip(grp_keys, keys if isinstance(keys, tuple) else (keys,))}
        d.update({
            "method": r["method"],
            "CompositeScore": r["CompositeScore"],
            "BestScore": best["CompositeScore"],
            "DeltaToBest": r["CompositeScore"] - best["CompositeScore"],
            "IsBest": bool(r["CompositeScore"] == best["CompositeScore"]),
        })
        rows.append(d)

pd.DataFrame(rows).to_csv(p_pair, index=False, encoding="utf-8-sig")
print(f"[ok] wrote {p_pair}")
"@

# 3) 임시 파일에 쓰고 실행
$tmpPy = Join-Path $env:TEMP "run_sim_pipeline_tmp.py"
Set-Content -Path $tmpPy -Value $py -Encoding UTF8

# 가상환경 python 경로 우선
$pyexe = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
if (-not (Test-Path $pyexe)) { $pyexe = "python" }

& $pyexe $tmpPy $rootAbs
$exit = $LASTEXITCODE

Remove-Item $tmpPy -ErrorAction SilentlyContinue

if ($exit -ne 0) { exit $exit }
Write-Host "[done] pipeline finished."
