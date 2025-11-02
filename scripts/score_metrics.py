# scripts/score_metrics.py
import pandas as pd, argparse, numpy as np, pathlib as p

def z(x):
    x = pd.to_numeric(x, errors="coerce")
    return (x - x.mean())/x.std(ddof=0) if x.std(ddof=0) not in [0, np.nan] else (x*0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--w_ew", type=float, default=0.4)
    ap.add_argument("--w_es", type=float, default=0.4)
    ap.add_argument("--w_ruin", type=float, default=0.2)
    args = ap.parse_args()

    df = pd.read_csv(args.src)
    for c in ["EW","ES95","Ruin"]:
        if c not in df.columns: df[c]=np.nan

    # 방향성: EW↑ 좋음, ES95↑ 좋음, Ruin↓ 나쁨 → 점수에서 부호 반전
    z_EW, z_ES, z_RU = z(df["EW"]), z(df["ES95"]), z(df["Ruin"])
    comp = args.w_ew*z_EW + args.w_es*z_ES + args.w_ruin*(-z_RU)
    df["CompositeScore"] = comp

    out = p.Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[OK] _summary_scored.csv -> {out}")

if __name__ == "__main__":
    main()
