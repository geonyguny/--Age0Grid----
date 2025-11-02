# scripts/make_heatmaps.py
import argparse, pathlib as p
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def parse_mix(s: str):
    """
    alpha_mix가 'kr,us,gold' 순서라면 us,kr,gold를 반환해
    (질문 맥락의 기존 코드 호환: (us, kr, gold) 리턴)
    """
    try:
        a, b, c = [float(x) for x in str(s).split(",")]
        return b, a, c  # us, kr, gold
    except Exception:
        return np.nan, np.nan, np.nan

def coerce_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

def plot_heat(df, x, y, val, title, outpng):
    # 피벗(평균) → 값이 있는 축만 사용, 축 눈금은 숫자 정렬
    piv = df.pivot_table(index=y, columns=x, values=val, aggfunc="mean")
    # 정렬: x/y가 숫자라면 숫자 정렬
    try:
        piv = piv.sort_index(axis=0)
    except Exception:
        pass
    try:
        piv = piv.sort_index(axis=1)
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=(6.0, 4.6), dpi=180)
    im = ax.imshow(piv.values, aspect="auto", origin="lower")
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels([f"{v:g}" if isinstance(v, (int,float,np.floating)) else str(v) for v in piv.columns])
    ax.set_yticks(range(len(piv.index)));   ax.set_yticklabels([f"{v:g}" if isinstance(v, (int,float,np.floating)) else str(v) for v in piv.index])
    ax.set_xlabel(x); ax.set_ylabel(y); ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(val, rotation=90, va="center")

    outpng.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpng, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {outpng}")

ap = argparse.ArgumentParser()
ap.add_argument("--src", default=r".\outputs\_summary_scored.csv")
ap.add_argument("--tag_startswith", default="2D_", help="이 접두사로 시작하는 tag만 사용")
ap.add_argument("--x", default="mix_us", help="x축 컬럼명 (예: mix_us, hedge_sigma_k 등)")
ap.add_argument("--y", default="hedge_sigma_k", help="y축 컬럼명")
ap.add_argument("--zlist", default="EW,ES95", help="콤마 구분 메트릭 목록 (예: EW,ES95,CompositeScore)")
ap.add_argument("--outdir", default=r".\outputs\figs")
args = ap.parse_args()

df = pd.read_csv(args.src)

# 태그 필터
df = df[df["tag"].astype(str).str.startswith(args.tag_startswith)]

# 파생/보정 컬럼
if "alpha_mix" in df.columns:
    us, kr, au = zip(*df["alpha_mix"].map(parse_mix))
    df["mix_us"]   = np.round(us, 2)
    df["mix_kr"]   = np.round(kr, 2)
    df["mix_gold"] = np.round(au, 2)

# hedge_sigma_k 대체: hedge_ratio가 있으면 그것으로 채움
if "hedge_sigma_k" not in df.columns and "hedge_ratio" in df.columns:
    df["hedge_sigma_k"] = pd.to_numeric(df["hedge_ratio"], errors="coerce")

# 숫자화 가능 항목 숫자화
coerce_numeric(df, [args.x, args.y, "EW", "ES95", "CompositeScore", "Ruin", "hedge_ratio", "hedge_sigma_k"])

# 유효성 체크
need = [args.x, args.y]
for c in need:
    if c not in df.columns:
        raise SystemExit(f"[ERR] 데이터에 '{c}' 컬럼이 없음. --x/--y 설정과 원본 컬럼명을 확인하세요.")

zcols = [z.strip() for z in args.zlist.split(",") if z.strip()]
if not zcols:
    raise SystemExit("[ERR] zlist가 비어 있음")

outdir = p.Path(args.outdir)

for z in zcols:
    if z not in df.columns:
        print(f"[SKIP] '{z}' 컬럼 없음")
        continue
    sub = df[[args.x, args.y, z]].dropna()
    if sub.empty:
        print(f"[SKIP] '{z}' 데이터 없음")
        continue
    outpng = outdir / f"heatmap_{args.tag_startswith}_{args.x}_vs_{args.y}_{z}.png"
    plot_heat(sub, args.x, args.y, z, f"Heatmap: {z} ({args.x} × {args.y})", outpng)
