# scripts/make_heatmaps.py
<<<<<<< HEAD
from __future__ import annotations

import argparse
import pathlib as p
import re
from typing import Iterable, Tuple

import numpy as np
import pandas as pd

# 백엔드 설정은 반드시 import pyplot 전에
=======
import argparse, pathlib as p
import numpy as np
import pandas as pd
>>>>>>> f7103a2 (report)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

<<<<<<< HEAD

# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────
TAG_PATTERNS = [
    # 예) 2D_us0.6_h0.5  또는  2D_us0.6_h0.5_anything
    re.compile(r"^2D_us(?P<u>[0-9.]+)_h(?P<h>[0-9.]+)(?:_.+)?$"),
]


def parse_mix(s: str) -> Tuple[float, float, float]:
    """
    alpha_mix 가 'kr,us,gold' 순서일 때 (us, kr, gold) 를 반환.
    기존 코드 호환 목적.
=======
def parse_mix(s: str):
    """
    alpha_mix가 'kr,us,gold' 순서라면 us,kr,gold를 반환해
    (질문 맥락의 기존 코드 호환: (us, kr, gold) 리턴)
>>>>>>> f7103a2 (report)
    """
    try:
        a, b, c = [float(x) for x in str(s).split(",")]
        return b, a, c  # us, kr, gold
    except Exception:
        return np.nan, np.nan, np.nan

<<<<<<< HEAD

def coerce_numeric(df: pd.DataFrame, cols: Iterable[str]) -> None:
=======
def coerce_numeric(df: pd.DataFrame, cols):
>>>>>>> f7103a2 (report)
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

<<<<<<< HEAD

def _fmt_tick(v) -> str:
    return f"{v:g}" if isinstance(v, (int, float, np.floating)) else str(v)


def _infer_from_alpha_mix(df: pd.DataFrame) -> pd.DataFrame:
    """alpha_mix에서 mix_us/mix_kr/mix_gold 파생."""
    if "alpha_mix" in df.columns:
        us, kr, au = zip(*df["alpha_mix"].map(parse_mix))
        df["mix_us"] = np.round(us, 2)
        df["mix_kr"] = np.round(kr, 2)
        df["mix_gold"] = np.round(au, 2)
    return df


def _infer_from_tag(df: pd.DataFrame) -> pd.DataFrame:
    """
    alpha_mix/hedge 컬럼이 없을 때 태그에서 유추.
    예: '2D_us0.6_h0.5' → mix_us=0.6, mix_kr=0.4, mix_gold=0.0, hedge_sigma_k=0.5
    """
    if "tag" not in df.columns:
        return df

    def _parse_one(tag: str):
        s = str(tag)
        for pat in TAG_PATTERNS:
            m = pat.match(s)
            if m:
                try:
                    u = float(m.group("u"))
                    h = float(m.group("h"))
                except Exception:
                    continue
                return {
                    "mix_us": round(u, 2),
                    "mix_kr": round(1.0 - u, 2),
                    "mix_gold": 0.0,
                    "hedge_sigma_k": h,
                }
        return {}

    parsed = df["tag"].map(_parse_one)
    extra = pd.DataFrame(list(parsed))
    for c in ("mix_us", "mix_kr", "mix_gold", "hedge_sigma_k"):
        if c not in df.columns and c in extra.columns:
            df[c] = extra[c]
    return df


def _backfill_hedge_sigma(df: pd.DataFrame) -> pd.DataFrame:
    """hedge_sigma_k 없고 hedge_ratio 있으면 보정."""
    if "hedge_sigma_k" not in df.columns and "hedge_ratio" in df.columns:
        df["hedge_sigma_k"] = pd.to_numeric(df["hedge_ratio"], errors="coerce")
    return df


def _pivot(df: pd.DataFrame, x: str, y: str, val: str, agg: str) -> pd.DataFrame:
    aggfunc = {"mean": "mean", "median": "median"}[agg]
    piv = df.pivot_table(index=y, columns=x, values=val, aggfunc=aggfunc)
    # 숫자 인덱스/컬럼은 숫자 정렬 적용
=======
def plot_heat(df, x, y, val, title, outpng):
    # 피벗(평균) → 값이 있는 축만 사용, 축 눈금은 숫자 정렬
    piv = df.pivot_table(index=y, columns=x, values=val, aggfunc="mean")
    # 정렬: x/y가 숫자라면 숫자 정렬
>>>>>>> f7103a2 (report)
    try:
        piv = piv.sort_index(axis=0)
    except Exception:
        pass
    try:
        piv = piv.sort_index(axis=1)
    except Exception:
        pass
<<<<<<< HEAD
    return piv


def plot_heat(
    piv: pd.DataFrame,
    x: str,
    y: str,
    val: str,
    title: str,
    outpng: p.Path,
    annotate: bool,
    figsize: Tuple[float, float],
    dpi: int,
    cmap_name: str,
) -> None:
    # NaN 마스킹 후 회색으로 표시
    data = np.array(piv.values, dtype=float)
    mask = np.isnan(data)
    mdata = np.ma.masked_array(data, mask=mask)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    cmap = getattr(plt.cm, cmap_name, plt.cm.viridis)
    cmap = cmap.copy()
    cmap.set_bad("lightgray")
    im = ax.imshow(mdata, aspect="auto", origin="lower", cmap=cmap)

    xt = list(range(len(piv.columns)))
    yt = list(range(len(piv.index)))
    ax.set_xticks(xt)
    ax.set_xticklabels([_fmt_tick(v) for v in piv.columns])
    ax.set_yticks(yt)
    ax.set_yticklabels([_fmt_tick(v) for v in piv.index])

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)

    if annotate:
        for i in yt:
            for j in xt:
                if not mask[i, j]:
                    ax.text(j, i, f"{mdata[i, j]:.2f}", ha="center", va="center", fontsize=8)

=======

    fig, ax = plt.subplots(figsize=(6.0, 4.6), dpi=180)
    im = ax.imshow(piv.values, aspect="auto", origin="lower")
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels([f"{v:g}" if isinstance(v, (int,float,np.floating)) else str(v) for v in piv.columns])
    ax.set_yticks(range(len(piv.index)));   ax.set_yticklabels([f"{v:g}" if isinstance(v, (int,float,np.floating)) else str(v) for v in piv.index])
    ax.set_xlabel(x); ax.set_ylabel(y); ax.set_title(title)
>>>>>>> f7103a2 (report)
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(val, rotation=90, va="center")

    outpng.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpng, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {outpng}")

<<<<<<< HEAD

# ─────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────
def build_heatmaps(
    src: str,
    tag_startswith: str,
    x: str,
    y: str,
    zlist: Iterable[str],
    outdir: str,
    infer_from_tag: bool,
    annotate: bool,
    agg: str,
    save_pivots: bool,
    dpi: int,
    figsize: Tuple[float, float],
    cmap: str,
) -> None:
    df = pd.read_csv(src)

    if "tag" not in df.columns:
        raise SystemExit("[ERR] 데이터에 'tag' 컬럼이 없습니다.")
    df = df[df["tag"].astype(str).str.startswith(tag_startswith)]
    if df.empty:
        raise SystemExit(f"[ERR] '{tag_startswith}' 로 시작하는 tag 데이터가 없습니다.")

    # 파생/보정
    df = _infer_from_alpha_mix(df)
    df = _backfill_hedge_sigma(df)
    if infer_from_tag:
        df = _infer_from_tag(df)

    # 숫자화
    coerce_numeric(
        df,
        [
            x,
            y,
            "EW",
            "ES95",
            "CompositeScore",
            "Ruin",
            "hedge_ratio",
            "hedge_sigma_k",
            "mix_us",
            "mix_kr",
            "mix_gold",
        ],
    )

    # 유효성
    for c in (x, y):
        if c not in df.columns:
            raise SystemExit(f"[ERR] 데이터에 '{c}' 컬럼이 없음. --x/--y와 원본 컬럼명을 확인하세요.")

    # 진단
    print(
        f"[INFO] rows={len(df)} | x={x} y={y} | zlist={list(zlist)} | "
        f"infer_from_tag={infer_from_tag} | agg={agg}"
    )

    outdir_path = p.Path(outdir)

    for z in zlist:
        if z not in df.columns:
            print(f"[SKIP] '{z}' 컬럼 없음")
            continue

        sub = df[[x, y, z]].dropna()
        if sub.empty:
            print(f"[SKIP] '{z}' 데이터 없음")
            continue

        piv = _pivot(sub, x, y, z, agg)

        # 원하면 피벗 CSV도 남김
        if save_pivots:
            csv_path = outdir_path / f"heatmap_{tag_startswith}_{x}_vs_{y}_{z}_pivot.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            piv.to_csv(csv_path)
            print(f"[OK] {csv_path}")

        outpng = outdir_path / f"heatmap_{tag_startswith}_{x}_vs_{y}_{z}.png"
        plot_heat(
            piv,
            x,
            y,
            z,
            f"Heatmap: {z} ({x} × {y})",
            outpng,
            annotate=annotate,
            figsize=figsize,
            dpi=dpi,
            cmap_name=cmap,
        )


# ─────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=r".\outputs\_summary_scored.csv")
    ap.add_argument("--tag_startswith", default="2D_", help="이 접두사로 시작하는 tag만 사용")
    ap.add_argument("--x", default="mix_us", help="x축 컬럼명 (예: mix_us, hedge_sigma_k 등)")
    ap.add_argument("--y", default="hedge_sigma_k", help="y축 컬럼명")
    ap.add_argument(
        "--zlist",
        default="EW,ES95",
        help="콤마 구분 메트릭 목록 (예: EW,ES95,CompositeScore)",
    )
    ap.add_argument("--outdir", default=r".\outputs\figs")
    ap.add_argument(
        "--infer_from_tag",
        choices=["on", "off"],
        default="on",
        help="alpha_mix/hedge 컬럼 없을 때 태그에서 (us,h) 추론",
    )
    ap.add_argument("--annotate", choices=["on", "off"], default="off", help="각 셀에 값 표기")
    ap.add_argument("--agg", choices=["mean", "median"], default="mean", help="피벗 집계 방식")
    ap.add_argument("--save_pivots", choices=["on", "off"], default="off", help="피벗 CSV 저장")
    ap.add_argument("--dpi", type=int, default=180)
    ap.add_argument("--fig_w", type=float, default=6.6)
    ap.add_argument("--fig_h", type=float, default=4.8)
    ap.add_argument("--cmap", default="viridis", help="matplotlib 컬러맵 이름")
    args = ap.parse_args()

    # normalize flags
    args.infer_from_tag = args.infer_from_tag == "on"
    args.annotate = args.annotate == "on"
    args.save_pivots = args.save_pivots == "on"
    args.zlist = [z.strip() for z in str(args.zlist).split(",") if z.strip()]
    if not args.zlist:
        raise SystemExit("[ERR] zlist가 비어 있습니다.")
    return args


def main() -> None:
    args = parse_args()
    build_heatmaps(
        src=args.src,
        tag_startswith=args.tag_startswith,
        x=args.x,
        y=args.y,
        zlist=args.zlist,
        outdir=args.outdir,
        infer_from_tag=args.infer_from_tag,
        annotate=args.annotate,
        agg=args.agg,
        save_pivots=args.save_pivots,
        dpi=args.dpi,
        figsize=(args.fig_w, args.fig_h),
        cmap=args.cmap,
    )


if __name__ == "__main__":
    main()
=======
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
>>>>>>> f7103a2 (report)
