# scripts/make_heatmaps.py
from __future__ import annotations

import argparse, json
import pathlib as p
import re
from typing import Iterable, Tuple, Dict, Optional, List

import numpy as np
import pandas as pd

# pyplot import 전에 백엔드 지정
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────
TAG_PATTERNS = [
    # 예) 2D_us0.6_h0.5  또는  2D_us0.6_h0.5_anything
    re.compile(r"^2D_us(?P<u>[0-9.]+)_h(?P<h>[0-9.]+)(?:_.+)?$"),
]

def parse_mix(s: str) -> Tuple[float, float, float]:
    """alpha_mix 가 'kr,us,gold' 순서일 때 (us, kr, gold) 를 반환 (기존 코드 호환)."""
    try:
        a, b, c = [float(x) for x in str(s).split(",")]
        return b, a, c  # us, kr, gold
    except Exception:
        return np.nan, np.nan, np.nan

def coerce_numeric(df: pd.DataFrame, cols: Iterable[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

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
                    u = float(m.group("u")); h = float(m.group("h"))
                except Exception:
                    continue
                return {"mix_us": round(u, 2), "mix_kr": round(1.0 - u, 2),
                        "mix_gold": 0.0, "hedge_sigma_k": h}
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

def _round_if_numeric(s: pd.Series, n: Optional[int]) -> pd.Series:
    if n is None:
        return s
    try:
        v = pd.to_numeric(s, errors="coerce").round(n)
        return v
    except Exception:
        return s

def _apply_order(idx: List, order_str: Optional[str]) -> List:
    """order_str='0.2|0.6|0.8' 같은 순서를 강제. 누락값은 뒤에 붙임."""
    if not order_str:
        return idx
    want = [x.strip() for x in order_str.split("|") if x.strip() != ""]
    want_set = set(want)
    head = [x for x in want if x in idx]
    tail = [x for x in idx if x not in want_set]
    return head + tail

def _pivot(df: pd.DataFrame, x: str, y: str, val: str, agg: str,
           round_x: Optional[int], round_y: Optional[int],
           x_order: Optional[str], y_order: Optional[str]) -> pd.DataFrame:
    # 라운딩(선택)
    if round_x is not None and x in df.columns:
        df = df.copy()
        df[x] = _round_if_numeric(df[x], round_x)
    if round_y is not None and y in df.columns:
        if df is not None:
            df = df.copy()
        df[y] = _round_if_numeric(df[y], round_y)

    aggfunc = {"mean": "mean", "median": "median"}[agg]
    piv = df.pivot_table(index=y, columns=x, values=val, aggfunc=aggfunc)

    # 숫자 정렬
    for axis, is_col in ((0, False), (1, True)):
        try:
            if is_col:
                piv = piv.reindex(sorted(piv.columns, key=lambda t: (isinstance(t, str), t)))
            else:
                piv = piv.reindex(sorted(piv.index, key=lambda t: (isinstance(t, str), t)))
        except Exception:
            pass

    # 순서 강제
    if x_order:
        cols = list(piv.columns)
        cols = _apply_order(cols, x_order)
        piv = piv.reindex(columns=cols)
    if y_order:
        idx = list(piv.index)
        idx = _apply_order(idx, y_order)
        piv = piv.reindex(index=idx)

    return piv

def _parse_vmin_max(s: Optional[str]) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    """
    'ES95:0.25,1.0;EW:0.25,1.25;CompositeScore:-0.6,1.9'
      → {'ES95': (0.25,1.0), 'EW': (0.25,1.25), 'CompositeScore': (-0.6,1.9)}
    빈칸/부적절 값은 None으로 처리.
    """
    out: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    if not s:
        return out
    for part in str(s).split(";"):
        part = part.strip()
        if not part or ":" not in part:
            continue
        k, rng = part.split(":", 1)
        k = k.strip()
        vmin, vmax = None, None
        try:
            xs = [t.strip() for t in rng.split(",")]
            if len(xs) >= 1 and xs[0] != "":
                vmin = float(xs[0])
            if len(xs) >= 2 and xs[1] != "":
                vmax = float(xs[1])
        except Exception:
            pass
        out[k] = (vmin, vmax)
    return out

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
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    xtick_rotate: int = 0,
    ytick_rotate: int = 0,
    interp: str = "nearest",  # 'nearest' or 'bilinear'
) -> None:
    # NaN 마스킹 후 회색 처리
    data = np.array(piv.values, dtype=float)
    mask = np.isnan(data)
    mdata = np.ma.masked_array(data, mask=mask)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    cmap = getattr(plt.cm, cmap_name, plt.cm.viridis).copy()
    cmap.set_bad("lightgray")

    im = ax.imshow(
        mdata, aspect="auto", origin="lower",
        cmap=cmap, vmin=vmin, vmax=vmax, interpolation=interp
    )

    xt = list(range(len(piv.columns)))
    yt = list(range(len(piv.index)))
    ax.set_xticks(xt)
    ax.set_xticklabels([_fmt_tick(v) for v in piv.columns], rotation=xtick_rotate)
    ax.set_yticks(yt)
    ax.set_yticklabels([_fmt_tick(v) for v in piv.index], rotation=ytick_rotate)

    ax.set_xlabel(x); ax.set_ylabel(y); ax.set_title(title)

    if annotate:
        for i in yt:
            for j in xt:
                if not mask[i, j]:
                    ax.text(j, i, f"{mdata[i, j]:.2f}", ha="center", va="center", fontsize=8)

    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(val, rotation=90, va="center")

    outpng.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpng, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {outpng}")


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
    vmin_max: Optional[str],
    round_x: Optional[int],
    round_y: Optional[int],
    x_order: Optional[str],
    y_order: Optional[str],
    xtick_rotate: int,
    ytick_rotate: int,
    interp: str,
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
    coerce_numeric(df, [x, y, "EW", "ES95", "CompositeScore", "Ruin",
                        "hedge_ratio", "hedge_sigma_k", "mix_us", "mix_kr", "mix_gold"])

    # 유효성
    for c in (x, y):
        if c not in df.columns:
            raise SystemExit(f"[ERR] 데이터에 '{c}' 컬럼이 없음. --x/--y와 원본 컬럼명을 확인하세요.")

    # vmin/vmax 파싱
    vmm = _parse_vmin_max(vmin_max)

    # 진단
    print(f"[INFO] rows={len(df)} | x={x} y={y} | zlist={list(zlist)} | "
          f"infer_from_tag={infer_from_tag} | agg={agg} | round_x={round_x} round_y={round_y}")

    outdir_path = p.Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)

    # 실행 설정 로그
    (outdir_path / f"heatmap_{tag_startswith}_config.json").write_text(
        json.dumps({
            "src": src, "tag_startswith": tag_startswith, "x": x, "y": y,
            "zlist": list(zlist), "infer_from_tag": infer_from_tag, "annotate": annotate,
            "agg": agg, "dpi": dpi, "figsize": list(figsize), "cmap": cmap,
            "vmin_max": vmin_max, "round_x": round_x, "round_y": round_y,
            "x_order": x_order, "y_order": y_order,
            "xtick_rotate": xtick_rotate, "ytick_rotate": ytick_rotate, "interp": interp,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    for z in zlist:
        if z not in df.columns:
            print(f"[SKIP] '{z}' 컬럼 없음")
            continue

        sub = df[[x, y, z]].dropna()
        if sub.empty:
            print(f"[SKIP] '{z}' 데이터 없음")
            continue

        piv = _pivot(sub, x, y, z, agg, round_x, round_y, x_order, y_order)

        # 피벗 CSV 저장 옵션 + 원천 서브셋도 저장
        if save_pivots:
            csv_path = outdir_path / f"heatmap_{tag_startswith}_{x}_vs_{y}_{z}_pivot.csv"
            piv.to_csv(csv_path); print(f"[OK] {csv_path}")
            sub_path = outdir_path / f"heatmap_{tag_startswith}_{x}_vs_{y}_{z}_subset.csv"
            sub.to_csv(sub_path, index=False); print(f"[OK] {sub_path}")

        outpng = outdir_path / f"heatmap_{tag_startswith}_{x}_vs_{y}_{z}.png"
        vmin, vmax = vmm.get(z, (None, None))
        plot_heat(
            piv, x, y, z, f"Heatmap: {z} ({x} × {y})", outpng,
            annotate=annotate, figsize=figsize, dpi=dpi, cmap_name=cmap,
            vmin=vmin, vmax=vmax, xtick_rotate=xtick_rotate, ytick_rotate=ytick_rotate,
            interp=interp
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
    ap.add_argument("--zlist", default="EW,ES95,CompositeScore",
                    help="콤마 구분 메트릭 목록 (예: EW,ES95,CompositeScore)")
    ap.add_argument("--outdir", default=r".\outputs\figs")
    ap.add_argument("--infer_from_tag", choices=["on", "off"], default="on",
                    help="alpha_mix/hedge 컬럼 없을 때 태그에서 (us,h) 추론")
    ap.add_argument("--annotate", choices=["on", "off"], default="off", help="각 셀에 값 표기")
    ap.add_argument("--agg", choices=["mean", "median"], default="mean", help="피벗 집계 방식")
    ap.add_argument("--save_pivots", choices=["on", "off"], default="off", help="피벗 CSV 저장")
    ap.add_argument("--dpi", type=int, default=180)
    ap.add_argument("--fig_w", type=float, default=6.6)
    ap.add_argument("--fig_h", type=float, default=4.8)
    ap.add_argument("--cmap", default="viridis", help="matplotlib 컬러맵 이름")
    ap.add_argument("--vmin_max", default="",
                    help="색상 스케일: 'ES95:0.25,1.0;EW:0.25,1.25;CompositeScore:-0.6,1.9'")
    # 신규 옵션
    ap.add_argument("--round_x", type=int, default=None, help="x축 값 라운딩 자리수")
    ap.add_argument("--round_y", type=int, default=None, help="y축 값 라운딩 자리수")
    ap.add_argument("--x_order", default=None, help="x축 순서 강제, 예: '0.2|0.6|0.8|1.0'")
    ap.add_argument("--y_order", default=None, help="y축 순서 강제, 예: '0|0.5|1.0'")
    ap.add_argument("--xtick_rotate", type=int, default=0, help="x축 tick 라벨 회전각")
    ap.add_argument("--ytick_rotate", type=int, default=0, help="y축 tick 라벨 회전각")
    ap.add_argument("--interp", choices=["nearest", "bilinear"], default="nearest",
                    help="imshow 보간 방식")

    args = ap.parse_args()
    # normalize flags
    args.infer_from_tag = (args.infer_from_tag == "on")
    args.annotate = (args.annotate == "on")
    args.save_pivots = (args.save_pivots == "on")
    args.zlist = [z.strip() for z in str(args.zlist).split(",") if z.strip()]
    if not args.zlist:
        raise SystemExit("[ERR] zlist가 비어 있습니다.")
    return args

def main() -> None:
    args = parse_args()
    build_heatmaps(
        src=args.src, tag_startswith=args.tag_startswith, x=args.x, y=args.y,
        zlist=args.zlist, outdir=args.outdir, infer_from_tag=args.infer_from_tag,
        annotate=args.annotate, agg=args.agg, save_pivots=args.save_pivots,
        dpi=args.dpi, figsize=(args.fig_w, args.fig_h), cmap=args.cmap,
        vmin_max=args.vmin_max, round_x=args.round_x, round_y=args.round_y,
        x_order=args.x_order, y_order=args.y_order,
        xtick_rotate=args.xtick_rotate, ytick_rotate=args.ytick_rotate,
        interp=args.interp,
    )

if __name__ == "__main__":
    main()
