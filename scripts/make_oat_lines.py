# scripts/make_oat_lines.py
import argparse
import pathlib as p

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def coerce_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=r".\outputs\_summary_scored.csv")
    ap.add_argument("--tag_startswith", default="OAT_")
    ap.add_argument("--xcol", default="hedge_sigma_k")
    ap.add_argument("--group_by", default="method", help="라인 구분 컬럼(기본: method)")
    ap.add_argument("--metrics", default="EW,ES95", help="콤마 구분 (예: EW,ES95)")
    ap.add_argument("--outdir", default=r".\outputs\figs")
    args = ap.parse_args()

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]

    df = pd.read_csv(args.src)

    # 태그 필터
    if "tag" in df.columns:
        df = df[df["tag"].astype(str).str.startswith(args.tag_startswith)]

    # xcol 파생(없고 hedge_ratio가 있으면 대체)
    if args.xcol not in df.columns and "hedge_ratio" in df.columns:
        df[args.xcol] = coerce_num(df["hedge_ratio"])

    # 숫자화
    if args.xcol in df.columns:
        df[args.xcol] = coerce_num(df[args.xcol])

    # 그룹 컬럼 보정
    if args.group_by not in df.columns:
        df[args.group_by] = "ALL"

    # 출력 디렉터리
    outdir = p.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 메트릭별 라인 플롯
    for metric in metrics:
        if metric not in df.columns or args.xcol not in df.columns:
            continue

        # group -> (x, mean(metric))
        lines = []
        for g, gdf in df.groupby(args.group_by, dropna=False):
            s = (
                gdf[[args.xcol, metric]]
                .dropna()
                .groupby(args.xcol)[metric]
                .mean(numeric_only=True)
                .reset_index()
                .sort_values(args.xcol)
            )
            if len(s) > 0:
                lines.append((str(g), s))

        if not lines:
            continue

        fig, ax = plt.subplots(figsize=(5.6, 3.8), dpi=180)
        for name, s in lines:
            ax.plot(s[args.xcol], s[metric], marker="o", label=name)

        ax.set_xlabel(args.xcol)
        ax.set_ylabel(metric)
        ax.set_title(f"OAT: {args.xcol} → {metric}   [{args.group_by}]")
        ax.grid(True, alpha=0.25)
        if len(lines) > 1:
            ax.legend(title=args.group_by, fontsize=8)

        out = outdir / f"oat_{args.tag_startswith}_{args.xcol}_{metric}.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] {out}")


if __name__ == "__main__":
    main()
