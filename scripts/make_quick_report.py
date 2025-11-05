# scripts/make_quick_report.py
import os, glob, json, datetime
import pandas as pd

def rel(p): return os.path.normpath(p)

def collect_optimal_points(op_csv):
    if not os.path.exists(op_csv):
        return None, f"[WARN] optimal_points not found: {op_csv}"
    try:
        df = pd.read_csv(op_csv)
        return df, f"[OK] loaded optimal points: {op_csv} (rows={len(df)})"
    except Exception as e:
        return None, f"[ERR] failed to load optimal points: {e}"

def list_figs(patterns):
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    files = sorted(set(files))
    return files

def as_img_md(path):
    name = os.path.basename(path)
    # 이미지 미리보기 + 파일명
    return f"![{name}]({path})\n\n<sub>{name}</sub>\n"

def main():
    out_md = rel(r"outputs/paper/report_quick.md")
    os.makedirs(os.path.dirname(out_md), exist_ok=True)

    parts = []
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    parts.append(f"# Quick Report (DEV)\n\n- Generated: {now}\n- Profile: DEV\n\n")

    # ===== Optimal Points =====
    op_csv = rel(r"outputs/paper/tables/optimal_points.csv")
    df, msg = collect_optimal_points(op_csv)
    parts.append(f"## Optimal Points\n\n")
    parts.append(f"*{msg}*\n\n")
    if df is not None and len(df) > 0:
        try:
            parts.append(df.to_markdown(index=False))
            parts.append("\n\n")
        except Exception as e:
            parts.append(f"[ERR] markdown render failed: {e}\n\n")

    # ===== Figures (heatmaps with overlay) =====
    parts.append("## Figures (Heatmaps)\n\n")
    fig_patterns = [
        rel(r"outputs/figs/heatmap_DEV_2D__*.png"),
        rel(r"outputs/figs/heatmap_*DEV*__.png"),
        rel(r"outputs/figs/heatmap_*2D*.png"),
    ]
    figs = list_figs(fig_patterns)

    if not figs:
        parts.append("_No figures found._\n")
    else:
        for fp in figs:
            parts.append(as_img_md(fp))

    # ===== Save =====
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("".join(parts))
    print(f"[OK] wrote {out_md}")

if __name__ == "__main__":
    main()
