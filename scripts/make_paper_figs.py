# scripts/make_paper_figs.py
import os, sys, math, argparse, pathlib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless 안전
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────
# A) 공통 유틸: OutRoot 해석 + 데이터 가드 + 메시지 오버레이
# ─────────────────────────────────────────────────────────
def resolve_outroot(arg_path: str) -> tuple[str, bool]:
    """
    반환: (OR_abs, used_latest_fallback)
    - 인자가 디렉터리면 절대경로로 반환
    - 아니면 parent 아래 night_* 중 가장 최근 폴더를 선택(경고 플래그 True)
    """
    base = os.path.abspath(r".\outputs")
    if arg_path:
        p = os.path.abspath(arg_path)
    else:
        p = os.path.abspath(r".\outputs\night_latest")

    # 직접 지정한 경로가 디렉터리면 그대로 사용
    if os.path.isdir(p):
        return p, False

    # parent 탐색 (night_latest or 잘못된 경로일 때)
    parent = os.path.dirname(p) if os.path.dirname(p) else base
    parent = parent if os.path.isdir(parent) else base

    cand = []
    try:
        for name in os.listdir(parent):
            full = os.path.join(parent, name)
            if os.path.isdir(full) and name.startswith("night_"):
                try:
                    cand.append((os.path.getmtime(full), full))
                except OSError:
                    pass
    except FileNotFoundError:
        pass

    if not cand:
        raise FileNotFoundError(f"[resolve] no night_* folder under {parent}")
    cand.sort(reverse=True)
    return os.path.abspath(cand[0][1]), True


def guard_stats(arr_like) -> tuple[bool, bool, bool]:
    """
    반환: (is_empty, is_all_zero, is_single_value)
    """
    if arr_like is None:
        return True, False, False
    a = np.asarray(arr_like, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return True, False, False
    if np.all(a == 0):
        return False, True, True
    if np.nanmax(a) - np.nanmin(a) < 1e-12:
        return False, False, True
    return False, False, False


def annotate_center(ax, text: str, sub: str = "", alpha: float = 0.35):
    """
    축/눈금/데이터는 유지하면서 중앙에 반투명 박스로 안내문을 오버레이.
    - fig를 지우지 않고, ax도 clear하지 않음.
    """
    if not text:
        return
    bbox = dict(boxstyle="round,pad=0.6", facecolor="white", alpha=0.5, edgecolor="none")
    ax.text(0.5, 0.55, text, ha="center", va="center",
            transform=ax.transAxes, fontsize=13, color="0.25", style="italic",
            bbox=bbox, zorder=9999)
    if sub:
        ax.text(0.5, 0.40, sub, ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="0.4", zorder=9999)


def overlay_message(fig, ax, title: str, msg: str, sub: str = "", keep_axes: bool = True):
    """
    데이터가 없거나 단일점일 때:
    - keep_axes=True: 축은 유지 + 중앙 메시지(요청사항 반영)
    - keep_axes=False: 예전 방식(축 숨김) 메시지 화면
    """
    if keep_axes:
        ax.set_title(title)
        annotate_center(ax, msg, sub)
        fig.tight_layout()
    else:
        ax.cla()
        ax.axis("off")
        ax.set_title(title)
        ax.text(0.5, 0.55, msg, ha="center", va="center", fontsize=16, color="0.4", style="italic")
        if sub:
            ax.text(0.5, 0.40, sub, ha="center", va="center", fontsize=10, color="0.5")
        fig.tight_layout()


def save_fig(fig, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

# ─────────────────────────────────────────────────────────
# B) 인자/경로 및 데이터 로드
# ─────────────────────────────────────────────────────────
def load_data(or_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rep = os.path.join(or_path, "night_summary_report.csv")
    cln = os.path.join(or_path, "night_summary_clean.csv")

    if not os.path.exists(rep) or not os.path.exists(cln):
        raise FileNotFoundError(
            f"[load] summary files missing under OR={or_path}\n"
            f" - exists({os.path.basename(rep)}): {os.path.exists(rep)}\n"
            f" - exists({os.path.basename(cln)}): {os.path.exists(cln)}\n"
            f"Run night_run.ps1 with -DoSummary to generate them."
        )

    df_rep = pd.read_csv(rep)
    df_cln = pd.read_csv(cln)

    # 숫자형 보정
    for c in ["EW_avg", "ES95_avg", "Ruin_avg", "WT_avg"]:
        if c in df_rep.columns:
            df_rep[c] = pd.to_numeric(df_rep[c], errors="coerce")
    for c in ["EW", "ES95", "RuinPct", "mean_WT"]:
        if c in df_cln.columns:
            df_cln[c] = pd.to_numeric(df_cln[c], errors="coerce")

    # 결측 컬럼 기본값
    for col, default in [("method", ""), ("baseline", ""), ("w_fixed", "NA")]:
        if col not in df_rep.columns:
            df_rep[col] = default

    return df_rep, df_cln

# ─────────────────────────────────────────────────────────
# C) 라벨/정렬 보정
# ─────────────────────────────────────────────────────────
def normalize_labels(df_rep: pd.DataFrame) -> tuple[pd.DataFrame, list[str], float, float]:
    # baseline: 비어있으면 method 기반 치환
    df_rep = df_rep.copy()
    df_rep["baseline"] = df_rep["baseline"].fillna("")
    mask_empty = df_rep["baseline"].str.strip() == ""
    df_rep.loc[(df_rep["method"].str.lower() == "hjb") & mask_empty, "baseline"] = "HJB"
    df_rep.loc[(df_rep["method"].str.lower() == "rl") & mask_empty,  "baseline"] = "RL"
    df_rep.loc[(df_rep["method"].str.lower() == "rule") & mask_empty, "baseline"] = "(unknown)"

    # w_fixed 카테고리 정렬
    w_order = ["w_0", "w_0_3", "w_0_5", "w_0_7", "w_1", "NA"]
    df_rep["w_fixed"] = df_rep["w_fixed"].astype(str)
    df_rep["w_fixed"] = pd.Categorical(df_rep["w_fixed"], categories=w_order, ordered=True)

    def w_to_float(w: str) -> float:
        if not isinstance(w, str): return np.nan
        if not w.startswith("w_"):  return np.nan
        body = w[2:].replace("_", ".")
        try: return float(body)
        except: return np.nan

    df_rep["w_float"] = df_rep["w_fixed"].astype(str).map(w_to_float)
    df_rep = df_rep.sort_values(["method", "baseline", "w_fixed"])

    # HJB 벤치마크(있으면 1개만 참조)
    hjb_row = df_rep.loc[df_rep["method"].str.lower() == "hjb"].head(1)
    hjb_EW  = float(hjb_row["EW_avg"].iloc[0])   if len(hjb_row) and pd.notna(hjb_row["EW_avg"].iloc[0])   else np.nan
    hjb_ES  = float(hjb_row["ES95_avg"].iloc[0]) if len(hjb_row) and pd.notna(hjb_row["ES95_avg"].iloc[0]) else np.nan

    # baseline 목록
    baselines_present = [b for b in df_rep["baseline"].dropna().unique().tolist() if b != ""]

    return df_rep, baselines_present, hjb_EW, hjb_ES

# ─────────────────────────────────────────────────────────
# D) 플롯들
# ─────────────────────────────────────────────────────────
def plot_ew_vs_w(df_rep: pd.DataFrame, baselines: list[str], hjb_EW: float, OR: str, sub_hint: str, center_msg: str|None):
    fig, ax = plt.subplots(figsize=(8,5))
    lines_y = []

    if baselines:
        for bsl in baselines:
            d = df_rep.query("method=='rule' and baseline==@bsl and w_fixed!='NA'").copy()
            d = d.dropna(subset=["EW_avg","w_float"]).sort_values("w_float")
            if len(d)==0: continue
            ax.plot(d["w_float"], d["EW_avg"], marker="o", label=bsl)
            lines_y.append(d["EW_avg"].values)
    else:
        d = df_rep.query("method=='rule' and w_fixed!='NA'").dropna(subset=["EW_avg","w_float"]).sort_values("w_float")
        if len(d)>0:
            ax.plot(d["w_float"], d["EW_avg"], marker="o", label="rule")
            lines_y.append(d["EW_avg"].values)

    if not math.isnan(hjb_EW):
        ax.axhline(hjb_EW, linestyle="--", label="HJB", alpha=0.8)
        lines_y.append([hjb_EW])

    is_empty, is_all_zero, is_single = guard_stats(np.concatenate(lines_y) if lines_y else [])
    if is_empty or is_single:
        overlay_message(fig, ax, "EW vs w_fixed",
                        "Single-point result (identical samples)" if is_single else "No data to visualize",
                        sub_hint, keep_axes=True)
    else:
        ax.set_xlabel("w_fixed"); ax.set_ylabel("EW (avg)"); ax.set_title("EW vs w_fixed")
        ax.legend(); fig.tight_layout()
        if center_msg:
            annotate_center(ax, center_msg, sub_hint)

    save_fig(fig, os.path.join(OR, "fig_EW_vs_w_fixed.png"))


def plot_es_vs_w(df_rep: pd.DataFrame, baselines: list[str], hjb_ES: float, OR: str, sub_hint: str, center_msg: str|None):
    fig, ax = plt.subplots(figsize=(8,5))
    lines_y = []

    if baselines:
        for bsl in baselines:
            d = df_rep.query("method=='rule' and baseline==@bsl and w_fixed!='NA'").copy()
            d = d.dropna(subset=["ES95_avg","w_float"]).sort_values("w_float")
            if len(d)==0: continue
            ax.plot(d["w_float"], d["ES95_avg"], marker="o", label=bsl)
            lines_y.append(d["ES95_avg"].values)
    else:
        d = df_rep.query("method=='rule' and w_fixed!='NA'").dropna(subset=["ES95_avg","w_float"]).sort_values("w_float")
        if len(d)>0:
            ax.plot(d["w_float"], d["ES95_avg"], marker="o", label="rule")
            lines_y.append(d["ES95_avg"].values)

    if not math.isnan(hjb_ES):
        ax.axhline(hjb_ES, linestyle="--", label="HJB", alpha=0.8)
        lines_y.append([hjb_ES])

    is_empty, is_all_zero, is_single = guard_stats(np.concatenate(lines_y) if lines_y else [])
    if is_empty or is_single:
        overlay_message(fig, ax, "ES95 vs w_fixed",
                        "Single-point result (identical samples)" if is_single else "No data to visualize",
                        sub_hint, keep_axes=True)
    else:
        ax.set_xlabel("w_fixed"); ax.set_ylabel("ES95 (avg, lower is better)"); ax.set_title("ES95 vs w_fixed")
        ax.legend(); fig.tight_layout()
        if center_msg:
            annotate_center(ax, center_msg, sub_hint)

    save_fig(fig, os.path.join(OR, "fig_ES95_vs_w_fixed.png"))


def plot_risk_return(df_rep: pd.DataFrame, baselines: list[str], hjb_EW: float, hjb_ES: float, OR: str, sub_hint: str, center_msg: str|None):
    fig, ax = plt.subplots(figsize=(7,6))
    points = []

    if baselines:
        for bsl in baselines:
            d = df_rep.query("method=='rule' and baseline==@bsl and w_fixed!='NA'").copy()
            d = d.dropna(subset=["ES95_avg","EW_avg"])
            if len(d)==0: continue
            ax.scatter(d["ES95_avg"], d["EW_avg"], label=bsl)
            for _, r in d.iterrows():
                lbl = str(r["w_fixed"]).replace("w_","").replace("_",".")
                ax.annotate(lbl, (r["ES95_avg"], r["EW_avg"]), fontsize=7, alpha=0.7)
            points.append(np.c_[d["ES95_avg"].values, d["EW_avg"].values])
    else:
        d = df_rep.query("method=='rule' and w_fixed!='NA'").dropna(subset=["ES95_avg","EW_avg"])
        if len(d)>0:
            ax.scatter(d["ES95_avg"], d["EW_avg"], label="rule")
            for _, r in d.iterrows():
                lbl = str(r["w_fixed"]).replace("w_","").replace("_",".")
                ax.annotate(lbl, (r["ES95_avg"], r["EW_avg"]), fontsize=7, alpha=0.7)
            points.append(np.c_[d["ES95_avg"].values, d["EW_avg"].values])

    if not math.isnan(hjb_EW) and not math.isnan(hjb_ES):
        ax.scatter([hjb_ES], [hjb_EW], marker="*", s=120, label="HJB")
        points.append(np.array([[hjb_ES, hjb_EW]]))

    P = np.vstack(points) if points else np.empty((0,2))
    if P.shape[0] == 0 or (np.nanmax(P, axis=0) - np.nanmin(P, axis=0) < 1e-12).all():
        overlay_message(fig, ax, "Risk–Return",
                        "Single-point result (identical samples)" if P.shape[0] > 0 else "No data to visualize",
                        sub_hint, keep_axes=True)
    else:
        ax.set_xlabel("ES95 (lower better)"); ax.set_ylabel("EW (higher better)"); ax.set_title("Risk–Return")
        ax.legend(); fig.tight_layout()
        if center_msg:
            annotate_center(ax, center_msg, sub_hint)

    save_fig(fig, os.path.join(OR, "fig_risk_return.png"))


def _compute_frontier(df: pd.DataFrame) -> pd.DataFrame:
    """
    단순 프런티어(ES 오름차순 정렬 후 EW의 상향 누적 최대점).
    입력은 ES95_avg/EW_avg가 있는 df (rule/HJB/RL 혼합 가능).
    """
    d = df.dropna(subset=["ES95_avg", "EW_avg"]).copy()
    if len(d) == 0:
        return d
    d = d.sort_values("ES95_avg")  # 위험(작을수록 좋음) 기준
    ew_best = []
    cur_max = -np.inf
    for x in d["EW_avg"].values:
        cur_max = max(cur_max, x)
        ew_best.append(cur_max)
    d["_ew_cummax"] = ew_best
    # 프런티어 점(해당 위험수준에서 EW가 누적 최대를 갱신한 점들만 남김)
    d_front = d.loc[d["EW_avg"] >= d["_ew_cummax"] - 1e-12].copy()
    d_front = d_front.drop(columns=["_ew_cummax"])
    return d_front


def plot_frontier(df_rep: pd.DataFrame, OR: str, sub_hint: str, center_msg: str|None):
    """
    EW–ES frontier(규범적 효율면). rule/HJB/RL을 색/마커로 구분.
    - HJB/RL은 단일점 또는 소수 점일 수 있음.
    - rule은 w_fixed 라벨을 함께 표시.
    """
    fig, ax = plt.subplots(figsize=(7,6))

    # 방법별 색/마커(고정 색 지정은 피함; 기본 팔레트 사용)
    styles = {
        "rule": dict(marker="o", linestyle="none", alpha=0.85),
        "hjb":  dict(marker="*", linestyle="none", s=120),
        "rl":   dict(marker="^", linestyle="none"),
    }

    # 전체 점
    any_points = False
    for m in ["rule", "hjb", "rl"]:
        d = df_rep.loc[df_rep["method"].str.lower() == m].dropna(subset=["ES95_avg","EW_avg"]).copy()
        if len(d) == 0:
            continue
        any_points = True
        if m == "rule":
            ax.scatter(d["ES95_avg"], d["EW_avg"], label="rule", **{k:v for k,v in styles["rule"].items() if k!="s"})
            # 라벨(가독성 위해 소형)
            for _, r in d.iterrows():
                lbl = str(r.get("w_fixed","")).replace("w_","").replace("_",".")
                if lbl and lbl != "NA":
                    ax.annotate(lbl, (r["ES95_avg"], r["EW_avg"]), fontsize=7, alpha=0.7)
        elif m == "hjb":
            ax.scatter(d["ES95_avg"], d["EW_avg"], label="HJB", **styles["hjb"])
        elif m == "rl":
            ax.scatter(d["ES95_avg"], d["EW_avg"], label="RL", **styles["rl"])

    # 프런티어(모든 방법 합산 후 계산)
    d_all = df_rep.dropna(subset=["ES95_avg","EW_avg"]).copy()
    if len(d_all) > 0:
        fr = _compute_frontier(d_all)
    else:
        fr = pd.DataFrame(columns=["ES95_avg","EW_avg"])

    if len(fr) == 0 or not any_points:
        overlay_message(fig, ax, "EW–ES Frontier",
                        "No data to visualize" if not any_points else "Single-point result (frontier undefined)",
                        sub_hint, keep_axes=True)
    else:
        fr = fr.sort_values("ES95_avg")
        ax.plot(fr["ES95_avg"], fr["EW_avg"], linewidth=2.0, label="Frontier")
        ax.set_xlabel("ES95 (lower better)"); ax.set_ylabel("EW (higher better)")
        ax.set_title("EW–ES Frontier")
        ax.legend()
        fig.tight_layout()
        if center_msg:
            annotate_center(ax, center_msg, sub_hint)

    save_fig(fig, os.path.join(OR, "fig_frontier_EW_ES.png"))

# ─────────────────────────────────────────────────────────
# E) LaTeX 테이블
# ─────────────────────────────────────────────────────────
def save_latex_table(df_rep: pd.DataFrame, OR: str):
    latex_path = os.path.join(OR, "table_summary.tex")
    # 열 순서가 너무 길면 요약형으로 기본 정렬만 유지
    try:
        df_rep.to_latex(latex_path, index=False, float_format="%.6f")
    except Exception:
        with open(latex_path, "w", encoding="utf-8") as f:
            f.write(df_rep.to_string(index=False))
    return latex_path

# ─────────────────────────────────────────────────────────
# F) overlay 문구 자동 생성 (가능하면 요약)
# ─────────────────────────────────────────────────────────
def build_auto_center_msg(df_rep: pd.DataFrame, or_name: str, market_hint: str = "") -> str:
    # 대표값들 추출
    mset = sorted([m for m in df_rep["method"].dropna().astype(str).str.upper().unique() if m])
    bset = sorted([b for b in df_rep["baseline"].dropna().astype(str).unique() if b])
    wvals = sorted([w for w in df_rep["w_fixed"].dropna().astype(str).unique() if w and w != "NA"],
                   key=lambda s: [int(t) if t.isdigit() else t for t in s.replace("w_","").split("_")])
    parts = [f"{or_name}"]
    if mset: parts.append(f"Method={','.join(mset)}")
    if bset: parts.append(f"Baseline={','.join(bset)}")
    if wvals: parts.append(f"w={','.join([v.replace('w_','').replace('_','.') for v in wvals])}")
    if market_hint: parts.append(market_hint)
    return "\n".join(parts)

def market_meta_hint(df_cln: pd.DataFrame) -> str:
    """
    night_summary_clean.csv 안의 메타(가능하면) 요약 문자열 생성.
    우선순위: market_mode/use_real_rf/data_window/bootstrap_block.
    """
    cols = [c for c in df_cln.columns]
    def _first(col):
        if col in cols and df_cln[col].notna().any():
            return str(df_cln[col].dropna().iloc[0])
        return ""
    mode = _first("market_mode")
    use_real = _first("use_real_rf")
    win = _first("data_window")
    blk = _first("bootstrap_block")
    pieces = []
    if mode: pieces.append(f"mode={mode}")
    if blk:  pieces.append(f"block={blk}")
    if use_real: pieces.append(f"use_real_rf={use_real}")
    if win:  pieces.append(f"window={win}")
    return (" | " + ", ".join(pieces)) if pieces else ""

# ─────────────────────────────────────────────────────────
# G) main
# ─────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Generate paper figures from night_summary_* under a given OR.")
    ap.add_argument("outroot", nargs="?", help="outputs/night_YYYYMMDD_HHMM folder. If omitted, uses latest night_* (WARN).")
    ap.add_argument("--overlay", choices=["on","off"], default="on",
                    help="중앙 오버레이 메시지 표시 여부 (기본 on)")
    ap.add_argument("--overlay_text", type=str, default="",
                    help="중앙 오버레이에 강제로 표시할 커스텀 텍스트(줄바꿈 \\n 가능). 비우면 자동 요약.")
    ap.add_argument("--keep_axes_on_empty", choices=["on","off"], default="on",
                    help="데이터 부족/단일점일 때도 축을 유지할지 (요청 반영; 기본 on)")
    ap.add_argument("--frontier", choices=["on","off"], default="on",
                    help="EW–ES 프런티어 그림 생성 여부 (기본 on)")
    ap.add_argument("--save_md", choices=["on","off"], default="on",
                    help="간단 요약 .md 저장 (기본 on)")
    args = ap.parse_args()

    OR, used_latest = resolve_outroot(args.outroot)
    print(f"[OR] {OR}")
    if used_latest:
        print("[WARN] outroot not provided or invalid → using latest night_* folder under parent. "
              "To avoid mixing sessions, prefer passing the exact OR from night_run.ps1.")

    df_rep, df_cln = load_data(OR)
    df_rep, baselines, hjb_EW, hjb_ES = normalize_labels(df_rep)

    or_name = pathlib.Path(OR).name
    mkt_hint = market_meta_hint(df_cln)
    sub_hint = f"OutRoot={or_name}{mkt_hint}"

    # 중앙 오버레이 문구
    if args.overlay == "on":
        base_msg = args.overlay_text.strip()
        center_msg = base_msg if base_msg else build_auto_center_msg(df_rep, or_name, market_hint=mkt_hint.strip(" |"))
    else:
        center_msg = None

    # 플롯들
    plot_ew_vs_w(df_rep, baselines, hjb_EW, OR, sub_hint, center_msg)
    plot_es_vs_w(df_rep, baselines, hjb_ES, OR, sub_hint, center_msg)
    plot_risk_return(df_rep, baselines, hjb_EW, hjb_ES, OR, sub_hint, center_msg)
    if args.frontier == "on":
        plot_frontier(df_rep, OR, sub_hint, center_msg)

    # LaTeX
    latex_path = save_latex_table(df_rep, OR)

    # (선택) 요약 .md
    if args.save_md == "on":
        md_path = os.path.join(OR, "report_quick.md")
        try:
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(f"# Quick Report — {or_name}\n\n")
                f.write(f"- Market: {mkt_hint.strip(' |') or 'N/A'}\n")
                # 대표 결과(있으면) 간단 첨부
                try:
                    hjb = df_rep.loc[df_rep["method"].str.lower()=="hjb"].head(1)
                    rl = df_rep.loc[df_rep["method"].str.lower()=="rl"].head(1)
                    if len(hjb):
                        f.write(f"- HJB: EW={hjb['EW_avg'].iloc[0]:.4f}, ES95={hjb['ES95_avg'].iloc[0]:.4f}\n")
                    if len(rl):
                        f.write(f"- RL : EW={rl['EW_avg'].iloc[0]:.4f}, ES95={rl['ES95_avg'].iloc[0]:.4f}\n")
                except Exception:
                    pass
                f.write("\n## Artifacts\n")
                for fn in ["fig_EW_vs_w_fixed.png","fig_ES95_vs_w_fixed.png","fig_risk_return.png","fig_frontier_EW_ES.png","fig_ruin_bar.png","table_summary.tex"]:
                    p = os.path.join(OR, fn)
                    if os.path.exists(p):
                        f.write(f"- {fn}\n")
        except Exception as e:
            print(f"[report_md] skipped: {e}")

    # 로그 요약
    print("Saved under OR:")
    for fn in ["fig_EW_vs_w_fixed.png","fig_ES95_vs_w_fixed.png","fig_risk_return.png","fig_frontier_EW_ES.png","fig_ruin_bar.png","table_summary.tex","report_quick.md"]:
        p = os.path.join(OR, fn)
        if os.path.exists(p):
            print(" -", p)

if __name__ == "__main__":
    main()
