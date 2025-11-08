# scripts/make_paper_figs.py
from __future__ import annotations

"""
make_paper_figs.py — refactored (2025-11-07)

CLI (동일 유지):
  1) figs           : night_* 요약 폴더 기반 그림/요약
  2) oat-table      : 스냅샷 CSV → 단독변수(OAT) 테이블
  3) oat-heatmap    : 스냅샷 CSV → 2D 히트맵(+옵션 오버레이)

변경 요약:
- 공통 유틸/로깅/저장 일원화, 수치 캐스팅/결측 방어 강화, 태그/별칭 컬럼 해석 일관화
- 빈 데이터/단일 포인트/NaN pivot 등 경계 케이스 가독성 높은 메시지 출력
- paper_style 실패해도 예외 삼키지 않고 경고 후 기본 스타일로 진행
"""

import os, sys, math, argparse, pathlib as p, re, json
from typing import Tuple, List, Iterable, Dict, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────
# 전역 플래그 (CLI에서 토글)
# ─────────────────────────────────────────────────────────
KEEP_AXES_ON_EMPTY = True    # figs --keep_axes_on_empty
SAVE_PDF = False             # figs --paper_style on → True


# ─────────────────────────────────────────────────────────
# 로깅/공통 유틸
# ─────────────────────────────────────────────────────────
def _log(msg: str) -> None:
    print(f"[INFO] {msg}")

def _warn(msg: str) -> None:
    print(f"[WARN] {msg}")

def _err(msg: str) -> None:
    print(f"[ERR]  {msg}", file=sys.stderr)

def _num(x) -> float | np.nan:
    try:
        return float(x)
    except Exception:
        return np.nan

def _to_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def _safe_percent(v: float) -> str:
    if not np.isfinite(v): return "NA"
    return f"{v*100:.1f}%"


# 파일 저장(경로 보장 + png & 선택적 pdf)
def _save_figure(fig: plt.Figure, path_png: str, save_pdf: bool = False) -> None:
    try:
        os.makedirs(os.path.dirname(path_png), exist_ok=True)
        fig.savefig(path_png, dpi=300, bbox_inches="tight", facecolor="white")
        if save_pdf:
            base, _ = os.path.splitext(path_png)
            fig.savefig(base + ".pdf", bbox_inches="tight", facecolor="white")
    except Exception as e:
        _warn(f"figure save failed: {e}")


# 비어있는/단일값/상수 데이터 가드
def guard_stats(arr_like) -> tuple[bool, bool, bool]:
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
    if not text: return
    bbox = dict(boxstyle="round,pad=0.6", facecolor="white", alpha=0.5, edgecolor="none")
    ax.text(0.5, 0.55, text, ha="center", va="center",
            transform=ax.transAxes, fontsize=13, color="0.25", style="italic",
            bbox=bbox, zorder=9999)
    if sub:
        ax.text(0.5, 0.40, sub, ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="0.4", zorder=9999)


def overlay_message(fig, ax, title: str, msg: str, sub: str = "", keep_axes: bool = True):
    ax.set_title(title)
    if keep_axes:
        annotate_center(ax, msg, sub)
    else:
        ax.cla(); ax.axis("off")
        ax.text(0.5, 0.55, msg, ha="center", va="center", fontsize=16, color="0.4", style="italic")
        if sub:
            ax.text(0.5, 0.40, sub, ha="center", va="center", fontsize=10, color="0.5")
    fig.tight_layout()


def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def _ensure_columns(df: pd.DataFrame, defaults: dict) -> pd.DataFrame:
    d = df.copy()
    for k, v in defaults.items():
        if k not in d.columns:
            d[k] = v
        d[k] = d[k].fillna(v)
    return d


# ─────────────────────────────────────────────────────────
# outroot(night_*) 탐색
# ─────────────────────────────────────────────────────────
def resolve_outroot(arg_path: str | None) -> tuple[str, bool]:
    base = os.path.abspath(r".\outputs")
    if arg_path:
        pth = os.path.abspath(arg_path)
    else:
        pth = os.path.abspath(r".\outputs\night_latest")

    if os.path.isdir(pth):
        return pth, False

    parent = os.path.dirname(pth) if os.path.dirname(pth) else base
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


# ─────────────────────────────────────────────────────────
# night_* 요약 로딩/정규화
# ─────────────────────────────────────────────────────────
def _normalize_ruin_columns(df_rep: pd.DataFrame, df_cln: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rep = df_rep.copy()
    cln = df_cln.copy()

    if "Ruin_avg" in rep.columns:
        rep["Ruin_avg"] = pd.to_numeric(rep["Ruin_avg"], errors="coerce")
        over_one = rep["Ruin_avg"] > 1.0
        if over_one.any():
            rep.loc[over_one, "Ruin_avg"] = rep.loc[over_one, "Ruin_avg"] / 100.0

    if "RuinPct" in cln.columns:
        cln["RuinPct"] = pd.to_numeric(cln["RuinPct"], errors="coerce")
        cln["Ruin"] = cln["RuinPct"] / 100.0
        under_one = cln["RuinPct"] <= 1.0
        cln.loc[under_one, "Ruin"] = cln.loc[under_one, "RuinPct"]
    elif "Ruin" in cln.columns:
        cln["Ruin"] = pd.to_numeric(cln["Ruin"], errors="coerce")
        over_one = cln["Ruin"] > 1.0
        cln.loc[over_one, "Ruin"] = cln.loc[over_one, "Ruin"] / 100.0

    return rep, cln


def load_night_summary(or_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rep = os.path.join(or_path, "night_summary_report.csv")
    cln = os.path.join(or_path, "night_summary_clean.csv")

    if os.path.exists(rep) and os.path.exists(cln):
        df_rep = pd.read_csv(rep)
        df_cln = pd.read_csv(cln)
        _coerce_numeric(df_rep, ["EW_avg","ES95_avg","Ruin_avg","WT_avg"])
        _coerce_numeric(df_cln, ["EW","ES95","RuinPct","Ruin","mean_WT"])
        df_rep = _ensure_columns(df_rep, {"method":"","baseline":"","w_fixed":"NA"})
        df_cln = _ensure_columns(df_cln, {"method":"","baseline":"","w_fixed":"NA"})
        df_rep, df_cln = _normalize_ruin_columns(df_rep, df_cln)
        return df_rep, df_cln

    if (not os.path.exists(rep)) and os.path.exists(cln):
        df_cln = pd.read_csv(cln)
        _coerce_numeric(df_cln, ["EW","ES95","RuinPct","Ruin","mean_WT"])
        if "baseline" not in df_cln.columns: df_cln["baseline"] = ""
        if "w_fixed"  not in df_cln.columns: df_cln["w_fixed"]  = "NA"
        if "Ruin" not in df_cln.columns:
            df_cln["Ruin"] = pd.to_numeric(df_cln.get("RuinPct"), errors="coerce") / 100.0
        else:
            df_cln["Ruin"] = pd.to_numeric(df_cln["Ruin"], errors="coerce")
            over_one = df_cln["Ruin"] > 1.0
            df_cln.loc[over_one, "Ruin"] = df_cln.loc[over_one, "Ruin"] / 100.0

        grp = df_cln.groupby(["method","baseline","w_fixed"], dropna=False)
        df_rep = grp.agg(EW_avg=("EW","mean"),
                         ES95_avg=("ES95","mean"),
                         Ruin_avg=("Ruin","mean"),
                         WT_avg=("mean_WT","mean")).reset_index()
        df_rep = _ensure_columns(df_rep, {"method":"","baseline":"","w_fixed":"NA"})
        return df_rep, df_cln

    raise FileNotFoundError(
        f"[load] summary files missing under OR={or_path}\n"
        f" - exists(night_summary_report.csv): {os.path.exists(rep)}\n"
        f" - exists(night_summary_clean.csv): {os.path.exists(cln)}\n"
        f"Run night_run.ps1 with -DoSummary to generate them."
    )


def normalize_labels(df_rep: pd.DataFrame) -> tuple[pd.DataFrame, list[str], float, float]:
    df_rep = df_rep.copy()
    df_rep["baseline"] = df_rep["baseline"].fillna("")
    m_lower = df_rep["method"].astype(str).str.lower()
    mask_empty = df_rep["baseline"].str.strip() == ""

    # 비어있으면 메서드명으로 baseline 채움(시각화 가독성)
    df_rep.loc[(m_lower == "hjb") & mask_empty, "baseline"] = "HJB"
    df_rep.loc[(m_lower == "rl") & mask_empty,  "baseline"] = "RL"
    df_rep.loc[(m_lower == "rule") & mask_empty, "baseline"] = "(unknown)"

    # w_fixed 정렬/숫자 변환
    w_order = ["w_0", "w_0_3", "w_0_5", "w_0_7", "w_1", "NA"]
    df_rep["w_fixed"] = pd.Categorical(df_rep["w_fixed"].astype(str), categories=w_order, ordered=True)

    def w_to_float(w: str) -> float:
        if not isinstance(w, str): return np.nan
        if not w.startswith("w_"):  return np.nan
        body = w[2:].replace("_", ".")
        try: return float(body)
        except: return np.nan

    df_rep["w_float"] = df_rep["w_fixed"].astype(str).map(w_to_float)
    df_rep = df_rep.sort_values(["method", "baseline", "w_fixed"])

    hjb_row = df_rep.loc[m_lower == "hjb"].head(1)
    hjb_EW  = float(hjb_row["EW_avg"].iloc[0])   if len(hjb_row) and pd.notna(hjb_row["EW_avg"].iloc[0])   else np.nan
    hjb_ES  = float(hjb_row["ES95_avg"].iloc[0]) if len(hjb_row) and pd.notna(hjb_row["ES95_avg"].iloc[0]) else np.nan
    baselines_present = [b for b in df_rep["baseline"].dropna().unique().tolist() if b != ""]

    return df_rep, baselines_present, hjb_EW, hjb_ES


# ─────────────────────────────────────────────────────────
# 그림: EW/ES vs w_fixed, Risk–Return/Frontier, Ruin bar
# ─────────────────────────────────────────────────────────
def plot_ew_vs_w(df_rep: pd.DataFrame, baselines: list[str], hjb_EW: float, OR: str, sub_hint: str, center_msg: str|None):
    fig, ax = plt.subplots(figsize=(8,5)); lines_y = []
    if baselines:
        for bsl in baselines:
            d = df_rep.query("method=='rule' and baseline==@bsl and w_fixed!='NA'").copy()
            d = d.dropna(subset=["EW_avg","w_float"]).sort_values("w_float")
            if len(d)==0: continue
            ax.plot(d["w_float"], d["EW_avg"], marker="o", label=bsl); lines_y.append(d["EW_avg"].values)
    else:
        d = df_rep.query("method=='rule' and w_fixed!='NA'").dropna(subset=["EW_avg","w_float"]).sort_values("w_float")
        if len(d)>0:
            ax.plot(d["w_float"], d["EW_avg"], marker="o", label="rule"); lines_y.append(d["EW_avg"].values)

    if np.isfinite(hjb_EW):
        ax.axhline(hjb_EW, linestyle="--", label="HJB", alpha=0.8); lines_y.append([hjb_EW])

    is_empty, _, is_single = guard_stats(np.concatenate(lines_y) if lines_y else [])
    if is_empty or is_single:
        overlay_message(fig, ax, "EW vs w_fixed",
                        "Single-point result (identical samples)" if is_single else "No data to visualize",
                        sub_hint, keep_axes=KEEP_AXES_ON_EMPTY)
    else:
        ax.set_xlabel("w_fixed"); ax.set_ylabel("EW (avg)"); ax.set_title("EW vs w_fixed")
        ax.legend(); fig.tight_layout()
        if center_msg: annotate_center(ax, center_msg, sub_hint)

    _save_figure(fig, os.path.join(OR, "fig_EW_vs_w_fixed.png"), SAVE_PDF)
    plt.close(fig)


def plot_es_vs_w(df_rep: pd.DataFrame, baselines: list[str], hjb_ES: float, OR: str, sub_hint: str, center_msg: str|None):
    fig, ax = plt.subplots(figsize=(8,5)); lines_y = []
    if baselines:
        for bsl in baselines:
            d = df_rep.query("method=='rule' and baseline==@bsl and w_fixed!='NA'").copy()
            d = d.dropna(subset=["ES95_avg","w_float"]).sort_values("w_float")
            if len(d)==0: continue
            ax.plot(d["w_float"], d["ES95_avg"], marker="o", label=bsl); lines_y.append(d["ES95_avg"].values)
    else:
        d = df_rep.query("method=='rule' and w_fixed!='NA'").dropna(subset=["ES95_avg","w_float"]).sort_values("w_float")
        if len(d)>0:
            ax.plot(d["w_float"], d["ES95_avg"], marker="o", label="rule"); lines_y.append(d["ES95_avg"].values)

    if np.isfinite(hjb_ES):
        ax.axhline(hjb_ES, linestyle="--", label="HJB", alpha=0.8); lines_y.append([hjb_ES])

    is_empty, _, is_single = guard_stats(np.concatenate(lines_y) if lines_y else [])
    if is_empty or is_single:
        overlay_message(fig, ax, "ES95 vs w_fixed",
                        "Single-point result (identical samples)" if is_single else "No data to visualize",
                        sub_hint, keep_axes=KEEP_AXES_ON_EMPTY)
    else:
        ax.set_xlabel("w_fixed"); ax.set_ylabel("ES95 (avg, lower is better)"); ax.set_title("ES95 vs w_fixed")
        ax.legend(); fig.tight_layout()
        if center_msg: annotate_center(ax, center_msg, sub_hint)

    _save_figure(fig, os.path.join(OR, "fig_ES95_vs_w_fixed.png"), SAVE_PDF)
    plt.close(fig)


def _compute_frontier(df: pd.DataFrame) -> pd.DataFrame:
    d = df.dropna(subset=["ES95_avg", "EW_avg"]).copy()
    if len(d) == 0: return d
    d = d.sort_values("ES95_avg")
    ew_best, cur_max = [], -np.inf
    for x in d["EW_avg"].values:
        cur_max = max(cur_max, x); ew_best.append(cur_max)
    d["_ew_cummax"] = ew_best
    d_front = d.loc[d["EW_avg"] >= d["_ew_cummax"] - 1e-12].copy()
    d_front = d_front.drop(columns=["_ew_cummax"])
    return d_front


def plot_risk_return(df_rep: pd.DataFrame, baselines: list[str], hjb_EW: float, hjb_ES: float, OR: str, sub_hint: str, center_msg: str|None):
    fig, ax = plt.subplots(figsize=(7,6))
    styles = { "rule": dict(marker="o", alpha=0.85, s=35),
               "hjb":  dict(marker="*", s=120),
               "rl":   dict(marker="^", s=45) }
    any_points = False

    for m in ["rule","hjb","rl"]:
        d = df_rep.loc[df_rep["method"].astype(str).str.lower()==m].dropna(subset=["ES95_avg","EW_avg"]).copy()
        if len(d)==0: continue
        any_points = True
        if m=="rule":
            ax.scatter(d["ES95_avg"], d["EW_avg"], label="rule", **styles["rule"])
            for _, r in d.iterrows():
                lbl = str(r.get("w_fixed","")).replace("w_","").replace("_",".")
                if lbl and lbl!="NA": ax.annotate(lbl, (r["ES95_avg"], r["EW_avg"]), fontsize=7, alpha=0.7)
        elif m=="hjb":
            ax.scatter(d["ES95_avg"], d["EW_avg"], label="HJB", **styles["hjb"])
        elif m=="rl":
            ax.scatter(d["ES95_avg"], d["EW_avg"], label="RL",  **styles["rl"])

    d_all = df_rep.dropna(subset=["ES95_avg","EW_avg"]).copy()
    fr = _compute_frontier(d_all) if len(d_all)>0 else pd.DataFrame(columns=["ES95_avg","EW_avg"])

    if len(fr)==0 or not any_points:
        overlay_message(fig, ax, "Risk–Return",
                        "No data to visualize" if not any_points else "Single-point result (frontier undefined)",
                        sub_hint, keep_axes=KEEP_AXES_ON_EMPTY)
    else:
        fr = fr.sort_values("ES95_avg")
        ax.plot(fr["ES95_avg"], fr["EW_avg"], linewidth=2.0, label="Frontier")
        ax.set_xlabel("ES95 (lower better)"); ax.set_ylabel("EW (higher better)"); ax.set_title("Risk–Return")
        ax.legend(); fig.tight_layout()
        if center_msg: annotate_center(ax, center_msg, sub_hint)

    _save_figure(fig, os.path.join(OR, "fig_risk_return.png"), SAVE_PDF)
    plt.close(fig)


def plot_frontier(df_rep: pd.DataFrame, OR: str, sub_hint: str, center_msg: str|None):
    fig, ax = plt.subplots(figsize=(7,6))
    styles = {"rule": dict(marker="o", alpha=0.85, s=35),
              "hjb":  dict(marker="*", s=120),
              "rl":   dict(marker="^", s=45)}
    any_points = False

    for m in ["rule","hjb","rl"]:
        d = df_rep.loc[df_rep["method"].astype(str).str.lower()==m].dropna(subset=["ES95_avg","EW_avg"]).copy()
        if len(d)==0: continue
        any_points = True
        ax.scatter(d["ES95_avg"], d["EW_avg"], label=m.upper(), **styles.get(m, {}))

    d_all = df_rep.dropna(subset=["ES95_avg","EW_avg"]).copy()
    fr = _compute_frontier(d_all) if len(d_all) > 0 else pd.DataFrame(columns=["ES95_avg","EW_avg"])

    if len(fr) == 0 or not any_points:
        overlay_message(fig, ax, "EW–ES Frontier",
                        "No data to visualize" if not any_points else "Single-point result (frontier undefined)",
                        sub_hint, keep_axes=KEEP_AXES_ON_EMPTY)
    else:
        fr = fr.sort_values("ES95_avg")
        ax.plot(fr["ES95_avg"], fr["EW_avg"], linewidth=2.0, label="Frontier")
        ax.set_xlabel("ES95 (lower better)"); ax.set_ylabel("EW (higher better)")
        ax.set_title("EW–ES Frontier"); ax.legend(); fig.tight_layout()
        if center_msg: annotate_center(ax, center_msg, sub_hint)

    _save_figure(fig, os.path.join(OR, "fig_frontier_EW_ES.png"), SAVE_PDF)
    plt.close(fig)


def plot_ruin_bar(df_rep: pd.DataFrame, OR: str, sub_hint: str, center_msg: str | None):
    from matplotlib.ticker import PercentFormatter, AutoMinorLocator
    d_rule = df_rep.loc[df_rep["method"].astype(str).str.lower() == "rule"].copy()
    d_rule = d_rule.loc[d_rule["w_fixed"].astype(str) != "NA"].dropna(subset=["Ruin_avg"])
    if d_rule.empty:
        fig, ax = plt.subplots(figsize=(9,5))
        overlay_message(fig, ax, "Ruin (by baseline & w_fixed)", "No data to visualize", sub_hint, keep_axes=KEEP_AXES_ON_EMPTY)
        _save_figure(fig, os.path.join(OR, "fig_ruin_bar.png"), SAVE_PDF); plt.close(fig); return

    w_order = ["w_0","w_0_3","w_0_5","w_0_7","w_1"]
    d_rule["w_fixed"] = pd.Categorical(d_rule["w_fixed"].astype(str), categories=w_order, ordered=True)

    pv = d_rule.pivot_table(index="w_fixed", columns="baseline", values="Ruin_avg", aggfunc="mean", observed=False)
    pv = pv.reindex(w_order).dropna(how="all")

    if pv.empty:
        fig, ax = plt.subplots(figsize=(9,5))
        overlay_message(fig, ax, "Ruin (by baseline & w_fixed)", "No data to visualize", sub_hint, keep_axes=KEEP_AXES_ON_EMPTY)
        _save_figure(fig, os.path.join(OR, "fig_ruin_bar.png"), SAVE_PDF); plt.close(fig); return

    fig, ax = plt.subplots(figsize=(9.5,5.3))
    x = np.arange(len(pv.index)); cols = list(pv.columns); nb = max(len(cols),1)
    width = min(0.8/nb, 0.22)

    vals_all = pv.to_numpy(dtype=float); finite_vals = vals_all[np.isfinite(vals_all)]
    treat_as_prob = finite_vals.size > 0 and np.nanmax(finite_vals) <= 1.0+1e-12

    hatches = ["/","\\\\","xx","--","++","..","oo","**"]
    edge_kwargs = dict(edgecolor="0.25", linewidth=0.6)
    bar_groups = []

    for i, c in enumerate(cols):
        y = pv[c].to_numpy(dtype=float); offset = (i - (nb-1)/2.0)*width
        bars = ax.bar(x+offset, y, width=width, label=str(c), hatch=hatches[i%len(hatches)], **edge_kwargs)
        bar_groups.append((x+offset, y, bars))

    # HJB 참조선
    hjb_val = None
    try:
        hjb = df_rep.loc[df_rep["method"].astype(str).str.lower()=="hjb"].head(1)
        if len(hjb) and pd.notna(hjb["Ruin_avg"].iloc[0]):
            hjb_val = float(hjb["Ruin_avg"].iloc[0])
            if hjb_val > 1.0: hjb_val /= 100.0
    except Exception:
        pass

    # 축 범위
    if treat_as_prob:
        ymax = np.nanmax(finite_vals) if finite_vals.size else 0.0
        if hjb_val is not None: ymax = np.nanmax([ymax, hjb_val])
        y0,y1 = 0.0, min(1.0, max(1e-6, ymax)*1.15)
    else:
        ymax = np.nanmax(finite_vals) if finite_vals.size else 1.0
        if hjb_val is not None: ymax = np.nanmax([ymax, hjb_val])
        y0,y1 = 0.0, max(1e-6, ymax)*1.15
    ax.set_ylim(y0,y1)

    # 값 라벨
    span = y1-y0; lift = 0.035*span; y_cap = y0 + 0.98*span
    for (xs, ys, bars) in bar_groups:
        for bx, vy in zip(xs, ys):
            if not np.isfinite(vy): continue
            ytext = min(vy+lift, y_cap)
            label = f"{vy*100:.1f}%" if treat_as_prob else f"{vy:.2f}"
            ax.text(bx, ytext, label, ha="center", va="bottom", fontsize=8, color="0.25")

    if hjb_val is not None and np.isfinite(hjb_val):
        ax.axhline(hjb_val, linestyle="--", linewidth=1.2, alpha=0.9, label="HJB", color="0.35")
        lbl = f"HJB: {hjb_val*100:.1f}%" if treat_as_prob else f"HJB: {hjb_val:.2f}"
        ax.text(0.995, min(0.92, max(0.06, (hjb_val - y0)/max(span,1e-9))), lbl,
                transform=ax.transAxes, ha="right", va="center", fontsize=9, color="0.35")

    ax.set_xticks(x, [str(ix).replace("w_","").replace("_",".") for ix in pv.index])
    ax.set_xlabel("w_fixed"); ax.set_title("Ruin (by baseline & w_fixed)")
    if treat_as_prob:
        from matplotlib.ticker import PercentFormatter
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax.set_ylabel("Ruin probability")
    else:
        ax.set_ylabel("Ruin (avg)")

    ax.yaxis.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.35)
    from matplotlib.ticker import AutoMinorLocator
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.20)
    ax.xaxis.grid(False)
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)
    ax.spines["left"].set_linewidth(0.8); ax.spines["bottom"].set_linewidth(0.8)

    ncol = min(len(cols),4)
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5,1.12), ncol=ncol, frameon=False)
    for t in leg.get_texts(): t.set_fontsize(9)

    fig.tight_layout()
    if center_msg: annotate_center(ax, center_msg, sub_hint)
    _save_figure(fig, os.path.join(OR, "fig_ruin_bar.png"), SAVE_PDF)
    plt.close(fig)


def save_latex_table(df_rep: pd.DataFrame, OR: str):
    latex_path = os.path.join(OR, "table_summary.tex")
    try:
        df_rep.to_latex(latex_path, index=False, float_format="%.6f")
    except Exception:
        with open(latex_path, "w", encoding="utf-8") as f:
            f.write(df_rep.to_string(index=False))
    return latex_path


def build_auto_center_msg(df_rep: pd.DataFrame, or_name: str, market_hint: str = "") -> str:
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
# 스냅샷 CSV 기반: 태그 해석/별칭/파싱
# ─────────────────────────────────────────────────────────
_TAG2D_PAT = re.compile(r".*?_2D(?:_|-)us(?P<u>-?\d+(?:\.\d+)?)_(?:h|k)(?P<h>-?\d+(?:\.\d+)?)")
_TAG_LA_PAT = re.compile(r"(?:^|[_-])la(?P<v>-?\d+(?:\.\d+)?)")

VAR_FROM_TAG_PATTERNS = {
    "hedge_sigma_k": re.compile(r"(?:^|[_-])h(?P<v>-?\d+(?:\.\d+)?)"),
    "mix_us":        re.compile(r"(?:^|[_-])us(?P<v>-?\d+(?:\.\d+)?)"),
    "loss_aversion": re.compile(r"(?:^|[_-])la(?P<v>-?\d+(?:\.\d+)?)"),
    "bias_loss_aversion": re.compile(r"(?:^|[_-])la(?P<v>-?\d+(?:\.\d+)?)"),
}

def _infer_from_tag(df: pd.DataFrame) -> pd.DataFrame:
    if "tag" not in df.columns: return df

    def _parse_2d(tag: str):
        out = {}
        m = _TAG2D_PAT.match(str(tag))
        if m:
            try:
                u = float(m.group("u")); h = float(m.group("h"))
                out.update({"mix_us": round(u,2), "mix_kr": round(1.0-u,2), "mix_gold": 0.0, "hedge_sigma_k": h})
            except Exception:
                pass
        m2 = _TAG_LA_PAT.search(str(tag))
        if m2:
            try:
                out["loss_aversion"] = float(m2.group("v"))
                out["bias_loss_aversion"] = out["loss_aversion"]
            except Exception:
                pass
        return out

    extra_df = pd.DataFrame(list(df["tag"].map(_parse_2d)))
    for c in ("mix_us","mix_kr","mix_gold","hedge_sigma_k","loss_aversion","bias_loss_aversion"):
        if c not in df.columns and c in extra_df.columns:
            df[c] = extra_df[c]
    return df


def ensure_var_column(df: pd.DataFrame, var_name: str, round_n: int | None = 3) -> pd.DataFrame:
    """
    var_name 보장:
      - 동일 컬럼 있으면 캐스팅 후 반환
      - 별칭 매핑(예: loss_aversion <- bias_loss_aversion, la_k, la)
      - 그래도 없으면 tag에서 정규식으로 파싱
    """
    d = df.copy()
    if var_name in d.columns:
        d[var_name] = pd.to_numeric(d[var_name], errors="coerce")
        if round_n is not None: d[var_name] = d[var_name].round(round_n)
        return d

    alias_map = {
        "loss_aversion": ["bias_loss_aversion", "la_k", "la"],
        "bias_loss_aversion": ["loss_aversion", "la_k", "la"],
    }
    if var_name in alias_map:
        for a in alias_map[var_name]:
            if a in d.columns:
                d[var_name] = pd.to_numeric(d[a], errors="coerce")
                if round_n is not None: d[var_name] = d[var_name].round(round_n)
                return d

    if "tag" in d.columns:
        pat = VAR_FROM_TAG_PATTERNS.get(var_name)
        if pat:
            def _parse_tag(s: str) -> float:
                m = pat.search(str(s))
                return float(m.group("v")) if m else np.nan
            d[var_name] = d["tag"].map(_parse_tag)
            if round_n is not None: d[var_name] = d[var_name].round(round_n)
            return d

    raise SystemExit(f"[ERR] ensure_var_column: cannot resolve var '{var_name}'. df.columns={list(d.columns)[:20]} ...")


def load_snapshot(src: str, tag_startswith: str, method: str, es_mode: str) -> pd.DataFrame:
    df = pd.read_csv(src)
    if "tag" not in df.columns:
        raise SystemExit("[ERR] snapshot에 'tag' 컬럼이 없습니다.")
    if tag_startswith:
        df = df[df["tag"].astype(str).str.startswith(tag_startswith)]
    if method:
        df = df[df.get("method", "").astype(str).str.lower() == method.lower()]
    if es_mode:
        df = df[df.get("es_mode", "").astype(str).str.lower() == es_mode.lower()]
    if df.empty:
        raise SystemExit(f"[ERR] 필터 후 행이 없습니다. tag_startswith={tag_startswith}, method={method}, es_mode={es_mode}")
    return df


# ─────────────────────────────────────────────────────────
# OAT(단독변수) 테이블 생성
# ─────────────────────────────────────────────────────────
def make_oat_table(src: str, out: str, var: str, metrics: str, tag_startswith: str, method: str, es_mode: str, agg: str, round_var:int=3):
    df = load_snapshot(src, tag_startswith, method, es_mode)
    df = ensure_var_column(df, var, round_n=round_var)

    ms_all = [m.strip() for m in str(metrics).split(",") if m.strip()]
    ms = [m for m in ms_all if m in df.columns]
    if not ms:
        raise SystemExit(f"[ERR] 지정한 metrics가 데이터에 없습니다: {ms_all}")

    for c in [var] + ms:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    sub = df[[var] + ms].dropna(subset=[var])
    if sub.empty:
        raise SystemExit(f"[ERR] '{var}' 유효 데이터가 없습니다.")

    if agg not in {"median","mean","std","iqr"}:
        raise SystemExit(f"[ERR] agg must be one of median|mean|std|iqr, got {agg}")

    def _iqr(s): 
        return np.nanpercentile(s, 75) - np.nanpercentile(s, 25)

    aggfunc = {"median":"median","mean":"mean","std":"std","iqr":_iqr}[agg]
    tbl = sub.groupby(var, as_index=False)[ms].agg(aggfunc)
    tbl = tbl.sort_values(var).reset_index(drop=True)

    # 기준점(첫 행) 대비 Δ% (median/mean일 때만)
    if agg in {"median","mean"}:
        base = tbl.iloc[0].copy()
        for m in ms:
            denom = base[m]
            if pd.isna(denom) or float(denom) == 0.0:
                tbl[f"{m}_DeltaPct"] = np.nan
            else:
                tbl[f"{m}_DeltaPct"] = (tbl[m] - base[m]) / denom * 100.0

    outp = p.Path(out); outp.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(outp) as xw:
        tbl.to_excel(xw, sheet_name=f"{var}_{agg}", index=False)
    _log(f"wrote {outp}")


# ─────────────────────────────────────────────────────────
# 2D 히트맵
# ─────────────────────────────────────────────────────────
def _parse_vmin_max(s: Optional[str]) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    out: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    if not s: return out
    for part in str(s).split(";"):
        part = part.strip()
        if not part or ":" not in part: continue
        k, rng = part.split(":", 1)
        k = k.strip()
        vmin, vmax = None, None
        try:
            xs = [t.strip() for t in rng.split(",")]
            if len(xs)>=1 and xs[0]!="": vmin=float(xs[0])
            if len(xs)>=2 and xs[1]!="": vmax=float(xs[1])
        except Exception:
            pass
        out[k]=(vmin,vmax)
    return out


def _pivot(df: pd.DataFrame, x: str, y: str, val: str, agg: str) -> pd.DataFrame:
    if agg not in {"median","mean","std","iqr"}:
        raise SystemExit(f"[ERR] agg must be one of median|mean|std|iqr, got {agg}")

    def _iqr(s): 
        return np.nanpercentile(s, 75) - np.nanpercentile(s, 25)

    aggfunc = {"median":"median","mean":"mean","std":"std","iqr":_iqr}[agg]
    piv = df.pivot_table(index=y, columns=x, values=val, aggfunc=aggfunc)

    # 정렬 시도(숫자로 캐스팅 가능하면 숫자 기준)
    try:
        piv = piv.sort_index(axis=0, key=lambda s: pd.to_numeric(s, errors="ignore"))
    except Exception:
        try: piv = piv.sort_index(axis=0)
        except Exception: pass

    try:
        piv = piv.sort_index(axis=1, key=lambda s: pd.to_numeric(s, errors="ignore"))
    except Exception:
        try: piv = piv.sort_index(axis=1)
        except Exception: pass

    return piv


def _fmt_tick(v) -> str:
    return f"{v:g}" if isinstance(v, (int, float, np.floating)) else str(v)


def plot_heat(piv: pd.DataFrame, x: str, y: str, val: str, title: str,
              annotate: bool, figsize: Tuple[float,float], dpi: int, cmap_name: str,
              vmin: Optional[float]=None, vmax: Optional[float]=None) -> tuple[plt.Figure, plt.Axes]:
    data = np.array(piv.values, dtype=float)
    mask = np.isnan(data)
    mdata = np.ma.masked_array(data, mask=mask)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    cmap = getattr(plt.cm, cmap_name, plt.cm.viridis).copy()
    cmap.set_bad("lightgray")
    im = ax.imshow(mdata, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)

    xt = list(range(len(piv.columns))); yt = list(range(len(piv.index)))
    ax.set_xticks(xt); ax.set_xticklabels([_fmt_tick(v) for v in piv.columns])
    ax.set_yticks(yt); ax.set_yticklabels([_fmt_tick(v) for v in piv.index])
    ax.set_xlabel(x); ax.set_ylabel(y); ax.set_title(title)

    if annotate:
        for i in yt:
            for j in xt:
                if not mask[i,j]:
                    ax.text(j, i, f"{mdata[i,j]:.2f}", ha="center", va="center", fontsize=8)

    cbar = plt.colorbar(im, ax=ax); cbar.ax.set_ylabel(val, rotation=90, va="center")
    fig.tight_layout()
    return fig, ax


def _snap_index(val, ticks: List, numeric_ok: bool = True) -> int:
    try:
        if numeric_ok:
            arr = np.array(ticks, dtype=float)
            j = int(np.nanargmin(np.abs(arr - float(val))))
            return j
    except Exception:
        pass

    sval = str(val)
    if sval in ticks:
        return ticks.index(sval)

    def _to_num(s):
        try:
            return float(str(s).replace("w_","").replace("_","."))
        except:
            return np.nan

    base = _to_num(sval)
    diffs = [abs((_to_num(t) - base)) if np.isfinite(_to_num(t)) and np.isfinite(base) else np.inf for t in ticks]
    return int(np.nanargmin(diffs))


def _overlay_points_on_ax(ax: plt.Axes, piv: pd.DataFrame, points: List[dict], x: str, y: str, label: str):
    xs = list(piv.columns); ys = list(piv.index)
    Xp, Yp, texts = [], [], []
    for pt in points:
        xv = pt.get("x"); yv = pt.get("y")
        if xv is None or yv is None: 
            continue
        j = _snap_index(xv, xs)
        i = _snap_index(yv, ys)
        Xp.append(j); Yp.append(i); texts.append(pt.get("text",""))

    if not Xp: return

    ax.scatter(Xp, Yp, marker="*", s=220, facecolor="none", edgecolor="black", linewidths=1.6, label=label)
    ax.scatter(Xp, Yp, marker="*", s=120, color="white", linewidths=0.0)

    for xj, yi, txt in zip(Xp, Yp, texts):
        if txt:
            ax.text(xj+0.12, yi+0.12, txt, ha="left", va="bottom", fontsize=9, color="0.2",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", alpha=0.75))
    ax.legend(loc="upper left", frameon=False, fontsize=8)


def _load_overlay_points(path: str) -> tuple[List[dict], Dict[str,str]]:
    pts: List[dict] = []
    meta: Dict[str,str] = {}
    if not path: return pts, meta
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            pts = obj
        elif isinstance(obj, dict):
            pts = obj.get("points", [])
            meta = obj.get("meta", {})
        if not isinstance(pts, list):
            pts = []
    except Exception as e:
        _warn(f"overlay_points load failed: {e}")
    return pts, meta


def _apply_method_es_filter(df: pd.DataFrame, method: str, es_mode: str) -> pd.DataFrame:
    d = df.copy()
    if method:
        d = d[d.get("method","").astype(str).str.lower()==method.lower()]
    if es_mode:
        d = d[d.get("es_mode","").astype(str).str.lower()==es_mode.lower()]
    return d


def make_oat_heatmaps(src: str, outdir: str, tag_startswith: str, x: str, y: str,
                      zlist: str, agg: str, annotate: bool, round_x: Optional[int],
                      round_y: Optional[int], vmin_max: str, title_prefix: str, title_suffix: str,
                      dpi:int, fig_w: float, fig_h: float, cmap: str, infer_from_tag: bool=True,
                      overlay_points: str = "", overlay_label: str = "best",
                      method: str = "", es_mode: str = ""):
    df = pd.read_csv(src)
    if "tag" not in df.columns:
        raise SystemExit("[ERR] 데이터에 'tag' 컬럼이 없습니다.")

    df = df[df["tag"].astype(str).str.startswith(tag_startswith)]
    if df.empty:
        raise SystemExit(f"[ERR] '{tag_startswith}' 로 시작하는 tag 데이터가 없습니다.")

    if infer_from_tag:
        df = _infer_from_tag(df)

    df = _apply_method_es_filter(df, method, es_mode)
    if df.empty:
        raise SystemExit(f"[ERR] 필터 후 행이 없습니다. tag_startswith={tag_startswith}, method={method}, es_mode={es_mode}")

    # 수치 캐스팅
    cast_cols = [x, y, "EW","ES95","CompositeScore","Ruin",
                 "hedge_ratio","hedge_sigma_k","mix_us","mix_kr","mix_gold",
                 "loss_aversion","bias_loss_aversion"]
    for c in cast_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # x/y가 없으면 태그/별칭에서 생성
    if x not in df.columns:
        df = ensure_var_column(df, x, round_n=round_x if round_x is not None else 3)
    if y not in df.columns:
        df = ensure_var_column(df, y, round_n=round_y if round_y is not None else 3)

    if x not in df.columns or y not in df.columns:
        raise SystemExit(f"[ERR] 데이터에 '{x}' 혹은 '{y}' 컬럼이 없음.")

    if round_x is not None: df[x] = df[x].round(round_x)
    if round_y is not None: df[y] = df[y].round(round_y)

    zcols = [t.strip() for t in str(zlist).split(",") if t.strip()]
    vmm = _parse_vmin_max(vmin_max)

    outdir_path = p.Path(outdir); outdir_path.mkdir(parents=True, exist_ok=True)
    overlay_pts, overlay_meta = _load_overlay_points(overlay_points)

    _log(f"rows={len(df)} | x={x} y={y} | zlist={zcols} | agg={agg} | method={method or '-'} | es_mode={es_mode or '-'}")

    for z in zcols:
        if z not in df.columns:
            _warn(f"'{z}' 없음 → skip"); continue

        sub = df[[x,y,z]].dropna()
        if sub.empty:
            _warn(f"'{z}' 데이터 없음 → skip"); continue

        piv = _pivot(sub, x, y, z, agg)
        csv_path = outdir_path / f"heatmap_{tag_startswith}_{x}_vs_{y}_{z}_pivot.csv"
        try:
            piv.to_csv(csv_path)
            _log(f"{csv_path}")
        except Exception as e:
            _warn(f"pivot csv write failed: {e}")

        outpng = outdir_path / f"heatmap_{tag_startswith}_{x}_vs_{y}_{z}.png"
        vmin,vmax = vmm.get(z,(None,None))
        title = f"{title_prefix}Heatmap: {z} ({x} × {y}){(' ' + title_suffix) if title_suffix else ''}"

        if piv.size == 0 or (np.asarray(piv.values).size == 0):
            # 빈 pivot: 축만 남기거나 통째로 메시지
            fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
            overlay_message(fig, ax, f"{title}", "No data to visualize", "", keep_axes=KEEP_AXES_ON_EMPTY)
            _save_figure(fig, str(outpng), SAVE_PDF)
            plt.close(fig)
            continue

        fig, ax = plot_heat(piv, x, y, z, title, annotate, (fig_w,fig_h), dpi, cmap, vmin, vmax)

        # 오버레이 z-필터
        do_overlay = bool(overlay_pts)
        meta_z = str(overlay_meta.get("z","")).strip()
        if meta_z:
            do_overlay = do_overlay and (meta_z == str(z))
        if do_overlay:
            _overlay_points_on_ax(ax, piv, overlay_pts, x, y, overlay_label)

        _save_figure(fig, str(outpng), SAVE_PDF)
        plt.close(fig)


# ─────────────────────────────────────────────────────────
# CLI (subcommands: figs / oat-table / oat-heatmap)
# ─────────────────────────────────────────────────────────
def main():
    global KEEP_AXES_ON_EMPTY, SAVE_PDF
    ap = argparse.ArgumentParser(description="Paper figures & tables")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # 1) night_* 폴더 기반 그림/표
    ap_figs = sub.add_parser("figs", help="Generate figures from night_* summary folder")
    ap_figs.add_argument("outroot", nargs="?", help="outputs/night_YYYYMMDD_HHMM (omit=latest)")
    ap_figs.add_argument("--overlay", choices=["on","off"], default="on")
    ap_figs.add_argument("--overlay_text", type=str, default="")
    ap_figs.add_argument("--keep_axes_on_empty", choices=["on","off"], default="on")
    ap_figs.add_argument("--frontier", choices=["on","off"], default="on")
    ap_figs.add_argument("--save_md", choices=["on","off"], default="on")
    ap_figs.add_argument("--paper_style", choices=["on","off"], default="off")

    # 2) OAT 표 (스냅샷/요약 CSV)
    ap_tbl = sub.add_parser("oat-table", help="Build OAT summary table from snapshot CSV")
    ap_tbl.add_argument("--src", required=True, help="CSV (e.g., .\\outputs\\DEV_metrics_snapshot.csv)")
    ap_tbl.add_argument("--out", default=r".\outputs\paper\tables\OAT_summary.xlsx")
    ap_tbl.add_argument("--var", required=True, help="e.g., hedge_sigma_k, mix_us, loss_aversion")
    ap_tbl.add_argument("--metrics", default="EW,ES95,Ruin,CompositeScore")
    ap_tbl.add_argument("--tag_startswith", default="", help="e.g., DEV_OAT_, OVN_OAT_")
    ap_tbl.add_argument("--method", default="", help="hjb/rl filter")
    ap_tbl.add_argument("--es_mode", default="", help="wealth/cons filter")
    ap_tbl.add_argument("--agg", choices=["median","mean","std","iqr"], default="median")

    # 3) 2D 히트맵 (스냅샷 CSV)
    ap_hm = sub.add_parser("oat-heatmap", help="Build 2D heatmaps from snapshot CSV")
    ap_hm.add_argument("--src", required=True)
    ap_hm.add_argument("--outdir", default=r".\outputs\figs")
    ap_hm.add_argument("--tag_startswith", required=True, help="e.g., DEV_2D_, OVN_2D_")
    ap_hm.add_argument("--x", default="mix_us"); ap_hm.add_argument("--y", default="hedge_sigma_k")
    ap_hm.add_argument("--zlist", default="EW,ES95,CompositeScore")
    ap_hm.add_argument("--agg", choices=["median","mean","std","iqr"], default="median")
    ap_hm.add_argument("--annotate", choices=["on","off"], default="on")
    ap_hm.add_argument("--round_x", type=int, default=2); ap_hm.add_argument("--round_y", type=int, default=2)
    ap_hm.add_argument("--vmin_max", default="")
    ap_hm.add_argument("--title_prefix", default=""); ap_hm.add_argument("--title_suffix", default="")
    ap_hm.add_argument("--dpi", type=int, default=180); ap_hm.add_argument("--fig_w", type=float, default=6.6); ap_hm.add_argument("--fig_h", type=float, default=4.8)
    ap_hm.add_argument("--cmap", default="viridis")
    ap_hm.add_argument("--overlay_points", default="", help="JSON of points to overlay (from find_optima.py)")
    ap_hm.add_argument("--overlay_label", default="best", help="legend label for overlay points")
    ap_hm.add_argument("--method", default="", help="hjb/rl filter")
    ap_hm.add_argument("--es_mode", default="", help="wealth/cons filter")

    args = ap.parse_args()

    if args.cmd == "figs":
        KEEP_AXES_ON_EMPTY = (args.keep_axes_on_empty == "on")
        OR, used_latest = resolve_outroot(args.outroot)
        _log(f"OR={OR} (latest={used_latest})")

        center_msg_forced_none = False
        if args.paper_style == "on":
            try:
                import importlib.util
                style_path = os.path.join(os.path.dirname(__file__), "fig_style_paper.py")
                spec = importlib.util.spec_from_file_location("fig_style_paper", style_path)
                mod = importlib.util.module_from_spec(spec); assert spec and spec.loader
                spec.loader.exec_module(mod)  # type: ignore
                mod.apply_paper_style()
                SAVE_PDF = True
                center_msg_forced_none = True
                _log("paper_style applied; PDF export enabled; overlay disabled.")
            except Exception as e:
                SAVE_PDF = False
                _warn(f"paper_style failed: {e}")

        df_rep, df_cln = load_night_summary(OR)
        df_rep, baselines, hjb_EW, hjb_ES = normalize_labels(df_rep)
        or_name = p.Path(OR).name
        mkt_hint = market_meta_hint(df_cln)
        sub_hint = f"OutRoot={or_name}{mkt_hint}"
        center_msg = None if center_msg_forced_none or args.overlay=="off" else \
            (args.overlay_text.strip() or build_auto_center_msg(df_rep, or_name, market_hint=mkt_hint.strip(' |')))

        plot_ew_vs_w(df_rep, baselines, hjb_EW, OR, sub_hint, center_msg)
        plot_es_vs_w(df_rep, baselines, hjb_ES, OR, sub_hint, center_msg)
        plot_risk_return(df_rep, baselines, hjb_EW, hjb_ES, OR, sub_hint, center_msg)
        if args.frontier == "on":
            plot_frontier(df_rep, OR, sub_hint, center_msg)
        plot_ruin_bar(df_rep, OR, sub_hint, center_msg)

        latex_path = save_latex_table(df_rep, OR)

        if args.save_md == "on":
            md_path = os.path.join(OR, "report_quick.md")
            try:
                with open(md_path, "w", encoding="utf-8") as f:
                    f.write(f"# Quick Report — {or_name}\n\n")
                    f.write(f"- Market: {mkt_hint.strip(' |') or 'N/A'}\n")
                    try:
                        hjb = df_rep.loc[df_rep["method"].astype(str).str.lower()=="hjb"].head(1)
                        rl  = df_rep.loc[df_rep["method"].astype(str).str.lower()=="rl"].head(1)
                        if len(hjb) and pd.notna(hjb['EW_avg'].iloc[0]) and pd.notna(hjb['ES95_avg'].iloc[0]):
                            f.write(f"- HJB: EW={hjb['EW_avg'].iloc[0]:.4f}, ES95={hjb['ES95_avg'].iloc[0]:.4f}\n")
                        if len(rl) and pd.notna(rl['EW_avg'].iloc[0]) and pd.notna(rl['ES95_avg'].iloc[0]):
                            f.write(f"- RL : EW={rl['EW_avg'].iloc[0]:.4f}, ES95={rl['ES95_avg'].iloc[0]:.4f}\n")
                    except Exception:
                        pass
                    f.write("\n## Artifacts\n")
                    for fn in ["fig_EW_vs_w_fixed.png","fig_ES95_vs_w_fixed.png","fig_risk_return.png","fig_frontier_EW_ES.png","fig_ruin_bar.png","table_summary.tex"]:
                        pp = os.path.join(OR, fn)
                        if os.path.exists(pp): f.write(f"- {fn}\n")
            except Exception as e:
                _warn(f"report_md skipped: {e}")

        print("Saved under OR:", OR)
        return

    if args.cmd == "oat-table":
        make_oat_table(
            src=args.src, out=args.out, var=args.var, metrics=args.metrics,
            tag_startswith=args.tag_startswith, method=args.method, es_mode=args.es_mode, agg=args.agg
        )
        return

    if args.cmd == "oat-heatmap":
        make_oat_heatmaps(
            src=args.src, outdir=args.outdir, tag_startswith=args.tag_startswith,
            x=args.x, y=args.y, zlist=args.zlist, agg=args.agg,
            annotate=(args.annotate=="on"), round_x=args.round_x, round_y=args.round_y,
            vmin_max=args.vmin_max, title_prefix=args.title_prefix, title_suffix=args.title_suffix,
            dpi=args.dpi, fig_w=args.fig_w, fig_h=args.fig_h, cmap=args.cmap, infer_from_tag=True,
            overlay_points=args.overlay_points, overlay_label=args.overlay_label,
            method=args.method, es_mode=args.es_mode
        )
        return


if __name__ == "__main__":
    main()
