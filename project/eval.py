# project/eval.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import csv
import datetime
import inspect
import sys
from typing import Callable, Tuple, Optional, Dict, Any, Union, List

import numpy as _np
import pandas as pd

# ✅ env 패키지 내부 파일 직접 임포트 (init 노출 이슈 방지)
from .env.retirement_env import RetirementEnv


# =========================
# Constants (CSV header)
# =========================

_METRICS_HEADER: list[str] = [
    "ts", "asset", "method", "es_mode",
    "alpha", "lambda_term", "F_target",
    "EW", "ES95", "EL", "Ruin", "mean_WT",
    "HedgeHit", "HedgeKMean", "HedgeActiveW",
    "fee_annual", "w_max", "floor_on", "f_min_real",
    "hedge_on", "hedge_mode",
    "horizon_years", "steps_per_year",
    "seeds", "n_paths_eval", "tag",
    # consumption (bands & ES-like on consumption)
    "p10_c_last", "p50_c_last", "p90_c_last", "C_ES95_avg", "AlivePathRate",
    # annuity overlay (if present)
    "y_ann", "a_factor", "P",
    # sweep/overlay params persisted in logs
    "ann_alpha",
    # diagnostics for ES
    "es95_source", "es95_note",
    # expected utility reporting
    "EU", "EU_per_year", "delta_annual",
    # (NEW) utility/reporting config & loss baseline audit
    "crra_gamma", "u_scale", "report_utility", "F_target_used",
]


# =========================
# Helpers
# =========================

def _reset_env(env: RetirementEnv, seed: int) -> None:
    """reset(seed=...) 지원/미지원 환경 모두에서 안전하게 초기화."""
    try:
        sig = inspect.signature(env.reset)
        if "seed" in sig.parameters:
            env.reset(seed=seed)
            return
    except (TypeError, ValueError):
        pass
    try:
        env.reset()
    finally:
        if hasattr(env, "set_seed"):
            try:
                env.set_seed(seed)
            except Exception:
                pass


def _as_nd_state(raw: Any, env_like: Any) -> _np.ndarray:
    """
    정책에 전달할 상태를 항상 ndarray([t_norm, W])로 정규화.
    - dict: {"t": step_idx, "W": wealth} → [t_norm, W]
    - ndarray: float ravel, 길이<2면 W 보강
    - 기타: [0.0, env.W] 폴백
    """
    if isinstance(raw, dict):
        T = int(getattr(env_like, "T", 1) or 1)
        t_idx = float(raw.get("t", 0.0))
        t_norm = t_idx / float(max(1, T - 1))
        W_now = float(raw.get("W", getattr(env_like, "W", 0.0)))
        return _np.asarray([t_norm, W_now], dtype=float)
    if isinstance(raw, _np.ndarray):
        arr = _np.asarray(raw, dtype=float).ravel()
        if arr.size >= 2:
            return arr[:2]
        W_now = float(getattr(env_like, "W", 0.0))
        return _np.asarray([float(arr[0]) if arr.size else 0.0, W_now], dtype=float)
    return _np.asarray([0.0, float(getattr(env_like, "W", 0.0))], dtype=float)


def _clean_finite(a: _np.ndarray) -> tuple[_np.ndarray, int, int]:
    """
    NaN/Inf 제거 유틸.
    Returns: (finite_only, n_total, n_dropped)
    """
    arr = _np.asarray(a, dtype=float)
    mask = _np.isfinite(arr)
    return arr[mask], int(arr.size), int((~mask).sum())


def _crra_u(c: float, gamma: float) -> float:
    """CRRA 효용 (c>0). gamma≈1이면 로그 효용으로 처리."""
    c = max(float(c), 1e-12)
    if abs(float(gamma) - 1.0) < 1e-12:
        return float(_np.log(c))
    return float((c ** (1.0 - float(gamma)) - 1.0) / (1.0 - float(gamma)))


def _stderr(msg: str) -> None:
    try:
        print(msg, file=sys.stderr, flush=True)
    except Exception:
        pass


# =========================
# Core episode
# =========================

def run_episode(
    env: RetirementEnv,
    actor: Callable[[Any], Tuple[float, float]],
    seed: int = 0,
) -> Tuple[_np.ndarray, _np.ndarray, bool, dict[str, float]]:
    """
    한 에피소드 실행.
    Returns:
        W_hist: 월별 W
        C_hist: 월별 소비
        early_hit: 조기 파산 여부(사망과 구분)
        ep_stats: 헤지 계측 요약
    """
    _reset_env(env, seed=seed)

    W_hist: List[float] = []
    C_hist: List[float] = []
    early_hit = False

    # hedge counters
    hedge_hits = 0
    hedge_k_sum = 0.0
    hedge_active_w_sum = 0.0

    for _ in range(env.T):
        raw_state = env._obs() if hasattr(env, "_obs") else {"t": 0.0, "W": getattr(env, "W", 0.0)}
        state = _as_nd_state(raw_state, env)   # 상태 어댑터
        q, w = actor(state)
        _, _, done, _, info = env.step(q=q, w=w)

        W_hist.append(float(env.W))
        C_hist.append(float((info or {}).get("consumption", 0.0)))

        # ruin (separate from death) — env가 W<=0에서 done 처리하므로 조기 파산 트래킹만 분리
        if env.W <= 0.0 and not done:
            early_hit = True

        # hedge stats
        if isinstance(info, dict) and info.get("hedge_active", False):
            hedge_hits += 1
            hedge_k_sum += float(info.get("hedge_k", 0.0))
            hedge_active_w_sum += float(info.get("w", w))

        if done:
            break

    N = len(W_hist)
    ep_stats = {
        "hedge_hits": float(hedge_hits),
        "steps": float(N),
        "hedge_k_sum": float(hedge_k_sum),
        "hedge_active_w_sum": float(hedge_active_w_sum),
    }
    return _np.asarray(W_hist, dtype=float), _np.asarray(C_hist, dtype=float), bool(early_hit), ep_stats


# =========================
# Metrics
# =========================

def metrics_wealth(WT_samples: _np.ndarray, alpha: float = 0.95) -> Dict[str, float]:
    """ES95 = 하위 (1-α) 분위수 이하의 평균(자산 관점)."""
    WT = _np.asarray(WT_samples, dtype=float)
    if WT.size == 0:
        return dict(EW=0.0, ES95=0.0)
    EW = float(WT.mean())
    q = _np.quantile(WT, 1.0 - alpha)  # 5th pct of wealth
    tail = WT[WT <= q]
    ES_tail_mean = float(tail.mean()) if tail.size > 0 else float(q)
    return dict(EW=EW, ES95=ES_tail_mean)


def metrics_loss(WT_samples: _np.ndarray, F: float = 1.0, alpha: float = 0.95) -> Dict[str, float]:
    """Loss = max(F − W_T, 0). ES95는 손실분포의 α-조건부기대치(CVaR_α)."""
    WT = _np.asarray(WT_samples, dtype=float)
    if WT.size == 0:
        return dict(EW=0.0, EL=0.0, ES95=0.0)
    L = _np.maximum(F - WT, 0.0)
    EL = float(L.mean())
    qL = _np.quantile(L, alpha)      # VaR_α
    tail = L[L >= qL]
    ES = float(tail.mean()) if tail.size > 0 else float(qL)
    return dict(EW=float(WT.mean()), EL=EL, ES95=ES)


# =========================
# Soft assertions (consumption / wealth)
# =========================

def _soft_assert_streams(WT_arr: _np.ndarray, C_all: List[_np.ndarray], quiet: bool) -> List[str]:
    """
    WT_arr: 최종자산 배열이 아니라, 여기서는 경로별 최종값들만 있어도 최소한의 검사는 가능.
    C_all: 각 경로의 소비 시계열(고정 길이, NaN 패딩)
    반환: 경고 메시지 리스트(비어있으면 이상 없음)
    """
    warns: List[str] = []

    # Wealth warnings (terminal values)
    if WT_arr.size == 0 or not _np.isfinite(WT_arr).all():
        bad = int(_np.sum(~_np.isfinite(WT_arr))) if WT_arr.size > 0 else 0
        warns.append(f"wealth_nonfinite={bad}/{WT_arr.size}")
    if _np.any(WT_arr < -1e-9):
        nneg = int(_np.sum(WT_arr < -1e-9))
        wmin = float(_np.nanmin(WT_arr))
        warns.append(f"wealth_negative_paths={nneg} (min={wmin:.6g})")

    # Consumption warnings (sequence-level)
    if C_all:
        C_mat = _np.vstack(C_all)
        if not _np.isfinite(C_mat).all():
            bad = int(_np.sum(~_np.isfinite(C_mat)))
            warns.append(f"consumption_nonfinite_count={bad}")
        if _np.nanmin(C_mat) < -1e-12:
            nneg = int(_np.sum(C_mat < -1e-12))
            cmin = float(_np.nanmin(C_mat))
            warns.append(f"consumption_negative_count={nneg} (min={cmin:.6g})")

    if warns and not quiet:
        _stderr("[warn] soft-assert: " + "; ".join(warns))
    return warns


# =========================
# Evaluation (wealth/loss + consumption bands + EU)
# =========================

def evaluate(
    cfg: Any,
    actor,
    es_mode: str = "wealth",
    rng=None,                  # (호환성 유지용, 미사용)
    return_paths: bool = False,
) -> Union[Dict[str, float], Tuple[Dict[str, float], Dict[str, Any]]]:
    """
    Parameters
    ----------
    cfg : Any
    actor : policy callable
    es_mode : "wealth" | "loss"
    return_paths : bool
        True면 (metrics, {"eval_WT": [...]}) 튜플 반환. False면 metrics만 반환.

    Returns
    -------
    metrics : dict
        EW, ES95, EL(손실모드), Ruin, mean_WT, 소비 밴드 등 요약 메트릭
        + es_mode, es95_source(진단) + (옵션) EU, EU_per_year, delta_annual
    extras? : dict (optional)
        eval_WT : list[float]  # 경로별 최종자산 (CLI의 CVaR 재계산에 사용)
        ruin_flags : list[bool]
        T : int
    """
    env = RetirementEnv(cfg)
    T = int(getattr(env, "T", 0))

    # (debug) verify injected paths snapshot (quiet=off 때만)
    if str(getattr(cfg, "quiet", "on")).lower() != "on":
        def _cs(a):
            a = _np.asarray(a, dtype=float)
            return float(_np.nanmean(a[:16])) if a.size else float("nan")
        try:
            print(f"[eval] path_cs ret={_cs(getattr(env, 'path_risky', _np.array([], dtype=float))):.6f} "
                  f"rf={_cs(getattr(env, 'path_safe', _np.array([], dtype=float))):.6f}")
            pr = getattr(env, "path_risky", None); ps = getattr(env, "path_safe", None)
            if pr is not None and ps is not None:
                pr = _np.asarray(pr, dtype=float); ps = _np.asarray(ps, dtype=float)
                if pr.size >= 2 and ps.size >= 2:
                    print(f"[eval] head ret={pr[0]:.6f},{pr[1]:.6f} rf={ps[0]:.6f},{ps[1]:.6f}")
        except Exception:
            pass

    WT: List[float] = []
    early_flags: List[bool] = []
    C_all: List[_np.ndarray] = []  # consumption series (NaN padded to T)

    # EU accumulators (path level, optional)
    path_EU: List[float] = []

    # hedge aggregates
    agg_hedge_hits = 0.0
    agg_steps = 0.0
    agg_k_sum = 0.0
    agg_active_w_sum = 0.0

    seeds = getattr(cfg, "seeds", [0]) or [0]
    # CLI와 호환: n_paths(일반) / rl_n_paths_eval(RL)
    n_eval = int(getattr(cfg, "n_paths", getattr(cfg, "n_paths_eval", getattr(cfg, "rl_n_paths_eval", 1)) or 1))

    report_utility = str(getattr(cfg, "report_utility", "off")).lower() == "on"
    gamma = float(getattr(cfg, "crra_gamma", 3.0) or 3.0)
    u_scale = float(getattr(cfg, "u_scale", 1.0) or 1.0)
    delta_ann = getattr(cfg, "delta_annual", None)
    spm = int(getattr(env, "steps_per_year", 12) or 12)

    # delta_annual이 주어졌을 때만 월 할인율로 변환(정합성 보장)
    if delta_ann is None:
        delta_m = 1.0
    else:
        try:
            delta_ann_f = float(delta_ann)
            # 0<delta≤1 범위로 클리핑(수치적 안전)
            delta_ann_f = max(1e-12, min(1.0, delta_ann_f))
            delta_m = float(delta_ann_f ** (1.0 / float(spm)))
        except Exception:
            delta_m = 1.0

    for sd in seeds:
        base = int(sd) * 100_000
        for k in range(n_eval):
            W_hist, C_hist, early, ep_stats = run_episode(env, actor, seed=base + k)

            WT.append(float(W_hist[-1]) if W_hist.size > 0 else 0.0)
            early_flags.append(bool(early))

            if T <= 0:
                T = int(C_hist.size)
            row = _np.full(T, _np.nan, dtype=float)
            take = min(C_hist.size, T)
            if take > 0:
                row[:take] = C_hist[:take]
            C_all.append(row)

            # EU on the consumption stream (optional, robust)
            if report_utility:
                try:
                    eu = 0.0
                    for t_idx, c_t in enumerate(_np.asarray(C_hist, dtype=float)):
                        if _np.isfinite(c_t):
                            eu += (delta_m ** t_idx) * (u_scale * _crra_u(c_t, gamma))
                    path_EU.append(float(eu))
                except Exception:
                    # EU 계산 실패는 메트릭의 필수 항목이 아니므로 무음 처리
                    pass

            agg_hedge_hits += float(ep_stats.get("hedge_hits", 0.0))
            agg_steps += float(ep_stats.get("steps", 0.0))
            agg_k_sum += float(ep_stats.get("hedge_k_sum", 0.0))
            agg_active_w_sum += float(ep_stats.get("hedge_active_w_sum", 0.0))

    WT_arr = _np.asarray(WT, dtype=float)
    early_arr = _np.asarray(early_flags, dtype=bool)

    es_note_msgs: List[str] = []

    # wealth/loss & ruin  (★ NaN/Inf 제거)
    WT_fin, n_total, n_drop = _clean_finite(WT_arr)
    if n_drop > 0:
        es_note_msgs.append(f"dropped_nonfinite_WT={n_drop}/{n_total}")

    # Ruin은 유한 샘플 기준으로 계산
    if n_total == 0 or WT_fin.size == 0:
        m: Dict[str, float] = dict(EW=0.0, ES95=0.0, EL=0.0, mean_WT=0.0)
        ruin_rate = 0.0
        es_note_msgs.append("no_finite_WT")
    else:
        # early 플래그는 동일 길이이므로 유한 마스크로 필터
        finite_mask = _np.isfinite(WT_arr)
        early_fin = early_arr[finite_mask] if early_arr.size == finite_mask.size else early_arr
        ruin_rate = float(_np.mean(_np.logical_or(early_fin, WT_fin <= 0.0))) if WT_fin.size > 0 else 0.0

        if str(es_mode).lower() == "loss":
            # loss 기준선 선택(명시 없으면 1.0로 fallback) + 감사 로깅
            F_raw = getattr(cfg, "F_target", None)
            if F_raw is None:
                F = 1.0
                es_note_msgs.append("F_target_fallback_to_1.0")
            else:
                F = float(F_raw)
                if F == 0.0:
                    es_note_msgs.append("F_target_explicit_zero")
            alpha = float(getattr(cfg, "alpha", 0.95))
            m = metrics_loss(WT_fin, F=F, alpha=alpha)
            m["mean_WT"] = float(WT_fin.mean()) if WT_fin.size > 0 else 0.0
            m["es95_source"] = "computed_in_evaluate_loss"
            m["F_target_used"] = float(F)
        else:
            alpha = float(getattr(cfg, "alpha", 0.95))
            m = metrics_wealth(WT_fin, alpha=alpha)
            m["mean_WT"] = m["EW"]
            m["es95_source"] = "computed_in_evaluate_wealth"
            m["F_target_used"] = None

    m["Ruin"] = ruin_rate
    m["es_mode"] = str(es_mode).lower()

    # hedge summary
    m["HedgeHit"] = float(agg_hedge_hits / agg_steps) if agg_steps > 0 else 0.0
    m["HedgeKMean"] = float(agg_k_sum / max(agg_hedge_hits, 1.0)) if agg_hedge_hits > 0 else 0.0
    m["HedgeActiveW"] = float(agg_active_w_sum / max(agg_hedge_hits, 1.0)) if agg_hedge_hits > 0 else 0.0

    # -----------------------------
    # Consumption: bands + ES-like
    # -----------------------------
    if len(C_all) > 0 and T > 0:
        C_mat = _np.vstack(C_all)  # (Npaths, T)

        # AlivePathRate: 소비가 한 번이라도 관측된 경로 비율
        alive_rate = float(_np.mean(~_np.all(_np.isnan(C_mat), axis=1)))
        m["AlivePathRate"] = alive_rate

        # 유효 열(전부 NaN이 아닌 시점)만 골라서 밴드 계산
        valid_cols = _np.where(~_np.all(_np.isnan(C_mat), axis=0))[0]
        if valid_cols.size > 0:
            last_idx = int(valid_cols[-1])

            p10_v = _np.nanpercentile(C_mat[:, valid_cols], 10, axis=0)
            p50_v = _np.nanpercentile(C_mat[:, valid_cols], 50, axis=0)
            p90_v = _np.nanpercentile(C_mat[:, valid_cols], 90, axis=0)

            m["p10_c_last"] = float(_np.nanpercentile(C_mat[:, last_idx], 10))
            m["p50_c_last"] = float(_np.nanpercentile(C_mat[:, last_idx], 50))
            m["p90_c_last"] = float(_np.nanpercentile(C_mat[:, last_idx], 90))

            # ES-like on consumption: 경로별 평균소비의 하위 5% 분위수
            Cmean_paths = _np.nanmean(C_mat, axis=1)
            m["C_ES95_avg"] = float(_np.nanquantile(Cmean_paths, 0.05))

            # bands 저장은 --bands 토글(on일 때만 IO)
            if str(getattr(cfg, "bands", "on")).lower() == "on":
                bands_dir = os.path.join(getattr(cfg, "outputs", "./outputs"), "_bands")
                os.makedirs(bands_dir, exist_ok=True)
                bands = _np.full((3, T), _np.nan, dtype=float)
                bands[:, valid_cols] = _np.vstack([p10_v, p50_v, p90_v])
                pd.DataFrame({
                    "t": _np.arange(T, dtype=int),
                    "p10": bands[0], "p50": bands[1], "p90": bands[2]
                }).to_csv(os.path.join(bands_dir, "consumption_bands.csv"),
                         index=False, encoding="utf-8")
        else:
            m.update({"p10_c_last": 0.0, "p50_c_last": 0.0, "p90_c_last": 0.0, "C_ES95_avg": 0.0})

    # EU summary (+ config echo)
    try:
        if path_EU:
            m["EU"] = float(_np.nanmean(_np.asarray(path_EU, dtype=float)))
            # 연환산: 총 기간(T months) → years = T/spm
            spm_local = int(getattr(env, "steps_per_year", 12) or 12)
            yrs = float(max(1.0, (T or int(getattr(env, "T", spm_local))) / spm_local))
            m["EU_per_year"] = float(m["EU"] / yrs)
            m["delta_annual"] = getattr(cfg, "delta_annual", None)
        else:
            m["EU"] = None
            m["EU_per_year"] = None
            m["delta_annual"] = getattr(cfg, "delta_annual", None)
        # EU 리포팅 설정/파라미터도 메트릭에 반영
        m["report_utility"] = bool(report_utility)
        m["crra_gamma"] = float(gamma)
        m["u_scale"] = float(u_scale)
    except Exception:
        m["EU"] = None
        m["EU_per_year"] = None
        m["delta_annual"] = getattr(cfg, "delta_annual", None)
        m["report_utility"] = bool(report_utility)
        m["crra_gamma"] = float(gamma)
        m["u_scale"] = float(u_scale)

    # 소프트 어설션 실행
    try:
        warns = _soft_assert_streams(WT_arr, C_all, quiet=(str(getattr(cfg, "quiet", "on")).lower() == "on"))
        if warns:
            es_note_msgs.append("soft_assert: " + "; ".join(warns))
    except Exception:
        pass

    # (quiet=off) 분포 진단 출력
    if str(getattr(cfg, "quiet", "on")).lower() != "on":
        try:
            wt_std = float(_np.nanstd(WT_arr)) if WT_arr.size > 0 else float("nan")
            wt_min = float(_np.nanmin(WT_arr)) if WT_arr.size > 0 else float("nan")
            wt_max = float(_np.nanmax(WT_arr)) if WT_arr.size > 0 else float("nan")
            print(f"[dbg:evaluate] WT_std={wt_std:.6g} WT_min={wt_min:.6g} WT_max={wt_max:.6g} Ruin={m.get('Ruin'):.3f}")
        except Exception:
            pass

    # ES 노트 정리
    if es_note_msgs:
        m["es95_note"] = "; ".join(es_note_msgs)

    if return_paths:
        extras: Dict[str, Any] = {
            "eval_WT": WT_arr.tolist(),  # 원본(필터 전) 유지: 재현/진단 목적
            "ruin_flags": early_flags,
            "T": int(T),
        }
        return m, extras
    return m


# =========================
# Autosave (+ header auto-migration)
# =========================

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def _maybe_upgrade_header(csv_path: str, expected: list[str]) -> None:
    """
    기존 metrics.csv 헤더가 구버전이면 자동 백업 후 최신 헤더로 재작성.
    (기존 행은 유지, 새 컬럼은 공란으로 채움)
    """
    try:
        if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
            return
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            first = f.readline().rstrip("\r\n")
        current = [c.strip() for c in first.split(",")] if first else []
        if current == expected:
            return

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        bak = f"{csv_path}.bak_{ts}"
        os.replace(csv_path, bak)

        with open(bak, "r", encoding="utf-8", newline="") as fin, \
             open(csv_path, "w", encoding="utf-8", newline="") as fout:
            r = csv.DictReader(fin)
            w = csv.DictWriter(fout, fieldnames=expected)
            w.writeheader()
            for row in r:
                w.writerow({k: row.get(k, "") for k in expected})
        print(f"[metrics:migrate] header upgraded → {csv_path} (backup: {bak})")
    except Exception:
        # 어떤 문제든 조용히 패스 (append 시도는 아래에서 계속)
        pass


def _append_row(csv_path: str, row: dict) -> None:
    """단일 CSV에 헤더를 보장하며 행을 추가."""
    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_METRICS_HEADER)
        if write_header:
            w.writeheader()
        w.writerow(row)


def _sanitize_tag(tag: str) -> str:
    """폴더명 안전화를 위한 태그 정규화(영숫자, -, _, . 만 유지)."""
    safe = []
    for ch in str(tag):
        if ch.isalnum() or ch in "-_.":
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe).strip("_") or "untagged"


def save_metrics_autocsv(metrics: dict, cfg: Any, outputs: Optional[str] = None) -> str:
    """
    outputs/_logs/metrics.csv에 한 줄 append.
    추가로 tag가 있으면 outputs/_logs/<tag>/metrics.csv 에도 미러링.
    (extras는 파일에 쓰지 않음)
    """
    out_dir = outputs or getattr(cfg, "outputs", "./outputs")
    logs_dir = os.path.join(out_dir, "_logs")
    _ensure_dir(logs_dir)
    csv_path = os.path.join(logs_dir, "metrics.csv")

    _maybe_upgrade_header(csv_path, _METRICS_HEADER)

    method = getattr(cfg, "method", None)
    es_mode = getattr(cfg, "es_mode", None)
    hedge_on = str(getattr(cfg, "hedge", "off")).lower() == "on"
    hedge_mode = getattr(cfg, "hedge_mode", None)

    seeds_val = getattr(cfg, "seeds", None)
    if seeds_val:
        try:
            seeds_str = ",".join(str(s) for s in seeds_val)
        except Exception:
            seeds_str = str(seeds_val)
    else:
        seeds_str = None

    row = {
        "ts": _now_iso(),
        "asset": getattr(cfg, "asset", None),
        "method": method,
        "es_mode": es_mode,
        "alpha": getattr(cfg, "alpha", None),
        "lambda_term": getattr(cfg, "lambda_term", None),
        "F_target": getattr(cfg, "F_target", None),
        "EW": metrics.get("EW"),
        "ES95": metrics.get("ES95"),
        "EL": metrics.get("EL"),
        "Ruin": metrics.get("Ruin"),
        "mean_WT": metrics.get("mean_WT"),
        "HedgeHit": metrics.get("HedgeHit"),
        "HedgeKMean": metrics.get("HedgeKMean"),
        "HedgeActiveW": metrics.get("HedgeActiveW"),
        "fee_annual": getattr(cfg, "fee_annual", None),
        "w_max": getattr(cfg, "w_max", None),
        "floor_on": getattr(cfg, "floor_on", None),
        "f_min_real": getattr(cfg, "f_min_real", None),
        "hedge_on": bool(hedge_on),
        "hedge_mode": hedge_mode,
        "horizon_years": getattr(cfg, "horizon_years", None),
        "steps_per_year": getattr(cfg, "steps_per_year", None),
        "seeds": seeds_str,
        "n_paths_eval": getattr(cfg, "n_paths_eval", getattr(cfg, "rl_n_paths_eval", getattr(cfg, "n_paths", None))),
        "tag": getattr(cfg, "tag", None),
        # consumption
        "p10_c_last": metrics.get("p10_c_last"),
        "p50_c_last": metrics.get("p50_c_last"),
        "p90_c_last": metrics.get("p90_c_last"),
        "C_ES95_avg": metrics.get("C_ES95_avg"),
        "AlivePathRate": metrics.get("AlivePathRate"),
        # annuity overlay
        "y_ann": metrics.get("y_ann"),
        "a_factor": metrics.get("a_factor"),
        "P": metrics.get("P"),
        # sweep/overlay params
        "ann_alpha": getattr(cfg, "ann_alpha", None),
        # ES diagnostics
        "es95_source": metrics.get("es95_source"),
        "es95_note": metrics.get("es95_note"),
        # EU reporting
        "EU": metrics.get("EU"),
        "EU_per_year": metrics.get("EU_per_year"),
        "delta_annual": getattr(cfg, "delta_annual", None),
        # utility/reporting config & loss baseline audit
        "crra_gamma": getattr(cfg, "crra_gamma", None),
        "u_scale": getattr(cfg, "u_scale", None),
        "report_utility": (str(getattr(cfg, "report_utility", "off")).lower() == "on"),
        "F_target_used": metrics.get("F_target_used"),
    }

    # 글로벌 로그에 기록
    _append_row(csv_path, row)

    # 태그별 로그 미러링
    tag = getattr(cfg, "tag", None)
    if tag:
        safe_tag = _sanitize_tag(tag)
        tag_dir = os.path.join(logs_dir, safe_tag)
        _ensure_dir(tag_dir)
        tag_csv = os.path.join(tag_dir, "metrics.csv")
        _maybe_upgrade_header(tag_csv, _METRICS_HEADER)
        _append_row(tag_csv, row)

    return csv_path


# =========================
# (Optional) Frontier plot
# =========================

def plot_frontier_from_csv(csv_path: str, out_path: Optional[str] = None) -> Optional[str]:
    """EW–ES95 frontier를 metrics.csv에서 그려 저장(옵션)."""
    try:
        import csv as _csv
        import matplotlib.pyplot as plt  # optional

        xs, ys = [], []
        with open(csv_path, "r", encoding="utf-8") as f:
            r = _csv.DictReader(f)
            for row in r:
                try:
                    ew = float(row.get("EW", "nan"))
                    es = float(row.get("ES95", "nan"))
                    if _np.isfinite(ew) and _np.isfinite(es):
                        xs.append(ew); ys.append(es)
                except Exception:
                    continue

        if not xs:
            return None

        plt.figure()
        plt.scatter(xs, ys, s=16)
        plt.xlabel("EW (Expected Terminal Wealth)")
        plt.ylabel("ES95")
        plt.title("EW–ES95 frontier (from metrics.csv)")

        if out_path is None:
            base = os.path.dirname(csv_path)
            out_path = os.path.join(base, "frontier_EW_ES.png")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        return out_path
    except Exception:
        return None
