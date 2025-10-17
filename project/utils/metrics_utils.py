# project/utils/metrics_utils.py
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import numpy as np
# wealth 기반 ES 계산을 쓰는 경우 대비(없어도 정상 동작)
try:
    from project.metrics.es import es95_wealth  # optional
except Exception:  # pragma: no cover
    es95_wealth = None  # type: ignore


# ─────────────────────────────────────────────────────────
# 기본 유틸
# ─────────────────────────────────────────────────────────
def _first(d: Mapping[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _as_float(x: Any, nd: int | None = None) -> float:
    """안전한 float 변환 + NaN/Inf 가드(+선택 반올림)."""
    try:
        v = float(x)
        if math.isfinite(v):
            return round(v, nd) if isinstance(nd, int) else v
    except Exception:
        pass
    return float("nan")


def _w_label(w: Any) -> str:
    """w_fixed 숫자를 'w_0_3' 형태 라벨로 포맷."""
    if isinstance(w, str) and w.startswith("w_"):
        return w
    try:
        f = float(w)
        s = f"{f:.12g}"  # 불필요 0 제거
        return "w_" + s.replace(".", "_")
    except Exception:
        return "NA"


# ─────────────────────────────────────────────────────────
# 위험지표/손실 유틸
# ─────────────────────────────────────────────────────────
def terminal_losses(W_T: np.ndarray | List[float], F_target: float) -> np.ndarray:
    """L_i = max(F_target − W_T_i, 0)"""
    W_T = np.asarray(W_T, dtype=float).reshape(-1)
    return np.maximum(float(F_target) - W_T, 0.0)


def cvar_alpha(losses: np.ndarray | List[float], alpha: float = 0.95) -> float:
    """CVaR_α(loss) = E[ loss | loss ≥ VaR_α(loss) ]"""
    losses = np.asarray(losses, dtype=float).reshape(-1)
    if losses.size == 0:
        return 0.0
    # numpy 1.22+는 method= 사용, 하위 호환 가드
    try:
        q = np.quantile(losses, alpha, method="higher")
    except TypeError:
        q = np.quantile(losses, alpha)
    tail = losses[losses >= q]
    return float(tail.mean()) if tail.size else 0.0


def es95_from_wealth(W_T: np.ndarray | List[float], alpha: float = 0.95) -> float:
    """
    Wealth 기반 ES(좌측 꼬리). 외부 es95_wealth가 있으면 사용, 없으면 간단 구현.
    """
    if es95_wealth is not None:  # type: ignore
        try:
            return float(es95_wealth(W_T, alpha=alpha))  # type: ignore
        except Exception:
            pass
    W_T = np.asarray(W_T, dtype=float).reshape(-1)
    if W_T.size == 0:
        return float("nan")
    # 하위 α-분위수 이하의 평균(wealth tail의 기대값; 낮을수록 나쁨)
    try:
        q = np.quantile(W_T, 1 - (1 - alpha), method="lower")
    except TypeError:
        q = np.quantile(W_T, 1 - (1 - alpha))
    tail = W_T[W_T <= q]
    return float(np.mean(tail)) if tail.size else float("nan")


# ─────────────────────────────────────────────────────────
# metrics.csv 행 스키마 강제
#  - logging_io.write_metrics_csv()에서 사용
# ─────────────────────────────────────────────────────────
def ensure_metrics_schema(
    metrics: Dict[str, Any],
    meta: Dict[str, Any] | None,
    schema: List[str],
) -> Dict[str, Any]:
    """
    metrics/meta에서 표준 필드를 추출/보정해 `schema` 순서로 dict 반환.
    - 숫자형은 안전 변환(_as_float), 결측은 공백/NaN 처리
    - Ruin/RuinPct 호환
    - w_fixed 라벨 통일
    """
    m = metrics.get("metrics", metrics) if isinstance(metrics, dict) else {}
    meta = meta or metrics.get("meta", {}) if isinstance(metrics, dict) else (meta or {})

    # 공통 필드 스냅샷
    ruin = _first(m, ("RuinPct", "Ruin", "ruin"), default=0.0)
    w_fixed = _first(m, ("w_fixed", "w"), default="NA")
    baseline = _first(m, ("baseline",), default=_first(meta, ("baseline",), default=""))
    seeds = _first(meta, ("seeds",), default=_first(m, ("seeds",), default=""))

    # seeds → 문자열 통일
    if isinstance(seeds, (list, tuple)):
        seeds = ",".join(str(int(s)) for s in seeds)

    row = {
        "ts": _first(m, ("ts",), default=_first(meta, ("ts",), default="")),
        "asset": _first(m, ("asset",), default=_first(meta, ("asset",), default="")),
        "method": _first(m, ("method",), default=_first(meta, ("method",), default="")),
        "baseline": baseline,
        "es_mode": _first(m, ("es_mode",), default=""),
        "alpha": _first(m, ("alpha",), default=""),
        "lambda_term": _first(m, ("lambda_term",), default=""),
        "F_target": _first(m, ("F_target",), default=""),
        "EW": _as_float(_first(m, ("EW",), default=float("nan")), nd=6),
        "EL": _as_float(_first(m, ("EL",), default=float("nan")), nd=6),
        "ES95": _as_float(_first(m, ("ES95", "ES95_loss"), default=float("nan")), nd=6),
        "RuinPct": _as_float(ruin, nd=6),
        "mean_WT": _as_float(_first(m, ("mean_WT", "WT_avg"), default=float("nan")), nd=6),
        "HedgeHit": _as_float(_first(m, ("HedgeHit",), default=float("nan")), nd=6),
        "HedgeKMean": _as_float(_first(m, ("HedgeKMean",), default=float("nan")), nd=6),
        "HedgeActiveW": _as_float(_first(m, ("HedgeActiveW",), default=float("nan")), nd=6),
        "fee_annual": _first(m, ("fee_annual",), default=_first(meta, ("fee_annual",), default="")),
        "w_max": _first(m, ("w_max",), default=_first(meta, ("w_max",), default="")),
        "floor_on": _first(m, ("floor_on",), default=_first(meta, ("floor_on",), default=False)),
        "f_min_real": _first(m, ("f_min_real",), default=_first(meta, ("f_min_real",), default="")),
        "hedge_on": _first(m, ("hedge_on", "hedge"), default=_first(meta, ("hedge_on", "hedge"), default="")),
        "hedge_mode": _first(m, ("hedge_mode",), default=_first(meta, ("hedge_mode",), default="")),
        "hedge_sigma_k": _first(m, ("hedge_sigma_k",), default=_first(meta, ("hedge_sigma_k",), default="")),
        "hedge_cost": _first(m, ("hedge_cost",), default=_first(meta, ("hedge_cost",), default="")),
        "hedge_tx": _first(m, ("hedge_tx",), default=_first(meta, ("hedge_tx",), default="")),
        "horizon_years": _first(m, ("horizon_years",), default=_first(meta, ("horizon_years",), default="")),
        "steps_per_year": _first(m, ("steps_per_year",), default=_first(meta, ("steps_per_year",), default="")),
        "seeds": seeds,
        "n_paths_eval": _first(m, ("n_paths_eval","rl_n_paths_eval","n_paths"), default=""),
        "market_mode": _first(m, ("market_mode",), default=_first(meta, ("market_mode",), default="")),
        "market_csv": _first(m, ("market_csv",), default=_first(meta, ("market_csv",), default="")),
        "bootstrap_block": _first(m, ("bootstrap_block",), default=_first(meta, ("bootstrap_block",), default="")),
        "use_real_rf": _first(m, ("use_real_rf",), default=_first(meta, ("use_real_rf",), default="")),
        "time_total_hms": _first(m, ("time_total_hms",), default=_first(meta, ("time_total_hms",), default="")),
        "outputs_abs": _first(meta, ("outputs_abs",), default=""),
        "config_hash": _first(meta, ("config_hash",), default=""),
        "git_commit": _first(meta, ("git_commit",), default=""),
        "py_ver": _first(meta, ("py_ver",), default=""),
        "np_ver": _first(meta, ("np_ver",), default=""),
        "tag": _first(m, ("tag",), default=_first(meta, ("tag",), default="")),
    }

    # w_fixed 라벨 규격화 (스키마에 있을 때만)
    if "w_fixed" in schema:
        row["w_fixed"] = _w_label(_first(m, ("w_fixed", "w"), default="NA"))

    # 스키마 순서로 반환(누락 키는 공백)
    return {k: row.get(k, "") for k in schema}


# ─────────────────────────────────────────────────────────
# 검증 훅(초기 버전) — 필요시 확장
# ─────────────────────────────────────────────────────────
def validate_basic_rules(row: Mapping[str, Any], ctx: Mapping[str, Any] | None = None) -> Tuple[str, str]:
    """
    간단 점검:
      - 필수 지표(EW, ES95, RuinPct)가 숫자형인지
      - RuinPct ∈ [0,1] 대략 범위(느슨)
      - NaN/Inf 없는지
    반환: (status, note)  e.g., ("OK","") or ("WARN","Ruin out of range")
    """
    ew = _as_float(row.get("EW"))
    es = _as_float(row.get("ES95"))
    ru = _as_float(row.get("RuinPct"))

    if any(math.isnan(x) for x in (ew, es, ru)):
        return "WARN", "NaN in core metrics"

    if ru < 0.0 or ru > 1.0 + 1e-9:
        return "WARN", f"Ruin out of range: {ru}"

    return "OK", ""
