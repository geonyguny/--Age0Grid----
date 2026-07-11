# project/utils/logging_io.py
from __future__ import annotations

import os, csv, json, time, math, sys
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional

# ─────────────────────────────────────────────────────────
# 스키마(필드 순서 고정) + 버전
# ─────────────────────────────────────────────────────────
SCHEMA_VERSION = "metrics.v3"

METRIC_HEADER: List[str] = [
    # 식별/버전
    "schema","ts","tag","outputs_abs",
    # 실행 구성
    "asset","method","baseline","es_mode",
    "alpha","lambda_term","F_target",
    "fee_annual","w_max","w_fixed","q_floor","horizon_years","steps_per_year",
    # 시장/데이터 메타
    "market_mode","market_csv","bootstrap_block","use_real_rf","data_window",
    # 통계 파라미터
    "seeds","n_paths_eval",
    # 결과 지표
    "EW","EL","ES95","Ruin","mean_WT",
    # 헤지 관련
    "hedge_on","hedge_mode","hedge_sigma_k","hedge_cost","hedge_tx",
    "HedgeHit","HedgeKMean","HedgeActiveW",
    # 감사/결정성 메타
    "time_total_hms","config_hash","git_commit","py_ver","np_ver",
]

# ─────────────────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────────────────
def _as_dict(d: Any) -> Dict[str, Any]:
    if d is None: return {}
    if isinstance(d, Mapping): return dict(d)
    return dict(getattr(d, "__dict__", {}))

def _num(x: Any, nd: int = 6) -> float:
    try:
        v = float(x)
        if math.isfinite(v):
            # 0.0을 -0.0으로 찍지 않게 보정
            vv = round(v, nd)
            return 0.0 if vv == -0.0 else vv
    except Exception:
        pass
    return float("nan")

def _get(d: Mapping[str, Any], key: str, default: Any = None) -> Any:
    v = d.get(key, default)
    return v if v is not None else default

def _ensure_parent(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

def _to_onoff(v: Any, default: str = "") -> str:
    if v is None: return default
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("on","off"): return s
        if s in ("true","1","y","yes"): return "on"
        if s in ("false","0","n","no"): return "off"
        return s
    if isinstance(v, (bool,int)):
        return "on" if bool(v) else "off"
    return default

def _to_csv_list(v: Any) -> str:
    if v is None: return ""
    if isinstance(v, (list, tuple)):
        # 정수/실수 혼합도 문자열로 안전 직렬화
        return " ".join(str(i) for i in v)
    return str(v)

def _abs_or(v: Optional[str], fallback: str = "") -> str:
    try:
        if v: return os.path.abspath(v)
    except Exception:
        pass
    return fallback

# ─────────────────────────────────────────────────────────
# 표준 CSV 기록 함수
# ─────────────────────────────────────────────────────────
def write_metrics_csv(
    csv_path: str,
    args: Any,
    metrics: Dict[str, Any],
    meta: Optional[Dict[str, Any]] = None
) -> str:
    """
    metrics.csv에 스키마 고정으로 1행 append.
    - csv_path: 보통 <OutDir>/_logs/metrics.csv
    - args: argparse.Namespace 또는 dict
    - metrics: run 결과(dict)
    - meta: outputs_abs/config_hash/git_commit/py_ver/np_ver 등(선택)
    """
    _ensure_parent(csv_path)
    file_exists = os.path.isfile(csv_path)

    A = _as_dict(args)
    M = dict(meta or {})
    m = metrics.get("metrics", metrics) if isinstance(metrics, dict) else {}

    # seeds 표기: 리스트/튜플 → 공백-구분 문자열, 단일값은 그대로
    seeds_val = _get(A, "seeds", None)
    if isinstance(seeds_val, (list, tuple)):
        seeds_out = " ".join(str(int(s)) for s in seeds_val)
    else:
        seeds_out = "" if seeds_val is None else str(seeds_val)

    # outputs 절대경로
    outputs_abs = _abs_or(_get(M, "outputs_abs", _get(A, "outputs", "")))

    row: Dict[str, Any] = {
        # 식별/버전
        "schema": SCHEMA_VERSION,
        "ts": datetime.utcnow().isoformat(timespec="seconds"),
        "tag": _get(A, "tag", m.get("tag","")),
        "outputs_abs": outputs_abs,
        # 실행 구성
        "asset": _get(A, "asset", ""),
        "method": _get(A, "method", ""),
        "baseline": _get(A, "baseline", ""),
        "es_mode": _get(A, "es_mode", ""),
        "alpha": _get(A, "alpha", ""),
        "lambda_term": _get(A, "lambda_term", ""),
        "F_target": _get(A, "F_target", ""),
        "fee_annual": _get(A, "fee_annual", _get(A, "phi_adval", "")),
        "w_max": _get(A, "w_max", ""),
        "w_fixed": _get(A, "w_fixed", ""),          # RULE 스윕용
        "q_floor": _get(A, "q_floor", ""),
        "horizon_years": _get(A, "horizon_years", ""),
        "steps_per_year": _get(A, "steps_per_year", ""),
        # 시장/데이터 메타
        "market_mode": _get(A, "market_mode", ""),
        "market_csv": _abs_or(_get(A, "market_csv", "")),
        "bootstrap_block": _get(A, "bootstrap_block", ""),
        "use_real_rf": _to_onoff(_get(A, "use_real_rf", "")),
        "data_window": _get(A, "data_window",""),
        # 통계 파라미터
        "seeds": seeds_out,
        "n_paths_eval": _get(A, "rl_n_paths_eval", _get(A, "n_paths", "")),
        # 결과 지표
        "EW": _num(m.get("EW")),
        "EL": _num(m.get("EL")),
        "ES95": _num(m.get("ES95")),
        "Ruin": _num(m.get("Ruin")),
        "mean_WT": _num(m.get("mean_WT")),
        # 헤지
        "hedge_on": _to_onoff(_get(A, "hedge", _get(A, "hedge_on", ""))),
        "hedge_mode": _get(A, "hedge_mode", ""),
        "hedge_sigma_k": _get(A, "hedge_sigma_k", ""),
        "hedge_cost": _get(A, "hedge_cost", ""),
        "hedge_tx": _get(A, "hedge_tx", ""),
        "HedgeHit": _num(m.get("HedgeHit")),
        "HedgeKMean": _num(m.get("HedgeKMean")),
        "HedgeActiveW": _num(m.get("HedgeActiveW")),
        # 감사/결정성 메타
        "time_total_hms": m.get("time_total_hms", _get(M, "time_total_hms", "")),
        "config_hash": _get(M, "config_hash", ""),
        "git_commit": _get(M, "git_commit", ""),
        "py_ver": _get(M, "py_ver", sys.version.split()[0]),
        "np_ver": _get(M, "np_ver", ""),
    }

    # 필드 순서 고정
    ordered = [row.get(k, "") for k in METRIC_HEADER]

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(METRIC_HEADER)
        w.writerow(ordered)

    return csv_path

# ─────────────────────────────────────────────────────────
# 레거시 호환 진입점
# ─────────────────────────────────────────────────────────
def save_metrics_autocsv(csv_path: str, args: dict, metrics: dict, tag: str = ""):
    A = _as_dict(args)
    if not A.get("tag"):
        A["tag"] = tag
    return write_metrics_csv(csv_path=csv_path, args=A, metrics=metrics, meta=None)

# ─────────────────────────────────────────────────────────
# JSON 덤프(변경 없음)
# ─────────────────────────────────────────────────────────
def dump_result_json(path: str, args: dict, metrics: dict, extras: dict | None = None):
    _ensure_parent(path)
    payload = {
        "ts": int(time.time()),
        "args": _as_dict(args),
        "metrics": metrics,
        "extras": extras or {}
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)