# project/utils/logging_io.py
from __future__ import annotations

import os, csv, json, time, math
from datetime import datetime
from typing import Any, Dict, List, Mapping

# ─────────────────────────────────────────────────────────
# 스키마(필드 순서 고정)
#  - 기존 헤더와 호환 유지 + 결정성 메타 일부 추가
# ─────────────────────────────────────────────────────────
METRIC_HEADER: List[str] = [
    "ts","asset","method","baseline","es_mode","alpha","lambda_term","F_target",
    "EW","EL","ES95","Ruin","mean_WT",
    "HedgeHit","HedgeKMean","HedgeActiveW",
    "fee_annual","w_max","floor_on","f_min_real",
    "hedge_on","hedge_mode","hedge_sigma_k","hedge_cost","hedge_tx",
    "horizon_years","steps_per_year","seeds","n_paths_eval",
    "market_mode","market_csv","bootstrap_block","use_real_rf",
    # 결정성/감사 메타 (있으면 기록, 없어도 스키마는 고정)
    "time_total_hms","outputs_abs","config_hash","git_commit","py_ver","np_ver",
    "tag"
]

# ─────────────────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────────────────
def _as_dict(d: Any) -> Dict[str, Any]:
    """argparse.Namespace 또는 dict 모두 수용."""
    if d is None:
        return {}
    if isinstance(d, Mapping):
        return dict(d)
    return dict(getattr(d, "__dict__", {}))

def _num(x: Any, nd: int = 6) -> float:
    """숫자형 강제 + NaN/Inf 가드 + 반올림."""
    try:
        v = float(x)
        if math.isfinite(v):
            return round(v, nd)
    except Exception:
        pass
    return float("nan")

def _get(d: Mapping[str, Any], key: str, default: Any = None) -> Any:
    v = d.get(key, default)
    return v if v is not None else default

def _ensure_parent(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

# ─────────────────────────────────────────────────────────
# 표준 CSV 기록 함수 (신규): 내부에서 스키마 강제
# ─────────────────────────────────────────────────────────
def write_metrics_csv(csv_path: str, args: Any, metrics: Dict[str, Any], meta: Dict[str, Any] | None = None) -> str:
    """
    metrics.csv에 '필드 순서 고정'으로 1행 append.
    - csv_path: 보통 <OutDir>/_logs/metrics.csv
    - args: argparse.Namespace 또는 dict
    - metrics: run 결과(metrics dict 또는 상위 dict)
    - meta: outputs_abs/config_hash/git_commit/py_ver/np_ver 등(선택)
    """
    _ensure_parent(csv_path)
    file_exists = os.path.isfile(csv_path)

    A = _as_dict(args)
    M = dict(meta or {})
    # metrics가 상위 딕셔너리일 수도 있어 방어적으로 추출
    m = metrics.get("metrics", metrics) if isinstance(metrics, dict) else {}

    # seeds 표기 통일(리스트면 콤마조인, 정수면 그대로)
    seeds_val = _get(A, "seeds", None)
    if isinstance(seeds_val, (list, tuple)):
        seeds_str = ",".join(str(int(s)) for s in seeds_val)
    else:
        seeds_str = str(seeds_val) if seeds_val is not None else ""

    row: Dict[str, Any] = {
        "ts": datetime.utcnow().isoformat(timespec="seconds"),
        "asset": _get(A, "asset", ""),
        "method": _get(A, "method", ""),
        "baseline": _get(A, "baseline", ""),
        "es_mode": _get(A, "es_mode", ""),
        "alpha": _get(A, "alpha", ""),
        "lambda_term": _get(A, "lambda_term", ""),
        "F_target": _get(A, "F_target", ""),
        # 결과 지표
        "EW": _num(m.get("EW")),
        "EL": _num(m.get("EL")),
        "ES95": _num(m.get("ES95")),
        "Ruin": _num(m.get("Ruin")),
        "mean_WT": _num(m.get("mean_WT")),
        # 헤지 관련(없으면 NaN)
        "HedgeHit": _num(m.get("HedgeHit")),
        "HedgeKMean": _num(m.get("HedgeKMean")),
        "HedgeActiveW": _num(m.get("HedgeActiveW")),
        # 실행 파라미터 스냅샷
        "fee_annual": _get(A, "fee_annual", _get(A, "phi_adval", "")),
        "w_max": _get(A, "w_max", ""),
        "floor_on": _get(A, "floor_on", False),
        "f_min_real": _get(A, "f_min_real", ""),
        "hedge_on": _get(A, "hedge", _get(A, "hedge_on", "")),
        "hedge_mode": _get(A, "hedge_mode", ""),
        "hedge_sigma_k": _get(A, "hedge_sigma_k", ""),
        "hedge_cost": _get(A, "hedge_cost", ""),
        "hedge_tx": _get(A, "hedge_tx", ""),
        "horizon_years": _get(A, "horizon_years", ""),
        "steps_per_year": _get(A, "steps_per_year", ""),
        "seeds": seeds_str,
        "n_paths_eval": _get(A, "rl_n_paths_eval", _get(A, "n_paths", "")),
        "market_mode": _get(A, "market_mode", ""),
        "market_csv": _get(A, "market_csv", ""),
        "bootstrap_block": _get(A, "bootstrap_block", ""),
        "use_real_rf": _get(A, "use_real_rf", ""),
        # 결정성/감사 메타 (optional)
        "time_total_hms": m.get("time_total_hms", _get(M, "time_total_hms", "")),
        "outputs_abs": _get(M, "outputs_abs", _get(A, "outputs", "")),
        "config_hash": _get(M, "config_hash", ""),
        "git_commit": _get(M, "git_commit", ""),
        "py_ver": _get(M, "py_ver", ""),
        "np_ver": _get(M, "np_ver", ""),
        "tag": _get(A, "tag", m.get("tag", "")),
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
# 기존 인터페이스(호환) — 내부적으로 표준 함수로 위임
# ─────────────────────────────────────────────────────────
def save_metrics_autocsv(csv_path: str, args: dict, metrics: dict, tag: str = ""):
    """
    (레거시) 기존 호출부 호환:
      - csv_path: <OutDir>/_logs/metrics.csv 형태를 권장
      - tag 파라미터는 args['tag']가 비어있을 때만 사용
    """
    A = _as_dict(args)
    if not A.get("tag"):
        A["tag"] = tag
    return write_metrics_csv(csv_path=csv_path, args=A, metrics=metrics, meta=None)

# ─────────────────────────────────────────────────────────
# JSON 덤프 (그대로 유지, 폴더 보장/UTF-8)
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
