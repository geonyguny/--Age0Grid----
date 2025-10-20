# project/runner/pack_utils.py
from __future__ import annotations
from typing import Any, Dict, Iterable, Optional, Tuple, List
from ..eval import save_metrics_autocsv  # ← 추가: CSV 기록 백업용

from .cvar_utils import maybe_extract_WT

# ---- evaluate 동적 임포트 (있으면 사용) ----
def _import_evaluate():
    candidates = [
        "project.runner.evaluate",
        "project.evaluate",
        "project.runner.eval",
        "project.eval",
    ]
    for name in candidates:
        try:
            mod = __import__(name, fromlist=["evaluate"])
            return getattr(mod, "evaluate")
        except Exception:
            continue
    return None

_evaluate = _import_evaluate()  # type: ignore


# ---- n_paths 추정 ----
def _estimate_n_paths(args, out: Dict[str, Any]) -> Optional[int]:
    try:
        if isinstance(out, dict):
            np_exist = out.get("n_paths")
            if isinstance(np_exist, (int, float)) and int(np_exist) > 0:
                return int(np_exist)

        seeds = getattr(args, "seeds", [])
        if isinstance(seeds, int):
            n_seeds = 1
        elif isinstance(seeds, (list, tuple)):
            n_seeds = max(1, len(seeds))
        else:
            n_seeds = 1

        if str(getattr(args, "method", "hjb")).lower() == "rl":
            n_eval = int(getattr(args, "rl_n_paths_eval", 0) or 0)
            if n_eval > 0:
                return n_eval * n_seeds

        n_base = int(getattr(args, "n_paths", 0) or 0)
        if n_base > 0:
            return n_base * n_seeds
    except Exception:
        pass
    return None


# ---- stdout 경량화/포맷 ----
def prune_for_stdout(args, out: Dict[str, Any]) -> Any:
    """
    print_mode 에 따라 출력 페이로드를 경량화.
      - full: 원본 그대로
      - metrics: 지정된 metrics_keys 만 추출 + 메타(시간/태그/자산/방법/n_paths)
      - summary: 메트릭 서브셋 + 핵심 메타만
    또한 --no_paths 인 경우 extra.eval_WT/ruin_flags를 개수 필드로 대체.
    """
    def _sel_metrics(md: Dict[str, Any], keys: Iterable[str]) -> Dict[str, Any]:
        return {k: md[k] for k in md if k in keys}

    # 큰 배열 제거 옵션
    if getattr(args, "no_paths", False) and isinstance(out, dict):
        out = dict(out)
        extra = out.get("extra")
        if isinstance(extra, dict):
            for k in ("eval_WT", "ruin_flags"):
                if k in extra and isinstance(extra[k], (list, tuple)):
                    try:
                        extra[k + "_n"] = len(extra[k])
                    except Exception:
                        pass
                    # 실제 배열 제거
                    del extra[k]
            out["extra"] = extra

    mode = str(getattr(args, "print_mode", "full")).lower()
    if mode == "full":
        return out

    metrics = out["metrics"] if isinstance(out, dict) and isinstance(out.get("metrics"), dict) else out
    # cli.py 기본값이 EU/EU_per_year/delta_annual/F_target_used 포함이므로 그대로 따름
    keys = [s.strip() for s in str(getattr(args, "metrics_keys", "")).split(",") if s.strip()]

    n_paths_guess = _estimate_n_paths(args, out)

    # summary용 보조 메타(있으면 표시)
    age0_guess, sex_guess = None, None
    try:
        if isinstance(out, dict):
            age0_guess = out.get("age0")
            sex_guess = out.get("sex")
        if age0_guess is None:
            age0_guess = getattr(args, "age0", None)
        if sex_guess is None:
            sex_guess = getattr(args, "sex", None)
    except Exception:
        pass

    if mode == "metrics":
        mini = _sel_metrics(metrics, keys)
        if isinstance(out, dict):
            mini["time_total_s"] = out.get("time_total_s")
            mini["time_total_hms"] = out.get("time_total_hms")
        mini.update({
            "tag": (out.get("tag") if isinstance(out, dict) else None) or getattr(args, "tag", None),
            "asset": (out.get("asset") if isinstance(out, dict) else None) or getattr(args, "asset", None),
            "method": (out.get("method") if isinstance(out, dict) else None) or getattr(args, "method", None),
            "n_paths": n_paths_guess,
        })
        return mini

    if mode == "summary":
        args_dict = out.get("args", {}) if isinstance(out, dict) else {}
        top_tag    = (out.get("tag") if isinstance(out, dict) else None) or getattr(args, "tag", None)
        top_method = (out.get("method") if isinstance(out, dict) else None) or getattr(args, "method", None)
        top_asset  = (out.get("asset") if isinstance(out, dict) else None) or getattr(args, "asset", None)
        summary_obj = {
            "tag": top_tag,
            "asset": top_asset,
            "method": top_method,
            "age0": age0_guess if age0_guess is not None else (args_dict or {}).get("age0"),
            "sex": sex_guess if sex_guess is not None else (args_dict or {}).get("sex"),
            "metrics": _sel_metrics(metrics, keys),
            "n_paths": n_paths_guess,
            "T": (out.get("extra") or {}).get("T") if isinstance(out, dict) else None,
            "time_total_s": out.get("time_total_s") if isinstance(out, dict) else None,
            "time_total_hms": out.get("time_total_hms") if isinstance(out, dict) else None,
        }
        if isinstance(out, dict) and isinstance(out.get("cvar_calibration"), dict):
            summary_obj["cvar_calibration"] = out["cvar_calibration"]
        return summary_obj

    return out


# ---- cfg/actor 추출 ----
def try_extract_cfg_actor(res: Any) -> Tuple[Optional[Any], Optional[Any]]:
    if isinstance(res, tuple) and len(res) >= 2:
        return res[0], res[1]
    cfg = getattr(res, "cfg", None)
    actor = getattr(res, "actor", None) or getattr(res, "policy", None)
    if cfg is not None or actor is not None:
        return cfg, actor
    if isinstance(res, dict):
        cfg = res.get("args") or res.get("cfg")
        actor = res.get("actor") or res.get("policy") or res.get("pi")
        if cfg is not None or actor is not None:
            return cfg, actor
    return None, None


# ---- 필요 시 evaluate 재호출 (경로 포함) ----
def maybe_evaluate_with_es_mode(res: Any, es_mode: str, want_paths: bool = False) -> Dict[str, Any]:
    """
    run_once/run_rl 의 다양한 반환 포맷을 표준 페이로드로 래핑.
    want_paths=True 이고 경로가 비어있으며 evaluate 가 임포트되어 있고 (cfg, actor)가 있으면
    한 번 더 evaluate 를 호출해 경로를 채운다.
    """
    if isinstance(res, tuple) and len(res) >= 1 and isinstance(res[0], dict):
        pack: Dict[str, Any] = {"metrics": dict(res[0])}
        if len(res) >= 2 and isinstance(res[1], dict):
            pack["extra"] = dict(res[1])
        if "es_mode" not in pack["metrics"]:
            pack["metrics"]["es_mode"] = str(es_mode).lower()
    elif isinstance(res, dict):
        if "metrics" in res and isinstance(res["metrics"], dict):
            pack = {"metrics": dict(res["metrics"])}
            if isinstance(res.get("extra"), dict):
                pack["extra"] = dict(res["extra"])
            # 메타 필드 복사
            for k in ("asset","method","w_max","fee_annual","lambda_term","alpha","F_target","outputs","tag","n_paths","args"):
                if k in res:
                    pack[k] = res[k]
            if "es_mode" not in pack["metrics"]:
                pack["metrics"]["es_mode"] = str(es_mode).lower()
        else:
            pack = {"metrics": dict(res)}
            if "es_mode" not in pack["metrics"]:
                pack["metrics"]["es_mode"] = str(es_mode).lower()
    else:
        return {
            "result": "ok",
            "note": "evaluate not executed in cli (no evaluate import or unexpected return type).",
            "es_mode": str(es_mode).lower(),
        }

    # 이미 경로가 있으면 재평가 불필요
    have_paths = False
    try:
        wt0 = maybe_extract_WT({"metrics": pack.get("metrics", {}), "extra": pack.get("extra", {})})
        have_paths = wt0 is not None and len(list(wt0)) > 0
    except Exception:
        have_paths = False

    if want_paths and (not have_paths) and _evaluate is not None:
        cfg, actor = try_extract_cfg_actor(res)
        if cfg is None:
            cfg, actor = try_extract_cfg_actor(pack)
        if (cfg is not None) and (actor is not None):
            try:
                m = None
                try:
                    m = _evaluate(cfg, actor, es_mode=str(es_mode).lower(), return_paths=True)  # type: ignore
                except TypeError:
                    m = _evaluate(cfg, actor, es_mode=str(es_mode).lower())  # type: ignore

                if isinstance(m, tuple) and len(m) >= 1 and isinstance(m[0], dict):
                    pack["metrics"] = m[0]
                    if len(m) >= 2 and isinstance(m[1], dict):
                        pack["extra"] = m[1]
                elif isinstance(m, dict):
                    pack["metrics"] = m

                if "es_mode" not in pack.get("metrics", {}):
                    pack["metrics"]["es_mode"] = str(es_mode).lower()

                wt_paths = maybe_extract_WT({"metrics": pack.get("metrics", {}), "extra": pack.get("extra", {})})
                if wt_paths is not None:
                    if "extra" not in pack or not isinstance(pack["extra"], dict):
                        pack["extra"] = {}
                    pack["extra"]["eval_WT"] = list(wt_paths)
            except Exception:
                # 재평가 실패는 조용히 무시 (상위 단계가 후처리)
                pass

    return pack
