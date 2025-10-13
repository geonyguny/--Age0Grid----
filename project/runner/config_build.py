# project/runner/config_build.py
from __future__ import annotations
from typing import Any, Iterable, Sequence, Tuple
import numpy as _np

from ..config import SimConfig, ASSET_PRESETS
from .helpers import auto_eta_grid


# -------------------------
# small helpers
# -------------------------

def _get(obj: Any, name: str, default: Any = None) -> Any:
    """getattr(obj, name, default)를 짧게."""
    return getattr(obj, name, default)


def _as_tuple(x, *, empty_ok: bool = True) -> Tuple:
    """단일/시퀀스를 tuple로 정규화."""
    if x is None:
        return tuple() if empty_ok else (0,)
    if isinstance(x, (list, tuple)):
        return tuple(x)
    return (x,)


def _normalize_hjb_w_grid(cfg: SimConfig, raw) -> None:
    """
    HJB의 W-격자 정규화.
      - int: 0..w_max 균등분할
      - Iterable[float]: 클립/정렬/중복제거 후 사용
      - None/빈값: 8개 기본격자
    결과: cfg.hjb_w_grid = tuple(float)
    """
    w_max = float(getattr(cfg, "w_max", 1.0) or 1.0)

    if raw is None or (isinstance(raw, (list, tuple)) and len(raw) == 0):
        grid = _np.linspace(0.0, w_max, 8)
    elif isinstance(raw, int):
        n_w = max(2, int(raw))
        grid = _np.linspace(0.0, w_max, n_w)
    elif isinstance(raw, Iterable):
        arr = _np.asarray(list(raw), dtype=float)
        if arr.size < 2:
            arr = _np.linspace(0.0, w_max, 8)
        arr = _np.clip(arr, 0.0, w_max)
        grid = _np.unique(_np.round(arr, 6))
        if grid.size < 2:
            grid = _np.linspace(0.0, w_max, 8)
    else:
        grid = _np.linspace(0.0, w_max, 8)

    cfg.hjb_w_grid = tuple(_np.round(grid, 4))


def _set_fee_fields(cfg: SimConfig, args) -> None:
    """
    phi_adval(선취) vs fee_annual(연 운용보수) 정합성.
    - 둘 다 주어지면 phi_adval 우선.
    - 하나만 주어지면 동일값으로 두 필드 세팅(하위호환).
    - 미지정 시 디폴트 유지(0.004).
    """
    fee_annual = _get(args, "fee_annual", None)
    phi_adval  = _get(args, "phi_adval", None)

    if phi_adval is not None:
        cfg.phi_adval = float(phi_adval)
        cfg.fee_annual = float(fee_annual if fee_annual is not None else phi_adval)
    elif fee_annual is not None:
        cfg.fee_annual = float(fee_annual)
        cfg.phi_adval = float(fee_annual)
    else:
        cfg.fee_annual = float(getattr(cfg, "fee_annual", 0.004) or 0.004)
        cfg.phi_adval = float(getattr(cfg, "phi_adval", cfg.fee_annual) or cfg.fee_annual)


def _set_q_floor_monthly(cfg: SimConfig, args) -> None:
    """
    입력 q_floor가 '연 환산 소비비율'이면 월로 변환하여 cfg.q_floor에 저장.
    - args.q_floor가 None이면 기존 cfg 값 유지(또는 0.0).
    """
    qf_ann = _get(args, "q_floor", None)
    spm = int(getattr(cfg, "steps_per_year", 12) or 12)
    if qf_ann is None:
        cfg.q_floor = float(getattr(cfg, "q_floor", 0.0) or 0.0)
        cfg.q_floor_annual = float(_get(cfg, "q_floor_annual", 0.0) or 0.0)
        return

    qf_ann = float(qf_ann)
    qf_ann = max(0.0, min(0.999999, qf_ann))  # 안정화
    qf_m = 1.0 - (1.0 - qf_ann) ** (1.0 / spm)
    cfg.q_floor = float(qf_m)
    cfg.q_floor_annual = float(qf_ann)
    print(f"[cfg] q_floor_annual={qf_ann:.6f} → q_floor_monthly={qf_m:.6f} (steps_per_year={spm})")


def _choose_n_paths_eval(args, default_val: int = 100) -> int:
    """CLI의 --n_paths와 호환되는 평가 경로 수 결정."""
    n = int(_get(args, "n_paths", 0) or 0)
    if n <= 0:
        n = default_val
    return int(n)


def _maybe_set(cfg: SimConfig, args, names: Sequence[str]) -> None:
    """args에 존재하는 필드만 cfg에 옮긴다."""
    for nm in names:
        val = _get(args, nm, None)
        if val is not None:
            setattr(cfg, nm, val)


# -------------------------
# build
# -------------------------

def make_cfg(args) -> SimConfig:
    cfg = SimConfig()

    # 0) 자산 프리셋 반영
    if _get(args, "asset", None) in ASSET_PRESETS:
        for k, v in ASSET_PRESETS[args.asset].items():  # type: ignore
            setattr(cfg, k, v)
    cfg.asset = _get(args, "asset", getattr(cfg, "asset", None))

    # 1) steps_per_year & horizon 확정(최소값 보장)
    steps_per_year = int(_get(args, "steps_per_year", getattr(cfg, "steps_per_year", 12)) or 12)
    steps_per_year = max(1, steps_per_year)
    cfg.steps_per_year = steps_per_year
    cfg.horizon_years = int(_get(args, "horizon_years", getattr(cfg, "horizon_years", 15)) or 15)

    # 2) 주요 인자 일괄 반영(존재할 때만 세팅)
    bulk = dict(
        # 핵심
        w_max=_get(args, "w_max", getattr(cfg, "w_max", None)),
        horizon_years=cfg.horizon_years,
        lambda_term=_get(args, "lambda_term", getattr(cfg, "lambda_term", None)),
        alpha=_get(args, "alpha", getattr(cfg, "alpha", None)),
        baseline=_get(args, "baseline", getattr(cfg, "baseline", None)),
        p_annual=_get(args, "p_annual", getattr(cfg, "p_annual", None)),
        g_real_annual=_get(args, "g_real_annual", getattr(cfg, "g_real_annual", None)),
        w_fixed=_get(args, "w_fixed", getattr(cfg, "w_fixed", None)),
        floor_on=_get(args, "floor_on", getattr(cfg, "floor_on", None)),
        f_min_real=_get(args, "f_min_real", getattr(cfg, "f_min_real", None)),
        F_target=_get(args, "F_target", getattr(cfg, "F_target", None)),
        hjb_W_grid=_get(args, "hjb_W_grid", getattr(cfg, "hjb_W_grid", None)),
        hjb_Nshock=_get(args, "hjb_Nshock", getattr(cfg, "hjb_Nshock", None)),

        # 헤지/시장
        hedge=_get(args, "hedge", getattr(cfg, "hedge", "off")),
        hedge_on=(str(_get(args, "hedge", "off")).lower() == "on"),
        hedge_mode=_get(args, "hedge_mode", getattr(cfg, "hedge_mode", None)),
        hedge_cost=_get(args, "hedge_cost", getattr(cfg, "hedge_cost", None)),
        hedge_sigma_k=_get(args, "hedge_sigma_k", getattr(cfg, "hedge_sigma_k", None)),
        hedge_tx=_get(args, "hedge_tx", getattr(cfg, "hedge_tx", None)),
        market_mode=_get(args, "market_mode", getattr(cfg, "market_mode", "iid")),
        market_csv=_get(args, "market_csv", getattr(cfg, "market_csv", "")),
        bootstrap_block=_get(args, "bootstrap_block", getattr(cfg, "bootstrap_block", 24)),
        use_real_rf=_get(args, "use_real_rf", getattr(cfg, "use_real_rf", "on")),

        # 사망/연금
        mortality=_get(args, "mortality", getattr(cfg, "mortality", "off")),
        mortality_on=(str(_get(args, "mortality", "off")).lower() == "on"),
        mort_table=_get(args, "mort_table", getattr(cfg, "mort_table", None)),
        age0=_get(args, "age0", getattr(cfg, "age0", 65)),
        sex=_get(args, "sex", getattr(cfg, "sex", "M")),
        bequest_kappa=_get(args, "bequest_kappa", getattr(cfg, "bequest_kappa", None)),
        bequest_gamma=_get(args, "bequest_gamma", getattr(cfg, "bequest_gamma", None)),

        # RL
        rl_q_cap=_get(args, "rl_q_cap", getattr(cfg, "rl_q_cap", None)),
        teacher_eps0=_get(args, "teacher_eps0", getattr(cfg, "teacher_eps0", None)),
        teacher_decay=_get(args, "teacher_decay", getattr(cfg, "teacher_decay", None)),
        lw_scale=_get(args, "lw_scale", getattr(cfg, "lw_scale", None)),

        # 보상/효용
        survive_bonus=_get(args, "survive_bonus", getattr(cfg, "survive_bonus", None)),
        crra_gamma=_get(args, "crra_gamma", getattr(cfg, "crra_gamma", None)),
        u_scale=_get(args, "u_scale", getattr(cfg, "u_scale", None)),

        # Stage-wise CVaR
        cvar_stage_on=(str(_get(args, "cvar_stage", "off")).lower() == "on"),
        alpha_stage=_get(args, "alpha_stage", getattr(cfg, "alpha_stage", None)),
        lambda_stage=_get(args, "lambda_stage", getattr(cfg, "lambda_stage", None)),
        cstar_mode=_get(args, "cstar_mode", getattr(cfg, "cstar_mode", None)),
        cstar_m=_get(args, "cstar_m", getattr(cfg, "cstar_m", None)),

        # XAI
        xai_on=(str(_get(args, "xai_on", "off")).lower() == "on"),

        # 연금 오버레이
        ann_on=_get(args, "ann_on", getattr(cfg, "ann_on", "off")),
        ann_alpha=_get(args, "ann_alpha", getattr(cfg, "ann_alpha", 0.0)),
        ann_L=_get(args, "ann_L", getattr(cfg, "ann_L", 0.0)),
        ann_d=_get(args, "ann_d", getattr(cfg, "ann_d", 0)),
        ann_index=_get(args, "ann_index", getattr(cfg, "ann_index", "real")),

        # 태그/메타
        tag=_get(args, "tag", getattr(cfg, "tag", None)),
        steps_per_year=steps_per_year,  # 확정 반영

        # (NEW) EU 리포팅 옵션
        report_utility=_get(args, "report_utility", getattr(cfg, "report_utility", "off")),
        delta_annual=_get(args, "delta_annual", getattr(cfg, "delta_annual", None)),

        # (보강) 시장데이터 선택 파라미터(넘겨주기만)
        data_window=_get(args, "data_window", getattr(cfg, "data_window", None)),
        data_profile=_get(args, "data_profile", getattr(cfg, "data_profile", None)),
        alpha_mix=_get(args, "alpha_mix", getattr(cfg, "alpha_mix", None)),
        alpha_kr=_get(args, "alpha_kr", getattr(cfg, "alpha_kr", None)),
        alpha_us=_get(args, "alpha_us", getattr(cfg, "alpha_us", None)),
        alpha_au=_get(args, "alpha_au", getattr(cfg, "alpha_au", None)),
        h_FX=_get(args, "h_FX", getattr(cfg, "h_FX", None)),
        fx_hedge_cost=_get(args, "fx_hedge_cost", getattr(cfg, "fx_hedge_cost", None)),
    )
    for k, v in bulk.items():
        if v is not None:
            setattr(cfg, k, v)

    # 3) 수수료 필드 정합성(우선순위: phi_adval > fee_annual)
    _set_fee_fields(cfg, args)

    # 4) seeds & n_paths_eval 확정
    seeds = list(_as_tuple(_get(args, "seeds", (0,))))
    if len(seeds) == 0:
        seeds = [0]
    cfg.seeds = tuple(int(s) for s in seeds)

    cfg.n_paths_eval = _choose_n_paths_eval(args, default_val=100)
    # RL용 평가 경로 수 (미지정 시 n_paths_eval 사용)
    if getattr(args, "rl_n_paths_eval", None) is not None:
        cfg.rl_n_paths_eval = int(args.rl_n_paths_eval)
    else:
        cfg.rl_n_paths_eval = int(cfg.n_paths_eval)

    # 5) 출력/메타
    cfg.outputs = _get(args, "outputs", getattr(cfg, "outputs", "./outputs"))
    cfg.method = _get(args, "method", getattr(cfg, "method", "hjb"))
    cfg.es_mode = _get(args, "es_mode", getattr(cfg, "es_mode", "wealth"))

    # 6) q_floor(연→월) 안전 변환
    _set_q_floor_monthly(cfg, args)

    # 7) HJB 격자/충격 수 정규화 (+하한)
    _normalize_hjb_w_grid(cfg, _get(args, "hjb_w_grid", getattr(cfg, "hjb_w_grid", None)))
    cfg.hjb_Nshock = max(int(getattr(cfg, "hjb_Nshock", 256) or 256), 256)

    # 8) ETA grid 자동 구성
    auto_eta_grid(cfg, requested_n=_get(args, "hjb_eta_n", None))

    # 선택: 입력 그대로 단순 전달만 필요한 필드가 있다면 여기서 추가로 처리
    # _maybe_set(cfg, args, ["추가필드명", ...])

    # print(f"[cfg] seeds={cfg.seeds}, n_paths_eval={cfg.n_paths_eval}, steps_per_year={cfg.steps_per_year}")
    return cfg
