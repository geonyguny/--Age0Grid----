# -*- coding: utf-8 -*-
"""
Thin adapter for RL trainer expecting a simple Env factory `make_env()` with:
  - reset(seed: int | None) -> np.ndarray
  - step({"q": float, "w": float}) -> (obs, reward, done, info)

This wraps the existing RetirementEnv in project.env.retirement_env and guarantees:
  - info contains "L_term" at episode end (best-effort)
  - observation dtype is float32 np.ndarray
  - underlying env 3/4/5-tuple or dict returns are normalized to a 4-tuple

NOTE
- Minimal defaults. For richer wiring, either set ENV_* (below) or provide your own factory in runner.
ENV OVERRIDES (optional)
- ENV_IRP_F_TARGET: float, terminal shortfall target (L_term=max(F_target-W_T,0))
- ENV_IRP_W_MAX:    float, risky-asset cap (if RetirementEnv supports it)
- ENV_IRP_Q_FLOOR:  float, consumption floor (if RetirementEnv supports it)
- ENV_IRP_SEED:     int, default seed used when reset() is called with seed=None
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np

try:
    from project.env.retirement_env import RetirementEnv  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError(
        "project.env.retirement_env.RetirementEnv not found. Please ensure it exists."
    ) from e


# -----------------------
# CRRA 효용 헬퍼 함수
# -----------------------
def _crra_u(c: float, gamma: float) -> float:
    """
    단순 CRRA 효용 u(c) = (c^(1-gamma) - 1) / (1-gamma), gamma != 1
    - gamma = 1 에 근접하면 log(c)로 처리
    - c <= 0 인 경우 수치안정을 위해 매우 작은 양수로 클리핑
    """
    c_eff = float(max(c, 1e-12))
    g = float(gamma)
    if abs(g - 1.0) < 1e-9:
        return float(np.log(c_eff))
    return float((c_eff ** (1.0 - g) - 1.0) / (1.0 - g))


class IRPEnvAdapter:
    def __init__(
        self,
        f_target: float = 0.0,
        w_max: Optional[float] = None,
        q_floor: Optional[float] = None,
        base_kwargs: Optional[Dict[str, Any]] = None,
        q_cap: Optional[float] = None,
        cfg: Any = None,  # RetirementEnv 구성에 사용되는 cfg
    ):
        self.f_target = float(f_target)
        self.q_cap = float(q_cap) if q_cap is not None else None
        self._cfg = cfg

        kwargs = dict(base_kwargs or {})

        # CRRA 효용 관련 파라미터 (run._build_env_factory_from_args 에서 채워줌)
        self.crra_gamma: float = float(kwargs.get("crra_gamma", 3.0) or 3.0)
        self.u_scale: float = float(kwargs.get("u_scale", 0.05) or 0.05)

        # best-effort: common flags if supported by underlying env
        if w_max is not None:
            kwargs.setdefault("w_max", float(w_max))
        if q_floor is not None:
            kwargs.setdefault("q_floor", float(q_floor))

        # cfg 주입 경로 보장
        if self._cfg is not None:
            self._env = RetirementEnv(self._cfg, **kwargs)
        else:
            self._env = RetirementEnv(**kwargs)

        print(
            "[ANN-DBG-IRP] IRPEnvAdapter created underlying env:",
            type(self._env).__name__,
            "with kwargs keys=",
            list(kwargs.keys()),
        )

        self._last_info: Dict[str, Any] | None = None

    # ---------- Helpers ----------
    @staticmethod
    def _as_obs(ob: Any) -> np.ndarray:
        """Ensure observation is np.ndarray(float32). Accept dict or array-like."""
        if isinstance(ob, dict):
            # Try to form at least [t_norm, W]
            try:
                t = float(ob.get("t", ob.get("t_norm", 0.0)))
                T = float(ob.get("T", 1.0))
                t_norm = t / max(1.0, (T - 1.0)) if T else 0.0
            except Exception:
                t_norm = 0.0
            try:
                W = float(ob.get("W", ob.get("wealth", 0.0)))
            except Exception:
                W = 0.0
            arr = np.asarray([t_norm, W], dtype=np.float32)
        else:
            arr = np.asarray(ob, dtype=np.float32).ravel()
        return arr

    # ---------- Standard API ----------
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        # If no seed provided, optionally fall back to ENV_IRP_SEED
        if seed is None:
            _seed_env = os.getenv("ENV_IRP_SEED")
            if _seed_env is not None:
                try:
                    seed = int(_seed_env)
                except Exception:
                    seed = None
        if hasattr(self._env, "reset"):
            ob = self._env.reset(seed=seed)
        else:  # pragma: no cover
            ob = self._env.reset()
        return self._as_obs(ob)

    def _call_step_underlying(self, q: float, w: float):
        """Try step with keyword → list → tuple; return the first non-None result."""
        # 1) keyword
        try:
            res = self._env.step(q=q, w=w)
            if res is not None:
                return res
        except TypeError:
            pass
        # 2) list
        try:
            res = self._env.step([q, w])
            if res is not None:
                return res
        except TypeError:
            pass
        # 3) tuple
        try:
            res = self._env.step((q, w))
            if res is not None:
                return res
        except TypeError:
            pass
        return None

    def step(self, action: Dict[str, float]):
        # 1) parse action
        q = float(action.get("q", 0.0))
        w = float(action.get("w", 0.0))

        # ★ 소비 상한 캡(월간). 예: 0.01 → 월 1%
        # [FIX 2026-07] 기존 코드는 `self.q_cap is not None`만 검사했는데,
        # --rl_q_cap의 CLI 기본값이 0.0("제한 없음"을 의도)이라 self.q_cap이
        # 0.0(≠None)으로 설정되면 `q = min(q, 0.0)`이 되어 소비율이 매 스텝
        # 무조건 0으로 강제되고 있었다. 이로 인해 RL 정책이 무엇을 출력하든
        # 소비가 항상 0으로 깎여 (a) 보상이 클립 하한(-100)에 영구히 고정되고
        # (b) 학습이 epoch 수와 무관하게 전혀 진행되지 않는 문제가 있었다.
        # 0 이하 값은 "제한 없음"으로 해석하도록 수정.
        if self.q_cap is not None and self.q_cap > 0.0:
            q = min(q, self.q_cap)

        # 2) call underlying env robustly
        res = self._call_step_underlying(q, w)
        if res is None:
            raise TypeError(
                "Underlying env.step returned None or could not be called. "
                "Make sure RetirementEnv.step returns (obs, reward, done[, trunc], info)."
            )

        # 3) normalize return shape to (obs, reward, done, info)
        trunc = False  # default if not provided
        if isinstance(res, dict):
            ob = res.get("obs") or res.get("observation")
            _rew_raw = float(res.get("reward", 0.0))
            term = bool(
                res.get("done", False)
                or res.get("terminated", False)
                or res.get("terminal", False)
            )
            trunc = bool(res.get("truncated", False))
            info = dict(res.get("info", {}))
            done = term or trunc
        else:
            try:
                n = len(res)  # may raise if not sized
            except Exception:
                raise TypeError(
                    f"Underlying env.step returned unsupported type: {type(res)!r}"
                )

            if n == 5:
                ob, _rew_raw, term, trunc, info = res
                done = bool(term) or bool(trunc)
                info = dict(info or {})
            elif n == 4:
                ob, _rew_raw, done, info = res
                done = bool(done)
                trunc = False
                info = dict(info or {})
            elif n == 3:
                ob, _rew_raw, done = res
                done = bool(done)
                trunc = False
                info = {}
            else:
                raise TypeError("Underlying env.step must return 3, 4, or 5 elements")

        # 4) terminal info enrichment (W_T, L_term 등)
        if done:
            # Prefer info.W_T, else try env.W
            W_T = info.get("W_T")
            if W_T is None:
                try:
                    W_T = float(getattr(self._env, "W", 0.0))
                except Exception:
                    W_T = 0.0
            else:
                try:
                    W_T = float(W_T)
                except Exception:
                    W_T = 0.0
            info.setdefault("W_T", W_T)
            info.setdefault("terminal_wealth", W_T)

            # L_term fill (if f_target available)
            if "L_term" not in info:
                F = None
                if getattr(self, "f_target", None) is not None:
                    try:
                        F = float(self.f_target)
                    except Exception:
                        F = None
                if F is None and hasattr(self._env, "F_target"):
                    try:
                        F = float(getattr(self._env, "F_target"))
                    except Exception:
                        F = None
                if F is not None:
                    try:
                        info["L_term"] = float(max(F - W_T, 0.0))
                    except Exception:
                        pass

        # 5) ★★★ RL 보상 재정의: 소비에 대한 CRRA 효용으로 계산 ★★★
        #    - underlying env 가 제공하는 reward는 무시하고, info의 소비금액 기반으로 reward 산출
        c_t = None
        if isinstance(info, dict):
            c_t = info.get("consumption") or info.get("c_t") or info.get("C")
        try:
            c_val = float(c_t) if c_t is not None else 0.0
        except Exception:
            c_val = 0.0

        # CRRA 효용 계산 + 스케일
        u_t = _crra_u(c_val, self.crra_gamma)
        rew = float(self.u_scale * u_t)

        # [FIX 2026-07] 위에서 재계산한 reward는 RetirementEnv.step()이 원래
        # 적용하던 ±100 클립을 그대로 우회하고 있었다. _crra_u는 소비가 0에
        # 가까워지면(c_eff 하한 1e-12) u(c)가 -1e23 스케일까지 발산하는데,
        # 그 값이 아무 제약 없이 그대로 RL 학습 보상으로 들어가고 있었다.
        # 실제로 초기/미학습 정책이 q≈0을 출력할 때마다 reward가 약 -2.5e22
        # 수준으로 폭주하여 critic loss가 처음부터 inf가 되고, 이후 몇 백
        # epoch을 더 돌려도 학습이 전혀 진행되지 않는 현상(60/200 epoch 결과
        # 완전 동일)의 직접적인 원인이었다. RetirementEnv.step()과 동일한
        # 스케일(±100)로 클립하여 학습 안정성을 회복한다.
        rew = float(np.clip(rew, -100.0, 100.0))

        # (선택적) 파산 경로에 소규모 페널티 부여
        if done:
            W_T_val = info.get("W_T") or info.get("terminal_wealth") or 0.0
            try:
                W_T_val = float(W_T_val)
            except Exception:
                W_T_val = 0.0
            if W_T_val <= 0.0:
                # 지나치게 큰 페널티는 RL 안정성을 해칠 수 있으므로 아주 작은 값 수준으로 유지
                rew += -1.0

        self._last_info = dict(info or {})
        # RLTrainer 쪽은 (obs, reward, done, info) 4-튜플을 허용하므로 기존 형태 유지
        return self._as_obs(ob), float(rew), bool(done), info

    # Optional accessor
    def get_last_info(self) -> Dict[str, Any] | None:
        return self._last_info


# ------------- Factory -------------

def _env_from_envvars() -> IRPEnvAdapter:
    def _get_env(name: str, cast, default):
        v = os.getenv(name)
        if v is None:
            return default
        try:
            return cast(v)
        except Exception:
            return default

    f_target = _get_env("ENV_IRP_F_TARGET", float, 0.0)
    w_max = _get_env("ENV_IRP_W_MAX", float, None)
    q_floor = _get_env("ENV_IRP_Q_FLOOR", float, None)
    # ENV_IRP_SEED는 어댑터 인자가 아니라 reset()에서 자동 사용합니다.

    # Map additional keys only into RetirementEnv via base_kwargs as needed.
    base_kwargs: Dict[str, Any] = {}
    # Example:
    # hz = _get_env("ENV_IRP_HORIZON_YEARS", int, None)
    # if hz is not None:
    #     base_kwargs["horizon_years"] = hz

    return IRPEnvAdapter(
        f_target=f_target,
        w_max=w_max,
        q_floor=q_floor,
        base_kwargs=base_kwargs,
        cfg=None,
    )


def make_env():
    """Default zero-arg factory used by RLTrainer."""
    return _env_from_envvars()