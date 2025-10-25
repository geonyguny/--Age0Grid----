# project/env/irp_env.py
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
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    from project.env.retirement_env import RetirementEnv  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError(
        "project.env.retirement_env.RetirementEnv not found. Please ensure it exists."
    ) from e


class IRPEnvAdapter:
    def __init__(
        self,
        f_target: float = 0.0,
        w_max: Optional[float] = None,
        q_floor: Optional[float] = None,
        base_kwargs: Optional[Dict[str, Any]] = None,
         q_cap: Optional[float] = None):  # ★ 추가
    
        self.f_target = float(f_target)
        self.q_cap = float(q_cap) if q_cap is not None else None  # ★ 추가
        kwargs = dict(base_kwargs or {})
        # best-effort: common flags if supported by underlying env
        if w_max is not None:
            kwargs.setdefault("w_max", float(w_max))
        if q_floor is not None:
            kwargs.setdefault("q_floor", float(q_floor))

        self._env = RetirementEnv(**kwargs)
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
        if self.q_cap is not None:
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
            rew = float(res.get("reward", 0.0))
            term = bool(res.get("done", False) or res.get("terminated", False) or res.get("terminal", False))
            trunc = bool(res.get("truncated", False))
            info = dict(res.get("info", {}))
            done = term or trunc
        else:
            try:
                n = len(res)  # may raise if not sized
            except Exception:
                raise TypeError(f"Underlying env.step returned unsupported type: {type(res)!r}")

            if n == 5:
                ob, rew, term, trunc, info = res
                rew = float(rew)
                done = bool(term) or bool(trunc)
                info = dict(info or {})
            elif n == 4:
                ob, rew, done, info = res
                rew = float(rew)
                done = bool(done)
                trunc = False
                info = dict(info or {})
            elif n == 3:
                ob, rew, done = res
                rew = float(rew)
                done = bool(done)
                trunc = False
                info = {}
            else:
                raise TypeError("Underlying env.step must return 3, 4, or 5 elements")

        # 4) terminal info enrichment
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

        self._last_info = dict(info or {})
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
    )


def make_env():
    """Default zero-arg factory used by RLTrainer."""
    return _env_from_envvars()
