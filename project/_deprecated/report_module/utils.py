# project/report/utils.py
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Tuple

def onoff(v: Any, default: str = "on") -> str:
    s = str(v).strip().lower() if v is not None else default
    if s in ("on", "off"): return s
    if s in ("true","1","y","yes"): return "on"
    if s in ("false","0","n","no"): return "off"
    return default

@dataclass(frozen=True)
class PanelID:
    market_mode: str
    bootstrap_block: int
    data_profile: str
    alpha_mix: Tuple[float, float, float]  # (kr, us, au)
    h_FX: float
    fee_annual: float
    floor_on: str
    es_mode: str

    def to_key(self) -> str:
        payload = {
            "market_mode": self.market_mode,
            "bootstrap_block": self.bootstrap_block,
            "data_profile": self.data_profile,
            "alpha_mix": [round(x, 6) for x in self.alpha_mix],
            "h_FX": round(self.h_FX, 6),
            "fee_annual": round(self.fee_annual, 6),
            "floor_on": self.floor_on,
            "es_mode": self.es_mode,
        }
        s = json.dumps(payload, sort_keys=True)
        h = hashlib.md5(s.encode("utf-8")).hexdigest()[:10]
        return f"{payload['market_mode']}-{payload['bootstrap_block']}-{payload['data_profile']}-" \
               f"{payload['alpha_mix'][0]:.3f},{payload['alpha_mix'][1]:.3f},{payload['alpha_mix'][2]:.3f}-" \
               f"h{payload['h_FX']:.3f}-fee{payload['fee_annual']:.3f}-{payload['floor_on']}-{payload['es_mode']}-{h}"

def make_panel_id(meta: Dict[str, Any], metrics: Dict[str, Any]) -> str:
    mm = str(meta.get("market",{}).get("mode") or metrics.get("market_mode") or "iid")
    bb = int(meta.get("market",{}).get("bootstrap_block") or metrics.get("bootstrap_block") or 24)
    dp = str(meta.get("market",{}).get("data_profile") or metrics.get("data_profile") or "")
    am = metrics.get("alpha_mix_used") or meta.get("alpha_mix_used") or (1/3,1/3,1/3)
    if isinstance(am, (list,tuple)) and len(am)==3:
        am = (float(am[0]), float(am[1]), float(am[2]))
    else:
        am = (1/3,1/3,1/3)
    hfx = float(metrics.get("h_FX_used") or meta.get("h_FX_used") or 0.0)
    fee = float(metrics.get("fx_hedge_cost_annual") or 0.0)  # this is hedge cost; try fee_annual too
    fee_annual = float(meta.get("fee_annual") or metrics.get("fee_annual") or 0.0)
    floor = "on" if str(meta.get("use_floor") or metrics.get("floor_on") or "on").lower() in ("on","true","1","yes","y") else "off"
    es_mode = str(metrics.get("es_mode") or meta.get("es_mode") or "wealth")
    pid = PanelID(mm, bb, dp, am, hfx, fee_annual, floor, es_mode)
    return pid.to_key()
