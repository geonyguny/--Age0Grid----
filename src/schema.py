# -*- coding: utf-8 -*-
from __future__ import annotations

METRICS_SCHEMA_KEYS = [
    "method", "tag",
    "EW", "ES95", "RuinPct",
    "bias_on",
    "window", "hedge", "mix",  # 프로젝트 상황에 맞춰 남길 메타 키
    "seed", "n_paths_eval",
    "es95_n", "es95_source",
    "data_profile", "commit_hash", "timestamp",
]
def normalize_metrics(d: dict) -> dict:
    out = {k: None for k in METRICS_SCHEMA_KEYS}
    out.update({k: v for k, v in d.items() if k in METRICS_SCHEMA_KEYS})
    return out
