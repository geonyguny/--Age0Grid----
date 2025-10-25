# project/utils/metrics_io.py
from __future__ import annotations
from pathlib import Path
import json, csv, hashlib, time, subprocess
from typing import Dict, Any

def safe_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding=encoding)
    tmp.replace(path)

def safe_write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    tmp.replace(path)

def save_metrics(out_dir: Path, metrics: Dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # JSON
    json_path = out_dir / "metrics.json"
    safe_write_text(json_path, json.dumps(metrics, indent=2))
    # CSV (단일행)
    csv_path = out_dir / "metrics.csv"
    safe_write_csv(csv_path, [metrics], list(metrics.keys()))

def compute_code_hash() -> str:
    """git 해시(있는 경우) 또는 파일셋 해시 근사값."""
    try:
        rev = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
        return rev
    except Exception:
        return "nogit-" + str(int(time.time()))

def write_manifest(out_dir: Path, args: Any, extras: Dict[str, Any] | None = None) -> None:
    manifest = {
        "tag": getattr(args, "tag", None),
        "outputs": str(getattr(args, "outputs", "")),
        "args": vars(args) if hasattr(args, "__dict__") else None,
        "code_hash": compute_code_hash(),
        "timestamp": int(time.time()),
        "version": "v2",
    }
    if extras:
        manifest.update(extras)
    safe_write_text(Path(out_dir) / "manifest.json", json.dumps(manifest, indent=2))
