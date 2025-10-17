import hashlib, json

def config_hash(cfg: dict) -> str:
    try:
        s = json.dumps(cfg, sort_keys=True, default=str, ensure_ascii=False)
    except Exception:
        s = str(cfg)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]
