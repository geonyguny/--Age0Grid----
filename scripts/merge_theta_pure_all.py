from __future__ import annotations
from pathlib import Path
import pandas as pd

ROOT = Path(".")
out_dir = ROOT / "outputs"

# *_BASE_theta_pure.csv 만 대상으로 (테스트용 M55_theta_pure는 제외)
files = sorted(out_dir.glob("THEORY_*_BASE_theta_pure.csv"))

if not files:
    print("[ERR] No THEORY_*_BASE_theta_pure.csv files found in outputs/")
    raise SystemExit(1)

rows = []
for f in files:
    df = pd.read_csv(f)
    # 파일명에서 tag 추출: THEORY_M55_BASE_theta_pure -> M55_BASE
    tag = f.stem.replace("THEORY_", "").replace("_theta_pure", "")
    if "tag" not in df.columns:
        df.insert(0, "tag", tag)
    else:
        df["tag"] = df["tag"].fillna(tag)
    rows.append(df)

all_df = pd.concat(rows, ignore_index=True)

out_path = out_dir / "THEORY_theta_pure_all.csv"
all_df.to_csv(out_path, index=False, encoding="utf-8-sig")

print("[OK] merged {} files -> {} rows".format(len(files), len(all_df)))
print("[OK] saved:", out_path)
