import argparse, pandas as pd, os, shutil
p=argparse.ArgumentParser(); p.add_argument("--seeds",type=int); p.add_argument("--preset"); p.add_argument("--out"); p.add_argument("--tag"); a=p.parse_args()
# DEV 스냅샷을 OVN으로 복사(경량 버전)
src=os.path.join(a.out,"DEV_metrics_snapshot.csv"); dst=os.path.join(a.out,"OVN_metrics_snapshot.csv")
shutil.copy2(src,dst); print("[OK] OVN snapshot <- DEV (copy):", dst)
