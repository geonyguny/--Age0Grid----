# scripts/check_snapshot_schema.py
import sys, pandas as pd

REQ = [
    "tag","method","es_mode","seed",
    "mix_us","hedge_sigma_k",
    "EW","ES95","CompositeScore"
]

def main():
    if len(sys.argv) < 2:
        print("[ERR] usage: python check_snapshot_schema.py <snapshot_csv>")
        sys.exit(2)

    src = sys.argv[1]
    try:
        df = pd.read_csv(src)
    except Exception as e:
        print(f"[ERR] failed to read '{src}': {e}")
        sys.exit(2)

    missing = [c for c in REQ if c not in df.columns]
    if missing:
        print("[ERR] missing columns:", missing)
        # 힌트: ensure_var_column() 결과를 스냅샷에 포함하도록 score_snapshot 파이프라인 점검
        if "tag" in df.columns:
            sample = df["tag"].astype(str).head(3).tolist()
            print("[HINT] sample tags:", sample)
        sys.exit(1)

    # 경고: 타입 체크(예: 수치형 컬럼)
    numeric_cols = ["mix_us","hedge_sigma_k","EW","ES95","CompositeScore"]
    bad = []
    for c in numeric_cols:
        if not pd.api.types.is_numeric_dtype(df[c]):
            bad.append(c)
    if bad:
        print("[WARN] non-numeric columns detected:", bad)

    print("[OK] all required columns present.")

if __name__ == "__main__":
    main()
