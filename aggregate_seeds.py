# -*- coding: utf-8 -*-
"""
seed_results.jsonl (한 줄에 JSON 하나씩, CLI --print_mode summary 출력)을 읽어서
EW, ES95, Ruin, mean_WT의 시드별 평균±표준편차를 계산하고 표로 출력한다.

사용법:
    python aggregate_seeds.py seed_results.jsonl
    (인자를 생략하면 기본으로 ./seed_results.jsonl 을 찾는다)
"""
import sys
import json
import statistics as st
from pathlib import Path

DEFAULT_PATH = "seed_results.jsonl"
METRIC_KEYS = ["EW", "ES95", "Ruin", "mean_WT", "EU", "EU_per_year"]


def load_records(path: Path):
    records = []
    # utf-8-sig: BOM이 있으면 자동으로 제거하고, 없으면 그냥 utf-8처럼 동작한다.
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[경고] {line_no}번째 줄 JSON 파싱 실패, 건너뜀: {e}")
                continue
            records.append(obj)
    return records


def main():
    path = Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH)
    if not path.exists():
        print(f"[오류] 파일을 찾을 수 없습니다: {path}")
        sys.exit(1)

    records = load_records(path)
    if not records:
        print("[오류] 유효한 JSON 레코드가 없습니다.")
        sys.exit(1)

    print(f"총 {len(records)}개 시드 결과 로드 완료: {path}\n")

    # tag에서 시드 라벨 추정(hjb_seed0 등), 없으면 순번 사용
    rows = []
    for i, r in enumerate(records):
        tag = r.get("tag", f"run{i}")
        m = r.get("metrics", {})
        row = {"tag": tag}
        for k in METRIC_KEYS:
            v = m.get(k)
            row[k] = float(v) if v is not None else float("nan")
        rows.append(row)

    # 개별 결과 표
    header = "tag".ljust(18) + "".join(k.rjust(14) for k in METRIC_KEYS)
    print(header)
    print("-" * len(header))
    for row in rows:
        line = row["tag"].ljust(18)
        for k in METRIC_KEYS:
            line += f"{row[k]:14.6f}"
        print(line)

    print("\n=== 집계 (평균 ± 표준편차, n={}) ===".format(len(rows)))
    for k in METRIC_KEYS:
        vals = [row[k] for row in rows if row[k] == row[k]]  # NaN 제거
        if not vals:
            print(f"{k:10s}: 데이터 없음")
            continue
        mean = st.mean(vals)
        std = st.pstdev(vals) if len(vals) > 1 else 0.0
        print(f"{k:10s}: {mean:.6f} ± {std:.6f}  (n={len(vals)})")

    # 변동계수(CV)도 참고로 — 상대적 안정성 판단에 유용
    print("\n=== 변동계수 (CV = std/mean, 낮을수록 안정적) ===")
    for k in METRIC_KEYS:
        vals = [row[k] for row in rows if row[k] == row[k]]
        if not vals:
            continue
        mean = st.mean(vals)
        std = st.pstdev(vals) if len(vals) > 1 else 0.0
        cv = (std / mean) if mean != 0 else float("nan")
        print(f"{k:10s}: CV = {cv:.4f}")


if __name__ == "__main__":
    main()