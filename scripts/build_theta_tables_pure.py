from __future__ import annotations
from pathlib import Path
import sys
import argparse

import numpy as np
import pandas as pd

# -------------------------------------------------------
# project root 세팅 및 모듈 import
# -------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent  # ...\01_simul
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project.env.retirement_env import RetirementEnv


def parse_theta_list(s: str) -> list[float]:
    """쉼표로 구분된 문자열을 float 리스트로 변환."""
    vals: list[float] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    return vals


def build_theta_table_for_case(
    *,
    sex: str,
    age0: int,
    mort_table: str,
    horizon_years: int,
    steps_per_year: int,
    r_f_real_annual: float,
    phi_adval: float,
    ann_index: str,
    theta_list: list[float],
) -> pd.DataFrame:
    """
    주어진 (sex, age0, mort_table, r_f, phi, index)에 대해
    theta_list(ann_alpha grid)별 y_ann, a_factor 테이블 생성.
    """
    rows = []
    for theta in theta_list:
        Cfg = type("Cfg", (object,), dict(
            steps_per_year   = steps_per_year,
            horizon_years    = horizon_years,
            W0               = 1.0,
            # annuity params
            ann_alpha        = float(theta),
            phi_adval        = phi_adval,
            ann_index        = ann_index,
            # mortality (무조건 on + token 사용)
            mort_table       = mort_table,
            sex              = sex,
            age0             = age0,
            mortality        = "on",
            # 기타
            use_real_rf      = "on",
            r_f_real_annual  = r_f_real_annual,
        ))()

        # RetirementEnv 내부에서 반드시 생명표를 로드하고,
        # annuity overlay 초기화(y_ann, ann_a_factor)를 수행
        env = RetirementEnv(Cfg)

        y_ann    = float(getattr(env, "y_ann", np.nan))
        a_factor = float(getattr(env, "ann_a_factor", np.nan))

        rows.append(dict(
            sex             = sex,
            age0            = age0,
            horizon_years   = horizon_years,
            steps_per_year  = steps_per_year,
            mort_table      = mort_table,
            r_f_real_annual = r_f_real_annual,
            phi_adval       = phi_adval,
            ann_index       = ann_index,
            ann_alpha       = float(theta),
            y_ann           = y_ann,
            y_ann_annual    = y_ann * steps_per_year,
            a_factor        = a_factor,
        ))

    df = (
        pd.DataFrame(rows)
          .sort_values("ann_alpha")
          .reset_index(drop=True)
    )
    return df


def merge_all_theta_pure(out_dir: Path) -> None:
    """
    outputs 디렉터리 내 THEORY_*_theta_pure.csv들을 모두 모아서
    THEORY_theta_pure_all.csv로 통합.
    - 개별 파일: THEORY_M55_BASE_theta_pure.csv, THEORY_F55_BASE_theta_pure.csv, ...
    - 통합 파일: THEORY_theta_pure_all.csv
    """
    pattern = "THEORY_*_theta_pure.csv"
    files = list(out_dir.glob(pattern))
    if not files:
        print(f"[WARN] no files matching {pattern} in {out_dir}")
        return

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"[WARN] failed to read {f}: {e}")
            continue

        # 파일명에서 tag 추출 (예: THEORY_M55_BASE_theta_pure.csv → M55_BASE)
        tag_name = f.stem.replace("THEORY_", "").replace("_theta_pure", "")
        df.insert(0, "tag", tag_name)
        dfs.append(df)

    if not dfs:
        print("[WARN] no valid dataframes loaded for merge.")
        return

    all_df = (
        pd.concat(dfs, ignore_index=True)
          .sort_values(["age0", "sex", "ann_alpha"])
          .reset_index(drop=True)
    )
    out_path = out_dir / "THEORY_theta_pure_all.csv"
    all_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("===================================================")
    print(" MERGED PURE ANNUITY THETA TABLE (ALL TAGS)")
    print("---------------------------------------------------")
    # 너무 길어지지 않도록 head(30)만 화면에 표시
    print(all_df.head(30).to_string(index=False, float_format=lambda x: f"{x:0.6f}"))
    print("---------------------------------------------------")
    print("Saved merged CSV =>", out_path)


def main():
    parser = argparse.ArgumentParser(
        description="Build pure annuity theta tables (with mortality, per tag) "
                    "and also a merged CSV across all tags."
    )
    parser.add_argument("--sex", type=str, required=True, help="M or F")
    parser.add_argument("--age0", type=int, required=True, help="retirement age")
    parser.add_argument("--mort_table", type=str, required=True,
                        help="mortality token, e.g. base, cohort_2020")
    parser.add_argument("--horizon_years", type=int, default=15)
    parser.add_argument("--steps_per_year", type=int, default=12)
    parser.add_argument("--r_f_real_annual", type=float, default=0.02)
    parser.add_argument("--phi_adval", type=float, default=0.05)
    parser.add_argument("--ann_index", type=str, default="real")
    parser.add_argument("--theta_list", type=str, required=True,
                        help="comma-separated list, e.g. 0.0,0.1,0.2")
    parser.add_argument("--tag", type=str, required=True,
                        help="tag used in output filename (e.g. M55_BASE)")

    args = parser.parse_args()

    sex  = args.sex.upper()
    age0 = int(args.age0)

    theta_list = parse_theta_list(args.theta_list)

    df = build_theta_table_for_case(
        sex=sex,
        age0=age0,
        mort_table=args.mort_table,
        horizon_years=int(args.horizon_years),
        steps_per_year=int(args.steps_per_year),
        r_f_real_annual=float(args.r_f_real_annual),
        phi_adval=float(args.phi_adval),
        ann_index=str(args.ann_index),
        theta_list=theta_list,
    )

    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"THEORY_{args.tag}_theta_pure.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("===================================================")
    print(f" PURE ANNUITY THETA TABLE  tag={args.tag}")
    print(f" sex={sex}, age0={age0}, horizon={int(args.horizon_years)}y")
    print(
        " mort_table={mt}, r_f_real={rf:.3f}, phi_adval={phi:.3f}, index={idx}".format(
            mt=args.mort_table,
            rf=float(args.r_f_real_annual),
            phi=float(args.phi_adval),
            idx=args.ann_index,
        )
    )
    print("---------------------------------------------------")
    print(df.to_string(index=False, float_format=lambda x: f"{x:0.6f}"))
    print("---------------------------------------------------")
    print("Saved CSV =>", out_path)

    # --- 통합 테이블도 동시에 생성 ---
    merge_all_theta_pure(out_dir)


if __name__ == "__main__":
    main()
