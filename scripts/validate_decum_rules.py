# scripts/validate_decum_rules.py
import argparse, os, pandas as pd, numpy as np

RULES = [
    dict(name="lambda_up_EW_ES_down",    need_cols=["lambda_stage","EW","ES95"], group=["method","es_mode"]),
    dict(name="wmax_down_EW_ES_down",    need_cols=["w_max","EW","ES95"],         group=["method","es_mode"]),
    dict(name="fee_up_EW_down",          need_cols=["fee_annual","EW"],           group=["method","es_mode"]),
    dict(name="qfloor_up_Ruin_down_EW_down", need_cols=["q_floor","Ruin","EW"],   group=["method","es_mode"]),
    dict(name="hedge_sigma_up_EW_down",  need_cols=["hedge_sigma_k","EW"],        group=["method","es_mode"]),
]

def monotone_check(df, x, y, expect_sign="neg"):
    d = df[[x,y]].dropna().sort_values(x)
    if len(d) < 2: return None
    # slope of simple regression
    x0 = d[x].to_numpy()
    y0 = d[y].to_numpy()
    slope = np.polyfit(x0, y0, 1)[0]
    if expect_sign=="neg":
        return slope <= 0
    elif expect_sign=="pos":
        return slope >= 0
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default=r"outputs/paper/validation_report.md")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    df = pd.read_csv(args.src)
    lines = ["# Validation Report (7.5 rules)\n"]

    for rule in RULES:
        ok = "SKIP"
        reason = ""
        if not all(c in df.columns for c in rule["need_cols"]):
            reason = f"missing cols: {set(rule['need_cols'])-set(df.columns)}"
        else:
            gcols = [c for c in rule["group"] if c in df.columns]
            ok_list = []
            for _,d in df.groupby(gcols) if gcols else [(None, df)]:
                if rule["name"]=="lambda_up_EW_ES_down":
                    r1 = monotone_check(d, "lambda_stage","EW","neg")
                    r2 = monotone_check(d, "lambda_stage","ES95","neg")
                    ok_list.append((r1 is True) and (r2 is True))
                elif rule["name"]=="wmax_down_EW_ES_down":
                    # reverse: as w_max decreases -> EW,ES decrease -> slope pos wrt w_max
                    r1 = monotone_check(d, "w_max","EW","pos")
                    r2 = monotone_check(d, "w_max","ES95","pos")
                    ok_list.append((r1 is True) and (r2 is True))
                elif rule["name"]=="fee_up_EW_down":
                    ok_list.append(monotone_check(d, "fee_annual","EW","neg") is True)
                elif rule["name"]=="qfloor_up_Ruin_down_EW_down":
                    r1 = monotone_check(d, "q_floor","Ruin","neg")  # q_floor up -> Ruin down
                    r2 = monotone_check(d, "q_floor","EW","neg")    # q_floor up -> EW down
                    ok_list.append((r1 is True) and (r2 is True))
                elif rule["name"]=="hedge_sigma_up_EW_down":
                    ok_list.append(monotone_check(d, "hedge_sigma_k","EW","neg") is True)
            if ok_list:
                ok = "PASS" if all(ok_list) else "FAIL"
        lines.append(f"- **{rule['name']}**: {ok}" + (f"  _{reason}_" if reason else ""))

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[OK] wrote {args.out}")

if __name__=="__main__":
    main()
