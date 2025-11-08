# scripts/figs/make_paper_tables.py
import argparse, os
import pandas as pd

def safe_read(p):
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outputs', required=True)
    ap.add_argument('--dest', required=True)
    args = ap.parse_args()

    out = args.outputs
    dest = args.dest

    opt_core  = safe_read(os.path.join(out,'OPT_table_core.csv'))
    bh_core   = safe_read(os.path.join(out,'BH_table_core.csv'))
    compare   = safe_read(os.path.join(out,'OPT_BH_compare.csv'))
    top_both  = safe_read(os.path.join(out,'OPT_BH_top.csv'))
    snap_dev  = safe_read(os.path.join(out,'DEV_metrics_snapshot.csv'))
    opt_sum   = safe_read(os.path.join(out,'OPT_summary_benefit.csv'))
    bh_sum    = safe_read(os.path.join(out,'BH_summary_benefit.csv'))
    best_sex  = safe_read(os.path.join(out,'DEV_OPT_best_by_sex_mort.csv'))

    with pd.ExcelWriter(dest, engine='openpyxl') as xw:
        if not opt_core.empty: opt_core.to_excel(xw, index=False, sheet_name='opt_core')
        if not bh_core.empty:  bh_core.to_excel(xw, index=False, sheet_name='bh_core')
        if not compare.empty:  compare.to_excel(xw, index=False, sheet_name='opt_bh_compare')
        if not top_both.empty: top_both.to_excel(xw, index=False, sheet_name='opt_bh_top')
        if not best_sex.empty: best_sex.to_excel(xw, index=False, sheet_name='best_by_sex_mort')
        if not opt_sum.empty:  opt_sum.to_excel(xw, index=False, sheet_name='opt_summary_raw')
        if not bh_sum.empty:   bh_sum.to_excel(xw, index=False, sheet_name='bh_summary_raw')
        if not snap_dev.empty: snap_dev.to_excel(xw, index=False, sheet_name='metrics_snapshot')

    print(f"[OK] wrote {dest}")

if __name__ == '__main__':
    main()
