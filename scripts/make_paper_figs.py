# scripts/make_paper_figs.py
import os, sys, math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────
# 1) OutRoot 해석: placeholder나 잘못된 경로면 최신 night_*로 대체
# ─────────────────────────────────────────────────────────
def resolve_outroot(arg_path: str) -> str:
    base = r'.\outputs'
    p = arg_path if arg_path else r'.\outputs\night_latest'

    # 유효 디렉터리면 그대로
    if os.path.isdir(p):
        return p

    # placeholder(YYYY 포함)거나 존재하지 않으면 최신 night_* 찾기
    parent = os.path.dirname(p) if os.path.dirname(p) else base
    if not os.path.isdir(parent):
        parent = base

    cand = []
    if os.path.isdir(parent):
        for name in os.listdir(parent):
            full = os.path.join(parent, name)
            if os.path.isdir(full) and name.startswith('night_'):
                cand.append((os.path.getmtime(full), full))
    if not cand:
        raise FileNotFoundError(f"no night_* folder under {parent}")

    cand.sort(reverse=True)
    return cand[0][1]

OR = resolve_outroot(sys.argv[1] if len(sys.argv) > 1 else r'.\outputs\night_latest')
rep = os.path.join(OR, 'night_summary_report.csv')
cln = os.path.join(OR, 'night_summary_clean.csv')

if not os.path.exists(rep):
    raise FileNotFoundError(f"not found: {rep}\n(run night_run.ps1 with -DoSummary)")
if not os.path.exists(cln):
    raise FileNotFoundError(f"not found: {cln}\n(run night_run.ps1 with -DoSummary)")

# ─────────────────────────────────────────────────────────
# 2) 데이터 로드 & 컬럼 정규화
# ─────────────────────────────────────────────────────────
df_rep = pd.read_csv(rep)
df_cln = pd.read_csv(cln)

# 숫자형
for c in ['EW_avg','ES95_avg','Ruin_avg','WT_avg']:
    if c in df_rep.columns: df_rep[c] = pd.to_numeric(df_rep[c], errors='coerce')
for c in ['EW','ES95','RuinPct','mean_WT']:
    if c in df_cln.columns: df_cln[c] = pd.to_numeric(df_cln[c], errors='coerce')

# 결측 컬럼 보정
for col, default in [('method',''), ('baseline',''), ('w_fixed','NA')]:
    if col not in df_rep.columns:
        df_rep[col] = default

# baseline 공백/NaN → '(unknown)'으로 라벨링
df_rep['baseline'] = df_rep['baseline'].fillna('')
df_rep.loc[df_rep['baseline'].str.strip()=='', 'baseline'] = '(unknown)'

# w_fixed 순서 & 변환
w_order = ['w_0','w_0_3','w_0_5','w_0_7','w_1','NA']
if 'w_fixed' in df_rep.columns:
    df_rep['w_fixed'] = df_rep['w_fixed'].astype(str)
    df_rep['w_fixed'] = pd.Categorical(df_rep['w_fixed'], categories=w_order, ordered=True)

def w_to_float(w):
    # 'w_0_3' -> 0.3, 'w_1'->1.0, 'NA'->np.nan
    if not isinstance(w, str):
        return np.nan
    if not w.startswith('w_'): 
        return np.nan
    body = w[2:].replace('_','.')
    try:
        return float(body)
    except:
        return np.nan

df_rep['w_float'] = df_rep['w_fixed'].astype(str).map(w_to_float)

# 정렬
df_rep = df_rep.sort_values(['method','baseline','w_fixed'])

# HJB 레퍼런스
hjb_row = df_rep.loc[df_rep['method']=='hjb'].head(1)
hjb_EW = float(hjb_row['EW_avg'].iloc[0]) if len(hjb_row) and not pd.isna(hjb_row['EW_avg'].iloc[0]) else np.nan
hjb_ES = float(hjb_row['ES95_avg'].iloc[0]) if len(hjb_row) and not pd.isna(hjb_row['ES95_avg'].iloc[0]) else np.nan

# 실제로 존재하는 baseline만 사용
baselines_present = [b for b in df_rep['baseline'].dropna().unique().tolist() if b != '']
# 기대 리스트가 전혀 없으면 한 줄(필터 없이)만 그리도록 플래그
has_any_baseline = len(baselines_present) > 0

# ─────────────────────────────────────────────────────────
# 3) 그래프들
# ─────────────────────────────────────────────────────────

# 3-1) EW vs w_fixed
fig1 = plt.figure(figsize=(8,5))
plotted = False
if has_any_baseline:
    for bsl in baselines_present:
        d = df_rep.query("method=='rule' and baseline==@bsl and w_fixed!='NA'").copy()
        d = d.dropna(subset=['EW_avg','w_float'])
        if len(d)==0: continue
        d = d.sort_values('w_float')
        plt.plot(d['w_float'], d['EW_avg'], marker='o', label=bsl)
        plotted = True
else:
    d = df_rep.query("method=='rule' and w_fixed!='NA'").dropna(subset=['EW_avg','w_float']).sort_values('w_float')
    if len(d)>0:
        plt.plot(d['w_float'], d['EW_avg'], marker='o', label='rule')
        plotted = True

if not math.isnan(hjb_EW):
    plt.axhline(hjb_EW, linestyle='--', label='HJB', alpha=0.8); plotted = True

plt.xlabel('w_fixed'); plt.ylabel('EW (avg)'); plt.title('EW vs w_fixed')
if plotted: plt.legend()
plt.tight_layout()
fig1_path = os.path.join(OR,'fig_EW_vs_w_fixed.png'); plt.savefig(fig1_path, dpi=300); plt.close()

# 3-2) ES95 vs w_fixed
fig2 = plt.figure(figsize=(8,5))
plotted = False
if has_any_baseline:
    for bsl in baselines_present:
        d = df_rep.query("method=='rule' and baseline==@bsl and w_fixed!='NA'").copy()
        d = d.dropna(subset=['ES95_avg','w_float'])
        if len(d)==0: continue
        d = d.sort_values('w_float')
        plt.plot(d['w_float'], d['ES95_avg'], marker='o', label=bsl)
        plotted = True
else:
    d = df_rep.query("method=='rule' and w_fixed!='NA'").dropna(subset=['ES95_avg','w_float']).sort_values('w_float')
    if len(d)>0:
        plt.plot(d['w_float'], d['ES95_avg'], marker='o', label='rule'); plotted = True

if not math.isnan(hjb_ES):
    plt.axhline(hjb_ES, linestyle='--', label='HJB', alpha=0.8); plotted = True

plt.xlabel('w_fixed'); plt.ylabel('ES95 (avg, lower is better)'); plt.title('ES95 vs w_fixed')
if plotted: plt.legend()
plt.tight_layout()
fig2_path = os.path.join(OR,'fig_ES95_vs_w_fixed.png'); plt.savefig(fig2_path, dpi=300); plt.close()

# 3-3) Risk–Return (EW vs ES95)
fig3 = plt.figure(figsize=(7,6))
plotted = False
if has_any_baseline:
    for bsl in baselines_present:
        d = df_rep.query("method=='rule' and baseline==@bsl and w_fixed!='NA'").copy()
        d = d.dropna(subset=['ES95_avg','EW_avg'])
        if len(d)==0: continue
        plt.scatter(d['ES95_avg'], d['EW_avg'], label=bsl)
        for _, r in d.iterrows():
            lbl = str(r['w_fixed']).replace('w_','').replace('_','.')
            plt.annotate(lbl, (r['ES95_avg'], r['EW_avg']), fontsize=7, alpha=0.7)
        plotted = True
else:
    d = df_rep.query("method=='rule' and w_fixed!='NA'").dropna(subset=['ES95_avg','EW_avg'])
    if len(d)>0:
        plt.scatter(d['ES95_avg'], d['EW_avg'], label='rule')
        for _, r in d.iterrows():
            lbl = str(r['w_fixed']).replace('w_','').replace('_','.')
            plt.annotate(lbl, (r['ES95_avg'], r['EW_avg']), fontsize=7, alpha=0.7)
        plotted = True

if not math.isnan(hjb_EW) and not math.isnan(hjb_ES):
    plt.scatter([hjb_ES],[hjb_EW], marker='*', s=120, label='HJB'); plotted = True

plt.xlabel('ES95 (lower better)'); plt.ylabel('EW (higher better)'); plt.title('Risk–Return')
if plotted: plt.legend()
plt.tight_layout()
fig3_path = os.path.join(OR,'fig_risk_return.png'); plt.savefig(fig3_path, dpi=300); plt.close()

# 3-4) Ruin bar (baseline × w_fixed)
fig4_path = None
d_rule = df_rep.query("method=='rule' and w_fixed!='NA'").copy()
if len(d_rule)>0 and 'Ruin_avg' in d_rule.columns:
    pivot = d_rule.pivot_table(index='w_fixed', columns='baseline', values='Ruin_avg', aggfunc='first')
    # 원하는 순서 중 실제로 존재하는 것만 재정렬
    exist = [w for w in w_order if w in pivot.index]
    if len(exist)>0:
        pivot = pivot.reindex(index=exist)
        ax = pivot.plot(kind='bar', figsize=(8,4))
        ax.set_ylabel('Ruin Probability'); ax.set_title('Ruin by w_fixed and baseline')
        plt.tight_layout()
        fig4_path = os.path.join(OR,'fig_ruin_bar.png'); plt.savefig(fig4_path, dpi=300); plt.close()
    else:
        plt.close()
else:
    plt.close()

# LaTeX 테이블(그대로)
latex_path = os.path.join(OR, 'table_summary.tex')
try:
    df_rep.to_latex(latex_path, index=False, float_format="%.6f")
except Exception as e:
    # LaTeX 설치가 없거나 인코딩 문제여도 넘어가게
    with open(latex_path,'w',encoding='utf-8') as f:
        f.write(df_rep.to_string(index=False))

print("OutRoot:", OR)
print("Saved:")
print(" -", fig1_path)
print(" -", fig2_path)
print(" -", fig3_path)
print(" -", fig4_path if fig4_path else "(no ruin fig)")
print(" -", latex_path)
