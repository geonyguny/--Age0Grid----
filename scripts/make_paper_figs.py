# tools/make_paper_figs.py
import os, sys, math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

OR = sys.argv[1] if len(sys.argv) > 1 else r'.\outputs\night_latest'
rep = os.path.join(OR, 'night_summary_report.csv')
cln = os.path.join(OR, 'night_summary_clean.csv')

df_rep = pd.read_csv(rep)
df_cln = pd.read_csv(cln)

# 숫자형 처리
for c in ['EW_avg','ES95_avg','Ruin_avg','WT_avg']:
    if c in df_rep.columns: df_rep[c] = pd.to_numeric(df_rep[c], errors='coerce')
for c in ['EW','ES95','RuinPct','mean_WT']:
    if c in df_cln.columns: df_cln[c] = pd.to_numeric(df_cln[c], errors='coerce')

# 정렬 및 범주형 순서
w_order = ['w_0','w_0_3','w_0_5','w_0_7','w_1','NA']
df_rep['w_fixed'] = pd.Categorical(df_rep['w_fixed'], categories=w_order, ordered=True)
df_rep = df_rep.sort_values(['method','baseline','w_fixed'])

# HJB 레퍼런스 값
hjb_row = df_rep.query("method=='hjb'").head(1)
hjb_EW = float(hjb_row['EW_avg']) if len(hjb_row) else np.nan
hjb_ES = float(hjb_row['ES95_avg']) if len(hjb_row) else np.nan

# 1) 성능 곡선 (EW vs w_fixed)
fig1 = plt.figure(figsize=(6,4))
for bsl in ['4pct','cpb','vpw','kgr']:
    d = df_rep.query("method=='rule' and baseline==@bsl and w_fixed!='NA'")
    if len(d)==0: continue
    x = d['w_fixed'].astype(str).str.replace('w_','').str.replace('_','.')
    y = d['EW_avg'].values
    plt.plot(x, y, marker='o', label=bsl)
if not math.isnan(hjb_EW):
    plt.axhline(hjb_EW, linestyle='--', label='HJB', alpha=0.7)
plt.xlabel('w_fixed'); plt.ylabel('EW (avg)'); plt.title('EW vs w_fixed')
plt.legend(); plt.tight_layout()
fig1_path = os.path.join(OR,'fig_EW_vs_w_fixed.png'); plt.savefig(fig1_path, dpi=300); plt.close()

# 2) 리스크 곡선 (ES95 vs w_fixed)
fig2 = plt.figure(figsize=(6,4))
for bsl in ['4pct','cpb','vpw','kgr']:
    d = df_rep.query("method=='rule' and baseline==@bsl and w_fixed!='NA'")
    if len(d)==0: continue
    x = d['w_fixed'].astype(str).str.replace('w_','').str.replace('_','.')
    y = d['ES95_avg'].values
    plt.plot(x, y, marker='o', label=bsl)
if not math.isnan(hjb_ES):
    plt.axhline(hjb_ES, linestyle='--', label='HJB', alpha=0.7)
plt.xlabel('w_fixed'); plt.ylabel('ES95 (avg, lower is better)'); plt.title('ES95 vs w_fixed')
plt.legend(); plt.tight_layout()
fig2_path = os.path.join(OR,'fig_ES95_vs_w_fixed.png'); plt.savefig(fig2_path, dpi=300); plt.close()

# 3) 리스크-리턴 산점도 (EW vs ES95)
fig3 = plt.figure(figsize=(6,5))
for bsl in ['4pct','cpb','vpw','kgr']:
    d = df_rep.query("method=='rule' and baseline==@bsl and w_fixed!='NA'")
    if len(d)==0: continue
    plt.scatter(d['ES95_avg'], d['EW_avg'], label=bsl)
    for _, r in d.iterrows():
        plt.annotate(str(r['w_fixed']).replace('w_','').replace('_','.'),(r['ES95_avg'], r['EW_avg']), fontsize=7, alpha=0.7)
if not math.isnan(hjb_EW) and not math.isnan(hjb_ES):
    plt.scatter([hjb_ES],[hjb_EW], marker='*', s=120, label='HJB')
plt.xlabel('ES95 (lower better)'); plt.ylabel('EW (higher better)'); plt.title('Risk–Return')
plt.legend(); plt.tight_layout()
fig3_path = os.path.join(OR,'fig_risk_return.png'); plt.savefig(fig3_path, dpi=300); plt.close()

# 4) 파산확률 막대 (Ruin_avg)
fig4 = plt.figure(figsize=(7,4))
d = df_rep.query("method=='rule' and w_fixed!='NA'")
if len(d)>0:
    pivot = d.pivot_table(index='w_fixed', columns='baseline', values='Ruin_avg', aggfunc='first').loc[['w_0','w_0_3','w_0_5','w_0_7','w_1']]
    pivot.plot(kind='bar')
    if not math.isnan(hjb_row['Ruin_avg'].values[0] if len(hjb_row) else np.nan):
        pass # 보통 HJB가 0이면 시각적 의미 적어 생략
    plt.ylabel('Ruin Probability'); plt.title('Ruin by w_fixed and baseline')
    plt.tight_layout()
    fig4_path = os.path.join(OR,'fig_ruin_bar.png'); plt.savefig(fig4_path, dpi=300); plt.close()
else:
    plt.close(fig4)

# (선택) 논문용 LaTeX 테이블 내보내기
latex_path = os.path.join(OR, 'table_summary.tex')
with open(latex_path,'w',encoding='utf-8') as f:
    f.write(df_rep.to_latex(index=False, float_format="%.6f"))

print("Saved:", fig1_path, fig2_path, fig3_path, fig4_path if len(d)>0 else "(no ruin fig)", latex_path)
