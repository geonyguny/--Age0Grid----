import numpy as np, pandas as pd

df = pd.read_csv(r'D:\01_simul\outputs\_summary.csv')
labels = {0:'s0',1:'s1',2:'s2',3:'s3',4:'s4'}
esm_num = pd.to_numeric(df['es_mode'], errors='coerce')
df['es_mode_lab'] = esm_num.round().astype('Int64').map(labels)

mask = (df['es_metric']=='wealth') & (df['window']=='2015-01to')

print("\n[es_mode별 ES95 분포 top 15]")
print(df[mask].groupby(['es_mode_lab','ES95']).size()
        .sort_values(ascending=False).head(15))

print("\n[조합별 ES95 유니크 개수 분포]")
uniq_cnt = (df[mask]
            .groupby(['mix_kr','mix_us','mix_gold','hedge_ratio','es_mode_lab'])['ES95']
            .nunique())
print(uniq_cnt.value_counts().sort_index())
