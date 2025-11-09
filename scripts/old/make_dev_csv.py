# scripts/make_dev_csv.py
import os, csv, math, random
random.seed(7)

PATH = os.path.abspath("project/data/market/kr_us_gold_bootstrap_mini.csv")
os.makedirs(os.path.dirname(PATH), exist_ok=True)

# 아주 작은 월별 시계열(60개월)
N = 60
rows = [("date","ret_kr_eq","ret_us_eq_krw","ret_gold_krw","rf_real","rf_nom","cpi","ret_fx_usdkrw")]
for t in range(N):
    # date: YYYY-MM
    y = 2000 + (t//12)
    m = 1 + (t%12)
    date = f"{y:04d}-{m:02d}"
    # 간단한 난수(평균 0.005, 표준편차 0.04 정도인 수익률)
    def rr(mu=0.005, sigma=0.04): return mu + random.gauss(0.0, sigma)
    rows.append((date, rr(), rr(), rr(0.003,0.03), 0.001, 0.002, 100.0+t, rr(0.0,0.02)))

with open(PATH, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerows(rows)

print(PATH)
