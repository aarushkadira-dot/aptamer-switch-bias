import pandas as pd, numpy as np
from scipy import stats
F="/sessions/dazzling-sweet-pascal/mnt/Figure 2 and 3/GSE149225_toehold_processed_datafile.csv"
d=pd.read_csv(F,low_memory=False); d=d.dropna(subset=['ON_OFF']); d=d[d['QC_ON_OFF']>=3].copy()
d['V']=d['off_id'].astype(str).str[20:50]          # variable trigger-complementary region, 30 nt
for b in 'ACGT': d['n'+b]=d['V'].str.count(b)
d['GC']=d['nG']+d['nC']
print(f"n={len(d)}  variable region = switch nt 21-50 (30 nt)")
print(f"library mean composition:  A {d.nA.mean()/30:.3f}  C {d.nC.mean()/30:.3f}  G {d.nG.mean()/30:.3f}  T {d.nT.mean()/30:.3f}")

def inversion(col,label,minn=200):
    g=d.groupby(col).agg(cov=('ON_OFF','size'),perf=('ON_OFF','mean'))
    g=g[g['cov']>=minn]
    r=stats.spearmanr(g['cov'],g['perf'])
    print(f"\n=== {label} ===")
    print(f"  {'bin':>5} {'coverage':>9} {'%lib':>7} {'mean ON_OFF':>12}")
    for i,row in g.iterrows():
        print(f"  {i:>5} {int(row['cov']):>9} {100*row['cov']/len(d):>6.2f}% {row['perf']:>12.4f}")
    print(f"  Spearman(coverage, performance) = {r.statistic:+.3f}   p={r.pvalue:.4g}   ({len(g)} bins)")
    return g,r

inversion('GC','GC content of variable region')
inversion('nC','C count in variable region')

print("\n=== quartile view: best vs worst represented composition ===")
g=d.groupby('GC').agg(cov=('ON_OFF','size'),perf=('ON_OFF','mean'))
g=g[g['cov']>=200].sort_values('cov')
lo=g.head(max(1,len(g)//4)); hi=g.tail(max(1,len(g)//4))
wl=np.average(lo['perf'],weights=lo['cov']); wh=np.average(hi['perf'],weights=hi['cov'])
print(f"  least-represented GC bins  (n={int(lo['cov'].sum())}): mean ON_OFF = {wl:.4f}")
print(f"  most-represented  GC bins  (n={int(hi['cov'].sum())}): mean ON_OFF = {wh:.4f}")
print(f"  ratio = {wl/wh:.2f}x")
