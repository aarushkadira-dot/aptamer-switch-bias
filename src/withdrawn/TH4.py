import pandas as pd, numpy as np
from scipy import stats
import statsmodels.api as sm
F="/sessions/dazzling-sweet-pascal/mnt/Figure 2 and 3/GSE149225_toehold_processed_datafile.csv"
d=pd.read_csv(F,low_memory=False); d=d.dropna(subset=['ON_OFF']); d=d[d['QC_ON_OFF']>=3].copy()
d['V']=d['off_id'].astype(str).str[20:50]
for b in 'ACGT': d['n'+b]=d['V'].str.count(b)
d['GC']=d['nG']+d['nC']
print("=== per-base effect on ON_OFF, toehold library (n=%d) ==="%len(d))
for b in 'ACGT':
    r=stats.spearmanr(d['n'+b],d['ON_OFF'])
    print(f"  n{b}: library freq {d['n'+b].mean()/30:.3f}   Spearman vs ON_OFF = {r.statistic:+.4f}  p={r.pvalue:.3g}")
print("\n=== C effect CONTROLLING for total GC (is it C specifically, or just GC?) ===")
out=[]
for gc,sub in d.groupby('GC'):
    if len(sub)<300 or sub['nC'].nunique()<4: continue
    r=stats.spearmanr(sub['nC'],sub['ON_OFF'])
    out.append((gc,len(sub),r.statistic,r.pvalue))
print(f"  {'GC':>4} {'n':>6} {'rho(nC|GC)':>11} {'p':>10}")
for gc,n,rho,p in out: print(f"  {gc:>4} {n:>6} {rho:>+11.4f} {p:>10.3g}")
rhos=np.array([o[2] for o in out]); ns=np.array([o[1] for o in out])
print(f"  weighted mean partial rho = {np.average(rhos,weights=ns):+.4f}   ({(rhos>0).sum()}/{len(rhos)} strata positive)")
print("\n=== control: source organism (genome GC confound) ===")
d['org']=d['source_sequence'].astype(str)
big=d['org'].value_counts(); big=big[big>=500].index
sub=d[d['org'].isin(big)]
print(f"  {len(big)} organisms with n>=500, total n={len(sub)}")
X=pd.get_dummies(sub['org'],drop_first=True).astype(float)
X['nC']=sub['nC'].values; X['GC']=sub['GC'].values; X['GC2']=sub['GC'].values**2
X=sm.add_constant(X)
m=sm.OLS(sub['ON_OFF'].values,X).fit()
print(f"  OLS ON_OFF ~ nC + GC + GC^2 + organism fixed effects")
print(f"    beta_nC = {m.params['nC']:+.5f}   SE {m.bse['nC']:.5f}   t={m.tvalues['nC']:+.2f}   p={m.pvalues['nC']:.3g}")
print(f"    -> each extra C in the 30-nt variable region changes ON_OFF by {m.params['nC']:+.4f}")
print(f"    R2={m.rsquared:.4f}")
print("\n=== where is the library, vs where is the optimum? ===")
print(f"  library modal nC = {int(d['nC'].mode().iloc[0])}   mean nC = {d['nC'].mean():.2f}  ({d['nC'].mean()/30:.1%} C)")
g=d.groupby('nC').agg(n=('ON_OFF','size'),perf=('ON_OFF','mean')); g=g[g['n']>=200]
print(f"  best-performing nC bin = {g['perf'].idxmax()}  ({g['perf'].idxmax()/30:.1%} C), mean ON_OFF {g['perf'].max():.4f}")
print(f"  fraction of library at or above the optimum bin: {(d['nC']>=g['perf'].idxmax()).mean():.2%}")
