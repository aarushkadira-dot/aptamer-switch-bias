import pandas as pd, numpy as np
from scipy import stats
import statsmodels.api as sm
a=pd.read_csv('/tmp/atp_fixB.csv'); a=a[np.isfinite(a['ratio'])].copy()
for b in 'ACGT': a['n'+b]=a['seq'].astype(str).str.count(b)

print("=== STRESS 1: is the ATP C-effect just read-depth noise? (C-rich seqs are rarer -> fewer reads) ===")
print(f"  corr(nC, read depth B) = {stats.spearmanr(a['nC'],a['B']).statistic:+.4f}")
a['dq']=pd.qcut(a['B'],10,labels=False,duplicates='drop')
print(f"  {'depth decile':>13} {'median B':>9} {'n':>8} {'rho(nC,ratio)':>14}")
rr=[]
for q,s in a.groupby('dq'):
    r=stats.spearmanr(s['nC'],s['ratio']); rr.append((len(s),r.statistic))
    print(f"  {q:>13} {s['B'].median():>9.0f} {len(s):>8} {r.statistic:>+14.4f}")
n=np.array([x[0] for x in rr]); v=np.array([x[1] for x in rr])
print(f"  weighted mean within-depth rho = {np.average(v,weights=n):+.4f}  ({(v>0).sum()}/{len(v)} deciles positive)")
print(f"  -> depth does NOT explain it" if np.average(v,weights=n)>0.15 else "  -> depth may explain part of it")

print("\n=== STRESS 2: compositional constraint (counts sum to L). Regress with 3 of 4 bases, T as reference ===")
for nm,df,L,yc in [('ATP',a,10,'ratio')]:
    X=sm.add_constant(df[['nA','nC','nG']].astype(float))
    m=sm.OLS(stats.zscore(df[yc]),X).fit()
    print(f"  {nm}: (reference base = T)")
    for k in ['nA','nC','nG']:
        print(f"    beta_{k} = {m.params[k]:+.4f}  SE {m.bse[k]:.4f}  t={m.tvalues[k]:+.1f}")
F="/sessions/dazzling-sweet-pascal/mnt/Figure 2 and 3/GSE149225_toehold_processed_datafile.csv"
d=pd.read_csv(F,low_memory=False); d=d.dropna(subset=['ON_OFF']); d=d[d['QC_ON_OFF']>=3].copy()
d['V']=d['off_id'].astype(str).str[20:50]
for b in 'ACGT': d['n'+b]=d['V'].str.count(b)
X=pd.get_dummies(d['source_sequence'].astype(str),drop_first=True).astype(float)
for k in ['nA','nC','nG']: X[k]=d[k].values
X=sm.add_constant(X)
m2=sm.OLS(stats.zscore(d['ON_OFF']),X).fit()
print(f"  toehold: (reference base = T, + organism fixed effects)")
for k in ['nA','nC','nG']:
    print(f"    beta_{k} = {m2.params[k]:+.4f}  SE {m2.bse[k]:.4f}  t={m2.tvalues[k]:+.1f}")

print("\n=== STRESS 3: which bases actually AGREE across the two libraries? ===")
ra={b:stats.spearmanr(a['n'+b],a['ratio']).statistic for b in 'ACGT'}
rt={b:stats.spearmanr(d['n'+b],d['ON_OFF']).statistic for b in 'ACGT'}
print(f"  {'base':>5} {'ATP rho':>9} {'toehold rho':>13} {'agree?':>8}")
for b in 'ACGT':
    ag="YES" if np.sign(ra[b])==np.sign(rt[b]) else "no  <-- FLIPS"
    print(f"  {b:>5} {ra[b]:>+9.4f} {rt[b]:>+13.4f} {ag:>8}")

print("\n=== STRESS 4: effect size reality check ===")
for nm,df,yc,bcol in [('ATP',a,'ratio','nC'),('toehold',d,'ON_OFF','nC')]:
    g=df.groupby(bcol)[yc].agg(['size','mean']); g=g[g['size']>=200]
    lo,hi=g['mean'].iloc[0],g['mean'].max()
    print(f"  {nm}: mean function at lowest-C bin {lo:.4f} -> best-C bin {hi:.4f}  ({hi/lo:.2f}x)")
    print(f"       variance in function explained by nC alone: R2 = {stats.pearsonr(df[bcol],df[yc]).statistic**2:.4f}")
