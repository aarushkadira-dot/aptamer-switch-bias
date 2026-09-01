import numpy as np
from scipy import stats
import numpy.linalg as la
cm={'A':'T','C':'G','G':'C','T':'A'}; rc=lambda s:"".join(cm[c] for c in reversed(s))
ATP="ACCTGGGGGAGTATTGCGGAGGAAGGT"
GLU="CCCGTGTTAGATAGTAAGTGCAATCTCGGC"
P_ATP=np.array([34,63,77,83,85,72,52,43,31,37,43,67,127,162,207,229,248,193,142,98,70,45,33,25,21,9],float)
P_GLU=np.array([40,52,52,53,54,24,11,8,3,4,3,5,17,20,18,20,21,17,15,20,24,23,28,29,28,25,16,7,6,4],float)
def pur(apt,W):
    n=len(apt); out=[]
    for i in range(n):
        a=max(0,i-W//2); b=min(n,i+W//2+1); w=apt[a:b]
        out.append((w.count('A')+w.count('G'))/len(w))
    return np.array(out)
print("=== CHECK 1: robustness to window size ===")
print(f"  {'window':>7} {'r ATP':>9} {'p':>9} {'r GLU':>9} {'p':>9}")
for W in [3,5,7,9,11]:
    a=pur(ATP,W)[:len(P_ATP)]; g=pur(GLU,W)[:len(P_GLU)]
    ra=stats.pearsonr(a,P_ATP); rg=stats.pearsonr(g,P_GLU)
    print(f"  {W:>7} {ra.statistic:+9.3f} {ra.pvalue:9.4f} {rg.statistic:+9.3f} {rg.pvalue:9.4f}")
print("\n=== CHECK 2: multiple testing (7 features tested) ===")
print("  Bonferroni-corrected p:  ATP 0.031*7 = 0.217 (NOT significant)")
print("                           GLU 0.0003*7 = 0.0021 (survives)")
print("\n=== CHECK 3: is it just position? (partial corr controlling for distance from 5' end) ===")
for nm,apt,P in [('ATP',ATP,P_ATP),('GLU',GLU,P_GLU)]:
    a=pur(apt,5)[:len(P)]; pos=np.arange(len(P),dtype=float)
    X=np.column_stack([np.ones(len(P)),pos,pos**2])
    ra=P-X@la.lstsq(X,P,rcond=None)[0]
    rp=a-X@la.lstsq(X,a,rcond=None)[0]
    r=stats.pearsonr(rp,ra)
    print(f"  {nm}: raw r={stats.pearsonr(a,P).statistic:+.3f}   after removing quadratic position trend r={r.statistic:+.3f} p={r.pvalue:.4f}")
print("\n=== CHECK 4: permutation test (shuffle aptamer, recompute) ===")
rng=np.random.default_rng(0)
for nm,apt,P in [('ATP',ATP,P_ATP),('GLU',GLU,P_GLU)]:
    obs=stats.pearsonr(pur(apt,5)[:len(P)],P).statistic
    null=[]
    for _ in range(5000):
        s=''.join(rng.permutation(list(apt)))
        null.append(stats.pearsonr(pur(s,5)[:len(P)],P).statistic)
    null=np.array(null)
    p=(null<=obs).mean()
    print(f"  {nm}: observed r={obs:+.3f}   permutation p={p:.4f}  (null mean {null.mean():+.3f}, sd {null.std():.3f})")
