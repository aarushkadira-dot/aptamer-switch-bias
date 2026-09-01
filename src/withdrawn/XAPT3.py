import numpy as np
from scipy import stats
cm={'A':'T','C':'G','G':'C','T':'A'}; rc=lambda s:"".join(cm[c] for c in reversed(s))
ATP="ACCTGGGGGAGTATTGCGGAGGAAGGT"; GLU="CCCGTGTTAGATAGTAAGTGCAATCTCGGC"
P_ATP=np.array([34,63,77,83,85,72,52,43,31,37,43,67,127,162,207,229,248,193,142,98,70,45,33,25,21,9],float)
P_GLU=np.array([40,52,52,53,54,24,11,8,3,4,3,5,17,20,18,20,21,17,15,20,24,23,28,29,28,25,16,7,6,4],float)
NN={'AA':-1.00,'TT':-1.00,'AT':-0.88,'TA':-0.58,'CA':-1.45,'TG':-1.45,'GT':-1.44,'AC':-1.44,
    'CT':-1.28,'AG':-1.28,'GA':-1.30,'TC':-1.30,'CG':-2.17,'GC':-2.24,'GG':-1.84,'CC':-1.84}
def feat(apt,name,W=5):
    n=len(apt); o=[]
    for i in range(n):
        w=apt[max(0,i-W//2):min(n,i+W//2+1)]; L=len(w)
        if   name=='purine': o.append((w.count('A')+w.count('G'))/L)
        elif name=='C':      o.append(w.count('C')/L)
        elif name=='G':      o.append(w.count('G')/L)
        elif name=='GC':     o.append((w.count('G')+w.count('C'))/L)
        elif name=='compC':  o.append(rc(w).count('C')/L)
        elif name=='dG':     o.append(-sum(NN.get(w[k:k+2],0) for k in range(L-1)))
    return np.array(o)
FEATS=['purine','C','G','GC','compC','dG']
rng=np.random.default_rng(0); NP=20000
print("PERMUTATION NULL (shuffle aptamer sequence, preserve composition) -- the honest test\n")
print(f"  {'feature':>8} | {'r ATP':>7} {'perm p':>8} | {'r GLU':>7} {'perm p':>8} | {'Fisher combined p':>18}")
print("  "+"-"*72)
res={}
for f in FEATS:
    row=[]
    for apt,P in [(ATP,P_ATP),(GLU,P_GLU)]:
        obs=stats.pearsonr(feat(apt,f)[:len(P)],P).statistic
        null=np.array([stats.pearsonr(feat(''.join(rng.permutation(list(apt))),f)[:len(P)],P).statistic for _ in range(NP)])
        p=((np.abs(null)>=abs(obs)).sum()+1)/(NP+1)   # two-sided
        row.append((obs,p))
    (ra,pa),(rg,pg)=row
    X2=-2*(np.log(pa)+np.log(pg)); pc=stats.chi2.sf(X2,4)
    res[f]=pc
    flag=" <<<" if pc<0.05/len(FEATS) else (" *" if pc<0.05 else "")
    print(f"  {f:>8} | {ra:+7.3f} {pa:8.4f} | {rg:+7.3f} {pg:8.4f} | {pc:18.4f}{flag}")
print(f"\n  Bonferroni threshold for {len(FEATS)} features: p < {0.05/len(FEATS):.4f}")
best=min(res,key=res.get)
print(f"  Best feature: '{best}' combined p={res[best]:.4f}  -> {'SURVIVES' if res[best]<0.05/len(FEATS) else 'DOES NOT SURVIVE'} correction")
