import numpy as np, pandas as pd, gc
from scipy import stats
from collections import defaultdict
apt="ACCTGGGGGAGTATTGCGGAGGAAGGT"; cm={'A':'T','C':'G','G':'C','T':'A'}
rc=lambda s:"".join(cm[c] for c in reversed(s)); L="ACGT"; LA=len(apt)
# Authors' Fig 2a: how often each aptamer position was targeted by their top-1000 SDs
POS=list(range(1,27)); CNT=[34,63,77,83,85,72,52,43,31,37,43,67,127,162,207,229,248,193,142,98,70,45,33,25,21,9]
auth=np.array(CNT,dtype=float)
pred=np.load("R_predfull.npy")
allc=np.arange(1048576,dtype=np.int64)
A=np.empty((1048576,10),dtype=np.int8)
for p in range(10): A[:,p]=(allc//(4**p))%4
s2=lambda a:"".join(L[b] for b in a)
sub=np.random.default_rng(1).choice(1048576,150000,replace=False)
# 5-mer mean predicted switching
sc=defaultdict(list)
for i in sub:
    q=s2(A[i]); v=pred[i]
    for j in range(6): sc[q[j:j+5]].append(v)
mean={k:np.mean(v) for k,v in sc.items() if len(v)>=40}
gm=np.mean(list(mean.values()))
# model-derived targeting propensity of each aptamer position:
# average model preference over every 5-nt window covering that position
prop=np.full(LA,np.nan)
for p in range(LA):
    vals=[]
    for st in range(max(0,p-4),min(p+1,LA-4)):
        w=rc(apt[st:st+5])
        if w in mean: vals.append(mean[w]-gm)
    if vals: prop[p]=np.mean(vals)
m=~np.isnan(prop)
mp=prop[:26][m[:26]]; ap=auth[m[:26]]
print("Aptamer position: authors' experimental targeting count vs MODEL preference")
print(f"  {'nt':>3} {'authors count':>14} {'model propensity':>17}")
for i in range(26):
    if not m[i]: continue
    bar="#"*int(auth[i]/10)
    print(f"  {i+1:>3} {int(auth[i]):>14} {prop[i]:>17.3f}  {bar}")
r=stats.pearsonr(ap,mp); rho=stats.spearmanr(ap,mp)
print(f"\n  n={len(ap)}  Pearson r = {r.statistic:+.3f}  p = {r.pvalue:.2e}")
print(f"           Spearman = {rho.statistic:+.3f}  p = {rho.pvalue:.2e}")
print(f"\n  authors' peak position : nt {POS[int(np.argmax(auth))]}")
print(f"  model's peak position  : nt {int(np.nanargmax(prop))+1}")
