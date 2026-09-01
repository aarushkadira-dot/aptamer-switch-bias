import pandas as pd, numpy as np, itertools, gc
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy import stats
from collections import defaultdict
glu="CCCGTGTTAGATAGTAAGTGCAATCTCGGC"; cm={'A':'T','C':'G','G':'C','T':'A'}
rc=lambda s:"".join(cm[c] for c in reversed(s)); L="ACGT"; LG=len(glu)
print(f"glucose aptamer NNGmin (Fig 4a): {glu}  len={LG}")
# authors' Fig 4B: positional targeting counts from their top-1000 glucose SDs
AUTH=[40,52,52,53,54,24,11,8,3,4,3,5,17,20,18,20,21,17,15,20,24,23,28,29,28,25,16,7,6,4]
print(f"their Fig4B peak at position {int(np.argmax(AUTH))+1} (count {max(AUTH)})")
d=pd.read_csv("glu_L.csv"); d=d[d.seq.str.len()==10]
g=d.groupby("seq")["ratio"].agg(["mean","size"]); del d; gc.collect()
s=g.index.to_numpy(); y=g["mean"].to_numpy().astype(np.float32); N=len(s)
print(f"\nglucose 10 mM: {N:,} unique 10-mers  (per-sequence split-half reliability 0.012)")
M={'A':0,'C':1,'G':2,'T':3}
A=np.array([[M[c] for c in q] for q in s],dtype=np.int8)
pairs=list(itertools.combinations(range(10),2))
def feats(Ar):
    n=len(Ar); Z=np.empty((n,55),dtype=np.int16); Z[:,:10]=Ar
    for k,(i,j) in enumerate(pairs): Z[:,10+k]=Ar[:,i].astype(np.int16)*4+Ar[:,j]
    return Z
X=feats(A); cat=[True]*55
rng=np.random.default_rng(0); perm=rng.permutation(N); hold=perm[:50000]; pool=perm[50000:]
yt=y[hold]
m=HistGradientBoostingRegressor(max_iter=1200,learning_rate=.1,max_leaf_nodes=31,
    early_stopping=False,categorical_features=cat,max_bins=16,l2_regularization=1.0).fit(X[pool],y[pool])
p=m.predict(X[hold])
r2=1-((yt-p)**2).sum()/((yt-yt.mean())**2).sum()
print(f"PER-SEQUENCE prediction: held-out R2 = {r2:+.4f}   (ATP was +0.494)")
print("  -> as expected, individual glucose sequences are not predictable.\n")
print("But does the AGGREGATE targeting profile still emerge?")
allc=np.arange(1048576,dtype=np.int64)
Aall=np.empty((1048576,10),dtype=np.int8)
for q_ in range(10): Aall[:,q_]=(allc//(4**q_))%4
pred=np.empty(1048576,dtype=np.float32)
for i in range(0,1048576,200000): pred[i:i+200000]=m.predict(feats(Aall[i:i+200000]))
s2=lambda a:"".join(L[b] for b in a)
sc=defaultdict(list)
sub=np.random.default_rng(1).choice(1048576,150000,replace=False)
for i in sub:
    q=s2(Aall[i]); v=pred[i]
    for j in range(6): sc[q[j:j+5]].append(v)
mean={k:np.mean(v) for k,v in sc.items() if len(v)>=40}
gm=np.mean(list(mean.values()))
prop=np.full(LG,np.nan)
for q_ in range(LG):
    vals=[]
    for st in range(max(0,q_-4),min(q_+1,LG-4)):
        w=rc(glu[st:st+5])
        if w in mean: vals.append(mean[w]-gm)
    if vals: prop[q_]=np.mean(vals)
ok=~np.isnan(prop)
a=np.array(AUTH,dtype=float)[ok]; b=prop[ok]
print(f"  {'nt':>3} {'their count':>12} {'model':>9}")
for i in range(LG):
    if not ok[i]: continue
    print(f"  {i+1:>3} {AUTH[i]:>12} {prop[i]:>9.4f}  {'#'*int(AUTH[i]/3)}")
r=stats.pearsonr(a,b); rho=stats.spearmanr(a,b)
print(f"\n  n={ok.sum()}  Pearson r={r.statistic:+.3f} p={r.pvalue:.2e}   Spearman={rho.statistic:+.3f} p={rho.pvalue:.2e}")
print(f"  their peak: nt {int(np.argmax(AUTH))+1}   model peak: nt {int(np.nanargmax(prop))+1}")
print(f"\n  ATP comparison was r=+0.859, p=1.9e-08")
