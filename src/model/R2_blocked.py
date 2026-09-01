import pandas as pd, numpy as np, itertools, gc
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy import stats
d=pd.read_csv("atp_fixB.csv"); d=d[d.seq.str.len()==10]
g=d.groupby("seq")["ratio"].agg(["mean","size"]); del d; gc.collect()
s=g.index.to_numpy(); y=g["mean"].to_numpy().astype(np.float32); N=len(s)
M={'A':0,'C':1,'G':2,'T':3}
A=np.array([[M[c] for c in q] for q in s],dtype=np.int8)
pw=(4**np.arange(10)).astype(np.int64); code=(A.astype(np.int64)*pw).sum(1)
idx_of=np.full(1048576,-1,dtype=np.int32); idx_of[code]=np.arange(N)
pairs=list(itertools.combinations(range(10),2))
X=np.empty((N,55),dtype=np.int16); X[:,:10]=A
for i_,(i,j) in enumerate(pairs): X[:,10+i_]=A[:,i].astype(np.int16)*4+A[:,j]
cat=[True]*55
rng=np.random.default_rng(0); perm=rng.permutation(N)
TEST=perm[:10000]; POOL=perm[10000:]; tc=code[TEST]
print(f"coverage: {N:,}/1,048,576 = {N/1048576*100:.1f}% of sequence space")
poolset=np.zeros(1048576,dtype=bool); poolset[code[POOL]]=True
nb1=np.zeros(len(tc),dtype=np.int32)
for p in range(10):
    cur=(tc//pw[p])%4
    for b in range(4):
        nb1+=((b!=cur)&poolset[tc+(b-cur)*pw[p]]).astype(np.int32)
print(f"mean distance-1 training neighbours per test seq: {nb1.mean():.1f}/30; with zero: {(nb1==0).sum()}")
def ball(codes,r):
    out=[codes]; cur=codes
    for _ in range(r):
        acc=[]
        for p in range(10):
            c=(cur//pw[p])%4
            for b in range(4):
                m=b!=c; acc.append((cur+(b-c)*pw[p])[m])
        cur=np.unique(np.concatenate(acc)); out.append(cur)
    return np.unique(np.concatenate(out))
yt=y[TEST]
def run(tr,tag):
    m=HistGradientBoostingRegressor(max_iter=1200,learning_rate=.1,max_leaf_nodes=31,
        early_stopping=False,categorical_features=cat,max_bins=16,l2_regularization=1.0).fit(X[tr],y[tr])
    p=m.predict(X[TEST])
    r2=1-((yt-p)**2).sum()/((yt-yt.mean())**2).sum()
    t=len(set(np.argsort(-yt)[:100].tolist())&set(np.argsort(-p)[:100].tolist()))
    print(f"  {tag:50s} n={len(tr):>7,}  R2={r2:.3f}  rank={stats.spearmanr(yt,p).statistic:.3f}  top100={t}",flush=True)
    return r2
print("\nSAME 10,000 test sequences everywhere; only TRAINING differs (corrected labels)")
run(POOL,"full training (random split)")
for r in [1,2]:
    Bl=ball(tc,r); ex=idx_of[Bl[poolset[Bl]]]
    keep=np.setdiff1d(POOL,ex)
    b=run(keep,f"BLOCKED: nothing within Hamming <= {r}")
    c=run(rng.choice(POOL,len(keep),replace=False),f"size-matched random control (r={r})")
    print(f"    -> interpolation contribution at radius {r}: {c-b:+.3f} R2\n")
