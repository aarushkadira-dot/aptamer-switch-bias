import os, pandas as pd, numpy as np, itertools, gc
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy import stats
d=pd.read_csv(os.environ.get("ATP_DATA","data/atp_corrected.csv.gz")); d=d[d.seq.str.len()==10]
g=d.groupby("seq").agg(y=("ratio","mean"),k=("ratio","size"),B=("B","mean")); del d; gc.collect()
s=g.index.to_numpy(); y=g["y"].to_numpy().astype(np.float32)
k=g["k"].to_numpy(); B=g["B"].to_numpy().astype(np.float32); N=len(s)
M={'A':0,'C':1,'G':2,'T':3}
A=np.array([[M[c] for c in q] for q in s],dtype=np.int8)
pairs=list(itertools.combinations(range(10),2))
X=np.empty((N,55),dtype=np.int16); X[:,:10]=A
for i_,(i,j) in enumerate(pairs): X[:,10+i_]=A[:,i].astype(np.int16)*4+A[:,j]
cat=[True]*55
rng=np.random.default_rng(0); perm=rng.permutation(N)
hold=perm[:60000]; pool=perm[60000:]
print(f"N={N:,}  hold={len(hold):,}  pool={len(pool):,}  (sum={len(hold)+len(pool):,})")
assert len(set(hold.tolist())&set(pool.tolist()))==0, "LEAK"
print("leak check: train/hold disjoint at sequence level  OK")
def fit(t,tr=pool): return HistGradientBoostingRegressor(max_iter=1200,learning_rate=.1,
    max_leaf_nodes=31,early_stopping=False,categorical_features=cat,max_bins=16,
    l2_regularization=1.0).fit(X[tr],t[tr])
def r2(a,b): return 1-((a-b)**2).sum()/((a-a.mean())**2).sum()
yt=y[hold]
pg=fit(y).predict(X[hold]); np.save("R_pgbm.npy",pg); np.save("R_hold.npy",hold)
rho=stats.spearmanr(yt,pg).statistic
t100=len(set(np.argsort(-yt)[:100].tolist())&set(np.argsort(-pg)[:100].tolist()))
print(f"\nBASELINE GBM (corrected labels): R2={r2(yt,pg):.3f}  rank={rho:.3f}  top100={t100}")
print(f"   (uncorrected labels were:      R2=0.499        rank=0.486    top100=12)")
print("\nNEGATIVE CONTROL — shuffle the labels, refit. Must give R2 ~ 0.")
ysh=y.copy(); ysh[pool]=rng.permutation(ysh[pool])
psh=fit(ysh).predict(X[hold])
print(f"   shuffled-label R2 on held-out = {r2(yt,psh):+.4f}   rank={stats.spearmanr(yt,psh).statistic:+.4f}")
print("\nCONFOUNDS (corrected labels)")
pB=fit(B).predict(X[hold]); print(f"   sequence -> brightness B : R2={r2(B[hold],pB):.3f}")
fB=HistGradientBoostingRegressor(max_iter=200,learning_rate=.1,early_stopping=False).fit(B[pool].reshape(-1,1),y[pool])
yB=y-fB.predict(B.reshape(-1,1)).astype(np.float32)
print(f"   variance removed by B    : {(1-yB[pool].var()/y[pool].var())*100:.1f}%   "
      f"-> seq predicts residual R2={r2(yB[hold],fit(yB).predict(X[hold])):.3f}")
kf=k.astype(np.float32)
fk=HistGradientBoostingRegressor(max_iter=200,learning_rate=.1,early_stopping=False).fit(kf[pool].reshape(-1,1),y[pool])
yk=y-fk.predict(kf.reshape(-1,1)).astype(np.float32)
print(f"   variance removed by k    : {(1-yk[pool].var()/y[pool].var())*100:.1f}%   "
      f"-> seq predicts residual R2={r2(yk[hold],fit(yk).predict(X[hold])):.3f}")
C=np.stack([(A==b).sum(1) for b in range(4)],1).astype(np.float32)
mc=HistGradientBoostingRegressor(max_iter=400,learning_rate=.1,early_stopping=False).fit(C[pool],y[pool])
print(f"   composition alone        : R2={r2(yt,mc.predict(C[hold])):.3f}")
yc=y-mc.predict(C).astype(np.float32)
print(f"   variance removed by comp : {(1-yc[pool].var()/y[pool].var())*100:.1f}%   "
      f"-> seq predicts residual R2={r2(yc[hold],fit(yc).predict(X[hold])):.3f}")
