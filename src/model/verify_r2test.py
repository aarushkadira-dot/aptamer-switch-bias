import pandas as pd, numpy as np, itertools, gc, warnings; warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy import stats
D="/sessions/dazzling-sweet-pascal/mnt/outputs/deliverables/aptamer-switch-bias/data/atp_corrected.csv.gz"
d=pd.read_csv(D); d=d[d.seq.str.len()==10]
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
print(f"N={N:,}  hold={len(hold):,}  pool={len(pool):,}")
assert len(set(hold.tolist())&set(pool.tolist()))==0
def fit(t,tr=pool): return HistGradientBoostingRegressor(max_iter=1200,learning_rate=.1,
    max_leaf_nodes=31,early_stopping=False,categorical_features=cat,max_bins=16,
    l2_regularization=1.0).fit(X[tr],t[tr])
def r2(a,b): return 1-((a-b)**2).sum()/((a-a.mean())**2).sum()
yt=y[hold]
pg=fit(y).predict(X[hold])
rho=stats.spearmanr(yt,pg).statistic
t100=len(set(np.argsort(-yt)[:100].tolist())&set(np.argsort(-pg)[:100].tolist()))
print(f"\n*** GBM on corrected labels: R2 = {r2(yt,pg):.4f}   Spearman = {rho:.4f}   top100 = {t100}")
print(f"    README claims ensemble 0.525, GBM alone implied ~0.494")
ysh=y.copy(); ysh[pool]=rng.permutation(ysh[pool])
print(f"    shuffled-label control: R2 = {r2(yt,fit(ysh).predict(X[hold])):+.4f}   (README: -0.043)")
np.save("/tmp/R_pgbm.npy",pg); np.save("/tmp/R_hold.npy",hold); np.save("/tmp/R_y.npy",y)
