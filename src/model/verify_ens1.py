import pandas as pd, numpy as np, itertools, gc, warnings; warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingRegressor
D="/sessions/dazzling-sweet-pascal/mnt/outputs/deliverables/aptamer-switch-bias/data/atp_corrected.csv.gz"
d=pd.read_csv(D); d=d[d.seq.str.len()==10]
g=d.groupby("seq")["ratio"].agg(["mean","size"]); del d; gc.collect()
s=g.index.to_numpy(); y=g["mean"].to_numpy().astype(np.float32); N=len(s)
M={'A':0,'C':1,'G':2,'T':3}
A=np.array([[M[c] for c in q] for q in s],dtype=np.int8)
pairs=list(itertools.combinations(range(10),2))
X=np.empty((N,55),dtype=np.int16); X[:,:10]=A
for i_,(i,j) in enumerate(pairs): X[:,10+i_]=A[:,i].astype(np.int16)*4+A[:,j]
cat=[True]*55
rng=np.random.default_rng(0); perm=rng.permutation(N)
hold=perm[:60000]; pool=perm[60000:]; val=pool[:40000]; train=pool[40000:]
np.save("/tmp/E_y.npy",y); np.save("/tmp/E_val.npy",val); np.save("/tmp/E_hold.npy",hold)
g1=HistGradientBoostingRegressor(max_iter=1200,learning_rate=.1,max_leaf_nodes=31,
    early_stopping=False,categorical_features=cat,max_bins=16,l2_regularization=1.0).fit(X[train],y[train])
np.save("/tmp/E_gv.npy",g1.predict(X[val]))
print("GBM(train) -> val predictions saved")
