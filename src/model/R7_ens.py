import os, pandas as pd, numpy as np, itertools, gc, torch, torch.nn as nn
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy import stats
d=pd.read_csv(os.environ.get("ATP_DATA","data/atp_corrected.csv.gz")); d=d[d.seq.str.len()==10]
g=d.groupby("seq")["ratio"].agg(["mean","size"]); del d; gc.collect()
s=g.index.to_numpy(); y=g["mean"].to_numpy().astype(np.float32); N=len(s)
M={'A':0,'C':1,'G':2,'T':3}
A=np.array([[M[c] for c in q] for q in s],dtype=np.int8)
pairs=list(itertools.combinations(range(10),2))
X=np.empty((N,55),dtype=np.int16); X[:,:10]=A
for i_,(i,j) in enumerate(pairs): X[:,10+i_]=A[:,i].astype(np.int16)*4+A[:,j]
Xo=np.zeros((N,4,10),dtype=np.float32)
for j in range(10): Xo[np.arange(N),A[:,j],j]=1.
cat=[True]*55
rng=np.random.default_rng(0); perm=rng.permutation(N)
hold=perm[:60000]; pool=perm[60000:]; val=pool[:40000]; train=pool[40000:]
class Net(nn.Module):
    def __init__(s_,ch=96,dr=0.15):
        super().__init__()
        s_.c=nn.ModuleList([nn.Conv1d(4,ch,k,padding=k//2) for k in (2,3,4)])
        s_.bn=nn.BatchNorm1d(ch*3)
        s_.h=nn.Sequential(nn.Flatten(),nn.Dropout(dr),nn.Linear(ch*3*10,256),nn.ReLU(),
                           nn.Linear(256,64),nn.ReLU(),nn.Linear(64,1))
    def forward(s_,x):
        h=torch.cat([c(x)[:,:,:10] for c in s_.c],1)
        return s_.h(torch.relu(s_.bn(h))).squeeze(1)
net=Net(); net.load_state_dict(torch.load("ck3_raw.pt",map_location="cpu")["bestnet"]); net.eval()
def pr(idx):
    o=[]
    with torch.no_grad():
        for i in range(0,len(idx),8192): o.append(net(torch.tensor(Xo[idx[i:i+8192]])).numpy())
    return np.concatenate(o)
cv,chd=pr(val),pr(hold)
g1=HistGradientBoostingRegressor(max_iter=1200,learning_rate=.1,max_leaf_nodes=31,
    early_stopping=False,categorical_features=cat,max_bins=16,l2_regularization=1.0).fit(X[train],y[train])
gv=g1.predict(X[val]); gh=np.load("R_pgbm.npy")
def met(yy,p):
    r2=1-((yy-p)**2).sum()/((yy-yy.mean())**2).sum()
    t=len(set(np.argsort(-yy)[:100].tolist())&set(np.argsort(-p)[:100].tolist()))
    return r2,stats.spearmanr(yy,p).statistic,t
yv,yh=y[val],y[hold]
print("STEP 1 — select blend weight on VALIDATION (held-out untouched)")
best=(-9,None)
for a in np.arange(0.1,0.95,0.1):
    r=met(yv,a*gv+(1-a)*cv)[0]
    print(f"   alpha={a:.1f}  val R2 {r:.4f}")
    if r>best[0]: best=(r,a)
a=best[1]; print(f"   -> selected alpha = {a:.1f}\n")
print("STEP 2 — held-out (corrected labels), single evaluation")
print(f"   {'model':40s} {'R2':>8} {'rank':>8} {'top100':>8}")
rows=[("gradient boosting",gh),("CNN",chd),
      ("ensemble alpha=0.5 (untuned)",0.5*gh+0.5*chd),
      (f"ensemble alpha={a:.1f} (val-selected)",a*gh+(1-a)*chd)]
for nm,p in rows:
    r2,rho,t=met(yh,p); print(f"   {nm:40s} {r2:8.3f} {rho:8.3f} {t:8d}")
pe=a*gh+(1-a)*chd
br=np.random.default_rng(7); dif=[]
for _ in range(2000):
    i=br.integers(0,60000,60000); yy=yh[i]
    f=lambda p:1-((yy-p[i])**2).sum()/((yy-yy.mean())**2).sum()
    dif.append(f(pe)-f(gh))
dif=np.array(dif)
print(f"\n   ensemble - GBM: {dif.mean():+.4f}  95% CI [{np.percentile(dif,2.5):+.4f}, "
      f"{np.percentile(dif,97.5):+.4f}]  wins {(dif>0).mean()*100:.1f}%")
np.save("R_ens.npy",pe)
