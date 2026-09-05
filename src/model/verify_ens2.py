import numpy as np, pandas as pd, torch, torch.nn as nn, warnings; warnings.filterwarnings("ignore")
from scipy import stats
D="/sessions/dazzling-sweet-pascal/mnt/outputs/deliverables/aptamer-switch-bias/data/atp_corrected.csv.gz"
d=pd.read_csv(D); d=d[d.seq.str.len()==10]
g=d.groupby("seq")["ratio"].agg(["mean","size"]); del d
seqs=g.index.to_numpy()
M={'A':0,'C':1,'G':2,'T':3}
ia=np.array([[M[c] for c in s] for s in seqs],dtype=np.int64)
X=np.zeros((len(seqs),4,10),dtype=np.float32)
for j in range(10): X[np.arange(len(seqs)),ia[:,j],j]=1.
y=np.load("/tmp/E_y.npy"); val=np.load("/tmp/E_val.npy"); hold=np.load("/tmp/E_hold.npy")
gv=np.load("/tmp/E_gv.npy"); gh=np.load("/tmp/R_pgbm.npy")
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
net=Net(); ck=torch.load("/tmp/ck3_raw.pt",map_location="cpu")
net.load_state_dict(ck["bestnet"]); net.eval()
def pr(idx):
    o=[]
    with torch.no_grad():
        for i in range(0,len(idx),8192): o.append(net(torch.tensor(X[idx[i:i+8192]])).numpy())
    return np.concatenate(o)
cv,chd=pr(val),pr(hold)
def met(yy,p):
    r2=1-((yy-p)**2).sum()/((yy-yy.mean())**2).sum()
    t=len(set(np.argsort(-yy)[:100].tolist())&set(np.argsort(-p)[:100].tolist()))
    return r2,stats.spearmanr(yy,p).statistic,t
yv,yh=y[val],y[hold]
print(f"CNN checkpoint: {ck['ep']} epochs, best val R2 {ck['best']:.4f}\n")
print("STEP 1 - select blend weight on VALIDATION")
best=(-9,None)
for a in np.arange(0.1,0.95,0.1):
    r=met(yv,a*gv+(1-a)*cv)[0]
    print(f"   alpha={a:.1f}  val R2 {r:.4f}")
    if r>best[0]: best=(r,a)
a=best[1]; print(f"   -> selected alpha = {a:.1f}\n")
print("STEP 2 - HELD-OUT, single evaluation")
print(f"   {'model':40s} {'R2':>8} {'rank':>8} {'top100':>8}")
for nm,p in [("gradient boosting",gh),("CNN",chd),
             ("ensemble alpha=0.5 (untuned)",0.5*gh+0.5*chd),
             (f"ensemble alpha={a:.1f} (val-selected)",a*gh+(1-a)*chd)]:
    r2,rho,t=met(yh,p); print(f"   {nm:40s} {r2:8.4f} {rho:8.3f} {t:8d}")
pe=a*gh+(1-a)*chd
br=np.random.default_rng(7); dif=[]
for _ in range(2000):
    i=br.integers(0,60000,60000); yy=yh[i]
    f=lambda p:1-((yy-p[i])**2).sum()/((yy-yy.mean())**2).sum()
    dif.append(f(pe)-f(gh))
dif=np.array(dif)
print(f"\n   ensemble - GBM: {dif.mean():+.4f}  95% CI [{np.percentile(dif,2.5):+.4f}, {np.percentile(dif,97.5):+.4f}]  wins {(dif>0).mean()*100:.1f}%")
print(f"\n   README claims: headline R2 0.525, ensemble over GBM +0.031, CI [+0.023,+0.038], 100% of resamples")
