import numpy as np, pandas as pd, torch, torch.nn as nn, os, time, warnings; warnings.filterwarnings("ignore")
from scipy import stats
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(4)
D="/sessions/dazzling-sweet-pascal/mnt/outputs/deliverables/aptamer-switch-bias/data/atp_corrected.csv.gz"
d=pd.read_csv(D); d=d[d.seq.str.len()==10]
g=d.groupby("seq")["ratio"].agg(["mean","size"]); del d
seqs=g.index.to_numpy(); y=g["mean"].to_numpy().astype(np.float32); nrep=g["size"].to_numpy().astype(np.float32)
M={'A':0,'C':1,'G':2,'T':3}
ia=np.array([[M[c] for c in s] for s in seqs],dtype=np.int64)
X=np.zeros((len(seqs),4,10),dtype=np.float32)
for j in range(10): X[np.arange(len(seqs)),ia[:,j],j]=1.
rng=np.random.default_rng(0); perm=rng.permutation(len(seqs))
hold,pool=perm[:60000],perm[60000:]; val,train=pool[:40000],pool[40000:]
print(f"N={len(seqs):,} hold={len(hold):,} val={len(val):,} train={len(train):,}",flush=True)
class Net(nn.Module):
    def __init__(s,ch=96,dr=0.15):
        super().__init__()
        s.c=nn.ModuleList([nn.Conv1d(4,ch,k,padding=k//2) for k in (2,3,4)])
        s.bn=nn.BatchNorm1d(ch*3)
        s.h=nn.Sequential(nn.Flatten(),nn.Dropout(dr),nn.Linear(ch*3*10,256),nn.ReLU(),
                          nn.Linear(256,64),nn.ReLU(),nn.Linear(64,1))
    def forward(s,x):
        h=torch.cat([c(x)[:,:,:10] for c in s.c],1)
        return s.h(torch.relu(s.bn(h))).squeeze(1)
def r2(yt,p): return 1-((yt-p)**2).sum()/((yt-yt.mean())**2).sum()
Xt=torch.tensor(X); Yt=torch.tensor(y); W=torch.tensor(np.sqrt(nrep))
tr=torch.tensor(train); va=torch.tensor(val); ho=torch.tensor(hold)
ck="/tmp/ck3_raw.pt"
net=Net(); opt=torch.optim.AdamW(net.parameters(),lr=2e-3,weight_decay=1e-4)
start=0; best=-9
if os.path.exists(ck):
    s=torch.load(ck); net.load_state_dict(s["net"]); opt.load_state_dict(s["opt"])
    start=s["ep"]; best=s["best"]
    if "bestnet" in s: pass
    print(f"resumed epoch {start} best {best:.4f}",flush=True)
bs=4096; EP=int(os.environ.get("EP","12")); t0=time.time()
for ep in range(start,start+EP):
    net.train(); o=tr[torch.randperm(len(tr))]
    for i in range(0,len(o),bs):
        b=o[i:i+bs]
        loss=((net(Xt[b])-Yt[b])**2*W[b]).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad(): pv=net(Xt[va]).numpy()
    v=r2(y[val],pv)
    print(f"  epoch {ep:3d}  val R2 {v:.4f}   ({time.time()-t0:.0f}s)",flush=True)
    st={"net":net.state_dict(),"opt":opt.state_dict(),"ep":ep+1,"best":max(best,v)}
    if v>best:
        best=v; st["bestnet"]={k:v_.clone() for k,v_ in net.state_dict().items()}
    elif os.path.exists(ck):
        old=torch.load(ck)
        if "bestnet" in old: st["bestnet"]=old["bestnet"]
    torch.save(st,ck)
print(f"best val R2 = {best:.4f}",flush=True)
