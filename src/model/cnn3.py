import numpy as np, pandas as pd, torch, torch.nn as nn, sys, time, os
from scipy import stats
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(4)
d=pd.read_csv("atp_fixB.csv"); d=d[d.seq.str.len()==10]
g=d.groupby("seq")["ratio"].agg(["mean","size"]); del d
seqs=g.index.to_numpy(); y=g["mean"].to_numpy().astype(np.float32); nrep=g["size"].to_numpy().astype(np.float32)
M={'A':0,'C':1,'G':2,'T':3}
ia=np.array([[M[c] for c in s] for s in seqs],dtype=np.int64)
X=np.zeros((len(seqs),4,10),dtype=np.float32)
for j in range(10): X[np.arange(len(seqs)),ia[:,j],j]=1.
rng=np.random.default_rng(0); perm=rng.permutation(len(seqs))
hold,pool=perm[:60000],perm[60000:]; val,train=pool[:40000],pool[40000:]
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
def met(yt,p):
    r2=1-((yt-p)**2).sum()/((yt-yt.mean())**2).sum()
    top=set(np.argsort(-yt)[:100].tolist()); pt=set(np.argsort(-p)[:100].tolist())
    return r2, stats.spearmanr(yt,p).statistic, len(top&pt)
TAG=sys.argv[1]; EP=int(sys.argv[2]); TGT=sys.argv[3]
ck=f"ck3_{TAG}.pt"
yy=np.log2(y) if TGT=="log" else y
Xt=torch.tensor(X); Yt=torch.tensor(yy); W=torch.tensor(np.sqrt(nrep))
tr=torch.tensor(train); va=torch.tensor(val); ho=torch.tensor(hold)
net=Net(); opt=torch.optim.AdamW(net.parameters(),lr=2e-3,weight_decay=1e-4)
start=0; best=-9
if os.path.exists(ck):
    s=torch.load(ck); net.load_state_dict(s["net"]); opt.load_state_dict(s["opt"])
    start=s["ep"]; best=s["best"]; print(f"resumed at epoch {start}, best {best:.4f}",flush=True)
bs=4096
for ep in range(start,start+EP):
    net.train(); o=tr[torch.randperm(len(tr))]
    for i in range(0,len(o),bs):
        b=o[i:i+bs]
        loss=((net(Xt[b])-Yt[b])**2*W[b]).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad(): pv=net(Xt[va]).numpy()
    if TGT=="log": pv=2**pv
    r2=met(y[val],pv)[0]
    print(f"  epoch {ep:3d}  val R2 {r2:.4f}",flush=True)
    if r2>best:
        best=r2
        torch.save({"net":net.state_dict(),"opt":opt.state_dict(),"ep":ep+1,"best":best,
                    "bestnet":{k:v.clone() for k,v in net.state_dict().items()}},ck)
    else:
        s=torch.load(ck); s["ep"]=ep+1; s["net"]=net.state_dict(); s["opt"]=opt.state_dict(); torch.save(s,ck)
s=torch.load(ck); net.load_state_dict(s["bestnet"]); net.eval()
with torch.no_grad(): p=net(Xt[ho]).numpy()
if TGT=="log": p=2**p
np.save(f"p3_{TAG}.npy",p)
r2,rho,t=met(y[hold],p)
print(f"\n  HELD-OUT: R2={r2:.3f}  rank={rho:.3f}  top100={t}   (baseline GBM: 0.499 / 0.486 / 12)",flush=True)
