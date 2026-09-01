import pandas as pd, numpy as np, gc, torch, torch.nn as nn
from sklearn.ensemble import HistGradientBoostingRegressor
cm={'A':'T','C':'G','G':'C','T':'A'}; rc=lambda s:"".join(cm[c] for c in reversed(s))
apt="ACCTGGGGGAGTATTGCGGAGGAAGGT"; R=rc(apt); LR=len(R)
M={'A':0,'C':1,'G':2,'T':3}
d=pd.read_csv("atp_fixB.csv"); d=d[d.seq.str.len()==10]
g=d.groupby("seq")["ratio"].agg(["mean","size"]); del d; gc.collect()
y=g["mean"].to_numpy().astype(np.float32); N=len(y)
CNT=np.load("CNT.npy"); BEST=np.load("BESTL.npy")
hold=np.load("R_hold.npy"); pool=np.setdiff1d(np.arange(N),hold); yt=y[hold]
def r2(p): return 1-((yt-p)**2).sum()/((yt-yt.mean())**2).sum()
# dinucleotide composition of every (p,L) block
DI=np.zeros((LR,11,16),dtype=np.float32)
for p in range(LR):
    for L in range(3,11):
        if p+L>LR: continue
        for k in range(p,p+L-1): DI[p,L,M[R[k]]*4+M[R[k+1]]]+=1
CNTf=CNT.astype(np.float32).reshape(N,-1)
DIf=DI.reshape(-1,16)
print("BASELINES")
b=BEST.reshape(-1,1).astype(np.float32)
m=HistGradientBoostingRegressor(max_iter=300,learning_rate=.1,early_stopping=False).fit(b[pool],y[pool])
print(f"  longest contiguous block, ALONE (1 feature)          R2={r2(m.predict(b[hold])):.4f}")
m=HistGradientBoostingRegressor(max_iter=600,learning_rate=.1,early_stopping=False).fit(CNTf[pool],y[pool])
print(f"  full block table (297 features, black box)           R2={r2(m.predict(CNTf[hold])):.4f}")
print("\nBIOPHYSICAL MODEL: Z = sum_blocks exp(-dG/RT), dG = init + sum(stacking)")
print("                   ratio = base + amp * Z/(1+Z)      [19 parameters]")
dev="cpu"
Xc=torch.tensor(CNTf); Yt=torch.tensor(y); D=torch.tensor(DIf)
tr=torch.tensor(pool); ho=torch.tensor(hold)
class Nuc(nn.Module):
    def __init__(s):
        super().__init__()
        s.stack=nn.Parameter(torch.full((16,),-1.0))
        s.init=nn.Parameter(torch.tensor(4.0))
        s.base=nn.Parameter(torch.tensor(1.02))
        s.amp=nn.Parameter(torch.tensor(2.0))
    def dG(s): return s.init + D@s.stack
    def forward(s,c):
        Z=(c*torch.exp(-s.dG()).clamp(max=1e6)).sum(1)
        return s.base+s.amp*Z/(1+Z)
net=Nuc(); opt=torch.optim.Adam(net.parameters(),lr=0.05)
for ep in range(400):
    perm=tr[torch.randperm(len(tr))][:60000]
    opt.zero_grad(); loss=((net(Xc[perm])-Yt[perm])**2).mean(); loss.backward(); opt.step()
    if ep%100==0:
        with torch.no_grad(): print(f"    epoch {ep:>3}  train MSE {loss.item():.5f}  held-out R2 {r2(net(Xc[ho]).numpy()):.4f}",flush=True)
with torch.no_grad():
    p=net(Xc[ho]).numpy(); print(f"\n  BIOPHYSICAL (19 params)                             R2={r2(p):.4f}")
    st=net.stack.detach().numpy(); L="ACGT"
    print(f"  fitted initiation penalty: {net.init.item():+.2f}  base {net.base.item():.3f}  amp {net.amp.item():.3f}")
    print("  fitted stacking energies (arbitrary units, lower = more stable):")
    order=np.argsort(st)
    for i in order:
        if DIf[:,i].sum()==0: continue
        print(f"    {L[i//4]}{L[i%4]}  {st[i]:+.3f}")
print("\n  reference: free positional model 0.4584, GBM+pairs 0.494, ensemble 0.525")
