import pandas as pd, numpy as np, gc, torch, torch.nn as nn
cm={'A':'T','C':'G','G':'C','T':'A'}; rc=lambda s:"".join(cm[c] for c in reversed(s))
apt="ACCTGGGGGAGTATTGCGGAGGAAGGT"; R=rc(apt); LR=len(R)
M={'A':0,'C':1,'G':2,'T':3}; Rv=np.array([M[c] for c in R],dtype=np.int8)
d=pd.read_csv("atp_fixB.csv"); d=d[d.seq.str.len()==10]
g=d.groupby("seq")["ratio"].agg(["mean","size"]); del d; gc.collect()
s=g.index.to_numpy(); y=g["mean"].to_numpy().astype(np.float32); N=len(s)
A=np.array([[M[c] for c in q] for q in s],dtype=np.int8)
OFFS=list(range(0,LR-9))          # 18 registers with full 10-nt overlap
nO=len(OFFS)
MATCH=np.zeros((N,nO,10),dtype=bool)
for k,off in enumerate(OFFS): MATCH[:,k,:]=(A==Rv[off:off+10][None,:])
ADJ=(MATCH[:,:,:-1]&MATCH[:,:,1:])                       # (N,18,9) stack formed
NM=MATCH.sum(2).astype(np.int8)                          # matches per register
DID=np.zeros((nO,9),dtype=np.int64)                      # dinucleotide id per (offset,i)
for k,off in enumerate(OFFS):
    for i in range(9): DID[k,i]=Rv[off+i]*4+Rv[off+i+1]
print(f"registers {nO}, matches/register mean {NM.mean():.2f}, best register mean {NM.max(1).mean():.2f}")
hold=np.load("R_hold.npy"); pool=np.setdiff1d(np.arange(N),hold); yt=y[hold]
def r2(p): return 1-((yt-p)**2).sum()/((yt-yt.mean())**2).sum()
ADJt=torch.tensor(ADJ.astype(np.float32)); NMt=torch.tensor(NM.astype(np.float32))
DIDt=torch.tensor(DID); Yt=torch.tensor(y)
tr=torch.tensor(pool); ho=torch.tensor(hold)
class Coop(nn.Module):
    """dG(register) = init + sum_i [pair i and i+1 both formed] * stack(dinuc) + mm*(#mismatches)
       Z = sum over registers exp(-dG);  ratio = base + amp * Z/(1+Z)"""
    def __init__(s):
        super().__init__()
        s.stack=nn.Parameter(torch.full((16,),-0.8))
        s.init=nn.Parameter(torch.tensor(3.0))
        s.mm=nn.Parameter(torch.tensor(0.3))
        s.base=nn.Parameter(torch.tensor(1.02)); s.amp=nn.Parameter(torch.tensor(1.5))
    def forward(s,adj,nm):
        st=s.stack[DIDt]                              # (18,9)
        dG=s.init + (adj*st.unsqueeze(0)).sum(2) + s.mm*(10.0-nm)
        Z=torch.exp(-dG.clamp(-30,30)).sum(1)
        return s.base+s.amp*Z/(1+Z)
net=Coop(); opt=torch.optim.Adam(net.parameters(),lr=0.03)
best=-9
for ep in range(700):
    p=tr[torch.randperm(len(tr))][:80000]
    opt.zero_grad(); l=((net(ADJt[p],NMt[p])-Yt[p])**2).mean(); l.backward(); opt.step()
    if ep%100==0 or ep==699:
        with torch.no_grad():
            v=r2(net(ADJt[ho],NMt[ho]).numpy())
            best=max(best,v)
            print(f"  epoch {ep:>3}  train MSE {l.item():.5f}   held-out R2 {v:.4f}",flush=True)
with torch.no_grad():
    print(f"\n  COOPERATIVE BIOPHYSICAL MODEL (21 params)   R2={r2(net(ADJt[ho],NMt[ho]).numpy()):.4f}")
    st=net.stack.detach().numpy(); L="ACGT"
    print(f"  initiation {net.init.item():+.2f}   mismatch penalty {net.mm.item():+.3f}   "
          f"base {net.base.item():.3f}  amp {net.amp.item():.3f}")
    used=sorted(set(DID.ravel().tolist()))
    print("  fitted stacking energies (lower = more stable):")
    for i in sorted(used,key=lambda z:st[z]):
        print(f"    {L[i//4]}{L[i%4]}  {st[i]:+.3f}")
print("\n  reference: perfect-block model 0.129 | free positional 0.458 | GBM+pairs 0.494")
