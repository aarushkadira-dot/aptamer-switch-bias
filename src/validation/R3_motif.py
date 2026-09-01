import pandas as pd, numpy as np, itertools, gc
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy import stats
from collections import defaultdict
apt="ACCTGGGGGAGTATTGCGGAGGAAGGT"; comp={'A':'T','C':'G','G':'C','T':'A'}
rc=lambda s:"".join(comp[c] for c in reversed(s)); L="ACGT"
MO={"m1":rc(apt[12:17]),"m2":rc(apt[16:20]),"m3":rc(apt[0:5])}
d=pd.read_csv("atp_fixB.csv"); d=d[d.seq.str.len()==10]
g=d.groupby("seq")["ratio"].agg(["mean","size"]); del d; gc.collect()
s=g.index.to_numpy(); y=g["mean"].to_numpy().astype(np.float32); N=len(s)
M={'A':0,'C':1,'G':2,'T':3}
A=np.array([[M[c] for c in q] for q in s],dtype=np.int8)
pairs=list(itertools.combinations(range(10),2))
def feats(Ar):
    n=len(Ar); Z=np.empty((n,55),dtype=np.int16); Z[:,:10]=Ar
    for k,(i,j) in enumerate(pairs): Z[:,10+k]=Ar[:,i].astype(np.int16)*4+Ar[:,j]
    return Z
X=feats(A); cat=[True]*55
hold=np.load("R_hold.npy"); pg=np.load("R_pgbm.npy")
hs=s[hold]; hy=y[hold]
frac=lambda seqs,m: np.mean([m in q for q in seqs])
print("HELD-OUT motif recovery (corrected labels)")
mtop=hs[np.argsort(-pg)[:1000]]; atop=hs[np.argsort(-hy)[:1000]]
print(f"  {'motif':5s} {'seq':7s} {'bg':>7} {'true top1k':>11} {'MODEL top1k':>12} {'OR':>7} {'p':>11}")
for k,v in MO.items():
    bg=frac(hs,v); a=sum(v in q for q in mtop)
    c=sum(v in q for q in hs)-a
    orr,p=stats.fisher_exact([[a,1000-a],[c,len(hs)-1000-c]])
    print(f"  {k:5s} {v:7s} {bg*100:6.2f}% {frac(atop,v)*100:10.1f}% {frac(mtop,v)*100:11.1f}% {orr:7.2f} {p:11.2e}")
anym=lambda q: any(v in q for v in MO.values())
print(f"  any motif: bg {np.mean([anym(q) for q in hs])*100:.1f}%  true {np.mean([anym(q) for q in atop])*100:.1f}%  MODEL {np.mean([anym(q) for q in mtop])*100:.1f}%")
print("\nPROSPECTIVE — retrain on all data, score the full 4^10 space")
mdl=HistGradientBoostingRegressor(max_iter=1200,learning_rate=.1,max_leaf_nodes=31,
    early_stopping=False,categorical_features=cat,max_bins=16,l2_regularization=1.0).fit(X,y)
allc=np.arange(1048576,dtype=np.int64)
Aall=np.empty((1048576,10),dtype=np.int8)
for p in range(10): Aall[:,p]=(allc//(4**p))%4
pred=np.empty(1048576,dtype=np.float32)
for i in range(0,1048576,200000): pred[i:i+200000]=mdl.predict(feats(Aall[i:i+200000]))
s2=lambda a:"".join(L[b] for b in a)
sc=defaultdict(list)
sub=np.random.default_rng(1).choice(1048576,150000,replace=False)
for i in sub:
    q=s2(Aall[i]); v=pred[i]
    for j in range(6): sc[q[j:j+5]].append(v)
mean={k:np.mean(v) for k,v in sc.items() if len(v)>=40}
order=sorted(mean,key=lambda z:-mean[z]); rank={z:i+1 for i,z in enumerate(order)}
print(f"  ranked {len(order):,} 5-mers by model preference")
print(f"  {'aptamer nt':>11}  {'window':7s} {'revcomp(SD)':12s} {'rank':>6}")
rows=[]
for st in range(len(apt)-4):
    w=apt[st:st+5]; r=rc(w)
    if r in rank:
        rows.append((st+1,w,r,rank[r]))
for st,w,r,rk in sorted(rows,key=lambda z:z[3])[:8]:
    tag="  <-- "+[x for x,vv in MO.items() if vv==r][0] if r in MO.values() else ""
    print(f"  {st:>4}-{st+4:<6}  {w:7s} {r:12s} {rk:>6}{tag}")
allr=[z[3] for z in rows]
print(f"\n  median rank of aptamer-complementary 5-mers: {np.median(allr):,.0f} of {len(order):,} (chance {len(order)/2:.0f})")
print(f"  Mann-Whitney vs uniform: p = {stats.mannwhitneyu(allr,np.arange(1,len(order)+1),alternative='less').pvalue:.2e}")
for k,v in MO.items():
    print(f"  {k} ({v}) rank {rank.get(v,'NA')}")
np.save("R_predfull.npy",pred)
