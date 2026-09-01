import numpy as np, pandas as pd, gc
from scipy import stats
L="ACGT"; M={'A':0,'C':1,'G':2,'T':3}
d=pd.read_csv("atp_fixB.csv"); d=d[d.seq.str.len()==10]
g=d.groupby("seq")["ratio"].agg(["mean","size"]); del d; gc.collect()
s=g.index.to_numpy(); y=g["mean"].to_numpy().astype(np.float32); k=g["size"].to_numpy()
pw=(4**np.arange(10)).astype(np.int64)
A=np.array([[M[c] for c in q] for q in s],dtype=np.int8)
code=(A.astype(np.int64)*pw).sum(1)
val=np.full(1048576,np.nan,dtype=np.float32); val[code]=y
print("MEASURED mutational fragility: all Hamming-1 pairs where BOTH were measured")
rows=[]
for p in range(10):
    cur=(code//pw[p])%4
    for b in range(4):
        m=b!=cur
        nb=code[m]+(b-cur[m])*pw[p]
        v=val[nb]; ok=~np.isnan(v)
        rows.append(np.column_stack([y[m][ok],v[ok]]))
P=np.vstack(rows)
print(f"  {len(P):,} measured neighbour pairs")
hi=P.max(1); lo=P.min(1)
print(f"\n  {'parent switching (decile)':>28} {'n':>9} {'mean drop to neighbour':>24} {'relative':>10}")
q=np.quantile(P[:,0],np.linspace(0,1,11))
for i in range(10):
    m=(P[:,0]>=q[i])&(P[:,0]<q[i+1] if i<9 else P[:,0]<=q[10])
    if m.sum()<200: continue
    drop=(P[m,0]-P[m,1]).mean()
    print(f"  {q[i]:11.3f}-{q[i+1]:<9.3f} {m.sum():>9,} {drop:>24.4f} {drop/P[m,0].mean():>10.3f}")
print("\n  TOP measured switches vs random, single-mutation effect:")
for tag,thr in [("top 1%",np.percentile(P[:,0],99)),("top 0.1%",np.percentile(P[:,0],99.9))]:
    m=P[:,0]>=thr
    print(f"   {tag:9s} n={m.sum():>7,}  parent {P[m,0].mean():.3f} -> neighbour {P[m,1].mean():.3f}  "
          f"loses {(1-P[m,1].mean()/P[m,0].mean())*100:.1f}%")
mid=(P[:,0]>np.percentile(P[:,0],45))&(P[:,0]<np.percentile(P[:,0],55))
print(f"   median    n={mid.sum():>7,}  parent {P[mid,0].mean():.3f} -> neighbour {P[mid,1].mean():.3f}  "
      f"loses {(1-P[mid,1].mean()/P[mid,0].mean())*100:.1f}%")
print("\n  (regression to the mean inflates this; the control is the BOTTOM decile,")
print("   which should GAIN by the same logic:)")
m=P[:,0]<=q[1]
print(f"   bottom decile n={m.sum():>7,}  parent {P[m,0].mean():.3f} -> neighbour {P[m,1].mean():.3f}  "
      f"changes {(P[m,1].mean()/P[m,0].mean()-1)*100:+.1f}%")
