import pandas as pd, numpy as np, gc, itertools
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy import stats
cm={'A':'T','C':'G','G':'C','T':'A'}; rc=lambda s:"".join(cm[c] for c in reversed(s))
apt="ACCTGGGGGAGTATTGCGGAGGAAGGT"
d=pd.read_csv("atp_fixB.csv"); d=d[d.seq.str.len()==10]
g=d.groupby("seq")["ratio"].agg(["mean","size"]); del d; gc.collect()
s=g.index.to_numpy(); y=g["mean"].to_numpy().astype(np.float32); N=len(s)
M={'A':0,'C':1,'G':2,'T':3}; L="ACGT"
A=np.array([[M[c] for c in q] for q in s],dtype=np.int8)
hold=np.load("R_hold.npy"); pool=np.setdiff1d(np.arange(N),hold); yt=y[hold]
def r2(p): return 1-((yt-p)**2).sum()/((yt-yt.mean())**2).sum()
tri=list(itertools.combinations(range(10),3))
pairs=list(itertools.combinations(range(10),2))
X3=np.empty((N,55+len(tri)),dtype=np.int16); X3[:,:10]=A
for k,(i,j) in enumerate(pairs): X3[:,10+k]=A[:,i].astype(np.int16)*4+A[:,j]
for k,(i,j,l) in enumerate(tri): X3[:,55+k]=(A[:,i].astype(np.int16)*16+A[:,j]*4+A[:,l])
m=HistGradientBoostingRegressor(max_iter=1200,learning_rate=.1,max_leaf_nodes=31,
    early_stopping=False,categorical_features=[True]*(55+len(tri)),max_bins=70,
    l2_regularization=1.0).fit(X3[pool],y[pool])
print(f"  additive 0.458 | +pairs 0.494 | +triples {r2(m.predict(X3[hold])):.3f}")
print("\nLEARNED POSITION-WEIGHT MATRIX (mean measured ratio, training data only)")
print("  position:      "+"".join(f"{p:>7}" for p in range(10)))
PWM=np.zeros((10,4))
for b in range(4):
    row=[]
    for p in range(10):
        mask=A[pool,p]==b
        PWM[p,b]=y[pool][mask].mean(); row.append(PWM[p,b])
    print(f"      {L[b]}:        "+"".join(f"{v:7.3f}" for v in row))
bestb=[L[int(np.argmax(PWM[p]))] for p in range(10)]
print("  best base:     "+"".join(f"{c:>7}" for c in bestb))
print(f"\n  optimal SD by PWM: {''.join(bestb)}")
R=rc(apt)
print(f"  revcomp(aptamer) = {R}")
best=None
for off in range(len(R)-9):
    w=R[off:off+10]; sc=sum(a==b for a,b in zip(w,bestb))
    if best is None or sc>best[0]: best=(sc,off,w)
print(f"  closest 10-nt window of revcomp(aptamer): {best[2]} (aptamer nt {len(apt)-best[1]-9}-{len(apt)-best[1]})")
print(f"  identity: {best[0]}/10")
print("\n  per-position: does the PWM's favourite base match that window?")
print("    PWM  "+"  ".join(bestb))
print("    apt  "+"  ".join(best[2]))
print("    match"+"  ".join(" Y" if a==b else " ." for a,b in zip(bestb,best[2])))
sc=PWM.max(1)-PWM.min(1)
print(f"\n  positional information content (max-min mean ratio per position):")
print("    "+"".join(f"{v:7.3f}" for v in sc))
print(f"  most discriminating positions: {np.argsort(-sc)[:4]}")
