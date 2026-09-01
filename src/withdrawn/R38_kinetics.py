import numpy as np, pandas as pd, gc
from scipy import stats
V=np.load("V_raw.npy"); seqs=np.load("seq_raw.npy",allow_pickle=True)
L=np.array([len(s) for s in seqs])
tg=[2,4,6,8]; bl=[(1,3),(3,5),(5,7),(7,9)]      # 4 properly bracketed exposures
B5=V[:,[1,3,5,7,9]]
gate=(B5>=150).all(1)&(B5<=1200).all(1)&(L==10)
for t in tg: gate&=(V[:,t]>=100)&(V[:,t]<=1200)
R=np.vstack([V[:,t]/((V[:,a]+V[:,b])/2) for t,(a,b) in zip(tg,bl)]).T
ok=gate&np.isfinite(R).all(1)
R=R[ok]; sq=seqs[ok]
print(f"{len(R):,} clusters with 4 clean sequential exposures")
x=np.arange(4)-1.5
slope=((R-R.mean(1,keepdims=True))*x).sum(1)/ (x**2).sum()   # per-cluster trend across cycles
level=R.mean(1)
print(f"mean level {level.mean():.4f}   mean slope {slope.mean():+.5f}   sd(slope) {slope.std():.4f}")
print(f"corr(level, slope) = {stats.pearsonr(level,slope).statistic:+.4f}")
df=pd.DataFrame({"seq":sq,"level":level,"slope":slope})
print("\nIS THE PER-CYCLE TREND A REAL SEQUENCE PROPERTY, OR NOISE?")
print("(split-half: do independent clusters of the same sequence agree on the trend?)")
d2=df.sample(frac=1.,random_state=1).copy(); d2["i"]=d2.groupby("seq").cumcount()
c=d2.groupby("seq").size(); d2=d2[d2.seq.isin(c[c>=2].index)]
for col in ["level","slope"]:
    p=d2.assign(h=d2.i%2).pivot_table(index="seq",columns="h",values=col,aggfunc="mean").dropna()
    r=stats.pearsonr(p[0],p[1]).statistic
    sb=2*r/(1+r)
    print(f"  {col:6s}: split-half r = {r:.4f}   (Spearman-Brown full = {sb:.4f})   n={len(p):,}")
print("\nMEAN TRAJECTORY across the 4 exposures, by switching strength:")
g=df.groupby("seq").agg(level=("level","mean"),slope=("slope","mean"),n=("level","size"))
hi=g.nlargest(2000,"level").index; lo=g.nsmallest(2000,"level").index
for tag,idx in [("top 2000 switches",hi),("bottom 2000",lo)]:
    m=np.isin(sq,idx)
    print(f"  {tag:20s} " + "  ".join(f"cyc{i+1} {R[m][:,i].mean():.4f}" for i in range(4))
          + f"   slope {slope[m].mean():+.5f}")
print("\nARE STRONG SWITCHES MORE OR LESS REVERSIBLE?")
print(f"  corr(level, slope) across sequences = {stats.pearsonr(g.level,g.slope).statistic:+.4f}")
g.to_csv("kinetics.csv")
