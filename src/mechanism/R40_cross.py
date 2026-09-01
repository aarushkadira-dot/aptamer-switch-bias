import numpy as np, pandas as pd
from scipy import stats
V=np.load("V_raw.npy"); seqs=np.load("seq_raw.npy",allow_pickle=True)
L=np.array([len(s) for s in seqs])
tg=[2,4,6,8]; bl=[(1,3),(3,5),(5,7),(7,9)]
B5=V[:,[1,3,5,7,9]]
gate=(B5>=150).all(1)&(B5<=1200).all(1)&(L==10)
for t in tg: gate&=(V[:,t]>=100)&(V[:,t]<=1200)
R=np.vstack([V[:,t]/((V[:,a]+V[:,b])/2) for t,(a,b) in zip(tg,bl)]).T
ok=gate&np.isfinite(R).all(1); R=R[ok]; sq=seqs[ok]
x=np.arange(4)-1.5
slope=((R-R.mean(1,keepdims=True))*x).sum(1)/(x**2).sum()
level=R.mean(1)
df=pd.DataFrame({"seq":sq,"level":level,"slope":slope})
df=df.sample(frac=1.,random_state=1); df["i"]=df.groupby("seq").cumcount()
c=df.groupby("seq").size(); df=df[df.seq.isin(c[c>=2].index)]
a=df[df.i==0].set_index("seq"); b=df[df.i==1].set_index("seq")
j=a.join(b,lsuffix="_A",rsuffix="_B",how="inner")
print(f"sequences with >=2 independent clusters: {len(j):,}\n")
print("SAME cluster (shared noise possible):")
print(f"  corr(slope_A, level_A) = {stats.pearsonr(j.slope_A,j.level_A).statistic:+.4f}")
print("\nCROSS cluster (independent measurements, shared noise impossible):")
r1=stats.pearsonr(j.slope_A,j.level_B); r2=stats.pearsonr(j.slope_B,j.level_A)
print(f"  corr(slope_A, level_B) = {r1.statistic:+.4f}   p={r1.pvalue:.2e}")
print(f"  corr(slope_B, level_A) = {r2.statistic:+.4f}   p={r2.pvalue:.2e}")
print("\nreplicate agreement of each quantity (for reference):")
print(f"  level : {stats.pearsonr(j.level_A,j.level_B).statistic:+.4f}")
print(f"  slope : {stats.pearsonr(j.slope_A,j.slope_B).statistic:+.4f}")
print("\nVERDICT: if the cross-cluster correlations are ~0 while the same-cluster one")
print("is strongly negative, the 'fatigue' is a within-measurement artifact, not a")
print("property of the sequence.")
