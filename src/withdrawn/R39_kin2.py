import numpy as np, pandas as pd
from scipy import stats
V=np.load("V_raw.npy"); seqs=np.load("seq_raw.npy",allow_pickle=True)
L=np.array([len(s) for s in seqs])
tg=[2,4,6,8]; bl=[(1,3),(3,5),(5,7),(7,9)]
B5=V[:,[1,3,5,7,9]]
gate=(B5>=150).all(1)&(B5<=1200).all(1)&(L==10)
for t in tg: gate&=(V[:,t]>=100)&(V[:,t]<=1200)
R=np.vstack([V[:,t]/((V[:,a]+V[:,b])/2) for t,(a,b) in zip(tg,bl)]).T
ok=gate&np.isfinite(R).all(1); R=R[ok]
tgt=V[ok][:,tg].mean(1); buf=V[ok][:,[1,3,5,7,9]].mean(1)
x=np.arange(4)-1.5
slope=((R-R.mean(1,keepdims=True))*x).sum(1)/(x**2).sum()
level=R.mean(1)
print("Is the decline of strong switches BIOLOGY (fatigue) or a BLEACHING artifact?")
print(f"  corr(slope, switching level)   = {stats.pearsonr(slope,level).statistic:+.4f}")
print(f"  corr(slope, raw target signal) = {stats.pearsonr(slope,tgt).statistic:+.4f}")
print(f"  corr(slope, raw buffer signal) = {stats.pearsonr(slope,buf).statistic:+.4f}")
# partial: slope ~ level controlling for raw brightness
import numpy.linalg as la
Xd=np.column_stack([np.ones_like(tgt),tgt,buf])
res_s=slope-Xd@la.lstsq(Xd,slope,rcond=None)[0]
res_l=level-Xd@la.lstsq(Xd,level,rcond=None)[0]
print(f"\n  PARTIAL corr(slope, level | raw brightness) = {stats.pearsonr(res_s,res_l).statistic:+.4f}")
print("  -> if this stays strongly negative, the fatigue is not just bleaching.")
print("\nDecline within NARROW raw-brightness bands (bleaching held ~constant):")
q=np.quantile(tgt,[.2,.4,.6,.8])
for i,(lo,hi) in enumerate(zip([tgt.min()]+list(q),list(q)+[tgt.max()])):
    m=(tgt>=lo)&(tgt<hi)
    if m.sum()<2000: continue
    r=stats.pearsonr(slope[m],level[m]).statistic
    print(f"   target band {lo:6.0f}-{hi:6.0f}  n={m.sum():>7,}  corr(slope,level) = {r:+.4f}")
