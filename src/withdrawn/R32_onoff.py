import numpy as np, pandas as pd, itertools, gc
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy import stats
V=np.load("V_raw.npy"); seqs=np.load("seq_raw.npy",allow_pickle=True)
L=np.array([len(s) for s in seqs])
BUF=[1,3,5,7,9]; TGT=[2,4,6,8]     # drop the FM-bracketed 5th target
B5=V[:,BUF]
gate=(B5>=150).all(1)&(B5<=1200).all(1)&(L==10)
for t in TGT+[10]: gate&=(V[:,t]>=100)&(V[:,t]<=1200)
# photobleaching-corrected OFF and ON, each normalised to the run's own decay
buf=V[:,BUF].mean(1); tgt=V[:,TGT].mean(1)
df=pd.DataFrame({"seq":seqs[gate],"buf":buf[gate],"tgt":tgt[gate]})
g=df.groupby("seq").agg(buf=("buf","mean"),tgt=("tgt","mean"),n=("buf","size"))
g=g[[len(i)==10 for i in g.index]]
s=g.index.to_numpy(); N=len(s)
lb=np.log(g.buf.to_numpy()).astype(np.float32)     # OFF state (quenched, bound)
lt=np.log(g.tgt.to_numpy()).astype(np.float32)     # ON state (released)
ratio=(lt-lb).astype(np.float32)                   # log switching
print(f"{N:,} sequences   corr(logOFF, logON) = {stats.pearsonr(lb,lt).statistic:.4f}")
print(f"sd(logOFF) {lb.std():.4f}   sd(logON) {lt.std():.4f}   sd(log ratio) {ratio.std():.4f}")
M={'A':0,'C':1,'G':2,'T':3}
A=np.array([[M[c] for c in q] for q in s],dtype=np.int8)
pairs=list(itertools.combinations(range(10),2))
X=np.empty((N,55),dtype=np.int16); X[:,:10]=A
for k,(i,j) in enumerate(pairs): X[:,10+k]=A[:,i].astype(np.int16)*4+A[:,j]
rng=np.random.default_rng(0); perm=rng.permutation(N); hold=perm[:60000]; pool=perm[60000:]
def fit(t,tag):
    m=HistGradientBoostingRegressor(max_iter=1200,learning_rate=.1,max_leaf_nodes=31,
        early_stopping=False,categorical_features=[True]*55,max_bins=16,
        l2_regularization=1.0).fit(X[pool],t[pool])
    p=m.predict(X[hold]); tt=t[hold]
    r2=1-((tt-p)**2).sum()/((tt-tt.mean())**2).sum()
    print(f"  {tag:44s} R2={r2:.4f}",flush=True); return p,r2
print("\nHOW PREDICTABLE IS EACH STATE FROM SEQUENCE?")
pb,rb=fit(lb,"OFF state  (buffer, aptamer-bound/quenched)")
pt,rt=fit(lt,"ON state   (target present, released)")
pr,rr=fit(ratio,"log switching ratio (ON - OFF)")
print("\nWHICH STATE DRIVES THE SWITCHING SIGNAL?")
yt=ratio[hold]
print(f"  variance of log-ratio explained by predicted OFF alone : "
      f"{stats.pearsonr(yt,-pb).statistic**2:.4f}")
print(f"  variance of log-ratio explained by predicted ON alone  : "
      f"{stats.pearsonr(yt,pt).statistic**2:.4f}")
print(f"\n  corr(log ratio, logOFF) = {stats.pearsonr(ratio,lb).statistic:+.4f}")
print(f"  corr(log ratio, logON ) = {stats.pearsonr(ratio,lt).statistic:+.4f}")
print("\n  DECOMPOSITION of switching variance:")
vb,vt=lb.var(),lt.var(); cv=np.cov(lb,lt)[0,1]
print(f"    var(logOFF)          {vb:.5f}  ({vb/ratio.var()*100:5.1f}% of switching variance)")
print(f"    var(logON)           {vt:.5f}  ({vt/ratio.var()*100:5.1f}%)")
print(f"    -2*cov(OFF,ON)       {-2*cv:.5f}")
print(f"    = var(log ratio)     {ratio.var():.5f}")
