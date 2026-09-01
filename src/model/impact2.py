import numpy as np, pandas as pd
from scipy import stats
V=np.load("V_raw.npy"); seqs=np.load("seq_raw.npy",allow_pickle=True)
L=np.array([len(s) for s in seqs]); tg=[2,4,6,8,10]
B5=V[:,[1,3,5,7,9]]
gate=(B5>=150).all(1)&(B5<=1200).all(1)&(V[:,tg]>=100).all(1)&(V[:,tg]<=1200).all(1)&(L==10)
def make(R):
    rsd=R.std(1,ddof=1)/R.mean(1)
    m=gate&(rsd<=0.30)&np.isfinite(R).all(1)
    return np.median(R,1),m
bl=[(1,3),(3,5),(5,7),(7,9),(9,11)]
R_old=np.vstack([V[:,t]/((V[:,b1]+V[:,b2])/2) for t,(b1,b2) in zip(tg,bl)]).T
R_a=R_old.copy(); R_a[:,4]=V[:,10]/V[:,9]          # fix: single preceding buffer
R_b=R_old[:,:4]                                     # fix: drop exposure 5
r_o,m_o=make(R_old); r_a,m_a=make(R_a); r_b,m_b=make(R_b)
print(f"{'variant':28s} {'clusters kept':>14} {'median':>9} {'mean':>9}")
for nm,(r,m) in [("shipped (FM in bracket)",(r_o,m_o)),("A: exp5 / buf9 only",(r_a,m_a)),("B: drop exp5",(r_b,m_b))]:
    print(f"{nm:28s} {m.sum():14,} {np.median(r[m]):9.5f} {r[m].mean():9.5f}")
print("\nPer-cluster agreement on the intersection:")
for nm,(r,m) in [("A",(r_a,m_a)),("B",(r_b,m_b))]:
    c=m_o&m
    print(f"  shipped vs {nm}: n={c.sum():,}  pearson {stats.pearsonr(r_o[c],r[c]).statistic:.4f}  "
          f"mean shift {(r[c]-r_o[c]).mean():+.5f}  sd {(r[c]-r_o[c]).std():.5f}")
print("\nSequence-level (the actual labels) and top-100 stability:")
for nm,(r,m) in [("A",(r_a,m_a)),("B",(r_b,m_b))]:
    c=m_o&m
    j=(pd.DataFrame({"s":seqs[c],"o":r_o[c],"n":r[c]}).groupby("s").mean())
    t_o=set(j.nlargest(100,"o").index); t_n=set(j.nlargest(100,"n").index)
    print(f"  {nm}: n_seq={len(j):,}  pearson {stats.pearsonr(j.o,j.n).statistic:.4f}  "
          f"spearman {stats.spearmanr(j.o,j.n).statistic:.4f}  top-100 overlap {len(t_o&t_n)}/100")
print("\nReplicate agreement (the QC metric that justified the whole correction):")
for nm,(r,m) in [("shipped",(r_o,m_o)),("A",(r_a,m_a)),("B",(r_b,m_b))]:
    df=pd.DataFrame({"s":seqs[m],"v":r[m]})
    df=df.sample(frac=1.,random_state=1); df["i"]=df.groupby("s").cumcount()
    cnt=df.groupby("s").v.size(); df=df[df.s.isin(cnt[cnt>=2].index)]
    p=df.assign(h=df.i%2).pivot_table(index="s",columns="h",values="v",aggfunc="mean").dropna()
    print(f"  {nm:8s} split-half r = {stats.pearsonr(p[0],p[1]).statistic:.4f}  (n={len(p):,})")
for nm,r,m in [("A",r_a,m_a),("B",r_b,m_b)]:
    pd.DataFrame({"seq":seqs[m],"B":B5[m].mean(1),"ratio":r[m]}).to_csv(f"atp_fix{nm}.csv",index=False)
