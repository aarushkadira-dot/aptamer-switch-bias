import pandas as pd, numpy as np, gc
from scipy import stats
cm={'A':'T','C':'G','G':'C','T':'A'}; rc=lambda s:"".join(cm[c] for c in reversed(s))
apt="ACCTGGGGGAGTATTGCGGAGGAAGGT"
MO={"m1":rc(apt[12:17]),"m2":rc(apt[16:20]),"m3":rc(apt[0:5])}
def load(f):
    d=pd.read_csv(f); d=d[d.seq.str.len()==10]
    return d.groupby("seq")["ratio"].agg(["mean","size"])
old=load("atp_v2.csv")      # as-published normalisation (FM in the bracket)
new=load("atp_fixB.csv")    # corrected
print(f"old: {len(old):,} sequences   corrected: {len(new):,} sequences")
j=old[["mean"]].rename(columns={"mean":"old"}).join(new[["mean"]].rename(columns={"mean":"new"}),how="inner")
print(f"shared: {len(j):,}   pearson {stats.pearsonr(j.old,j.new).statistic:.4f}   "
      f"spearman {stats.spearmanr(j.old,j.new).statistic:.4f}")
print("\nQ1. Does the identity of the TOP SEQUENCES change?")
for k in [100,1000,5000]:
    a=set(j.nlargest(k,"old").index); b=set(j.nlargest(k,"new").index)
    print(f"   top {k:>5}: overlap {len(a&b):>5}/{k}  ({len(a&b)/k*100:5.1f}%)  -> {k-len(a&b)} sequences change status")
print("\nQ2. Do the PUBLISHED MOTIF FREQUENCIES change?")
print("   (paper reports m1 28.2%, m2 9.3%, m3 4.9% of the top 1000)")
t_old=j.nlargest(1000,"old").index; t_new=j.nlargest(1000,"new").index
print(f"   {'motif':6s} {'seq':7s} {'old top1000':>12} {'corrected':>11} {'change':>9}")
for k,v in MO.items():
    a=np.mean([v in q for q in t_old])*100; b=np.mean([v in q for q in t_new])*100
    print(f"   {k:6s} {v:7s} {a:11.1f}% {b:10.1f}% {b-a:+8.1f}pp")
print("\nQ3. Does the WINNER change? (single best sequence)")
print(f"   old best      : {j.old.idxmax()}  ratio {j.old.max():.3f}")
print(f"   corrected best: {j.new.idxmax()}  ratio {j.new.max():.3f}")
print("\nQ4. Does the corrected data agree BETTER with the 8 lab-measured constructs?")
BUF={"atp-1":[137,115,103,103,134],"atp-2":[110,112,109,116,118],"atp-3":[125,109,120,115,123],
     "atp-4":[115,128,105,126,116],"atp-5":[118,108,113,143,111],"atp-6":[116,106,105,180,140],
     "atp-7":[149,130,113,124,109],"atp-8":[165,154,119,150,148]}
ATP={"atp-1":[931,823,790,848],"atp-2":[816,838,813,800],"atp-3":[845,893,768,792],
     "atp-4":[820,809,853,773],"atp-5":[758,776,782,714],"atp-6":[691,664,717,610],
     "atp-7":[654,612,575,549],"atp-8":[675,661,628,594]}
VAR={"atp-1":"CGACGCGTAATC","atp-2":"ACTCCAGCGATC","atp-3":"CGTAGCTCACTC","atp-4":"CGCATCCAGGTC",
     "atp-5":"GAGCAATATATC","atp-6":"ATGCGCTACTC","atp-7":"CCGGGGGCTATC","atp-8":"CCGGGGGCGATC"}
rows=[]
for n,v in VAR.items():
    q=v[:10]
    if q in j.index:
        rows.append((n,np.mean(ATP[n])/np.mean(BUF[n]),j.loc[q,"old"],j.loc[q,"new"]))
print(f"   {'construct':10s} {'lab fold':>9} {'old screen':>11} {'corrected':>10}")
for n,f,o,ne in rows: print(f"   {n:10s} {f:8.2f}x {o:11.3f} {ne:10.3f}")
if len(rows)>=3:
    f=np.array([r[1] for r in rows]); o=np.array([r[2] for r in rows]); ne=np.array([r[3] for r in rows])
    print(f"   correlation with lab measurement:  old {stats.pearsonr(f,o).statistic:+.3f}   "
          f"corrected {stats.pearsonr(f,ne).statistic:+.3f}   (n={len(rows)})")
