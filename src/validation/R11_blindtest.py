import numpy as np, itertools
from scipy import stats
BUF={"atp-1":[137,115,103,103,134],"atp-2":[110,112,109,116,118],"atp-3":[125,109,120,115,123],
     "atp-4":[115,128,105,126,116],"atp-5":[118,108,113,143,111],"atp-6":[116,106,105,180,140],
     "atp-7":[149,130,113,124,109],"atp-8":[165,154,119,150,148]}
ATP={"atp-1":[931,823,790,848],"atp-2":[816,838,813,800],"atp-3":[845,893,768,792],
     "atp-4":[820,809,853,773],"atp-5":[758,776,782,714],"atp-6":[691,664,717,610],
     "atp-7":[654,612,575,549],"atp-8":[675,661,628,594]}
VAR={"atp-1":"CGACGCGTAATC","atp-2":"ACTCCAGCGATC","atp-3":"CGTAGCTCACTC",
     "atp-4":"CGCATCCAGGTC","atp-5":"GAGCAATATATC","atp-6":"ATGCGCTACTC",
     "atp-7":"CCGGGGGCTATC","atp-8":"CCGGGGGCGATC"}
names=list(VAR)
fold=np.array([np.mean(ATP[n])/np.mean(BUF[n]) for n in names])
print("MEASURED plate-reader fold-change (ATP / buffer), from Fig 3a source data:")
for n,f in zip(names,fold): print(f"  {n}: {f:6.2f}x")
pred=np.load("R_predfull.npy")
M={'A':0,'C':1,'G':2,'T':3}; pw=(4**np.arange(10)).astype(np.int64)
def score(q): return float(pred[int(sum(M[c]*pw[i] for i,c in enumerate(q)))])
frames={"first-10 (drop trailing TC)":lambda v: v[:10],
        "last-10":lambda v: v[-10:],
        "max over all 10-mer windows":None,
        "mean over all 10-mer windows":None}
print(f"\n{'frame':32s} {'Pearson r':>10} {'Spearman':>10} {'p (spearman)':>13}")
print("-"*70)
for fr,fn in frames.items():
    vals=[]
    for n in names:
        v=VAR[n]
        if fn: 
            q=fn(v)
            if len(q)<10: vals.append(np.nan); continue
            vals.append(score(q))
        else:
            ws=[v[i:i+10] for i in range(len(v)-9)]
            sc=[score(w) for w in ws]
            vals.append(max(sc) if "max" in fr else float(np.mean(sc)))
    vals=np.array(vals); m=~np.isnan(vals)
    if m.sum()<5: print(f"{fr:32s} {'n<5':>10}"); continue
    pr=stats.pearsonr(fold[m],vals[m]); sp=stats.spearmanr(fold[m],vals[m])
    print(f"{fr:32s} {pr.statistic:10.3f} {sp.statistic:10.3f} {sp.pvalue:13.3f}   (n={m.sum()})")
print("\nPer-construct, 'mean over windows' frame:")
for n in names:
    v=VAR[n]; ws=[v[i:i+10] for i in range(len(v)-9)]
    print(f"  {n}: measured {np.mean(ATP[n])/np.mean(BUF[n]):5.2f}x   model {np.mean([score(w) for w in ws]):.3f}")
print("\nNOTE: n=8, and all 8 were chosen by the authors BECAUSE they were top performers,")
print("so the dynamic range is compressed (5.7x-6.9x). This is a range-restricted test.")
