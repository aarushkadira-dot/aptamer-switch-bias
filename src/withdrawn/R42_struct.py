import numpy as np
from scipy import stats
from collections import defaultdict
apt="ACCTGGGGGAGTATTGCGGAGGAAGGT"; cm={'A':'T','C':'G','G':'C','T':'A'}
rc=lambda s:"".join(cm[c] for c in reversed(s)); L="ACGT"
# NMR secondary structure read from Fig 2c (1-indexed)
STEM=set([1,2,3,4,5, 23,24,25,26,27, 8,9,10,11,12, 16,17,18,19])
LOOP=set([6,7, 13,14,15, 20,21,22])
print("ATP aptamer secondary structure (Fig 2c, NMR):")
print("  helix / paired  :",sorted(STEM))
print("  loop / unpaired :",sorted(LOOP))
pred=np.load("R_predfull.npy")
allc=np.arange(1048576,dtype=np.int64)
A=np.empty((1048576,10),dtype=np.int8)
for p in range(10): A[:,p]=(allc//(4**p))%4
s2=lambda a:"".join(L[b] for b in a)
sc=defaultdict(list)
sub=np.random.default_rng(1).choice(1048576,150000,replace=False)
for i in sub:
    q=s2(A[i]); v=pred[i]
    for j in range(6): sc[q[j:j+5]].append(v)
mean={k:np.mean(v) for k,v in sc.items() if len(v)>=40}
gm=np.mean(list(mean.values()))
prop=np.full(27,np.nan)
for p in range(27):
    vals=[mean[rc(apt[st:st+5])]-gm for st in range(max(0,p-4),min(p+1,23)) if rc(apt[st:st+5]) in mean]
    if vals: prop[p]=np.mean(vals)
AUTH=[34,63,77,83,85,72,52,43,31,37,43,67,127,162,207,229,248,193,142,98,70,45,33,25,21,9]
st=[prop[p-1] for p in sorted(STEM) if not np.isnan(prop[p-1])]
lp=[prop[p-1] for p in sorted(LOOP) if not np.isnan(prop[p-1])]
print(f"\nMODEL targeting propensity:")
print(f"  paired (helix)   mean {np.mean(st):+.4f}   n={len(st)}")
print(f"  unpaired (loop)  mean {np.mean(lp):+.4f}   n={len(lp)}")
print(f"  Mann-Whitney p = {stats.mannwhitneyu(st,lp).pvalue:.4f}")
sa=[AUTH[p-1] for p in sorted(STEM) if p<=26]; la=[AUTH[p-1] for p in sorted(LOOP) if p<=26]
print(f"\nAUTHORS' measured targeting counts (Fig 2a):")
print(f"  paired (helix)   mean {np.mean(sa):6.1f}")
print(f"  unpaired (loop)  mean {np.mean(la):6.1f}")
print(f"  Mann-Whitney p = {stats.mannwhitneyu(sa,la).pvalue:.4f}")
print("\nIf the screen were a STRUCTURE PROBE, unpaired loops would be targeted MORE.")
print("Verdict below:")
print(f"  model : loops targeted {'MORE' if np.mean(lp)>np.mean(st) else 'LESS'} than helices")
print(f"  authors: loops targeted {'MORE' if np.mean(la)>np.mean(sa) else 'LESS'} than helices")
