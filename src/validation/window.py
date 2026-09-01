import numpy as np
from collections import defaultdict
apt="ACCTGGGGGAGTATTGCGGAGGAAGGT"; comp={'A':'T','C':'G','G':'C','T':'A'}
rc=lambda s:"".join(comp[c] for c in reversed(s)); L="ACGT"
pred=np.load("pred_full.npy")
allc=np.arange(1048576,dtype=np.int64)
A=np.empty((1048576,10),dtype=np.int8)
for p in range(10): A[:,p]=(allc//(4**p))%4
sc=defaultdict(list)
sub=np.random.default_rng(1).choice(1048576,150000,replace=False)
s2=lambda a:"".join(L[b] for b in a)
for i in sub:
    q=s2(A[i]); v=pred[i]
    for j in range(6): sc[q[j:j+5]].append(v)
mean={k:np.mean(v) for k,v in sc.items() if len(v)>=40}
order=sorted(mean,key=lambda k:-mean[k]); rank={k:i+1 for i,k in enumerate(order)}
print(f"ranked {len(order):,} 5-mers by the model's mean predicted switching\n")
print("For each 5-nt window of the ATP aptamer: where does its REVERSE COMPLEMENT rank?")
print(f"  {'aptamer nt':>11}  {'window':7s} {'revcomp(=SD)':13s} {'model rank':>11}  {'percentile':>10}")
rows=[]
for st in range(len(apt)-4):
    w=apt[st:st+5]; r=rc(w)
    if r in rank:
        pc=100*(1-rank[r]/len(order))
        rows.append((st+1,w,r,rank[r],pc))
        bar="#"*int(pc/3)
        print(f"  {st+1:>4}-{st+5:<6}  {w:7s} {r:13s} {rank[r]:>11,}  {pc:9.1f}% {bar}")
best=sorted(rows,key=lambda z:z[3])[:6]
print("\nTop-ranked aptamer windows (the regions the model says to target):")
for st,w,r,rk,pc in best: print(f"   nt {st}-{st+4}: {w} -> SD {r}  rank {rk}")
lo=min(z[0] for z in best); hi=max(z[0]+4 for z in best)
print(f"\n   -> these span aptamer nt {lo}-{hi}")
print("   Paper Fig 2a (independent, from their own MEME/Smith-Waterman analysis) peaks at nt 15-20,")
print("   max count 250 at nt 17. Paper's m1=nt13-17, m2=nt17-20.")
allr=[z[3] for z in rows]
print(f"\n   median rank of aptamer-complementary 5-mers: {np.median(allr):,.0f} of {len(order):,}")
print(f"   median rank of a random 5-mer would be {len(order)/2:,.0f}")
from scipy import stats
print(f"   Mann-Whitney vs uniform: p = {stats.mannwhitneyu(allr,np.arange(1,len(order)+1),alternative='less').pvalue:.2e}")
