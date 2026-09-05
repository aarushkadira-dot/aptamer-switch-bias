import gzip, collections, numpy as np
from scipy import stats
F="/sessions/dazzling-sweet-pascal/mnt/Figure 2 and 3/SRR24093234_1.fastq.gz"
LIM=4_000_000
print("Construct: adapter(33) - T(25) - N10 - TCTCAG - 3'Cy3")
print("DNA synthesis runs 3'->5'. Within N10, base i couples AFTER base i+1 exists,")
print("and BEFORE base i-1 exists. So:")
print("  synthesis-driven bias  -> dependence on the 3' neighbour (i+1) only")
print("  amplification/seq bias -> no directional asymmetry expected\n")
cnt=collections.Counter(); n=0
with gzip.open(F,"rt") as fh:
    lines=[]
    for line in fh:
        lines.append(line.rstrip("\n"))
        if len(lines)<4: continue
        s=lines[1]; lines=[]
        n+=1
        if n>LIM: break
        j=s.find("TCTCAG")
        if j>=10:
            w=s[j-10:j]
            if len(w)==10 and set(w)<=set("ACGT") and s[max(0,j-22):j-10].count("T")>=10:
                cnt[w]+=1
tot=sum(cnt.values())
print(f"reads scanned {n:,}   framed N10 {tot:,}   unique {len(cnt):,}\n")
B="ACGT"; bi={b:k for k,b in enumerate(B)}
marg=np.zeros((10,4))
for w,c in cnt.items():
    for i,ch in enumerate(w): marg[i,bi[ch]]+=c
marg/=marg.sum(1,keepdims=True)
print("per-position abundance-weighted composition (%)")
print("  pos " + "  ".join(f"{b:>6}" for b in B))
for i in range(10):
    print(f"  {i+1:>3} " + "  ".join(f"{100*marg[i,k]:6.2f}" for k in range(4)))
def conditional(offset):
    T=np.zeros((4,4))
    for w,c in cnt.items():
        for i in range(1,9):
            T[bi[w[i+offset]],bi[w[i]]]+=c
    return T/T.sum(1,keepdims=True), T
P3,T3=conditional(+1); P5,T5=conditional(-1)
base=np.zeros(4)
for w,c in cnt.items():
    for i in range(1,9): base[bi[w[i]]]+=c
base/=base.sum()
print("\nunconditional composition, positions 2-9: " + "  ".join(f"{b} {100*base[k]:.2f}%" for k,b in enumerate(B)))
def show(P,T,offset):
    nm = "3' neighbour (ALREADY coupled)" if offset>0 else "5' neighbour (NOT YET coupled)"
    print(f"\n=== conditioning on {nm} ===")
    print("  nb |" + "".join(f"{b:>8}" for b in B) + "   | max |dev|")
    dev=[]
    for k,nb in enumerate(B):
        d=np.abs(P[k]-base).max(); dev.append(d)
        print(f"   {nb} | " + " ".join(f"{100*P[k,m]:7.2f}" for m in range(4)) + f"   | {100*d:5.2f} pp")
    chi2,p,dof,_=stats.chi2_contingency(T)
    V=np.sqrt(chi2/(T.sum()*(min(T.shape)-1)))
    print(f"  chi2={chi2:,.0f} dof={dof} p={p:.3g}  Cramer's V={V:.5f}  mean|dev|={100*np.mean(dev):.2f} pp")
    return V
V3=show(P3,T3,+1); V5=show(P5,T5,-1)
print("\n=== DISCRIMINATOR: directional asymmetry ===")
print(f"  V, 3' neighbour (coupled before) : {V3:.5f}")
print(f"  V, 5' neighbour (coupled after)  : {V5:.5f}")
print(f"  ratio 3'/5' = {V3/V5:.3f}")
print("  synthesis-origin predicts ratio >> 1 ; amplification predicts ~1")
np.save("/tmp/dinuc_counts.npy", np.array([T3,T5]))
import pickle; pickle.dump(cnt, open("/tmp/n10_counts.pkl","wb"))
