import gzip, collections
F="/sessions/dazzling-sweet-pascal/mnt/Figure 2 and 3/SRR24093234_1.fastq.gz"
reads=[]; n=0
with gzip.open(F,"rt") as fh:
    lines=[]
    for line in fh:
        lines.append(line.rstrip("\n"))
        if len(lines)<4: continue
        reads.append(lines[1]); lines=[]
        n+=1
        if n>=200000: break
print(f"read length distribution: {collections.Counter(len(r) for r in reads).most_common(5)}\n")
print("first 6 reads:")
for r in reads[:6]: print("  ", r)
L=max(len(r) for r in reads)
print(f"\nper-cycle base composition (first {min(L,80)} cycles), to locate the constant regions:")
print("  cyc     A     C     G     T   | consensus / entropy")
import math
for i in range(min(L,80)):
    c=collections.Counter(r[i] for r in reads if len(r)>i and r[i] in "ACGT")
    t=sum(c.values())
    if t<1000: continue
    fr={b:c[b]/t for b in "ACGT"}
    H=-sum(p*math.log2(p) for p in fr.values() if p>0)
    cons=max(fr,key=fr.get)
    tag = cons if H<0.6 else ("." if H>1.9 else "?")
    print(f"  {i:>3} " + " ".join(f"{100*fr[b]:5.1f}" for b in "ACGT") + f"   | {tag}  H={H:.2f}")
